#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
import math

# --- Constants ---

# LINEAR SPEED LIMITS
MAX_LINEAR_SPEED = 0.25
MIN_LINEAR_SPEED = 0.00
MAX_REVERSE_SPEED = 0.1

# ANGULAR SPEED LIMITS
MAX_ANGULAR_SPEED = 0.45
MIN_ANGULAR_SPEED = 0.0

# SAFETY AND WALL-FOLLOWING DISTANCES
CRITICAL_DISTANCE = 0.38
WARNING_DISTANCE = 0.45
SIDE_CLEARANCE = 0.4

class WallFollowerNode(Node):
    """
    A ROS2 node that implements wall-following and collision avoidance logic.
    """
    def __init__(self):
        super().__init__('wall_follower_node')
        
        # --- Wall following parameters ---
        self.wall_distance = 0.5      # Desired distance to keep from the left wall
        self.wall_threshold = 0.4     # Threshold to detect being too close to the wall
        self.front_threshold = 0.5    # Distance to detect a wall in front and start turning
        self.max_speed = MAX_LINEAR_SPEED
        self.turn_speed = MAX_ANGULAR_SPEED
        
        # --- Safety parameters ---
        self.critical_distance = CRITICAL_DISTANCE
        self.warning_distance = WARNING_DISTANCE
        self.side_clearance = SIDE_CLEARANCE
        
        # --- Subscribers and Publishers ---
        self.create_subscription(LaserScan, '/scan', self._scan_received, 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(Bool,'/explore_trigger',self._explore_trigger,10)
        
        # --- Control Loop ---
        self.control_timer = self.create_timer(0.1, self.control_loop) # 10Hz control loop
        
        self.latest_scan = None
        self.can_explore = True
        
        self.get_logger().info('Wall Follower Node started.')
        self.get_logger().info(f'Speed limits: Linear={MIN_LINEAR_SPEED}-{MAX_LINEAR_SPEED}m/s, Angular={MIN_ANGULAR_SPEED}-{MAX_ANGULAR_SPEED}rad/s')
        self.get_logger().info(f'Safety distances: Critical={CRITICAL_DISTANCE}m, Warning={WARNING_DISTANCE}m')

    def _scan_received(self, msg):
        """Callback to store the latest laser scan data."""
        self.latest_scan = msg
    
    def _explore_trigger(self,msg):
        """Callback to trigger exploration"""
        self.can_explore = msg.data

        self.get_logger().info(f'Message data: {msg.data}')

        if msg.data:
            self.get_logger().info(f'Exploration resumed...')
        else:
            self.get_logger().info(f'Exploration stopped...')

    def control_loop(self):
        """Main control loop that executes wall-following and safety checks."""
        if not self.can_explore:
            # Publish stop command when not exploring
            stop_cmd = Twist()
            self.cmd_pub.publish(stop_cmd)
            return

        if not self.latest_scan:
            return
            
        # Get the desired movement from the wall-following logic
        twist_command = self.wall_following_behavior()
        
        # Apply safety override to prevent collisions
        safe_twist = self.apply_safety_override(twist_command)
        
        # Publish the final, safe command
        self.cmd_pub.publish(safe_twist)

    def wall_following_behavior(self):
        """
        Calculates Twist commands for left-wall-following.
        This behavior is the primary logic for navigation.
        """
        twist = Twist()
        
        ranges = list(self.latest_scan.ranges)
        num_ranges = len(ranges)
        
        # Define regions for front and side distance checks
        front_indices = self._get_scan_indices(0.0, 0.15, num_ranges)
        left_indices = self._get_scan_indices(0.25, 0.15, num_ranges)
        
        front_dist = self._get_min_distance(ranges, front_indices)
        left_dist = self._get_min_distance(ranges, left_indices)
        
        # --- Core Wall-Following Logic ---
        if front_dist < self.front_threshold:
            # Obstacle ahead: Turn right. Prioritize turning over forward motion.
            self.get_logger().info("Wall ahead, turning right.")
            twist.angular.z = -max(MIN_ANGULAR_SPEED, self.turn_speed * 0.8)
            twist.linear.x = 0.0
        elif left_dist > self.wall_distance * 1.5:
            # Lost the left wall: Turn left to find it again.
            self.get_logger().info("Lost wall, turning left to find.")
            twist.angular.z = max(MIN_ANGULAR_SPEED, self.turn_speed * 0.6)
            twist.linear.x = MAX_LINEAR_SPEED * 0.5
        else:
            # Follow the left wall
            twist.linear.x = MAX_LINEAR_SPEED
            
            if left_dist < self.wall_threshold:
                # Too close: Steer away from the wall (right).
                twist.angular.z = -max(MIN_ANGULAR_SPEED * 0.5, 0.2)
            elif left_dist > self.wall_distance * 1.5:
                # Too far: Steer towards the wall (left).
                twist.angular.z = max(MIN_ANGULAR_SPEED * 0.5, 0.15)
            else:
                # Ideal distance: Go straight.
                twist.angular.z = 0.0
                
        return twist

    def apply_safety_override(self, twist):
        """
        Overrides the given Twist command if a collision is imminent.
        This acts as a safety layer on top of other behaviors.
        """
        if not self.latest_scan:
            return twist
            
        ranges = list(self.latest_scan.ranges)
        num_ranges = len(ranges)
        
        # Define comprehensive safety zones
        front_indices = self._get_scan_indices(0.0, 0.25, num_ranges)      # Wide front
        front_left_indices = self._get_scan_indices(0.125, 0.15, num_ranges) # Front-left corner
        front_right_indices = self._get_scan_indices(0.875, 0.15, num_ranges)# Front-right corner
        left_indices = self._get_scan_indices(0.25, 0.2, num_ranges)       # Left side
        right_indices = self._get_scan_indices(0.75, 0.2, num_ranges)      # Right side
        
        # Get minimum distances in all zones
        front_dist = self._get_min_distance(ranges, front_indices)
        front_left_dist = self._get_min_distance(ranges, front_left_indices)
        front_right_dist = self._get_min_distance(ranges, front_right_indices)
        left_dist = self._get_min_distance(ranges, left_indices)
        right_dist = self._get_min_distance(ranges, right_indices)
        
        safe_twist = Twist()
        safe_twist.linear.x = twist.linear.x
        safe_twist.angular.z = twist.angular.z
        
        # PRIORITY 1: Critical Collision Avoidance
        if (front_dist < self.critical_distance or 
            front_left_dist < self.critical_distance or 
            front_right_dist < self.critical_distance):
            
            self.get_logger().warn("CRITICAL SAFETY: Obstacle ahead. Halting and turning.")
            safe_twist.linear.x = 0.0 # Stop forward movement
            
            # Turn away from the closest obstacle
            if left_dist > right_dist:
                safe_twist.angular.z = self.turn_speed  # Turn left
            else:
                safe_twist.angular.z = -self.turn_speed # Turn right
            
            return safe_twist

        # PRIORITY 2: Warning Zone - Slow down and adjust
        if (front_dist < self.warning_distance or 
            front_left_dist < self.warning_distance or 
            front_right_dist < self.warning_distance):
            
            self.get_logger().info("Safety Warning: Obstacle in warning zone, slowing down.")
            safe_twist.linear.x = min(safe_twist.linear.x, 0.1) # Reduce speed
            
            # Adjust turn to steer away
            if front_left_dist < front_right_dist:
                safe_twist.angular.z = min(safe_twist.angular.z, -0.2) # Bias turn right
            else:
                safe_twist.angular.z = max(safe_twist.angular.z, 0.2)  # Bias turn left

        # PRIORITY 3: Side Clearance Check
        if left_dist < self.side_clearance and safe_twist.angular.z > 0:
            self.get_logger().info("Side Safety: Reducing left turn due to close side obstacle.")
            safe_twist.angular.z = max(safe_twist.angular.z * 0.5, 0.0) # Reduce left turn
            
        if right_dist < self.side_clearance and safe_twist.angular.z < 0:
            self.get_logger().info("Side Safety: Reducing right turn due to close side obstacle.")
            safe_twist.angular.z = min(safe_twist.angular.z * 0.5, 0.0) # Reduce right turn

        # PRIORITY 4: Final Speed and Turn Limiting
        safe_twist.linear.x = max(-MAX_REVERSE_SPEED, min(safe_twist.linear.x, MAX_LINEAR_SPEED))
        safe_twist.angular.z = max(-MAX_ANGULAR_SPEED, min(safe_twist.angular.z, MAX_ANGULAR_SPEED))
        
        if safe_twist.linear.x > 0:
            safe_twist.linear.x = max(safe_twist.linear.x, MIN_LINEAR_SPEED)
        
        if abs(safe_twist.angular.z) > 0.01:
            if safe_twist.angular.z > 0:
                safe_twist.angular.z = max(safe_twist.angular.z, MIN_ANGULAR_SPEED)
            else:
                safe_twist.angular.z = min(safe_twist.angular.z, -MIN_ANGULAR_SPEED)
        
        return safe_twist

    def _get_scan_indices(self, center_fraction, width_fraction, num_ranges):
        """Helper to get a range of indices from a laser scan."""
        center = int(center_fraction * num_ranges) % num_ranges
        half_width = int((width_fraction * num_ranges) / 2)
        start = (center - half_width) % num_ranges
        end = (center + half_width) % num_ranges
        
        if start < end:
            return [(start, end)]
        else: # Handle wrap-around for front-facing scans
            return [(start, num_ranges - 1), (0, end)]

    def _get_min_distance(self, ranges, index_pairs):
        """Helper to find the minimum valid distance within a set of scan indices."""
        distances = []
        for start, end in index_pairs:
            sector = ranges[start:end + 1]
            valid_distances = [r for r in sector if not math.isinf(r) and not math.isnan(r) and r > 0.1]
            distances.extend(valid_distances)
        
        return min(distances, default=float('inf'))

def main():
    rclpy.init()
    node = WallFollowerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Stop the robot on shutdown
        twist = Twist()
        node.cmd_pub.publish(twist)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
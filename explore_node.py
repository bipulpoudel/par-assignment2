#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, String
import math
from collections import deque
import tf2_ros
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

class PledgeWallFollowerNode(Node):
    """
    A ROS2 node that implements the Pledge Algorithm for right wall following.
    The Pledge Algorithm tracks total rotation to ensure the robot can navigate
    out of any maze configuration and return to straight-line movement.
    """
    
    def __init__(self):
        super().__init__('pledge_wall_follower')
        
        # --- Pledge Algorithm Parameters ---
        self.declare_parameter('target_direction', 0.0)  # Target direction in radians (0 = east)
        self.declare_parameter('wall_following_distance', 0.4)  # Desired distance from wall
        self.declare_parameter('wall_detection_threshold', 0.6)  # Distance to start wall following
        self.declare_parameter('front_obstacle_threshold', 0.8)  # Distance to consider obstacle ahead
        
        # --- Speed Parameters ---
        self.declare_parameter('max_linear_speed', 0.4)
        self.declare_parameter('max_angular_speed', 0.6)
        self.declare_parameter('wall_following_speed', 0.25)
        self.declare_parameter('turn_speed', 0.45)
        
        # --- Safety Parameters ---
        self.declare_parameter('min_safe_distance', 0.3)
        self.declare_parameter('emergency_stop_distance', 0.25)
        
        # --- Smart Decision Making Parameters ---
        self.declare_parameter('memory_window_size', 10)  # Number of recent positions to remember
        self.declare_parameter('avoidance_radius', 0.8)   # Radius to avoid recent positions
        self.declare_parameter('position_update_threshold', 0.3)  # Min distance to update position
        self.declare_parameter('smart_decision_weight', 0.5)  # Weight for smart decision influence
        
        # Get parameters
        self.target_direction = self.get_parameter('target_direction').value
        self.wall_distance = self.get_parameter('wall_following_distance').value
        self.wall_threshold = self.get_parameter('wall_detection_threshold').value
        self.front_threshold = self.get_parameter('front_obstacle_threshold').value
        self.max_linear_speed = self.get_parameter('max_linear_speed').value
        self.max_angular_speed = self.get_parameter('max_angular_speed').value
        self.wall_speed = self.get_parameter('wall_following_speed').value
        self.turn_speed = self.get_parameter('turn_speed').value
        self.min_safe_distance = self.get_parameter('min_safe_distance').value
        self.emergency_distance = self.get_parameter('emergency_stop_distance').value
        self.memory_window_size = self.get_parameter('memory_window_size').value
        self.avoidance_radius = self.get_parameter('avoidance_radius').value
        self.position_update_threshold = self.get_parameter('position_update_threshold').value
        self.smart_decision_weight = self.get_parameter('smart_decision_weight').value
        
        # --- Pledge Algorithm State ---
        self.total_angle = 0.0  # Cumulative angle turned (key to Pledge Algorithm)
        self.current_heading = 0.0  # Current robot heading
        self.previous_heading = 0.0  # Previous heading for angle calculation
        self.wall_following_mode = False  # Whether currently following a wall
        self.pledge_active = True  # Whether Pledge algorithm is active
        
        # --- Wall Following State ---
        self.right_wall_detected = False
        self.front_obstacle_detected = False
        self.lost_wall_counter = 0  # Counter for when wall is lost
        self.max_lost_wall_count = 10  # Max cycles before considering wall truly lost
        
        # --- Control State ---
        self.latest_scan = None
        self.can_move = True
        self.first_scan_received = False
        self.robot_detected = False  # Robot detection state
        
        # --- Smart Memory System ---
        self.position_history = deque(maxlen=self.memory_window_size)  # Sliding window of positions
        self.last_recorded_position = None
        self.current_position = None
        
        # --- Status Logging ---
        self.status_counter = 0
        self.status_log_interval = 50  # Log every 50 cycles (5 seconds at 10Hz)
        
        # --- TF Buffer for Position Tracking ---
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # --- Publishers and Subscribers ---
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.trigger_sub = self.create_subscription(Bool, '/explore_trigger', self.trigger_callback, 10)
        self.robot_detected_sub = self.create_subscription(Bool, 'robot_detected', self.robot_detected_callback, 10)
        self.movement_priority_sub = self.create_subscription(String, '/movement_priority', self.movement_priority_callback, 10)
        
        # --- Control Timer ---
        self.control_timer = self.create_timer(0.1, self.control_loop)  # 10Hz
        
        # Movement priority control
        self.current_movement_priority = "none"
        self.can_publish_movement = True
        
        self.get_logger().info('Pledge Wall Follower Node Started')
        self.get_logger().info(f'Target direction: {math.degrees(self.target_direction):.1f}°')
        self.get_logger().info(f'Right wall following with Pledge Algorithm')
        self.get_logger().info(f'Wall distance: {self.wall_distance}m, Detection threshold: {self.wall_threshold}m')
        
    def scan_callback(self, msg):
        """Process laser scan data"""
        self.latest_scan = msg
        if not self.first_scan_received:
            self.first_scan_received = True
            self.get_logger().info('First laser scan received - ready to start wall following')
    
    def trigger_callback(self, msg):
        """Control whether wall following is active"""
        self.can_move = msg.data
        if msg.data:
            self.get_logger().info('Exploration activated')
        else:
            self.get_logger().info('Exploration deactivated')
            # Stop robot when deactivated, but only if we have priority
            if self.can_publish_movement:
                stop_cmd = Twist()
                self.cmd_pub.publish(stop_cmd)
    
    def control_loop(self):
        """Main control loop implementing Pledge Algorithm with smart memory"""
        if not self.can_move or not self.latest_scan or not self.first_scan_received or self.robot_detected:
            if self.robot_detected:
                # Ensure robot is completely stopped when robot is detected
                self.stop_robot()
            return
        
        # Only publish movement commands if we have priority
        if not self.can_publish_movement:
            return  # Higher priority node is controlling movement
        
        # Update current position
        self.update_current_position()
        
        # Update position history
        self.update_position_history()
        
        # Update sensor readings
        self.update_sensor_readings()
        
        # Implement Pledge Algorithm with smart decisions
        cmd = self.pledge_algorithm_smart()
        
        # Apply safety constraints
        safe_cmd = self.apply_safety_constraints(cmd)
        
        # Publish command only if we have movement priority
        if self.can_publish_movement:
            self.cmd_pub.publish(safe_cmd)
        
        # Log status periodically
        self.log_status()
    
    def update_current_position(self):
        """Update current position using TF"""
        try:
            trans = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
            self.current_position = (trans.transform.translation.x, trans.transform.translation.y)
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            # Fallback: estimate position from odometry or use last known position
            if self.current_position is None:
                self.current_position = (0.0, 0.0)  # Default starting position
    
    def update_position_history(self):
        """Update position history with smart filtering"""
        if self.current_position is None:
            return
            
        # Only add position if it's significantly different from last recorded
        if (self.last_recorded_position is None or 
            self.distance_between_points(self.current_position, self.last_recorded_position) > 
            self.position_update_threshold):
            
            self.position_history.append(self.current_position)
            self.last_recorded_position = self.current_position
    
    def update_sensor_readings(self):
        """Update sensor readings for wall detection"""
        if not self.latest_scan:
            return
        
        ranges = np.array(self.latest_scan.ranges)
        ranges[np.isinf(ranges)] = 10.0  # Replace inf with large value
        ranges[np.isnan(ranges)] = 10.0  # Replace nan with large value
        
        num_ranges = len(ranges)
        
        # Define sensor regions (assuming 0° is front, angles increase counterclockwise)
        # Right side: -90° to -30° (or 270° to 330°)
        right_start_idx = int(0.75 * num_ranges)  # ~270°
        right_end_idx = int(0.92 * num_ranges)    # ~330°
        right_indices = list(range(right_start_idx, num_ranges)) + list(range(0, int(0.08 * num_ranges)))
        
        # Front: -30° to +30°
        front_indices = list(range(int(0.92 * num_ranges), num_ranges)) + list(range(0, int(0.08 * num_ranges)))
        
        # Get distances
        right_distances = [ranges[i] for i in right_indices if 0 <= i < num_ranges]
        front_distances = [ranges[i] for i in front_indices if 0 <= i < num_ranges]
        
        # Calculate minimum distances
        right_dist = min(right_distances) if right_distances else 10.0
        front_dist = min(front_distances) if front_distances else 10.0
        
        # Update detection flags
        self.right_wall_detected = right_dist < self.wall_threshold
        self.front_obstacle_detected = front_dist < self.front_threshold
        
        # Track wall loss
        if not self.right_wall_detected:
            self.lost_wall_counter += 1
        else:
            self.lost_wall_counter = 0
    
    def pledge_algorithm_smart(self):
        """
        Implement Pledge Algorithm for right wall following with smart decisions
        
        Algorithm:
        1. If total_angle == 0 and no wall detected: move toward target direction
        2. If obstacle ahead or wall detected: start/continue wall following
        3. While wall following: follow right wall and track total angle
        4. Use smart decisions to avoid recently visited areas
        """
        cmd = Twist()
        
        # Check if we should exit wall following mode
        if (self.wall_following_mode and 
            abs(self.total_angle) < 0.1 and  # Back to original orientation
            not self.right_wall_detected and 
            not self.front_obstacle_detected and
            self.lost_wall_counter > self.max_lost_wall_count):
            
            self.wall_following_mode = False
            self.total_angle = 0.0  # Reset for precision
            self.get_logger().info('Pledge: Exiting wall following mode - returning to target direction')
        
        if not self.wall_following_mode:
            # Mode 1: Move toward target direction with smart avoidance
            cmd = self.move_toward_target_smart()
            
            # Check if we need to start wall following
            if self.front_obstacle_detected or self.right_wall_detected:
                self.wall_following_mode = True
                self.get_logger().info(f'Pledge: Starting wall following - total angle: {math.degrees(self.total_angle):.1f}°')
        else:
            # Mode 2: Follow right wall with smart decisions
            cmd = self.follow_right_wall_smart()
        
        return cmd
    
    def move_toward_target_smart(self):
        """Move in target direction when not wall following, avoiding recent areas"""
        cmd = Twist()
        
        # Calculate angle difference to target direction
        angle_diff = self.normalize_angle(self.target_direction - self.current_heading)
        
        # Check if target direction would lead to recently visited area
        avoidance_factor = self.calculate_avoidance_factor(self.target_direction)
        
        if abs(angle_diff) > 0.1:  # Need to turn toward target
            # Apply smart avoidance to turning decisions
            adjusted_angular = np.sign(angle_diff) * min(abs(angle_diff), self.turn_speed)
            
            # If high avoidance factor, modify turning behavior
            if avoidance_factor > 0.5:
                # Consider alternative direction
                alt_direction = self.target_direction + math.pi  # Opposite direction
                alt_angle_diff = self.normalize_angle(alt_direction - self.current_heading)
                alt_avoidance = self.calculate_avoidance_factor(alt_direction)
                
                if alt_avoidance < avoidance_factor:
                    adjusted_angular = np.sign(alt_angle_diff) * min(abs(alt_angle_diff), self.turn_speed)
                    self.get_logger().info('Smart: Using alternative direction to avoid recent area')
            
            cmd.angular.z = adjusted_angular
            cmd.linear.x = self.max_linear_speed * 0.5 * (1.0 - avoidance_factor * 0.5)
        else:
            # Move straight toward target, but reduce speed if approaching recent area
            cmd.linear.x = self.max_linear_speed * (1.0 - avoidance_factor * self.smart_decision_weight)
            cmd.angular.z = 0.0
        
        return cmd
    
    def follow_right_wall_smart(self):
        """Follow right wall while tracking total angle and avoiding recent areas"""
        cmd = Twist()
        
        if self.front_obstacle_detected:
            # Obstacle ahead: turn left (counterclockwise)
            # But check if this would lead to recently visited area
            left_direction = self.current_heading + math.pi/2
            avoidance_factor = self.calculate_avoidance_factor(left_direction)
            
            turn_speed = self.turn_speed * (1.0 - avoidance_factor * 0.3)
            cmd.angular.z = turn_speed
            cmd.linear.x = 0.0
            angular_change = turn_speed * 0.1  # dt = 0.1s
            self.total_angle += angular_change
            
            if avoidance_factor > 0.7:
                self.get_logger().info('Smart: High avoidance factor while turning left')
            
        elif not self.right_wall_detected:
            # No right wall: turn right (clockwise) to find wall
            # Check avoidance for right turn
            right_direction = self.current_heading - math.pi/2
            avoidance_factor = self.calculate_avoidance_factor(right_direction)
            
            turn_speed = self.turn_speed * 0.7 * (1.0 - avoidance_factor * 0.2)
            cmd.angular.z = -turn_speed
            cmd.linear.x = self.wall_speed * 0.5
            angular_change = -turn_speed * 0.1
            self.total_angle += angular_change
            
        else:
            # Following right wall with smart adjustments
            right_dist = self.get_right_wall_distance()
            forward_avoidance = self.calculate_avoidance_factor(self.current_heading)
            
            if right_dist < self.wall_distance * 0.8:
                # Too close to wall: steer left
                cmd.angular.z = 0.3 * (1.0 + forward_avoidance * 0.3)  # Stronger left if avoiding ahead
                cmd.linear.x = self.wall_speed * (1.0 - forward_avoidance * 0.4)
                angular_change = cmd.angular.z * 0.1
                self.total_angle += angular_change
                
            elif right_dist > self.wall_distance * 1.3:
                # Too far from wall: steer right
                cmd.angular.z = -0.2 * (1.0 - forward_avoidance * 0.3)  # Weaker right if avoiding ahead
                cmd.linear.x = self.wall_speed * (1.0 - forward_avoidance * 0.3)
                angular_change = cmd.angular.z * 0.1
                self.total_angle += angular_change
                
            else:
                # Good distance: move straight along wall
                cmd.linear.x = self.wall_speed * (1.0 - forward_avoidance * self.smart_decision_weight)
                cmd.angular.z = 0.0
        
        return cmd
    
    def get_right_wall_distance(self):
        """Get distance to right wall"""
        if not self.latest_scan:
            return 10.0
        
        ranges = np.array(self.latest_scan.ranges)
        ranges[np.isinf(ranges)] = 10.0
        ranges[np.isnan(ranges)] = 10.0
        
        num_ranges = len(ranges)
        
        # Right side indices (270° region)
        right_idx = int(0.75 * num_ranges)  # ~270°
        right_range = 20  # Check ±20 indices around right side
        
        right_distances = []
        for i in range(max(0, right_idx - right_range), 
                      min(num_ranges, right_idx + right_range)):
            right_distances.append(ranges[i])
        
        return min(right_distances) if right_distances else 10.0
    
    def apply_safety_constraints(self, cmd):
        """Apply safety constraints to prevent collisions"""
        if not self.latest_scan:
            return cmd
        
        ranges = np.array(self.latest_scan.ranges)
        ranges[np.isinf(ranges)] = 10.0
        ranges[np.isnan(ranges)] = 10.0
        
        # Check for emergency stop conditions
        min_front_distance = np.min(ranges[int(0.9*len(ranges)):] + ranges[:int(0.1*len(ranges))])
        
        safe_cmd = Twist()
        safe_cmd.linear.x = cmd.linear.x
        safe_cmd.angular.z = cmd.angular.z
        
        if min_front_distance < self.emergency_distance:
            # Emergency stop
            safe_cmd.linear.x = 0.0
            safe_cmd.angular.z = self.turn_speed  # Turn to avoid
            
        elif min_front_distance < self.min_safe_distance:
            # Slow down
            safe_cmd.linear.x = min(cmd.linear.x, 0.05)
        
        # Limit speeds
        safe_cmd.linear.x = max(-self.max_linear_speed, min(safe_cmd.linear.x, self.max_linear_speed))
        safe_cmd.angular.z = max(-self.max_angular_speed, min(safe_cmd.angular.z, self.max_angular_speed))
        
        return safe_cmd
    
    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    def calculate_avoidance_factor(self, direction):
        """Calculate how much to avoid a direction based on recent position history"""
        if len(self.position_history) < 2 or self.current_position is None:
            return 0.0
        
        # Calculate potential position if moving in this direction
        step_size = 0.5  # Look ahead distance
        potential_x = self.current_position[0] + step_size * math.cos(direction)
        potential_y = self.current_position[1] + step_size * math.sin(direction)
        potential_pos = (potential_x, potential_y)
        
        # Check proximity to recent positions
        max_avoidance = 0.0
        for i, hist_pos in enumerate(self.position_history):
            distance = self.distance_between_points(potential_pos, hist_pos)
            
            if distance < self.avoidance_radius:
                # Recent positions get higher weight
                recency_weight = (len(self.position_history) - i) / len(self.position_history)
                avoidance = (1.0 - distance / self.avoidance_radius) * recency_weight
                max_avoidance = max(max_avoidance, avoidance)
        
        return min(max_avoidance, 1.0)  # Clamp to [0, 1]
    
    def distance_between_points(self, pos1, pos2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def log_status(self):
        """Log current status periodically"""
        self.status_counter += 1
        if self.status_counter >= self.status_log_interval:
            self.status_counter = 0
            
            mode = "Target Seeking" if not self.wall_following_mode else "Wall Following"
            wall_status = "Detected" if self.right_wall_detected else "Lost"
            obstacle_status = "Detected" if self.front_obstacle_detected else "Clear"
            robot_status = "DETECTED - STOPPED" if self.robot_detected else "Not Detected"
            
            # Smart memory status
            memory_size = len(self.position_history)
            current_pos_str = f"({self.current_position[0]:.2f}, {self.current_position[1]:.2f})" if self.current_position else "Unknown"
            
            self.get_logger().info(
                f'Pledge Status - Mode: {mode} | '
                f'Total Angle: {math.degrees(self.total_angle):.1f}° | '
                f'Right Wall: {wall_status} | '
                f'Front Obstacle: {obstacle_status} | '
                f'Robot Detection: {robot_status} | '
                f'Position: {current_pos_str} | '
                f'Memory: {memory_size}/{self.memory_window_size} positions'
            )

    def robot_detected_callback(self, msg):
        """Callback for robot detection"""
        self.robot_detected = msg.data
        if msg.data:
            self.get_logger().info('Robot detected - stopping robot')
            self.stop_robot()
        else:
            self.get_logger().info('Robot not detected')

    def stop_robot(self):
        """Stop the robot"""
        if self.can_publish_movement:
            stop_cmd = Twist()
            self.cmd_pub.publish(stop_cmd)

    def movement_priority_callback(self, msg):
        """Callback for movement priority"""
        self.current_movement_priority = msg.data
        if msg.data == "none":
            self.can_publish_movement = True
        else:
            self.can_publish_movement = False


def main(args=None):
    rclpy.init(args=args)
    
    wall_follower = PledgeWallFollowerNode()
    
    try:
        rclpy.spin(wall_follower)
    except KeyboardInterrupt:
        pass
    
    wall_follower.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main() 

#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Empty, Bool

import random
import math
import time

# GLOBAL SPEED LIMITS - Easy to adjust for your robot
MAX_LINEAR_SPEED = 0.2    # Maximum forward/backward speed (m/s)
MAX_ANGULAR_SPEED = 0.4    # Maximum turning speed (rad/s)  
MIN_LINEAR_SPEED = 0.05    # Minimum speed to maintain movement
MAX_REVERSE_SPEED = 0.15   # Maximum reverse speed

# GLOBAL BEHAVIOR CONFIGURATION - Choose which behaviors to use
AVAILABLE_BEHAVIORS = [
    'zigzag',           # Sharp direction changes
    'smooth_curves',    # Flowing curved movement
    'stop_and_go',      # Random pauses
    'wall_hug',         # Loose wall following
    'chaotic'           # Completely random movementfre
]

# Preset behavior combinations for different game modes
BEHAVIOR_PRESETS = {
    'all': ['zigzag', 'smooth_curves', 'stop_and_go', 'wall_hug', 'chaotic'],
    'evasive': ['zigzag', 'chaotic', 'stop_and_go'],  # Hard to catch
    'smooth': ['smooth_curves', 'wall_hug'],          # Predictable movement
    'chaotic_only': ['chaotic'],                      # Pure randomness
    'basic': ['zigzag', 'smooth_curves'],             # Simple behaviors
}

# Select which behavior set to use
BEHAVIOR_MODE = 'smooth'  # Change this to: 'all', 'evasive', 'smooth', 'chaotic_only', or 'basic'

class RandomTarget(Node):
    def __init__(self):
        super().__init__('random_target')
        
        # Publishers and Subscribers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)    
        self.random_walk_sub = self.create_subscription(Bool, '/random_walk', self.random_walk_callback, 10)       

        self.game_over=False
        self.walking=True
        
        # Safety parameters - collision zones (no emergency stop)
        self.critical_distance = 0.35          # Sharp avoidance turn
        self.warning_distance = 0.5            # Gradual avoidance
        self.safe_distance = 0.7               # Preferred minimum distance
        
        # Movement parameters - using global limits
        self.min_speed = 0.15
        self.max_speed = MAX_LINEAR_SPEED      # Use global limit
        self.max_turn_speed = MAX_ANGULAR_SPEED # Use global limit
        
        # Safety state management
        self.in_avoidance_mode = False
        self.avoidance_start_time = 0
        self.min_avoidance_duration = 1.0      # Minimum time to spend avoiding
        
        # Randomness parameters
        self.direction_change_interval = random.uniform(2.0, 5.0)  # Change direction every 2-5 seconds
        self.last_direction_change = time.time()
        self.current_direction = random.uniform(-0.5, 0.5)  # Current turning bias
        self.speed_variation_interval = random.uniform(1.0, 3.0)  # Change speed every 1-3 seconds
        self.last_speed_change = time.time()
        self.current_speed = random.uniform(self.min_speed, self.max_speed)
        
        # Behavioral states - use global behavior configuration
        self.behavior_change_interval = random.uniform(8.0, 15.0)  # Change behavior every 8-15 seconds
        self.last_behavior_change = time.time()
        
        # Get behaviors from global configuration
        if BEHAVIOR_MODE in BEHAVIOR_PRESETS:
            self.active_behaviors = BEHAVIOR_PRESETS[BEHAVIOR_MODE]
        else:
            self.get_logger().warn(f"Unknown behavior mode '{BEHAVIOR_MODE}', using 'all'")
            self.active_behaviors = BEHAVIOR_PRESETS['all']
        
        self.current_behavior = random.choice(self.active_behaviors)
        
        # Stop and go behavior
        self.stop_duration = 0
        self.stop_start_time = 0
        self.is_stopped = False
        
        # Wall hugging state
        self.wall_side = random.choice(['left', 'right'])
        
        self.get_logger().info(f"Random Target initialized with behavior: {self.current_behavior}")
        self.get_logger().info(f'Active behaviors: {self.active_behaviors}')
        self.get_logger().info(f'Speed limits: Linear={MAX_LINEAR_SPEED}m/s, Angular={MAX_ANGULAR_SPEED}rad/s')

    def define_arcs(self, num_ranges):
        """Define comprehensive sensor regions for collision avoidance"""
        def get_arc(center_frac, arc_width_frac):
            center = int(center_frac * num_ranges) % num_ranges
            half_width = int((arc_width_frac * num_ranges) / 2)
            start = (center - half_width) % num_ranges
            end = (center + half_width) % num_ranges
            if start < end:
                return [(start, end)]
            else:
                return [(start, num_ranges - 1), (0, end)]
        
        return {
            'front': get_arc(0.0, 0.2),           # Front ±10% (wider for safety)
            'front_left': get_arc(0.125, 0.15),   # Front-left 
            'left': get_arc(0.25, 0.15),          # Left ±7.5%
            'back_left': get_arc(0.375, 0.1),     # Back-left
            'right': get_arc(0.75, 0.15),         # Right ±7.5%
            'front_right': get_arc(0.875, 0.15),  # Front-right
            'back': get_arc(0.5, 0.1),            # Back
            # Additional safety zones
            'wide_front': get_arc(0.0, 0.35),     # Wider front coverage
            'left_side': get_arc(0.25, 0.25),     # Wider left coverage  
            'right_side': get_arc(0.75, 0.25)     # Wider right coverage
        }

    def get_region_min(self, ranges, index_ranges):
        """Get minimum distance in a region with better filtering"""
        values = []
        for start, end in index_ranges:
            if start <= end:
                region = ranges[start:end + 1]
            else:
                region = ranges[start:] + ranges[:end + 1]
            # Filter out invalid readings and very close readings (likely noise)
            values += [r for r in region if not math.isinf(r) and not math.isnan(r) and r > 0.05]
        return min(values, default=float('inf'))

    def is_path_clear(self, distances, direction):
        """Check if a path is clear for safe movement"""
        if direction == 'forward':
            return (distances['front'] > self.safe_distance and 
                   distances['front_left'] > self.warning_distance and 
                   distances['front_right'] > self.warning_distance)
        elif direction == 'left':
            return (distances['left'] > self.safe_distance and 
                   distances['front_left'] > self.warning_distance and
                   distances['back_left'] > self.warning_distance)
        elif direction == 'right':
            return (distances['right'] > self.safe_distance and 
                   distances['front_right'] > self.warning_distance)
        return False

    def find_safe_direction(self, distances):
        """Find the safest direction to move"""
        directions = []
        
        # Check all possible directions and their safety scores
        if distances['front'] > self.safe_distance:
            directions.append(('forward', distances['front']))
        if distances['left'] > self.safe_distance:
            directions.append(('left', distances['left']))
        if distances['right'] > self.safe_distance:
            directions.append(('right', distances['right']))
        if distances['back'] > self.warning_distance:
            directions.append(('back', distances['back']))
        
        if directions:
            # Choose the direction with maximum clearance
            directions.sort(key=lambda x: x[1], reverse=True)
            return directions[0][0]
        
        # If no clearly safe direction, find the least dangerous
        min_distances = {
            'left': distances['left'],
            'right': distances['right'], 
            'back': distances['back']
        }
        return max(min_distances, key=min_distances.get)

    def collision_avoidance(self, distances):
        """Collision avoidance behavior without emergency stop"""
        twist = Twist()
        
        # Critical avoidance maneuver - but keep moving
        if (distances['front'] < self.critical_distance or 
            distances['wide_front'] < self.critical_distance):
            safe_direction = self.find_safe_direction(distances)
            
            # Always maintain some forward movement, just slow down
            twist.linear.x = 0.03  # Slow but steady movement
            
            if safe_direction == 'left':
                twist.angular.z = self.max_turn_speed
                self.get_logger().warn("Critical avoidance: Turning left")
            elif safe_direction == 'right':
                twist.angular.z = -self.max_turn_speed
                self.get_logger().warn("Critical avoidance: Turning right")
            elif safe_direction == 'back':
                twist.linear.x = -0.05  # Very slow reverse
                twist.angular.z = random.choice([-self.max_turn_speed, self.max_turn_speed])
                self.get_logger().warn("Critical avoidance: Slow reverse with turn")
            else:
                # No good options, turn toward least dangerous direction
                twist.angular.z = self.max_turn_speed if distances['left'] > distances['right'] else -self.max_turn_speed
                self.get_logger().warn("Critical avoidance: Sharp turn")
            
            return twist
        
        return None  # No critical avoidance needed

    def update_random_parameters(self):
        """Periodically update random movement parameters"""
        current_time = time.time()
        
        # Change movement direction randomly
        if current_time - self.last_direction_change > self.direction_change_interval:
            self.current_direction = random.uniform(-0.8, 0.8)
            self.direction_change_interval = random.uniform(2.0, 5.0)
            self.last_direction_change = current_time
            self.get_logger().info(f"Changed direction bias to: {self.current_direction:.2f}")
        
        # Change speed randomly  
        if current_time - self.last_speed_change > self.speed_variation_interval:
            self.current_speed = random.uniform(self.min_speed, self.max_speed)
            self.speed_variation_interval = random.uniform(1.0, 3.0)
            self.last_speed_change = current_time
            self.get_logger().info(f"Changed speed to: {self.current_speed:.2f}")
        
        # Change overall behavior
        if current_time - self.last_behavior_change > self.behavior_change_interval:
            old_behavior = self.current_behavior
            # Choose from active behaviors, excluding current one
            available_behaviors = [b for b in self.active_behaviors if b != old_behavior]
            if available_behaviors:  # Only change if other behaviors available
                self.current_behavior = random.choice(available_behaviors)
                self.behavior_change_interval = random.uniform(8.0, 15.0)
                self.last_behavior_change = current_time
                self.wall_side = random.choice(['left', 'right'])  # Reset wall side
                self.get_logger().info(f"Changed behavior from {old_behavior} to {self.current_behavior}")
            else:
                # Only one behavior active, just reset timer
                self.behavior_change_interval = random.uniform(8.0, 15.0)
                self.last_behavior_change = current_time

    def zigzag_behavior(self, distances):
        """Sharp zigzag movement pattern with safety"""
        twist = Twist()
        
        # Safety check first
        if not self.is_path_clear(distances, 'forward'):
            safe_direction = self.find_safe_direction(distances)
            if safe_direction == 'left':
                twist.angular.z = 1.0
            elif safe_direction == 'right':
                twist.angular.z = -1.0
            else:
                twist.angular.z = random.choice([-1.0, 1.0])
            twist.linear.x = 0.1  # Slow movement during avoidance
            self.get_logger().info("Zigzag: Safety avoidance")
        else:
            twist.linear.x = min(self.current_speed, 0.25)  # Limit speed for safety
            # Sharp direction changes but not too extreme
            twist.angular.z = self.current_direction * 1.0
        
        return twist

    def smooth_curves_behavior(self, distances):
        """Smooth curved movement with safety"""
        twist = Twist()
        
        if not self.is_path_clear(distances, 'forward'):
            safe_direction = self.find_safe_direction(distances)
            if safe_direction == 'left':
                twist.angular.z = 0.6
            elif safe_direction == 'right':
                twist.angular.z = -0.6
            else:
                twist.angular.z = 0.8 * (1 if distances['left'] > distances['right'] else -1)
            twist.linear.x = 0.15
            self.get_logger().info("Smooth curves: Safety avoidance")
        else:
            twist.linear.x = self.current_speed * 0.8  # Slightly slower for safety
            twist.angular.z = self.current_direction * 0.6
        
        return twist

    def stop_and_go_behavior(self, distances):
        """Randomly stop and start movement with safety"""
        twist = Twist()
        current_time = time.time()
        
        # Check if we should start a new stop (only if safe)
        if (not self.is_stopped and random.random() < 0.02 and 
            self.is_path_clear(distances, 'forward')):
            self.is_stopped = True
            self.stop_duration = random.uniform(0.5, 2.0)
            self.stop_start_time = current_time
            self.get_logger().info(f"Stop and go: Stopping for {self.stop_duration:.1f} seconds")
        
        # Check if stop period is over
        if self.is_stopped:
            if current_time - self.stop_start_time > self.stop_duration:
                self.is_stopped = False
                self.get_logger().info("Stop and go: Resuming movement")
            else:
                # Stay stopped only if it's safe, otherwise move
                if distances['front'] > self.safe_distance:
                    return twist  # Safe to stay stopped
                else:
                    self.is_stopped = False  # Unsafe to stop, resume movement
        
        # Normal movement when not stopped
        if not self.is_path_clear(distances, 'forward'):
            safe_direction = self.find_safe_direction(distances)
            if safe_direction == 'left':
                twist.angular.z = 0.8
            elif safe_direction == 'right':
                twist.angular.z = -0.8
            else:
                twist.angular.z = 1.0 * (1 if distances['left'] > distances['right'] else -1)
            twist.linear.x = 0.1
        else:
            twist.linear.x = self.current_speed * 0.7
            twist.angular.z = self.current_direction * 0.4
        
        return twist

    def wall_hug_behavior(self, distances):
        """Loosely follow walls for unpredictable movement with safety"""
        twist = Twist()
        
        wall_dist = distances['left'] if self.wall_side == 'left' else distances['right']
        
        # Safety override
        if not self.is_path_clear(distances, 'forward'):
            safe_direction = self.find_safe_direction(distances)
            if safe_direction == 'left':
                twist.angular.z = 0.8
            elif safe_direction == 'right':
                twist.angular.z = -0.8
            else:
                twist.angular.z = 1.0 * (1 if distances['left'] > distances['right'] else -1)
            twist.linear.x = 0.1
            self.get_logger().info("Wall hug: Safety override")
        elif wall_dist > 1.2:
            # No wall on chosen side, turn toward it (but safely)
            turn_direction = 0.4 if self.wall_side == 'left' else -0.4
            twist.angular.z = turn_direction
            twist.linear.x = self.current_speed * 0.6
            self.get_logger().info(f"Wall hug: Searching for {self.wall_side} wall")
        else:
            # Loosely follow the wall with random variations
            twist.linear.x = self.current_speed * 0.8
            wall_follow_turn = 0.2 if self.wall_side == 'left' else -0.2
            twist.angular.z = wall_follow_turn + self.current_direction * 0.2
        
        return twist

    def chaotic_behavior(self, distances):
        """Completely unpredictable movement with safety constraints"""
        twist = Twist()
        
        if not self.is_path_clear(distances, 'forward'):
            # Even chaotic behavior respects safety
            safe_direction = self.find_safe_direction(distances)
            if safe_direction == 'left':
                twist.angular.z = random.uniform(0.5, self.max_turn_speed)
            elif safe_direction == 'right':
                twist.angular.z = random.uniform(-self.max_turn_speed, -0.5)
            else:
                twist.angular.z = random.choice([-1.0, 1.0]) * random.uniform(0.8, 1.2)
            twist.linear.x = random.uniform(0.05, 0.15)
            self.get_logger().info("Chaotic: Safety-constrained randomness")
        else:
            # Chaotic but not suicidal movement
            twist.linear.x = random.uniform(0.1, min(self.max_speed, 0.3))
            twist.angular.z = random.uniform(-0.8, 0.8)
        
        return twist

    def random_walk_callback(self, msg):
        self.walking = msg.data

        self.get_logger().info(f'Message data: {msg.data}')

        if msg.data:
            self.get_logger().info(f'Walking resumed...')
        else:
            self.get_logger().info(f'Walking stopped...')


    def scan_callback(self, msg: LaserScan):
        if self.walking:
            self.cmd_pub.publish(Twist())
            self.get_logger().info("Walking stopped!")
            return

        ranges = list(msg.ranges)
        num_ranges = len(ranges)
        side_defs = self.define_arcs(num_ranges)
        
        # Calculate distances for all regions
        distances = {}
        for region, indices in side_defs.items():
            distances[region] = self.get_region_min(ranges, indices)
        
        # PRIORITY 1: Critical collision avoidance (no stop, just avoid)
        critical_twist = self.collision_avoidance(distances)
        if critical_twist is not None:
            self.in_avoidance_mode = True
            self.avoidance_start_time = time.time()
            # Apply global speed limits before publishing
            critical_twist = self.apply_speed_limits(critical_twist)
            self.cmd_pub.publish(critical_twist)
            return
        
        # PRIORITY 2: Check if we're still in avoidance mode
        current_time = time.time()
        if (self.in_avoidance_mode and 
            current_time - self.avoidance_start_time < self.min_avoidance_duration):
            # Continue avoidance behavior for minimum duration
            safe_direction = self.find_safe_direction(distances)
            twist = Twist()
            if safe_direction == 'left':
                twist.angular.z = 0.8
            elif safe_direction == 'right':  
                twist.angular.z = -0.8
            else:
                twist.angular.z = 0.8 if distances['left'] > distances['right'] else -0.8
            
            # Keep moving slowly during avoidance
            twist.linear.x = 0.1
            
            # Apply global speed limits before publishing
            twist = self.apply_speed_limits(twist)
            self.cmd_pub.publish(twist)
            return
        else:
            self.in_avoidance_mode = False
        
        # PRIORITY 3: Normal behavior with safety checks
        # Update random parameters
        self.update_random_parameters()
        
        # Choose behavior based on current behavior type
        if self.current_behavior == 'zigzag':
            twist = self.zigzag_behavior(distances)
        elif self.current_behavior == 'smooth_curves':
            twist = self.smooth_curves_behavior(distances)
        elif self.current_behavior == 'stop_and_go':
            twist = self.stop_and_go_behavior(distances)
        elif self.current_behavior == 'wall_hug':
            twist = self.wall_hug_behavior(distances)
        elif self.current_behavior == 'chaotic':
            twist = self.chaotic_behavior(distances)
        else:
            # Fallback to safe random behavior
            twist = Twist()
            if self.is_path_clear(distances, 'forward'):
                twist.linear.x = self.current_speed * 0.7
                twist.angular.z = random.uniform(-0.4, 0.4)
            else:
                safe_direction = self.find_safe_direction(distances)
                twist.angular.z = 0.6 if safe_direction == 'left' else -0.6
                twist.linear.x = 0.1
        
        # PRIORITY 4: Final safety validation (no complete stops)
        # Double-check that we won't hit anything but keep moving
        if (distances['front'] < self.warning_distance or 
            distances['front_left'] < self.warning_distance or
            distances['front_right'] < self.warning_distance):
            
            # Reduce speed but don't stop completely
            twist.linear.x = min(twist.linear.x, 0.08)
            twist.linear.x = max(twist.linear.x, MIN_LINEAR_SPEED)  # Ensure minimum movement
                
            # Ensure turn is away from obstacles
            if distances['left'] > distances['right']:
                twist.angular.z = abs(twist.angular.z)  # Force left turn
            else:
                twist.angular.z = -abs(twist.angular.z)  # Force right turn
        
        # PRIORITY 5: Speed limiting based on surroundings (no stops)
        # Slow down in tight spaces but maintain movement
        min_surrounding_distance = min(
            distances['front'], distances['left'], distances['right'],
            distances['front_left'], distances['front_right']
        )
        
        if min_surrounding_distance < self.safe_distance:
            speed_factor = max(0.3, min_surrounding_distance / self.safe_distance)  # Minimum 30% speed
            twist.linear.x *= speed_factor
            twist.linear.x = max(twist.linear.x, MIN_LINEAR_SPEED)  # Absolute minimum movement
        
        # Apply global speed limits before publishing
        twist = self.apply_speed_limits(twist)
        self.cmd_pub.publish(twist)

    def apply_speed_limits(self, twist):
        """Apply global speed limits to any Twist message"""
        # Limit linear speed
        twist.linear.x = max(-MAX_REVERSE_SPEED, min(twist.linear.x, MAX_LINEAR_SPEED))
        
        # Limit angular speed  
        twist.angular.z = max(-MAX_ANGULAR_SPEED, min(twist.angular.z, MAX_ANGULAR_SPEED))
        
        # Ensure minimum forward movement when moving forward
        if twist.linear.x > 0:
            twist.linear.x = max(twist.linear.x, MIN_LINEAR_SPEED)
            
        return twist

def main(args=None):
    rclpy.init(args=args)
    node = RandomTarget()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Stop the robot
        twist = Twist()
        node.cmd_pub.publish(twist)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

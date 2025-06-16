#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool, Empty
from geometry_msgs.msg import Twist
import json
import time
import random
import math

class GameCoordinator(Node):
    def __init__(self):
        super().__init__('game_coordinator')
        
        # Parameters
        self.declare_parameter('self_aruco_id', 0)
        self.declare_parameter('timeout_duration', 10.0)  # 10 seconds timeout
        self.declare_parameter('random_walk_duration', 20.0)  # 20 seconds random walk after hiding
        
        self.self_aruco_id = self.get_parameter('self_aruco_id').get_parameter_value().integer_value
        self.timeout_duration = self.get_parameter('timeout_duration').get_parameter_value().double_value
        self.random_walk_duration = self.get_parameter('random_walk_duration').get_parameter_value().double_value
        
        # Game state
        self.game_state = 'exploring'  # exploring, following_marker, hiding, stopped, random_walk
        self.state_start_time = None
        self.tagged_by = None
        self.tagged_robot = None
        
        # Publishers
        self.explore_trigger_pub = self.create_publisher(Bool, '/explore_trigger', 10)
        self.start_hiding_pub = self.create_publisher(Empty, '/start_hiding', 10)
        self.stop_hiding_pub = self.create_publisher(Empty, '/stop_hiding', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.server_command_pub = self.create_publisher(String, '/server_command', 10)
        
        # Subscribers
        self.robot_tagged_sub = self.create_subscription(String, '/robot_tagged', self.robot_tagged_callback, 10)
        self.server_response_sub = self.create_subscription(String, '/server_response', self.server_response_callback, 10)
        
        # Timers
        self.state_timer = self.create_timer(0.1, self.state_machine_update)  # 10Hz state machine
        self.random_walk_timer = self.create_timer(0.5, self.random_walk_update)  # Random walk updates
        
        # Random walk parameters
        self.random_walk_linear_speed = 0.2
        self.random_walk_angular_speed = 0.3
        self.random_walk_change_interval = 2.0  # Change direction every 2 seconds
        self.last_random_walk_change = 0
        self.current_random_cmd = Twist()
        
        # Start in exploring state
        self.set_state('exploring')
        
        self.get_logger().info(f'Game Coordinator initialized for robot {self.self_aruco_id}')
        self.get_logger().info(f'Timeout duration: {self.timeout_duration}s, Random walk duration: {self.random_walk_duration}s')
    
    def set_state(self, new_state):
        """Set new game state and handle state transitions"""
        if self.game_state != new_state:
            self.get_logger().info(f'State transition: {self.game_state} -> {new_state}')
            self.game_state = new_state
            self.state_start_time = self.get_clock().now()
            
            # Handle state entry actions
            if new_state == 'exploring':
                self.start_exploring()
            elif new_state == 'following_marker':
                self.stop_exploring()
            elif new_state == 'hiding':
                self.start_hiding()
            elif new_state == 'stopped':
                self.stop_all_movement()
            elif new_state == 'random_walk':
                self.start_random_walk()
    
    def start_exploring(self):
        """Start exploration behavior"""
        explore_msg = Bool()
        explore_msg.data = True
        self.explore_trigger_pub.publish(explore_msg)
        self.get_logger().info('Started exploring')
    
    def stop_exploring(self):
        """Stop exploration behavior"""
        explore_msg = Bool()
        explore_msg.data = False
        self.explore_trigger_pub.publish(explore_msg)
        self.get_logger().info('Stopped exploring')
    
    def start_hiding(self):
        """Start hiding behavior"""
        self.stop_exploring()
        start_msg = Empty()
        self.start_hiding_pub.publish(start_msg)
        self.get_logger().info('Started hiding behavior')
    
    def stop_hiding(self):
        """Stop hiding behavior"""
        stop_msg = Empty()
        self.stop_hiding_pub.publish(stop_msg)
        self.get_logger().info('Stopped hiding behavior')
    
    def stop_all_movement(self):
        """Stop all robot movement"""
        self.stop_exploring()
        self.stop_hiding()
        stop_cmd = Twist()
        self.cmd_vel_pub.publish(stop_cmd)
        self.get_logger().info('Stopped all movement')
    
    def start_random_walk(self):
        """Start random walk behavior"""
        self.stop_exploring()
        self.stop_hiding()
        self.last_random_walk_change = 0
        self.generate_random_movement()
        self.get_logger().info('Started random walk')
    
    def generate_random_movement(self):
        """Generate random movement command"""
        self.current_random_cmd = Twist()
        
        # Random linear speed (forward biased)
        self.current_random_cmd.linear.x = random.uniform(0.1, self.random_walk_linear_speed)
        
        # Random angular speed (can turn left or right)
        self.current_random_cmd.angular.z = random.uniform(-self.random_walk_angular_speed, self.random_walk_angular_speed)
        
        self.get_logger().info(f'Random walk: linear={self.current_random_cmd.linear.x:.2f}, angular={self.current_random_cmd.angular.z:.2f}')
    
    def random_walk_update(self):
        """Update random walk behavior"""
        if self.game_state == 'random_walk':
            current_time = time.time()
            
            # Change direction periodically
            if current_time - self.last_random_walk_change > self.random_walk_change_interval:
                self.generate_random_movement()
                self.last_random_walk_change = current_time
            
            # Publish random walk command
            self.cmd_vel_pub.publish(self.current_random_cmd)
    
    def robot_tagged_callback(self, msg):
        """Handle robot tagged event"""
        try:
            # Parse the message - should contain info about who tagged whom
            if ',' in msg.data:
                # Format: "tagger_id,tagged_id"
                parts = msg.data.split(',')
                tagger_id = int(parts[0])
                tagged_id = int(parts[1])
            else:
                # Legacy format: just tagger_id, assume this robot was tagged
                tagger_id = int(msg.data)
                tagged_id = self.self_aruco_id
            
            self.get_logger().info(f'Tag event: robot {tagger_id} tagged robot {tagged_id}')
            
            # Send tag event to server
            server_msg = String()
            server_msg.data = json.dumps({
                'command': 'robot_tagged',
                'tagger_id': tagger_id,
                'tagged_id': tagged_id,
                'timestamp': time.time()
            })
            self.server_command_pub.publish(server_msg)
            
            # Handle tag event for this robot
            if tagger_id == self.self_aruco_id:
                # This robot tagged someone - start hiding
                self.tagged_robot = tagged_id
                self.set_state('hiding')
                self.get_logger().info(f'I tagged robot {tagged_id} - starting to hide')
                
            elif tagged_id == self.self_aruco_id:
                # This robot was tagged - stop for timeout duration
                self.tagged_by = tagger_id
                self.set_state('stopped')
                self.get_logger().info(f'I was tagged by robot {tagger_id} - stopping for {self.timeout_duration}s')
            
        except Exception as e:
            self.get_logger().error(f'Error processing robot_tagged message: {e}')
    
    def server_response_callback(self, msg):
        """Handle server response"""
        try:
            response = json.loads(msg.data)
            self.get_logger().info(f'Server response: {response}')
            
            # Handle different server responses
            if response.get('command') == 'role_update':
                role = response.get('role')
                robot_id = response.get('robot_id')
                
                if robot_id == self.self_aruco_id:
                    self.get_logger().info(f'My role updated to: {role}')
                    # Handle role-specific behavior here if needed
                    
        except Exception as e:
            self.get_logger().error(f'Error processing server response: {e}')
    
    def state_machine_update(self):
        """Update state machine and handle timing"""
        if self.state_start_time is None:
            return
        
        current_time = self.get_clock().now()
        elapsed_time = (current_time - self.state_start_time).nanoseconds / 1e9
        
        # Handle state timeouts
        if self.game_state == 'hiding' and elapsed_time >= self.timeout_duration:
            # Hiding timeout - switch to random walk
            self.stop_hiding()
            self.set_state('random_walk')
            self.get_logger().info(f'Hiding timeout ({self.timeout_duration}s) - switching to random walk')
            
        elif self.game_state == 'stopped' and elapsed_time >= self.timeout_duration:
            # Stop timeout - resume exploring
            self.set_state('exploring')
            self.get_logger().info(f'Stop timeout ({self.timeout_duration}s) - resuming exploration')
            
        elif self.game_state == 'random_walk' and elapsed_time >= self.random_walk_duration:
            # Random walk timeout - resume exploring
            self.set_state('exploring')
            self.get_logger().info(f'Random walk timeout ({self.random_walk_duration}s) - resuming exploration')
    
    def get_state(self):
        """Get current game state"""
        return self.game_state
    
    def get_elapsed_time(self):
        """Get elapsed time in current state"""
        if self.state_start_time is None:
            return 0.0
        current_time = self.get_clock().now()
        return (current_time - self.state_start_time).nanoseconds / 1e9

def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = GameCoordinator()
        node.get_logger().info('Game Coordinator started!')
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool, Empty
from geometry_msgs.msg import Twist
import json
import time
import random

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
        self.game_state = 'wall_following' if self.self_aruco_id == 1 else 'exploring'  # Robot 1 starts wall following, Robot 0 starts exploring
        self.state_start_time = None
        self.tagged_by = None
        self.tagged_robot = None
        self.current_role = 'hider' if self.self_aruco_id == 1 else 'seeker'  # Robot 0 is initial seeker, Robot 1 is initial hider
        self.robot_detected = False  # Track if another robot is detected
        
        # Publishers
        self.explore_trigger_pub = self.create_publisher(Bool, '/explore_trigger', 10)
        self.start_hiding_pub = self.create_publisher(Empty, '/start_hiding', 10)
        self.stop_hiding_pub = self.create_publisher(Empty, '/stop_hiding', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.server_command_pub = self.create_publisher(String, '/server_command', 10)
        self.game_status_pub = self.create_publisher(String, '/game_status', 10)
        
        # Subscribers
        self.robot_tagged_sub = self.create_subscription(String, '/robot_tagged', self.robot_tagged_callback, 10)
        self.server_response_sub = self.create_subscription(String, '/server_response', self.server_response_callback, 10)
        self.game_status_sub = self.create_subscription(String, '/game_status', self.game_status_callback, 10)
        self.movement_priority_sub = self.create_subscription(String, '/movement_priority', self.movement_priority_callback, 10)
        
        # Timers
        self.state_timer = self.create_timer(0.1, self.state_machine_update)  # 10Hz state machine
        self.random_walk_timer = self.create_timer(0.5, self.random_walk_update)  # Random walk updates
        
        # Random walk parameters
        self.random_walk_linear_speed = 0.2
        self.random_walk_angular_speed = 0.3
        self.random_walk_change_interval = 2.0  # Change direction every 2 seconds
        self.last_random_walk_change = 0
        self.current_random_cmd = Twist()
        
        # Movement priority control
        self.current_movement_priority = "none"  # Track who has movement control
        self.can_publish_movement = True  # Whether this node can publish movement commands
        
        # Start with appropriate initial behavior
        if self.self_aruco_id == 1:
            self.set_state('wall_following')  # Robot 1 (hider) starts with wall following
        else:
            self.set_state('exploring')  # Robot 0 (seeker) starts exploring
        
        self.get_logger().info(f'Game Coordinator initialized for robot {self.self_aruco_id}')
        self.get_logger().info(f'Initial role: {self.current_role}, Initial state: {self.game_state}')
        self.get_logger().info(f'Timeout duration: {self.timeout_duration}s, Random walk duration: {self.random_walk_duration}s')
        self.get_logger().info(f'Robot behavior: Robot 0 seeks, Robot 1 starts wall following (via explore node) until robot detected')
    
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
            elif new_state == 'wall_following':
                self.start_wall_following()
    
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
        self.stop_wall_following()
        if self.can_publish_movement:
            stop_cmd = Twist()
            self.cmd_vel_pub.publish(stop_cmd)
            self.get_logger().info('Stopped all movement')
    
    def start_random_walk(self):
        """Start random walk behavior"""
        self.stop_exploring()
        self.stop_hiding()
        self.stop_wall_following()
        self.last_random_walk_change = 0
        self.generate_random_movement()
        self.get_logger().info('Started random walk')
        
        # Publish game status for random walk start
        game_status_msg = String()
        game_status_msg.data = f"Robot {self.self_aruco_id} started random walk after hiding"
        self.game_status_pub.publish(game_status_msg)
    
    def start_wall_following(self):
        """Start wall following behavior using explore node"""
        self.stop_hiding()
        explore_msg = Bool()
        explore_msg.data = True
        self.explore_trigger_pub.publish(explore_msg)
        self.get_logger().info('Started wall following behavior via explore node')
    
    def stop_wall_following(self):
        """Stop wall following behavior"""
        explore_msg = Bool()
        explore_msg.data = False
        self.explore_trigger_pub.publish(explore_msg)
        self.get_logger().info('Stopped wall following behavior')
    
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
        if self.game_state == 'random_walk' and self.can_publish_movement:
            current_time = time.time()
            
            # Change direction periodically
            if current_time - self.last_random_walk_change > self.random_walk_change_interval:
                self.generate_random_movement()
                self.last_random_walk_change = current_time
            
            # Publish random walk command only if we have movement priority
            if self.can_publish_movement:
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
                # This robot tagged someone - role switches, now become hider and go far away
                self.tagged_robot = tagged_id
                self.current_role = 'hider'
                self.set_state('hiding')
                self.get_logger().info(f'I tagged robot {tagged_id} - role switched to hider, starting to hide')
                
                # Publish game status for role switch
                game_status_msg = String()
                game_status_msg.data = f"Robot {self.self_aruco_id} becomes hider now, robot {tagged_id} is seeker"
                self.game_status_pub.publish(game_status_msg)
                
            elif tagged_id == self.self_aruco_id:
                # This robot was tagged - role switches, now become seeker 
                self.tagged_by = tagger_id
                self.current_role = 'seeker'
                self.set_state('exploring')  # Immediately start seeking
                self.get_logger().info(f'I was tagged by robot {tagger_id} - role switched to seeker, starting to seek')
                
                # Publish game status for role switch
                game_status_msg = String()
                game_status_msg.data = f"Robot {tagger_id} becomes hider now, robot {self.self_aruco_id} is seeker"
                self.game_status_pub.publish(game_status_msg)
            
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
                    self.current_role = role
                    self.get_logger().info(f'My role updated to: {role}')
                    # Handle role-specific behavior here if needed
                    
        except Exception as e:
            self.get_logger().error(f'Error processing server response: {e}')
    
    def game_status_callback(self, msg):
        """Handle game status updates (robot detection)"""
        try:
            # Check if ArUco tags are detected
            if "ArUco tags detected:" in msg.data:
                detected_ids_str = msg.data.replace("ArUco tags detected: ", "")
                detected_ids = eval(detected_ids_str)  # Parse the list
                
                # Check if other robot is detected
                other_robot_id = 0 if self.self_aruco_id == 1 else 1
                if other_robot_id in detected_ids:
                    if not self.robot_detected:
                        self.robot_detected = True
                        self.get_logger().info(f'Other robot {other_robot_id} detected!')
                        
                        # If this robot is hider and sees another robot, switch to seeking
                        if self.current_role == 'hider' and self.game_state == 'wall_following':
                            self.get_logger().info('Hider detected seeker - switching to exploring/seeking mode')
                            self.stop_wall_following()  # Stop wall following first
                            self.set_state('exploring')
                else:
                    if self.robot_detected:
                        self.robot_detected = False
                        self.get_logger().info('Other robot no longer detected')
                        
                        # If hider loses sight of seeker, resume wall following
                        if self.current_role == 'hider' and self.game_state == 'exploring':
                            self.get_logger().info('Hider lost sight of seeker - resuming wall following')
                            self.stop_exploring()  # Stop exploring first
                            self.set_state('wall_following')
            
            elif "No ArUco tags detected" in msg.data:
                if self.robot_detected:
                    self.robot_detected = False
                    self.get_logger().info('No robots detected')
                    
                    # If hider loses sight of seeker, resume wall following
                    if self.current_role == 'hider' and self.game_state == 'exploring':
                        self.get_logger().info('Hider lost sight of seeker - resuming wall following')
                        self.stop_exploring()  # Stop exploring first
                        self.set_state('wall_following')
                        
        except Exception as e:
            self.get_logger().error(f'Error processing game status: {e}')
    
    def state_machine_update(self):
        """Update state machine and handle timing"""
        if self.state_start_time is None:
            return
        
        current_time = self.get_clock().now()
        elapsed_time = (current_time - self.state_start_time).nanoseconds / 1e9
        
        # Handle state timeouts
        if self.game_state == 'hiding' and elapsed_time >= self.timeout_duration:
            # Hiding timeout - switch behavior based on role
            self.stop_hiding()
            if self.current_role == 'hider':
                # Hider should resume wall following after hiding
                self.set_state('wall_following')
                self.get_logger().info(f'Hiding timeout ({self.timeout_duration}s) - hider resuming wall following')
            else:
                # Seeker should resume exploring
                self.set_state('exploring')
                self.get_logger().info(f'Hiding timeout ({self.timeout_duration}s) - seeker resuming exploration')
            
        elif self.game_state == 'stopped' and elapsed_time >= self.timeout_duration:
            # Stop timeout - resume appropriate behavior based on role
            if self.current_role == 'hider':
                self.set_state('wall_following')
                self.get_logger().info(f'Stop timeout ({self.timeout_duration}s) - hider resuming wall following')
            else:
                self.set_state('exploring')
                self.get_logger().info(f'Stop timeout ({self.timeout_duration}s) - seeker resuming exploration')
            
        elif self.game_state == 'random_walk' and elapsed_time >= self.random_walk_duration:
            # Random walk timeout - resume appropriate behavior based on role
            if self.current_role == 'hider':
                self.set_state('wall_following')
                self.get_logger().info(f'Random walk timeout ({self.random_walk_duration}s) - hider resuming wall following')
            else:
                self.set_state('exploring')
                self.get_logger().info(f'Random walk timeout ({self.random_walk_duration}s) - seeker resuming exploration')
    
    def get_state(self):
        """Get current game state"""
        return self.game_state
    
    def get_elapsed_time(self):
        """Get elapsed time in current state"""
        if self.state_start_time is None:
            return 0.0
        current_time = self.get_clock().now()
        return (current_time - self.state_start_time).nanoseconds / 1e9

    def movement_priority_callback(self, msg):
        """Handle movement priority updates to prevent command conflicts"""
        self.current_movement_priority = msg.data
        
        # Game coordinator can only publish movement if no higher priority node is active
        if msg.data.startswith("aruco_detector"):
            self.can_publish_movement = False
            self.get_logger().debug(f'ArUco detector has movement priority - stopping game coordinator movement')
        elif msg.data == "none":
            self.can_publish_movement = True
            self.get_logger().debug('Movement priority released - game coordinator can move')
        else:
            self.can_publish_movement = True  # Allow other nodes like hider_node

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

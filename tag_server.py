#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float64
from geometry_msgs.msg import Twist
import socket
import json
import threading
import time

class TagServer(Node):
    def __init__(self):
        super().__init__('tag_server')
        
        # Parameters for socket connection
        self.declare_parameter('server_host', '192.168.100.47')  # Change this to laptop IP
        self.declare_parameter('server_port', 8888)
        self.declare_parameter('connection_timeout', 5.0)
        self.declare_parameter('reconnect_interval', 2.0)
        self.declare_parameter('self_aruco_id', 0)
        
        self.server_host = self.get_parameter('server_host').get_parameter_value().string_value
        self.server_port = self.get_parameter('server_port').get_parameter_value().integer_value
        self.connection_timeout = self.get_parameter('connection_timeout').get_parameter_value().double_value
        self.reconnect_interval = self.get_parameter('reconnect_interval').get_parameter_value().double_value
        self.self_aruco_id = self.get_parameter('self_aruco_id').get_parameter_value().integer_value
        
        # Socket connection
        self.socket = None
        self.connected = False
        self.running = True
        self.last_data = None
        
        # ROS Publishers - publish received data to ROS topics
        self.server_status_pub = self.create_publisher(String, '/tag_server/status', 10)
        self.robot_cmd_pub = self.create_publisher(Twist, '/tag_server/robot_commands', 10)
        self.marker_data_pub = self.create_publisher(String, '/tag_server/marker_data', 10)
        self.distance_pub = self.create_publisher(Float64, '/tag_server/distance_to_target', 10)
        self.raw_data_pub = self.create_publisher(String, '/tag_server/raw_data', 10)
        self.server_response_pub = self.create_publisher(String, '/server_response', 10)
        
        # ROS Subscribers - receive commands to send to server
        self.server_command_sub = self.create_subscription(String, '/server_command', self.server_command_callback, 10)
        
        # Status tracking
        self.messages_received = 0
        self.last_message_time = None
        
        # Command queue for sending to server
        self.command_queue = []
        self.command_lock = threading.Lock()
        
        # Start connection thread
        self.connection_thread = threading.Thread(target=self.connection_loop, daemon=True)
        self.connection_thread.start()
        
        # Create timer for publishing status
        self.create_timer(1.0, self.publish_status)
        
        self.get_logger().info(f'Tag Server initialized - connecting to {self.server_host}:{self.server_port}')
        self.get_logger().info(f'Robot ID: {self.self_aruco_id}')

    def server_command_callback(self, msg):
        """Handle commands to send to the Python server"""
        try:
            command_data = json.loads(msg.data)
            
            # Add robot ID to command if not present
            if 'robot_id' not in command_data:
                command_data['robot_id'] = self.self_aruco_id
            
            # Add to command queue
            with self.command_lock:
                self.command_queue.append(command_data)
            
            self.get_logger().info(f'Queued command for server: {command_data}')
            
        except json.JSONDecodeError as e:
            self.get_logger().error(f'Invalid JSON in server command: {e}')
        except Exception as e:
            self.get_logger().error(f'Error processing server command: {e}')

    def send_command_to_server(self, command_data):
        """Send command data to Python server"""
        if not self.connected or not self.socket:
            self.get_logger().warn('Not connected to server - cannot send command')
            return False
        
        try:
            command_json = json.dumps(command_data) + '\n'
            self.socket.send(command_json.encode('utf-8'))
            self.get_logger().info(f'Sent command to server: {command_data}')
            return True
        except Exception as e:
            self.get_logger().error(f'Error sending command to server: {e}')
            self.connected = False
            return False

    def connect_to_server(self):
        """Establish connection to Python server"""
        try:
            if self.socket:
                self.socket.close()
            
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(self.connection_timeout)
            
            self.get_logger().info(f'Attempting to connect to {self.server_host}:{self.server_port}...')
            self.socket.connect((self.server_host, self.server_port))
            
            self.connected = True
            self.get_logger().info('Successfully connected to Python server!')
            return True
            
        except socket.timeout:
            self.get_logger().warn(f'Connection timeout to {self.server_host}:{self.server_port}')
            return False
        except ConnectionRefusedError:
            self.get_logger().warn(f'Connection refused by {self.server_host}:{self.server_port} - is the server running?')
            return False
        except Exception as e:
            self.get_logger().error(f'Connection error: {e}')
            return False

    def connection_loop(self):
        """Main connection loop running in separate thread"""
        buffer = ""
        
        while self.running:
            if not self.connected:
                # Try to connect
                if self.connect_to_server():
                    buffer = ""  # Reset buffer on new connection
                else:
                    time.sleep(self.reconnect_interval)
                    continue
            
            try:
                # Send any queued commands
                with self.command_lock:
                    while self.command_queue:
                        command = self.command_queue.pop(0)
                        if not self.send_command_to_server(command):
                            # Re-queue if send failed
                            self.command_queue.insert(0, command)
                            break
                
                # Set socket to non-blocking for receiving
                self.socket.settimeout(0.1)
                
                # Receive data from server
                data = self.socket.recv(1024).decode('utf-8')
                
                if not data:
                    # Server closed connection
                    self.get_logger().warn('Server closed connection')
                    self.connected = False
                    continue
                
                # Add to buffer
                buffer += data
                
                # Process complete messages (separated by newlines)
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    if line.strip():
                        self.process_received_data(line.strip())
                        
            except socket.timeout:
                # This is normal - just continue
                continue
            except ConnectionResetError:
                self.get_logger().warn('Connection reset by server')
                self.connected = False
            except Exception as e:
                self.get_logger().error(f'Socket error: {e}')
                self.connected = False
                
            # Small delay to prevent busy waiting
            time.sleep(0.01)
        
        # Cleanup
        if self.socket:
            self.socket.close()

    def process_received_data(self, data_str):
        """Process received JSON data and publish to ROS topics"""
        try:
            data = json.loads(data_str)
            self.last_data = data
            self.messages_received += 1
            self.last_message_time = self.get_clock().now()
            
            # Publish raw data
            raw_msg = String()
            raw_msg.data = data_str
            self.raw_data_pub.publish(raw_msg)
            
            # Process game state updates
            if 'game_state' in data:
                game_state = data['game_state']
                self.get_logger().debug(f'Received game state: {game_state}')
                
                # Check if this robot's state changed
                my_robot_id = str(self.self_aruco_id)
                if my_robot_id in game_state:
                    my_state = game_state[my_robot_id]
                    
                    # Publish server response with robot state
                    response_msg = String()
                    response_msg.data = json.dumps({
                        'command': 'role_update',
                        'robot_id': self.self_aruco_id,
                        'role': my_state.get('role'),
                        'status': my_state.get('status'),
                        'timestamp': time.time()
                    })
                    self.server_response_pub.publish(response_msg)
            
            # Process game events
            if 'game_events' in data:
                game_events = data['game_events']
                for event in game_events:
                    if event.get('type') == 'robot_tagged':
                        self.get_logger().info(f'Game event: {event}')
                    elif event.get('type') == 'timeout_recovery':
                        robot_id = event.get('robot_id')
                        if robot_id == self.self_aruco_id:
                            self.get_logger().info('Timeout recovery - can resume exploring')
                    elif event.get('type') == 'hiding_complete':
                        robot_id = event.get('robot_id')
                        if robot_id == self.self_aruco_id:
                            self.get_logger().info('Hiding complete - starting random walk')
            
            # Process and publish robot commands
            if 'robot_commands' in data:
                cmd_data = data['robot_commands']
                twist_msg = Twist()
                twist_msg.linear.x = float(cmd_data.get('linear_x', 0.0))
                twist_msg.angular.z = float(cmd_data.get('angular_z', 0.0))
                self.robot_cmd_pub.publish(twist_msg)
                
                self.get_logger().debug(f'Published robot command: linear={twist_msg.linear.x:.3f}, angular={twist_msg.angular.z:.3f}')
            
            # Process and publish marker data
            if 'marker_data' in data:
                marker_data = data['marker_data']
                
                # Publish marker data as JSON string
                marker_msg = String()
                marker_msg.data = json.dumps(marker_data)
                self.marker_data_pub.publish(marker_msg)
                
                # Publish distance separately
                distance_msg = Float64()
                distance_msg.data = float(marker_data.get('distance_to_target', 0.0))
                self.distance_pub.publish(distance_msg)
                
                if marker_data.get('detected_markers'):
                    self.get_logger().debug(f'Markers detected: {marker_data["detected_markers"]}, target: {marker_data.get("target_marker")}, distance: {distance_msg.data:.2f}m')
            
        except json.JSONDecodeError as e:
            self.get_logger().error(f'JSON decode error: {e}')
        except Exception as e:
            self.get_logger().error(f'Data processing error: {e}')

    def publish_status(self):
        """Publish connection status"""
        status_msg = String()
        
        if self.connected and self.last_message_time:
            time_since_last = (self.get_clock().now() - self.last_message_time).nanoseconds / 1e9
            if time_since_last < 1.0:  # Recent data
                status = f"CONNECTED - {self.messages_received} messages received"
            else:
                status = f"CONNECTED but no recent data ({time_since_last:.1f}s ago)"
        else:
            status = f"DISCONNECTED - attempting to reconnect to {self.server_host}:{self.server_port}"
        
        status_msg.data = status
        self.server_status_pub.publish(status_msg)
        
        # Log status periodically
        if self.messages_received % 50 == 0 and self.messages_received > 0:
            self.get_logger().info(f'Status: {status}')

    def get_latest_data(self):
        """Get the most recent data received from server"""
        return self.last_data

    def is_connected(self):
        """Check if connected to server"""
        return self.connected

    def shutdown(self):
        """Graceful shutdown"""
        self.get_logger().info('Shutting down Tag Server...')
        self.running = False
        if self.socket:
            self.socket.close()

def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = TagServer()
        node.get_logger().info('Tag Server started - receiving data from Python server!')
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if 'node' in locals():
            node.shutdown()
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 
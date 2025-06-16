#!/usr/bin/env python3

import socket
import threading
import json
import time
import logging
from typing import Any

# Configuration - Change this IP to your laptop's IP address
LAPTOP_IP = "192.168.100.47"  # Change this to your laptop's IP

class PythonServer:
    def __init__(self, host=None, port=8888):
        """
        Initialize the Python server for socket communication with ROS nodes
        
        Args:
            host (str): Server host address. If None, will use LAPTOP_IP
            port (int): Server port number
        """
        self.host = host if host is not None else LAPTOP_IP
        self.port = port
        self.server_socket = None
        self.client_connections = []
        self.running = False

        # Game state management
        self.robot_tagger_state = {
            '0': {
                'role': 'seeker',
                'status': 'active',
                'last_tagged_time': None,
                'tagged_by': None,
                'tagged_count': 0
            },
            '1': {
                'role': 'hider',
                'status': 'active',
                'last_tagged_time': None,
                'tagged_by': None,
                'tagged_count': 0
            },
            '2': {
                'role': 'hider',
                'status': 'active',
                'last_tagged_time': None,
                'tagged_by': None,
                'tagged_count': 0
            }
        }
        
        # Game events queue for processing
        self.game_events = []
        self.events_lock = threading.Lock()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Data to send to clients
        self.shared_data = {
            'timestamp': time.time(),
            'server_status': 'active',
            'message_count': 0,
            'game_state': self.robot_tagger_state,
            'robot_commands': {
                'linear_x': 0.0,
                'angular_z': 0.0,
                'action': 'idle'
            },
            'marker_data': {
                'detected_markers': [],
                'target_marker': None,
                'distance_to_target': 0.0
            },
            'game_events': []
        }
        
    def process_tag_event(self, tagger_id: int, tagged_id: int):
        """Process a robot tagging event and update roles"""
        self.logger.info(f"Processing tag event: Robot {tagger_id} tagged Robot {tagged_id}")
        
        tagger_key = str(tagger_id)
        tagged_key = str(tagged_id)
        
        current_time = time.time()
        
        # Validate robot IDs
        if tagger_key not in self.robot_tagger_state or tagged_key not in self.robot_tagger_state:
            self.logger.error(f"Invalid robot IDs: tagger={tagger_key}, tagged={tagged_key}")
            return
        
        # Update tagged robot state
        self.robot_tagger_state[tagged_key]['last_tagged_time'] = current_time
        self.robot_tagger_state[tagged_key]['tagged_by'] = tagger_id
        self.robot_tagger_state[tagged_key]['tagged_count'] += 1
        self.robot_tagger_state[tagged_key]['status'] = 'tagged_timeout'
        
        # Update tagger robot state
        self.robot_tagger_state[tagger_key]['status'] = 'hiding'
        
        # Role assignment: The tagged robot will become seeker after timeout
        # This happens in process_timeout_recovery() after 10 seconds
        self.logger.info(f"Robot {tagged_id} will become seeker after 10-second timeout")
        self.logger.info(f"Robot {tagger_id} is now hiding")
        
        # Add event to shared data
        tag_event = {
            'type': 'robot_tagged',
            'tagger_id': tagger_id,
            'tagged_id': tagged_id,
            'timestamp': current_time,
            'new_roles': {
                str(robot_id): state['role'] 
                for robot_id, state in self.robot_tagger_state.items()
            }
        }
        
        # Add to events queue
        with self.events_lock:
            self.game_events.append(tag_event)
            # Keep only last 10 events
            if len(self.game_events) > 10:
                self.game_events = self.game_events[-10:]
        
        self.logger.info(f"Updated game state: {self.robot_tagger_state}")
        
    def process_timeout_recovery(self):
        """Process robots recovering from timeout"""
        current_time = time.time()
        timeout_duration = 10.0  # 10 seconds (changed from 20)
        
        for robot_id, state in self.robot_tagger_state.items():
            if (state['status'] == 'tagged_timeout' and 
                state['last_tagged_time'] and 
                current_time - state['last_tagged_time'] >= timeout_duration):
                
                # Timeout recovered - robot can resume exploring
                state['status'] = 'active'
                
                # Role switching: Tagged robot becomes the seeker
                # First, find the current seeker and make them a hider
                current_seeker = None
                for rid, rstate in self.robot_tagger_state.items():
                    if rstate['role'] == 'seeker':
                        current_seeker = rid
                        break
                
                if current_seeker:
                    self.robot_tagger_state[current_seeker]['role'] = 'hider'
                    self.logger.info(f"Role switch: Robot {current_seeker} is now hider")
                
                # Tagged robot becomes the new seeker
                state['role'] = 'seeker'
                self.logger.info(f"Robot {robot_id} timeout recovered - resuming active state as seeker")
                
                # Add recovery event
                recovery_event = {
                    'type': 'timeout_recovery',
                    'robot_id': int(robot_id),
                    'timestamp': current_time,
                    'new_role': 'seeker'
                }
                
                with self.events_lock:
                    self.game_events.append(recovery_event)
                    if len(self.game_events) > 10:
                        self.game_events = self.game_events[-10:]
    
    def process_hiding_timeout(self):
        """Process robots finishing hiding behavior"""
        current_time = time.time()
        hiding_duration = 20.0  # 20 seconds
        
        for robot_id, state in self.robot_tagger_state.items():
            if (state['status'] == 'hiding' and 
                state['last_tagged_time'] and 
                current_time - state['last_tagged_time'] >= hiding_duration):
                
                # Hiding completed - robot should start random walk
                state['status'] = 'random_walk'
                self.logger.info(f"Robot {robot_id} hiding completed - starting random walk")
                
                # Add hiding completion event
                hiding_event = {
                    'type': 'hiding_complete',
                    'robot_id': int(robot_id),
                    'timestamp': current_time
                }
                
                with self.events_lock:
                    self.game_events.append(hiding_event)
                    if len(self.game_events) > 10:
                        self.game_events = self.game_events[-10:]
    
    def add_game_event(self, event_data):
        """Add a game event from external source"""
        try:
            if isinstance(event_data, str):
                event = json.loads(event_data)
            else:
                event = event_data
            
            if event.get('command') == 'robot_tagged':
                tagger_id = event.get('tagger_id')
                tagged_id = event.get('tagged_id')
                
                if tagger_id is not None and tagged_id is not None:
                    self.process_tag_event(tagger_id, tagged_id)
                else:
                    self.logger.error(f"Invalid tag event data: {event}")
            else:
                self.logger.info(f"Unknown game event: {event}")
                
        except Exception as e:
            self.logger.error(f"Error processing game event: {e}")

    def start_server(self):
        """Start the server and listen for connections"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            self.running = True
            
            self.logger.info(f"Python Server started on {self.host}:{self.port}")
            self.logger.info("Waiting for ROS node connections...")
            
            # Start data update thread
            data_thread = threading.Thread(target=self.update_data_loop, daemon=True)
            data_thread.start()
            
            # Start game logic thread
            game_thread = threading.Thread(target=self.game_logic_loop, daemon=True)
            game_thread.start()
            
            # Accept connections
            while self.running:
                try:
                    client_socket, client_address = self.server_socket.accept()
                    self.logger.info(f"New connection from {client_address}")
                    
                    # Handle client in separate thread
                    client_thread = threading.Thread(
                        target=self.handle_client,
                        args=(client_socket, client_address),
                        daemon=True
                    )
                    client_thread.start()
                    
                except socket.error as e:
                    if self.running:
                        self.logger.error(f"Socket error: {e}")
                    break
                    
        except Exception as e:
            self.logger.error(f"Server startup error: {e}")
        finally:
            self.cleanup()
    
    def game_logic_loop(self):
        """Game logic processing loop"""
        while self.running:
            try:
                # Process timeout recoveries
                self.process_timeout_recovery()
                
                # Process hiding timeouts
                self.process_hiding_timeout()
                
                # Sleep for a short interval
                time.sleep(1.0)  # Check every second
                
            except Exception as e:
                self.logger.error(f"Error in game logic loop: {e}")
    
    def handle_client(self, client_socket, client_address):
        """Handle individual client connection"""
        self.client_connections.append(client_socket)
        buffer = ""
        
        try:
            while self.running:
                # Set socket to non-blocking for sending/receiving
                client_socket.settimeout(0.1)
                
                # Try to receive data from client
                try:
                    data = client_socket.recv(1024).decode('utf-8')
                    if data:
                        buffer += data
                        # Process complete messages (separated by newlines)
                        while '\n' in buffer:
                            line, buffer = buffer.split('\n', 1)
                            if line.strip():
                                self.process_client_command(line.strip(), client_address)
                except socket.timeout:
                    # No data received, continue to sending
                    pass
                except socket.error:
                    # Client disconnected
                    break
                
                # Send data to client
                data_json = json.dumps(self.shared_data) + '\n'
                try:
                    client_socket.send(data_json.encode('utf-8'))
                    time.sleep(0.1)
                except socket.error as e:
                    self.logger.warning(f"Client {client_address} disconnected: {e}")
                    break
                    
        except Exception as e:
            self.logger.error(f"Error handling client {client_address}: {e}")
        finally:
            # Remove client from list
            if client_socket in self.client_connections:
                self.client_connections.remove(client_socket)
            client_socket.close()
            self.logger.info(f"Client {client_address} connection closed")
    
    def process_client_command(self, command_str, client_address):
        """Process command received from client"""
        try:
            command = json.loads(command_str)
            self.logger.info(f"Received command from {client_address}: {command}")
            
            # Process the command
            if command.get('command') == 'robot_tagged':
                self.add_game_event(command)
            else:
                self.logger.info(f"Unknown command from {client_address}: {command}")
                
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON from {client_address}: {e}")
        except Exception as e:
            self.logger.error(f"Error processing command from {client_address}: {e}")
    
    def update_data_loop(self):
        """Continuously update the data to be sent to clients"""
        while self.running:
            # Update timestamp and message count
            self.shared_data['timestamp'] = time.time()
            self.shared_data['message_count'] += 1
            
            # Update game state
            self.shared_data['game_state'] = self.robot_tagger_state.copy()
            
            # Update game events
            with self.events_lock:
                self.shared_data['game_events'] = self.game_events.copy()
            
            # Simulate robot movement commands based on game state
            import math
            t = time.time()
            
            # Default idle commands
            self.shared_data['robot_commands'] = {
                'linear_x': 0.0,
                'angular_z': 0.0,
                'action': 'idle'
            }
            
            # Simulate marker detection data
            if self.shared_data['message_count'] % 100 < 30:
                self.shared_data['marker_data'] = {
                    'detected_markers': [0, 1, 2],
                    'target_marker': 1,
                    'distance_to_target': abs(2.0 * math.sin(t * 0.2))
                }
            else:
                self.shared_data['marker_data'] = {
                    'detected_markers': [],
                    'target_marker': None,
                    'distance_to_target': 0.0
                }
            
            time.sleep(0.05)  # Update at 20 Hz
    
    def update_robot_command(self, linear_x: float, angular_z: float, action: str = 'custom'):
        """Update robot command data to be sent to clients"""
        self.shared_data['robot_commands'] = {
            'linear_x': linear_x,
            'angular_z': angular_z,
            'action': action
        }
        self.logger.info(f"Updated robot command: linear_x={linear_x}, angular_z={angular_z}, action={action}")
    
    def update_marker_data(self, detected_markers: list, target_marker: int = None, distance: float = 0.0):
        """Update marker detection data to be sent to clients"""
        self.shared_data['marker_data'] = {
            'detected_markers': detected_markers,
            'target_marker': target_marker,
            'distance_to_target': distance
        }
        self.logger.info(f"Updated marker data: detected={detected_markers}, target={target_marker}, distance={distance}")
    
    def broadcast_custom_data(self, data_key: str, data_value: Any):
        """Add custom data to be broadcast to all clients"""
        self.shared_data[data_key] = data_value
        self.logger.info(f"Broadcasting custom data: {data_key} = {data_value}")
    
    def get_connected_clients_count(self) -> int:
        """Get number of connected clients"""
        return len(self.client_connections)
    
    def stop_server(self):
        """Stop the server gracefully"""
        self.logger.info("Stopping Python Server...")
        self.running = False
        if self.server_socket:
            self.server_socket.close()
    
    def cleanup(self):
        """Clean up resources"""
        # Close all client connections
        for client_socket in self.client_connections:
            try:
                client_socket.close()
            except:
                pass
        self.client_connections.clear()
        
        # Close server socket
        if self.server_socket:
            self.server_socket.close()
        
        self.logger.info("Python Server stopped and cleaned up")

def main():
    """Main function to run the server"""
    print(f"Starting server on IP: {LAPTOP_IP}")
    server = PythonServer(host=LAPTOP_IP, port=8888)
    
    try:
        # Start server in main thread
        server.start_server()
    except KeyboardInterrupt:
        print("\nReceived interrupt signal...")
    finally:
        server.stop_server()
        print("Server shutdown complete")

if __name__ == '__main__':
    main() 
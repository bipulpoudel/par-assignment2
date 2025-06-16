#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Bool
from cv_bridge import CvBridge
import cv2
import numpy as np
from cv2 import aruco
import math

class DetectorAndSeekrNode(Node):
    def __init__(self):
        super().__init__('detector_and_seeker')
        
        # Parameters
        self.declare_parameter('image_topic', '/oak/rgb/image_raw')
        self.declare_parameter('camera_info_topic', '/oak/rgb/camera_info')
        self.declare_parameter('aruco_dict_id', 0)
        self.declare_parameter('aruco_marker_id', 0)  # This robot's marker ID
        self.declare_parameter('marker_length', 0.05)
        self.declare_parameter('max_linear_speed', 0.8)  # Increased to 1.0 m/s for high speed
        self.declare_parameter('max_angular_speed', 1.2)  # Increased to 1.2 rad/s for fast turning
        self.declare_parameter('stop_distance', 0.7)  # Slightly closer approach for efficiency
        
        # Obstacle detection parameters - optimized for continuous movement
        self.declare_parameter('enable_obstacle_detection', False)
        self.declare_parameter('obstacle_critical_distance', 0.25)  # Very slow distance (no full stop)
        self.declare_parameter('obstacle_slow_distance', 0.6)  # Start slowing down distance
        self.declare_parameter('side_obstacle_distance', 0.25)  # Side obstacle detection
        self.declare_parameter('forward_angle_range', 25.0)  # Narrower forward detection
        self.declare_parameter('side_angle_range', 35.0)  # Narrower side detection
        self.declare_parameter('min_continuous_speed', 0.08)  # Minimum speed to keep moving
        
        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        camera_info_topic = self.get_parameter('camera_info_topic').get_parameter_value().string_value
        aruco_dict_id = self.get_parameter('aruco_dict_id').get_parameter_value().integer_value
        self.aruco_marker_id = self.get_parameter('aruco_marker_id').get_parameter_value().integer_value
        self.marker_length = self.get_parameter('marker_length').get_parameter_value().double_value
        self.max_linear_speed = self.get_parameter('max_linear_speed').get_parameter_value().double_value
        self.max_angular_speed = self.get_parameter('max_angular_speed').get_parameter_value().double_value
        self.stop_distance = self.get_parameter('stop_distance').get_parameter_value().double_value
        
        # Obstacle detection parameters
        self.enable_obstacle_detection = self.get_parameter('enable_obstacle_detection').get_parameter_value().bool_value
        self.obstacle_critical_distance = self.get_parameter('obstacle_critical_distance').get_parameter_value().double_value
        self.obstacle_slow_distance = self.get_parameter('obstacle_slow_distance').get_parameter_value().double_value
        self.side_obstacle_distance = self.get_parameter('side_obstacle_distance').get_parameter_value().double_value
        self.forward_angle_range = self.get_parameter('forward_angle_range').get_parameter_value().double_value
        self.side_angle_range = self.get_parameter('side_angle_range').get_parameter_value().double_value
        self.min_continuous_speed = self.get_parameter('min_continuous_speed').get_parameter_value().double_value
        
        # State
        self.bridge = CvBridge()
        self.camera_matrix = None
        self.dist_coeffs = None
        self.camera_info_received = False
        self.visited_markers = set()
        self.last_marker_time = None
        self.seeking = False  # Only start moving when seeking is True
        
        # Obstacle detection state
        self.latest_scan = None
        self.forward_obstacle_distance = float('inf')
        self.left_obstacle_distance = float('inf')
        self.right_obstacle_distance = float('inf')
        self.obstacle_detected = False
        
        # Smoothed obstacle distances to prevent oscillation
        self.smooth_forward_distance = float('inf')
        self.smooth_left_distance = float('inf')
        self.smooth_right_distance = float('inf')
        self.obstacle_smoothing = 0.7  # Smoothing factor for obstacle distances
        
        # Smooth movement parameters - optimized for speed + smoothness
        self.declare_parameter('smoothing_factor', 0.5)  # Increased for more responsiveness
        self.declare_parameter('max_acceleration', 1.0)  # Increased to 1.5 m/s² for faster response
        self.declare_parameter('max_angular_acceleration', 2.5)  # Increased to 2.5 rad/s² for snappy turns
        
        self.smoothing_factor = self.get_parameter('smoothing_factor').get_parameter_value().double_value
        self.max_acceleration = self.get_parameter('max_acceleration').get_parameter_value().double_value
        self.max_angular_acceleration = self.get_parameter('max_angular_acceleration').get_parameter_value().double_value
        
        # Previous command tracking for smooth transitions
        self.prev_linear_x = 0.0
        self.prev_angular_z = 0.0
        self.last_cmd_time = None
        
        # ArUco setup
        self.aruco_dict = aruco.getPredefinedDictionary(aruco_dict_id)
        try:
            self.aruco_params = aruco.DetectorParameters()
        except AttributeError:
            self.aruco_params = aruco.DetectorParameters_create()
        
        # Publishers and subscribers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.robot_tagged_pub = self.create_publisher(String, '/robot_tagged', 10)
        self.explore_trigger_pub = self.create_publisher(Bool, '/explore_trigger', 10)
        self.image_sub = self.create_subscription(Image, image_topic, self.image_callback, 10)
        self.camera_info_sub = self.create_subscription(CameraInfo, camera_info_topic, self.camera_info_callback, 10)

        # Obstacle detection subscription
        if self.enable_obstacle_detection:
            self.laser_sub = self.create_subscription(
                LaserScan,
                '/scan',
                self.laser_callback,
                10)
            self.get_logger().info('Obstacle detection enabled - subscribed to /scan')
        
        # Safety timer - stop if no markers seen for a while and resume exploring
        self.create_timer(0.5, self.safety_check)
        
        self.get_logger().info('Simple ArUco Mover initialized')
        self.get_logger().info(f'Robot marker ID: {self.aruco_marker_id}')
        self.get_logger().info(f'Will stop at {self.stop_distance}m from markers and publish /robot_tagged')
        self.get_logger().info(f'Smooth movement enabled: smoothing={self.smoothing_factor}, max_accel={self.max_acceleration}m/s²')
        self.get_logger().info(f'Seeking mode: {self.seeking} - robot will only move when seeking=True')
        self.get_logger().info('Subscribed to /explore_trigger to control seeking behavior')
        if self.enable_obstacle_detection:
            self.get_logger().info(f'Continuous obstacle avoidance: critical={self.obstacle_critical_distance}m, slow={self.obstacle_slow_distance}m')

    def camera_info_callback(self, msg):
        if not self.camera_info_received:
            self.camera_matrix = np.array(msg.k).reshape(3, 3)
            self.dist_coeffs = np.array(msg.d)
            self.camera_info_received = True
            self.get_logger().info('Camera calibration received')

    def laser_callback(self, msg):
        """Process laser scan for obstacle detection in forward and side directions only"""
        if not self.enable_obstacle_detection:
            return
            
        self.latest_scan = msg
        
        # Reset obstacle distances
        self.forward_obstacle_distance = float('inf')
        self.left_obstacle_distance = float('inf')
        self.right_obstacle_distance = float('inf')
        
        if not msg.ranges:
            return
        
        num_ranges = len(msg.ranges)
        angle_increment = msg.angle_increment
        angle_min = msg.angle_min
        
        # Convert angle ranges to indices
        forward_angle_rad = math.radians(self.forward_angle_range / 2)
        side_angle_rad = math.radians(self.side_angle_range)
        
        # Find forward, left, and right regions (ignore backward)
        for i, range_val in enumerate(msg.ranges):
            if range_val < msg.range_min or range_val > msg.range_max:
                continue
                
            angle = angle_min + i * angle_increment
            
            # Forward detection (±forward_angle_range degrees from front)
            if abs(angle) <= forward_angle_rad:
                self.forward_obstacle_distance = min(self.forward_obstacle_distance, range_val)
            
            # Left side detection (0 to side_angle_range degrees)
            elif 0 < angle <= side_angle_rad:
                self.left_obstacle_distance = min(self.left_obstacle_distance, range_val)
            
            # Right side detection (-side_angle_range to 0 degrees)
            elif -side_angle_rad <= angle < 0:
                self.right_obstacle_distance = min(self.right_obstacle_distance, range_val)
        
        # Apply smoothing to obstacle distances to prevent oscillation
        alpha = self.obstacle_smoothing
        if self.smooth_forward_distance == float('inf'):
            # Initialize smoothed values
            self.smooth_forward_distance = self.forward_obstacle_distance
            self.smooth_left_distance = self.left_obstacle_distance
            self.smooth_right_distance = self.right_obstacle_distance
        else:
            # Smooth the obstacle distances
            self.smooth_forward_distance = (alpha * self.smooth_forward_distance + 
                                          (1 - alpha) * self.forward_obstacle_distance)
            self.smooth_left_distance = (alpha * self.smooth_left_distance + 
                                       (1 - alpha) * self.left_obstacle_distance)
            self.smooth_right_distance = (alpha * self.smooth_right_distance + 
                                        (1 - alpha) * self.right_obstacle_distance)
        
        # Check for obstacle detection using smoothed values
        self.obstacle_detected = (
            self.smooth_forward_distance < self.obstacle_slow_distance or
            self.smooth_left_distance < self.side_obstacle_distance or
            self.smooth_right_distance < self.side_obstacle_distance
        )
        
        # Log obstacle status periodically
        if self.obstacle_detected:
            self.get_logger().debug(f'Smooth obstacles: forward={self.smooth_forward_distance:.2f}m, '
                                  f'left={self.smooth_left_distance:.2f}m, right={self.smooth_right_distance:.2f}m')

    def image_callback(self, msg):
        if not self.camera_info_received:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge exception: {e}')
            return

        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

        cmd = Twist()  # Default: stop

        if ids is not None:
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                corners, self.marker_length, self.camera_matrix, self.dist_coeffs)
            
            # Find closest unvisited marker
            closest_marker = None
            closest_distance = float('inf')
            closest_tvec = None
            
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id not in self.visited_markers:
                    tvec = tvecs[i][0]
                    distance = np.linalg.norm(tvec)
                    if distance < closest_distance:
                        closest_distance = distance
                        closest_marker = marker_id
                        closest_tvec = tvec
            
            if closest_marker is not None:
                self.last_marker_time = self.get_clock().now()
                
                # Send False to explore_trigger when marker is detected
                explore_msg = Bool()
                explore_msg.data = False
                self.explore_trigger_pub.publish(explore_msg)
                
                # Set seeking to True when marker is detected
                self.seeking = True
                
                # Only calculate movement if seeking is enabled
                if self.seeking:
                    cmd = self.calculate_movement(closest_marker, closest_tvec, closest_distance)
                else:
                    self.get_logger().info('Marker detected but seeking disabled - not moving')
        
        # Apply smooth movement before publishing, but only if seeking is enabled
        if self.seeking:
            smooth_cmd = self.apply_smoothing(cmd)
            self.cmd_vel_pub.publish(smooth_cmd)
        else:
            # If not seeking, publish stop command
            stop_cmd = Twist()
            self.cmd_vel_pub.publish(stop_cmd)

    def calculate_movement(self, marker_id, tvec, distance):
        """Calculate movement command based on marker position"""
        cmd = Twist()
        
        # If close enough, stop and mark as visited
        if distance <= self.stop_distance:
            self.visited_markers.add(marker_id)
            self.get_logger().info(f'Reached marker {marker_id}! Distance: {distance:.2f}m - ROBOT TAGGED!')
            
            # Publish robot_tagged message with tagger_id,tagged_id format
            tagged_msg = String()
            tagged_msg.data = f"{self.aruco_marker_id},{marker_id}"  # Format: "tagger_id,tagged_id"
            self.robot_tagged_pub.publish(tagged_msg)
            self.get_logger().info(f'Published /robot_tagged: Robot {self.aruco_marker_id} tagged Robot {marker_id}')
            
            # Send True to explore_trigger after tagging to resume exploring
            explore_msg = Bool()
            explore_msg.data = True
            self.explore_trigger_pub.publish(explore_msg)
            
            # Force complete stop - reset smoothing values to ensure no residual movement
            self.prev_linear_x = 0.0
            self.prev_angular_z = 0.0
            return cmd  # Stop (zero velocity)
        
        # DEBUG: Print raw tvec values to understand coordinate system
        self.get_logger().info(f'DEBUG - Marker {marker_id} tvec: x={tvec[0]:.3f}, y={tvec[1]:.3f}, z={tvec[2]:.3f}')
        
        # Calculate angle to marker (tvec[0] is x offset, tvec[2] is forward distance)
        angle_to_marker = math.atan2(tvec[0], tvec[2])
        
        # DEBUG: Determine movement direction based on tvec
        if tvec[2] > 0:
            # Marker is in front - move forward
            move_direction = 1.0
            direction_str = "FORWARD (marker ahead)"
        else:
            # Marker is behind - move backward or turn around
            move_direction = -1.0
            direction_str = "BACKWARD (marker behind)"
        
        # Forward speed control with correct direction - MORE AGGRESSIVE
        if distance > 2.0:
            # Far away: full speed
            cmd.linear.x = self.max_linear_speed * move_direction
        elif distance > 1.0:
            # Medium distance: 80% speed
            cmd.linear.x = self.max_linear_speed * 0.8 * move_direction
        else:
            # Close: scale with distance but keep minimum speed higher
            speed_factor = max(0.4, distance / 1.5)  # Minimum 40% speed, more aggressive scaling
            cmd.linear.x = self.max_linear_speed * speed_factor * move_direction
        
        # Simultaneous turning - MORE RESPONSIVE and FASTER
        if abs(angle_to_marker) > 0.8:  # Large angle (>46 degrees)
            # Less speed reduction, faster turning
            cmd.linear.x *= 0.7  # Less speed reduction (was 0.5)
            cmd.angular.z = -np.sign(angle_to_marker) * self.max_angular_speed * 0.9  # Faster turning
        elif abs(angle_to_marker) > 0.3:  # Medium angle (>17 degrees)  
            cmd.linear.x *= 0.85  # Minimal speed reduction (was 0.7)
            cmd.angular.z = -np.sign(angle_to_marker) * self.max_angular_speed * 0.7  # Good turning speed
        else:  # Small angle (<17 degrees)
            # Full speed forward with proportional turning
            cmd.angular.z = -angle_to_marker * 1.5  # More aggressive proportional turning
        
        # Apply obstacle avoidance if enabled
        if self.enable_obstacle_detection and self.latest_scan is not None:
            cmd = self.apply_obstacle_avoidance(cmd, marker_id)
        
        self.get_logger().info(f'Movement to marker {marker_id}: {direction_str}')
        self.get_logger().info(f'Distance={distance:.2f}m, angle={angle_to_marker:.2f}rad, cmd: linear.x={cmd.linear.x:.3f}, angular.z={cmd.angular.z:.3f}')
        
        return cmd

    def apply_obstacle_avoidance(self, cmd, marker_id):
        """Modify movement command for CONTINUOUS obstacle avoidance - never full stop"""
        original_linear = cmd.linear.x
        original_angular = cmd.angular.z
        
        # Use smoothed distances for stable behavior
        forward_dist = self.smooth_forward_distance
        left_dist = self.smooth_left_distance
        right_dist = self.smooth_right_distance
        
        # CONTINUOUS forward speed control - NEVER stop completely
        if forward_dist < self.obstacle_critical_distance:
            # Very close obstacle - move at minimum continuous speed
            speed_factor = max(0.15, forward_dist / self.obstacle_critical_distance)
            cmd.linear.x = max(self.min_continuous_speed, abs(cmd.linear.x) * speed_factor)
            if original_linear < 0:  # Preserve direction
                cmd.linear.x = -cmd.linear.x
            self.get_logger().info(f'Critical obstacle: {forward_dist:.2f}m, continuous speed: {cmd.linear.x:.2f}m/s')
        elif forward_dist < self.obstacle_slow_distance:
            # Moderate distance - smooth speed reduction
            speed_factor = 0.3 + 0.7 * (forward_dist / self.obstacle_slow_distance)
            cmd.linear.x *= speed_factor
            self.get_logger().debug(f'Slowing for obstacle: {forward_dist:.2f}m, factor: {speed_factor:.2f}')
        
        # CONTINUOUS side obstacle avoidance - gentle steering
        left_close = left_dist < self.side_obstacle_distance
        right_close = right_dist < self.side_obstacle_distance
        
        if left_close or right_close:
            if left_close and right_close:
                # Both sides close - slow down but keep moving
                cmd.linear.x *= 0.4
                cmd.linear.x = max(self.min_continuous_speed, abs(cmd.linear.x))
                if original_linear < 0:
                    cmd.linear.x = -cmd.linear.x
                # Slight wiggle to find way through
                cmd.angular.z += 0.1 * (1 if left_dist > right_dist else -1)
                self.get_logger().info(f'Tight space: left={left_dist:.2f}m, right={right_dist:.2f}m - continuous crawl')
            elif left_close:
                # Left obstacle - smooth turn right
                avoidance_strength = max(0.1, (self.side_obstacle_distance - left_dist) / self.side_obstacle_distance)
                cmd.angular.z += -0.4 * avoidance_strength  # Turn right (negative)
                cmd.linear.x *= (0.6 + 0.4 * (left_dist / self.side_obstacle_distance))
                self.get_logger().debug(f'Avoiding left: {left_dist:.2f}m, turn strength: {avoidance_strength:.2f}')
            elif right_close:
                # Right obstacle - smooth turn left
                avoidance_strength = max(0.1, (self.side_obstacle_distance - right_dist) / self.side_obstacle_distance)
                cmd.angular.z += 0.4 * avoidance_strength  # Turn left (positive)
                cmd.linear.x *= (0.6 + 0.4 * (right_dist / self.side_obstacle_distance))
                self.get_logger().debug(f'Avoiding right: {right_dist:.2f}m, turn strength: {avoidance_strength:.2f}')
        
        # Ensure minimum continuous movement
        if abs(cmd.linear.x) > 0.001 and abs(cmd.linear.x) < self.min_continuous_speed:
            cmd.linear.x = self.min_continuous_speed if cmd.linear.x > 0 else -self.min_continuous_speed
        
        # Log significant changes
        if abs(cmd.linear.x - original_linear) > 0.02 or abs(cmd.angular.z - original_angular) > 0.05:
            self.get_logger().debug(f'Continuous avoidance: linear {original_linear:.2f}→{cmd.linear.x:.2f}, angular {original_angular:.2f}→{cmd.angular.z:.2f}')
        
        return cmd

    def apply_smoothing(self, target_cmd):
        """Apply smooth acceleration and velocity filtering to commands"""
        current_time = self.get_clock().now()
        
        # Initialize timing
        if self.last_cmd_time is None:
            self.last_cmd_time = current_time
            self.prev_linear_x = target_cmd.linear.x
            self.prev_angular_z = target_cmd.angular.z
            return target_cmd
        
        # If target command is zero (stop), force immediate stop without smoothing
        if abs(target_cmd.linear.x) < 0.001 and abs(target_cmd.angular.z) < 0.001:
            self.get_logger().debug('Forcing immediate stop - no smoothing applied')
            self.prev_linear_x = 0.0
            self.prev_angular_z = 0.0
            return target_cmd  # Return zero command immediately
        
        # Calculate time step
        dt = (current_time - self.last_cmd_time).nanoseconds / 1e9
        self.last_cmd_time = current_time
        
        if dt > 0.1:  # Reset if too much time has passed (> 100ms)
            dt = 0.1
        
        smooth_cmd = Twist()
        
        # Smooth linear velocity with acceleration limits
        linear_diff = target_cmd.linear.x - self.prev_linear_x
        max_linear_change = self.max_acceleration * dt
        
        if abs(linear_diff) > max_linear_change:
            # Limit acceleration
            linear_change = max_linear_change * (1 if linear_diff > 0 else -1)
            smooth_cmd.linear.x = self.prev_linear_x + linear_change
        else:
            # Apply smoothing filter
            smooth_cmd.linear.x = (self.prev_linear_x * (1 - self.smoothing_factor) + 
                                 target_cmd.linear.x * self.smoothing_factor)
        
        # Smooth angular velocity with acceleration limits
        angular_diff = target_cmd.angular.z - self.prev_angular_z
        max_angular_change = self.max_angular_acceleration * dt
        
        if abs(angular_diff) > max_angular_change:
            # Limit angular acceleration
            angular_change = max_angular_change * (1 if angular_diff > 0 else -1)
            smooth_cmd.angular.z = self.prev_angular_z + angular_change
        else:
            # Apply smoothing filter
            smooth_cmd.angular.z = (self.prev_angular_z * (1 - self.smoothing_factor) + 
                                  target_cmd.angular.z * self.smoothing_factor)
        
        # Update previous values
        self.prev_linear_x = smooth_cmd.linear.x
        self.prev_angular_z = smooth_cmd.angular.z
        
        # Debug smoothing (optional)
        if abs(target_cmd.linear.x - smooth_cmd.linear.x) > 0.01 or abs(target_cmd.angular.z - smooth_cmd.angular.z) > 0.01:
            self.get_logger().debug(f'Smoothing: target=[{target_cmd.linear.x:.3f}, {target_cmd.angular.z:.3f}] → smooth=[{smooth_cmd.linear.x:.3f}, {smooth_cmd.angular.z:.3f}]')
        
        return smooth_cmd

    def safety_check(self):
        """Stop robot if no markers seen for too long and resume exploring"""
        if self.last_marker_time is not None:
            time_since_marker = (self.get_clock().now() - self.last_marker_time).nanoseconds / 1e9
            if time_since_marker > 2.0:  # 2 seconds without seeing a marker
                cmd = Twist()  # Stop
                # Force immediate stop for safety - reset smoothing
                self.prev_linear_x = 0.0
                self.prev_angular_z = 0.0
                self.cmd_vel_pub.publish(cmd)  # Publish zero directly
                self.seeking = False  # Disable seeking
                
                # Resume exploring when no markers are detected
                explore_msg = Bool()
                explore_msg.data = True
                self.explore_trigger_pub.publish(explore_msg)
                
                self.get_logger().debug('No markers detected - robot stopped and resuming exploration')

def main(args=None):
    rclpy.init(args=args)
    try:
        node = DetectorAndSeekrNode()
        node.get_logger().info('Simple ArUco navigation started - will move directly to detected markers!')
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if 'node' in locals():
            # Stop the robot before shutting down
            cmd = Twist()
            node.cmd_vel_pub.publish(cmd)
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
import tf2_ros
import math

from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Empty, String, Bool
from nav2_msgs.action import NavigateToPose

class HiderNode(Node):
    def __init__(self):
        super().__init__('hider_node')
        
        # Parameters
        self.declare_parameter('goal_tolerance', 0.5)    # Distance to consider goal reached
        self.declare_parameter('min_obstacle_distance', 0.5)  # Minimum distance from obstacles
        self.declare_parameter('search_resolution', 0.2)  # Resolution for searching map points
        self.declare_parameter('min_improvement', 1.0)    # Minimum distance improvement to switch goals
        self.declare_parameter('search_interval', 1.0)    # How often to search for better goals (reduced to 1 second)
        self.declare_parameter('self_aruco_id', 0)  # ArUco ID for game status messages
        self.declare_parameter('hiding_timeout', 30.0)  # Timeout for hiding behavior in seconds
        
        # Obstacle avoidance parameters (RE-ENABLED for hiding behavior)
        self.declare_parameter('obstacle_stop_distance', 0.4)    # Distance to stop for obstacles
        self.declare_parameter('obstacle_slow_distance', 0.8)    # Distance to slow down for obstacles
        self.declare_parameter('avoidance_angular_speed', 0.5)   # Angular speed for avoidance
        self.declare_parameter('max_linear_speed', 0.3)          # Maximum forward speed
        self.declare_parameter('side_clearance', 0.6)            # Required side clearance
        self.declare_parameter('emergency_stop_distance', 0.3)   # Emergency stop distance
        self.declare_parameter('path_clear_distance', 1.0)       # Distance to consider path clear ahead
        self.declare_parameter('obstacle_check_angle', 0.5)      # Angle range to check for obstacles (radians)
        
        # Get parameters
        self.goal_tolerance = self.get_parameter('goal_tolerance').get_parameter_value().double_value
        self.min_obstacle_distance = self.get_parameter('min_obstacle_distance').get_parameter_value().double_value
        self.search_resolution = self.get_parameter('search_resolution').get_parameter_value().double_value
        self.min_improvement = self.get_parameter('min_improvement').get_parameter_value().double_value
        self.search_interval = self.get_parameter('search_interval').get_parameter_value().double_value
        self.self_aruco_id = self.get_parameter('self_aruco_id').get_parameter_value().integer_value
        self.hiding_timeout = self.get_parameter('hiding_timeout').get_parameter_value().double_value
        
        # Obstacle avoidance parameters (RE-ENABLED)
        self.obstacle_stop_distance = self.get_parameter('obstacle_stop_distance').get_parameter_value().double_value
        self.obstacle_slow_distance = self.get_parameter('obstacle_slow_distance').get_parameter_value().double_value
        self.avoidance_angular_speed = self.get_parameter('avoidance_angular_speed').get_parameter_value().double_value
        self.max_linear_speed = self.get_parameter('max_linear_speed').get_parameter_value().double_value
        self.side_clearance = self.get_parameter('side_clearance').get_parameter_value().double_value
        self.emergency_stop_distance = self.get_parameter('emergency_stop_distance').get_parameter_value().double_value
        self.path_clear_distance = self.get_parameter('path_clear_distance').get_parameter_value().double_value
        self.obstacle_check_angle = self.get_parameter('obstacle_check_angle').get_parameter_value().double_value
        
        # State variables
        self.current_pose = None
        self.hiding_start_pose = None
        self.hiding_start_time = None
        self.hiding_active = False
        self.map_data = None
        self.navigation_goal_handle = None
        self.current_best_distance = 0.0  # Track the best distance found so far
        self.current_goal_pose = None     # Track current navigation goal
        self.last_search_time = None      # Track when we last searched for better goals
        self.goal_reached = False         # Track if we've reached our destination
        
        # Obstacle avoidance state (RE-ENABLED)
        self.latest_scan = None
        self.front_distance = float('inf')
        self.left_distance = float('inf')
        self.right_distance = float('inf')
        self.obstacle_detected = False
        self.emergency_stop = False
        self.avoidance_mode = False
        
        # 180-degree turn state for hiding behavior
        self.performing_turn = False
        self.turn_start_time = None
        self.turn_start_heading = None
        self.target_turn_duration = 3.0  # seconds to complete 180-degree turn
        self.turn_angular_speed = 1.0  # rad/s for turning
        
        # Navigation action client
        self.nav_action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        
        # TF setup
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Publishers
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.game_status_pub = self.create_publisher(String, '/game_status', 10)
        self.random_walk_pub = self.create_publisher(Bool, '/random_walk', 10)
        
        # Subscribers
        self.odom_sub = self.create_subscription(Odometry, '/odometry/filtered', self.odom_callback, 10)
        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)
        self.start_trigger_sub = self.create_subscription(Empty, '/start_hiding', self.start_hiding_callback, 10)
        self.stop_trigger_sub = self.create_subscription(Empty, '/stop_hiding', self.stop_hiding_callback, 10)
        self.laser_sub = self.create_subscription(LaserScan, '/scan', self.laser_callback, 10)  # RE-ENABLED
        
        # Timer for searching better goals during navigation (more frequent searches)
        self.search_timer = self.create_timer(self.search_interval, self.search_for_better_goal)
        
        # Timer for checking if goal is reached
        self.goal_check_timer = self.create_timer(0.5, self.check_goal_reached)
        
        # Timer for obstacle avoidance control loop (RE-ENABLED)
        self.avoidance_timer = self.create_timer(0.1, self.obstacle_avoidance_loop)
        
        # Timer for 180-degree turn control
        self.turn_timer = self.create_timer(0.1, self.turn_control_loop)
        
        # Timer for hiding timeout
        self.timeout_timer = self.create_timer(1.0, self.check_hiding_timeout)
        
        self.get_logger().info('🎯 Hider Node initialized with FRONTIER-BASED hiding and dynamic goal updating')
        self.get_logger().info(f'Goal tolerance: {self.goal_tolerance}m')
        self.get_logger().info(f'Minimum improvement for goal switch: {self.min_improvement}m')
        self.get_logger().info(f'Search interval: {self.search_interval}s')
        self.get_logger().info('🔍 FRONTIER DETECTION: Will target boundaries between known and unknown areas')
        self.get_logger().info('🚨 LASER-BASED OBSTACLE AVOIDANCE ENABLED for hiding behavior')
        self.get_logger().info(f'Emergency stop distance: {self.emergency_stop_distance}m')
        self.get_logger().info(f'Obstacle slow distance: {self.obstacle_slow_distance}m')
        self.get_logger().info(f'Side clearance required: {self.side_clearance}m')
        self.get_logger().info(f'Path clear distance: {self.path_clear_distance}m')
        self.get_logger().info('🔄 180° TURN-AROUND BEHAVIOR ENABLED')
        self.get_logger().info(f'Turn speed: {self.turn_angular_speed:.1f} rad/s, Duration: {self.target_turn_duration:.1f}s')
        self.get_logger().info(f'⏰ HIDING TIMEOUT: {self.hiding_timeout:.1f}s - will switch to random walk after timeout')
        self.get_logger().info('Waiting for map data and /start_hiding trigger...')

    def odom_callback(self, msg):
        """Update current pose from odometry"""
        self.current_pose = msg.pose.pose

    def map_callback(self, msg):
        """Store map data for finding farthest points"""
        self.map_data = msg

    def laser_callback(self, msg):
        """Process laser scan data for obstacle avoidance"""
        self.latest_scan = msg
        
        if not msg.ranges:
            return
        
        # Calculate front, left, and right distances
        num_readings = len(msg.ranges)
        
        # Front: center readings (straight ahead)
        front_start = max(0, num_readings // 2 - num_readings // 8)
        front_end = min(num_readings, num_readings // 2 + num_readings // 8)
        front_ranges = [r for r in msg.ranges[front_start:front_end] if not math.isinf(r) and not math.isnan(r)]
        self.front_distance = min(front_ranges) if front_ranges else float('inf')
        
        # Left side readings
        left_start = max(0, int(num_readings * 0.75))
        left_end = num_readings
        left_ranges = [r for r in msg.ranges[left_start:left_end] if not math.isinf(r) and not math.isnan(r)]
        self.left_distance = min(left_ranges) if left_ranges else float('inf')
        
        # Right side readings  
        right_start = 0
        right_end = min(num_readings, int(num_readings * 0.25))
        right_ranges = [r for r in msg.ranges[right_start:right_end] if not math.isinf(r) and not math.isnan(r)]
        self.right_distance = min(right_ranges) if right_ranges else float('inf')
        
        # Update obstacle detection status
        self.obstacle_detected = (
            self.front_distance < self.obstacle_slow_distance or
            self.left_distance < self.side_clearance or 
            self.right_distance < self.side_clearance
        )
        
        self.emergency_stop = self.front_distance < self.emergency_stop_distance



    def start_hiding_callback(self, msg):
        """Start the hiding behavior with 180-degree turn first, then path planning"""
        if not self.hiding_active and self.current_pose and self.map_data:
            # Stop random walk first
            random_walk_msg = Bool()
            random_walk_msg.data = True  # Stop random walking
            self.random_walk_pub.publish(random_walk_msg)
            
            self.hiding_active = True
            self.hiding_start_time = self.get_clock().now()
            self.hiding_start_pose = self.current_pose
            self.current_best_distance = 0.0
            self.current_goal_pose = None
            self.last_search_time = self.get_clock().now()
            
            self.get_logger().info('🏃 Starting hiding behavior with 180° turn first!')
            self.get_logger().info('📋 Navigation planning will start AFTER 180° turn is completed')
            
            # Publish game status for hiding start
            status_msg = String()
            status_msg.data = f"Robot {self.self_aruco_id} started hiding behavior - performing 180° turn first"
            self.game_status_pub.publish(status_msg)
            
            # Start with 180-degree turn before path planning
            self.start_180_turn()
        elif not self.map_data:
            self.get_logger().warn('Cannot start hiding: map data not available')
        elif not self.current_pose:
            self.get_logger().warn('Cannot start hiding: current pose not available')

    def stop_hiding_callback(self, msg):
        """Stop the hiding behavior"""
        if self.hiding_active:
            self.hiding_active = False
            self.cancel_navigation()
            self.current_best_distance = 0.0
            self.current_goal_pose = None
            
            # Stop 180-degree turn if in progress
            self.performing_turn = False
            self.turn_start_time = None
            self.turn_start_heading = None
            
            # Stop robot movement
            stop_cmd = Twist()
            self.cmd_vel_pub.publish(stop_cmd)
            
            # Stop random walk as well
            random_walk_msg = Bool()
            random_walk_msg.data = True  # Stop random walking
            self.random_walk_pub.publish(random_walk_msg)
            
            self.get_logger().info('Stopping hiding behavior and random walk')

    def world_to_map(self, world_x, world_y):
        """Convert world coordinates to map coordinates"""
        if not self.map_data:
            return None, None
        
        map_x = int((world_x - self.map_data.info.origin.position.x) / self.map_data.info.resolution)
        map_y = int((world_y - self.map_data.info.origin.position.y) / self.map_data.info.resolution)
        
        return map_x, map_y

    def map_to_world(self, map_x, map_y):
        """Convert map coordinates to world coordinates"""
        if not self.map_data:
            return None, None
        
        world_x = map_x * self.map_data.info.resolution + self.map_data.info.origin.position.x
        world_y = map_y * self.map_data.info.resolution + self.map_data.info.origin.position.y
        
        return world_x, world_y

    def is_point_free(self, map_x, map_y):
        """Check if a point in the map is free (navigable)"""
        if not self.map_data:
            return False
        
        width = self.map_data.info.width
        height = self.map_data.info.height
        
        # Check bounds
        if map_x < 0 or map_x >= width or map_y < 0 or map_y >= height:
            return False
        
        # Check if cell is free (0 = free, 100 = occupied, -1 = unknown)
        index = map_y * width + map_x
        if index >= len(self.map_data.data):
            return False
        
        cell_value = self.map_data.data[index]
        return cell_value == 0  # Free space

    def is_point_safe(self, map_x, map_y):
        """Check if a point is safe (free and has minimum distance from obstacles)"""
        if not self.is_point_free(map_x, map_y):
            return False
        
        # Check minimum distance from obstacles
        min_cells = int(self.min_obstacle_distance / self.map_data.info.resolution)
        
        for dx in range(-min_cells, min_cells + 1):
            for dy in range(-min_cells, min_cells + 1):
                check_x = map_x + dx
                check_y = map_y + dy
                
                # If any nearby cell is occupied, this point is not safe
                if not self.is_point_free(check_x, check_y):
                    # Check if it's actually an obstacle (not just unknown)
                    if (check_x >= 0 and check_x < self.map_data.info.width and
                        check_y >= 0 and check_y < self.map_data.info.height):
                        index = check_y * self.map_data.info.width + check_x
                        if index < len(self.map_data.data) and self.map_data.data[index] == 100:
                            return False
        
        return True

    def is_frontier_point(self, map_x, map_y):
        """Check if a point is a frontier (boundary between known free space and unknown space)"""
        if not self.map_data:
            return False
        
        width = self.map_data.info.width
        height = self.map_data.info.height
        
        # Point must be in bounds
        if map_x < 0 or map_x >= width or map_y < 0 or map_y >= height:
            return False
        
        # Point must be free space
        if not self.is_point_free(map_x, map_y):
            return False
        
        # Check if any neighboring cell is unknown (-1)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                
                neighbor_x = map_x + dx
                neighbor_y = map_y + dy
                
                # Check bounds
                if (neighbor_x >= 0 and neighbor_x < width and 
                    neighbor_y >= 0 and neighbor_y < height):
                    
                    index = neighbor_y * width + neighbor_x
                    if index < len(self.map_data.data):
                        # If neighbor is unknown (-1), this is a frontier
                        if self.map_data.data[index] == -1:
                            return True
        
        return False

    def find_farthest_frontier(self, verbose=True, from_current_position=False):
        """Find the farthest accessible frontier point in the known map"""
        if not self.map_data:
            return None, 0.0
            
        # Use current position if specified (for replanning) or if no start pose available
        if from_current_position and self.current_pose:
            start_x = self.current_pose.position.x
            start_y = self.current_pose.position.y
        elif self.hiding_start_pose:
            start_x = self.hiding_start_pose.position.x
            start_y = self.hiding_start_pose.position.y
        else:
            return None, 0.0
        
        best_point = None
        max_distance = 0.0
        frontiers_found = 0
        points_checked = 0
        
        # Search through the map with specified resolution
        width = self.map_data.info.width
        height = self.map_data.info.height
        resolution = self.map_data.info.resolution
        
        # Calculate step size based on search resolution
        step = max(1, int(self.search_resolution / resolution))
        
        if verbose:
            reference_point = "current position" if from_current_position else "start position"
            self.get_logger().info(f'🔍 Searching for frontier points from {reference_point} (boundaries of exploration)...')
        
        for map_y in range(0, height, step):
            for map_x in range(0, width, step):
                points_checked += 1
                
                # Check if this is a safe frontier point
                if self.is_frontier_point(map_x, map_y) and self.is_point_safe(map_x, map_y):
                    frontiers_found += 1
                    
                    # Convert to world coordinates
                    world_x, world_y = self.map_to_world(map_x, map_y)
                    
                    # Calculate distance from start position
                    dx = world_x - start_x
                    dy = world_y - start_y
                    distance = math.sqrt(dx*dx + dy*dy)
                    
                    if distance > max_distance:
                        max_distance = distance
                        best_point = PoseStamped()
                        best_point.header.frame_id = 'map'
                        best_point.header.stamp = self.get_clock().now().to_msg()
                        best_point.pose.position.x = world_x
                        best_point.pose.position.y = world_y
                        best_point.pose.position.z = 0.0
                        best_point.pose.orientation.w = 1.0
        
        if verbose:
            self.get_logger().info(f'Frontier search complete: {frontiers_found} frontiers found from {points_checked} points checked')
        
        if best_point:
            if verbose:
                self.get_logger().info(f'🎯 Farthest frontier found: {max_distance:.2f}m away (optimal hiding spot!)')
        else:
            if verbose:
                self.get_logger().warn('⚠️  No safe frontier points found! Falling back to farthest accessible point...')
            # Fallback to any farthest accessible point if no frontiers found
            return self.find_farthest_accessible_point_fallback(verbose, from_current_position)
        
        return best_point, max_distance

    def find_farthest_accessible_point_fallback(self, verbose=True, from_current_position=False):
        """Fallback method: Find the farthest accessible point in the known map (not necessarily frontier)"""
        if not self.map_data:
            return None, 0.0
            
        # Use current position if specified (for replanning) or if no start pose available
        if from_current_position and self.current_pose:
            start_x = self.current_pose.position.x
            start_y = self.current_pose.position.y
        elif self.hiding_start_pose:
            start_x = self.hiding_start_pose.position.x
            start_y = self.hiding_start_pose.position.y
        else:
            return None, 0.0
        
        best_point = None
        max_distance = 0.0
        safe_points_found = 0
        
        # Search through the map with specified resolution
        width = self.map_data.info.width
        height = self.map_data.info.height
        
        # Calculate step size based on search resolution
        step = max(1, int(self.search_resolution / self.map_data.info.resolution))
        
        for map_y in range(0, height, step):
            for map_x in range(0, width, step):
                if self.is_point_safe(map_x, map_y):
                    safe_points_found += 1
                    
                    # Convert to world coordinates
                    world_x, world_y = self.map_to_world(map_x, map_y)
                    
                    # Calculate distance from start position
                    dx = world_x - start_x
                    dy = world_y - start_y
                    distance = math.sqrt(dx*dx + dy*dy)
                    
                    if distance > max_distance:
                        max_distance = distance
                        best_point = PoseStamped()
                        best_point.header.frame_id = 'map'
                        best_point.header.stamp = self.get_clock().now().to_msg()
                        best_point.pose.position.x = world_x
                        best_point.pose.position.y = world_y
                        best_point.pose.position.z = 0.0
                        best_point.pose.orientation.w = 1.0
        
        if best_point:
            if verbose:
                self.get_logger().info(f'📍 Fallback: Farthest accessible point: {max_distance:.2f}m away ({safe_points_found} safe points found)')
        else:
            if verbose:
                self.get_logger().warn('❌ No safe accessible point found in map!')
        
        return best_point, max_distance

    def navigate_to_farthest_point(self):
        """Send navigation goal to the farthest frontier point"""
        # Don't start navigation if still performing 180-degree turn
        if self.performing_turn:
            self.get_logger().info('Still performing 180° turn - navigation will start after turn completion')
            return
            
        if not self.nav_action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Navigation action server not available!')
            return
        
        target_pose, target_distance = self.find_farthest_frontier()
        if not target_pose:
            self.get_logger().error('Could not find a safe target frontier!')
            return
        
        # Update our tracking variables
        self.current_best_distance = target_distance
        self.current_goal_pose = target_pose
        
        # Create navigation goal
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = target_pose
        
        # Publish goal for visualization
        self.goal_pub.publish(target_pose)
        
        self.get_logger().info(f'🎯 Navigating to frontier hiding point: {target_distance:.2f}m away')
        
        # Send goal to navigation
        future = self.nav_action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.navigation_feedback_callback
        )
        future.add_done_callback(self.navigation_goal_response_callback)

    def search_for_better_goal(self):
        """Continuously search for better hiding goals during navigation"""
        if not self.hiding_active or not self.hiding_start_pose or self.goal_reached:
            return
            
        # Don't search for goals while performing 180-degree turn
        if self.performing_turn:
            return
        
        # Don't search too frequently but more often than before
        current_time = self.get_clock().now()
        if self.last_search_time:
            time_since_last_search = (current_time - self.last_search_time).nanoseconds / 1e9
            if time_since_last_search < self.search_interval:
                return
        
        self.last_search_time = current_time
        
        # Find the current best frontier point
        better_point, better_distance = self.find_farthest_frontier(verbose=False)
        
        # More aggressive goal switching - use smaller improvement threshold
        improvement_threshold = max(0.5, self.min_improvement * 0.5)  # At least 0.5m improvement
        
        if better_point and better_distance > self.current_best_distance + improvement_threshold:
            self.get_logger().info(f'🎯 Found significantly better frontier goal! Distance: {better_distance:.2f}m (previous: {self.current_best_distance:.2f}m)')
            self.get_logger().info(f'Improvement: {better_distance - self.current_best_distance:.2f}m')
            
            # Cancel current navigation
            self.cancel_navigation()
            
            # Update tracking variables
            self.current_best_distance = better_distance
            self.current_goal_pose = better_point
            self.goal_reached = False  # Reset goal reached flag
            
            # Navigate to the better point
            self.send_navigation_goal(better_point, better_distance)
        else:
            # Still search but be less verbose - only log significant findings
            elapsed_time = (current_time - self.hiding_start_time).nanoseconds / 1e9 if self.hiding_start_time else 0
            if int(elapsed_time) % 5 == 0:  # Every 5 seconds
                if better_point:
                    improvement = better_distance - self.current_best_distance
                    if improvement > 0.2:  # Only log if there's some improvement
                        self.get_logger().info(f'🔍 Searched for better frontiers: best found {better_distance:.2f}m (current: {self.current_best_distance:.2f}m, improvement: {improvement:.2f}m)')
                    elif int(elapsed_time) % 10 == 0:  # Less frequent updates when no significant improvement
                        self.get_logger().info(f'🔍 Continuous frontier search: current target {self.current_best_distance:.2f}m, checking for better positions...')
                else:
                    self.get_logger().info('🔍 Searched for better frontiers: no safe frontier points found')

    def send_navigation_goal(self, target_pose, target_distance):
        """Send a navigation goal to bt_navigator"""
        if not self.nav_action_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error('Navigation action server not available!')
            return
        
        # Create navigation goal
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = target_pose
        
        # Publish goal for visualization
        self.goal_pub.publish(target_pose)
        
        self.get_logger().info(f'Updated navigation goal: {target_distance:.2f}m away')
        
        # Send goal to navigation
        future = self.nav_action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.navigation_feedback_callback
        )
        future.add_done_callback(self.navigation_goal_response_callback)

    def navigation_goal_response_callback(self, future):
        """Handle navigation goal response"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Navigation goal rejected!')
            return
        
        self.navigation_goal_handle = goal_handle
        self.get_logger().info('Navigation goal accepted by bt_navigator')
        
        # Wait for result
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.navigation_result_callback)

    def navigation_feedback_callback(self, feedback_msg):
        """Handle navigation feedback"""
        feedback = feedback_msg.feedback
        if self.hiding_active and self.hiding_start_pose:
            current_distance = math.sqrt(
                (feedback.current_pose.pose.position.x - self.hiding_start_pose.position.x)**2 +
                (feedback.current_pose.pose.position.y - self.hiding_start_pose.position.y)**2
            )
            # Only log occasionally to avoid spam
            if hasattr(self, '_last_feedback_log'):
                if (self.get_clock().now() - self._last_feedback_log).nanoseconds / 1e9 < 3.0:
                    return
            self._last_feedback_log = self.get_clock().now()
            self.get_logger().info(f'Navigation feedback: distance from start = {current_distance:.2f}m')

    def navigation_result_callback(self, future):
        """Handle navigation result"""
        result = future.result().result
        if result:
            self.get_logger().info('Navigation completed successfully!')
            
            # Check if we actually reached our goal
            if self.current_pose and self.current_goal_pose:
                dx = self.current_pose.position.x - self.current_goal_pose.pose.position.x
                dy = self.current_pose.position.y - self.current_goal_pose.pose.position.y
                distance_to_goal = math.sqrt(dx*dx + dy*dy)
                
                if distance_to_goal <= self.goal_tolerance:
                    self.goal_reached = True
                    self.hiding_active = False
                    
                    # Calculate final distance from start
                    if self.hiding_start_pose:
                        dx_start = self.current_pose.position.x - self.hiding_start_pose.position.x
                        dy_start = self.current_pose.position.y - self.hiding_start_pose.position.y
                        final_distance = math.sqrt(dx_start*dx_start + dy_start*dy_start)
                        
                        self.get_logger().info(f'Successfully reached farthest hiding point!')
                        self.get_logger().info(f'Final distance from start: {final_distance:.2f}m')
                        self.get_logger().info(f'Target distance was: {self.current_best_distance:.2f}m')
                        self.get_logger().info('Hiding complete - ready for random walk behavior')
                        
                        # Publish game status for reaching farthest distance
                        game_status_msg = String()
                        game_status_msg.data = f"Robot {self.self_aruco_id} reached farthest hiding distance: {final_distance:.2f}m"
                        self.game_status_pub.publish(game_status_msg)
                        
                        # Publish game status for starting random walk
                        game_status_msg = String()
                        game_status_msg.data = f"Robot {self.self_aruco_id} started random walk after hiding"
                        self.game_status_pub.publish(game_status_msg)
                        
                        # Start random walk by sending False to random_walk topic
                        random_walk_msg = Bool()
                        random_walk_msg.data = False  # Start random walking
                        self.random_walk_pub.publish(random_walk_msg)
                        
                        self.get_logger().info('🚶 Random walk behavior activated after reaching goal!')
                    
                    # Reset tracking variables
                    self.current_best_distance = 0.0
                    self.current_goal_pose = None
                else:
                    self.get_logger().info(f'Navigation completed but not close enough to goal (distance: {distance_to_goal:.2f}m)')
                    # Continue searching for better goals since we haven't reached our target
            else:
                self.get_logger().info('Navigation completed but pose information unavailable')
        else:
            self.get_logger().warn('Navigation failed or was cancelled')
            # Don't set goal_reached to True if navigation failed
            # Continue searching for alternative goals

    def cancel_navigation(self):
        """Cancel current navigation goal"""
        return
        if self.navigation_goal_handle:
            self.get_logger().info('Cancelling current navigation goal')
            self.navigation_goal_handle.cancel_goal_async()
            self.navigation_goal_handle = None

    def check_goal_reached(self):
        """Check if the goal is reached and provide periodic status updates"""
        if not self.hiding_active or self.goal_reached:
            return
            
        if self.current_pose and self.current_goal_pose and self.hiding_start_pose:
            # Check distance to current goal
            dx = self.current_pose.position.x - self.current_goal_pose.pose.position.x
            dy = self.current_pose.position.y - self.current_goal_pose.pose.position.y
            distance_to_goal = math.sqrt(dx*dx + dy*dy)
            
            # Check distance from start position
            dx_start = self.current_pose.position.x - self.hiding_start_pose.position.x
            dy_start = self.current_pose.position.y - self.hiding_start_pose.position.y
            distance_from_start = math.sqrt(dx_start*dx_start + dy_start*dy_start)
            
            # Log progress every 3 seconds
            if hasattr(self, '_last_status_log'):
                time_since_log = (self.get_clock().now() - self._last_status_log).nanoseconds / 1e9
                if time_since_log < 3.0:
                    return
            self._last_status_log = self.get_clock().now()
            
            self.get_logger().info(f'Hiding progress: {distance_from_start:.2f}m from start, {distance_to_goal:.2f}m to goal (target: {self.current_best_distance:.2f}m)')
            
            # Check if goal is reached
            if distance_to_goal <= self.goal_tolerance:
                self.goal_reached = True
                self.hiding_active = False
                self.cancel_navigation()
                
                self.get_logger().info(f'Farthest hiding point reached! Final distance from start: {distance_from_start:.2f}m')
                
                # Publish game status for reaching farthest distance
                status_msg = String()
                status_msg.data = f"Robot {self.self_aruco_id} reached farthest hiding distance: {distance_from_start:.2f}m"
                self.game_status_pub.publish(status_msg)
                
                # Start random walk immediately after reaching goal
                status_msg = String()
                status_msg.data = f"Robot {self.self_aruco_id} started random walk after hiding"
                self.game_status_pub.publish(status_msg)
                
                # Start random walk by sending False to random_walk topic
                random_walk_msg = Bool()
                random_walk_msg.data = False  # Start random walking
                self.random_walk_pub.publish(random_walk_msg)
                
                self.get_logger().info('🚶 Random walk behavior activated after reaching goal!')
                
                # Reset tracking variables
                self.current_best_distance = 0.0
                self.current_goal_pose = None

    def obstacle_avoidance_loop(self):
        """Main obstacle avoidance control loop"""
        if not self.hiding_active:
            return
            
        if self.emergency_stop:
            self.get_logger().warn('Emergency stop active - pausing navigation')
            
            # Publish emergency stop status
            status_msg = String()
            status_msg.data = f"Robot {self.self_aruco_id} EMERGENCY STOP - obstacle too close"
            self.game_status_pub.publish(status_msg)
            
            return
        
        if self.obstacle_detected:
            self.get_logger().info('Obstacle detected - cancelling navigation')
            
            # Publish obstacle detection status
            status_msg = String()
            status_msg.data = f"Robot {self.self_aruco_id} detected obstacle - cancelling navigation"
            self.game_status_pub.publish(status_msg)
            
            self.cancel_navigation()
            return

    def get_minimum_front_distance(self):
        """Get the minimum distance to obstacles in front of the robot"""
        if not self.latest_scan or not self.latest_scan.ranges:
            return float('inf')
            
        num_readings = len(self.latest_scan.ranges)
        # Check front 60 degrees (30 degrees on each side of center)
        front_start = max(0, num_readings // 2 - num_readings // 6)
        front_end = min(num_readings, num_readings // 2 + num_readings // 6)
        
        min_distance = float('inf')
        for i in range(front_start, front_end):
            range_value = self.latest_scan.ranges[i]
            if not math.isinf(range_value) and not math.isnan(range_value):
                min_distance = min(min_distance, range_value)
                
        return min_distance



    def start_180_turn(self):
        """Initiate a 180-degree turn before starting navigation"""
        self.performing_turn = True
        self.turn_start_time = self.get_clock().now()
        
        # Get current heading from pose orientation
        if self.current_pose and self.current_pose.orientation:
            # Convert quaternion to yaw angle manually
            x = self.current_pose.orientation.x
            y = self.current_pose.orientation.y
            z = self.current_pose.orientation.z
            w = self.current_pose.orientation.w
            
            # Yaw (z-axis rotation) from quaternion
            yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
            self.turn_start_heading = yaw
        else:
            self.turn_start_heading = 0.0  # Default heading if pose not available
        
        self.get_logger().info('🔄 Starting 180° turn')

    def turn_control_loop(self):
        """Control loop for 180-degree turn"""
        if not self.performing_turn or not self.hiding_active:
            return
        
        if self.emergency_stop:
            self.get_logger().warn('Emergency stop active - pausing turn')
            return
        
        current_time = self.get_clock().now()
        elapsed_time = (current_time - self.turn_start_time).nanoseconds / 1e9
        
        if elapsed_time >= self.target_turn_duration:
            # Turn completed
            self.complete_180_turn()
            return
        
        # Continue turning
        turn_cmd = Twist()
        turn_cmd.angular.z = self.turn_angular_speed  # Positive = counter-clockwise
        
        # Publish turn command
        self.cmd_vel_pub.publish(turn_cmd)
        
    def complete_180_turn(self):
        """Complete the 180-degree turn and start navigation"""
        self.performing_turn = False
        
        # Stop turning
        stop_cmd = Twist()
        self.cmd_vel_pub.publish(stop_cmd)
        
        self.get_logger().info('✅ 180° turn completed! Now finding farthest hiding spot...')
        
        # Publish game status for turn completion
        status_msg = String()
        status_msg.data = f"Robot {self.self_aruco_id} completed 180° turn - starting navigation to farthest point"
        self.game_status_pub.publish(status_msg)
        
        # Start navigation immediately after turn completion
        if self.hiding_active:  # Still hiding
            self.navigate_to_farthest_point()

    def check_hiding_timeout(self):
        """Check if hiding timeout has been reached and switch to random walk"""
        if not self.hiding_active or not self.hiding_start_time:
            return
        
        current_time = self.get_clock().now()
        elapsed_time = (current_time - self.hiding_start_time).nanoseconds / 1e9
        
        if elapsed_time >= self.hiding_timeout:
            # Calculate final distance reached when timing out
            final_distance_reached = 0.0
            if self.current_pose and self.hiding_start_pose:
                dx = self.current_pose.position.x - self.hiding_start_pose.position.x
                dy = self.current_pose.position.y - self.hiding_start_pose.position.y
                final_distance_reached = math.sqrt(dx*dx + dy*dy)
            
            self.get_logger().info(f'⏰ Hiding timeout reached ({elapsed_time:.1f}s)! Switching to random walk...')
            self.get_logger().info(f'Final distance reached: {final_distance_reached:.2f}m')
            
            # Stop hiding behavior
            self.hiding_active = False
            self.cancel_navigation()
            
            # Stop 180-degree turn if in progress
            self.performing_turn = False
            self.turn_start_time = None
            self.turn_start_heading = None
            
            # Stop robot movement
            stop_cmd = Twist()
            self.cmd_vel_pub.publish(stop_cmd)
            
            # Publish game status with final distance
            status_msg = String()
            status_msg.data = f"Robot {self.self_aruco_id} hiding timeout - reached {final_distance_reached:.2f}m, starting random walk"
            self.game_status_pub.publish(status_msg)
            
            # Start random walk by sending False to random_walk topic
            # (False means start walking, True means stop walking in the random_walk node)
            random_walk_msg = Bool()
            random_walk_msg.data = False  # Start random walking
            self.random_walk_pub.publish(random_walk_msg)
            
            self.get_logger().info('🚶 Random walk behavior activated!')


def main():
    rclpy.init()
    node = HiderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main() 

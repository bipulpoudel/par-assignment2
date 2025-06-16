#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
import tf2_ros
import math

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from std_msgs.msg import Empty
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
        
        # Get parameters
        self.goal_tolerance = self.get_parameter('goal_tolerance').get_parameter_value().double_value
        self.min_obstacle_distance = self.get_parameter('min_obstacle_distance').get_parameter_value().double_value
        self.search_resolution = self.get_parameter('search_resolution').get_parameter_value().double_value
        self.min_improvement = self.get_parameter('min_improvement').get_parameter_value().double_value
        self.search_interval = self.get_parameter('search_interval').get_parameter_value().double_value
        
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
        
        # Navigation action client
        self.nav_action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        
        # TF setup
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Publishers
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        
        # Subscribers
        self.odom_sub = self.create_subscription(Odometry, '/odometry/filtered', self.odom_callback, 10)
        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)
        self.start_trigger_sub = self.create_subscription(Empty, '/start_hiding', self.start_hiding_callback, 10)
        self.stop_trigger_sub = self.create_subscription(Empty, '/stop_hiding', self.stop_hiding_callback, 10)
        
        # Timer for searching better goals during navigation (more frequent searches)
        self.search_timer = self.create_timer(self.search_interval, self.search_for_better_goal)
        
        # Timer for checking if goal is reached
        self.goal_check_timer = self.create_timer(0.5, self.check_goal_reached)
        
        self.get_logger().info('Hider Node initialized with bt_navigator and enhanced dynamic goal updating')
        self.get_logger().info(f'Goal tolerance: {self.goal_tolerance}m')
        self.get_logger().info(f'Minimum improvement for goal switch: {self.min_improvement}m')
        self.get_logger().info(f'Search interval: {self.search_interval}s')
        self.get_logger().info('No timeout logic - will reach farthest point then signal for random walk')
        self.get_logger().info('Waiting for map data and /start_hiding trigger...')

    def odom_callback(self, msg):
        """Update current pose from odometry"""
        self.current_pose = msg.pose.pose

    def map_callback(self, msg):
        """Store map data for finding farthest points"""
        self.map_data = msg
        self.get_logger().info(f'Map received: {msg.info.width}x{msg.info.height}, resolution: {msg.info.resolution}m/cell')

    def start_hiding_callback(self, msg):
        """Start the hiding behavior using bt_navigator"""
        if not self.hiding_active and self.current_pose and self.map_data:
            self.hiding_active = True
            self.hiding_start_time = self.get_clock().now()
            self.hiding_start_pose = self.current_pose
            self.current_best_distance = 0.0
            self.current_goal_pose = None
            self.last_search_time = self.get_clock().now()
            
            self.get_logger().info(f'Starting hiding behavior with bt_navigator!')
            self.get_logger().info(f'Start position: ({self.hiding_start_pose.position.x:.2f}, {self.hiding_start_pose.position.y:.2f})')
            
            # Find and navigate to the farthest accessible point
            self.navigate_to_farthest_point()
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
            self.get_logger().info('Stopping hiding behavior')

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

    def find_farthest_accessible_point(self, verbose=True):
        """Find the farthest accessible point in the known map"""
        if not self.map_data or not self.hiding_start_pose:
            return None, 0.0
        
        start_x = self.hiding_start_pose.position.x
        start_y = self.hiding_start_pose.position.y
        
        if verbose:
            self.get_logger().info('Searching for farthest accessible point in map...')
        
        best_point = None
        max_distance = 0.0
        points_checked = 0
        safe_points_found = 0
        
        # Search through the map with specified resolution
        width = self.map_data.info.width
        height = self.map_data.info.height
        resolution = self.map_data.info.resolution
        
        # Calculate step size based on search resolution
        step = max(1, int(self.search_resolution / resolution))
        
        for map_y in range(0, height, step):
            for map_x in range(0, width, step):
                points_checked += 1
                
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
        
        if verbose:
            self.get_logger().info(f'Map search complete: {points_checked} points checked, {safe_points_found} safe points found')
            
            if best_point:
                self.get_logger().info(f'Farthest point found: ({best_point.pose.position.x:.2f}, {best_point.pose.position.y:.2f})')
                self.get_logger().info(f'Distance from start: {max_distance:.2f}m')
            else:
                self.get_logger().warn('No safe accessible point found in map!')
        
        return best_point, max_distance

    def navigate_to_farthest_point(self):
        """Send navigation goal to the farthest accessible point"""
        if not self.nav_action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Navigation action server not available!')
            return
        
        target_pose, target_distance = self.find_farthest_accessible_point()
        if not target_pose:
            self.get_logger().error('Could not find a safe target point!')
            return
        
        # Update our tracking variables
        self.current_best_distance = target_distance
        self.current_goal_pose = target_pose
        
        # Create navigation goal
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = target_pose
        
        # Publish goal for visualization
        self.goal_pub.publish(target_pose)
        
        self.get_logger().info(f'Sending navigation goal to bt_navigator: ({target_pose.pose.position.x:.2f}, {target_pose.pose.position.y:.2f})')
        self.get_logger().info(f'Target distance from start: {target_distance:.2f}m')
        
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
        
        # Don't search too frequently but more often than before
        current_time = self.get_clock().now()
        if self.last_search_time:
            time_since_last_search = (current_time - self.last_search_time).nanoseconds / 1e9
            if time_since_last_search < self.search_interval:
                return
        
        self.last_search_time = current_time
        
        # Find the current best point
        better_point, better_distance = self.find_farthest_accessible_point(verbose=False)
        
        # More aggressive goal switching - use smaller improvement threshold
        improvement_threshold = max(0.5, self.min_improvement * 0.5)  # At least 0.5m improvement
        
        if better_point and better_distance > self.current_best_distance + improvement_threshold:
            self.get_logger().info(f'Found significantly better hiding goal! Distance: {better_distance:.2f}m (previous: {self.current_best_distance:.2f}m)')
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
                        self.get_logger().info(f'Searched for better goals: best found {better_distance:.2f}m (current: {self.current_best_distance:.2f}m, improvement: {improvement:.2f}m)')
                    elif int(elapsed_time) % 10 == 0:  # Less frequent updates when no significant improvement
                        self.get_logger().info(f'Continuous search: current target {self.current_best_distance:.2f}m, checking for better positions...')
                else:
                    self.get_logger().info('Searched for better goals: no safe points found')

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
        
        self.get_logger().info(f'Sending new navigation goal: ({target_pose.pose.position.x:.2f}, {target_pose.pose.position.y:.2f})')
        self.get_logger().info(f'Target distance from start: {target_distance:.2f}m')
        
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
                self.get_logger().info('Hiding complete - ready for random walk behavior')
                
                # Reset tracking variables
                self.current_best_distance = 0.0
                self.current_goal_pose = None


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
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import Bool
from nav2_msgs.action import NavigateToPose
import numpy as np
import math
from collections import deque
import tf2_ros
from rclpy.duration import Duration


class FrontierExplorer(Node):
    def __init__(self):
        super().__init__('frontier_explorer')
        
        # Parameters
        self.declare_parameter('robot_radius', 0.3)
        self.declare_parameter('transform_tolerance', 0.3)
        self.declare_parameter('min_frontier_size', 10)
        self.declare_parameter('potential_scale', 3.0)
        self.declare_parameter('gain_scale', 1.0)
        self.declare_parameter('goal_tolerance', 0.5)
        
        # Get parameters
        self.robot_radius = self.get_parameter('robot_radius').value
        self.transform_tolerance = self.get_parameter('transform_tolerance').value
        self.min_frontier_size = self.get_parameter('min_frontier_size').value
        self.potential_scale = self.get_parameter('potential_scale').value
        self.gain_scale = self.get_parameter('gain_scale').value
        self.goal_tolerance = self.get_parameter('goal_tolerance').value
        
        # TF buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        
        # Subscribers
        self.map_sub = self.create_subscription(
            OccupancyGrid, 'map', self.map_callback, 10)
        self.robot_detected_sub = self.create_subscription(
            Bool, 'robot_detected', self.robot_detected_callback, 10)
        
        # Action client for navigation
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        
        # State variables
        self.map_data = None
        self.current_goal = None
        self.exploring = False
        self.frontiers = []
        self.robot_detected = False
        self.exploration_complete = False
        self.target_position = None
        
        # Timer for exploration
        self.exploration_timer = self.create_timer(2.0, self.exploration_cycle)
        
        self.get_logger().info('Frontier Explorer Node Started - ROSbot 3 PRO Exploration Mode')

    def map_callback(self, msg):
        """Process incoming occupancy grid map"""
        self.map_data = msg
        
    def robot_detected_callback(self, msg):
        """Handle robot detection message"""
        self.robot_detected = msg.data
        if self.robot_detected:
            self.get_logger().info('Robot detected! Stopping all movement.')
            self.exploring = False
            self.exploration_complete = True
            self.cancel_current_navigation()
            self.stop_robot()
        
    def exploration_cycle(self):
        """Main exploration cycle"""
        if not self.map_data or self.robot_detected or self.exploration_complete:
            return
        
        # Check for early completion
        if self.current_goal and self.is_within_goal_tolerance():
            self.handle_early_arrival()
        
        if not self.exploring and not self.current_goal:
            self.start_exploration()
        elif self.exploring and not self.current_goal:
            self.find_and_goto_frontier()
    
    def start_exploration(self):
        """Initialize exploration"""
        if not self.check_system_status():
            self.get_logger().error('❌ System not ready for exploration - waiting...')
            return
            
        self.exploring = True
        self.get_logger().info('🚀 Starting frontier-based exploration')
        self.find_and_goto_frontier()
    
    def find_and_goto_frontier(self):
        """Find frontiers and navigate to the best one"""
        if not self.map_data:
            return
            
        frontiers = self.find_frontiers()
        
        if not frontiers:
            self.get_logger().info('🎉 EXPLORATION COMPLETE! 🎉')
            self.get_logger().info('No more frontiers found - map fully explored')
            self.get_logger().info('🛑 Stopping robot...')
            self.exploring = False
            self.exploration_complete = True
            self.stop_robot()
            return
        
        robot_pose = self.get_robot_pose()
        if not robot_pose:
            return
        
        best_frontier = self.evaluate_frontiers(frontiers, robot_pose)
        
        if best_frontier:
            self.navigate_to_frontier(best_frontier)
    
    def find_frontiers(self):
        """Find frontier cells using wavefront frontier detection"""
        if not self.map_data:
            return []
        
        width = self.map_data.info.width
        height = self.map_data.info.height
        data = np.array(self.map_data.data).reshape((height, width))
        
        frontiers = []
        visited = np.zeros((height, width), dtype=bool)
        
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                if visited[y, x] or data[y, x] != -1:
                    continue
                    
                is_frontier = False
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < height and 0 <= nx < width:
                            if data[ny, nx] == 0:
                                is_frontier = True
                                break
                    if is_frontier:
                        break
                
                if is_frontier:
                    frontier_cells = self.bfs_frontier_cells(data, x, y, visited)
                    if len(frontier_cells) >= self.min_frontier_size:
                        frontiers.append(frontier_cells)
        
        return frontiers
    
    def bfs_frontier_cells(self, data, start_x, start_y, visited):
        """Use BFS to find connected frontier cells"""
        height, width = data.shape
        frontier_cells = []
        queue = deque([(start_x, start_y)])
        visited[start_y, start_x] = True
        
        while queue:
            x, y = queue.popleft()
            frontier_cells.append((x, y))
            
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    
                    if (0 <= nx < width and 0 <= ny < height and 
                        not visited[ny, nx] and data[ny, nx] == -1):
                        
                        is_frontier = False
                        for ddy in [-1, 0, 1]:
                            for ddx in [-1, 0, 1]:
                                if ddx == 0 and ddy == 0:
                                    continue
                                nny, nnx = ny + ddy, nx + ddx
                                if (0 <= nnx < width and 0 <= nny < height and 
                                    data[nny, nnx] == 0):
                                    is_frontier = True
                                    break
                            if is_frontier:
                                break
                        
                        if is_frontier:
                            visited[ny, nx] = True
                            queue.append((nx, ny))
        
        return frontier_cells
    
    def evaluate_frontiers(self, frontiers, robot_pose):
        """Evaluate frontiers and return the best one"""
        if not frontiers:
            return None
            
        best_frontier = None
        best_score = -float('inf')
        
        robot_x = robot_pose.pose.position.x
        robot_y = robot_pose.pose.position.y
        
        for frontier in frontiers:
            centroid_x = sum(cell[0] for cell in frontier) / len(frontier)
            centroid_y = sum(cell[1] for cell in frontier) / len(frontier)
            
            world_x = (centroid_x * self.map_data.info.resolution + 
                      self.map_data.info.origin.position.x)
            world_y = (centroid_y * self.map_data.info.resolution + 
                      self.map_data.info.origin.position.y)
            
            distance = math.sqrt((world_x - robot_x)**2 + (world_y - robot_y)**2)
            size_gain = len(frontier)
            
            if distance > 0:
                score = (self.gain_scale * size_gain) / (self.potential_scale * distance)
            else:
                score = self.gain_scale * size_gain
            
            if score > best_score:
                best_score = score
                best_frontier = {
                    'cells': frontier,
                    'centroid': (world_x, world_y),
                    'size': len(frontier),
                    'score': score
                }
        
        return best_frontier
    
    def navigate_to_frontier(self, frontier):
        """Navigate to the selected frontier"""
        target_x, target_y = frontier['centroid'][0], frontier['centroid'][1]
        
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = target_x
        goal_msg.pose.pose.position.y = target_y
        goal_msg.pose.pose.position.z = 0.0
        goal_msg.pose.pose.orientation.w = 1.0
        
        self.get_logger().info(f'🎯 Navigating to frontier at ({target_x:.2f}, {target_y:.2f})')
        
        if not self.nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('❌ Navigation server not available!')
            return
        
        try:
            future = self.nav_client.send_goal_async(goal_msg)
            future.add_done_callback(self.goal_response_callback)
            self.current_goal = {'type': 'frontier', 'data': frontier}
            self.target_position = (target_x, target_y)
        except Exception as e:
            self.get_logger().error(f'❌ Failed to send navigation goal: {e}')
    
    def goal_response_callback(self, future):
        """Handle navigation goal response"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('🚫 Navigation goal rejected!')
            self.current_goal = None
            self.target_position = None
            return
        
        self.get_logger().info('✅ Navigation goal accepted')
        self.current_goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.goal_result_callback)
    
    def goal_result_callback(self, future):
        """Handle navigation result"""
        self.get_logger().info('Navigation completed')
        self.current_goal = None
        self.target_position = None
    
    def cancel_current_navigation(self):
        """Cancel current navigation goal"""
        try:
            if hasattr(self, 'current_goal_handle') and self.current_goal_handle:
                cancel_future = self.current_goal_handle.cancel_goal_async()
                self.get_logger().info('Navigation goal canceled')
        except Exception as e:
            self.get_logger().warn(f'Failed to cancel navigation: {e}')
    
    def is_within_goal_tolerance(self):
        """Check if robot is within goal tolerance distance"""
        if not self.target_position:
            return False
            
        robot_pose = self.get_robot_pose()
        if not robot_pose:
            return False
            
        robot_x = robot_pose.pose.position.x
        robot_y = robot_pose.pose.position.y
        target_x, target_y = self.target_position
        
        distance = math.sqrt((target_x - robot_x)**2 + (target_y - robot_y)**2)
        return distance <= self.goal_tolerance
    
    def handle_early_arrival(self):
        """Handle early arrival when robot is within tolerance"""
        self.get_logger().info('Early arrival - robot within tolerance distance')
        self.current_goal = None
        self.target_position = None
    
    def get_robot_pose(self):
        """Get current robot pose in map frame"""
        try:
            transform = self.tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time(),
                timeout=Duration(seconds=self.transform_tolerance))
            
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = transform.transform.translation.x
            pose.pose.position.y = transform.transform.translation.y
            pose.pose.position.z = transform.transform.translation.z
            pose.pose.orientation = transform.transform.rotation
            
            return pose
            
        except Exception as e:
            self.get_logger().warn(f'Could not get robot pose: {e}')
            return None
    
    def stop_robot(self):
        """Stop the robot by publishing zero velocities"""
        stop_msg = Twist()
        self.cmd_vel_pub.publish(stop_msg)
        self.get_logger().info('🛑 Robot stopped - Exploration complete!')

    def check_system_status(self):
        """Check system status for navigation readiness"""
        issues = []
        
        if not self.map_data:
            issues.append("❌ No map data received")
        else:
            self.get_logger().info("✅ Map data available")
        
        robot_pose = self.get_robot_pose()
        if not robot_pose:
            issues.append("❌ Cannot get robot pose (localization issue)")
        else:
            self.get_logger().info("✅ Robot pose available")
        
        if not self.nav_client.wait_for_server(timeout_sec=1.0):
            issues.append("❌ Navigation server not available")
        else:
            self.get_logger().info("✅ Navigation server available")
        
        if issues:
            self.get_logger().error("🔍 System Status Issues Found:")
            for issue in issues:
                self.get_logger().error(f"  {issue}")
        else:
            self.get_logger().info("🎉 All systems ready for exploration!")
        
        return len(issues) == 0


def main(args=None):
    rclpy.init(args=args)
    
    frontier_explorer = FrontierExplorer()
    
    try:
        rclpy.spin(frontier_explorer)
    except KeyboardInterrupt:
        pass
    
    frontier_explorer.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main() 
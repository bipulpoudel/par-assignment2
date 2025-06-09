#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import Bool
import numpy as np
import math

class LawnMowerPlanner(Node):
    def __init__(self):
        super().__init__('lawn_mower_planner')
        
        # Parameters
        self.declare_parameter('path_spacing', 1.0)  # Distance between parallel paths
        self.declare_parameter('path_overlap', 0.2)  # Overlap between paths
        self.declare_parameter('boundary_buffer', 0.5)  # Buffer from obstacles
        self.declare_parameter('min_segment_length', 1.0)  # Minimum segment length
        self.declare_parameter('auto_generate', True)  # Auto-generate on map update
        
        # Get parameters
        self.path_spacing = self.get_parameter('path_spacing').value
        self.path_overlap = self.get_parameter('path_overlap').value
        self.boundary_buffer = self.get_parameter('boundary_buffer').value
        self.min_segment_length = self.get_parameter('min_segment_length').value
        self.auto_generate = self.get_parameter('auto_generate').value
        
        # Publishers
        self.path_pub = self.create_publisher(Path, 'lawn_mower_path', 10)
        self.boundary_pub = self.create_publisher(MarkerArray, 'boundary_markers', 10)
        self.coverage_pub = self.create_publisher(MarkerArray, 'coverage_markers', 10)
        
        # Subscribers
        self.map_sub = self.create_subscription(
            OccupancyGrid, 'map', self.map_callback, 10)
        self.trigger_sub = self.create_subscription(
            Bool, 'generate_lawn_mower_path', self.trigger_callback, 10)
        
        # State variables
        self.map_data = None
        self.boundary_points = []
        self.coverage_path = []
        self.path_generated = False
        
        # Timer for auto-generation
        if self.auto_generate:
            self.generation_timer = self.create_timer(5.0, self.auto_generation_cycle)
        
        self.get_logger().info('🌿 Lawn Mower Path Planner Node Started')
        self.get_logger().info(f'Path spacing: {self.path_spacing}m')
        self.get_logger().info(f'Boundary buffer: {self.boundary_buffer}m')

    def map_callback(self, msg):
        """Process incoming occupancy grid map"""
        self.map_data = msg
        if self.auto_generate and not self.path_generated:
            self.get_logger().info('📍 Map received, checking for path generation...')
    
    def trigger_callback(self, msg):
        """Manual trigger for path generation"""
        if msg.data:
            self.get_logger().info('🎯 Manual path generation triggered')
            self.generate_lawn_mower_path()
    
    def auto_generation_cycle(self):
        """Auto-generate path when conditions are met"""
        if not self.map_data or self.path_generated:
            return
        
        # Check if map has sufficient free space
        if self.has_sufficient_free_space():
            self.get_logger().info('🚀 Auto-generating lawn mower path...')
            self.generate_lawn_mower_path()

    def has_sufficient_free_space(self):
        """Check if map has enough free space for path generation"""
        if not self.map_data:
            return False
        
        width = self.map_data.info.width
        height = self.map_data.info.height
        data = np.array(self.map_data.data).reshape((height, width))
        
        free_cells = np.sum(data == 0)  # Count free space cells
        total_cells = width * height
        free_ratio = free_cells / total_cells
        
        # Require at least 10% free space
        return free_ratio > 0.1

    def generate_lawn_mower_path(self):
        """Main function to generate the lawn mower coverage path"""
        if not self.map_data:
            self.get_logger().error('❌ No map data available')
            return
        
        self.get_logger().info('🌿 Generating lawn mower coverage path...')
        
        try:
            # Step 1: Extract free space and create buffered boundary
            free_space_grid = self.extract_free_space_grid()
            
            # Step 2: Find boundary of traversable area
            boundary_points = self.find_boundary_points(free_space_grid)
            
            if len(boundary_points) < 4:
                self.get_logger().error('❌ Insufficient boundary points for path generation')
                return
            
            # Step 3: Create bounding box and determine coverage area
            bbox = self.compute_bounding_box(boundary_points)
            
            # Step 4: Generate zigzag coverage lines
            coverage_lines = self.generate_zigzag_lines(bbox, free_space_grid)
            
            # Step 5: Create boustrophedon path
            lawn_mower_path = self.create_boustrophedon_path(coverage_lines)
            
            # Step 6: Publish results
            self.publish_path(lawn_mower_path)
            self.visualize_coverage_pattern(boundary_points, coverage_lines, bbox)
            
            self.boundary_points = boundary_points
            self.coverage_path = lawn_mower_path
            self.path_generated = True
            
            self.get_logger().info(f'✅ Lawn mower path generated with {len(lawn_mower_path)} waypoints')
            
        except Exception as e:
            self.get_logger().error(f'❌ Error generating lawn mower path: {e}')

    def extract_free_space_grid(self):
        """Extract and process free space from occupancy grid"""
        width = self.map_data.info.width
        height = self.map_data.info.height
        resolution = self.map_data.info.resolution
        
        data = np.array(self.map_data.data).reshape((height, width))
        
        # Create buffered free space grid
        free_space = (data == 0)  # Free space
        
        # Apply morphological erosion to create buffer from obstacles
        buffer_cells = max(1, int(self.boundary_buffer / resolution))
        free_space_buffered = self.erode_binary_image(free_space, buffer_cells)
        
        return free_space_buffered

    def erode_binary_image(self, binary_image, erosion_size):
        """Simple morphological erosion without OpenCV"""
        height, width = binary_image.shape
        eroded = np.zeros_like(binary_image)
        
        for y in range(erosion_size, height - erosion_size):
            for x in range(erosion_size, width - erosion_size):
                if binary_image[y, x]:
                    # Check if all pixels in erosion kernel are free
                    all_free = True
                    for dy in range(-erosion_size, erosion_size + 1):
                        for dx in range(-erosion_size, erosion_size + 1):
                            if not binary_image[y + dy, x + dx]:
                                all_free = False
                                break
                        if not all_free:
                            break
                    eroded[y, x] = all_free
        
        return eroded

    def find_boundary_points(self, free_space_grid):
        """Find boundary points of the free space area"""
        height, width = free_space_grid.shape
        resolution = self.map_data.info.resolution
        origin_x = self.map_data.info.origin.position.x
        origin_y = self.map_data.info.origin.position.y
        
        boundary_points = []
        
        # Find boundary pixels (free pixels adjacent to non-free pixels)
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                if free_space_grid[y, x]:  # Current pixel is free
                    # Check if adjacent to non-free pixel
                    is_boundary = False
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dy == 0 and dx == 0:
                                continue
                            ny, nx = y + dy, x + dx
                            if not free_space_grid[ny, nx]:
                                is_boundary = True
                                break
                        if is_boundary:
                            break
                    
                    if is_boundary:
                        # Convert to world coordinates
                        world_x = x * resolution + origin_x
                        world_y = y * resolution + origin_y
                        boundary_points.append([world_x, world_y])
        
        return boundary_points

    def compute_bounding_box(self, points):
        """Compute axis-aligned bounding box"""
        if not points:
            return None
        
        points_array = np.array(points)
        min_x, min_y = np.min(points_array, axis=0)
        max_x, max_y = np.max(points_array, axis=0)
        
        return {
            'min_x': min_x,
            'max_x': max_x,
            'min_y': min_y,
            'max_y': max_y,
            'width': max_x - min_x,
            'height': max_y - min_y,
            'center_x': (min_x + max_x) / 2,
            'center_y': (min_y + max_y) / 2
        }

    def generate_zigzag_lines(self, bbox, free_space_grid):
        """Generate zigzag coverage lines within the boundary"""
        if not bbox:
            return []
        
        coverage_lines = []
        
        # Determine sweep direction based on aspect ratio
        if bbox['width'] >= bbox['height']:
            # Horizontal sweep (left-right motion, vertical progression)
            num_lines = int(bbox['height'] / self.path_spacing) + 1
            
            for i in range(num_lines):
                y_coord = bbox['min_y'] + i * self.path_spacing
                if y_coord > bbox['max_y']:
                    break
                
                # Find line segments at this y coordinate
                segments = self.find_horizontal_line_segments(y_coord, bbox, free_space_grid)
                
                if segments:
                    coverage_lines.append({
                        'y': y_coord,
                        'segments': segments,
                        'direction': 'horizontal',
                        'line_index': i
                    })
        else:
            # Vertical sweep (up-down motion, horizontal progression)
            num_lines = int(bbox['width'] / self.path_spacing) + 1
            
            for i in range(num_lines):
                x_coord = bbox['min_x'] + i * self.path_spacing
                if x_coord > bbox['max_x']:
                    break
                
                # Find line segments at this x coordinate
                segments = self.find_vertical_line_segments(x_coord, bbox, free_space_grid)
                
                if segments:
                    coverage_lines.append({
                        'x': x_coord,
                        'segments': segments,
                        'direction': 'vertical',
                        'line_index': i
                    })
        
        return coverage_lines

    def find_horizontal_line_segments(self, y_coord, bbox, free_space_grid):
        """Find horizontal line segments at given y coordinate"""
        resolution = self.map_data.info.resolution
        origin_x = self.map_data.info.origin.position.x
        origin_y = self.map_data.info.origin.position.y
        
        # Convert y coordinate to grid space
        grid_y = int((y_coord - origin_y) / resolution)
        height, width = free_space_grid.shape
        
        if grid_y < 0 or grid_y >= height:
            return []
        
        # Find continuous free space segments along this row
        segments = []
        segment_start = None
        
        x_start = int((bbox['min_x'] - origin_x) / resolution)
        x_end = int((bbox['max_x'] - origin_x) / resolution)
        
        x_start = max(0, x_start)
        x_end = min(width - 1, x_end)
        
        for grid_x in range(x_start, x_end + 1):
            if free_space_grid[grid_y, grid_x]:
                if segment_start is None:
                    segment_start = grid_x
            else:
                if segment_start is not None:
                    # End of segment
                    start_world_x = segment_start * resolution + origin_x
                    end_world_x = (grid_x - 1) * resolution + origin_x
                    
                    # Check minimum segment length
                    if end_world_x - start_world_x >= self.min_segment_length:
                        segments.append({
                            'start': [start_world_x, y_coord],
                            'end': [end_world_x, y_coord]
                        })
                    segment_start = None
        
        # Handle segment that extends to the end
        if segment_start is not None:
            start_world_x = segment_start * resolution + origin_x
            end_world_x = x_end * resolution + origin_x
            
            if end_world_x - start_world_x >= self.min_segment_length:
                segments.append({
                    'start': [start_world_x, y_coord],
                    'end': [end_world_x, y_coord]
                })
        
        return segments

    def find_vertical_line_segments(self, x_coord, bbox, free_space_grid):
        """Find vertical line segments at given x coordinate"""
        resolution = self.map_data.info.resolution
        origin_x = self.map_data.info.origin.position.x
        origin_y = self.map_data.info.origin.position.y
        
        # Convert x coordinate to grid space
        grid_x = int((x_coord - origin_x) / resolution)
        height, width = free_space_grid.shape
        
        if grid_x < 0 or grid_x >= width:
            return []
        
        # Find continuous free space segments along this column
        segments = []
        segment_start = None
        
        y_start = int((bbox['min_y'] - origin_y) / resolution)
        y_end = int((bbox['max_y'] - origin_y) / resolution)
        
        y_start = max(0, y_start)
        y_end = min(height - 1, y_end)
        
        for grid_y in range(y_start, y_end + 1):
            if free_space_grid[grid_y, grid_x]:
                if segment_start is None:
                    segment_start = grid_y
            else:
                if segment_start is not None:
                    # End of segment
                    start_world_y = segment_start * resolution + origin_y
                    end_world_y = (grid_y - 1) * resolution + origin_y
                    
                    # Check minimum segment length
                    if end_world_y - start_world_y >= self.min_segment_length:
                        segments.append({
                            'start': [x_coord, start_world_y],
                            'end': [x_coord, end_world_y]
                        })
                    segment_start = None
        
        # Handle segment that extends to the end
        if segment_start is not None:
            start_world_y = segment_start * resolution + origin_y
            end_world_y = y_end * resolution + origin_y
            
            if end_world_y - start_world_y >= self.min_segment_length:
                segments.append({
                    'start': [x_coord, start_world_y],
                    'end': [x_coord, end_world_y]
                })
        
        return segments

    def create_boustrophedon_path(self, coverage_lines):
        """Create boustrophedon (alternating direction) coverage path"""
        path_points = []
        
        if not coverage_lines:
            return path_points
        
        reverse_direction = False
        
        for line in coverage_lines:
            line_points = []
            
            # Collect all points from segments in this line
            for segment in line['segments']:
                if len(line_points) == 0:
                    line_points.append(segment['start'])
                line_points.append(segment['end'])
            
            # Reverse direction for boustrophedon pattern
            if reverse_direction:
                line_points.reverse()
            
            # Add connecting path to previous line if needed
            if path_points and line_points:
                # Add connection from last point to first point of new line
                last_point = path_points[-1]
                first_point = line_points[0]
                
                # Add intermediate connection point if needed
                distance = math.sqrt((first_point[0] - last_point[0])**2 + 
                                   (first_point[1] - last_point[1])**2)
                if distance > self.path_spacing * 2:
                    # Add intermediate waypoint
                    mid_point = [
                        (last_point[0] + first_point[0]) / 2,
                        (last_point[1] + first_point[1]) / 2
                    ]
                    path_points.append(mid_point)
            
            # Add line points to main path
            path_points.extend(line_points)
            reverse_direction = not reverse_direction
        
        return path_points

    def publish_path(self, path_points):
        """Publish the lawn mower path for visualization"""
        if not path_points:
            return
        
        path_msg = Path()
        path_msg.header.frame_id = 'map'
        path_msg.header.stamp = self.get_clock().now().to_msg()
        
        for point in path_points:
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = float(point[0])
            pose.pose.position.y = float(point[1])
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
        
        self.path_pub.publish(path_msg)
        self.get_logger().info(f'📍 Published lawn mower path with {len(path_points)} waypoints')

    def visualize_coverage_pattern(self, boundary_points, coverage_lines, bbox):
        """Create visualization markers for the coverage pattern"""
        # Visualize boundary
        self.visualize_boundary(boundary_points, bbox)
        
        # Visualize coverage lines
        self.visualize_coverage_lines(coverage_lines, bbox)

    def visualize_boundary(self, boundary_points, bbox):
        """Visualize boundary and bounding box"""
        marker_array = MarkerArray()
        
        # Clear previous markers
        clear_marker = Marker()
        clear_marker.header.frame_id = 'map'
        clear_marker.header.stamp = self.get_clock().now().to_msg()
        clear_marker.ns = 'boundary'
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)
        
        # Boundary points
        if boundary_points:
            boundary_marker = Marker()
            boundary_marker.header.frame_id = 'map'
            boundary_marker.header.stamp = self.get_clock().now().to_msg()
            boundary_marker.ns = 'boundary'
            boundary_marker.id = 0
            boundary_marker.type = Marker.POINTS
            boundary_marker.action = Marker.ADD
            boundary_marker.scale.x = 0.1
            boundary_marker.scale.y = 0.1
            boundary_marker.color.r = 1.0
            boundary_marker.color.g = 0.0
            boundary_marker.color.b = 0.0
            boundary_marker.color.a = 0.8
            
            for point in boundary_points[::5]:  # Subsample for visualization
                p = Point()
                p.x = float(point[0])
                p.y = float(point[1])
                p.z = 0.0
                boundary_marker.points.append(p)
            
            marker_array.markers.append(boundary_marker)
        
        # Bounding box
        if bbox:
            bbox_marker = Marker()
            bbox_marker.header.frame_id = 'map'
            bbox_marker.header.stamp = self.get_clock().now().to_msg()
            bbox_marker.ns = 'boundary'
            bbox_marker.id = 1
            bbox_marker.type = Marker.LINE_STRIP
            bbox_marker.action = Marker.ADD
            bbox_marker.scale.x = 0.05
            bbox_marker.color.r = 0.0
            bbox_marker.color.g = 0.0
            bbox_marker.color.b = 1.0
            bbox_marker.color.a = 1.0
            
            # Bounding box corners
            corners = [
                [bbox['min_x'], bbox['min_y']],
                [bbox['max_x'], bbox['min_y']],
                [bbox['max_x'], bbox['max_y']],
                [bbox['min_x'], bbox['max_y']],
                [bbox['min_x'], bbox['min_y']]  # Close the box
            ]
            
            for corner in corners:
                p = Point()
                p.x = float(corner[0])
                p.y = float(corner[1])
                p.z = 0.0
                bbox_marker.points.append(p)
            
            marker_array.markers.append(bbox_marker)
        
        self.boundary_pub.publish(marker_array)

    def visualize_coverage_lines(self, coverage_lines, bbox):
        """Visualize coverage lines and pattern"""
        marker_array = MarkerArray()
        marker_id = 0
        
        # Clear previous markers
        clear_marker = Marker()
        clear_marker.header.frame_id = 'map'
        clear_marker.header.stamp = self.get_clock().now().to_msg()
        clear_marker.ns = 'coverage'
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)
        
        # Coverage lines
        for i, line in enumerate(coverage_lines):
            for j, segment in enumerate(line['segments']):
                segment_marker = Marker()
                segment_marker.header.frame_id = 'map'
                segment_marker.header.stamp = self.get_clock().now().to_msg()
                segment_marker.ns = 'coverage'
                segment_marker.id = marker_id
                marker_id += 1
                segment_marker.type = Marker.LINE_STRIP
                segment_marker.action = Marker.ADD
                segment_marker.scale.x = 0.1
                
                # Alternate colors for visualization
                if i % 2 == 0:
                    segment_marker.color.r = 0.0
                    segment_marker.color.g = 1.0
                    segment_marker.color.b = 0.0
                else:
                    segment_marker.color.r = 1.0
                    segment_marker.color.g = 0.5
                    segment_marker.color.b = 0.0
                segment_marker.color.a = 1.0
                
                # Add segment points
                start_point = Point()
                start_point.x = float(segment['start'][0])
                start_point.y = float(segment['start'][1])
                start_point.z = 0.0
                segment_marker.points.append(start_point)
                
                end_point = Point()
                end_point.x = float(segment['end'][0])
                end_point.y = float(segment['end'][1])
                end_point.z = 0.0
                segment_marker.points.append(end_point)
                
                marker_array.markers.append(segment_marker)
        
        # Direction arrow
        if coverage_lines and bbox:
            arrow_marker = Marker()
            arrow_marker.header.frame_id = 'map'
            arrow_marker.header.stamp = self.get_clock().now().to_msg()
            arrow_marker.ns = 'coverage'
            arrow_marker.id = marker_id
            arrow_marker.type = Marker.ARROW
            arrow_marker.action = Marker.ADD
            arrow_marker.scale.x = 2.0
            arrow_marker.scale.y = 0.3
            arrow_marker.scale.z = 0.3
            arrow_marker.color.r = 1.0
            arrow_marker.color.g = 1.0
            arrow_marker.color.b = 0.0
            arrow_marker.color.a = 1.0
            
            # Position arrow in center
            arrow_marker.pose.position.x = bbox['center_x']
            arrow_marker.pose.position.y = bbox['center_y']
            arrow_marker.pose.position.z = 0.5
            
            # Orient arrow based on sweep direction
            if coverage_lines[0]['direction'] == 'horizontal':
                arrow_marker.pose.orientation.w = 1.0
            else:
                arrow_marker.pose.orientation.z = 0.707
                arrow_marker.pose.orientation.w = 0.707
            
            marker_array.markers.append(arrow_marker)
        
        self.coverage_pub.publish(marker_array)
        self.get_logger().info('🎨 Published coverage pattern visualization')


def main(args=None):
    rclpy.init(args=args)
    
    lawn_mower_planner = LawnMowerPlanner()
    
    try:
        rclpy.spin(lawn_mower_planner)
    except KeyboardInterrupt:
        pass
    
    lawn_mower_planner.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main() 
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped, TransformStamped
from sensor_msgs.msg import NavSatFix
import numpy as np
import threading
import tf2_ros
from tf2_ros import TransformException, TransformBroadcaster
import math

class MapMergeNode(Node):
    def __init__(self):
        super().__init__('map_merge_node')
        
        # QoS profile for map topics (latched, reliable, keep last)
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Subscribers for robot maps
        self.robot1_map_sub = self.create_subscription(
            OccupancyGrid,
            '/robot1/map',
            self.robot1_map_callback,
            qos_profile
        )
        
        self.robot2_map_sub = self.create_subscription(
            OccupancyGrid,
            '/robot2/map',
            self.robot2_map_callback,
            qos_profile
        )
        
        # Subscribers for robot positions
        self.robot1_pos_sub = self.create_subscription(
            PoseStamped,
            '/robot1/position',
            self.robot1_position_callback,
            qos_profile
        )
        
        self.robot2_pos_sub = self.create_subscription(
            PoseStamped,
            '/robot2/position',
            self.robot2_position_callback,
            qos_profile
        )
        
        # Subscribers for robot GPS coordinates
        self.robot1_gps_sub = self.create_subscription(
            NavSatFix,
            '/robot1/gps_position',
            self.robot1_gps_callback,
            qos_profile
        )
        
        self.robot2_gps_sub = self.create_subscription(
            NavSatFix,
            '/robot2/gps_position',
            self.robot2_gps_callback,
            qos_profile
        )
        
        # Publisher for merged map
        self.merged_map_pub = self.create_publisher(
            OccupancyGrid,
            '/merged_map',
            qos_profile
        )
        
        # TF2 setup
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # Store latest maps and positions
        self.robot1_map = None
        self.robot2_map = None
        self.robot1_position = None
        self.robot2_position = None
        
        # Store GPS coordinates
        self.robot1_gps = None
        self.robot2_gps = None
        
        # Parameters for map merging
        self.declare_parameter('merged_frame_id', 'merged_map')
        self.declare_parameter('map_resolution', 0.05)  # Default resolution 5cm/pixel
        self.declare_parameter('map_width', 4000)  # Default 4000 pixels
        self.declare_parameter('map_height', 4000)  # Default 4000 pixels
        self.declare_parameter('origin_x', -100.0)  # Default origin at -100m
        self.declare_parameter('origin_y', -100.0)  # Default origin at -100m
        self.declare_parameter('merge_frequency', 2.0)  # Merge frequency in Hz
        self.declare_parameter('publish_tf', True)  # Whether to publish TF transforms
        
        # Spatial offset parameters for robot maps
        self.declare_parameter('robot1_offset_x', 0.0)  # Robot1 map offset in merged frame
        self.declare_parameter('robot1_offset_y', 0.0)  # Robot1 map offset in merged frame
        self.declare_parameter('robot2_offset_x', 10.0)  # Robot2 map offset in merged frame (adjust for L-shape)
        self.declare_parameter('robot2_offset_y', 0.0)  # Robot2 map offset in merged frame
        self.declare_parameter('auto_calculate_offset', True)  # Automatically calculate offset from robot positions
        self.declare_parameter('offset_update_frequency', 0.5)  # How often to recalculate offset (Hz)
        
        # Manual initial position parameters (known starting positions)
        self.declare_parameter('use_known_positions', False)  # Use known initial positions
        self.declare_parameter('robot1_initial_x', 0.0)  # Robot1 known starting position X
        self.declare_parameter('robot1_initial_y', 0.0)  # Robot1 known starting position Y
        self.declare_parameter('robot2_initial_x', 10.0)  # Robot2 known starting position X
        self.declare_parameter('robot2_initial_y', 5.0)   # Robot2 known starting position Y
        
        # GPS-based offset calculation parameters
        self.declare_parameter('use_gps_coordinates', True)  # Use GPS coordinates for offset calculation
        self.declare_parameter('gps_origin_lat', 0.0)  # GPS latitude origin for local coordinate system
        self.declare_parameter('gps_origin_lon', 0.0)  # GPS longitude origin for local coordinate system
        
        self.merged_frame_id = self.get_parameter('merged_frame_id').value
        self.map_resolution = self.get_parameter('map_resolution').value
        self.map_width = self.get_parameter('map_width').value
        self.map_height = self.get_parameter('map_height').value
        self.origin_x = self.get_parameter('origin_x').value
        self.origin_y = self.get_parameter('origin_y').value
        self.merge_frequency = self.get_parameter('merge_frequency').value
        self.publish_tf = self.get_parameter('publish_tf').value
        
        # Robot offset parameters
        self.robot1_offset_x = self.get_parameter('robot1_offset_x').value
        self.robot1_offset_y = self.get_parameter('robot1_offset_y').value
        self.robot2_offset_x = self.get_parameter('robot2_offset_x').value
        self.robot2_offset_y = self.get_parameter('robot2_offset_y').value
        self.auto_calculate_offset = self.get_parameter('auto_calculate_offset').value
        self.offset_update_frequency = self.get_parameter('offset_update_frequency').value
        
        # Known position parameters
        self.use_known_positions = self.get_parameter('use_known_positions').value
        self.robot1_initial_x = self.get_parameter('robot1_initial_x').value
        self.robot1_initial_y = self.get_parameter('robot1_initial_y').value
        self.robot2_initial_x = self.get_parameter('robot2_initial_x').value
        self.robot2_initial_y = self.get_parameter('robot2_initial_y').value
        
        # GPS parameters
        self.use_gps_coordinates = self.get_parameter('use_gps_coordinates').value
        self.gps_origin_lat = self.get_parameter('gps_origin_lat').value
        self.gps_origin_lon = self.get_parameter('gps_origin_lon').value
        
        # Calculated offsets (will be updated automatically if enabled)
        self.calculated_robot1_offset_x = self.robot1_offset_x
        self.calculated_robot1_offset_y = self.robot1_offset_y
        self.calculated_robot2_offset_x = self.robot2_offset_x
        self.calculated_robot2_offset_y = self.robot2_offset_y
        
        # Reference positions for offset calculation
        self.robot1_reference_position = None
        self.robot2_reference_position = None
        self.offset_calculated = False
        
        # Thread lock for map data
        self.map_lock = threading.Lock()
        
        # Timer for periodic map merging
        self.merge_timer = self.create_timer(1.0 / self.merge_frequency, self.merge_and_publish_maps)
        
        # Timer for TF publishing (higher frequency)
        if self.publish_tf:
            self.tf_timer = self.create_timer(0.1, self.publish_tf_transforms)
        
        # Timer for automatic offset calculation
        if self.auto_calculate_offset:
            self.offset_timer = self.create_timer(1.0 / self.offset_update_frequency, self.calculate_robot_offsets)
        
        # Counters for logging
        self.robot1_map_count = 0
        self.robot2_map_count = 0
        self.merged_map_count = 0
        
        self.get_logger().info("Map merge node initialized")
        self.get_logger().info(f"Merged map parameters: {self.map_width}x{self.map_height} pixels, "
                              f"resolution: {self.map_resolution}m/pixel")
        self.get_logger().info(f"Origin: ({self.origin_x}, {self.origin_y})")
        if self.auto_calculate_offset:
            if self.use_gps_coordinates:
                self.get_logger().info("GPS-based offset calculation ENABLED - will use GPS coordinates")
                if self.gps_origin_lat != 0.0 or self.gps_origin_lon != 0.0:
                    self.get_logger().info(f"GPS origin set: ({self.gps_origin_lat:.8f}, {self.gps_origin_lon:.8f})")
                else:
                    self.get_logger().info("GPS origin will be auto-set from first robot GPS position")
            elif self.use_known_positions:
                self.get_logger().info("Known position calculation ENABLED - will use manual positions")
            else:
                self.get_logger().info("Automatic offset calculation ENABLED - will calculate from robot positions")
        else:
            self.get_logger().info(f"Robot1 spatial offset: ({self.robot1_offset_x}, {self.robot1_offset_y})")
            self.get_logger().info(f"Robot2 spatial offset: ({self.robot2_offset_x}, {self.robot2_offset_y})")
        if self.publish_tf:
            self.get_logger().info("TF publishing enabled for merged_map frame")
        
        # Status timer
        self.status_timer = self.create_timer(10.0, self.status_callback)
    
    def status_callback(self):
        """Log status information"""
        self.get_logger().info(f"Map status - Robot1: {self.robot1_map_count} maps, "
                              f"Robot2: {self.robot2_map_count} maps, "
                              f"Merged: {self.merged_map_count} maps published")
    
    def robot1_map_callback(self, msg):
        """Callback for robot1 map"""
        with self.map_lock:
            self.robot1_map = msg
            self.robot1_map_count += 1
            self.get_logger().debug(f"Received robot1 map #{self.robot1_map_count}")
    
    def robot2_map_callback(self, msg):
        """Callback for robot2 map"""
        with self.map_lock:
            self.robot2_map = msg
            self.robot2_map_count += 1
            self.get_logger().debug(f"Received robot2 map #{self.robot2_map_count}")
    
    def robot1_position_callback(self, msg):
        """Callback for robot1 position"""
        self.robot1_position = msg
        self.get_logger().debug(f"Received robot1 position: ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})")
    
    def robot2_position_callback(self, msg):
        """Callback for robot2 position"""
        self.robot2_position = msg
        self.get_logger().debug(f"Received robot2 position: ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})")
    
    def robot1_gps_callback(self, msg):
        """Callback for robot1 GPS coordinates"""
        self.robot1_gps = msg
        self.get_logger().debug(f"Received robot1 GPS coordinates: ({msg.latitude}, {msg.longitude})")
    
    def robot2_gps_callback(self, msg):
        """Callback for robot2 GPS coordinates"""
        self.robot2_gps = msg
        self.get_logger().debug(f"Received robot2 GPS coordinates: ({msg.latitude}, {msg.longitude})")
    
    def gps_to_local_coordinates(self, lat, lon, origin_lat, origin_lon):
        """
        Convert GPS coordinates (lat, lon) to local metric coordinates (x, y)
        using a simple equirectangular projection relative to an origin point.
        
        Args:
            lat, lon: GPS coordinates to convert
            origin_lat, origin_lon: Origin point for local coordinate system
            
        Returns:
            x, y: Local coordinates in meters
        """
        # Earth radius in meters
        R = 6378137.0
        
        # Convert degrees to radians
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)
        origin_lat_rad = math.radians(origin_lat)
        origin_lon_rad = math.radians(origin_lon)
        
        # Calculate differences
        dlat = lat_rad - origin_lat_rad
        dlon = lon_rad - origin_lon_rad
        
        # Convert to meters using equirectangular projection
        x = R * dlon * math.cos(origin_lat_rad)
        y = R * dlat
        
        return x, y
    
    def calculate_gps_origin(self):
        """
        Calculate GPS origin automatically if not set manually.
        Uses the first robot's GPS position as origin.
        """
        if self.robot1_gps is not None and (self.gps_origin_lat == 0.0 and self.gps_origin_lon == 0.0):
            self.gps_origin_lat = self.robot1_gps.latitude
            self.gps_origin_lon = self.robot1_gps.longitude
            self.get_logger().info(f"Auto-set GPS origin to Robot1's position: ({self.gps_origin_lat:.8f}, {self.gps_origin_lon:.8f})")
            return True
        return self.gps_origin_lat != 0.0 or self.gps_origin_lon != 0.0
    
    def calculate_robot_offsets(self):
        """Automatically calculate spatial offsets based on robot positions and map data"""
        if not self.auto_calculate_offset:
            return
            
        # Check if we have both robot positions and maps
        if (self.robot1_position is None or self.robot2_position is None or 
            self.robot1_map is None or self.robot2_map is None):
            return
        
        try:
            # Priority 1: Use GPS coordinates if available
            if (self.use_gps_coordinates and 
                self.robot1_gps is not None and self.robot2_gps is not None and 
                not self.offset_calculated):
                
                # Ensure we have a GPS origin
                if not self.calculate_gps_origin():
                    self.get_logger().warn("GPS origin not set, cannot calculate GPS-based offsets")
                    return
                
                # Convert GPS coordinates to local metric coordinates
                robot1_gps_x, robot1_gps_y = self.gps_to_local_coordinates(
                    self.robot1_gps.latitude, self.robot1_gps.longitude,
                    self.gps_origin_lat, self.gps_origin_lon
                )
                
                robot2_gps_x, robot2_gps_y = self.gps_to_local_coordinates(
                    self.robot2_gps.latitude, self.robot2_gps.longitude,
                    self.gps_origin_lat, self.gps_origin_lon
                )
                
                # Get map origins and robot positions for proper alignment
                robot1_map_origin_x = self.robot1_map.info.origin.position.x
                robot1_map_origin_y = self.robot1_map.info.origin.position.y
                robot2_map_origin_x = self.robot2_map.info.origin.position.x
                robot2_map_origin_y = self.robot2_map.info.origin.position.y
                
                # Get robot positions in their respective map coordinate systems
                robot1_current_x = self.robot1_position.pose.position.x if self.robot1_position else 0.0
                robot1_current_y = self.robot1_position.pose.position.y if self.robot1_position else 0.0
                robot2_current_x = self.robot2_position.pose.position.x if self.robot2_position else 0.0
                robot2_current_y = self.robot2_position.pose.position.y if self.robot2_position else 0.0
                
                # Calculate map offsets to align robot positions with GPS positions
                # Formula: GPS_position - robot_position_in_map - map_origin = map_offset
                self.calculated_robot1_offset_x = robot1_gps_x - robot1_current_x - robot1_map_origin_x
                self.calculated_robot1_offset_y = robot1_gps_y - robot1_current_y - robot1_map_origin_y
                self.calculated_robot2_offset_x = robot2_gps_x - robot2_current_x - robot2_map_origin_x
                self.calculated_robot2_offset_y = robot2_gps_y - robot2_current_y - robot2_map_origin_y
                
                self.offset_calculated = True
                
                distance = np.sqrt((robot2_gps_x - robot1_gps_x)**2 + (robot2_gps_y - robot1_gps_y)**2)
                
                self.get_logger().info("=== USING GPS COORDINATES ===")
                self.get_logger().info(f"GPS Origin: ({self.gps_origin_lat:.8f}, {self.gps_origin_lon:.8f})")
                self.get_logger().info(f"Robot1 GPS: ({self.robot1_gps.latitude:.8f}, {self.robot1_gps.longitude:.8f}) → Local: ({robot1_gps_x:.2f}, {robot1_gps_y:.2f})")
                self.get_logger().info(f"Robot2 GPS: ({self.robot2_gps.latitude:.8f}, {self.robot2_gps.longitude:.8f}) → Local: ({robot2_gps_x:.2f}, {robot2_gps_y:.2f})")
                self.get_logger().info(f"Robot1 position in map: ({robot1_current_x:.2f}, {robot1_current_y:.2f})")
                self.get_logger().info(f"Robot2 position in map: ({robot2_current_x:.2f}, {robot2_current_y:.2f})")
                self.get_logger().info(f"Robot1 map origin: ({robot1_map_origin_x:.2f}, {robot1_map_origin_y:.2f})")
                self.get_logger().info(f"Robot2 map origin: ({robot2_map_origin_x:.2f}, {robot2_map_origin_y:.2f})")
                self.get_logger().info(f"Calculated Robot1 map offset: ({self.calculated_robot1_offset_x:.2f}, {self.calculated_robot1_offset_y:.2f})")
                self.get_logger().info(f"Calculated Robot2 map offset: ({self.calculated_robot2_offset_x:.2f}, {self.calculated_robot2_offset_y:.2f})")
                self.get_logger().info(f"Distance between robots (GPS): {distance:.2f} meters")
                self.get_logger().info("=== END GPS COORDINATES ===")
                return
            
            # Priority 2: Use known positions if GPS not available
            if self.use_known_positions and not self.offset_calculated:
                self.calculated_robot1_offset_x = self.robot1_initial_x
                self.calculated_robot1_offset_y = self.robot1_initial_y
                self.calculated_robot2_offset_x = self.robot2_initial_x
                self.calculated_robot2_offset_y = self.robot2_initial_y
                
                self.offset_calculated = True
                
                distance = np.sqrt((self.robot2_initial_x - self.robot1_initial_x)**2 + 
                                 (self.robot2_initial_y - self.robot1_initial_y)**2)
                
                self.get_logger().info("=== USING KNOWN ROBOT POSITIONS ===")
                self.get_logger().info(f"Robot1 known position: ({self.robot1_initial_x:.2f}, {self.robot1_initial_y:.2f})")
                self.get_logger().info(f"Robot2 known position: ({self.robot2_initial_x:.2f}, {self.robot2_initial_y:.2f})")
                self.get_logger().info(f"Distance between robots: {distance:.2f} meters")
                self.get_logger().info(f"Applied Robot1 offset: ({self.calculated_robot1_offset_x:.2f}, {self.calculated_robot1_offset_y:.2f})")
                self.get_logger().info(f"Applied Robot2 offset: ({self.calculated_robot2_offset_x:.2f}, {self.calculated_robot2_offset_y:.2f})")
                self.get_logger().info("=== END KNOWN POSITIONS ===")
                return
            
            # Get robot positions in their respective coordinate frames
            robot1_pos_x = self.robot1_position.pose.position.x
            robot1_pos_y = self.robot1_position.pose.position.y
            robot2_pos_x = self.robot2_position.pose.position.x
            robot2_pos_y = self.robot2_position.pose.position.y
            
            # Get map origins
            robot1_map_origin_x = self.robot1_map.info.origin.position.x
            robot1_map_origin_y = self.robot1_map.info.origin.position.y
            robot2_map_origin_x = self.robot2_map.info.origin.position.x
            robot2_map_origin_y = self.robot2_map.info.origin.position.y
            
            # If this is the first calculation, store reference positions
            if not self.offset_calculated:
                self.robot1_reference_position = (robot1_pos_x, robot1_pos_y)
                self.robot2_reference_position = (robot2_pos_x, robot2_pos_y)
                
                # Calculate the offset needed to align the maps
                # Assume robot1 is at origin (0,0) in merged frame
                self.calculated_robot1_offset_x = 0.0
                self.calculated_robot1_offset_y = 0.0
                
                # Calculate robot2 offset based on relative positions
                # This assumes robots are in known relative positions
                relative_x = robot2_pos_x - robot1_pos_x
                relative_y = robot2_pos_y - robot1_pos_y
                
                # Add map origin differences
                origin_diff_x = robot2_map_origin_x - robot1_map_origin_x
                origin_diff_y = robot2_map_origin_y - robot1_map_origin_y
                
                self.calculated_robot2_offset_x = relative_x + origin_diff_x
                self.calculated_robot2_offset_y = relative_y + origin_diff_y
                
                self.offset_calculated = True
                
                self.get_logger().info("=== CALCULATED ROBOT OFFSETS ===")
                self.get_logger().info(f"Robot1 position: ({robot1_pos_x:.2f}, {robot1_pos_y:.2f})")
                self.get_logger().info(f"Robot2 position: ({robot2_pos_x:.2f}, {robot2_pos_y:.2f})")
                self.get_logger().info(f"Robot1 map origin: ({robot1_map_origin_x:.2f}, {robot1_map_origin_y:.2f})")
                self.get_logger().info(f"Robot2 map origin: ({robot2_map_origin_x:.2f}, {robot2_map_origin_y:.2f})")
                self.get_logger().info(f"Calculated Robot1 offset: ({self.calculated_robot1_offset_x:.2f}, {self.calculated_robot1_offset_y:.2f})")
                self.get_logger().info(f"Calculated Robot2 offset: ({self.calculated_robot2_offset_x:.2f}, {self.calculated_robot2_offset_y:.2f})")
                self.get_logger().info(f"Distance between robots: {np.sqrt(relative_x**2 + relative_y**2):.2f} meters")
                self.get_logger().info("=== END CALCULATED OFFSETS ===")
                
        except Exception as e:
            self.get_logger().error(f"Error calculating robot offsets: {e}")
    
    def world_to_map_coords(self, world_x, world_y):
        """Convert world coordinates to map pixel coordinates"""
        map_x = int((world_x - self.origin_x) / self.map_resolution)
        map_y = int((world_y - self.origin_y) / self.map_resolution)
        return map_x, map_y
    
    def merge_occupancy_values(self, val1, val2):
        """
        Merge two occupancy grid values
        -1: unknown, 0: free, 100: occupied
        """
        # If either value is occupied, result is occupied
        if val1 == 100 or val2 == 100:
            return 100
        # If either value is free, and the other is not occupied, result is free
        elif val1 == 0 or val2 == 0:
            return 0
        # If both are unknown, result is unknown
        else:
            return -1
    
    def merge_maps(self, map1, map2):
        """Merge two occupancy grid maps into a single map"""
        if map1 is None or map2 is None:
            return None
        
        # Create merged map message
        merged_map = OccupancyGrid()
        merged_map.header.stamp = self.get_clock().now().to_msg()
        merged_map.header.frame_id = self.merged_frame_id
        
        # Set map info
        merged_map.info.resolution = self.map_resolution
        merged_map.info.width = self.map_width
        merged_map.info.height = self.map_height
        merged_map.info.origin.position.x = self.origin_x
        merged_map.info.origin.position.y = self.origin_y
        merged_map.info.origin.position.z = 0.0
        merged_map.info.origin.orientation.w = 1.0
        
        # Initialize merged map data with unknown values (-1)
        merged_data = np.full(self.map_width * self.map_height, -1, dtype=np.int8)
        
        # Process map1 (robot1)
        if map1 is not None:
            self.integrate_map_data(merged_data, map1, "robot1")
        
        # Process map2 (robot2)
        if map2 is not None:
            self.integrate_map_data(merged_data, map2, "robot2")
        
        merged_map.data = merged_data.tolist()
        return merged_map
    
    def integrate_map_data(self, merged_data, robot_map, robot_name):
        """Integrate a robot's map data into the merged map"""
        try:
            # Get robot map properties
            robot_width = robot_map.info.width
            robot_height = robot_map.info.height
            robot_resolution = robot_map.info.resolution
            robot_origin_x = robot_map.info.origin.position.x
            robot_origin_y = robot_map.info.origin.position.y
            
            # Apply robot-specific spatial offsets
            if robot_name == "robot1":
                spatial_offset_x = self.calculated_robot1_offset_x
                spatial_offset_y = self.calculated_robot1_offset_y
            elif robot_name == "robot2":
                spatial_offset_x = self.calculated_robot2_offset_x
                spatial_offset_y = self.calculated_robot2_offset_y
            else:
                spatial_offset_x = 0.0
                spatial_offset_y = 0.0
            
            # Convert robot map data to numpy array
            robot_data = np.array(robot_map.data, dtype=np.int8).reshape(robot_height, robot_width)
            
            # Calculate offset between robot map origin and merged map origin in merged map coordinates
            # Include the spatial offset for proper robot positioning
            offset_x = int((robot_origin_x + spatial_offset_x - self.origin_x) / self.map_resolution)
            offset_y = int((robot_origin_y + spatial_offset_y - self.origin_y) / self.map_resolution)
            
            # Calculate scaling factor if resolutions are different
            scale_factor = robot_resolution / self.map_resolution
            
            for robot_y in range(robot_height):
                for robot_x in range(robot_width):
                    robot_value = robot_data[robot_y, robot_x]
                    
                    # Skip unknown values (-1) when integrating
                    if robot_value == -1:
                        continue
                    
                    # Calculate corresponding position in merged map
                    merged_x = int(offset_x + robot_x * scale_factor)
                    merged_y = int(offset_y + robot_y * scale_factor)
                    
                    # Check bounds
                    if (0 <= merged_x < self.map_width and 0 <= merged_y < self.map_height):
                        merged_idx = merged_y * self.map_width + merged_x
                        
                        # Merge the occupancy values
                        current_value = merged_data[merged_idx]
                        merged_data[merged_idx] = self.merge_occupancy_values(current_value, robot_value)
            
            self.get_logger().debug(f"Integrated {robot_name} map data: "
                                   f"{robot_width}x{robot_height} at offset ({offset_x}, {offset_y}) "
                                   f"with spatial offset ({spatial_offset_x:.1f}, {spatial_offset_y:.1f})")
        
        except Exception as e:
            self.get_logger().error(f"Error integrating {robot_name} map: {e}")
    
    def merge_and_publish_maps(self):
        """Merge maps and publish the result"""
        try:
            with self.map_lock:
                # Check if we have at least one map
                if self.robot1_map is None and self.robot2_map is None:
                    return
                
                # Create merged map
                merged_map = self.merge_maps(self.robot1_map, self.robot2_map)
                
                if merged_map is not None:
                    # Publish merged map
                    self.merged_map_pub.publish(merged_map)
                    self.merged_map_count += 1
                    
                    # Calculate statistics
                    data_array = np.array(merged_map.data)
                    occupied_cells = np.sum(data_array == 100)
                    free_cells = np.sum(data_array == 0)
                    unknown_cells = np.sum(data_array == -1)
                    total_cells = len(data_array)
                    
                    self.get_logger().debug(f"Published merged map #{self.merged_map_count}")
                    self.get_logger().debug(f"Map stats - Occupied: {occupied_cells}, "
                                           f"Free: {free_cells}, Unknown: {unknown_cells}, "
                                           f"Total: {total_cells}")
        
        except Exception as e:
            self.get_logger().error(f"Error in merge_and_publish_maps: {e}")

    def publish_tf_transforms(self):
        """Publish TF transform for merged_map frame"""
        if not self.publish_tf:
            return
            
        try:
            current_time = self.get_clock().now().to_msg()
            
            # Publish merged_map to map transform (identity)
            t = TransformStamped()
            t.header.stamp = current_time
            t.header.frame_id = 'map'  # Parent frame
            t.child_frame_id = self.merged_frame_id
            
            # Identity transform (no translation or rotation)
            t.transform.translation.x = 0.0
            t.transform.translation.y = 0.0
            t.transform.translation.z = 0.0
            t.transform.rotation.x = 0.0
            t.transform.rotation.y = 0.0
            t.transform.rotation.z = 0.0
            t.transform.rotation.w = 1.0
            
            # Publish robot1/map to merged_map transform
            t1 = TransformStamped()
            t1.header.stamp = current_time
            t1.header.frame_id = self.merged_frame_id
            t1.child_frame_id = 'robot1/map'
            
            # Use robot1 spatial offset
            t1.transform.translation.x = self.calculated_robot1_offset_x
            t1.transform.translation.y = self.calculated_robot1_offset_y  
            t1.transform.translation.z = 0.0
            t1.transform.rotation.x = 0.0
            t1.transform.rotation.y = 0.0
            t1.transform.rotation.z = 0.0
            t1.transform.rotation.w = 1.0
            
            # Publish robot2/map to merged_map transform  
            t2 = TransformStamped()
            t2.header.stamp = current_time
            t2.header.frame_id = self.merged_frame_id
            t2.child_frame_id = 'robot2/map'
            
            # Use robot2 spatial offset
            t2.transform.translation.x = self.calculated_robot2_offset_x
            t2.transform.translation.y = self.calculated_robot2_offset_y
            t2.transform.translation.z = 0.0
            t2.transform.rotation.x = 0.0
            t2.transform.rotation.y = 0.0
            t2.transform.rotation.z = 0.0
            t2.transform.rotation.w = 1.0
            
            # Send all transforms
            self.tf_broadcaster.sendTransform([t, t1, t2])
            
        except Exception as e:
            self.get_logger().debug(f"Error publishing TF transform: {e}")


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = MapMergeNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error in map merge node: {e}")
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

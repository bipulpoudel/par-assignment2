#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import NavSatFix
import tf2_ros
from tf2_ros import TransformException
import threading
import os

class Robot1MapRepublisher:
    def __init__(self):
        # Get current domain ID from environment variable
        current_domain_id = int(os.environ.get('ROS_DOMAIN_ID', '0'))
        
        # Set ROBOT1_TARGET_DOMAIN_ID environment variable for other nodes to use
        os.environ['ROBOT1_TARGET_DOMAIN_ID'] = str(current_domain_id)
        
        # Create separate contexts for different domains
        self.subscriber_context = Context()
        self.publisher_context = Context()
        
        # Initialize contexts with different domain IDs
        # Subscriber uses current robot's domain ID, publisher uses domain 0
        rclpy.init(context=self.subscriber_context, domain_id=current_domain_id)
        rclpy.init(context=self.publisher_context, domain_id=0)
        
        # Create subscriber node on current domain
        self.subscriber_node = Node('robot1_map_subscriber', context=self.subscriber_context)
        
        # Log the current domain ID and environment variable setting
        self.subscriber_node.get_logger().info(f"Current robot domain ID: {current_domain_id}")
        self.subscriber_node.get_logger().info(f"Set ROBOT1_TARGET_DOMAIN_ID environment variable to: {current_domain_id}")
        
        # Create publisher node on domain 0
        self.publisher_node = Node('robot1_map_publisher', context=self.publisher_context)
        
        # TF2 setup for getting robot position
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self.subscriber_node)
        
        # Create executors for each context
        self.subscriber_executor = SingleThreadedExecutor(context=self.subscriber_context)
        self.publisher_executor = SingleThreadedExecutor(context=self.publisher_context)
        
        # Add nodes to their respective executors
        self.subscriber_executor.add_node(self.subscriber_node)
        self.publisher_executor.add_node(self.publisher_node)
        
        # QoS profile for map topics (latched, reliable, keep last)
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Subscriber to /map on current domain
        self.map_sub = self.subscriber_node.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            qos_profile
        )

        # Subscriber to /gps on current domain
        self.gps_sub = self.subscriber_node.create_subscription(
            NavSatFix,
            '/gps',
            self.gps_callback,
            qos_profile
        )

        # Publishers on domain 0
        self.map_pub = self.publisher_node.create_publisher(
            OccupancyGrid,
            '/robot1/map',
            qos_profile
        )
        
        # Publisher for robot1 position
        self.position_pub = self.publisher_node.create_publisher(
            PoseStamped,
            '/robot1/position',
            qos_profile
        )
        
        # Publisher for robot1 GPS
        self.gps_pub = self.publisher_node.create_publisher(
            NavSatFix,
            '/robot1/gps_position',
            qos_profile
        )

        # Parameters
        self.publisher_node.declare_parameter('map_frame', 'robot1/map')
        self.publisher_node.declare_parameter('robot_base_frame', 'base_link')
        self.publisher_node.declare_parameter('position_frame', 'merged_map')
        
        self.map_frame = self.publisher_node.get_parameter('map_frame').value
        self.robot_base_frame = self.publisher_node.get_parameter('robot_base_frame').value
        self.position_frame = self.publisher_node.get_parameter('position_frame').value

        self.subscriber_node.get_logger().info(f"Robot1 map republisher initialized")
        self.subscriber_node.get_logger().info(f"Subscribing to /map on domain {current_domain_id}, publishing to /robot1/map on domain 0")
        self.subscriber_node.get_logger().info(f"Subscribing to /gps on domain {current_domain_id}, publishing to /robot1/gps_position on domain 0")
        self.subscriber_node.get_logger().info(f"Publishing robot1 position to /robot1/position on domain 0")

        # Add a timer to periodically log that we're waiting for map data and publish position
        self.timer = self.subscriber_node.create_timer(5.0, self.timer_callback)
        self.position_timer = self.subscriber_node.create_timer(0.1, self.publish_position)  # 10 Hz
        self.map_received_count = 0
        self.gps_received_count = 0

    def timer_callback(self):
        """Timer callback to show node is alive and waiting for map data"""
        self.subscriber_node.get_logger().info(f"Node alive - waiting for /map data. Maps received so far: {self.map_received_count}")

    def get_robot_position(self):
        """Get robot position using TF2, always in merged_map frame"""
        try:
            # First try to get position directly in merged_map frame
            try:
                transform = self.tf_buffer.lookup_transform(
                    'merged_map',
                    self.robot_base_frame,
                    rclpy.time.Time()
                )
                frame_used = 'merged_map'
                self.subscriber_node.get_logger().debug("Using direct transform to merged_map")
                
            except TransformException:
                # If merged_map frame doesn't exist, try to get position in map frame
                # and assume map frame == merged_map frame (they should be aligned)
                try:
                    transform = self.tf_buffer.lookup_transform(
                        'map',
                        self.robot_base_frame,
                        rclpy.time.Time()
                    )
                    frame_used = 'merged_map'  # Publish as merged_map frame
                    self.subscriber_node.get_logger().debug("Using map frame, publishing as merged_map")
                    
                except TransformException as e:
                    self.subscriber_node.get_logger().debug(f"Could not get robot position in any frame: {e}")
                    return None
            
            # Convert to PoseStamped with robot1/map frame
            pose = PoseStamped()
            pose.header.frame_id = 'robot1/map'  # Use robot1's own frame
            pose.header.stamp = self.subscriber_node.get_clock().now().to_msg()
            pose.pose.position.x = transform.transform.translation.x
            pose.pose.position.y = transform.transform.translation.y
            pose.pose.position.z = transform.transform.translation.z
            pose.pose.orientation = transform.transform.rotation
            
            return pose
            
        except Exception as e:
            self.subscriber_node.get_logger().debug(f"Error getting robot position: {e}")
            return None

    def publish_position(self):
        """Publish robot1 position in robot1/map frame"""
        position = self.get_robot_position()
        if position is not None:
            self.position_pub.publish(position)
            self.subscriber_node.get_logger().debug(f"Published robot1 position in robot1/map frame: ({position.pose.position.x:.2f}, {position.pose.position.y:.2f})")
        else:
            self.subscriber_node.get_logger().debug("Could not get robot1 position for publishing")

    def map_callback(self, msg):
        """Callback to republish /map to /robot1/map with updated frame_id"""
        try:
            self.map_received_count += 1
            if self.map_received_count % 10 == 1:  # Log every 10th message
                self.subscriber_node.get_logger().info(f"Received map message #{self.map_received_count}, republishing to /robot1/map")
            
            # Create a new message with updated frame_id
            republished_map = OccupancyGrid()
            republished_map.header = msg.header
            republished_map.header.frame_id = self.map_frame
            republished_map.info = msg.info
            republished_map.data = msg.data

            # Publish to /robot1/map on domain 0
            self.map_pub.publish(republished_map)
            
            if self.map_received_count % 10 == 1:  # Log every 10th message
                self.subscriber_node.get_logger().info("Successfully republished map to /robot1/map on domain 0")
        except Exception as e:
            self.subscriber_node.get_logger().error(f"Error republishing map: {e}")
    
    def gps_callback(self, msg):
        """Callback to republish /gps to /robot1/gps_position"""
        try:
            self.gps_received_count += 1
            if self.gps_received_count % 5 == 1:  # Log every 5th GPS message
                self.subscriber_node.get_logger().info(f"Received GPS message #{self.gps_received_count}, republishing to /robot1/gps_position")
            
            # Create a new message (GPS messages don't need frame_id changes)
            republished_gps = NavSatFix()
            republished_gps.header = msg.header
            republished_gps.status = msg.status
            republished_gps.latitude = msg.latitude
            republished_gps.longitude = msg.longitude
            republished_gps.altitude = msg.altitude
            republished_gps.position_covariance = msg.position_covariance
            republished_gps.position_covariance_type = msg.position_covariance_type

            # Publish to /robot1/gps_position on domain 0
            self.gps_pub.publish(republished_gps)
            
            if self.gps_received_count % 5 == 1:  # Log every 5th GPS message
                self.subscriber_node.get_logger().info(f"Successfully republished GPS to /robot1/gps_position on domain 0: "
                                                      f"lat={msg.latitude:.8f}, lon={msg.longitude:.8f}")
        except Exception as e:
            self.subscriber_node.get_logger().error(f"Error republishing GPS: {e}")

    def spin(self):
        """Spin both executors using threading"""
        # Create threads for spinning each executor
        subscriber_thread = threading.Thread(
            target=self.subscriber_executor.spin,
            daemon=True
        )
        publisher_thread = threading.Thread(
            target=self.publisher_executor.spin,
            daemon=True
        )
        
        # Start both threads
        subscriber_thread.start()
        publisher_thread.start()
        
        try:
            # Keep main thread alive
            subscriber_thread.join()
            publisher_thread.join()
        except KeyboardInterrupt:
            pass
    
    def destroy(self):
        """Clean up nodes, executors and contexts"""
        self.subscriber_executor.shutdown()
        self.publisher_executor.shutdown()
        self.subscriber_node.destroy_node()
        self.publisher_node.destroy_node()
        rclpy.shutdown(context=self.subscriber_context)
        rclpy.shutdown(context=self.publisher_context)

def main(args=None):
    republisher = Robot1MapRepublisher()
    try:
        republisher.spin()
    except KeyboardInterrupt:
        pass
    finally:
        republisher.destroy()

if __name__ == '__main__':
    main()
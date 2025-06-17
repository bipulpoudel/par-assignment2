#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# Configuration - Default IP address (change this to your laptop's IP)
DEFAULT_LAPTOP_IP = "192.168.100.47"

def generate_launch_description():
    # Declare launch arguments
    self_aruco_id_arg = DeclareLaunchArgument(
        'self_aruco_id',
        default_value='0',
        description='ArUco marker ID for this robot'
    )
    
    server_host_arg = DeclareLaunchArgument(
        'server_host',
        default_value=DEFAULT_LAPTOP_IP,
        description=f'IP address of the Python server (default: {DEFAULT_LAPTOP_IP})'
    )
    
    server_port_arg = DeclareLaunchArgument(
        'server_port',
        default_value='8888',
        description='Port number for the Python server'
    )
    
    # Get launch configuration
    self_aruco_id = LaunchConfiguration('self_aruco_id')
    server_host = LaunchConfiguration('server_host')
    server_port = LaunchConfiguration('server_port')
    
    # Node configurations
    game_coordinator_node = Node(
        package='swarm_tag',
        executable='game_coordinator',
        name='game_coordinator',
        parameters=[{
            'self_aruco_id': self_aruco_id,
            'timeout_duration': 10.0,
            'random_walk_duration': 20.0
        }],
        output='screen'
    )
    
    aruco_marker_detector_node = Node(
        package='swarm_tag',
        executable='aruco_marker_detector',
        name='aruco_marker_detector',
        parameters=[{
            'self_aruco_id': self_aruco_id,
            'aruco_marker_id': self_aruco_id  # Use same ID as parameter
        }],
        output='screen'
    )
    
    hider_node = Node(
        package='swarm_tag',
        executable='hider_node',
        name='hider_node',
        parameters=[{
            'self_aruco_id': self_aruco_id
        }],
        output='screen'
    )
    
    tag_server_node = Node(
        package='swarm_tag',
        executable='tag_server',
        name='tag_server',
        parameters=[{
            'self_aruco_id': self_aruco_id,
            'server_host': server_host,
            'server_port': server_port
        }],
        output='screen'
    )
    
    explore_node = Node(
        package='swarm_tag',
        executable='explore_node',
        name='explore_node',
        parameters=[{
            'self_aruco_id': self_aruco_id
        }],
        output='screen'
    )
    
    # Log info messages
    log_info = LogInfo(
        msg=['Starting swarm nodes with self_aruco_id: ', self_aruco_id]
    )
    
    log_server_info = LogInfo(
        msg=['Python server should be started separately on: ', server_host, ':', server_port]
    )
    
    return LaunchDescription([
        self_aruco_id_arg,
        server_host_arg,
        server_port_arg,
        log_info,
        log_server_info,
        game_coordinator_node,
        aruco_marker_detector_node,
        hider_node,
        tag_server_node,
        explore_node
    ]) 

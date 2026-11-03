import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    pkg_sensado = get_package_share_directory('sensado')
    
    world_file = os.path.join(pkg_sensado, 'worlds', 'pared.world')
    urdf_file = os.path.join(pkg_sensado, 'urdf', 'robot_pared.urdf.xacro')
    
    with open(urdf_file, 'r') as infp:
        robot_desc = infp.read()

    # 1. Iniciar Gazebo Fortress
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': f'-r {world_file}'}.items(),
    )

    # 2. Publicador del estado del robot
    rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_desc}]
    )

    # 3. Spawn del robot en Gazebo
    spawn = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=['-topic', 'robot_description', '-name', 'robot_desmalezador', '-z', '0.2'],
        output='screen'
    )

    # 4. Puente de Tópicos (ROS 2 <-> Gazebo)
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/cmd_vel@geometry_msgs/msg/Twist@ignition.msgs.Twist',
            '/scan@sensor_msgs/msg/LaserScan@ignition.msgs.LaserScan',
            '/odom@nav_msgs/msg/Odometry@ignition.msgs.Odometry',
            '/tf@tf2_msgs/msg/TFMessage@ignition.msgs.Pose_V'
        ],
        output='screen'
    )

    # 5. Nodos de control (con 5 segundos de retraso para que Gazebo cargue bien)
    nodo_test = Node(
        package='sensado',
        executable='test_pared',
        output='screen'
    )
    
    nodo_control = Node(
        package='sensado',
        executable='control_pared',
        output='screen'
    )

    return LaunchDescription([
        gz_sim,
        rsp,
        spawn,
        bridge,
        TimerAction(period=5.0, actions=[nodo_test, nodo_control])
    ])
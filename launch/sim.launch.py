import os
import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    pkg_sensado = get_package_share_directory('sensado')
    ros_gz_sim_pkg_path = get_package_share_directory('ros_gz_sim')
    sensado_share_path = FindPackageShare('sensado')

    xacro_file = os.path.join(pkg_sensado, 'urdf', 'robot_desmalezador.urdf.xacro')
    world_file = os.path.join(pkg_sensado, 'worlds', 'cultivo_arboles.world')
    gz_launch_path = PathJoinSubstitution([ros_gz_sim_pkg_path, 'launch', 'gz_sim.launch.py'])

    doc = xacro.parse(open(xacro_file))
    xacro.process_doc(doc)
    params = {'robot_description': doc.toxml()}

    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[params]
    )

    node_spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=['-string', doc.toxml(), '-name', 'robot_desmalezador', '-x', '-6.0', '-y', '0.0', '-z', '0.1'],
        output='screen'
    )

    # --- DEFINICIÓN DE LOS 3 NUEVOS NODOS ---
    node_percepcion = Node(
        package='sensado',
        executable='procesamiento_lidar',
        output='screen'
    )

    node_tfmini = Node(
        package='sensado',
        executable='procesamiento_tfmini',
        output='screen'
    )

    node_navegacion = Node(
        package='sensado',
        executable='rutina_navegacion',
        output='screen'
    )

    # Temporizador vital: Lanza los 3 nodos Python a la vez tras 4 segundos
    delayed_nodos_python = TimerAction(
        period=4.0,
        actions=[node_percepcion, node_tfmini, node_navegacion]
    )

    return LaunchDescription([
        SetEnvironmentVariable('GZ_SIM_RESOURCE_PATH', PathJoinSubstitution([sensado_share_path])),
        
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(gz_launch_path),
            launch_arguments={'gz_args': f'-r {world_file}', 'on_exit_shutdown': 'True'}.items(),
        ),
        
        node_robot_state_publisher,
        node_spawn_entity,
        
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            arguments=[
                '/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist',
                '/scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan',
                '/marker@visualization_msgs/msg/Marker@gz.msgs.Marker',
                '/tf_mini/scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan'
            ],
            output='screen'
        ),
        
        delayed_nodos_python
    ])
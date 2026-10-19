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

    # El nodo Python que contiene la rutina de HyperSVD
    node_rutina = Node(
        package='sensado',
        executable='rutina_navegacion',
        output='screen'
    )

    # Temporizador vital para evitar el colapso de Gazebo OpenGL
    delayed_rutina = TimerAction(
        period=3.0,
        actions=[node_rutina]
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
                # --- NUEVA LÍNEA: Puente para enviar los marcadores de ROS a Gazebo ---
                '/marker@visualization_msgs/msg/Marker@gz.msgs.Marker'
            ],
            output='screen'
        ),
        
        # Iniciará 3 segundos DESPUÉS de que Gazebo esté listo
        delayed_rutina
    ])
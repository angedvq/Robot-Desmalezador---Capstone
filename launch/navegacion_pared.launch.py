import os
import pty
import sys
import threading
import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription, SetEnvironmentVariable, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _drain_pty(fd):
    while True:
        try:
            data = os.read(fd, 1024)
            if not data:
                break
        except OSError:
            break


def generate_launch_description():
    pkg_sensado = get_package_share_directory('sensado')
    ros_gz_sim_pkg_path = get_package_share_directory('ros_gz_sim')
    sensado_share_path = FindPackageShare('sensado')

    xacro_file = os.path.join(pkg_sensado, 'urdf', 'robot_sim_pared.urdf.xacro')
    world_file = os.path.join(pkg_sensado, 'worlds', 'sim_pared.world')
    gz_launch_path = PathJoinSubstitution([ros_gz_sim_pkg_path, 'launch', 'gz_sim.launch.py'])

    doc = xacro.parse(open(xacro_file))
    xacro.process_doc(doc)
    params = {'robot_description': doc.toxml()}

    # Si /dev/serial0 no existe (PC de simulación), creamos un puerto serie virtual (PTY)
    # para que stm32_bridge.py se ejecute en simulación sin lanzar excepción de puerto inexistente.
    serial_port = '/dev/serial0'
    if not os.path.exists(serial_port):
        master_fd, slave_fd = pty.openpty()
        serial_port = os.ttyname(slave_fd)
        threading.Thread(target=_drain_pty, args=(master_fd,), daemon=True).start()

    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[params]
    )

    node_spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=['-string', doc.toxml(), '-name', 'robot_sim_pared', '-x', '-2.5', '-y', '0.0', '-z', '0.1'],
        output='screen'
    )

    # Si existe el puerto del RPLidar físico (/dev/ttyUSB0), también puede lanzarse,
    # pero en simulación /scan proviene del puente de Gazebo (ros_gz_bridge).
    node_test_pared = ExecuteProcess(
        cmd=[sys.executable, '-m', 'sensado.test_pared'],
        output='screen'
    )

    node_control_pared = ExecuteProcess(
        cmd=[sys.executable, '-m', 'sensado.control_pared_sim'],
        output='screen'
    )

    node_stm32_bridge = ExecuteProcess(
        cmd=[sys.executable, '-m', 'sensado.stm32_bridge', '--ros-args', '-p', f'port:={serial_port}'],
        output='screen'
    )

    delayed_nodos_python = TimerAction(
        period=4.0,
        actions=[node_test_pared, node_control_pared, node_stm32_bridge]
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
                '/odom@nav_msgs/msg/Odometry@gz.msgs.Odometry',
                '/scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan',
            ],
            output='screen'
        ),

        delayed_nodos_python
    ])

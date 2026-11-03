from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    lidar_port = LaunchConfiguration('lidar_port')
    tfmini_port = LaunchConfiguration('tfmini_port')
    stm32_port = LaunchConfiguration('stm32_port')
    distancia_meta = LaunchConfiguration('distancia_meta')
    distancia_frenado = LaunchConfiguration('distancia_frenado')
    velocidad_avance = LaunchConfiguration('velocidad_avance')
    kp = LaunchConfiguration('kp')

    return LaunchDescription([
        DeclareLaunchArgument(
            'lidar_port',
            default_value='/dev/ttyUSB0',
            description='Puerto USB del RPLidar A1'
        ),
        DeclareLaunchArgument(
            'tfmini_port',
            default_value='/dev/serial0',
            description='Puerto UART del sensor frontal TF-Mini'
        ),
        DeclareLaunchArgument(
            'stm32_port',
            default_value='/dev/ttyAMA1',
            description='Puerto UART hacia la placa STM32'
        ),
        DeclareLaunchArgument(
            'distancia_meta',
            default_value='0.50',
            description='Distancia objetivo a la pared izquierda en metros'
        ),
        DeclareLaunchArgument(
            'distancia_frenado',
            default_value='0.50',
            description='Distancia frontal de detención por obstáculo en metros'
        ),
        DeclareLaunchArgument(
            'velocidad_avance',
            default_value='0.20',
            description='Velocidad lineal de avance en m/s'
        ),
        DeclareLaunchArgument(
            'kp',
            default_value='100.0',
            description='Ganancia proporcional rumbo/distancia (grados por metro)'
        ),

        # 1. Driver oficial RPLidar A1 (orientación por defecto: motor/+X hacia atrás del robot)
        Node(
            package='rplidar_ros',
            executable='rplidar_node',
            name='rplidar_node',
            parameters=[{
                'channel_type': 'serial',
                'serial_port': lidar_port,
                'serial_baudrate': 115200,
                'frame_id': 'laser_link',
                'inverted': False,
                'angle_compensate': True,
                'flip_x_axis': False,
            }],
            output='screen'
        ),

        # 2. Percepción de distancia perpendicular a la pared izquierda
        Node(
            package='sensado',
            executable='real_test_pared',
            name='real_lidar_pared_lateral',
            output='screen'
        ),

        # 3. Driver y detector del sensor frontal TF-Mini
        Node(
            package='sensado',
            executable='driver_tfmini',
            name='tfmini_driver',
            parameters=[{'port': tfmini_port}],
            output='screen'
        ),
        Node(
            package='sensado',
            executable='procesamiento_tfmini',
            name='tf_mini_detector',
            parameters=[{'distancia_frenado': distancia_frenado}],
            output='screen'
        ),

        # 4. Puente UART hacia STM32
        Node(
            package='sensado',
            executable='stm32_bridge',
            name='stm32_bridge',
            parameters=[{'port': stm32_port}],
            output='screen'
        ),

        # 5. Controlador autónomo de seguimiento de pared con watchdogs y frenado
        Node(
            package='sensado',
            executable='real_control_pared',
            name='real_control_pared',
            parameters=[{
                'distancia_meta': distancia_meta,
                'distancia_frenado': distancia_frenado,
                'velocidad_avance': velocidad_avance,
                'kp': kp,
            }],
            output='screen'
        ),
    ])

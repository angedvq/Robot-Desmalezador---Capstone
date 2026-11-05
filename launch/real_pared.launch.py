from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
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
    max_grados = LaunchConfiguration('max_grados')
    max_grados_evasion = LaunchConfiguration('max_grados_evasion')
    usar_tfmini = LaunchConfiguration('usar_tfmini')

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
            default_value='0.70',
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
            default_value='35.0',
            description='Ganancia proporcional suave rumbo/distancia (grados por metro)'
        ),
        DeclareLaunchArgument(
            'max_grados',
            default_value='12.0',
            description='Corrección angular máxima suave en grados'
        ),
        DeclareLaunchArgument(
            'max_grados_evasion',
            default_value='25.0',
            description='Corrección angular máxima de evasión ante pilares/salientes en grados'
        ),
        DeclareLaunchArgument(
            'usar_tfmini',
            default_value='true',
            description='Habilitar sensor frontal TF-Mini (true/false)'
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

        # 3. Driver y detector del sensor frontal TF-Mini (condicional a usar_tfmini)
        Node(
            package='sensado',
            executable='driver_tfmini',
            name='tfmini_driver',
            parameters=[{'port': tfmini_port}],
            condition=IfCondition(usar_tfmini),
            output='screen'
        ),
        Node(
            package='sensado',
            executable='procesamiento_tfmini',
            name='tf_mini_detector',
            parameters=[{'distancia_frenado': distancia_frenado}],
            condition=IfCondition(usar_tfmini),
            output='screen'
        ),

        # 4. Puente UART hacia STM32 (Protocolo V2 con soporte STOP_All / START_Mov / START_Tracking)
        Node(
            package='sensado',
            executable='stm32_bridge',
            name='stm32_bridge',
            parameters=[{'port': stm32_port}],
            output='screen'
        ),

        # 5. Controlador autónomo suave de seguimiento de pared con watchdogs y comando STOP
        Node(
            package='sensado',
            executable='real_control_pared',
            name='real_control_pared',
            parameters=[{
                'distancia_meta': distancia_meta,
                'distancia_frenado': distancia_frenado,
                'velocidad_avance': velocidad_avance,
                'kp': kp,
                'max_grados': max_grados,
                'max_grados_evasion': max_grados_evasion,
                'usar_tfmini': usar_tfmini,
            }],
            output='screen'
        ),
    ])

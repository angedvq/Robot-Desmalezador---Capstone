#!/usr/bin/env python3
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
    meta_arboles = LaunchConfiguration('meta_arboles')
    timeout_sin_arbol = LaunchConfiguration('timeout_sin_arbol')
    tiempo_delay_brazo = LaunchConfiguration('tiempo_delay_brazo')
    tiempo_brazo_extendido = LaunchConfiguration('tiempo_brazo_extendido')
    tiempo_brazo_retraido = LaunchConfiguration('tiempo_brazo_retraido')
    desplazamiento_brazo_mm = LaunchConfiguration('desplazamiento_brazo_mm')
    usar_tfmini = LaunchConfiguration('usar_tfmini')

    return LaunchDescription([
        # ---------------------------------------------------------------------
        # Argumentos configurables por terminal
        # ---------------------------------------------------------------------
        DeclareLaunchArgument(
            'lidar_port',
            default_value='/dev/ttyUSB0',
            description='Puerto USB del RPLidar A1',
        ),
        DeclareLaunchArgument(
            'tfmini_port',
            default_value='/dev/serial0',
            description='Puerto UART del sensor frontal TF-Mini',
        ),
        DeclareLaunchArgument(
            'stm32_port',
            default_value='/dev/ttyAMA1',
            description='Puerto UART hacia la placa STM32',
        ),
        DeclareLaunchArgument(
            'distancia_meta',
            default_value='0.70',
            description='Distancia perpendicular objetivo a la hilera de árboles (m)',
        ),
        DeclareLaunchArgument(
            'distancia_frenado',
            default_value='0.50',
            description='Distancia frontal de detención por obstáculo con TF-Mini (m)',
        ),
        DeclareLaunchArgument(
            'velocidad_avance',
            default_value='0.20',
            description='Velocidad lineal de avance del robot (m/s)',
        ),
        DeclareLaunchArgument(
            'meta_arboles',
            default_value='4',
            description='Cantidad total de árboles a procesar antes de finalizar',
        ),
        DeclareLaunchArgument(
            'timeout_sin_arbol',
            default_value='20.0',
            description='Tiempo máximo sin detectar un árbol antes de detenerse (s)',
        ),
        DeclareLaunchArgument(
            'tiempo_delay_brazo',
            default_value='1.0',
            description='Delay de avance tras detección LiDAR para alinear el brazo trasero (s)',
        ),
        DeclareLaunchArgument(
            'tiempo_brazo_extendido',
            default_value='4.0',
            description='Tiempo que el brazo permanece extendido frente al tronco (s)',
        ),
        DeclareLaunchArgument(
            'tiempo_brazo_retraido',
            default_value='3.0',
            description='Tiempo de espera para retracción del brazo antes de avanzar (s)',
        ),
        DeclareLaunchArgument(
            'desplazamiento_brazo_mm',
            default_value='200.0',
            description='Magnitud enviada a /brazo_mm (+X extiende, -X retrae)',
        ),
        DeclareLaunchArgument(
            'usar_tfmini',
            default_value='true',
            description='Habilitar sensor frontal TF-Mini (true/false)',
        ),

        # ---------------------------------------------------------------------
        # Driver RPLidar A1
        # ---------------------------------------------------------------------
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
            output='screen',
        ),

        # ---------------------------------------------------------------------
        # Detección de troncos 
        # ---------------------------------------------------------------------
        Node(
            package='sensado',
            executable='real_cf_deteccion_troncos',
            name='real_cf_deteccion_troncos',
            output='screen',
        ),

        # ---------------------------------------------------------------------
        # Sensor frontal de obstáculos 
        # ---------------------------------------------------------------------
        Node(
            package='sensado',
            executable='driver_tfmini',
            name='tfmini_driver',
            parameters=[{'port': tfmini_port}],
            condition=IfCondition(usar_tfmini),
            output='screen',
        ),
        Node(
            package='sensado',
            executable='procesamiento_tfmini',
            name='tf_mini_detector',
            parameters=[{'distancia_frenado': distancia_frenado}],
            condition=IfCondition(usar_tfmini),
            output='screen',
        ),

        # ---------------------------------------------------------------------
        # Puente UART hacia STM32 
        # ---------------------------------------------------------------------
        Node(
            package='sensado',
            executable='real_stm32_bridge',
            name='real_stm32_bridge',
            parameters=[{'port': stm32_port}],
            output='screen',
        ),

        # ---------------------------------------------------------------------
        # Control autónomo de rutina de árboles 
        # ---------------------------------------------------------------------
        Node(
            package='sensado',
            executable='real_rutina_navegacion',
            name='real_rutina_navegacion',
            parameters=[{
                'distancia_meta': distancia_meta,
                'distancia_frenado': distancia_frenado,
                'velocidad_avance': velocidad_avance,
                'meta_arboles': meta_arboles,
                'timeout_sin_arbol': timeout_sin_arbol,
                'tiempo_delay_brazo': tiempo_delay_brazo,
                'tiempo_brazo_extendido': tiempo_brazo_extendido,
                'tiempo_brazo_retraido': tiempo_brazo_retraido,
                'desplazamiento_brazo_mm': desplazamiento_brazo_mm,
                'usar_tfmini': usar_tfmini,
            }],
            output='screen',
        ),
    ])

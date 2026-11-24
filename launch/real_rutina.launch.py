#!/usr/bin/env python3
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    lidar_port = LaunchConfiguration('lidar_port')
    tfmini_port = LaunchConfiguration('tfmini_port')
    stm32_port = LaunchConfiguration('stm32_port')
    distancia_meta = LaunchConfiguration('distancia_meta')
    distancia_frenado = LaunchConfiguration('distancia_frenado')
    velocidad_avance = LaunchConfiguration('velocidad_avance')
    velocidad_correccion = LaunchConfiguration('velocidad_correccion')
    kd_rumbo = LaunchConfiguration('kd_rumbo')
    umbral_angulo_lento = LaunchConfiguration('umbral_angulo_lento')
    umbral_angulo_reanudar = LaunchConfiguration('umbral_angulo_reanudar')
    meta_arboles = LaunchConfiguration('meta_arboles')
    timeout_sin_arbol = LaunchConfiguration('timeout_sin_arbol')
    tiempo_barrido_brazo = LaunchConfiguration('tiempo_barrido_brazo')
    tiempo_delay_brazo = LaunchConfiguration('tiempo_delay_brazo')
    tiempo_brazo_extendido = LaunchConfiguration('tiempo_brazo_extendido')
    desplazamiento_brazo_mm = LaunchConfiguration('desplazamiento_brazo_mm')
    usar_tfmini = LaunchConfiguration('usar_tfmini')

    rutina_node = Node(
        package='sensado',
        executable='real_rutina_navegacion',
        name='real_rutina_navegacion',
        parameters=[{
            'distancia_meta': distancia_meta,
            'distancia_frenado': distancia_frenado,
            'velocidad_avance': velocidad_avance,
            'velocidad_correccion': velocidad_correccion,
            'kd_rumbo': kd_rumbo,
            'umbral_angulo_lento': umbral_angulo_lento,
            'umbral_angulo_reanudar': umbral_angulo_reanudar,
            'meta_arboles': meta_arboles,
            'timeout_sin_arbol': timeout_sin_arbol,
            'tiempo_barrido_brazo': tiempo_barrido_brazo,
            'tiempo_delay_brazo': tiempo_delay_brazo,
            'tiempo_brazo_extendido': tiempo_brazo_extendido,
            'desplazamiento_brazo_mm': desplazamiento_brazo_mm,
            'usar_tfmini': usar_tfmini,
        }],
        output='screen',
    )

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
            default_value='0.85',
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
            'velocidad_correccion',
            default_value='0.065',
            description='Velocidad reducida al corregir ángulos de rumbo grandes (m/s)',
        ),
        DeclareLaunchArgument(
            'kd_rumbo',
            default_value='5.0',
            description='Ganancia derivativa del control PID de rumbo (grados por m/s de variación de error)',
        ),
        DeclareLaunchArgument(
            'umbral_angulo_lento',
            default_value='18.0',
            description='Ángulo de corrección (grados) desde el cual el robot baja su velocidad a velocidad_correccion',
        ),
        DeclareLaunchArgument(
            'umbral_angulo_reanudar',
            default_value='6.0',
            description='Ángulo de corrección (grados) bajo el cual el robot vuelve a velocidad_avance',
        ),
        DeclareLaunchArgument(
            'meta_arboles',
            default_value='4',
            description='Cantidad total de árboles a procesar antes de finalizar',
        ),
        DeclareLaunchArgument(
            'timeout_sin_arbol',
            default_value='50.0',
            description='Tiempo máximo sin detectar un árbol antes de detenerse (s)',
        ),
        DeclareLaunchArgument(
            'tiempo_barrido_brazo',
            default_value='2.5',
            description='Tiempo que el robot avanza con el brazo extendido realizando el barrido (s)',
        ),
        DeclareLaunchArgument(
            'tiempo_delay_brazo',
            default_value='1.5',
            description='Parámetro legado de delay (no usado en modo barrido)',
        ),
        DeclareLaunchArgument(
            'tiempo_brazo_extendido',
            default_value='4.0',
            description='Tiempo que el robot se mantiene detenido con el brazo ya extendido (s)',
        ),
        DeclareLaunchArgument(
            'desplazamiento_brazo_mm',
            default_value='600.0',
            description='Magnitud enviada a /brazo_mm (+600 mm extiende, -600 mm retrae)',
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
        # Odometría y generación de mapa geométrico IEEE 1873
        # ---------------------------------------------------------------------
        Node(
            package='sensado',
            executable='odometria',
            name='odometria',
            output='screen',
        ),
        Node(
            package='sensado',
            executable='mapas_ieee1873',
            name='mapas_ieee1873',
            parameters=[{'meta_arboles': meta_arboles}],
            output='screen',
        ),

        # ---------------------------------------------------------------------
        # Control autónomo de rutina de árboles y cierre automático al terminar
        # ---------------------------------------------------------------------
        rutina_node,
        RegisterEventHandler(
            OnProcessExit(
                target_action=rutina_node,
                on_exit=[EmitEvent(event=Shutdown(reason='Rutina de navegacion finalizada'))],
            )
        ),
    ])

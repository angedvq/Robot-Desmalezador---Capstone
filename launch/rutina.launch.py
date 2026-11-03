import os
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # ---------------------------------------------------------
        # 1. Drivers de Sensores (Hardware a ROS 2)
        # ---------------------------------------------------------
        Node(
            package='sensado',
            executable='driver_mapir',
            name='mapir_publisher',
            output='screen'
        ),
        Node(
            package='sensado',
            executable='driver_tfmini',
            name='tfmini_driver',
            output='screen'
        ),


        # ---------------------------------------------------------
        # 2. Nodos de Procesamiento y Percepción
        # ---------------------------------------------------------
        Node(
            package='sensado',
            executable='procesamiento_tfmini',
            name='tf_mini_detector',
            output='screen'
        ),
        Node(
            package='sensado',
            executable='procesamiento_lidar',
            name='percepcion_troncos',
            output='screen'
        ),
        Node(
            package='sensado',
            executable='procesamiento_nir',
            name='procesador_maleza_nir',
            output='screen'
        ),

        # ---------------------------------------------------------
        # 3. Control, Navegación y Puente STM32
        # ---------------------------------------------------------
        Node(
            package='sensado',
            executable='stm32_bridge',
            name='stm32_bridge',
            output='screen'
        ),
        Node(
            package='sensado',
            executable='rutina_navegacion',
            name='control_navegacion',
            output='screen'
        )
    ])
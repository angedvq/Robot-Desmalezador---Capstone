#!/usr/bin/env python3
"""
===============================================================================
ARCHIVO LAUNCH DE SIMULACIÓN: sim_rutina.launch.py
===============================================================================
Lanza la simulación en Gazebo Fortress de la rutina autónoma de detección de
troncos, guiado paralelo a 0.70 m con IMU, delay de 1.0 s para alineación del
brazo trasero y visualización de Markers (Árbol Próximo + Árbol Lateral).

COMPONENTES LANZADOS:
1. Gazebo Fortress (`gz_sim.launch.py`) cargando el mundo existente
   `worlds/cultivo_arboles.world` (4 árboles en y = -1.0 m separados cada 2 m:
   x = -4.0, -2.0, 0.0 y +2.0 m).
2. `robot_state_publisher` + spawn de `urdf/robot_desmalezador.urdf.xacro`:
   - El robot se instancia en `(x = -6.0, y = -1.82, z = 0.10, yaw = 0.0)`.
   - Como el LiDAR está en la esquina superior izquierda (`y_lidar = +0.12 m`
     sobre el chasis), queda en `y = -1.70 m`, dejando la hilera de árboles
     exactamente a **0.70 m a la izquierda del LiDAR**.
3. `ros_gz_bridge`:
   Puentea `/cmd_vel`, `/odom`, `/tf`, `/joint_states`, `/scan`,
   `/tf_mini/scan`, `/imu/data` y `/brazo/cmd_pos`.
4. Nodos de simulación en `sensado/simulaciones/`:
   - `sim_cf_deteccion_troncos.py`: Percepción DBSCAN + HyperSVD + Markers 3D
     en Gazebo Fortress (vía servicio `/marker`) y RViz2 (`/tree_markers`).
   - `procesamiento_tfmini.py`: Detección de obstáculos frontales (< 0.50 m).
   - `sim_stm32_bridge.py`: Puente Protocolo V2 sobre PTY virtual + control de
     la articulación prismática del brazo trasero en Gazebo (`/brazo/cmd_pos`).
   - `sim_rutina_navegacion.py`: Máquina de estados autónoma (4 árboles o 20 s
     de timeout, delay de 1.0 s para alinear el brazo trasero, guiado a 0.70 m).
5. `rviz2` (opcional con `usar_rviz:=true`, activado por defecto):
   Abre una vista cenital con los conos de visión del LiDAR y los Markers del
   **Árbol Próximo (Cian/Amarillo)** y del **Árbol Lateral (Verde)**.
===============================================================================
"""

import os
import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_sensado = get_package_share_directory('sensado')
    ros_gz_sim_pkg_path = get_package_share_directory('ros_gz_sim')
    sensado_share_path = FindPackageShare('sensado')

    xacro_file = os.path.join(pkg_sensado, 'urdf', 'robot_desmalezador.urdf.xacro')
    world_file = os.path.join(pkg_sensado, 'worlds', 'cultivo_arboles.world')
    rviz_config_file = os.path.join(
        pkg_sensado, 'simulaciones', 'sim_markers.rviz'
    )
    gz_launch_path = PathJoinSubstitution(
        [ros_gz_sim_pkg_path, 'launch', 'gz_sim.launch.py']
    )

    doc = xacro.parse(open(xacro_file))
    xacro.process_doc(doc)
    params = {'robot_description': doc.toxml()}

    usar_rviz = LaunchConfiguration('usar_rviz')
    distancia_meta = LaunchConfiguration('distancia_meta')
    velocidad_avance = LaunchConfiguration('velocidad_avance')
    meta_arboles = LaunchConfiguration('meta_arboles')
    timeout_sin_arbol = LaunchConfiguration('timeout_sin_arbol')
    tiempo_delay_brazo = LaunchConfiguration('tiempo_delay_brazo')
    tiempo_brazo_extendido = LaunchConfiguration('tiempo_brazo_extendido')
    tiempo_brazo_retraido = LaunchConfiguration('tiempo_brazo_retraido')

    # Posición inicial de spawn:
    # Los árboles de `cultivo_arboles.world` están escalonados alrededor de y = -1.0 m
    # con separaciones irregulares (x = -4.15, -1.80, -0.05, +2.35 m).
    # Con base_footprint en y = -1.86 m y el LiDAR en la esquina superior izquierda
    # (+0.12 m en Y), el LiDAR arranca en y = -1.74 m, obligando al control de
    # rumbo + IMU a corregir activamente sobre el terreno irregular.
    spawn_x = -6.0
    spawn_y = -1.86
    spawn_yaw = 0.0

    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[params],
    )

    node_spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-string', doc.toxml(),
            '-name', 'robot_desmalezador',
            '-x', str(spawn_x),
            '-y', str(spawn_y),
            '-z', '0.1',
            '-Y', str(spawn_yaw),
        ],
        output='screen',
    )

    node_percepcion = Node(
        package='sensado',
        executable='sim_cf_deteccion_troncos',
        name='sim_cf_deteccion_troncos',
        parameters=[{
            'spawn_x': spawn_x,
            'spawn_y': spawn_y,
            'spawn_yaw': spawn_yaw,
            'usar_gazebo_markers': True,
        }],
        output='screen',
    )

    node_tfmini = Node(
        package='sensado',
        executable='procesamiento_tfmini',
        name='tf_mini_detector',
        output='screen',
    )

    node_stm32_bridge = Node(
        package='sensado',
        executable='sim_stm32_bridge',
        name='sim_stm32_bridge',
        output='screen',
    )

    node_navegacion = Node(
        package='sensado',
        executable='sim_rutina_navegacion',
        name='sim_rutina_navegacion',
        parameters=[{
            'distancia_meta': distancia_meta,
            'velocidad_avance': velocidad_avance,
            'meta_arboles': meta_arboles,
            'timeout_sin_arbol': timeout_sin_arbol,
            'tiempo_delay_brazo': tiempo_delay_brazo,
            'tiempo_brazo_extendido': tiempo_brazo_extendido,
            'tiempo_brazo_retraido': tiempo_brazo_retraido,
        }],
        output='screen',
    )

    node_rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2_markers',
        arguments=['-d', rviz_config_file],
        condition=IfCondition(usar_rviz),
        output='log',
    )

    # Lanzar los nodos Python tras 4.0 s para dar tiempo a que Gazebo cargue el mundo
    delayed_nodos_python = TimerAction(
        period=4.0,
        actions=[
            node_percepcion,
            node_tfmini,
            node_stm32_bridge,
            node_navegacion,
            node_rviz,
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'usar_rviz',
            default_value='true',
            description='Abrir RViz2 automáticamente para visualizar los Markers (true/false)',
        ),
        DeclareLaunchArgument(
            'distancia_meta',
            default_value='0.70',
            description='Distancia perpendicular objetivo a la hilera de árboles (m)',
        ),
        DeclareLaunchArgument(
            'velocidad_avance',
            default_value='0.20',
            description='Velocidad lineal de avance (m/s)',
        ),
        DeclareLaunchArgument(
            'meta_arboles',
            default_value='4',
            description='Cantidad total de árboles a procesar antes de detenerse',
        ),
        DeclareLaunchArgument(
            'timeout_sin_arbol',
            default_value='20.0',
            description='Tiempo máximo sin detectar árboles antes de detenerse (s)',
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
            description='Tiempo de espera para retracción del brazo antes de reanudar (s)',
        ),

        SetEnvironmentVariable(
            'GZ_SIM_RESOURCE_PATH', PathJoinSubstitution([sensado_share_path])
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(gz_launch_path),
            launch_arguments={
                'gz_args': f'-r {world_file}',
                'on_exit_shutdown': 'True',
            }.items(),
        ),

        node_robot_state_publisher,
        node_spawn_entity,

        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            arguments=[
                '/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist',
                '/odom@nav_msgs/msg/Odometry@gz.msgs.Odometry',
                '/tf@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V',
                '/joint_states@sensor_msgs/msg/JointState[gz.msgs.Model',
                '/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
                '/tf_mini/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
                '/imu/data@sensor_msgs/msg/Imu[gz.msgs.IMU',
                '/brazo/cmd_pos@std_msgs/msg/Float64]gz.msgs.Double',
                '/world/cultivo_arboles_world/dynamic_pose/info@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V',
            ],
            output='screen',
        ),

        delayed_nodos_python,
    ])

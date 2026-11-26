#!/usr/bin/env python3
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, RegisterEventHandler
from launch.conditions import IfCondition, UnlessCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Puertos de hardware
    lidar_port = LaunchConfiguration('lidar_port')
    tfmini_port = LaunchConfiguration('tfmini_port')
    stm32_port = LaunchConfiguration('stm32_port')
    mapir_port = LaunchConfiguration('mapir_port')

    # Parámetros de navegación, corrección PID de trayectoria y barrido del brazo
    distancia_meta = LaunchConfiguration('distancia_meta')
    distancia_frenado = LaunchConfiguration('distancia_frenado')
    velocidad_avance = LaunchConfiguration('velocidad_avance')
    velocidad_correccion = LaunchConfiguration('velocidad_correccion')
    kd_rumbo = LaunchConfiguration('kd_rumbo')
    umbral_angulo_lento = LaunchConfiguration('umbral_angulo_lento')
    umbral_angulo_reanudar = LaunchConfiguration('umbral_angulo_reanudar')
    meta_arboles = LaunchConfiguration('meta_arboles')
    timeout_sin_arbol = LaunchConfiguration('timeout_sin_arbol')
    tiempo_verificacion_maleza = LaunchConfiguration('tiempo_verificacion_maleza')
    tiempo_barrido_brazo = LaunchConfiguration('tiempo_barrido_brazo')
    desplazamiento_brazo_mm = LaunchConfiguration('desplazamiento_brazo_mm')
    usar_tfmini = LaunchConfiguration('usar_tfmini')
    usar_mapir = LaunchConfiguration('usar_mapir')

    # Parámetros de la cámara MAPIR (calibración con panel QR + detección directa NIR)
    requiere_calibracion_panel = LaunchConfiguration('requiere_calibracion_panel')
    umbral_nir = LaunchConfiguration('umbral_nir')
    factor_panel_pct = LaunchConfiguration('factor_panel_pct')
    roi_ancho_cm = LaunchConfiguration('roi_ancho_cm')
    roi_alto_cm = LaunchConfiguration('roi_alto_cm')
    min_ancho_cm = LaunchConfiguration('min_ancho_cm')
    min_alto_cm = LaunchConfiguration('min_alto_cm')
    px_por_cm = LaunchConfiguration('px_por_cm')
    area_min_px = LaunchConfiguration('area_min_px')
    bloquear_auto_exposicion = LaunchConfiguration('bloquear_auto_exposicion')
    gui = LaunchConfiguration('gui')
    log_calibracion = LaunchConfiguration('log_calibracion')

    mapir_rutina_node = Node(
        package='sensado',
        executable='mapir_rutina_navegacion',
        name='mapir_rutina_navegacion',
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
            'tiempo_verificacion_maleza': tiempo_verificacion_maleza,
            'tiempo_barrido_brazo': tiempo_barrido_brazo,
            'desplazamiento_brazo_mm': desplazamiento_brazo_mm,
            'usar_tfmini': usar_tfmini,
        }],
        condition=IfCondition(usar_mapir),
        output='screen',
    )

    real_rutina_node = Node(
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
            'desplazamiento_brazo_mm': desplazamiento_brazo_mm,
            'usar_tfmini': usar_tfmini,
        }],
        condition=UnlessCondition(usar_mapir),
        output='screen',
    )

    return LaunchDescription([
        # ---------------------------------------------------------------------
        # Argumentos configurables por terminal (limpios, sin parámetros obsoletos)
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
            'mapir_port',
            default_value='2',
            description='Índice de video V4L2 de la cámara MAPIR (/dev/videoX, con auto-detección)',
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
            description='Velocidad lineal habitual de avance del robot (m/s)',
        ),
        DeclareLaunchArgument(
            'velocidad_correccion',
            default_value='0.065',
            description='Velocidad reducida al corregir ángulos de rumbo grandes (m/s)',
        ),
        DeclareLaunchArgument(
            'kd_rumbo',
            default_value='5.0',
            description='Ganancia derivativa del control PID de rumbo',
        ),
        DeclareLaunchArgument(
            'umbral_angulo_lento',
            default_value='18.0',
            description='Ángulo de corrección (grados) desde el cual el robot baja a velocidad_correccion',
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
            'tiempo_verificacion_maleza',
            default_value='2.0',
            description='Tiempo detenido frente al tronco verificando maleza con la cámara MAPIR (s)',
        ),
        DeclareLaunchArgument(
            'tiempo_barrido_brazo',
            default_value='2.5',
            description='Tiempo que el robot avanza con el brazo extendido realizando el barrido (s)',
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
        DeclareLaunchArgument(
            'usar_mapir',
            default_value='true',
            description='Habilitar cámara MAPIR y rutina con verificación de maleza (true/false)',
        ),
        DeclareLaunchArgument(
            'requiere_calibracion_panel',
            default_value='true',
            description='Esperar calibración con el panel de reflectancia MAPIR antes de iniciar avance',
        ),
        DeclareLaunchArgument(
            'umbral_nir',
            default_value='180',
            description='Umbral NIR inicial/respaldo (se actualiza automáticamente al calibrar con el panel)',
        ),
        DeclareLaunchArgument(
            'factor_panel_pct',
            default_value='60',
            description='Porcentaje de interpolación entre el parche 23%R (suelo) y 51%R (vegetación)',
        ),
        DeclareLaunchArgument(
            'roi_ancho_cm',
            default_value='18.0',
            description='Ancho físico de la Región de Interés (ROI) centrada de la cámara MAPIR (cm)',
        ),
        DeclareLaunchArgument(
            'roi_alto_cm',
            default_value='10.0',
            description='Alto físico de la Región de Interés (ROI) centrada de la cámara MAPIR (cm)',
        ),
        DeclareLaunchArgument(
            'min_ancho_cm',
            default_value='3.0',
            description='Ancho mínimo del bounding box de maleza válida (cm)',
        ),
        DeclareLaunchArgument(
            'min_alto_cm',
            default_value='3.0',
            description='Alto mínimo del bounding box de maleza válida (cm)',
        ),
        DeclareLaunchArgument(
            'px_por_cm',
            default_value='16.0',
            description='Factor de conversión píxeles por centímetro a 1280x720 (~16 px/cm)',
        ),
        DeclareLaunchArgument(
            'area_min_px',
            default_value='250',
            description='Área mínima de contorno (píxeles) para validar maleza',
        ),
        DeclareLaunchArgument(
            'bloquear_auto_exposicion',
            default_value='true',
            description='Congelar auto-exposición y auto-WB de la cámara MAPIR tras calibrar',
        ),
        DeclareLaunchArgument(
            'gui',
            default_value='true',
            description='Mantener abierta la GUI de la MAPIR para calibrar con el panel y monitorear malezas',
        ),
        DeclareLaunchArgument(
            'log_calibracion',
            default_value='true',
            description='Imprimir telemetría de detección MAPIR en consola',
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
        # Cámara MAPIR y detección de maleza NIR con panel de calibración
        # ---------------------------------------------------------------------
        Node(
            package='sensado',
            executable='mapir_driver',
            name='mapir_driver',
            parameters=[{
                'video_port': mapir_port,
                'bloquear_auto_exposicion': bloquear_auto_exposicion,
            }],
            condition=IfCondition(usar_mapir),
            output='screen',
        ),
        Node(
            package='sensado',
            executable='mapir_procesamiento_nir',
            name='mapir_procesamiento_nir',
            parameters=[{
                'requiere_calibracion_panel': requiere_calibracion_panel,
                'umbral_nir': umbral_nir,
                'factor_panel_pct': factor_panel_pct,
                'roi_ancho_cm': roi_ancho_cm,
                'roi_alto_cm': roi_alto_cm,
                'min_ancho_cm': min_ancho_cm,
                'min_alto_cm': min_alto_cm,
                'px_por_cm': px_por_cm,
                'area_min_px': area_min_px,
                'gui': gui,
                'log_calibracion': log_calibracion,
            }],
            condition=IfCondition(usar_mapir),
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
        # Control autónomo (CON o SIN cámara MAPIR) y cierre limpio al finalizar
        # ---------------------------------------------------------------------
        mapir_rutina_node,
        real_rutina_node,
        RegisterEventHandler(
            OnProcessExit(
                target_action=mapir_rutina_node,
                on_exit=[EmitEvent(event=Shutdown(reason='Rutina de navegacion MAPIR finalizada'))],
            )
        ),
        RegisterEventHandler(
            OnProcessExit(
                target_action=real_rutina_node,
                on_exit=[EmitEvent(event=Shutdown(reason='Rutina de navegacion finalizada'))],
            )
        ),
    ])

#!/usr/bin/env python3
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    lidar_port = LaunchConfiguration('lidar_port')
    tfmini_port = LaunchConfiguration('tfmini_port')
    stm32_port = LaunchConfiguration('stm32_port')
    mapir_port = LaunchConfiguration('mapir_port')
    distancia_meta = LaunchConfiguration('distancia_meta')
    distancia_frenado = LaunchConfiguration('distancia_frenado')
    velocidad_avance = LaunchConfiguration('velocidad_avance')
    meta_arboles = LaunchConfiguration('meta_arboles')
    timeout_sin_arbol = LaunchConfiguration('timeout_sin_arbol')
    tiempo_delay_brazo = LaunchConfiguration('tiempo_delay_brazo')
    tiempo_verificacion_maleza = LaunchConfiguration('tiempo_verificacion_maleza')
    tiempo_brazo_extendido = LaunchConfiguration('tiempo_brazo_extendido')
    desplazamiento_brazo_mm = LaunchConfiguration('desplazamiento_brazo_mm')
    usar_tfmini = LaunchConfiguration('usar_tfmini')
    usar_mapir = LaunchConfiguration('usar_mapir')

    # Parámetros de calibración / umbralización de la cámara MAPIR
    modo_umbral = LaunchConfiguration('modo_umbral')
    umbral_nir_min = LaunchConfiguration('umbral_nir_min')
    umbral_nir_fijo = LaunchConfiguration('umbral_nir_fijo')
    delta_contraste_fondo = LaunchConfiguration('delta_contraste_fondo')
    roi_ancho_cm = LaunchConfiguration('roi_ancho_cm')
    roi_alto_cm = LaunchConfiguration('roi_alto_cm')
    min_ancho_cm = LaunchConfiguration('min_ancho_cm')
    min_alto_cm = LaunchConfiguration('min_alto_cm')
    px_por_cm = LaunchConfiguration('px_por_cm')
    area_min_px = LaunchConfiguration('area_min_px')
    mostrar_ventana = LaunchConfiguration('mostrar_ventana')
    gui = LaunchConfiguration('gui')
    log_calibracion = LaunchConfiguration('log_calibracion')

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
            'mapir_port',
            default_value='2',
            description='Índice de video V4L2 de la cámara MAPIR (/dev/videoX, con auto-detección)',
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
            default_value='50.0',
            description='Tiempo máximo sin detectar un árbol antes de detenerse (s)',
        ),
        DeclareLaunchArgument(
            'tiempo_delay_brazo',
            default_value='1.5',
            description='Delay de avance tras detección LiDAR para alinear el brazo trasero y MAPIR (s)',
        ),
        DeclareLaunchArgument(
            'tiempo_verificacion_maleza',
            default_value='2.0',
            description='Tiempo detenido frente al tronco verificando maleza con la cámara MAPIR (s)',
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
        DeclareLaunchArgument(
            'usar_mapir',
            default_value='true',
            description='Habilitar cámara MAPIR y rutina con verificación de maleza (true/false)',
        ),
        DeclareLaunchArgument(
            'modo_umbral',
            default_value='hibrido',
            description='Modo de umbralización MAPIR: hibrido (fondo + delta) o fijo',
        ),
        DeclareLaunchArgument(
            'umbral_nir_min',
            default_value='150',
            description='Intensidad NIR mínima absoluta para considerar maleza (0..255)',
        ),
        DeclareLaunchArgument(
            'umbral_nir_fijo',
            default_value='180',
            description='Umbral NIR estático cuando modo_umbral:=fijo (0..255)',
        ),
        DeclareLaunchArgument(
            'delta_contraste_fondo',
            default_value='35',
            description='Diferencia mínima de intensidad NIR sobre la mediana del suelo en modo híbrido',
        ),
        DeclareLaunchArgument(
            'roi_ancho_cm',
            default_value='15.0',
            description='Ancho físico de la Región de Interés (ROI) centrada de la cámara MAPIR (cm)',
        ),
        DeclareLaunchArgument(
            'roi_alto_cm',
            default_value='15.0',
            description='Alto físico de la Región de Interés (ROI) centrada de la cámara MAPIR (cm)',
        ),
        DeclareLaunchArgument(
            'min_ancho_cm',
            default_value='3.0',
            description='Ancho mínimo del bounding box de maleza para ser considerada válida (cm)',
        ),
        DeclareLaunchArgument(
            'min_alto_cm',
            default_value='3.0',
            description='Alto mínimo del bounding box de maleza para ser considerada válida (cm)',
        ),
        DeclareLaunchArgument(
            'px_por_cm',
            default_value='16.0',
            description='Factor de conversión píxeles por centímetro a 1280x720 (~16 px/cm -> 15 cm = 240 px, 3 cm = 48 px)',
        ),
        DeclareLaunchArgument(
            'area_min_px',
            default_value='250',
            description='Área mínima de contorno (píxeles) para validar maleza',
        ),
        DeclareLaunchArgument(
            'mostrar_ventana',
            default_value='false',
            description='Mostrar ventana OpenCV con trackbars de calibración (requiere monitor/DISPLAY)',
        ),
        DeclareLaunchArgument(
            'gui',
            default_value='false',
            description='Alias de mostrar_ventana para abrir la GUI de detección MAPIR con bounding boxes',
        ),
        DeclareLaunchArgument(
            'log_calibracion',
            default_value='true',
            description='Imprimir telemetría de calibración NIR (Fondo, P95, Umbral, Área) en consola',
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
        # Cámara MAPIR y detección calibrada de maleza NIR (sensado/integracion_mapir)
        # ---------------------------------------------------------------------
        Node(
            package='sensado',
            executable='mapir_driver',
            name='mapir_driver',
            parameters=[{'video_port': mapir_port}],
            condition=IfCondition(usar_mapir),
            output='screen',
        ),
        Node(
            package='sensado',
            executable='mapir_procesamiento_nir',
            name='mapir_procesamiento_nir',
            parameters=[{
                'modo_umbral': modo_umbral,
                'umbral_nir_min': umbral_nir_min,
                'umbral_nir_fijo': umbral_nir_fijo,
                'delta_contraste_fondo': delta_contraste_fondo,
                'roi_ancho_cm': roi_ancho_cm,
                'roi_alto_cm': roi_alto_cm,
                'min_ancho_cm': min_ancho_cm,
                'min_alto_cm': min_alto_cm,
                'px_por_cm': px_por_cm,
                'area_min_px': area_min_px,
                'mostrar_ventana': mostrar_ventana,
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
        # Control autónomo CON cámara MAPIR (por defecto: usar_mapir:=true)
        # ---------------------------------------------------------------------
        Node(
            package='sensado',
            executable='mapir_rutina_navegacion',
            name='mapir_rutina_navegacion',
            parameters=[{
                'distancia_meta': distancia_meta,
                'distancia_frenado': distancia_frenado,
                'velocidad_avance': velocidad_avance,
                'meta_arboles': meta_arboles,
                'timeout_sin_arbol': timeout_sin_arbol,
                'tiempo_delay_brazo': tiempo_delay_brazo,
                'tiempo_verificacion_maleza': tiempo_verificacion_maleza,
                'tiempo_brazo_extendido': tiempo_brazo_extendido,
                'desplazamiento_brazo_mm': desplazamiento_brazo_mm,
                'usar_tfmini': usar_tfmini,
            }],
            condition=IfCondition(usar_mapir),
            output='screen',
        ),

        # ---------------------------------------------------------------------
        # Control autónomo SIN cámara MAPIR (respaldo intacto: usar_mapir:=false)
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
                'desplazamiento_brazo_mm': desplazamiento_brazo_mm,
                'usar_tfmini': usar_tfmini,
            }],
            condition=UnlessCondition(usar_mapir),
            output='screen',
        ),
    ])

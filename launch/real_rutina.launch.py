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
    mapir_port = LaunchConfiguration('mapir_port')
    distancia_meta = LaunchConfiguration('distancia_meta')
    distancia_frenado = LaunchConfiguration('distancia_frenado')
    velocidad_avance = LaunchConfiguration('velocidad_avance')
    velocidad_correccion = LaunchConfiguration('velocidad_correccion')
    kd_rumbo = LaunchConfiguration('kd_rumbo')
    umbral_angulo_lento = LaunchConfiguration('umbral_angulo_lento')
    umbral_angulo_reanudar = LaunchConfiguration('umbral_angulo_reanudar')
    meta_arboles = LaunchConfiguration('meta_arboles')
    timeout_sin_arbol = LaunchConfiguration('timeout_sin_arbol')
    tiempo_calibracion_bomba = LaunchConfiguration('tiempo_calibracion_bomba')
    tiempo_barrido_brazo = LaunchConfiguration('tiempo_barrido_brazo')
    tiempo_delay_brazo = LaunchConfiguration('tiempo_delay_brazo')
    tiempo_verificacion_maleza = LaunchConfiguration('tiempo_verificacion_maleza')
    tiempo_brazo_extendido = LaunchConfiguration('tiempo_brazo_extendido')
    desplazamiento_brazo_mm = LaunchConfiguration('desplazamiento_brazo_mm')
    usar_tfmini = LaunchConfiguration('usar_tfmini')
    usar_mapir = LaunchConfiguration('usar_mapir')
    usar_bomba = LaunchConfiguration('usar_bomba')

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

    # Nodo unificado que gestiona los 4 casos posibles de (usar_mapir x usar_bomba):
    # - usar_mapir:=true  usar_bomba:=true  -> Calibra bomba 5s + inspecciona MAPIR + barrido con bomba si hay maleza
    # - usar_mapir:=true  usar_bomba:=false -> Sin calibración bomba + inspecciona MAPIR + barrido sin bomba si hay maleza
    # - usar_mapir:=false usar_bomba:=true  -> Calibra bomba 5s + sin MAPIR + barrido con bomba en cada árbol
    # - usar_mapir:=false usar_bomba:=false -> Sin calibración bomba + sin MAPIR + barrido sin bomba en cada árbol
    bomba_rutina_node = Node(
        package='sensado',
        executable='bomba_rutina_navegacion',
        name='bomba_rutina_navegacion',
        parameters=[{
            'usar_mapir': usar_mapir,
            'usar_bomba': usar_bomba,
            'distancia_meta': distancia_meta,
            'distancia_frenado': distancia_frenado,
            'velocidad_avance': velocidad_avance,
            'velocidad_correccion': velocidad_correccion,
            'kd_rumbo': kd_rumbo,
            'umbral_angulo_lento': umbral_angulo_lento,
            'umbral_angulo_reanudar': umbral_angulo_reanudar,
            'meta_arboles': meta_arboles,
            'timeout_sin_arbol': timeout_sin_arbol,
            'tiempo_calibracion_bomba': tiempo_calibracion_bomba,
            'tiempo_barrido_brazo': tiempo_barrido_brazo,
            'tiempo_delay_brazo': tiempo_delay_brazo,
            'tiempo_verificacion_maleza': tiempo_verificacion_maleza,
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
            'tiempo_calibracion_bomba',
            default_value='5.0',
            description='Tiempo inicial con brazo extendido y bomba activa para llenar la manguera al encender (s)',
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
            description='Habilitar cámara MAPIR y verificación de maleza antes de barrer (true/false)',
        ),
        DeclareLaunchArgument(
            'usar_bomba',
            default_value='true',
            description='Habilitar calibración inicial de 5s y activación de la bomba durante el barrido (true/false)',
        ),
        DeclareLaunchArgument(
            'modo_umbral',
            default_value='hibrido',
            description='Modo de umbralización MAPIR: hibrido (fondo + delta) o fijo',
        ),
        DeclareLaunchArgument(
            'umbral_nir_min',
            default_value='25',
            description='Intensidad NIR mínima absoluta para considerar maleza en baja luz/pasillo (0..255)',
        ),
        DeclareLaunchArgument(
            'umbral_nir_fijo',
            default_value='180',
            description='Umbral NIR estático cuando modo_umbral:=fijo (0..255)',
        ),
        DeclareLaunchArgument(
            'delta_contraste_fondo',
            default_value='14',
            description='Diferencia base de intensidad sobre el suelo en modo híbrido adaptativo',
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
        # Cámara MAPIR y detección calibrada de maleza NIR (sensado/integracion_mapir_bomba)
        # Se lanzan únicamente cuando usar_mapir:=true
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
        # Control autónomo de rutina (cubre los 4 casos de usar_mapir x usar_bomba)
        # ---------------------------------------------------------------------
        bomba_rutina_node,
        RegisterEventHandler(
            OnProcessExit(
                target_action=bomba_rutina_node,
                on_exit=[EmitEvent(event=Shutdown(reason='Rutina de navegacion finalizada'))],
            )
        ),
    ])

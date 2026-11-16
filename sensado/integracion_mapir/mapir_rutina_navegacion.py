#!/usr/bin/env python3
"""
Nodo de control autónomo de navegación por hilera de árboles con inspección de maleza
mediante cámara MAPIR (`mapir_rutina_navegacion.py`).

Copia adaptada de `sensado/real_rutina_navegacion.py` ubicada en `sensado/integracion_mapir/`
para mantener intacta la rutina original sin cámara.

Flujo de la máquina de estados:
1. `AVANZANDO`:
   Avanza en línea recta buscando troncos con el RPLidar A1 (`/arboles_detectados`) y
   vigilando obstáculos frontales con el TF-Mini.
2. `ALINEANDO_BRAZO`:
   Al detectar un tronco, sigue avanzando durante `tiempo_delay_brazo` (1.5 s) para que la
   parte trasera del robot (cámara MAPIR y brazo) quede alineada frente al tronco, y se detiene.
3. `VERIFICANDO_MALEZA`:
   Con el robot detenido frente al tronco, consulta durante `tiempo_verificacion_maleza` (2.0 s)
   el tópico `/rutina/maleza_detectada` publicado por `mapir_procesamiento_nir`:
   - Si y solo si hay maleza: pasa a `EXTENDIENDO_BRAZO` (+600 mm, espera código 4 de STM32) ->
     `MANTENIENDO_BRAZO` (4.0 s detenido con brazo extendido) -> `RETRAYENDO_BRAZO` (-600 mm,
     espera código 3 de STM32) -> `ENFRIAMIENTO` (o `FINALIZADO`).
   - Si NO hay maleza: NO acciona el brazo y pasa directamente a `ENFRIAMIENTO` (o `FINALIZADO`
     si ya alcanzó `meta_arboles`) para continuar hacia el siguiente tronco.
"""

import math
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseArray
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Float32, Int32

STM32_BRAZO_RETRAIDO = 3   # Código enviado por la STM32 cuando el brazo terminó de retraerse
STM32_BRAZO_EXTENDIDO = 4  # Código enviado por la STM32 cuando el brazo terminó de extenderse


class MapirRutinaNavegacion(Node):
    def __init__(self):
        super().__init__('mapir_rutina_navegacion')

        # ---------------------------------------------------------------------
        # Declaración de parámetros
        # ---------------------------------------------------------------------
        # Parámetros geométricos y de avance
        self.declare_parameter('distancia_meta', 0.70)        # Distancia a la hilera (m)
        self.declare_parameter('distancia_frenado', 0.50)     # Umbral obstáculo TF-Mini (m)
        self.declare_parameter('velocidad_avance', 0.20)      # Velocidad lineal (m/s)

        # Parámetros de corrección suave de rumbo (LiDAR + IMU)
        self.declare_parameter('kp_rumbo', 30.0)
        self.declare_parameter('max_grados_rumbo', 10.0)
        self.declare_parameter('slew_rate_deg_s', 6.0)
        self.declare_parameter('alpha_dist', 0.40)

        # Parámetros de la rutina de árboles, inspección MAPIR y brazo
        self.declare_parameter('meta_arboles', 4)
        self.declare_parameter('timeout_sin_arbol', 50.0)
        self.declare_parameter('tiempo_delay_brazo', 1.5)          # Delay tras detección LiDAR para alinear brazo/MAPIR (s)
        self.declare_parameter('tiempo_verificacion_maleza', 2.0)  # Ventana de inspección MAPIR estando detenido (s)
        self.declare_parameter('tiempo_estabilizacion_camara', 0.3) # Espera tras frenar antes de muestrear MAPIR (s)
        self.declare_parameter('tiempo_brazo_extendido', 4.0)      # Tiempo detenido con el brazo ya extendido (s)
        self.declare_parameter('tiempo_enfriamiento', 3.0)         # Ceguera temporal para avanzar al siguiente tronco (s)
        self.declare_parameter('desplazamiento_brazo_mm', 600.0)   # +600 mm (+60 cm) extiende, -600 mm retrae

        # Watchdogs de sensores reales
        self.declare_parameter('usar_tfmini', True)
        self.declare_parameter('lidar_timeout', 1.5)
        self.declare_parameter('tfmini_timeout', 1.5)

        # Lectura de parámetros
        self.distancia_meta = float(self.get_parameter('distancia_meta').value)
        self.distancia_frenado = float(self.get_parameter('distancia_frenado').value)
        self.velocidad_avance = float(self.get_parameter('velocidad_avance').value)
        self.kp_rumbo = float(self.get_parameter('kp_rumbo').value)
        self.max_grados_rumbo = float(self.get_parameter('max_grados_rumbo').value)
        self.slew_rate_deg_s = float(self.get_parameter('slew_rate_deg_s').value)
        self.alpha_dist = float(self.get_parameter('alpha_dist').value)

        self.meta_arboles = int(self.get_parameter('meta_arboles').value)
        self.timeout_sin_arbol = float(self.get_parameter('timeout_sin_arbol').value)
        self.tiempo_delay_brazo = float(self.get_parameter('tiempo_delay_brazo').value)
        self.tiempo_verificacion_maleza = float(self.get_parameter('tiempo_verificacion_maleza').value)
        self.tiempo_estabilizacion_camara = float(self.get_parameter('tiempo_estabilizacion_camara').value)
        self.tiempo_brazo_extendido = float(self.get_parameter('tiempo_brazo_extendido').value)
        self.tiempo_enfriamiento = float(self.get_parameter('tiempo_enfriamiento').value)
        self.desplazamiento_brazo_mm = abs(float(self.get_parameter('desplazamiento_brazo_mm').value))

        self.usar_tfmini = bool(self.get_parameter('usar_tfmini').value)
        self.lidar_timeout = float(self.get_parameter('lidar_timeout').value)
        self.tfmini_timeout = float(self.get_parameter('tfmini_timeout').value)

        # ---------------------------------------------------------------------
        # Publicadores hacia STM32
        # ---------------------------------------------------------------------
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.stop_pub = self.create_publisher(Bool, '/navegacion/stop', 10)
        self.grados_pub = self.create_publisher(Float32, '/navegacion/corregir_grados', 10)
        self.brazo_pub = self.create_publisher(Float32, '/brazo_mm', 10)

        # ---------------------------------------------------------------------
        # Suscriptores de percepción (LiDAR, TF-Mini, MAPIR) y STM32
        # ---------------------------------------------------------------------
        self.tree_sub = self.create_subscription(PoseArray, '/arboles_detectados', self.tree_callback, 10)
        self.dist_sub = self.create_subscription(
            Float32, '/lidar/distancia_tronco_lateral', self.dist_tronco_callback, 10
        )
        self.obs_sub = self.create_subscription(Bool, '/obstaculo_frontal', self.obs_callback, 10)
        self.tfmini_sub = self.create_subscription(LaserScan, '/tf_mini/scan', self.tfmini_callback, 10)
        self.estado_brazo_sub = self.create_subscription(
            Int32, '/stm32/estado_brazo', self.estado_brazo_callback, 10
        )
        self.maleza_sub = self.create_subscription(
            Bool, '/rutina/maleza_detectada', self.maleza_callback, 10
        )
        self.cobertura_sub = self.create_subscription(
            Float32, '/mapir/porcentaje_maleza', self.cobertura_callback, 10
        )

        # ---------------------------------------------------------------------
        # Variables de estado y temporizadores
        # ---------------------------------------------------------------------
        self.estado_actual = 'AVANZANDO'
        self.arboles_procesados = 0
        self.arboles_con_maleza = 0
        self.arbol_en_mira = False

        # Estado de maleza (MAPIR)
        self.maleza_actual_topic = False
        self.porcentaje_maleza_actual = 0.0
        self.ultimo_mapir_ts = None
        self.maleza_detectada_en_parada = False
        self.max_cobertura_en_parada = 0.0

        # Estado del brazo (STM32)
        self.brazo_extendido = False
        self.orden_extender_enviada = False
        self.orden_retraer_enviada = False
        self.stm32_brazo_extendido = False
        self.stm32_brazo_retraido = False
        self.brazo_retraido_confirmado = True

        # Variables de guiado lateral y rumbo IMU
        self.distancia_tronco_actual = None
        self.ultimo_tronco_guia_ts = None
        self.rumbo_suavizado = 0.0
        self.ultimo_rumbo_enviado = 0.0
        self.ultimo_envio_grados_ts = 0.0

        # Watchdogs y estado de obstáculo frontal
        self.ultimo_lidar_ts = None
        self.ultimo_tfmini_ts = None
        self.obstaculo_topic = False
        self.obstaculo_scan = False
        self.en_pausa_por_watchdog = True
        self.en_pausa_por_obstaculo = False

        # Timestamps
        now = time.monotonic()
        self.tiempo_inicio_estado = now
        self.tiempo_ultimo_arbol = now
        self.tiempo_transcurrido_guardado = 0.0
        self.tiempo_sin_ver_arbol_guardado = 0.0

        # Bucle principal a 20 Hz
        self.dt = 0.05
        self.control_timer = self.create_timer(self.dt, self.control_loop)

        self.get_logger().info(
            f"mapir_rutina_navegacion iniciado | Meta: {self.meta_arboles} árboles | "
            f"Delay alineación: {self.tiempo_delay_brazo:.1f}s | "
            f"Inspección MAPIR: {self.tiempo_verificacion_maleza:.1f}s | "
            f"Espera brazo extendido: {self.tiempo_brazo_extendido:.1f}s | "
            f"Recorrido brazo: {self.desplazamiento_brazo_mm:.0f}mm"
        )

    # =========================================================================
    # Callbacks
    # =========================================================================
    def estado_brazo_callback(self, msg: Int32):
        """
        Recibe los avisos de fin de movimiento del brazo desde la STM32:
        - 4 (STM32_BRAZO_EXTENDIDO): El stepper terminó de extenderse por completo.
        - 3 (STM32_BRAZO_RETRAIDO):  El stepper terminó de retraerse por completo.
        """
        codigo = int(msg.data)
        if codigo == STM32_BRAZO_EXTENDIDO:
            self.stm32_brazo_extendido = True
            self.get_logger().info(
                "[STM32 -> Navegación] Código 4 recibido: Brazo totalmente EXTENDIDO."
            )
        elif codigo == STM32_BRAZO_RETRAIDO:
            self.stm32_brazo_retraido = True
            self.brazo_extendido = False
            self.brazo_retraido_confirmado = True
            self.get_logger().info(
                "[STM32 -> Navegación] Código 3 recibido: Brazo totalmente RETRAÍDO."
            )

    def maleza_callback(self, msg: Bool):
        """
        Recibe la confirmación de presencia de maleza desde `mapir_procesamiento_nir`
        (/rutina/maleza_detectada).
        """
        now = time.monotonic()
        self.ultimo_mapir_ts = now
        self.maleza_actual_topic = bool(msg.data)

        # Si estamos detenidos verificando maleza frente al tronco y ya pasó el breve
        # tiempo de estabilización tras frenar, registrar detección positiva
        if self.estado_actual == 'VERIFICANDO_MALEZA':
            if (now - self.tiempo_inicio_estado) >= self.tiempo_estabilizacion_camara:
                if self.maleza_actual_topic:
                    self.maleza_detectada_en_parada = True

    def cobertura_callback(self, msg: Float32):
        """Recibe el porcentaje de área de maleza en la ROI (/mapir/porcentaje_maleza)."""
        if math.isfinite(msg.data):
            self.porcentaje_maleza_actual = float(msg.data)
            if self.estado_actual == 'VERIFICANDO_MALEZA':
                self.max_cobertura_en_parada = max(
                    self.max_cobertura_en_parada, self.porcentaje_maleza_actual
                )

    def tree_callback(self, msg: PoseArray):
        """Recibe las detecciones de troncos en la ventana lateral."""
        self.ultimo_lidar_ts = time.monotonic()
        self.arbol_en_mira = len(msg.poses) > 0

    def dist_tronco_callback(self, msg: Float32):
        """Recibe la distancia perpendicular al tronco más cercano."""
        now = time.monotonic()
        self.ultimo_lidar_ts = now

        if math.isfinite(msg.data) and msg.data > 0.0:
            nueva = float(msg.data)
            if self.distancia_tronco_actual is None:
                self.distancia_tronco_actual = nueva
            else:
                self.distancia_tronco_actual = (
                    self.alpha_dist * nueva + (1.0 - self.alpha_dist) * self.distancia_tronco_actual
                )
            self.ultimo_tronco_guia_ts = now
        else:
            self.distancia_tronco_actual = None

    def obs_callback(self, msg: Bool):
        """Recibe flag de obstáculo frontal desde procesamiento_tfmini."""
        self.obstaculo_topic = bool(msg.data)

    def tfmini_callback(self, msg: LaserScan):
        """Monitorea /tf_mini/scan para actualizar el watchdog y detectar obstáculos."""
        self.ultimo_tfmini_ts = time.monotonic()
        detectado = False
        for r in msg.ranges:
            if math.isnan(r) or (math.isinf(r) and r > 0.0):
                continue
            if r < self.distancia_frenado:
                detectado = True
                break
        self.obstaculo_scan = detectado

    # =========================================================================
    # Funciones auxiliares de actuación y seguridad
    # =========================================================================
    def comandar_brazo(self, accion: str):
        """
        Envía el comando de brazo hacia la STM32 a través de /brazo_mm:
        - 'EXTENDER': envía +desplazamiento_brazo_mm
        - 'RETRAER':  envía -desplazamiento_brazo_mm
        """
        msg_brazo = Float32()
        if accion == 'EXTENDER':
            msg_brazo.data = +self.desplazamiento_brazo_mm
            self.brazo_extendido = True
            self.get_logger().info(
                f"[BRAZO] Extendiendo brazo (+{self.desplazamiento_brazo_mm / 10.0:.1f} cm)..."
            )
        elif accion == 'RETRAER':
            msg_brazo.data = -self.desplazamiento_brazo_mm
            self.brazo_extendido = False
            self.get_logger().info(
                f"[BRAZO] Retrayendo brazo (-{self.desplazamiento_brazo_mm / 10.0:.1f} cm)..."
            )
        else:
            msg_brazo.data = 0.0
            self.brazo_extendido = False

        self.brazo_pub.publish(msg_brazo)

    def detener_todo_emergencia(self):
        """Envía STOP_All y velocidad 0.0 m/s."""
        msg_stop = Bool()
        msg_stop.data = True
        self.stop_pub.publish(msg_stop)

        msg_vel = Twist()
        msg_vel.linear.x = 0.0
        msg_vel.angular.z = 0.0
        self.cmd_pub.publish(msg_vel)

        self.rumbo_suavizado = 0.0

    def _pausar_relojes(self, now: float):
        """Congela los tiempos transcurridos durante una pausa por obstáculo o watchdog."""
        self.tiempo_transcurrido_guardado = now - self.tiempo_inicio_estado
        self.tiempo_sin_ver_arbol_guardado = now - self.tiempo_ultimo_arbol

    def _restaurar_relojes(self, now: float):
        """Restaura los tiempos transcurridos al reanudarse la operación."""
        self.tiempo_inicio_estado = now - self.tiempo_transcurrido_guardado
        self.tiempo_ultimo_arbol = now - self.tiempo_sin_ver_arbol_guardado

    def _finalizar_o_continuar_siguiente_arbol(self, now: float, motivo_log: str):
        """
        Decide si finalizar la rutina (cuando ya se alcanzó `meta_arboles`) o
        pasar a `ENFRIAMIENTO` para avanzar hacia el siguiente árbol.
        """
        if self.arboles_procesados >= self.meta_arboles:
            self.estado_actual = 'FINALIZADO'
            self.detener_todo_emergencia()
            self.get_logger().info(
                f"Meta alcanzada ({self.arboles_procesados}/{self.meta_arboles} árboles inspeccionados, "
                f"{self.arboles_con_maleza} con maleza). {motivo_log} Robot detenido por completo."
            )
            return True
        else:
            self.estado_actual = 'ENFRIAMIENTO'
            self.tiempo_inicio_estado = now
            self.tiempo_ultimo_arbol = now
            self.get_logger().info(
                f"Árbol {self.arboles_procesados}/{self.meta_arboles}: {motivo_log} "
                f"Reanudando marcha (enfriamiento de {self.tiempo_enfriamiento:.1f}s)..."
            )
            return False

    # =========================================================================
    # Bucle Principal de Control
    # =========================================================================
    def control_loop(self):
        now = time.monotonic()

        # Si la rutina ya terminó, mantener todo apagado con STOP_All
        if self.estado_actual == 'FINALIZADO':
            self.detener_todo_emergencia()
            return

        # Verificación de Watchdogs de Sensores
        detenido_en_arbol = self.estado_actual in (
            'VERIFICANDO_MALEZA',
            'EXTENDIENDO_BRAZO',
            'MANTENIENDO_BRAZO',
            'RETRAYENDO_BRAZO',
        )
        lidar_ok = (self.ultimo_lidar_ts is not None and (now - self.ultimo_lidar_ts) <= self.lidar_timeout)
        tfmini_ok = (
            (not self.usar_tfmini)
            or (self.ultimo_tfmini_ts is not None and (now - self.ultimo_tfmini_ts) <= self.tfmini_timeout)
        )

        # Si algún sensor falla mientras el robot se desplaza, freno de emergencia.
        if (not lidar_ok or not tfmini_ok) and not detenido_en_arbol:
            if not self.en_pausa_por_watchdog:
                self._pausar_relojes(now)
                self.en_pausa_por_watchdog = True
            self.detener_todo_emergencia()

            faltantes = []
            if not lidar_ok:
                faltantes.append('LiDAR (/scan)')
            if not tfmini_ok:
                faltantes.append('TF-Mini (/tf_mini/scan)')
            self.get_logger().warn(
                f"Watchdog sensores: esperando datos de {', '.join(faltantes)}. STOP_All activo.",
                throttle_duration_sec=2.0,
            )
            return

        elif self.en_pausa_por_watchdog and lidar_ok and tfmini_ok:
            self._restaurar_relojes(now)
            self.en_pausa_por_watchdog = False
            self.get_logger().info("Sensores ok. Iniciando/reanudando rutina con integración MAPIR")

        # ---------------------------------------------------------------------
        # Obstáculo Frontal (solo detiene cuando el robot pretende avanzar)
        # ---------------------------------------------------------------------
        hay_obstaculo = self.usar_tfmini and (self.obstaculo_topic or self.obstaculo_scan)
        if hay_obstaculo and not detenido_en_arbol:
            if not self.en_pausa_por_obstaculo:
                self._pausar_relojes(now)
                self.en_pausa_por_obstaculo = True
            self.detener_todo_emergencia()
            self.get_logger().warn(
                f"Obstáculo frontal (< {self.distancia_frenado:.2f}m). "
                "Esperando despeje de la ruta...",
                throttle_duration_sec=2.0,
            )
            return

        elif self.en_pausa_por_obstaculo and not hay_obstaculo:
            self._restaurar_relojes(now)
            self.en_pausa_por_obstaculo = False
            self.get_logger().info("Ruta despejada. Reanudando...")

        # Operación normal: desactivar bandera de emergencia STOP_All
        msg_stop = Bool()
        msg_stop.data = False
        self.stop_pub.publish(msg_stop)

        tiempo_transcurrido = now - self.tiempo_inicio_estado
        tiempo_sin_ver_arbol = now - self.tiempo_ultimo_arbol

        msg_vel = Twist()

        # ---------------------------------------------------------------------
        # Máquina de Estados: Navegación + Inspección MAPIR + Brazo
        # ---------------------------------------------------------------------
        if self.estado_actual == 'AVANZANDO':
            # Garantía de seguridad: solo avanzar si el brazo está retraído
            if not self.brazo_retraido_confirmado or self.brazo_extendido:
                msg_vel.linear.x = 0.0
                self.cmd_pub.publish(msg_vel)
                return

            msg_vel.linear.x = self.velocidad_avance

            # Criterio de parada por timeout sin detectar árboles
            if tiempo_sin_ver_arbol >= self.timeout_sin_arbol:
                self.estado_actual = 'FINALIZADO'
                self.detener_todo_emergencia()
                self.get_logger().error(
                    f"TIMEOUT: {tiempo_sin_ver_arbol:.1f}s sin detectar ningún árbol. "
                    "Deteniendo robot."
                )
                return

            # Detección de tronco confirmada a la altura del LiDAR
            if self.arbol_en_mira:
                self.arboles_procesados += 1
                self.tiempo_inicio_estado = now
                self.tiempo_ultimo_arbol = now
                self.estado_actual = 'ALINEANDO_BRAZO'
                self.get_logger().info(
                    f"Tronco {self.arboles_procesados}/{self.meta_arboles} detectado. "
                    f"Avanzando {self.tiempo_delay_brazo:.1f}s para alinear cámara MAPIR y brazo..."
                )

        elif self.estado_actual == 'ALINEANDO_BRAZO':
            # Sigue avanzando durante tiempo_delay_brazo para alinear la parte trasera con el tronco
            if tiempo_transcurrido < self.tiempo_delay_brazo:
                msg_vel.linear.x = self.velocidad_avance
            else:
                # Detener ruedas (STOP_Mov) e iniciar inspección de maleza con la cámara MAPIR
                msg_vel.linear.x = 0.0
                self.maleza_detectada_en_parada = False
                self.max_cobertura_en_parada = 0.0
                self.estado_actual = 'VERIFICANDO_MALEZA'
                self.tiempo_inicio_estado = now
                self.get_logger().info(
                    f"Robot detenido frente al tronco {self.arboles_procesados}/{self.meta_arboles}. "
                    f"Verificando presencia de maleza con cámara MAPIR ({self.tiempo_verificacion_maleza:.1f}s)..."
                )

        elif self.estado_actual == 'VERIFICANDO_MALEZA':
            # Robot detenido mientras la cámara MAPIR analiza si existe maleza al pie del tronco
            msg_vel.linear.x = 0.0

            # También considerar la lectura actual una vez pasado el tiempo de estabilización
            if tiempo_transcurrido >= self.tiempo_estabilizacion_camara and self.maleza_actual_topic:
                self.maleza_detectada_en_parada = True

            # CASO A: Se confirma presencia de maleza -> Activar secuencia de brazo
            if self.maleza_detectada_en_parada:
                self.arboles_con_maleza += 1
                self.orden_extender_enviada = False
                self.orden_retraer_enviada = False
                self.stm32_brazo_extendido = False
                self.stm32_brazo_retraido = False
                self.brazo_retraido_confirmado = False
                self.estado_actual = 'EXTENDIENDO_BRAZO'
                self.tiempo_inicio_estado = now
                self.get_logger().info(
                    f"[MAPIR] Maleza DETECTADA en tronco {self.arboles_procesados}/{self.meta_arboles} "
                    f"(cobertura: {self.max_cobertura_en_parada:.1f}%). "
                    f"Extendiendo brazo +{self.desplazamiento_brazo_mm:.0f}mm (esperando código 4 de STM32)..."
                )

            # CASO B: Terminó el tiempo de inspección y NO hay maleza -> NO activar brazo y continuar
            elif tiempo_transcurrido >= self.tiempo_verificacion_maleza:
                self.get_logger().info(
                    f"[MAPIR] SIN maleza en tronco {self.arboles_procesados}/{self.meta_arboles}. "
                    "NO se acciona el brazo."
                )
                if self._finalizar_o_continuar_siguiente_arbol(
                    now, motivo_log="Sin maleza (brazo omitido)."
                ):
                    return

        elif self.estado_actual == 'EXTENDIENDO_BRAZO':
            # Ruedas detenidas mientras el motor stepper recorre los +600 mm hasta que la STM32 envíe código 4
            msg_vel.linear.x = 0.0

            if not self.orden_extender_enviada and tiempo_transcurrido >= 0.10:
                self.stm32_brazo_extendido = False
                self.comandar_brazo('EXTENDER')
                self.orden_extender_enviada = True

            if self.orden_extender_enviada and self.stm32_brazo_extendido:
                self.estado_actual = 'MANTENIENDO_BRAZO'
                self.tiempo_inicio_estado = now
                self.get_logger().info(
                    f"STM32 confirmó brazo totalmente extendido (código 4). "
                    f"Manteniendo robot detenido con brazo extendido por {self.tiempo_brazo_extendido:.1f}s..."
                )

        elif self.estado_actual == 'MANTENIENDO_BRAZO':
            # Robot detenido con el brazo completamente extendido durante 4.0 s
            msg_vel.linear.x = 0.0

            if tiempo_transcurrido >= self.tiempo_brazo_extendido:
                self.stm32_brazo_retraido = False
                self.estado_actual = 'RETRAYENDO_BRAZO'
                self.tiempo_inicio_estado = now
                self.comandar_brazo('RETRAER')
                self.orden_retraer_enviada = True
                self.get_logger().info(
                    f"Espera con brazo extendido completada ({tiempo_transcurrido:.1f}s). "
                    f"Retrayendo brazo -{self.desplazamiento_brazo_mm:.0f}mm "
                    "(esperando confirmación código 3 de STM32)..."
                )

        elif self.estado_actual == 'RETRAYENDO_BRAZO':
            # Ruedas detenidas mientras el brazo se retrae; espera el mensaje código 3 de la STM32
            msg_vel.linear.x = 0.0

            if self.orden_retraer_enviada and self.stm32_brazo_retraido and not self.brazo_extendido:
                self.brazo_retraido_confirmado = True
                if self._finalizar_o_continuar_siguiente_arbol(
                    now, motivo_log="Maleza tratada y brazo retraído (código 3 STM32)."
                ):
                    return

        elif self.estado_actual == 'ENFRIAMIENTO':
            # Avanza ignorando el tronco que acaba de procesar, si y solo si el brazo está retraído
            if not self.brazo_retraido_confirmado or self.brazo_extendido:
                msg_vel.linear.x = 0.0
                self.cmd_pub.publish(msg_vel)
                return

            msg_vel.linear.x = self.velocidad_avance

            if tiempo_transcurrido >= self.tiempo_enfriamiento:
                self.estado_actual = 'AVANZANDO'
                self.tiempo_inicio_estado = now
                self.tiempo_ultimo_arbol = now
                self.get_logger().info(
                    f"Buscando árbol {self.arboles_procesados + 1}/{self.meta_arboles}..."
                )

        self.cmd_pub.publish(msg_vel)


def main(args=None):
    rclpy.init(args=args)
    node = MapirRutinaNavegacion()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.detener_todo_emergencia()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

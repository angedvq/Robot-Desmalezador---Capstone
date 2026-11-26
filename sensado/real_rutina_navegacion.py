#!/usr/bin/env python3
"""
Nodo de control autónomo de navegación por hilera de árboles sin cámara MAPIR
(`real_rutina_navegacion.py`), con control PID de trayectoria, doble régimen de velocidad
y barrido del brazo de 2.5 s en cada árbol detectado.
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


class RealRutinaNavegacion(Node):
    def __init__(self):
        super().__init__('real_rutina_navegacion')

        # ---------------------------------------------------------------------
        # Declaración de parámetros (limpios, sin parámetros obsoletos)
        # ---------------------------------------------------------------------
        # Parámetros geométricos y de avance
        self.declare_parameter('distancia_meta', 0.85)         # Distancia objetivo a la hilera (m)
        self.declare_parameter('distancia_frenado', 0.50)      # Umbral obstáculo TF-Mini (m)
        self.declare_parameter('velocidad_avance', 0.20)       # Velocidad lineal habitual (m/s)
        self.declare_parameter('velocidad_correccion', 0.065)  # Velocidad reducida al corregir ángulos grandes (m/s)

        # Parámetros de corrección suave de rumbo PID (LiDAR + IMU)
        self.declare_parameter('kp_rumbo', 30.0)               # Ganancia proporcional (grados por metro de error)
        self.declare_parameter('ki_rumbo', 5.0)                # Ganancia integral para vencer offset IMU y fricción
        self.declare_parameter('kd_rumbo', 5.0)                # Ganancia derivativa (grados por m/s de variación de error)
        self.declare_parameter('min_grados_rumbo', 7.0)        # Ángulo mínimo efectivo para mover las 4 ruedas
        self.declare_parameter('max_grados_rumbo', 30.0)       # Límite de ángulo suave (tope máximo de giro)
        self.declare_parameter('slew_rate_deg_s', 15.0)        # Rampa máxima de giro (grados/s)
        self.declare_parameter('alpha_dist', 0.40)             # Filtro exponencial de distancia lateral al tronco
        self.declare_parameter('umbral_angulo_lento', 18.0)    # Ángulo (>=) que activa velocidad_correccion (0.065 m/s)
        self.declare_parameter('umbral_angulo_reanudar', 6.0)  # Ángulo (<=) bajo el cual vuelve a velocidad_avance (0.20 m/s)

        # Parámetros de la rutina de árboles y barrido del brazo
        self.declare_parameter('meta_arboles', 4)
        self.declare_parameter('timeout_sin_arbol', 50.0)
        self.declare_parameter('tiempo_barrido_brazo', 2.5)    # Tiempo avanzando con el brazo extendido haciendo barrido (s)
        self.declare_parameter('tiempo_enfriamiento', 3.0)     # Ceguera temporal para avanzar al siguiente tronco (s)
        self.declare_parameter('desplazamiento_brazo_mm', 600.0)  # +600 mm (+60 cm) extiende, -600 mm retrae

        # Watchdogs de sensores reales
        self.declare_parameter('usar_tfmini', True)
        self.declare_parameter('lidar_timeout', 1.5)
        self.declare_parameter('tfmini_timeout', 1.5)

        # Lectura de parámetros
        self.distancia_meta = float(self.get_parameter('distancia_meta').value)
        self.distancia_frenado = float(self.get_parameter('distancia_frenado').value)
        self.velocidad_avance = float(self.get_parameter('velocidad_avance').value)
        self.velocidad_correccion = float(self.get_parameter('velocidad_correccion').value)
        self.kp_rumbo = float(self.get_parameter('kp_rumbo').value)
        self.ki_rumbo = float(self.get_parameter('ki_rumbo').value)
        self.kd_rumbo = float(self.get_parameter('kd_rumbo').value)
        self.min_grados_rumbo = float(self.get_parameter('min_grados_rumbo').value)
        self.max_grados_rumbo = float(self.get_parameter('max_grados_rumbo').value)
        self.slew_rate_deg_s = float(self.get_parameter('slew_rate_deg_s').value)
        self.alpha_dist = float(self.get_parameter('alpha_dist').value)
        self.umbral_angulo_lento = float(self.get_parameter('umbral_angulo_lento').value)
        self.umbral_angulo_reanudar = float(self.get_parameter('umbral_angulo_reanudar').value)

        self.meta_arboles = int(self.get_parameter('meta_arboles').value)
        self.timeout_sin_arbol = float(self.get_parameter('timeout_sin_arbol').value)
        self.tiempo_barrido_brazo = float(self.get_parameter('tiempo_barrido_brazo').value)
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
        self.stop_mov_pub = self.create_publisher(Bool, '/navegacion/stop_mov', 10)
        self.grados_pub = self.create_publisher(Float32, '/navegacion/corregir_grados', 10)
        self.brazo_pub = self.create_publisher(Float32, '/brazo_mm', 10)

        # ---------------------------------------------------------------------
        # Suscriptores de percepción y confirmación de brazo STM32
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

        # ---------------------------------------------------------------------
        # Variables de estado y temporizadores
        # ---------------------------------------------------------------------
        self.estado_actual = 'AVANZANDO'
        self.arboles_procesados = 0
        self.arboles_con_maleza = 0
        self.arbol_en_mira = False
        self.brazo_extendido = False
        self.orden_extender_enviada = False
        self.orden_retraer_enviada = False
        self.stm32_brazo_extendido = False
        self.stm32_brazo_retraido = False
        self.brazo_retraido_confirmado = True

        # Métricas acumuladas para el reporte final de la rutina
        self.inicio_rutina_ts = None
        self.tiempo_rutina_total = 0.0
        self.distancia_recorrida = 0.0
        self.velocidad_promedio = 0.0
        self.rutina_finalizada = False
        self.motivo_fin_rutina = ""
        self.ciclos_cierre = 0

        # Variables de guiado lateral y rumbo IMU (PID + adaptación de velocidad)
        self.distancia_tronco_actual = None
        self.ultimo_tronco_guia_ts = None
        self.error_previo = None
        self.derivada_error_filtrada = 0.0
        self.rumbo_integral = 0.0
        self.rumbo_suavizado = 0.0
        self.ultimo_rumbo_enviado = 0.0
        self.ultimo_envio_grados_ts = 0.0
        self.en_correccion_lenta = False

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
            f"real_rutina_navegacion iniciado | Meta: {self.meta_arboles} árboles | "
            f"Dist. meta: {self.distancia_meta:.2f}m | "
            f"Modo BARRIDO: {self.tiempo_barrido_brazo:.1f}s con brazo extendido | "
            f"Recorrido brazo: {self.desplazamiento_brazo_mm:.0f}mm (confirmación STM32: 4=ext, 3=ret)"
        )

    # =========================================================================
    # Callbacks
    # =========================================================================
    def estado_brazo_callback(self, msg: Int32):
        codigo = int(msg.data)
        if codigo == STM32_BRAZO_EXTENDIDO:
            self.stm32_brazo_extendido = True
            self.get_logger().info(
                "[STM32 -> Navegación] Código 4 recibido: Brazo EXTENDIDO."
            )
        elif codigo == STM32_BRAZO_RETRAIDO:
            self.stm32_brazo_retraido = True
            self.brazo_extendido = False
            self.brazo_retraido_confirmado = True
            self.get_logger().info(
                "[STM32 -> Navegación] Código 3 recibido: Brazo RETRAÍDO."
            )

    def tree_callback(self, msg: PoseArray):
        self.ultimo_lidar_ts = time.monotonic()
        self.arbol_en_mira = len(msg.poses) > 0

    def dist_tronco_callback(self, msg: Float32):
        now = time.monotonic()
        self.ultimo_lidar_ts = now

        if math.isfinite(msg.data) and msg.data > 0.0:
            nueva = float(msg.data)
            if self.distancia_tronco_actual is None:
                self.distancia_tronco_actual = nueva
                self.error_previo = None
                self.derivada_error_filtrada = 0.0
            else:
                self.distancia_tronco_actual = (
                    self.alpha_dist * nueva + (1.0 - self.alpha_dist) * self.distancia_tronco_actual
                )
            self.ultimo_tronco_guia_ts = now
        else:
            self.distancia_tronco_actual = None
            self.error_previo = None
            self.derivada_error_filtrada = 0.0

    def obs_callback(self, msg: Bool):
        self.obstaculo_topic = bool(msg.data)

    def tfmini_callback(self, msg: LaserScan):
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

    def comandar_stop_mov(self):
        """
        Envía la orden explícita STOP_Mov (Orden 11) a la STM32 a través de /navegacion/stop_mov
        y fija /cmd_vel en 0.0 m/s para detener las ruedas inmediatamente sin apagar el stepper.
        """
        msg_stop_mov = Bool()
        msg_stop_mov.data = True
        self.stop_mov_pub.publish(msg_stop_mov)

        msg_vel = Twist()
        msg_vel.linear.x = 0.0
        msg_vel.angular.z = 0.0
        self.cmd_pub.publish(msg_vel)

    def detener_todo_emergencia(self):
        msg_stop = Bool()
        msg_stop.data = True
        self.stop_pub.publish(msg_stop)

        msg_vel = Twist()
        msg_vel.linear.x = 0.0
        msg_vel.angular.z = 0.0
        self.cmd_pub.publish(msg_vel)

        self.rumbo_suavizado = 0.0
        self.rumbo_integral = 0.0
        self.error_previo = None
        self.derivada_error_filtrada = 0.0
        self.en_correccion_lenta = False

    def _pausar_relojes(self, now: float):
        self.tiempo_transcurrido_guardado = now - self.tiempo_inicio_estado
        self.tiempo_sin_ver_arbol_guardado = now - self.tiempo_ultimo_arbol

    def _restaurar_relojes(self, now: float):
        self.tiempo_inicio_estado = now - self.tiempo_transcurrido_guardado
        self.tiempo_ultimo_arbol = now - self.tiempo_sin_ver_arbol_guardado

    def _finalizar_rutina_con_reporte(self, motivo: str):
        self.estado_actual = 'FINALIZADO'
        self.motivo_fin_rutina = motivo
        self.detener_todo_emergencia()

        if not self.rutina_finalizada:
            self.rutina_finalizada = True
            self.ciclos_cierre = 0
            minutos = int(self.tiempo_rutina_total // 60)
            segundos = self.tiempo_rutina_total % 60.0
            reporte = (
                "\n"
                "============================================================\n"
                "                 REPORTE FINAL DE LA RUTINA                 \n"
                "============================================================\n"
                f" Motivo de término   : {self.motivo_fin_rutina}\n"
                f" Tiempo de rutina    : {self.tiempo_rutina_total:.2f} s ({minutos}m {segundos:04.1f}s)\n"
                f" Velocidad promedio  : {self.velocidad_promedio:.3f} m/s "
                f"(distancia est.: {self.distancia_recorrida:.2f} m)\n"
                f" Árboles encontrados : {self.arboles_procesados} / {self.meta_arboles}\n"
                f" Malezas encontradas : {self.arboles_con_maleza}\n"
                "============================================================"
            )
            self.get_logger().info(reporte)

    def _actualizar_y_publicar_rumbo(self, now: float, permitir_correccion_lateral: bool):
        tronco_guia_vigente = (
            permitir_correccion_lateral
            and self.distancia_tronco_actual is not None
            and self.ultimo_tronco_guia_ts is not None
            and (now - self.ultimo_tronco_guia_ts) <= 0.5
        )

        angulo_correccion = 0.0

        if tronco_guia_vigente:
            error = self.distancia_tronco_actual - self.distancia_meta

            if self.error_previo is None:
                self.derivada_error_filtrada = 0.0
            else:
                derivada_bruta = (error - self.error_previo) / self.dt
                self.derivada_error_filtrada = (
                    0.30 * derivada_bruta + 0.70 * self.derivada_error_filtrada
                )
            self.error_previo = error

            if abs(error) < 0.03:
                rumbo_objetivo = self.rumbo_integral
                angulo_correccion = abs(rumbo_objetivo - self.rumbo_suavizado)
            else:
                self.rumbo_integral += error * self.ki_rumbo * self.dt
                self.rumbo_integral = max(
                    -0.6 * self.max_grados_rumbo,
                    min(0.6 * self.max_grados_rumbo, self.rumbo_integral),
                )

                termino_p = error * self.kp_rumbo
                if abs(error) >= 0.04 and abs(termino_p) < self.min_grados_rumbo:
                    termino_p = math.copysign(self.min_grados_rumbo, error)

                termino_d = max(
                    -12.0,
                    min(12.0, self.kd_rumbo * self.derivada_error_filtrada),
                )

                correccion_pd = termino_p + termino_d
                rumbo_objetivo = max(
                    -self.max_grados_rumbo,
                    min(self.max_grados_rumbo, self.rumbo_integral + correccion_pd),
                )

                angulo_correccion = max(
                    abs(correccion_pd),
                    abs(rumbo_objetivo - self.rumbo_suavizado),
                    abs(rumbo_objetivo),
                )
        else:
            self.error_previo = None
            self.derivada_error_filtrada = 0.0
            rumbo_objetivo = self.rumbo_integral
            angulo_correccion = abs(rumbo_objetivo - self.rumbo_suavizado)

        if not permitir_correccion_lateral or not tronco_guia_vigente:
            if self.en_correccion_lenta:
                self.en_correccion_lenta = False
                self.get_logger().info(
                    f"[VELOCIDAD] Fin de tramo de corrección -> "
                    f"Restaurando velocidad habitual ({self.velocidad_avance:.2f} m/s)"
                )
        else:
            if not self.en_correccion_lenta and angulo_correccion >= self.umbral_angulo_lento:
                self.en_correccion_lenta = True
                self.get_logger().info(
                    f"[VELOCIDAD] Ángulo a corregir grande ({angulo_correccion:.1f}° >= "
                    f"{self.umbral_angulo_lento:.1f}°) -> Reduciendo velocidad a "
                    f"{self.velocidad_correccion:.3f} m/s para completar giro"
                )
            elif self.en_correccion_lenta and angulo_correccion <= self.umbral_angulo_reanudar:
                self.en_correccion_lenta = False
                self.get_logger().info(
                    f"[VELOCIDAD] Ángulo corregido ({angulo_correccion:.1f}° <= "
                    f"{self.umbral_angulo_reanudar:.1f}°) -> Restaurando velocidad habitual "
                    f"({self.velocidad_avance:.2f} m/s)"
                )

        max_paso = self.slew_rate_deg_s * self.dt
        delta = rumbo_objetivo - self.rumbo_suavizado
        if abs(delta) <= max_paso:
            self.rumbo_suavizado = rumbo_objetivo
        else:
            self.rumbo_suavizado += math.copysign(max_paso, delta)

        diferencia = abs(self.rumbo_suavizado - self.ultimo_rumbo_enviado)
        retorno_a_cero = (self.rumbo_suavizado == 0.0 and self.ultimo_rumbo_enviado != 0.0)
        requiere_refresco = (now - self.ultimo_envio_grados_ts) >= 1.0

        if diferencia >= 0.5 or retorno_a_cero or requiere_refresco:
            msg_grados = Float32()
            msg_grados.data = float(self.rumbo_suavizado)
            self.grados_pub.publish(msg_grados)

            if diferencia >= 0.5 or retorno_a_cero:
                ref_str = (f"Tronco a {self.distancia_tronco_actual:.2f}m" if tronco_guia_vigente else "Claro IMU 0°")
                vel_actual = self.velocidad_correccion if self.en_correccion_lenta else self.velocidad_avance
                self.get_logger().info(
                    f"[RUMBO] {ref_str} -> SET_Heading: {self.rumbo_suavizado:.1f}° "
                    f"(obj: {rumbo_objetivo:.1f}°, vel: {vel_actual:.3f} m/s)"
                )
            self.ultimo_rumbo_enviado = self.rumbo_suavizado
            self.ultimo_envio_grados_ts = now

    # =========================================================================
    # Bucle Principal de Control
    # =========================================================================
    def control_loop(self):
        now = time.monotonic()

        if self.estado_actual == 'FINALIZADO':
            self.detener_todo_emergencia()
            self.ciclos_cierre += 1
            if self.ciclos_cierre >= 4:
                self.control_timer.cancel()
                raise SystemExit(0)
            return

        operando_brazo_detenido = self.estado_actual in (
            'EXTENDIENDO_BRAZO',
            'RETRAYENDO_BRAZO',
        )
        lidar_ok = (self.ultimo_lidar_ts is not None and (now - self.ultimo_lidar_ts) <= self.lidar_timeout)
        tfmini_ok = (
            (not self.usar_tfmini)
            or (self.ultimo_tfmini_ts is not None and (now - self.ultimo_tfmini_ts) <= self.tfmini_timeout)
        )

        if (not lidar_ok or not tfmini_ok) and not operando_brazo_detenido:
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
            if self.inicio_rutina_ts is None:
                self.inicio_rutina_ts = now
            self._restaurar_relojes(now)
            self.en_pausa_por_watchdog = False
            self.get_logger().info("Sensores ok. Iniciando/reanudando rutina de árboles")

        hay_obstaculo = self.usar_tfmini and (self.obstaculo_topic or self.obstaculo_scan)
        if hay_obstaculo and not operando_brazo_detenido:
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

        msg_stop = Bool()
        msg_stop.data = False
        self.stop_pub.publish(msg_stop)

        tiempo_transcurrido = now - self.tiempo_inicio_estado
        tiempo_sin_ver_arbol = now - self.tiempo_ultimo_arbol

        msg_vel = Twist()

        if self.estado_actual == 'AVANZANDO':
            if not self.brazo_retraido_confirmado or self.brazo_extendido:
                msg_vel.linear.x = 0.0
                self.cmd_pub.publish(msg_vel)
                return

            self._actualizar_y_publicar_rumbo(now, permitir_correccion_lateral=True)
            msg_vel.linear.x = (
                self.velocidad_correccion if self.en_correccion_lenta else self.velocidad_avance
            )

            if tiempo_sin_ver_arbol >= self.timeout_sin_arbol:
                self.get_logger().error(
                    f"TIMEOUT: {tiempo_sin_ver_arbol:.1f}s sin detectar ningún árbol. "
                    "Deteniendo robot y generando reporte final."
                )
                self._finalizar_rutina_con_reporte(
                    motivo=f"Timeout ({tiempo_sin_ver_arbol:.1f}s sin detectar árboles)"
                )
                return

            if self.arbol_en_mira:
                self.arboles_procesados += 1
                self.tiempo_inicio_estado = now
                self.tiempo_ultimo_arbol = now
                msg_vel.linear.x = 0.0
                self.comandar_stop_mov()
                self.rumbo_integral = 0.0
                self.rumbo_suavizado = 0.0
                self.ultimo_rumbo_enviado = 0.0
                self.error_previo = None
                self.derivada_error_filtrada = 0.0
                self.en_correccion_lenta = False
                self.orden_extender_enviada = False
                self.orden_retraer_enviada = False
                self.stm32_brazo_extendido = False
                self.stm32_brazo_retraido = False
                self.brazo_retraido_confirmado = False
                self.estado_actual = 'EXTENDIENDO_BRAZO'
                self.get_logger().info(
                    f"Tronco {self.arboles_procesados}/{self.meta_arboles} detectado. "
                    f"Enviando STOP_Mov inmediatamente y extendiendo brazo +{self.desplazamiento_brazo_mm:.0f}mm..."
                )
                return

        elif self.estado_actual == 'EXTENDIENDO_BRAZO':
            msg_vel.linear.x = 0.0

            if not self.orden_extender_enviada and tiempo_transcurrido >= 0.10:
                self.stm32_brazo_extendido = False
                self.comandar_brazo('EXTENDER')
                self.orden_extender_enviada = True
                self.arboles_con_maleza += 1

            if self.orden_extender_enviada and self.stm32_brazo_extendido:
                self.estado_actual = 'BARRIENDO_BRAZO'
                self.tiempo_inicio_estado = now
                self.get_logger().info(
                    f"Brazo extendido (código 4). "
                    f"Iniciando barrido: avanzando durante {self.tiempo_barrido_brazo:.1f}s con brazo extendido..."
                )

        elif self.estado_actual == 'BARRIENDO_BRAZO':
            if tiempo_transcurrido < self.tiempo_barrido_brazo:
                msg_vel.linear.x = self.velocidad_avance
                self._actualizar_y_publicar_rumbo(now, permitir_correccion_lateral=False)
            else:
                msg_vel.linear.x = 0.0
                self.comandar_stop_mov()
                self.rumbo_integral = 0.0
                self.rumbo_suavizado = 0.0
                self.ultimo_rumbo_enviado = 0.0
                self.stm32_brazo_retraido = False
                self.orden_retraer_enviada = False
                self.estado_actual = 'RETRAYENDO_BRAZO'
                self.tiempo_inicio_estado = now
                self.get_logger().info(
                    f"Barrido completado ({tiempo_transcurrido:.1f}s). "
                    f"Enviando STOP_Mov y retrayendo brazo -{self.desplazamiento_brazo_mm:.0f}mm..."
                )
                return

        elif self.estado_actual == 'RETRAYENDO_BRAZO':
            msg_vel.linear.x = 0.0

            if not self.orden_retraer_enviada and tiempo_transcurrido >= 0.10:
                self.comandar_brazo('RETRAER')
                self.orden_retraer_enviada = True

            if self.orden_retraer_enviada and self.stm32_brazo_retraido and not self.brazo_extendido:
                self.brazo_retraido_confirmado = True
                if self.arboles_procesados >= self.meta_arboles:
                    self.get_logger().info(
                        f"Meta alcanzada ({self.arboles_procesados}/{self.meta_arboles} árboles procesados). "
                        "Brazo retraído y robot detenido por completo."
                    )
                    self._finalizar_rutina_con_reporte(
                        motivo=f"Meta alcanzada ({self.arboles_procesados}/{self.meta_arboles} árboles)"
                    )
                    return
                else:
                    self.estado_actual = 'ENFRIAMIENTO'
                    self.tiempo_inicio_estado = now
                    self.tiempo_ultimo_arbol = now
                    self.get_logger().info(
                        f"Árbol {self.arboles_procesados}/{self.meta_arboles} completado y brazo retraído. "
                        f"Reanudando marcha (enfriamiento de {self.tiempo_enfriamiento:.1f}s)..."
                    )

        elif self.estado_actual == 'ENFRIAMIENTO':
            if not self.brazo_retraido_confirmado or self.brazo_extendido:
                msg_vel.linear.x = 0.0
                self.cmd_pub.publish(msg_vel)
                return

            msg_vel.linear.x = self.velocidad_avance
            self._actualizar_y_publicar_rumbo(now, permitir_correccion_lateral=False)

            if tiempo_transcurrido >= self.tiempo_enfriamiento:
                self.estado_actual = 'AVANZANDO'
                self.tiempo_inicio_estado = now
                self.tiempo_ultimo_arbol = now
                self.get_logger().info(
                    f"Buscando árbol {self.arboles_procesados + 1}/{self.meta_arboles}..."
                )

        self.cmd_pub.publish(msg_vel)

        if self.inicio_rutina_ts is not None and not self.rutina_finalizada:
            self.tiempo_rutina_total += self.dt
            self.distancia_recorrida += abs(float(msg_vel.linear.x)) * self.dt
            if self.tiempo_rutina_total > 1e-6:
                self.velocidad_promedio = self.distancia_recorrida / self.tiempo_rutina_total


def main(args=None):
    rclpy.init(args=args)
    node = RealRutinaNavegacion()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        if rclpy.ok():
            node.detener_todo_emergencia()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

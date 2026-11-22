#!/usr/bin/env python3
import math
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseArray, Point
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Float32, Int32

STM32_BRAZO_RETRAIDO = 3   # Código enviado por la STM32 cuando el brazo terminó de retraerse
STM32_BRAZO_EXTENDIDO = 4  # Código enviado por la STM32 cuando el brazo terminó de extenderse


class RealRutinaNavegacion(Node):
    def __init__(self):
        super().__init__('real_rutina_navegacion')

        # ---------------------------------------------------------------------
        # Declaración de parámetros 
        # ---------------------------------------------------------------------
        # Parámetros geométricos y de avance
        self.declare_parameter('distancia_meta', 0.70)        # Distancia a la hilera (m)
        self.declare_parameter('distancia_frenado', 0.50)     # Umbral obstáculo TF-Mini (m)
        self.declare_parameter('velocidad_avance', 0.20)      # Velocidad lineal habitual (m/s)
        self.declare_parameter('velocidad_correccion', 0.05)  # Velocidad reducida al corregir ángulos grandes (m/s)

        # Parámetros de corrección suave de rumbo PID (LiDAR + IMU)
        self.declare_parameter('kp_rumbo', 45.0)              # Ganancia proporcional (grados por metro de error)
        self.declare_parameter('ki_rumbo', 20.0)              # Ganancia integral para vencer offset IMU y fricción
        self.declare_parameter('kd_rumbo', 15.0)              # Ganancia derivativa (grados por m/s de variación de error)
        self.declare_parameter('min_grados_rumbo', 7.0)       # Ángulo mínimo efectivo para mover las 4 ruedas
        self.declare_parameter('max_grados_rumbo', 30.0)      # Límite de ángulo suave. Tope maximo de giro
        self.declare_parameter('slew_rate_deg_s', 15.0)       # Rampa máxima de giro. Velocidad de giro
        self.declare_parameter('alpha_dist', 0.40)            # Filtro distancia tronco
        self.declare_parameter('umbral_angulo_lento', 12.0)   # Ángulo de corrección (>=) que activa baja velocidad (0.05 m/s)
        self.declare_parameter('umbral_angulo_reanudar', 6.0) # Ángulo (<=) bajo el cual se considera corregido y vuelve a 0.20 m/s

        # Parámetros de la rutina de árboles y brazo
        self.declare_parameter('meta_arboles', 4)      
        self.declare_parameter('timeout_sin_arbol', 50.0)    
        self.declare_parameter('distancia_lidar_brazo', 0.28)     # Separación longitudinal entre LiDAR y brazo (28 cm)
        self.declare_parameter('tiempo_delay_brazo', 2.2)         # Timeout máximo de respaldo para alinear brazo (s)
        self.declare_parameter('tiempo_brazo_extendido', 4.0)     # Tiempo detenido con el brazo ya extendido (s)
        self.declare_parameter('tiempo_enfriamiento', 3.0)        # Ceguera temporal para avanzar a sgte tronco (s)
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
        self.distancia_lidar_brazo = abs(float(self.get_parameter('distancia_lidar_brazo').value))
        self.tiempo_delay_brazo = float(self.get_parameter('tiempo_delay_brazo').value)
        self.tiempo_brazo_extendido = float(self.get_parameter('tiempo_brazo_extendido').value)
        self.tiempo_enfriamiento = float(self.get_parameter('tiempo_enfriamiento').value)
        self.desplazamiento_brazo_mm = abs(float(self.get_parameter('desplazamiento_brazo_mm').value))

        self.usar_tfmini = bool(self.get_parameter('usar_tfmini').value)
        self.lidar_timeout = float(self.get_parameter('lidar_timeout').value)
        self.tfmini_timeout = float(self.get_parameter('tfmini_timeout').value)

        # ---------------------------------------------------------------------
        # Publicadores hacia stm32
        # ---------------------------------------------------------------------
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.stop_pub = self.create_publisher(Bool, '/navegacion/stop', 10)
        self.grados_pub = self.create_publisher(Float32, '/navegacion/corregir_grados', 10)
        self.brazo_pub = self.create_publisher(Float32, '/brazo_mm', 10)

        # ---------------------------------------------------------------------
        # Suscriptores de percepción y confirmación de brazo STM32
        # ---------------------------------------------------------------------
        self.tree_sub = self.create_subscription(PoseArray, '/arboles_detectados', self.tree_callback, 10)
        self.dist_sub = self.create_subscription(Float32, '/lidar/distancia_tronco_lateral', self.dist_tronco_callback, 10)
        self.pos_alin_sub = self.create_subscription(
            Point, '/lidar/posicion_tronco_alineacion', self.pos_alineacion_callback, 10
        )
        self.obs_sub = self.create_subscription(Bool, '/obstaculo_frontal', self.obs_callback, 10)
        self.tfmini_sub = self.create_subscription(LaserScan, '/tf_mini/scan', self.tfmini_callback, 10)
        self.estado_brazo_sub = self.create_subscription(Int32, '/stm32/estado_brazo', self.estado_brazo_callback, 10)

        # ---------------------------------------------------------------------
        # Variables de estado y temporizadores
        # ---------------------------------------------------------------------
        self.estado_actual = 'AVANZANDO'
        self.arboles_procesados = 0
        self.arbol_en_mira = False
        self.brazo_extendido = False
        self.orden_extender_enviada = False
        self.orden_retraer_enviada = False
        self.stm32_brazo_extendido = False
        self.stm32_brazo_retraido = False
        self.brazo_retraido_confirmado = True

        # Variables de guiado lateral, triangulación del brazo y rumbo IMU (PID + adaptación de velocidad)
        self.distancia_tronco_actual = None
        self.ultimo_tronco_guia_ts = None
        self.tronco_alineacion_x = None
        self.tronco_alineacion_y = None
        self.tronco_alineacion_diag = None
        self.ultimo_tronco_alin_ts = None
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
            f"Triangulación brazo-LiDAR: {self.distancia_lidar_brazo * 100.0:.0f}cm (timeout respaldo: {self.tiempo_delay_brazo:.1f}s) | "
            f"Espera con brazo extendido: {self.tiempo_brazo_extendido:.1f}s | "
            f"Recorrido brazo: {self.desplazamiento_brazo_mm:.0f}mm (confirmación por STM32: 4=ext, 3=ret)"
        )

    # =========================================================================
    # Callbacks
    # =========================================================================
    def estado_brazo_callback(self, msg: Int32):
        """
        Recibe los avisos de fin de movimiento del brazo desde la STM32:
        - 4 (STM32_BRAZO_EXTENDIDO): El stepper terminó de extenderse 
        - 3 (STM32_BRAZO_RETRAIDO):  El stepper terminó de retraerse
        """
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
        """Recibe las detecciones de troncos en la ventana lateral"""
        self.ultimo_lidar_ts = time.monotonic()
        self.arbol_en_mira = len(msg.poses) > 0

    def dist_tronco_callback(self, msg: Float32):
        """Recibe la distancia perpendicular al tronco más cercano en diagonal frontal-izquierdo. 
        Si recibe NaN, significa que no hay tronco (espacio entre árboles) """
        now = time.monotonic()
        self.ultimo_lidar_ts = now

        if math.isfinite(msg.data) and msg.data > 0.0:
            nueva = float(msg.data)
            if self.distancia_tronco_actual is None:
                self.distancia_tronco_actual = nueva
                self.error_previo = None
                self.derivada_error_filtrada = 0.0
            else:
                # Filtrar y suavizar conduccion
                self.distancia_tronco_actual = (
                    self.alpha_dist * nueva + (1.0 - self.alpha_dist) * self.distancia_tronco_actual
                )
            self.ultimo_tronco_guia_ts = now
        else:
            # En el claro entre árboles no hay lectura lateral
            self.distancia_tronco_actual = None
            self.error_previo = None
            self.derivada_error_filtrada = 0.0

    def pos_alineacion_callback(self, msg: Point):
        """
        Recibe desde /lidar/posicion_tronco_alineacion las coordenadas del tronco que
        está siendo rebasado por el LiDAR:
        - msg.x: x_robot (m) -> negativo cuando el LiDAR ya adelantó al tronco.
                 Cuando x_robot <= -distancia_lidar_brazo (-0.28 m), el brazo trasero
                 queda exactamente centrado con el tronco.
        - msg.y: y_robot (m) -> distancia lateral perpendicular al tronco.
        - msg.z: d_diag (m)  -> distancia diagonal real entre el LiDAR y el centro del tronco
                 (hipotenusa del triángulo rectángulo: d_obj = sqrt(y_robot^2 + L_brazo^2)).
        """
        now = time.monotonic()
        self.ultimo_lidar_ts = now
        if math.isfinite(msg.x) and math.isfinite(msg.y) and math.isfinite(msg.z):
            self.tronco_alineacion_x = float(msg.x)
            self.tronco_alineacion_y = float(msg.y)
            self.tronco_alineacion_diag = float(msg.z)
            self.ultimo_tronco_alin_ts = now
        else:
            self.tronco_alineacion_x = None
            self.tronco_alineacion_y = None
            self.tronco_alineacion_diag = None

    def obs_callback(self, msg: Bool):
        """Recibe flag de obstáculo frontal desde procesamiento_tfmini"""
        self.obstaculo_topic = bool(msg.data)

    def tfmini_callback(self, msg: LaserScan):
        """
        Monitorea /tf_mini/scan para actualizar el watchdog del
        sensor frontal y verificar distancias menores a distancia_frenado
        """
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
        Envía el comando de brazo hacia la stm32 a través de /brazo_mm:
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
        """
        Envía STOP_All y velocidad 0.0 m/s
        """
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
        """Congela los tiempos transcurridos durante una pausa por obstáculo o watchdog"""
        self.tiempo_transcurrido_guardado = now - self.tiempo_inicio_estado
        self.tiempo_sin_ver_arbol_guardado = now - self.tiempo_ultimo_arbol

    def _restaurar_relojes(self, now: float):
        """Restaura los tiempos transcurridos al reanudarse la operación"""
        self.tiempo_inicio_estado = now - self.tiempo_transcurrido_guardado
        self.tiempo_ultimo_arbol = now - self.tiempo_sin_ver_arbol_guardado

    def _actualizar_y_publicar_rumbo(self, now: float, permitir_correccion_lateral: bool):
        """
        Calcula y publica el rumbo objetivo hacia la STM32 por /navegacion/corregir_grados:
        - Si `permitir_correccion_lateral` es True y hay un tronco guía reciente (< 0.5 s),
          aplica control PID (Proporcional + Integral + Derivativo) + ángulo mínimo efectivo
          (`min_grados_rumbo`) para vencer la fricción de las 4 ruedas y amortiguar sobrepasos.
        - Evalúa la magnitud del ángulo a corregir (`angulo_correccion`): si supera
          `umbral_angulo_lento` (12° por defecto), activa `self.en_correccion_lenta = True`
          para bajar la velocidad lineal a `velocidad_correccion` (0.05 m/s) hasta que el
          ángulo se reduzca por debajo de `umbral_angulo_reanudar` (6° por defecto).
        - Cuando el error entra en la banda muerta (< 3 cm) o no hay tronco guía, mantiene
          el rumbo base acumulado (`self.rumbo_integral`).
        """
        tronco_guia_vigente = (
            permitir_correccion_lateral
            and self.distancia_tronco_actual is not None
            and self.ultimo_tronco_guia_ts is not None
            and (now - self.ultimo_tronco_guia_ts) <= 0.5
        )

        angulo_correccion = 0.0

        # Si tenemos un tronco guia vigente, aplica corrección PID (Proporcional + Integral + Derivativa)
        if tronco_guia_vigente:
            error = self.distancia_tronco_actual - self.distancia_meta

            # Cálculo del término derivativo filtrado sobre la evolución continua del error lateral
            if self.error_previo is None:
                self.derivada_error_filtrada = 0.0
            else:
                derivada_bruta = (error - self.error_previo) / self.dt
                self.derivada_error_filtrada = (
                    0.30 * derivada_bruta + 0.70 * self.derivada_error_filtrada
                )
            self.error_previo = error

            if abs(error) < 0.03:  # Dentro de +-3 cm de 0.70 m: mantener rumbo base actual
                rumbo_objetivo = self.rumbo_integral
                angulo_correccion = abs(rumbo_objetivo - self.rumbo_suavizado)
            else:
                # 1. Término Integral (acotado al 60% del rumbo máximo)
                self.rumbo_integral += error * self.ki_rumbo * self.dt
                self.rumbo_integral = max(
                    -0.6 * self.max_grados_rumbo,
                    min(0.6 * self.max_grados_rumbo, self.rumbo_integral),
                )

                # 2. Término Proporcional con ángulo mínimo efectivo para mover las 4 ruedas
                termino_p = error * self.kp_rumbo
                if abs(error) >= 0.04 and abs(termino_p) < self.min_grados_rumbo:
                    termino_p = math.copysign(self.min_grados_rumbo, error)

                # 3. Término Derivativo (acotado a +-12° para amortiguar sin amplificar ruido del LiDAR)
                termino_d = max(
                    -12.0,
                    min(12.0, self.kd_rumbo * self.derivada_error_filtrada),
                )

                correccion_pd = termino_p + termino_d
                rumbo_objetivo = max(
                    -self.max_grados_rumbo,
                    min(self.max_grados_rumbo, self.rumbo_integral + correccion_pd),
                )

                # Magnitud del ángulo que el robot necesita corregir en este instante
                angulo_correccion = max(
                    abs(correccion_pd),
                    abs(rumbo_objetivo - self.rumbo_suavizado),
                    abs(rumbo_objetivo),
                )
        else:
            # En el claro entre árboles o últimos 25 cm antes del tronco, mantener el rumbo base alineado
            self.error_previo = None
            self.derivada_error_filtrada = 0.0
            rumbo_objetivo = self.rumbo_integral
            angulo_correccion = abs(rumbo_objetivo - self.rumbo_suavizado)

        # ---------------------------------------------------------------------
        # Lógica de separación de casos de velocidad según ángulo de corrección
        # ---------------------------------------------------------------------
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
                    f"{self.velocidad_correccion:.2f} m/s para completar giro"
                )
            elif self.en_correccion_lenta and angulo_correccion <= self.umbral_angulo_reanudar:
                self.en_correccion_lenta = False
                self.get_logger().info(
                    f"[VELOCIDAD] Ángulo corregido ({angulo_correccion:.1f}° <= "
                    f"{self.umbral_angulo_reanudar:.1f}°) -> Restaurando velocidad habitual "
                    f"({self.velocidad_avance:.2f} m/s)"
                )

        # Limitador de pendiente (slew-rate) para que todo giro sea suave
        max_paso = self.slew_rate_deg_s * self.dt
        delta = rumbo_objetivo - self.rumbo_suavizado
        if abs(delta) <= max_paso:
            self.rumbo_suavizado = rumbo_objetivo
        else:
            self.rumbo_suavizado += math.copysign(max_paso, delta)

        # Se envia orden a stm32 solo si hubo un cambio significante (0.5°) para no saturar
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
                    f"(obj: {rumbo_objetivo:.1f}°, vel: {vel_actual:.2f} m/s)"
                )
            self.ultimo_rumbo_enviado = self.rumbo_suavizado
            self.ultimo_envio_grados_ts = now

    # =========================================================================
    # Bucle Principal de Control 
    # =========================================================================
    def control_loop(self):
        now = time.monotonic()

        # Si la rutina ya terminó, mantener todo apagado con STOP_All
        if self.estado_actual == 'FINALIZADO':
            self.detener_todo_emergencia()
            return

        # Verificación de Watchdogs de Sensores. Revisa si pasó mucho tiempo desde la ultima recepcion de datos de sensores
        operando_brazo_detenido = self.estado_actual in (
            'EXTENDIENDO_BRAZO',
            'MANTENIENDO_BRAZO',
            'RETRAYENDO_BRAZO',
        )
        lidar_ok = (self.ultimo_lidar_ts is not None and (now - self.ultimo_lidar_ts) <= self.lidar_timeout)
        tfmini_ok = ((not self.usar_tfmini) or (self.ultimo_tfmini_ts is not None and (now - self.ultimo_tfmini_ts) <= self.tfmini_timeout))

        # Si algun sensor no está funcionando mientras el robot se desplaza, freno de emergencia.
        # Si ya está detenido accionando el brazo, NO enviamos STOP_All 
        # para no interrumpir el recorrido del stepper
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

        # Si todo esta ok, quitar la pausa
        elif self.en_pausa_por_watchdog and lidar_ok and tfmini_ok:
            self._restaurar_relojes(now)
            self.en_pausa_por_watchdog = False
            self.get_logger().info("Sensores ok. Iniciando/reanudando rutina de árboles")

        # ---------------------------------------------------------------------
        # Obstáculo Frontal (solo detiene cuando el robot pretende avanzar)
        # ---------------------------------------------------------------------
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

        # Continuar ruta
        elif self.en_pausa_por_obstaculo and not hay_obstaculo:
            self._restaurar_relojes(now)
            self.en_pausa_por_obstaculo = False
            self.get_logger().info("Ruta despejada. Reanudando...")

        # Operación normal 
        msg_stop = Bool()
        msg_stop.data = False
        self.stop_pub.publish(msg_stop)

        # Tiempos activos en el estado actual y desde el último árbol visto
        tiempo_transcurrido = now - self.tiempo_inicio_estado
        tiempo_sin_ver_arbol = now - self.tiempo_ultimo_arbol

        msg_vel = Twist()

        # ---------------------------------------------------------------------
        # Máquina de Estados de Navegación y Accionamiento de Brazo
        # ---------------------------------------------------------------------
        if self.estado_actual == 'AVANZANDO':
            # Garantía de seguridad: solo avanzar si el brazo está retraído
            if not self.brazo_retraido_confirmado or self.brazo_extendido:
                msg_vel.linear.x = 0.0
                self.cmd_pub.publish(msg_vel)
                return

            self._actualizar_y_publicar_rumbo(now, permitir_correccion_lateral=True)
            msg_vel.linear.x = (
                self.velocidad_correccion if self.en_correccion_lenta else self.velocidad_avance
            )

            # Criterio de parada por timeout de 20 segundos sin detectar árboles
            if tiempo_sin_ver_arbol >= self.timeout_sin_arbol:
                self.estado_actual = 'FINALIZADO'
                self.detener_todo_emergencia()
                self.get_logger().error(
                    f"TIMEOUT: {tiempo_sin_ver_arbol:.1f}s sin detectar ningún árbol "
                    f"Deteniendo robot"
                )
                return

            # Detección de tronco confirmada a la altura del LiDAR
            if self.arbol_en_mira:
                self.arboles_procesados += 1
                self.tiempo_inicio_estado = now
                self.tiempo_ultimo_arbol = now
                self.estado_actual = 'ALINEANDO_BRAZO'
                y_ref = (
                    self.tronco_alineacion_y
                    if self.tronco_alineacion_y is not None
                    else self.distancia_meta
                )
                d_diag_obj = math.hypot(y_ref, self.distancia_lidar_brazo)
                self.get_logger().info(
                    f"Tronco {self.arboles_procesados}/{self.meta_arboles} detectado a la altura del LiDAR. "
                    f"Triangulando brazo: avanzando hasta que el LiDAR supere al tronco por "
                    f"{self.distancia_lidar_brazo * 100.0:.0f} cm (diag objetivo ~{d_diag_obj * 100.0:.1f} cm)"
                )

        elif self.estado_actual == 'ALINEANDO_BRAZO':
            # Triangulación con el LiDAR para centrar el brazo (ubicado `distancia_lidar_brazo` detrás del LiDAR):
            # Formamos el triángulo rectángulo entre:
            # - Cateto lateral `y_robot`: distancia perpendicular del tronco al robot (~0.70 m).
            # - Cateto longitudinal `L_brazo = distancia_lidar_brazo`: separación LiDAR-brazo (0.28 m).
            # - Hipotenusa `d_diag_objetivo = sqrt(y_robot^2 + L_brazo^2)` (~0.754 m cuando y_robot = 0.70 m).
            # El brazo queda centrado frente al tronco cuando el LiDAR ya pasó al árbol (`x_robot < 0`)
            # y el cateto longitudinal alcanza `-L_brazo` (equivalentemente `d_diag >= d_diag_objetivo`).
            lectura_alin_vigente = (
                self.tronco_alineacion_x is not None
                and self.tronco_alineacion_y is not None
                and self.tronco_alineacion_diag is not None
                and self.ultimo_tronco_alin_ts is not None
                and (now - self.ultimo_tronco_alin_ts) <= 0.4
            )

            alineado_por_lidar = False
            restante_longitudinal = self.distancia_lidar_brazo
            d_diag_objetivo = math.hypot(self.distancia_meta, self.distancia_lidar_brazo)

            if lectura_alin_vigente:
                d_diag_objetivo = math.hypot(self.tronco_alineacion_y, self.distancia_lidar_brazo)
                # Cuánto le falta avanzar al LiDAR para quedar `distancia_lidar_brazo` por delante del tronco
                restante_longitudinal = self.tronco_alineacion_x + self.distancia_lidar_brazo
                if (
                    self.tronco_alineacion_x <= -self.distancia_lidar_brazo
                    or (self.tronco_alineacion_x < -0.05 and self.tronco_alineacion_diag >= d_diag_objetivo)
                ):
                    alineado_por_lidar = True

            timeout_respaldo = tiempo_transcurrido >= self.tiempo_delay_brazo

            if not alineado_por_lidar and not timeout_respaldo:
                # Si faltan <= 8 cm para llegar al centro del tronco, reducir a velocidad de aproximación fina (0.05 m/s)
                if lectura_alin_vigente and restante_longitudinal <= 0.08:
                    msg_vel.linear.x = self.velocidad_correccion
                else:
                    msg_vel.linear.x = self.velocidad_avance
                self._actualizar_y_publicar_rumbo(now, permitir_correccion_lateral=False)
            else:
                # Detener primero las ruedas (STOP_Mov) antes de iniciar la extensión del brazo.
                # Al detenerse, real_stm32_bridge ejecuta RESET_TRACKING (fijando la orientación
                # ya alineada frente al árbol como nuevo 0.0°), por lo que reiniciamos los acumuladores.
                msg_vel.linear.x = 0.0
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
                self.tiempo_inicio_estado = now

                if alineado_por_lidar:
                    motivo_alin = (
                        f"Triangulación LiDAR completada (x={self.tronco_alineacion_x * 100.0:.1f} cm, "
                        f"y={self.tronco_alineacion_y * 100.0:.1f} cm, "
                        f"diag={self.tronco_alineacion_diag * 100.0:.1f}/{d_diag_objetivo * 100.0:.1f} cm)"
                    )
                else:
                    motivo_alin = f"Timeout de respaldo alcanzado ({tiempo_transcurrido:.2f}s)"

                self.get_logger().info(
                    f"Brazo alineado con tronco {self.arboles_procesados} [{motivo_alin}]. "
                    f"Deteniendo ruedas y extendiendo brazo +{self.desplazamiento_brazo_mm:.0f}mm "
                )

        elif self.estado_actual == 'EXTENDIENDO_BRAZO':
            # Ruedas detenidas mientras el motor stepper recorre los +600 mm hasta que la STM32 envíe código 4
            msg_vel.linear.x = 0.0

            # Enviar la orden de extender tras 0.10 s de haber frenado las ruedas
            if not self.orden_extender_enviada and tiempo_transcurrido >= 0.10:
                self.stm32_brazo_extendido = False
                self.comandar_brazo('EXTENDER')
                self.orden_extender_enviada = True

            # Pasa al estado de espera de 4s únicamente cuando la STM32 confirma código 4 (brazo extendido)
            if self.orden_extender_enviada and self.stm32_brazo_extendido:
                self.estado_actual = 'MANTENIENDO_BRAZO'
                self.tiempo_inicio_estado = now
                self.get_logger().info(
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
                )

        elif self.estado_actual == 'RETRAYENDO_BRAZO':
            # Ruedas detenidas mientras el brazo se retrae; espera el mensaje código 3 de la STM32
            msg_vel.linear.x = 0.0

            # Avanza si y solo si se envió la orden de retraer y la STM32 confirmó código 3 (brazo totalmente retraído)
            if self.orden_retraer_enviada and self.stm32_brazo_retraido and not self.brazo_extendido:
                self.brazo_retraido_confirmado = True
                if self.arboles_procesados >= self.meta_arboles:
                    self.estado_actual = 'FINALIZADO'
                    self.detener_todo_emergencia()
                    self.get_logger().info(
                        f"Meta alcanzada ({self.arboles_procesados}/{self.meta_arboles} árboles procesados). "
                        "Brazo retraído y robot detenido por completo."
                    )
                    return
                else:
                    self.estado_actual = 'ENFRIAMIENTO'
                    self.tiempo_inicio_estado = now
                    self.tiempo_ultimo_arbol = now
                    self.get_logger().info(
                        f"Árbol {self.arboles_procesados}/{self.meta_arboles} completado y brazo retraído"
                        f"Reanudando marcha (enfriamiento de {self.tiempo_enfriamiento:.1f}s)..."
                    )

        elif self.estado_actual == 'ENFRIAMIENTO':
            # Avanza ignorando el tronco que acaba de procesar, si y solo si el brazo ya fue retraído
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


def main(args=None):
    rclpy.init(args=args)
    node = RealRutinaNavegacion()
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
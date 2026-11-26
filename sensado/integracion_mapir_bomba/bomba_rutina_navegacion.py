#!/usr/bin/env python3
"""
Nodo de control autónomo de navegación por hilera de árboles con integración
configurable de cámara MAPIR (`usar_mapir`) y bomba de rociado (`usar_bomba`).

Ubicación: `sensado/integracion_mapir_bomba/bomba_rutina_navegacion.py`

Soporta los 4 modos de operación posibles:
1. `usar_mapir=True`,  `usar_bomba=True`  (MAPIR + Bomba):
   - Al encender: realiza calibración de bomba (extiende brazo -> activa bomba 5s -> apaga bomba -> retrae brazo).
   - En cada árbol: se detiene (`STOP_Mov`) e inspecciona maleza con la cámara MAPIR (2s).
     Si y solo si hay maleza: extiende brazo -> activa bomba -> barre por 2.5s -> detiene bomba -> retrae brazo.
     Si NO hay maleza: continúa al siguiente árbol sin mover el brazo ni activar la bomba.
2. `usar_mapir=True`,  `usar_bomba=False` (Solo MAPIR, sin bomba):
   - Al encender: inicia directamente buscando árboles (sin cebado de bomba).
   - En cada árbol: se detiene e inspecciona con la cámara MAPIR.
     Si y solo si hay maleza: extiende brazo -> barre por 2.5s (sin bomba) -> retrae brazo.
     Si NO hay maleza: continúa al siguiente árbol sin mover el brazo.
3. `usar_mapir=False`, `usar_bomba=True`  (Solo Bomba, sin MAPIR):
   - Al encender: realiza calibración de bomba (extiende brazo -> activa bomba 5s -> apaga bomba -> retrae brazo).
   - En cada árbol: se detiene (`STOP_Mov`) -> extiende brazo -> activa bomba -> barre por 2.5s -> detiene bomba -> retrae brazo.
4. `usar_mapir=False`, `usar_bomba=False` (Ni Bomba ni MAPIR):
   - Al encender: inicia directamente buscando árboles (sin cebado de bomba).
   - En cada árbol: se detiene (`STOP_Mov`) -> extiende brazo -> barre por 2.5s (sin bomba) -> retrae brazo.
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


class BombaRutinaNavegacion(Node):
    def __init__(self, nombre_nodo='bomba_rutina_navegacion', default_usar_mapir=True, default_usar_bomba=True):
        super().__init__(nombre_nodo)

        # ---------------------------------------------------------------------
        # Declaración de parámetros
        # ---------------------------------------------------------------------
        # Flags de activación de subsistemas (permiten cubrir los 4 casos)
        self.declare_parameter('usar_mapir', default_usar_mapir)
        self.declare_parameter('usar_bomba', default_usar_bomba)

        # Parámetros geométricos y de avance
        self.declare_parameter('distancia_meta', 0.85)         # Distancia a la hilera (m)
        self.declare_parameter('distancia_frenado', 0.50)      # Umbral obstáculo TF-Mini (m)
        self.declare_parameter('velocidad_avance', 0.20)       # Velocidad lineal habitual (m/s)
        self.declare_parameter('velocidad_correccion', 0.065)  # Velocidad reducida al corregir ángulos grandes (m/s)

        # Parámetros de corrección suave de rumbo PID (LiDAR + IMU)
        self.declare_parameter('kp_rumbo', 30.0)               # Ganancia proporcional (grados por metro de error)
        self.declare_parameter('ki_rumbo', 5.0)                # Ganancia integral para vencer offset IMU y fricción
        self.declare_parameter('kd_rumbo', 5.0)                # Ganancia derivativa (grados por m/s de variación de error)
        self.declare_parameter('min_grados_rumbo', 7.0)        # Ángulo mínimo efectivo para mover las 4 ruedas
        self.declare_parameter('max_grados_rumbo', 30.0)       # Límite de ángulo suave. Tope máximo de giro
        self.declare_parameter('slew_rate_deg_s', 15.0)        # Rampa máxima de giro. Velocidad de giro
        self.declare_parameter('alpha_dist', 0.40)             # Filtro distancia tronco
        self.declare_parameter('umbral_angulo_lento', 18.0)    # Ángulo de corrección (>=) que activa baja velocidad (0.065 m/s)
        self.declare_parameter('umbral_angulo_reanudar', 6.0)  # Ángulo (<=) bajo el cual se considera corregido y vuelve a 0.20 m/s

        # Parámetros de la rutina de árboles, inspección MAPIR, bomba y barrido del brazo
        self.declare_parameter('meta_arboles', 4)
        self.declare_parameter('timeout_sin_arbol', 50.0)
        self.declare_parameter('tiempo_calibracion_bomba', 5.0)     # Tiempo inicial cebando la manguera con brazo extendido (s)
        self.declare_parameter('tiempo_barrido_brazo', 2.5)         # Tiempo avanzando con el brazo extendido haciendo barrido (s)
        self.declare_parameter('tiempo_delay_brazo', 1.5)           # Parámetro legado
        self.declare_parameter('tiempo_verificacion_maleza', 2.0)   # Ventana de inspección MAPIR estando detenido (s)
        self.declare_parameter('tiempo_estabilizacion_camara', 0.3) # Espera tras frenar antes de muestrear MAPIR (s)
        self.declare_parameter('tiempo_brazo_extendido', 2.0)       # Parámetro legado / respaldo (s)
        self.declare_parameter('tiempo_enfriamiento', 3.0)          # Ceguera temporal para avanzar al siguiente tronco (s)
        self.declare_parameter('desplazamiento_brazo_mm', 600.0)    # +600 mm (+60 cm) extiende, -600 mm retrae

        # Watchdogs de sensores reales
        self.declare_parameter('usar_tfmini', True)
        self.declare_parameter('lidar_timeout', 1.5)
        self.declare_parameter('tfmini_timeout', 1.5)

        # Lectura de parámetros
        self.usar_mapir = bool(self.get_parameter('usar_mapir').value)
        self.usar_bomba = bool(self.get_parameter('usar_bomba').value)

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
        self.tiempo_calibracion_bomba = float(self.get_parameter('tiempo_calibracion_bomba').value)
        self.tiempo_barrido_brazo = float(self.get_parameter('tiempo_barrido_brazo').value)
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
        self.stop_mov_pub = self.create_publisher(Bool, '/navegacion/stop_mov', 10)
        self.grados_pub = self.create_publisher(Float32, '/navegacion/corregir_grados', 10)
        self.brazo_pub = self.create_publisher(Float32, '/brazo_mm', 10)
        self.bomba_pub = self.create_publisher(Bool, '/bomba', 10)

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
        # Si `usar_bomba=True`, inicia en `CAL_EXTENDIENDO_BRAZO` para cebar la manguera 5s.
        # Si `usar_bomba=False`, salta la calibración de bomba e inicia directo en `AVANZANDO`.
        # ---------------------------------------------------------------------
        self.estado_actual = 'CAL_EXTENDIENDO_BRAZO' if self.usar_bomba else 'AVANZANDO'
        self.calibracion_bomba_completada = not self.usar_bomba
        self.arboles_procesados = 0
        self.arboles_con_maleza = 0
        self.arbol_en_mira = False

        # Estado de maleza (MAPIR)
        self.maleza_actual_topic = False
        self.porcentaje_maleza_actual = 0.0
        self.ultimo_mapir_ts = None
        self.maleza_detectada_en_parada = False
        self.max_cobertura_en_parada = 0.0

        # Estado del brazo y de la bomba (STM32)
        self.brazo_extendido = False
        self.bomba_activa = False
        self.orden_extender_enviada = False
        self.orden_retraer_enviada = False
        self.stm32_brazo_extendido = False
        self.stm32_brazo_retraido = False
        self.brazo_retraido_confirmado = not self.usar_bomba
        self.ts_codigo_brazo_extendido = None
        self.ts_codigo_brazo_retraido = None
        self.ts_orden_extender = None
        self.rehoming_en_calibracion = False
        self.micro_avance_end_enviado = False

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
            f"{nombre_nodo} iniciado | Meta: {self.meta_arboles} árboles | "
            f"MAPIR: {'SI' if self.usar_mapir else 'NO'} | "
            f"BOMBA: {'SI (calibración ' + f'{self.tiempo_calibracion_bomba:.1f}s)' if self.usar_bomba else 'NO (solo barrido seco)'} | "
            f"Barrido: {self.tiempo_barrido_brazo:.1f}s | "
            f"Recorrido brazo: {self.desplazamiento_brazo_mm:.0f}mm (confirmación STM32: 4=ext, 3=ret)"
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
            self.ts_codigo_brazo_extendido = time.monotonic()
            self.micro_avance_end_enviado = False
            self.get_logger().info(
                "[STM32 -> Navegación] Código 4 recibido: Brazo totalmente EXTENDIDO."
            )
        elif codigo == STM32_BRAZO_RETRAIDO:
            self.stm32_brazo_retraido = True
            self.ts_codigo_brazo_retraido = time.monotonic()
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
        - 'RETRAER':  envía -desplazamiento_brazo_mm (garantizando apagar la bomba antes)
        """
        msg_brazo = Float32()
        if accion == 'EXTENDER':
            msg_brazo.data = +self.desplazamiento_brazo_mm
            self.brazo_extendido = True
            self.get_logger().info(
                f"[BRAZO] Extendiendo brazo (+{self.desplazamiento_brazo_mm / 10.0:.1f} cm)..."
            )
        elif accion == 'RETRAER':
            if self.bomba_activa:
                self.comandar_bomba(False)
            msg_brazo.data = -self.desplazamiento_brazo_mm
            self.brazo_extendido = False
            self.get_logger().info(
                f"[BRAZO] Retrayendo brazo (-{self.desplazamiento_brazo_mm / 10.0:.1f} cm)..."
            )
        else:
            if self.bomba_activa:
                self.comandar_bomba(False)
            msg_brazo.data = 0.0
            self.brazo_extendido = False

        self.brazo_pub.publish(msg_brazo)

    def comandar_bomba(self, encender: bool):
        """
        Envía orden a la bomba (/bomba -> START_BOMBA=7 / STOP_BOMBA=8 en STM32).
        - Si `self.usar_bomba` es False, mantiene la bomba apagada siempre.
        - Si `self.usar_bomba` es True, solo permite encenderla si y solo si la STM32 ya
          confirmó que el brazo está totalmente extendido (`self.stm32_brazo_extendido == True`, código 4).
        """
        if encender:
            if not self.usar_bomba:
                return
            if not (self.brazo_extendido and self.stm32_brazo_extendido):
                self.get_logger().warn(
                    "[BOMBA] Bloqueo de seguridad: no se puede activar la bomba si el brazo no está completamente extendido (código 4)."
                )
                return
            self.bomba_activa = True
            msg_bomba = Bool()
            msg_bomba.data = True
            self.bomba_pub.publish(msg_bomba)
            self.get_logger().info("[BOMBA] Bomba ACTIVADA (START_Bomba).")
        else:
            if self.bomba_activa or self.usar_bomba:
                self.bomba_activa = False
                msg_bomba = Bool()
                msg_bomba.data = False
                self.bomba_pub.publish(msg_bomba)
                self.get_logger().info("[BOMBA] Bomba DETENIDA (STOP_Bomba).")

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
        """Envía STOP_All, apaga la bomba y fija velocidad 0.0 m/s."""
        self.bomba_activa = False
        msg_bomba = Bool()
        msg_bomba.data = False
        self.bomba_pub.publish(msg_bomba)

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
        """Congela los tiempos transcurridos durante una pausa por obstáculo o watchdog."""
        self.tiempo_transcurrido_guardado = now - self.tiempo_inicio_estado
        self.tiempo_sin_ver_arbol_guardado = now - self.tiempo_ultimo_arbol

    def _restaurar_relojes(self, now: float):
        """Restaura los tiempos transcurridos al reanudarse la operación."""
        self.tiempo_inicio_estado = now - self.tiempo_transcurrido_guardado
        self.tiempo_ultimo_arbol = now - self.tiempo_sin_ver_arbol_guardado

    def _finalizar_rutina_con_reporte(self, motivo: str):
        """
        Detiene el robot, imprime el reporte completo en consola
        y activa la secuencia de cierre limpio del nodo.
        """
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
                f" Modo de operación   : MAPIR={'SI' if self.usar_mapir else 'NO'} | "
                f"BOMBA={'SI' if self.usar_bomba else 'NO'}\n"
                f" Tiempo de rutina    : {self.tiempo_rutina_total:.2f} s ({minutos}m {segundos:04.1f}s)\n"
                f" Velocidad promedio  : {self.velocidad_promedio:.3f} m/s "
                f"(distancia est.: {self.distancia_recorrida:.2f} m)\n"
                f" Árboles encontrados : {self.arboles_procesados} / {self.meta_arboles}\n"
                f" Malezas encontradas : {self.arboles_con_maleza}\n"
                "============================================================"
            )
            self.get_logger().info(reporte)

    def _actualizar_y_publicar_rumbo(self, now: float, permitir_correccion_lateral: bool):
        """
        Calcula y publica el rumbo objetivo hacia la STM32 por /navegacion/corregir_grados:
        - Si `permitir_correccion_lateral` es True y hay un tronco guía reciente (< 0.5 s),
          aplica control PID (Proporcional + Integral + Derivativo) + ángulo mínimo efectivo
          (`min_grados_rumbo`) para vencer la fricción de las 4 ruedas y amortiguar sobrepasos.
        - Evalúa la magnitud del ángulo a corregir (`angulo_correccion`): si supera
          `umbral_angulo_lento` (18° por defecto), activa `self.en_correccion_lenta = True`
          para bajar la velocidad lineal a `velocidad_correccion` (0.065 m/s) hasta que el
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

        # Lógica de separación de casos de velocidad según ángulo de corrección
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

        # Limitador de pendiente (slew-rate)
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

    def _finalizar_o_continuar_siguiente_arbol(self, now: float, motivo_log: str):
        """
        Decide si finalizar la rutina con reporte (cuando ya se alcanzó `meta_arboles`) o
        pasar a `ENFRIAMIENTO` para avanzar hacia el siguiente árbol.
        """
        if self.arboles_procesados >= self.meta_arboles:
            self.get_logger().info(
                f"Meta alcanzada ({self.arboles_procesados}/{self.meta_arboles} árboles inspeccionados, "
                f"{self.arboles_con_maleza} con maleza). {motivo_log} Robot detenido por completo."
            )
            self._finalizar_rutina_con_reporte(
                motivo=f"Meta alcanzada ({self.arboles_procesados}/{self.meta_arboles} árboles)"
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

        # Si la rutina ya terminó, mantener todo apagado con STOP_All y cerrar el nodo
        if self.estado_actual == 'FINALIZADO':
            self.detener_todo_emergencia()
            self.ciclos_cierre += 1
            if self.ciclos_cierre >= 4:
                self.control_timer.cancel()
                raise SystemExit(0)
            return

        # Verificación de Watchdogs de Sensores
        detenido_en_arbol = self.estado_actual in (
            'CAL_EXTENDIENDO_BRAZO',
            'CAL_BOMBA_ACTIVA',
            'CAL_RETRAYENDO_BRAZO',
            'VERIFICANDO_MALEZA',
            'EXTENDIENDO_BRAZO',
            'RETRAYENDO_BRAZO',
        )
        lidar_ok = (self.ultimo_lidar_ts is not None and (now - self.ultimo_lidar_ts) <= self.lidar_timeout)
        tfmini_ok = (
            (not self.usar_tfmini)
            or (self.ultimo_tfmini_ts is not None and (now - self.ultimo_tfmini_ts) <= self.tfmini_timeout)
        )

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
            if self.inicio_rutina_ts is None:
                self.inicio_rutina_ts = now
            self._restaurar_relojes(now)
            self.en_pausa_por_watchdog = False
            self.get_logger().info(
                f"Sensores ok. Iniciando/reanudando rutina "
                f"(MAPIR={'SI' if self.usar_mapir else 'NO'}, BOMBA={'SI' if self.usar_bomba else 'NO'})"
            )

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
        # Máquina de Estados:
        # Fase 0: Calibración inicial de bomba al encender (solo si usar_bomba=True)
        # Fase 1: Navegación + Inspección MAPIR (si usar_mapir=True) + Barrido (con o sin Bomba)
        # ---------------------------------------------------------------------
        if self.estado_actual == 'CAL_EXTENDIENDO_BRAZO':
            msg_vel.linear.x = 0.0

            # Si al iniciar se detectó que el brazo ya estaba extendido (< 4.5 s en llegar a END),
            # primero lo retraemos a HOME (código 3) para que la STM32 reinicie screw_step en INTR_HOME()
            # y luego lo volvemos a extender el recorrido completo (> 390 mm).
            if self.rehoming_en_calibracion:
                if not self.orden_retraer_enviada and tiempo_transcurrido >= 0.50:
                    self.stm32_brazo_retraido = False
                    self.comandar_brazo('RETRAER')
                    self.orden_retraer_enviada = True
                elif self.orden_retraer_enviada and self.stm32_brazo_retraido:
                    t_desde_home = (
                        (now - self.ts_codigo_brazo_retraido)
                        if self.ts_codigo_brazo_retraido is not None
                        else 1.0
                    )
                    if t_desde_home >= 0.55:
                        self.rehoming_en_calibracion = False
                        self.orden_extender_enviada = False
                        self.stm32_brazo_extendido = False
                        self.tiempo_inicio_estado = now
                        self.get_logger().info(
                            "[CALIBRACIÓN BOMBA] Brazo sincronizado en HOME (código 3). "
                            "Re-extendiendo recorrido completo para habilitar bomba en STM32..."
                        )
                return

            if not self.orden_extender_enviada and tiempo_transcurrido >= 0.50:
                self.stm32_brazo_extendido = False
                self.comandar_brazo('EXTENDER')
                self.orden_extender_enviada = True
                self.ts_orden_extender = now
                self.get_logger().info(
                    f"[CALIBRACIÓN BOMBA] Extendiendo brazo +{self.desplazamiento_brazo_mm:.0f}mm "
                    "al encender el robot (esperando código 4 de STM32)..."
                )

            if self.orden_extender_enviada and self.stm32_brazo_extendido:
                duracion_ext = (
                    (self.ts_codigo_brazo_extendido - self.ts_orden_extender)
                    if (self.ts_codigo_brazo_extendido is not None and self.ts_orden_extender is not None)
                    else 10.0
                )
                # Un recorrido completo de ~400 mm a 40 mm/s tarda ~10 s. Si llegó a END en < 4.5 s,
                # el brazo no partió desde HOME (ej. rutina anterior abortada) y screw_mm en la STM32
                # puede ser < 390 mm (MARGEN_MM en bomba.c), lo que bloquearía la bomba.
                if duracion_ext < 4.5:
                    self.get_logger().warn(
                        f"[CALIBRACIÓN BOMBA] El brazo tocó END en solo {duracion_ext:.1f}s "
                        "(ya estaba extendido al iniciar, por lo que screw_mm < 390mm en STM32). "
                        "Retrayendo a HOME primero para sincronizar posición..."
                    )
                    self.rehoming_en_calibracion = True
                    self.orden_retraer_enviada = False
                    self.stm32_brazo_retraido = False
                    self.stm32_brazo_extendido = False
                    self.tiempo_inicio_estado = now
                    return

                # Activar bomba de inmediato al recibir código 4 y aplicar micro-avance de +8 mm
                # tras el retroceso de -10 mm de INTR_END() para asegurar screw_mm > 390 mm en bomba.c
                if not self.bomba_activa:
                    self.comandar_bomba(True)

                t_desde_end = (
                    (now - self.ts_codigo_brazo_extendido)
                    if self.ts_codigo_brazo_extendido is not None
                    else 1.0
                )
                if not self.micro_avance_end_enviado and t_desde_end >= 0.45:
                    msg_micro = Float32()
                    msg_micro.data = 8.0  # +8 mm (+0.8 cm) recupera screw_mm > 390 mm sin tocar switch END
                    self.brazo_pub.publish(msg_micro)
                    self.micro_avance_end_enviado = True
                    self.comandar_bomba(True)
                elif self.micro_avance_end_enviado and t_desde_end >= 0.80:
                    self.comandar_bomba(True)
                    self.estado_actual = 'CAL_BOMBA_ACTIVA'
                    self.tiempo_inicio_estado = now
                    self.get_logger().info(
                        f"[CALIBRACIÓN BOMBA] Brazo extendido confirmado (código 4). "
                        f"Activando bomba por {self.tiempo_calibracion_bomba:.1f}s para llenar la manguera..."
                    )

        elif self.estado_actual == 'CAL_BOMBA_ACTIVA':
            msg_vel.linear.x = 0.0

            if tiempo_transcurrido >= self.tiempo_calibracion_bomba:
                self.comandar_bomba(False)
                self.stm32_brazo_retraido = False
                self.orden_retraer_enviada = False
                self.estado_actual = 'CAL_RETRAYENDO_BRAZO'
                self.tiempo_inicio_estado = now
                self.get_logger().info(
                    f"[CALIBRACIÓN BOMBA] Cebado de {self.tiempo_calibracion_bomba:.1f}s completado. "
                    f"Bomba apagada. Retrayendo brazo -{self.desplazamiento_brazo_mm:.0f}mm..."
                )

        elif self.estado_actual == 'CAL_RETRAYENDO_BRAZO':
            msg_vel.linear.x = 0.0

            if not self.orden_retraer_enviada and tiempo_transcurrido >= 0.15:
                self.comandar_brazo('RETRAER')
                self.orden_retraer_enviada = True

            if self.orden_retraer_enviada and self.stm32_brazo_retraido and not self.brazo_extendido:
                # Esperar 0.50 s tras código 3 para que INTR_HOME() termine MOVE_Stepper(+10 mm)
                t_desde_home = (
                    (now - self.ts_codigo_brazo_retraido)
                    if self.ts_codigo_brazo_retraido is not None
                    else 1.0
                )
                if t_desde_home >= 0.50:
                    self.brazo_retraido_confirmado = True
                    self.calibracion_bomba_completada = True
                    self.estado_actual = 'AVANZANDO'
                    self.tiempo_inicio_estado = now
                    self.tiempo_ultimo_arbol = now
                    self.get_logger().info(
                        "[CALIBRACIÓN BOMBA] Brazo retraído confirmado (código 3). "
                        "Calibración de bomba finalizada. Iniciando rutina de búsqueda de árboles..."
                    )

        elif self.estado_actual == 'AVANZANDO':
            # Garantía de seguridad: solo avanzar si el brazo está retraído y la bomba apagada
            if not self.brazo_retraido_confirmado or self.brazo_extendido or self.bomba_activa:
                msg_vel.linear.x = 0.0
                self.cmd_pub.publish(msg_vel)
                return

            self._actualizar_y_publicar_rumbo(now, permitir_correccion_lateral=True)
            msg_vel.linear.x = (
                self.velocidad_correccion if self.en_correccion_lenta else self.velocidad_avance
            )

            # Criterio de parada por timeout sin detectar árboles
            if tiempo_sin_ver_arbol >= self.timeout_sin_arbol:
                self.get_logger().error(
                    f"TIMEOUT: {tiempo_sin_ver_arbol:.1f}s sin detectar ningún árbol. "
                    "Deteniendo robot y generando reporte final."
                )
                self._finalizar_rutina_con_reporte(
                    motivo=f"Timeout ({tiempo_sin_ver_arbol:.1f}s sin detectar árboles)"
                )
                return

            # Detección de tronco confirmada a la altura del LiDAR
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

                if self.usar_mapir:
                    # Con MAPIR activa: detenerse y verificar durante 2s si hay maleza antes de mover el brazo
                    self.maleza_detectada_en_parada = False
                    self.max_cobertura_en_parada = 0.0
                    self.estado_actual = 'VERIFICANDO_MALEZA'
                    self.get_logger().info(
                        f"Tronco {self.arboles_procesados}/{self.meta_arboles} detectado. "
                        f"Enviando STOP_Mov inmediatamente y verificando maleza con cámara MAPIR "
                        f"({self.tiempo_verificacion_maleza:.1f}s)..."
                    )
                else:
                    # Sin MAPIR: pasar directamente a extender el brazo para hacer el barrido
                    self.arboles_con_maleza += 1
                    self.orden_extender_enviada = False
                    self.orden_retraer_enviada = False
                    self.stm32_brazo_extendido = False
                    self.stm32_brazo_retraido = False
                    self.brazo_retraido_confirmado = False
                    self.estado_actual = 'EXTENDIENDO_BRAZO'
                    self.get_logger().info(
                        f"Tronco {self.arboles_procesados}/{self.meta_arboles} detectado (modo sin MAPIR). "
                        f"Enviando STOP_Mov inmediatamente y extendiendo brazo +{self.desplazamiento_brazo_mm:.0f}mm..."
                    )
                return

        elif self.estado_actual == 'VERIFICANDO_MALEZA':
            msg_vel.linear.x = 0.0

            if tiempo_transcurrido >= self.tiempo_estabilizacion_camara and self.maleza_actual_topic:
                self.maleza_detectada_en_parada = True

            # CASO A: La MAPIR confirma presencia de maleza -> Activar secuencia de BARRIDO del brazo
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
                    f"Extendiendo brazo +{self.desplazamiento_brazo_mm:.0f}mm para iniciar barrido "
                    f"({'CON bomba' if self.usar_bomba else 'SIN bomba'})..."
                )

            # CASO B: Terminó el tiempo de inspección y NO hay maleza -> NO activar brazo ni bomba y continuar
            elif tiempo_transcurrido >= self.tiempo_verificacion_maleza:
                self.get_logger().info(
                    f"[MAPIR] SIN maleza en tronco {self.arboles_procesados}/{self.meta_arboles}. "
                    "NO se acciona el brazo ni la bomba."
                )
                if self._finalizar_o_continuar_siguiente_arbol(
                    now, motivo_log="Sin maleza (barrido omitido)."
                ):
                    return

        elif self.estado_actual == 'EXTENDIENDO_BRAZO':
            msg_vel.linear.x = 0.0

            # Garantizar que hayan pasado al menos 0.50 s desde el último código 3 (INTR_HOME)
            # para que drv8825.c no esté ocupado en MOVE_Stepper(BACK_MARGIN) (state == RUN)
            home_listo = (
                self.ts_codigo_brazo_retraido is None
                or (now - self.ts_codigo_brazo_retraido) >= 0.50
            )
            if not self.orden_extender_enviada and tiempo_transcurrido >= 0.15 and home_listo:
                self.stm32_brazo_extendido = False
                self.comandar_brazo('EXTENDER')
                self.orden_extender_enviada = True

            # Cuando la STM32 confirma código 4 (brazo extendido), esperamos 0.45 s a que termine
            # MOVE_Stepper(-10 mm) de INTR_END(), compensamos +8 mm para asegurar screw_mm > 390 mm
            # en bomba.c, encendemos la bomba con las ruedas aún detenidas (0.35 s para superar
            # el pulso WAKE de 200 ms de la bomba sin competir con el arranque de los 4 motores)
            # y luego iniciamos BARRIENDO_BRAZO.
            if self.orden_extender_enviada and self.stm32_brazo_extendido:
                if self.usar_bomba:
                    if not self.bomba_activa:
                        self.comandar_bomba(True)
                    t_desde_end = (
                        (now - self.ts_codigo_brazo_extendido)
                        if self.ts_codigo_brazo_extendido is not None
                        else 1.0
                    )
                    if not self.micro_avance_end_enviado and t_desde_end >= 0.45:
                        msg_micro = Float32()
                        msg_micro.data = 8.0  # +8 mm (+0.8 cm) recupera screw_mm > 390 mm en bomba.c
                        self.brazo_pub.publish(msg_micro)
                        self.micro_avance_end_enviado = True
                        self.comandar_bomba(True)
                    elif self.micro_avance_end_enviado and t_desde_end >= 0.85:
                        self.comandar_bomba(True)
                        self.estado_actual = 'BARRIENDO_BRAZO'
                        self.tiempo_inicio_estado = now
                        self.get_logger().info(
                            f"STM32 confirmó brazo extendido (código 4). "
                            f"Bomba ACTIVADA e iniciando barrido durante {self.tiempo_barrido_brazo:.1f}s..."
                        )
                else:
                    self.estado_actual = 'BARRIENDO_BRAZO'
                    self.tiempo_inicio_estado = now
                    self.get_logger().info(
                        f"STM32 confirmó brazo extendido (código 4). "
                        f"Iniciando (sin bomba) barrido durante {self.tiempo_barrido_brazo:.1f}s..."
                    )

        elif self.estado_actual == 'BARRIENDO_BRAZO':
            # Avanza en línea recta con el brazo extendido (y bomba activa si usar_bomba=True) durante `tiempo_barrido_brazo`
            if tiempo_transcurrido < self.tiempo_barrido_brazo:
                msg_vel.linear.x = self.velocidad_avance
                self._actualizar_y_publicar_rumbo(now, permitir_correccion_lateral=False)
            else:
                msg_vel.linear.x = 0.0
                if self.usar_bomba or self.bomba_activa:
                    self.comandar_bomba(False)
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
                    f"{'Bomba DETENIDA, ' if self.usar_bomba else ''}enviando STOP_Mov y retrayendo brazo "
                    f"-{self.desplazamiento_brazo_mm:.0f}mm (esperando confirmación código 3 de STM32)..."
                )
                return

        elif self.estado_actual == 'RETRAYENDO_BRAZO':
            msg_vel.linear.x = 0.0

            if not self.orden_retraer_enviada and tiempo_transcurrido >= 0.10:
                self.comandar_brazo('RETRAER')
                self.orden_retraer_enviada = True

            if self.orden_retraer_enviada and self.stm32_brazo_retraido and not self.brazo_extendido:
                self.brazo_retraido_confirmado = True
                if self._finalizar_o_continuar_siguiente_arbol(
                    now, motivo_log="Barrido completado y brazo retraído (código 3 STM32)."
                ):
                    return

        elif self.estado_actual == 'ENFRIAMIENTO':
            if not self.brazo_retraido_confirmado or self.brazo_extendido or self.bomba_activa:
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

        # Actualizar tiempo de rutina y velocidad promedio para el reporte final
        if self.inicio_rutina_ts is not None and not self.rutina_finalizada:
            self.tiempo_rutina_total += self.dt
            self.distancia_recorrida += abs(float(msg_vel.linear.x)) * self.dt
            if self.tiempo_rutina_total > 1e-6:
                self.velocidad_promedio = self.distancia_recorrida / self.tiempo_rutina_total


def main(args=None):
    rclpy.init(args=args)
    node = BombaRutinaNavegacion()
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

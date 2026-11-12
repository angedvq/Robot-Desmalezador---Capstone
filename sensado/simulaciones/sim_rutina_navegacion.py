#!/usr/bin/env python3
"""
===============================================================================
NODO DE SIMULACIÓN: sim_rutina_navegacion.py
===============================================================================
Ubicación: `sensado/simulaciones/sim_rutina_navegacion.py`
Equivalente en simulación de `real_rutina_navegacion.py`.

PROPÓSITO:
Ejecutar en Gazebo Fortress exactamente la misma máquina de estados autónoma
que `real_rutina_navegacion.py`:
1. `AVANZANDO`:
   - Avanza paralelo a la hilera de árboles a 0.70 m de distancia objetivo.
   - Mientras un árbol próximo es detectado en la zona de guiado amplio
     (`/lidar/distancia_tronco_lateral`), calcula una corrección suave de rumbo
     (`SET_HeadingRef`) hacia los 0.70 m.
   - En los claros entre árboles, el rumbo objetivo vuelve suavemente a `0.0°` y
     el robot mantiene la línea recta apoyándose en la IMU.
   - Si pasan más de `timeout_sin_arbol` segundos (20.0 s) sin detectar ningún
     tronco, pasa a `FINALIZADO` y detiene todo.
2. `ALINEANDO_BRAZO` (Delay de 1.0 s en movimiento):
   - Al confirmarse un tronco en el costado del LiDAR (esquina superior
     izquierda, `x = +0.15 m`), el robot sigue avanzando recto en `0.0°` durante
     `1.0 s` (`0.20 m` extra de recorrido) para que el brazo trasero
     (`x = -0.17 m`) quede exactamente frente al tronco.
3. `EXTENDIENDO_BRAZO`:
   - Detiene las ruedas (`STOP_Mov` en `sim_stm32_bridge`) y envía `+200 mm`
     (`+20 cm`) a `/brazo_mm` para desplegar el brazo trasero hacia el tronco
     durante `tiempo_brazo_extendido` (4.0 s).
4. `RETRAYENDO_BRAZO`:
   - Envía `-200 mm` (`-20 cm`) a `/brazo_mm` para retraer el brazo trasero y
     espera `tiempo_brazo_retraido` (3.0 s) con el robot detenido.
   - Al completar el 4° árbol (`meta_arboles = 4`), pasa a `FINALIZADO`.
5. `ENFRIAMIENTO`:
   - Avanza recto durante `3.0 s` ignorando el árbol recién procesado antes de
     volver a `AVANZANDO`.
6. `FINALIZADO`:
   - Detiene todo el sistema con `STOP_All` (`/navegacion/stop = True`).

CÓMO EMULA EL CONTROL DE RUMBO DE LA STM32 + IMU EN GAZEBO:
En el robot real, la STM32 recibe `SET_HeadingRef` (en grados) y usa su PID
interno con la IMU física para ajustar las ruedas. En Gazebo, este nodo lee el
sensor IMU simulado (`/imu/data`, con respaldo en `/odom`) y aplica ese mismo
lazo proporcional sobre `cmd_vel.angular.z` para seguir `rumbo_suavizado`.
===============================================================================
"""

import math
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseArray
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, LaserScan
from std_msgs.msg import Bool, Float32


def _yaw_desde_cuaternion(q) -> float:
    """Extrae el ángulo yaw (rad) a partir de un cuaternión `geometry_msgs/Quaternion`."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class SimRutinaNavegacion(Node):
    """
    Controlador de navegación autónoma en hilera de árboles y accionamiento de
    brazo trasero para simulación en Gazebo Fortress.
    """

    def __init__(self):
        super().__init__('sim_rutina_navegacion')

        # ---------------------------------------------------------------------
        # 1. Parámetros configurables (idénticos a `real_rutina_navegacion.py`)
        # ---------------------------------------------------------------------
        self.declare_parameter('distancia_meta', 0.70)
        self.declare_parameter('distancia_frenado', 0.50)
        self.declare_parameter('velocidad_avance', 0.20)

        self.declare_parameter('kp_rumbo', 38.0)
        self.declare_parameter('max_grados_rumbo', 12.0)
        self.declare_parameter('slew_rate_deg_s', 8.0)
        self.declare_parameter('alpha_dist', 0.40)
        self.declare_parameter('kp_yaw_imu', 2.6)

        self.declare_parameter('meta_arboles', 4)
        self.declare_parameter('timeout_sin_arbol', 20.0)
        self.declare_parameter('tiempo_delay_brazo', 1.0)
        self.declare_parameter('tiempo_brazo_extendido', 4.0)
        self.declare_parameter('tiempo_brazo_retraido', 3.0)
        self.declare_parameter('tiempo_enfriamiento', 3.0)
        self.declare_parameter('desplazamiento_brazo_mm', 200.0)

        self.declare_parameter('usar_tfmini', True)
        self.declare_parameter('lidar_timeout', 1.5)
        self.declare_parameter('tfmini_timeout', 1.5)

        self.distancia_meta = float(self.get_parameter('distancia_meta').value)
        self.distancia_frenado = float(self.get_parameter('distancia_frenado').value)
        self.velocidad_avance = float(self.get_parameter('velocidad_avance').value)
        self.kp_rumbo = float(self.get_parameter('kp_rumbo').value)
        self.max_grados_rumbo = float(self.get_parameter('max_grados_rumbo').value)
        self.slew_rate_deg_s = float(self.get_parameter('slew_rate_deg_s').value)
        self.alpha_dist = float(self.get_parameter('alpha_dist').value)
        self.kp_yaw_imu = float(self.get_parameter('kp_yaw_imu').value)

        self.meta_arboles = int(self.get_parameter('meta_arboles').value)
        self.timeout_sin_arbol = float(self.get_parameter('timeout_sin_arbol').value)
        self.tiempo_delay_brazo = float(self.get_parameter('tiempo_delay_brazo').value)
        self.tiempo_brazo_extendido = float(self.get_parameter('tiempo_brazo_extendido').value)
        self.tiempo_brazo_retraido = float(self.get_parameter('tiempo_brazo_retraido').value)
        self.tiempo_enfriamiento = float(self.get_parameter('tiempo_enfriamiento').value)
        self.desplazamiento_brazo_mm = abs(
            float(self.get_parameter('desplazamiento_brazo_mm').value)
        )

        self.usar_tfmini = bool(self.get_parameter('usar_tfmini').value)
        self.lidar_timeout = float(self.get_parameter('lidar_timeout').value)
        self.tfmini_timeout = float(self.get_parameter('tfmini_timeout').value)

        # ---------------------------------------------------------------------
        # 2. Publicadores hacia Gazebo y `sim_stm32_bridge`
        # ---------------------------------------------------------------------
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.stop_pub = self.create_publisher(Bool, '/navegacion/stop', 10)
        self.grados_pub = self.create_publisher(Float32, '/navegacion/corregir_grados', 10)
        self.brazo_pub = self.create_publisher(Float32, '/brazo_mm', 10)

        # ---------------------------------------------------------------------
        # 3. Suscriptores de percepción, TF-Mini e IMU simulada
        # ---------------------------------------------------------------------
        self.tree_sub = self.create_subscription(
            PoseArray, '/arboles_detectados', self.tree_callback, 10
        )
        self.dist_sub = self.create_subscription(
            Float32, '/lidar/distancia_tronco_lateral', self.dist_tronco_callback, 10
        )
        self.obs_sub = self.create_subscription(
            Bool, '/obstaculo_frontal', self.obs_callback, 10
        )
        self.tfmini_sub = self.create_subscription(
            LaserScan, '/tf_mini/scan', self.tfmini_callback, 10
        )
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10
        )
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10
        )

        # ---------------------------------------------------------------------
        # 4. Variables de estado
        # ---------------------------------------------------------------------
        self.estado_actual = 'AVANZANDO'
        self.arboles_procesados = 0
        self.arbol_en_mira = False
        self.brazo_extendido = False

        self.distancia_tronco_actual = None
        self.ultimo_tronco_guia_ts = None
        self.rumbo_suavizado = 0.0
        self.ultimo_rumbo_enviado = 0.0
        self.ultimo_envio_grados_ts = 0.0

        # Yaw medido por la IMU simulada (referenciado a 0.0 rad al iniciar)
        self.yaw_actual = 0.0
        self.yaw_referencia_inicial = None
        self.tiene_imu = False

        self.ultimo_lidar_ts = None
        self.ultimo_tfmini_ts = None
        self.obstaculo_topic = False
        self.obstaculo_scan = False
        self.en_pausa_por_watchdog = True
        self.en_pausa_por_obstaculo = False

        now = time.monotonic()
        self.tiempo_inicio_estado = now
        self.tiempo_ultimo_arbol = now
        self.tiempo_transcurrido_guardado = 0.0
        self.tiempo_sin_ver_arbol_guardado = 0.0

        self.dt = 0.05
        self.control_timer = self.create_timer(self.dt, self.control_loop)

        self.get_logger().info(
            f"sim_rutina_navegacion iniciado | Meta: {self.meta_arboles} árboles | "
            f"Distancia hilera: {self.distancia_meta:.2f}m | Delay brazo: {self.tiempo_delay_brazo:.1f}s | "
            f"Timeout sin árbol: {self.timeout_sin_arbol:.0f}s"
        )

    # =========================================================================
    # Callbacks de sensores
    # =========================================================================
    def imu_callback(self, msg: Imu):
        """
        Lee la orientación de la IMU simulada en Gazebo y fija la referencia
        `0.0°` en el primer mensaje recibido (igual que `START_Mov` en la STM32).
        """
        yaw_raw = _yaw_desde_cuaternion(msg.orientation)
        if self.yaw_referencia_inicial is None:
            self.yaw_referencia_inicial = yaw_raw
        diff = yaw_raw - self.yaw_referencia_inicial
        self.yaw_actual = math.atan2(math.sin(diff), math.cos(diff))
        self.tiene_imu = True

    def odom_callback(self, msg: Odometry):
        """Respaldo de orientación yaw si el tópico IMU aún no ha publicado."""
        if not self.tiene_imu:
            self.yaw_actual = _yaw_desde_cuaternion(msg.pose.pose.orientation)

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
            else:
                self.distancia_tronco_actual = (
                    self.alpha_dist * nueva
                    + (1.0 - self.alpha_dist) * self.distancia_tronco_actual
                )
            self.ultimo_tronco_guia_ts = now
        else:
            self.distancia_tronco_actual = None

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
    # Funciones auxiliares de actuación y control de rumbo
    # =========================================================================
    def comandar_brazo(self, accion: str):
        msg_brazo = Float32()
        if accion == 'EXTENDER':
            msg_brazo.data = +self.desplazamiento_brazo_mm
            self.brazo_extendido = True
            self.get_logger().info(
                f"[BRAZO-SIM] Extendiendo brazo trasero (+{self.desplazamiento_brazo_mm / 10.0:.1f} cm)..."
            )
        elif accion == 'RETRAER':
            msg_brazo.data = -self.desplazamiento_brazo_mm
            self.brazo_extendido = False
            self.get_logger().info(
                f"[BRAZO-SIM] Retrayendo brazo trasero (-{self.desplazamiento_brazo_mm / 10.0:.1f} cm)..."
            )
        else:
            msg_brazo.data = 0.0
            self.brazo_extendido = False

        self.brazo_pub.publish(msg_brazo)

    def detener_todo_emergencia(self):
        msg_stop = Bool()
        msg_stop.data = True
        self.stop_pub.publish(msg_stop)

        msg_vel = Twist()
        msg_vel.linear.x = 0.0
        msg_vel.angular.z = 0.0
        self.cmd_pub.publish(msg_vel)

        self.rumbo_suavizado = 0.0

    def _pausar_relojes(self, now: float):
        self.tiempo_transcurrido_guardado = now - self.tiempo_inicio_estado
        self.tiempo_sin_ver_arbol_guardado = now - self.tiempo_ultimo_arbol

    def _restaurar_relojes(self, now: float):
        self.tiempo_inicio_estado = now - self.tiempo_transcurrido_guardado
        self.tiempo_ultimo_arbol = now - self.tiempo_sin_ver_arbol_guardado

    def _actualizar_y_publicar_rumbo(self, now: float, permitir_correccion_lateral: bool) -> float:
        """
        Actualiza `self.rumbo_suavizado` (grados), lo publica hacia
        `sim_stm32_bridge` y retorna la velocidad angular `angular.z` (rad/s)
        calculada con el yaw de la IMU simulada para las ruedas de Gazebo.
        """
        tronco_guia_vigente = (
            permitir_correccion_lateral
            and self.distancia_tronco_actual is not None
            and self.ultimo_tronco_guia_ts is not None
            and (now - self.ultimo_tronco_guia_ts) <= 0.5
        )

        if tronco_guia_vigente:
            error = self.distancia_tronco_actual - self.distancia_meta
            if abs(error) < 0.03:
                rumbo_objetivo = 0.0
            else:
                rumbo_objetivo = max(
                    -self.max_grados_rumbo,
                    min(self.max_grados_rumbo, error * self.kp_rumbo),
                )
        else:
            rumbo_objetivo = 0.0

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
                ref_str = (
                    f"Árbol próximo a {self.distancia_tronco_actual:.2f}m"
                    if tronco_guia_vigente
                    else "Claro IMU 0°"
                )
                self.get_logger().info(
                    f"[RUMBO-SIM] {ref_str} -> SET_Heading: {self.rumbo_suavizado:.1f}° "
                    f"(obj: {rumbo_objetivo:.1f}°)"
                )
            self.ultimo_rumbo_enviado = self.rumbo_suavizado
            self.ultimo_envio_grados_ts = now

        # Lazo de seguimiento de rumbo usando el yaw de la IMU (emula el PID de la STM32)
        rumbo_rad = math.radians(self.rumbo_suavizado)
        error_yaw = math.atan2(
            math.sin(rumbo_rad - self.yaw_actual),
            math.cos(rumbo_rad - self.yaw_actual),
        )
        return max(-0.8, min(0.8, self.kp_yaw_imu * error_yaw))

    # =========================================================================
    # Bucle Principal de Control (20 Hz)
    # =========================================================================
    def control_loop(self):
        now = time.monotonic()

        if self.estado_actual == 'FINALIZADO':
            self.detener_todo_emergencia()
            return

        # 1. Watchdogs de sensores
        lidar_ok = (
            self.ultimo_lidar_ts is not None
            and (now - self.ultimo_lidar_ts) <= self.lidar_timeout
        )
        tfmini_ok = (
            (not self.usar_tfmini)
            or (
                self.ultimo_tfmini_ts is not None
                and (now - self.ultimo_tfmini_ts) <= self.tfmini_timeout
            )
        )

        if not lidar_ok or not tfmini_ok:
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
                f"Watchdog simulación: esperando {', '.join(faltantes)}...",
                throttle_duration_sec=2.0,
            )
            return
        elif self.en_pausa_por_watchdog:
            self._restaurar_relojes(now)
            self.en_pausa_por_watchdog = False
            self.get_logger().info(
                "Sensores de simulación activos. Iniciando rutina autónoma de árboles."
            )

        # 2. Detección de obstáculo frontal (TF-Mini < 0.50 m)
        hay_obstaculo = self.usar_tfmini and (self.obstaculo_topic or self.obstaculo_scan)
        if hay_obstaculo:
            if not self.en_pausa_por_obstaculo:
                self._pausar_relojes(now)
                self.en_pausa_por_obstaculo = True
            self.detener_todo_emergencia()
            self.get_logger().warn(
                f"Obstáculo frontal (< {self.distancia_frenado:.2f}m). "
                "Rutina en pausa con STOP_All. Esperando despeje...",
                throttle_duration_sec=2.0,
            )
            return
        elif self.en_pausa_por_obstaculo:
            self._restaurar_relojes(now)
            self.en_pausa_por_obstaculo = False
            self.get_logger().info("Obstáculo despejado. Reanudando rutina...")
            if self.estado_actual == 'EXTENDIENDO_BRAZO':
                self.comandar_brazo('EXTENDER')
            elif self.estado_actual == 'RETRAYENDO_BRAZO':
                self.comandar_brazo('RETRAER')

        msg_stop = Bool()
        msg_stop.data = False
        self.stop_pub.publish(msg_stop)

        tiempo_transcurrido = now - self.tiempo_inicio_estado
        tiempo_sin_ver_arbol = now - self.tiempo_ultimo_arbol

        msg_vel = Twist()

        # 3. Máquina de estados idéntica a `real_rutina_navegacion.py`
        if self.estado_actual == 'AVANZANDO':
            msg_vel.linear.x = self.velocidad_avance
            msg_vel.angular.z = self._actualizar_y_publicar_rumbo(
                now, permitir_correccion_lateral=True
            )

            if tiempo_sin_ver_arbol >= self.timeout_sin_arbol:
                self.estado_actual = 'FINALIZADO'
                self.detener_todo_emergencia()
                self.get_logger().error(
                    f"TIMEOUT: {tiempo_sin_ver_arbol:.1f}s sin detectar árboles. "
                    "Rutina finalizada con STOP_All."
                )
                return

            if self.arbol_en_mira:
                self.arboles_procesados += 1
                self.tiempo_inicio_estado = now
                self.tiempo_ultimo_arbol = now
                self.estado_actual = 'ALINEANDO_BRAZO'
                self.get_logger().info(
                    f"¡Árbol lateral {self.arboles_procesados}/{self.meta_arboles} detectado! "
                    f"Avanzando {self.tiempo_delay_brazo:.1f}s extra para alinear el brazo trasero..."
                )

        elif self.estado_actual == 'ALINEANDO_BRAZO':
            msg_vel.linear.x = self.velocidad_avance
            msg_vel.angular.z = self._actualizar_y_publicar_rumbo(
                now, permitir_correccion_lateral=False
            )

            if tiempo_transcurrido >= self.tiempo_delay_brazo:
                msg_vel.linear.x = 0.0
                msg_vel.angular.z = 0.0
                self.estado_actual = 'EXTENDIENDO_BRAZO'
                self.tiempo_inicio_estado = now
                self.get_logger().info(
                    f"Brazo trasero alineado con árbol {self.arboles_procesados}. "
                    f"Deteniendo ruedas (STOP_Mov) y extendiendo brazo por {self.tiempo_brazo_extendido:.1f}s..."
                )
                self.comandar_brazo('EXTENDER')

        elif self.estado_actual == 'EXTENDIENDO_BRAZO':
            msg_vel.linear.x = 0.0
            msg_vel.angular.z = 0.0

            if tiempo_transcurrido >= self.tiempo_brazo_extendido:
                self.estado_actual = 'RETRAYENDO_BRAZO'
                self.tiempo_inicio_estado = now
                self.comandar_brazo('RETRAER')
                self.get_logger().info(
                    f"Esperando {self.tiempo_brazo_retraido:.1f}s para retracción del brazo..."
                )

        elif self.estado_actual == 'RETRAYENDO_BRAZO':
            msg_vel.linear.x = 0.0
            msg_vel.angular.z = 0.0

            if tiempo_transcurrido >= self.tiempo_brazo_retraido:
                if self.arboles_procesados >= self.meta_arboles:
                    self.estado_actual = 'FINALIZADO'
                    self.detener_todo_emergencia()
                    self.get_logger().info(
                        f"¡Meta alcanzada ({self.arboles_procesados}/{self.meta_arboles} árboles)! "
                        "Brazo retraído y robot detenido por completo (STOP_All)."
                    )
                    return
                else:
                    self.estado_actual = 'ENFRIAMIENTO'
                    self.tiempo_inicio_estado = now
                    self.tiempo_ultimo_arbol = now
                    self.get_logger().info(
                        f"Árbol {self.arboles_procesados}/{self.meta_arboles} completado. "
                        f"Reanudando marcha (enfriamiento de {self.tiempo_enfriamiento:.1f}s)..."
                    )

        elif self.estado_actual == 'ENFRIAMIENTO':
            msg_vel.linear.x = self.velocidad_avance
            msg_vel.angular.z = self._actualizar_y_publicar_rumbo(
                now, permitir_correccion_lateral=False
            )

            if tiempo_transcurrido >= self.tiempo_enfriamiento:
                self.estado_actual = 'AVANZANDO'
                self.tiempo_inicio_estado = now
                self.tiempo_ultimo_arbol = now
                self.get_logger().info(
                    f"Buscando árbol {self.arboles_procesados + 1}/{self.meta_arboles} "
                    f"(timeout: {self.timeout_sin_arbol:.0f}s)..."
                )

        self.cmd_pub.publish(msg_vel)


def main(args=None):
    rclpy.init(args=args)
    node = SimRutinaNavegacion()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.detener_todo_emergencia()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
import math
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Float32


class RealControlPared(Node):
    def __init__(self):
        super().__init__('real_control_pared')

        # Parámetros de seguimiento suave de pared + evasión rápida de pilares/salientes
        self.declare_parameter('distancia_meta', 0.70)
        self.declare_parameter('distancia_frenado', 0.50)
        self.declare_parameter('velocidad_avance', 0.20)
        self.declare_parameter('kp', 35.0)
        self.declare_parameter('kp_evasion', 75.0)
        self.declare_parameter('max_grados', 12.0)
        self.declare_parameter('max_grados_evasion', 100.0)
        self.declare_parameter('slew_rate_deg_s', 6.0)
        self.declare_parameter('slew_evasion_deg_s', 30.0)
        self.declare_parameter('umbral_evasion', 0.12)
        self.declare_parameter('alpha_dist', 0.35)
        self.declare_parameter('usar_tfmini', True)
        self.declare_parameter('lidar_timeout', 0.8)
        self.declare_parameter('tfmini_timeout', 0.8)

        self.distancia_meta = float(self.get_parameter('distancia_meta').value)
        self.distancia_frenado = float(self.get_parameter('distancia_frenado').value)
        self.velocidad_avance = float(self.get_parameter('velocidad_avance').value)
        self.kp = float(self.get_parameter('kp').value)
        self.kp_evasion = float(self.get_parameter('kp_evasion').value)
        self.max_grados = float(self.get_parameter('max_grados').value)
        self.max_grados_evasion = float(self.get_parameter('max_grados_evasion').value)
        self.slew_rate_deg_s = float(self.get_parameter('slew_rate_deg_s').value)
        self.slew_evasion_deg_s = float(self.get_parameter('slew_evasion_deg_s').value)
        self.umbral_evasion = float(self.get_parameter('umbral_evasion').value)
        self.alpha_dist = float(self.get_parameter('alpha_dist').value)
        self.usar_tfmini = bool(self.get_parameter('usar_tfmini').value)
        self.lidar_timeout = float(self.get_parameter('lidar_timeout').value)
        self.tfmini_timeout = float(self.get_parameter('tfmini_timeout').value)

        # Suscriptores de sensores reales
        self.sub_dist = self.create_subscription(
            Float32, '/lidar/distancia_lateral', self.dist_callback, 10
        )
        self.sub_obs = self.create_subscription(
            Bool, '/obstaculo_frontal', self.obs_callback, 10
        )
        self.sub_tfmini = self.create_subscription(
            LaserScan, '/tf_mini/scan', self.tfmini_callback, 10
        )

        # Publicadores hacia stm32_bridge
        self.pub_vel = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_stop = self.create_publisher(Bool, '/navegacion/stop', 10)
        self.pub_grados = self.create_publisher(Float32, '/navegacion/corregir_grados', 10)

        # El rumbo por defecto arranca siempre en 0.0°
        self.distancia_actual = None
        self.rumbo_suavizado = 0.0
        self.ultimo_rumbo_enviado = 0.0
        self.ultimo_envio_grados_ts = 0.0

        self.obstaculo_topic = False
        self.obstaculo_scan = False
        self.en_pausa_por_obstaculo = False
        self.en_pausa_por_watchdog = True

        self.ultimo_lidar_ts = None
        self.ultimo_tfmini_ts = None

        # Bucle de control a 20 Hz (dt = 0.05 s)
        self.dt = 0.05
        self.timer = self.create_timer(self.dt, self.control_loop)

        self.get_logger().info(
            f"real_control_pared iniciado | meta={self.distancia_meta:.2f}m, "
            f"suave=(kp={self.kp:.0f}, max={self.max_grados:.0f}°), "
            f"evasion=(kp={self.kp_evasion:.0f}, max={self.max_grados_evasion:.0f}°)"
        )

    def dist_callback(self, msg):
        if math.isfinite(msg.data) and msg.data > 0.0:
            nueva = float(msg.data)
            if self.distancia_actual is None:
                self.distancia_actual = nueva
            elif nueva < (self.distancia_actual - 0.08):
                # Caída repentina de distancia (ej. pilar o escalón): respuesta rápida sin rezago
                self.distancia_actual = 0.80 * nueva + 0.20 * self.distancia_actual
            else:
                # Seguimiento normal / retorno a la pared: filtrado suave
                self.distancia_actual = (
                    self.alpha_dist * nueva + (1.0 - self.alpha_dist) * self.distancia_actual
                )
            self.ultimo_lidar_ts = time.monotonic()

    def obs_callback(self, msg):
        self.obstaculo_topic = bool(msg.data)

    def tfmini_callback(self, msg):
        self.ultimo_tfmini_ts = time.monotonic()
        detectado = False
        for r in msg.ranges:
            if math.isnan(r) or (math.isinf(r) and r > 0.0):
                continue
            if r < self.distancia_frenado:
                detectado = True
                break
        self.obstaculo_scan = detectado

    def detener_robot(self):
        msg_stop_flag = Bool()
        msg_stop_flag.data = True
        self.pub_stop.publish(msg_stop_flag)

        msg_stop_vel = Twist()
        msg_stop_vel.linear.x = 0.0
        msg_stop_vel.angular.z = 0.0
        self.pub_vel.publish(msg_stop_vel)

        self.rumbo_suavizado = 0.0

    def control_loop(self):
        now = time.monotonic()

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
            self.en_pausa_por_watchdog = True
            self.detener_robot()
            faltantes = []
            if not lidar_ok:
                faltantes.append('LiDAR lateral')
            if not tfmini_ok:
                faltantes.append('TF-Mini frontal')
            self.get_logger().warn(
                f"Watchdog sensores: sin datos vigentes de {', '.join(faltantes)}. STOP enviado.",
                throttle_duration_sec=2.0
            )
            return
        elif self.en_pausa_por_watchdog:
            self.en_pausa_por_watchdog = False
            self.get_logger().info("Sensores operativos. Control autónomo habilitado (rumbo base 0.0°).")

        # 2. Detección de obstáculo frontal -> STOP_All inmediato
        if self.usar_tfmini and (self.obstaculo_topic or self.obstaculo_scan):
            if not self.en_pausa_por_obstaculo:
                self.en_pausa_por_obstaculo = True
            self.detener_robot()
            self.get_logger().warn(
                f"Obstáculo frontal (< {self.distancia_frenado:.2f}m). Comando STOP_All activo...",
                throttle_duration_sec=2.0
            )
            return
        elif self.en_pausa_por_obstaculo:
            self.en_pausa_por_obstaculo = False
            self.get_logger().info("Ruta despejada. Reanudando movimiento...")

        msg_run = Bool()
        msg_run.data = False
        self.pub_stop.publish(msg_run)

        # 3. Cálculo asimétrico de rumbo:
        # - Seguimiento normal / retorno a la pared: suave (kp=35, máx +12°)
        # - Pilar / intrusión cercana (error < -umbral_evasion): evasión agresiva (kp=75, máx -25°)
        error = self.distancia_actual - self.distancia_meta
        en_evasion = error < -self.umbral_evasion

        if abs(error) < 0.03:
            rumbo_objetivo = 0.0
        elif en_evasion:
            rumbo_objetivo = max(-self.max_grados_evasion, error * self.kp_evasion)
        else:
            rumbo_objetivo = max(-self.max_grados, min(self.max_grados, error * self.kp))

        # 4. Rampa asimétrica: rápida al abrirse para esquivar un pilar, suave al volver a la pared
        abriendose_por_peligro = en_evasion and (rumbo_objetivo < self.rumbo_suavizado)
        tasa_deg_s = self.slew_evasion_deg_s if abriendose_por_peligro else self.slew_rate_deg_s
        max_paso = tasa_deg_s * self.dt

        delta = rumbo_objetivo - self.rumbo_suavizado
        if abs(delta) <= max_paso:
            self.rumbo_suavizado = rumbo_objetivo
        else:
            self.rumbo_suavizado += math.copysign(max_paso, delta)

        # 5. Publicar SET_HeadingRef
        diferencia_angular = abs(self.rumbo_suavizado - self.ultimo_rumbo_enviado)
        retorno_a_cero = (self.rumbo_suavizado == 0.0 and self.ultimo_rumbo_enviado != 0.0)
        requiere_refresco = (now - self.ultimo_envio_grados_ts) >= 1.0

        if diferencia_angular >= 0.5 or retorno_a_cero or requiere_refresco:
            msg_grados = Float32()
            msg_grados.data = float(self.rumbo_suavizado)
            self.pub_grados.publish(msg_grados)

            if diferencia_angular >= 0.5 or retorno_a_cero:
                modo = "EVASION" if abriendose_por_peligro else "SUAVE"
                self.get_logger().info(
                    f"[{modo}] Dist: {self.distancia_actual:.2f}m -> SET_Heading: {self.rumbo_suavizado:.1f}° "
                    f"(obj: {rumbo_objetivo:.1f}°)"
                )
            self.ultimo_rumbo_enviado = self.rumbo_suavizado
            self.ultimo_envio_grados_ts = now

        # 6. Enviar velocidad lineal constante
        msg_vel = Twist()
        msg_vel.linear.x = self.velocidad_avance
        self.pub_vel.publish(msg_vel)


def main(args=None):
    rclpy.init(args=args)
    node = RealControlPared()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.detener_robot()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

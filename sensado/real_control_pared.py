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

        # Parámetros ajustables por launch / CLI (SSH)
        self.declare_parameter('distancia_meta', 0.50)
        self.declare_parameter('distancia_frenado', 0.50)
        self.declare_parameter('velocidad_avance', 0.20)
        self.declare_parameter('kp', 100.0)
        self.declare_parameter('max_grados', 20.0)
        self.declare_parameter('lidar_timeout', 0.8)
        self.declare_parameter('tfmini_timeout', 0.8)

        self.distancia_meta = float(self.get_parameter('distancia_meta').value)
        self.distancia_frenado = float(self.get_parameter('distancia_frenado').value)
        self.velocidad_avance = float(self.get_parameter('velocidad_avance').value)
        self.kp = float(self.get_parameter('kp').value)
        self.max_grados = float(self.get_parameter('max_grados').value)
        self.lidar_timeout = float(self.get_parameter('lidar_timeout').value)
        self.tfmini_timeout = float(self.get_parameter('tfmini_timeout').value)

        # Suscriptores de sensores reales (sin dependencia de /odom de Gazebo)
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
        self.pub_grados = self.create_publisher(Float32, '/navegacion/corregir_grados', 10)

        self.distancia_actual = self.distancia_meta
        self.ultimo_rumbo_enviado = 0.0
        self.ultimo_envio_grados_ts = 0.0

        self.obstaculo_topic = False
        self.obstaculo_scan = False
        self.en_pausa_por_obstaculo = False
        self.en_pausa_por_watchdog = True

        # Marcas de tiempo monotónicas para watchdogs de sensores
        self.ultimo_lidar_ts = None
        self.ultimo_tfmini_ts = None

        # Bucle de control a 20 Hz
        self.timer = self.create_timer(0.05, self.control_loop)

        self.get_logger().info(
            f"real_control_pared iniciado | meta={self.distancia_meta:.2f}m, "
            f"vel={self.velocidad_avance:.2f}m/s, frenado={self.distancia_frenado:.2f}m"
        )

    def dist_callback(self, msg):
        if math.isfinite(msg.data) and msg.data > 0.0:
            self.distancia_actual = float(msg.data)
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
        msg_stop = Twist()
        msg_stop.linear.x = 0.0
        msg_stop.angular.z = 0.0
        self.pub_vel.publish(msg_stop)

    def control_loop(self):
        now = time.monotonic()

        # 1. Watchdogs de sensores: detener si falta LiDAR lateral o TF-Mini frontal
        lidar_ok = (
            self.ultimo_lidar_ts is not None
            and (now - self.ultimo_lidar_ts) <= self.lidar_timeout
        )
        tfmini_ok = (
            self.ultimo_tfmini_ts is not None
            and (now - self.ultimo_tfmini_ts) <= self.tfmini_timeout
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
                f"Watchdog sensores: sin datos vigentes de {', '.join(faltantes)}. Robot detenido.",
                throttle_duration_sec=2.0
            )
            return
        elif self.en_pausa_por_watchdog:
            self.en_pausa_por_watchdog = False
            self.get_logger().info("Sensores LiDAR y TF-Mini operativos. Control autónomo habilitado.")

        # 2. Detección de obstáculo frontal (< distancia_frenado o zona ciega)
        if self.obstaculo_topic or self.obstaculo_scan:
            if not self.en_pausa_por_obstaculo:
                self.en_pausa_por_obstaculo = True
            self.detener_robot()
            self.get_logger().warn(
                f"Obstáculo frontal a menos de {self.distancia_frenado:.2f}m. Robot detenido. Esperando despeje...",
                throttle_duration_sec=2.0
            )
            return
        elif self.en_pausa_por_obstaculo:
            self.en_pausa_por_obstaculo = False
            self.get_logger().info("Ruta despejada. Reanudando avance...")

        # 3. Error respecto a la pared IZQUIERDA:
        # actual < meta (muy cerca) -> error < 0 -> rumbo negativo (girar a la derecha)
        # actual > meta (muy lejos) -> error > 0 -> rumbo positivo (girar a la izquierda)
        error = self.distancia_actual - self.distancia_meta

        # Zona muerta de 2 cm para evitar oscilaciones por ruido del LiDAR
        if abs(error) < 0.02:
            rumbo_objetivo = 0.0
        else:
            rumbo_objetivo = error * self.kp
            rumbo_objetivo = max(-self.max_grados, min(self.max_grados, rumbo_objetivo))

        # 4. Enviar SET_Heading a STM32 (anti-saturación UART + refresco de seguridad cada 1s)
        diferencia_angular = abs(rumbo_objetivo - self.ultimo_rumbo_enviado)
        requiere_refresco = (now - self.ultimo_envio_grados_ts) >= 1.0

        if (
            diferencia_angular >= 1.0
            or (rumbo_objetivo == 0.0 and self.ultimo_rumbo_enviado != 0.0)
            or requiere_refresco
        ):
            msg_grados = Float32()
            msg_grados.data = float(rumbo_objetivo)
            self.pub_grados.publish(msg_grados)

            if diferencia_angular >= 1.0 or (rumbo_objetivo == 0.0 and self.ultimo_rumbo_enviado != 0.0):
                self.get_logger().info(
                    f"Dist: {self.distancia_actual:.2f}m -> SET_Heading: {rumbo_objetivo:.1f}°"
                )
            self.ultimo_rumbo_enviado = rumbo_objetivo
            self.ultimo_envio_grados_ts = now

        # 5. Enviar velocidad lineal de avance (stm32_bridge traduce linear.x a Orden 0)
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

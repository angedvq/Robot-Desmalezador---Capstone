#!/usr/bin/env python3
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TransformStamped
from std_msgs.msg import Bool
from tf2_ros import TransformBroadcaster


class Odometria(Node):
    def __init__(self):
        super().__init__('odometria')

        # --------------------------------------------------
        # Posición inicial y velocidad del robot
        # --------------------------------------------------
        self.x = 0.0
        self.velocidad = 0.0
        self.stop_activo = False
        self.t_anterior = time.monotonic()

        # --------------------------------------------------
        # Recibir velocidad ordenada por navegación y parada
        # --------------------------------------------------
        self.create_subscription(
            Twist, '/cmd_vel', self.recibir_velocidad, 10
        )
        self.create_subscription(
            Bool, '/navegacion/stop', self.recibir_stop, 10
        )

        # Publicador de transformación TF
        self.tf_broadcaster = TransformBroadcaster(self)

        # Actualizar odometría cada 0.05 segundos (20 Hz)
        self.create_timer(0.05, self.actualizar)

        self.get_logger().info('Nodo de odometría iniciado (odom -> base_footprint).')

    def recibir_velocidad(self, msg: Twist):
        self.velocidad = float(msg.linear.x)

    def recibir_stop(self, msg: Bool):
        self.stop_activo = bool(msg.data)
        if self.stop_activo:
            self.velocidad = 0.0

    def actualizar(self):
        # --------------------------------------------------
        # Calcular desplazamiento: d = v * t
        # --------------------------------------------------
        ahora = time.monotonic()
        dt = ahora - self.t_anterior
        self.t_anterior = ahora

        vel_efectiva = 0.0 if self.stop_activo else self.velocidad
        self.x += vel_efectiva * dt

        # --------------------------------------------------
        # Publicar transformación odom -> base_footprint
        # --------------------------------------------------
        tf = TransformStamped()

        tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id = 'odom'
        tf.child_frame_id = 'base_footprint'

        tf.transform.translation.x = float(self.x)
        tf.transform.translation.y = 0.0
        tf.transform.translation.z = 0.0

        # Orientación identidad (0 grados)
        tf.transform.rotation.x = 0.0
        tf.transform.rotation.y = 0.0
        tf.transform.rotation.z = 0.0
        tf.transform.rotation.w = 1.0

        self.tf_broadcaster.sendTransform(tf)


def main(args=None):
    rclpy.init(args=args)
    nodo = Odometria()

    try:
        rclpy.spin(nodo)
    except KeyboardInterrupt:
        pass
    finally:
        nodo.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

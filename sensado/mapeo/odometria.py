#!/usr/bin/env python3
import time
import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TransformStamped
from std_msgs.msg import Bool, Float32
from tf2_ros import TransformBroadcaster


class Odometria(Node):
    def __init__(self):
        super().__init__('odometria')

        # --------------------------------------------------
        # Posición y orientación inicial del robot
        # --------------------------------------------------
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

        self.yaw_referencia = 0.0
        self.rumbo_relativo = 0.0

        self.velocidad = 0.0
        self.stop_activo = False
        self.en_movimiento = False
        self.t_anterior = time.monotonic()

        # --------------------------------------------------
        # Suscripciones a las órdenes de navegación
        # --------------------------------------------------
        self.create_subscription(
            Twist, '/cmd_vel', self.recibir_velocidad, 10
        )
        self.create_subscription(
            Bool, '/navegacion/stop', self.recibir_stop, 10
        )
        self.create_subscription(
            Float32, '/navegacion/corregir_grados',
            self.recibir_rumbo, 10
        )

        # Publicador de transformación TF
        self.tf_broadcaster = TransformBroadcaster(self)

        # Actualización de odometría a 20 Hz
        self.create_timer(0.05, self.actualizar)

        self.get_logger().info('Odometría 2D iniciada.')

    def recibir_velocidad(self, msg: Twist):
        self.integrar_posicion()
        self.velocidad = float(msg.linear.x)
        self.actualizar_movimiento()

    def recibir_stop(self, msg: Bool):
        self.integrar_posicion()
        self.stop_activo = bool(msg.data)

        if self.stop_activo:
            self.velocidad = 0.0

        self.actualizar_movimiento()

    def recibir_rumbo(self, msg: Float32):
        self.integrar_posicion()
        self.rumbo_relativo = math.radians(float(msg.data))

        if self.en_movimiento:
            self.yaw = self.yaw_referencia + self.rumbo_relativo

    # --------------------------------------------------
    # Manejo de reinicios de referencia angular
    # --------------------------------------------------
    def actualizar_movimiento(self):
        movimiento = (
            not self.stop_activo
            and abs(self.velocidad) >= 1e-4
        )

        if self.en_movimiento and not movimiento:
            self.yaw_referencia = self.yaw
            self.rumbo_relativo = 0.0

        elif not self.en_movimiento and movimiento:
            self.yaw = self.yaw_referencia + self.rumbo_relativo

        self.en_movimiento = movimiento

    # --------------------------------------------------
    # Dead reckoning en dos dimensiones
    # --------------------------------------------------
    def integrar_posicion(self):
        ahora = time.monotonic()
        dt = ahora - self.t_anterior
        self.t_anterior = ahora

        if self.en_movimiento:
            self.x += self.velocidad * math.cos(self.yaw) * dt
            self.y += self.velocidad * math.sin(self.yaw) * dt

    def actualizar(self):
        self.integrar_posicion()

        # --------------------------------------------------
        # Publicar transformación odom -> base_footprint
        # --------------------------------------------------
        tf = TransformStamped()

        tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id = 'odom'
        tf.child_frame_id = 'base_footprint'

        tf.transform.translation.x = float(self.x)
        tf.transform.translation.y = float(self.y)
        tf.transform.translation.z = 0.0

        tf.transform.rotation.x = 0.0
        tf.transform.rotation.y = 0.0
        tf.transform.rotation.z = math.sin(self.yaw / 2.0)
        tf.transform.rotation.w = math.cos(self.yaw / 2.0)

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
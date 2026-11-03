#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32
import numpy as np


def calcular_distancia_pared(angles, ranges, range_min_msg, fov_min, fov_max, max_wall_range):
    """
    Calcula la distancia perpendicular desde el RPLidar A1 (esquina superior izquierda,
    motor/+X apuntando hacia atrás del robot) hasta la pared izquierda.
    Retorna float con la distancia en metros, o None si no hay suficientes puntos válidos.
    """
    angles = np.arctan2(np.sin(angles), np.cos(angles))
    min_r = max(float(range_min_msg), 0.15)

    valid = (
        (ranges >= min_r)
        & (ranges <= max_wall_range)
        & np.isfinite(ranges)
        & (angles >= fov_min)
        & (angles <= fov_max)
    )
    angles = angles[valid]
    ranges = ranges[valid]

    if len(ranges) < 5:
        return None

    # Filtrar saltos de fondo (puertas, huecos o fin de pared) quedándonos con el plano cercano
    r_ref = np.percentile(ranges, 20)
    near_mask = ranges <= (r_ref + 0.45)
    angles = angles[near_mask]
    ranges = ranges[near_mask]

    if len(ranges) < 5:
        return None

    # Marco del RPLidar A1 con motor (+X, 0°) apuntando hacia atrás del robot (-X_robot):
    # -90° (-Y_lidar) apunta directo hacia la pared izquierda (+Y_robot).
    X = ranges * np.cos(angles)
    Y = ranges * np.sin(angles)

    if np.std(X) > 1e-3:
        m, c = np.polyfit(X, Y, 1)
        # Refinar descartando puntos atípicos a más de 5 cm de la recta ajustada
        residuos = np.abs(m * X - Y + c) / np.sqrt(m**2 + 1.0)
        inliers = residuos < 0.05
        if np.count_nonzero(inliers) >= 5 and np.std(X[inliers]) > 1e-3:
            m, c = np.polyfit(X[inliers], Y[inliers], 1)
        return float(abs(c) / np.sqrt(m**2 + 1.0))

    return float(np.min(ranges))


class RealLidarParedLateral(Node):
    def __init__(self):
        super().__init__('real_lidar_pared_lateral')

        self.declare_parameter('fov_min_deg', -120.0)
        self.declare_parameter('fov_max_deg', -60.0)
        self.declare_parameter('max_wall_range', 2.5)

        self.fov_min = np.deg2rad(float(self.get_parameter('fov_min_deg').value))
        self.fov_max = np.deg2rad(float(self.get_parameter('fov_max_deg').value))
        self.max_wall_range = float(self.get_parameter('max_wall_range').value)

        self.subscription = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.dist_pub = self.create_publisher(Float32, '/lidar/distancia_lateral', 10)

        self.get_logger().info(
            "Nodo real_test_pared iniciado (RPLidar A1 esquina sup-izq, motor hacia atrás, pared a la izquierda)."
        )

    def scan_callback(self, msg):
        angles = msg.angle_min + np.arange(len(msg.ranges)) * msg.angle_increment
        ranges = np.array(msg.ranges, dtype=float)

        distancia_pared = calcular_distancia_pared(
            angles,
            ranges,
            msg.range_min,
            self.fov_min,
            self.fov_max,
            self.max_wall_range,
        )

        if distancia_pared is None:
            self.get_logger().warn(
                "Sin suficientes puntos válidos de pared izquierda en el rango lateral.",
                throttle_duration_sec=2.0
            )
            return

        msg_dist = Float32()
        msg_dist.data = distancia_pared
        self.dist_pub.publish(msg_dist)


def main(args=None):
    rclpy.init(args=args)
    node = RealLidarParedLateral()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

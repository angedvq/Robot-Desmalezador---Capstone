#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32
import numpy as np


def calcular_distancia_pared(
    angles,
    ranges,
    range_min_msg,
    fov_min,
    fov_max,
    max_wall_range,
    pillar_lookahead=0.95,
):
    """
    Calcula la distancia efectiva a la pared izquierda / pilares salientes.
    - En [-120°, -60°] estima la distancia perpendicular real a la pared.
    - En el corredor frontal-izquierdo (hasta pillar_lookahead m por delante de la esquina
      superior izquierda) detecta pilares o salientes antes de llegar a ellos y toma la
      menor holgura lateral para esquivarlos a tiempo.
    """
    angles = np.arctan2(np.sin(angles), np.cos(angles))
    min_r = max(float(range_min_msg), 0.15)

    valid_all = (ranges >= min_r) & (ranges <= max_wall_range) & np.isfinite(ranges)
    if not np.any(valid_all):
        return None

    ang_all = angles[valid_all]
    rng_all = ranges[valid_all]

    # Coordenadas en el marco del robot (LiDAR en esquina superior izquierda, motor/+X hacia atrás):
    # x_robot > 0: hacia adelante del robot | y_robot > 0: hacia la izquierda del robot
    x_robot = -rng_all * np.cos(ang_all)
    y_robot = -rng_all * np.sin(ang_all)

    # 1. Detección anticipada de pilares / salientes en el corredor frontal-izquierdo
    dist_pilar = None
    if pillar_lookahead > 0.0:
        corredor_adelante = (
            (x_robot >= 0.18)
            & (x_robot <= pillar_lookahead)
            & (y_robot >= -0.12)
            & (y_robot <= 0.58)
        )
        corredor_lateral = (
            (x_robot >= -0.15)
            & (x_robot < 0.18)
            & (y_robot >= 0.12)
            & (y_robot <= 0.58)
        )
        mask_pilar = corredor_adelante | corredor_lateral
        if np.count_nonzero(mask_pilar) >= 3:
            # 3er rayo más cercano para ignorar chispazos aislados sin perder pilares angostos
            holguras = np.sort(y_robot[mask_pilar])
            dist_pilar = float(max(0.05, holguras[2]))

    # 2. Distancia perpendicular a la pared en la ventana lateral [-120°, -60°]
    dist_pared = None
    lat_mask = (ang_all >= fov_min) & (ang_all <= fov_max)
    ang_lat = ang_all[lat_mask]
    rng_lat = rng_all[lat_mask]

    if len(rng_lat) >= 5:
        r_ref = np.percentile(rng_lat, 20)
        near_mask = rng_lat <= (r_ref + 0.45)
        ang_lat = ang_lat[near_mask]
        rng_lat = rng_lat[near_mask]

        if len(rng_lat) >= 5:
            X = rng_lat * np.cos(ang_lat)
            Y = rng_lat * np.sin(ang_lat)

            if np.std(X) > 1e-3:
                m, c = np.polyfit(X, Y, 1)
                residuos = np.abs(m * X - Y + c) / np.sqrt(m**2 + 1.0)
                inliers = residuos < 0.05
                if np.count_nonzero(inliers) >= 5 and np.std(X[inliers]) > 1e-3:
                    m, c = np.polyfit(X[inliers], Y[inliers], 1)
                dist_pared = float(abs(c) / np.sqrt(m**2 + 1.0))
            else:
                dist_pared = float(np.min(rng_lat))

    if dist_pared is not None and dist_pilar is not None:
        return min(dist_pared, dist_pilar)
    return dist_pared if dist_pared is not None else dist_pilar


class RealLidarParedLateral(Node):
    def __init__(self):
        super().__init__('real_lidar_pared_lateral')

        self.declare_parameter('fov_min_deg', -120.0)
        self.declare_parameter('fov_max_deg', -60.0)
        self.declare_parameter('max_wall_range', 3.0)
        self.declare_parameter('pillar_lookahead', 0.95)

        self.fov_min = np.deg2rad(float(self.get_parameter('fov_min_deg').value))
        self.fov_max = np.deg2rad(float(self.get_parameter('fov_max_deg').value))
        self.max_wall_range = float(self.get_parameter('max_wall_range').value)
        self.pillar_lookahead = float(self.get_parameter('pillar_lookahead').value)

        self.subscription = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.dist_pub = self.create_publisher(Float32, '/lidar/distancia_lateral', 10)

        self.get_logger().info(
            f"Nodo real_test_pared iniciado (pared izq + anticipación de pilares a {self.pillar_lookahead:.2f}m)."
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
            self.pillar_lookahead,
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

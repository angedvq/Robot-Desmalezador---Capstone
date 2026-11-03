#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32
import numpy as np

class LidarParedLateral(Node):
    def __init__(self):
        super().__init__('lidar_pared_lateral')
        
        self.subscription = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.dist_pub = self.create_publisher(Float32, '/lidar/distancia_lateral', 10)

        # El frente del LiDAR (0°) apunta hacia atrás del robot (opuesto al frente del robot).
        # Por lo tanto, la izquierda del robot (+90° del robot) corresponde a -90° en el LiDAR.
        self.fov_min = np.deg2rad(-120.0)
        self.fov_max = np.deg2rad(-60.0)

        self.get_logger().info("Nodo LiDAR lateral iniciado. Midiendo distancia perpendicular a la pared izquierda...")

    def scan_callback(self, msg):
        angles = msg.angle_min + np.arange(len(msg.ranges)) * msg.angle_increment
        # Normalizar ángulos a [-pi, pi] por si el driver publica en [0, 2*pi]
        angles = np.arctan2(np.sin(angles), np.cos(angles))
        ranges = np.array(msg.ranges, dtype=float)

        valid_ranges = (ranges > msg.range_min) & (ranges < msg.range_max) & np.isfinite(ranges)
        valid_angles = (angles >= self.fov_min) & (angles <= self.fov_max)
        
        valid = valid_ranges & valid_angles
        angles = angles[valid]
        ranges = ranges[valid]

        if len(ranges) >= 5:
            # Coordenadas cartesianas en el marco del LiDAR:
            # X: a lo largo de la pared (hacia atrás del robot)
            # Y: hacia la pared izquierda del robot (negativo en el marco del LiDAR)
            X = ranges * np.cos(angles)
            Y = ranges * np.sin(angles)
            
            # Ajuste lineal (Y = m * X + c) para obtener la distancia perpendicular real
            # desde el origen del sensor hasta la recta de la pared: d = |c| / sqrt(m^2 + 1)
            if np.std(X) > 1e-3:
                m, c = np.polyfit(X, Y, 1)
                distancia_pared = abs(c) / np.sqrt(m**2 + 1.0)
            else:
                distancia_pared = float(np.min(ranges))
            
            msg_dist = Float32()
            msg_dist.data = float(distancia_pared)
            self.dist_pub.publish(msg_dist)

def main(args=None):
    rclpy.init(args=args)
    node = LidarParedLateral()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

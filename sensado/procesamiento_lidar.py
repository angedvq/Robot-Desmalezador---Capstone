#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Float32
import math

class ProcesamientoLidar(Node):
    def __init__(self):
        super().__init__('procesador_lidar')
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        
        self.pub_bool = self.create_publisher(Bool, '/interno/tronco_geometrico', 10)
        self.pub_dist = self.create_publisher(Float32, '/interno/distancia_tronco', 10)
        
        self.get_logger().info("Nodo Lidar iniciado")

    def scan_callback(self, msg):
        puntos_validos = []
        
        for i, r in enumerate(msg.ranges):
            # Ignorar distancias infinitas, nulas, o fuera del rango de interes (mts)
            if math.isinf(r) or math.isnan(r) or r < 0.3 or r > 1.0: 
                continue
                
            # Calcular ángulo actual del punto
            angle = msg.angle_min + i * msg.angle_increment
            angle_norm = angle % (2 * math.pi)
            
            # FOV frontal
            if angle_norm >= 6.02 or angle_norm <= 0.26 :
                # Convertir a coordenadas cartesianas
                x = r * math.cos(angle)
                y = r * math.sin(angle)
                puntos_validos.append((x, y, r))
                
        msg_bool = Bool()
        msg_dist = Float32()
        msg_bool.data = False
        msg_dist.data = -1.0
        
        # Puntos mínimos para trazar el ancho 
        if len(puntos_validos) >= 3:
            # Extremos del grupo de puntos detectados en la ventana
            x_primero, y_primero, _ = puntos_validos[0]
            x_ultimo, y_ultimo, _ = puntos_validos[-1]
            
            # Ancho euclidiano del objeto
            ancho = math.sqrt((x_ultimo - x_primero)**2 + (y_ultimo - y_primero)**2)
            
            # Rango de ancho válido (mts)
            if 0.08 <= ancho <= 0.10: # actualmente medidas para la botella de prueba
                msg_bool.data = True
                # La distancia real al tronco será el punto más cercano de ese conjunto
                msg_dist.data = min([p[2] for p in puntos_validos])
                
        self.pub_bool.publish(msg_bool)
        self.pub_dist.publish(msg_dist)

def main(args=None):
    rclpy.init(args=args)
    nodo = ProcesamientoLidar()
    try:
        rclpy.spin(nodo)
    except KeyboardInterrupt:
        pass
    finally:
        nodo.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
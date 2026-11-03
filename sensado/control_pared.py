#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32

class ControlPared(Node):
    def __init__(self):
        super().__init__('control_pared')
        
        self.sub_dist = self.create_subscription(Float32, '/lidar/distancia_lateral', self.dist_callback, 10)
        
        self.pub_vel = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_grados = self.create_publisher(Float32, '/navegacion/corregir_grados', 10)
        
        # --- PARÁMETROS DEL CONTROLADOR ---
        self.distancia_meta = 0.50     
        self.velocidad_avance = 0.20   
        self.Kp = 100.0                
        
        self.distancia_actual = self.distancia_meta 
        self.ultimo_rumbo_enviado = 0.0 
        
        self.timer = self.create_timer(0.05, self.control_loop)
        self.get_logger().info("Control de pared iniciado. SET_Heading dinámico activado.")

    def dist_callback(self, msg):
        self.distancia_actual = msg.data

    def control_loop(self):
        # 1. Calcular el error (Asumiendo pared a la DERECHA)
        error = self.distancia_meta - self.distancia_actual
        
        # 2. Zona muerta (Deadzone) de 2 cm
        if abs(error) < 0.02:
            rumbo_objetivo = 0.0
        else:
            # Cálculo del SET_Heading
            rumbo_objetivo = error * self.Kp 
            # Saturar la corrección máxima a +/- 20 grados
            rumbo_objetivo = max(-20.0, min(20.0, rumbo_objetivo))
            
        # 3. Lógica Anti-Saturación UART (Hardware STM32)
        diferencia_angular = abs(rumbo_objetivo - self.ultimo_rumbo_enviado)
        
        if diferencia_angular >= 1.0 or (rumbo_objetivo == 0.0 and self.ultimo_rumbo_enviado != 0.0):
            msg_grados = Float32()
            msg_grados.data = float(rumbo_objetivo)
            self.pub_grados.publish(msg_grados)
            
            self.ultimo_rumbo_enviado = rumbo_objetivo
            self.get_logger().info(f"Dist: {self.distancia_actual:.2f}m -> SET_Heading: {rumbo_objetivo:.1f}°")

        # 4. Enviar el avance a la simulación 
        # (Esto reemplaza el trabajo de la STM32 dentro de Gazebo)
        msg_vel = Twist()
        msg_vel.linear.x = self.velocidad_avance
        msg_vel.angular.z = (rumbo_objetivo * 3.14159 / 180.0) * 0.5 
        self.pub_vel.publish(msg_vel)

def main(args=None):
    rclpy.init(args=args)
    node = ControlPared()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
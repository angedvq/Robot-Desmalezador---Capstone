#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Float32

class ControlParedSim(Node):
    def __init__(self):
        super().__init__('control_pared')
        
        self.sub_dist = self.create_subscription(Float32, '/lidar/distancia_lateral', self.dist_callback, 10)
        self.sub_odom = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.sub_obs = self.create_subscription(Bool, '/obstaculo_frontal', self.obs_callback, 10)
        self.sub_tfmini = self.create_subscription(LaserScan, '/tf_mini/scan', self.tfmini_callback, 10)
        
        self.pub_vel = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_grados = self.create_publisher(Float32, '/navegacion/corregir_grados', 10)
        
        # --- PARÁMETROS DEL CONTROLADOR ---
        self.distancia_meta = 0.50     # Distancia ideal a la pared en metros
        self.distancia_frenado = 0.50  # Distancia de frenado por obstáculo frontal en metros
        self.velocidad_avance = 0.20   # Velocidad base (m/s)
        self.Kp = 100.0                # Ganancia: cuántos grados gira por cada metro de error
        
        self.distancia_actual = self.distancia_meta 
        self.yaw_actual = 0.0
        self.ultimo_rumbo_enviado = 0.0 # Memoria para no saturar el UART

        self.obstaculo_topic = False
        self.obstaculo_scan = False
        self.en_pausa_por_obstaculo = False
        
        # Bucle de control (20 Hz)
        self.timer = self.create_timer(0.05, self.control_loop)
        
        self.get_logger().info("Control de pared iniciado. SET_Heading dinámico y detección frontal activados.")

    def dist_callback(self, msg):
        self.distancia_actual = msg.data

    def obs_callback(self, msg):
        self.obstaculo_topic = bool(msg.data)

    def tfmini_callback(self, msg):
        detectado = False
        for r in msg.ranges:
            if math.isinf(r) or math.isnan(r):
                continue
            if r < self.distancia_frenado:
                detectado = True
                break
        self.obstaculo_scan = detectado

    def odom_callback(self, msg):
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.yaw_actual = math.atan2(siny_cosp, cosy_cosp)

    def control_loop(self):
        # 0. Verificar obstáculo frontal a menos de 0.50 m
        obstaculo_detectado = self.obstaculo_topic or self.obstaculo_scan
        if obstaculo_detectado:
            if not self.en_pausa_por_obstaculo:
                self.en_pausa_por_obstaculo = True
            msg_stop = Twist()
            msg_stop.linear.x = 0.0
            msg_stop.angular.z = 0.0
            self.pub_vel.publish(msg_stop)
            self.get_logger().warn(
                'Obstáculo frontal a menos de 0.50m. Robot detenido. Esperando despeje de la ruta...',
                throttle_duration_sec=2.0
            )
            return
        else:
            if self.en_pausa_por_obstaculo:
                self.en_pausa_por_obstaculo = False
                self.get_logger().info('Ruta despejada. Reanudando avance...')

        # 1. Calcular el error (Asumiendo pared a la IZQUIERDA)
        # Si actual es 0.4 (muy cerca de la pared izquierda), el error es -0.1 (debe girar a la derecha / yaw negativo).
        error = self.distancia_actual - self.distancia_meta
        
        # 2. Zona muerta (Deadzone) de 2 cm para evitar oscilaciones nerviosas por ruido del LiDAR
        if abs(error) < 0.02:
            rumbo_objetivo = 0.0
        else:
            # 3. Cálculo del SET_Heading
            rumbo_objetivo = error * self.Kp 
            
            # Saturar la corrección máxima a +/- 20 grados para evitar giros violentos
            # El protocolo STM32 soporta +/- 180, pero para control lateral 20 es un límite seguro
            rumbo_objetivo = max(-20.0, min(20.0, rumbo_objetivo))

        # 4. Enviar el avance constante y seguimiento de SET_Heading en simulación.
        # (El puente STM32 toma linear.x cuando no hay comandos pendientes)
        rumbo_objetivo_rad = math.radians(rumbo_objetivo)
        error_yaw = rumbo_objetivo_rad - self.yaw_actual
        msg_vel = Twist()
        msg_vel.linear.x = self.velocidad_avance
        msg_vel.angular.z = max(-1.0, min(1.0, 2.0 * error_yaw))
        self.pub_vel.publish(msg_vel)
            
        # 5. Lógica Anti-Saturación UART
        # Solo publica (envía la orden a la STM32) si el ángulo pedido cambió en más de 1.0 grado
        # o si necesitamos volver exactamente a 0.0 (paralelo perfecto)
        diferencia_angular = abs(rumbo_objetivo - self.ultimo_rumbo_enviado)
        
        if diferencia_angular >= 1.0 or (rumbo_objetivo == 0.0 and self.ultimo_rumbo_enviado != 0.0):
            msg_grados = Float32()
            msg_grados.data = float(rumbo_objetivo)
            self.pub_grados.publish(msg_grados)
            
            self.ultimo_rumbo_enviado = rumbo_objetivo
            self.get_logger().info(f"Dist: {self.distancia_actual:.2f}m -> SET_Heading: {rumbo_objetivo:.1f}°")

def main(args=None):
    rclpy.init(args=args)
    node = ControlParedSim()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

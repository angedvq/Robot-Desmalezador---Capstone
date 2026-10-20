import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseArray
from std_msgs.msg import Bool

class ControlNavegacion(Node):
    def __init__(self):
        super().__init__('control_navegacion')
        
        # Publicador de velocidad
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Suscriptores
        self.tree_sub = self.create_subscription(PoseArray, '/arboles_detectados', self.tree_callback, 10)
        self.obs_sub = self.create_subscription(Bool, '/obstaculo_frontal', self.obs_callback, 10)

        # Variables de estado
        self.estado_actual = 'AVANZANDO'
        self.arboles_rociados = 0
        self.meta_arboles = 4
        self.velocidad_avance = 0.2
        self.obstaculo_detectado = False
        self.arbol_en_mira = False
        
        # Tiempos 
        self.tiempo_detencion = 5.0
        self.tiempo_enfriamiento = 3.0
        self.timeout_sin_arbol = 15.0
        
        self.tiempo_estado = self.get_clock().now()
        self.tiempo_ultimo_arbol = self.get_clock().now()

        # Pausar temporizadores
        self.en_pausa_por_obstaculo = False
        self.tiempo_transcurrido_guardado = 0.0
        self.tiempo_sin_ver_arbol_guardado = 0.0

        # Bucle de control a 20 Hz
        self.control_timer = self.create_timer(0.05, self.control_loop)

    def obs_callback(self, msg):
        self.obstaculo_detectado = msg.data

    def tree_callback(self, msg):
        # Si el arreglo de poses tiene al menos un elemento, hay un árbol confirmado enfrente
        self.arbol_en_mira = len(msg.poses) > 0

    def control_loop(self):
        msg = Twist()
        now = self.get_clock().now()

        # Deteccion de obstaculos 
        if self.obstaculo_detectado:
            if not self.en_pausa_por_obstaculo:
                # Acabamos de detectar un obstáculo: guardamos exactamente el tiempo activo que llevábamos
                self.tiempo_transcurrido_guardado = (now - self.tiempo_estado).nanoseconds / 1e9
                self.tiempo_sin_ver_arbol_guardado = (now - self.tiempo_ultimo_arbol).nanoseconds / 1e9
                self.en_pausa_por_obstaculo = True
                
            msg.linear.x = 0.0
            self.cmd_pub.publish(msg)
            self.get_logger().warn('Obstáculo a menos de 50cm. Rutina pausada. Esperando despeje de la ruta...', throttle_duration_sec=2.0)
            return 
        else:
            if self.en_pausa_por_obstaculo:
                # El obstáculo fue retirado: restauramos las marcas de tiempo restando lo que habíamos guardado
                self.tiempo_estado = now - rclpy.time.Duration(nanoseconds=int(self.tiempo_transcurrido_guardado * 1e9))
                self.tiempo_ultimo_arbol = now - rclpy.time.Duration(nanoseconds=int(self.tiempo_sin_ver_arbol_guardado * 1e9))
                self.en_pausa_por_obstaculo = False
                self.get_logger().warn('Ruta despejada. Reanudando...')

        # Cálculo de tiempos reales en movimiento
        tiempo_transcurrido = (now - self.tiempo_estado).nanoseconds / 1e9
        tiempo_sin_ver_arbol = (now - self.tiempo_ultimo_arbol).nanoseconds / 1e9

        # Máquina de estados
        if self.estado_actual == 'AVANZANDO':
            msg.linear.x = self.velocidad_avance
            
            if tiempo_sin_ver_arbol >= self.timeout_sin_arbol:
                self.estado_actual = 'FINALIZADO'
                self.get_logger().error(f'TIMEOUT: {tiempo_sin_ver_arbol:.1f} segundos sin detección. Rutina abortada.')
                
            elif self.arbol_en_mira:
                self.arboles_rociados += 1
                self.tiempo_estado = now
                self.tiempo_ultimo_arbol = now
                
                self.estado_actual = 'DETENIDO'
                self.get_logger().info(f'Árbol {self.arboles_rociados} detectado. Rociando por {self.tiempo_detencion}s...')
            
        elif self.estado_actual == 'DETENIDO':
            msg.linear.x = 0.0
            if tiempo_transcurrido >= self.tiempo_detencion:
                # Al terminar el rociado, comprobamos si era el último árbol 
                if self.arboles_rociados >= self.meta_arboles:
                    self.estado_actual = 'FINALIZADO'
                    self.get_logger().info(f'{self.meta_arboles}/{self.meta_arboles} árboles detectados y rociados. Rutina finalizada.')
                else:
                    self.estado_actual = 'ENFRIAMIENTO'
                    self.tiempo_estado = now
                    self.get_logger().info('Reanudando...')
                
        elif self.estado_actual == 'ENFRIAMIENTO':
            # Durante el enfriamiento ignoramos las detecciones del nodo de percepción
            msg.linear.x = self.velocidad_avance
            if tiempo_transcurrido >= self.tiempo_enfriamiento:
                self.estado_actual = 'AVANZANDO'
                self.tiempo_ultimo_arbol = now 
                
        elif self.estado_actual == 'FINALIZADO':
            msg.linear.x = 0.0

        self.cmd_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = ControlNavegacion()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()

if __name__ == '__main__':
    main()
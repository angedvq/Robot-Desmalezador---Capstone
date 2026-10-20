import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import serial
import time

class TFMiniDriver(Node):
    def __init__(self):
        super().__init__('tfmini_driver')
        self.pub = self.create_publisher(LaserScan, '/tf_mini/scan', 10)
        
        # Puerto UART nativo de la Raspberry Pi
        self.port_name = '/dev/serial0'
        self.baud_rate = 115200
        
        try:
            # Iniciar comunicación serial 
            self.ser = serial.Serial(self.port_name, self.baud_rate, timeout=0.1)
            self.get_logger().info(f"Conectado exitosamente al TF-Mini en {self.port_name}")
        except Exception as e:
            self.get_logger().error(f"Fallo al abrir puerto serial: {e}")
            return
            
        # leer datos a 20Hz para ROS 2
        self.timer = self.create_timer(0.05, self.read_and_publish) 

    def read_and_publish(self):
        if not self.ser.is_open:
            return

        self.ser.reset_input_buffer()
        time.sleep(0.01) 
        
        if self.ser.in_waiting >= 9:
            # El TF-Mini siempre envía un paquete de 9 bytes que empieza por 0x59 0x59
            buf = self.ser.read(9)
            
            if len(buf) == 9 and buf[0] == 0x59 and buf[1] == 0x59:
                distancia_cm = buf[2] + (buf[3] << 8)
                distancia_m = distancia_cm / 100.0
                
                # Valores extremos significan que no detecta nada o hay error
                if distancia_m <= 0.02 or distancia_m >= 12.0:
                    distancia_m = float('inf')

                # Armar el mensaje LaserScan con 1 solo rayo para que 'procesamiento_tfmini' lo lea igual que Gazebo
                scan = LaserScan()
                scan.header.stamp = self.get_clock().now().to_msg()
                scan.header.frame_id = 'tfmini_link'
                scan.angle_min = 0.0
                scan.angle_max = 0.0
                scan.angle_increment = 0.0
                scan.range_min = 0.05
                scan.range_max = 12.0
                scan.ranges = [distancia_m]
                
                self.pub.publish(scan)

def main(args=None):
    rclpy.init(args=args)
    node = TFMiniDriver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()

if __name__ == '__main__':
    main()
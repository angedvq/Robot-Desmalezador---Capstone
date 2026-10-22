import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool
import numpy as np

class TFMiniDetector(Node):
    def __init__(self):
        super().__init__('tf_mini_detector')
        self.sub = self.create_subscription(LaserScan, '/tf_mini/scan', self.scan_callback, 10)
        self.pub = self.create_publisher(Bool, '/obstaculo_frontal', 10)
        
        self.distancia_frenado = 0.5 

    def scan_callback(self, msg):
        obstaculo = Bool()
        obstaculo.data = False
        
        for r in msg.ranges:

            # Ignoramos +infinito (espacio completamente despejado) y NaNs
            if np.isposinf(r) or np.isnan(r):
                continue
                
            # Si hay un obstáculo a menos de 50cm o el sensor está cegado (-inf)
            if r < self.distancia_frenado:
                obstaculo.data = True
                
                break
            
        if obstaculo.data:
            print(f"Obstáculo a {self.distancia_frenado} cms")
        else:
            print(False)
            
        self.pub.publish(obstaculo)

def main(args=None):
    rclpy.init(args=args)
    node = TFMiniDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()

if __name__ == '__main__':
    main()
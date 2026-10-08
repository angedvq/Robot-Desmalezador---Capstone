#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class MapirPublisher(Node):
    def __init__(self):
        super().__init__('mapir_publisher')
        self.publisher_ = self.create_publisher(Image, '/image_raw', 10)
        self.bridge = CvBridge()
        
        self.puerto_video = 2
        self.cap = cv2.VideoCapture(self.puerto_video, cv2.CAP_V4L2)
        
        # Forzar formato MJPG y resolución nativa 
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        if not self.cap.isOpened():
            self.get_logger().error(f"No se pudo abrir /dev/video{self.puerto_video}")
            return

        self.get_logger().info("Publisher MAPIR iniciado")
        
        # Temporizador a ~30 FPS 
        self.timer = self.create_timer(0.033, self.timer_callback)

    def timer_callback(self):
        ret, frame = self.cap.read()
        if ret:
            # Convertir la matriz de OpenCV a un mensaje de ros2
            msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
            self.publisher_.publish(msg)
        else:
            self.get_logger().warn("Fallo al capturar el frame")

def main(args=None):
    rclpy.init(args=args)
    nodo = MapirPublisher()
    try:
        rclpy.spin(nodo)
    except KeyboardInterrupt:
        pass
    finally:
        nodo.cap.release()
        nodo.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
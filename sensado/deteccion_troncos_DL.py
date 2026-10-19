import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import torch

# MMDetection
from mmdet.apis import init_detector, inference_detector
from mmdet.registry import VISUALIZERS

class RTTrunkNode(Node):
    def __init__(self):
        super().__init__('rt_trunk_detector')
        self.bridge = CvBridge()
        
        # Rutas a tu configuración y pesos recién entrenados
        config_file = '/home/verine/Capstone/ros2_ws/src/sensado/mmdetection/work_dirs/mi_entrenamiento/mi_entrenamiento.py'
        checkpoint_file = '/home/verine/Capstone/ros2_ws/src/sensado/mmdetection/work_dirs/mi_entrenamiento/iter_10000.pth'
        
        self.get_logger().info("Cargando modelo RT-Trunk en la RTX 2060. Esto puede tardar unos segundos...")
        # Cargar el modelo a la GPU
        self.model = init_detector(config_file, checkpoint_file, device='cuda:0')
        
        # Configurar el visualizador para dibujar los polígonos
        self.visualizer = VISUALIZERS.build(self.model.cfg.visualizer)
        self.visualizer.dataset_meta = self.model.dataset_meta
        
        # 1. SUSCRIPTOR: Escucha la cámara RealSense (ajusta el tópico si tu cámara usa otro)
        self.subscription = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.image_callback,
            10)
            
        # 2. PUBLICADOR: Emite el video con los troncos detectados
        self.publisher = self.create_publisher(Image, '/sensado/tree_masks', 10)
        self.get_logger().info("¡Modelo listo! Esperando imágenes de la RealSense...")

    def image_callback(self, msg):
        # Convertir mensaje ROS a imagen de OpenCV
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        # Inferencia de Inteligencia Artificial
        result = inference_detector(self.model, img)
        
        # Dibujar los polígonos sobre la imagen
        self.visualizer.add_datasample(
            'realsense',
            img,
            data_sample=result,
            draw_gt=False,
            show=False,
            wait_time=0
        )
        annotated_img = self.visualizer.get_image()
        
        # Convertir de vuelta a mensaje ROS y publicar
        img_msg = self.bridge.cv2_to_imgmsg(annotated_img, encoding='bgr8')
        self.publisher.publish(img_msg)

def main(args=None):
    rclpy.init(args=args)
    node = RTTrunkNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
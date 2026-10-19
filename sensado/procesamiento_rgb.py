#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge
import cv2
import numpy as np

class ProcesadorRGB(Node):
    def __init__(self):
        super().__init__('procesador_rgb')
        # Suscripción a la webcam
        self.sub = self.create_subscription(Image, '/image_raw', self.image_callback, 10)
        # Publicador del veredicto cromático
        self.pub_bool = self.create_publisher(Bool, '/interno/tronco_color', 10)
        
        self.bridge = CvBridge()
        self.get_logger().info("Nodo procesador_rgb iniciado. Buscando objeto negro...")

    def image_callback(self, msg):
        try:
            # Convertir mensaje de ROS a imagen de OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Error convirtiendo imagen: {e}")
            return

        alto, ancho, _ = cv_image.shape
        
        # 1. Extraer Región de Interés (ROI)
        # Analizamos solo una franja vertical central (del 35% al 65% del ancho)
        # Esto asume que el LiDAR detendrá al robot con el tronco en el centro de la cámara
        x_inicio = int(ancho * 0.35)
        x_fin = int(ancho * 0.65)
        roi = cv_image[:, x_inicio:x_fin]

        # 2. Conversión a espacio HSV
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # 3. Máscara de color para la botella negra
        # El negro tiene bajo Value (luminosidad), el Hue y Saturation dan igual
        limite_inferior = np.array([0, 0, 0])
        limite_superior = np.array([180, 255, 60]) # V ajustado a 60 para tolerar reflejos de luz artificial
        mascara_negra = cv2.inRange(hsv_roi, limite_inferior, limite_superior)

        # 4. Cálculo de densidad de píxeles
        pixeles_totales = roi.shape[0] * roi.shape[1]
        pixeles_negros = cv2.countNonZero(mascara_negra)
        porcentaje_negro = (pixeles_negros / pixeles_totales) * 100

        # 5. Veredicto
        msg_bool = Bool()
        # Si más del 25% del centro de la imagen es negro, confirmamos el objeto
        if porcentaje_negro > 25.0:
            msg_bool.data = True
        else:
            msg_bool.data = False
            
        self.pub_bool.publish(msg_bool)

        # 6. Ventanas de depuración visual (Opcional, útil para calibrar)
        # Dibuja un rectángulo verde en la imagen original para mostrar la zona de análisis
        cv2.rectangle(cv_image, (x_inicio, 0), (x_fin, alto), (0, 255, 0), 2)
        cv2.imshow("Vista Camara", cv_image)
        cv2.imshow("Mascara Negra (ROI)", mascara_negra)
        cv2.waitKey(1) # Necesario para que la ventana gráfica se actualice

def main(args=None):
    rclpy.init(args=args)
    nodo = ProcesadorRGB()
    try:
        rclpy.spin(nodo)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        nodo.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge
import cv2
import numpy as np

class ProcesadorMalezaNIR(Node):
    def __init__(self):
        super().__init__('procesador_maleza_nir')
        self.sub = self.create_subscription(Image, '/image_raw', self.image_callback, 10)
        self.pub_confirmado = self.create_publisher(Bool, '/rutina/maleza_detectada', 10)
        self.bridge = CvBridge()

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            print(f"error: {e}")
            return 

        # Convertir a escala de grises
        gris = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # Umbralización para objetos brillantes 
        # Todo píxel con intensidad mayor a 180 se vuelve blanco (maleza), el resto negro.
        _, mascara_brillante = cv2.threshold(gris, 180, 255, cv2.THRESH_BINARY)

        kernel = np.ones((5,5), np.uint8)
        # Limpiar ruido pequeño 
        mascara_limpia = cv2.morphologyEx(mascara_brillante, cv2.MORPH_OPEN, kernel)
        # Cerrar agujeros negros dentro de las hojas detectadas
        mascara_limpia = cv2.morphologyEx(mascara_limpia, cv2.MORPH_CLOSE, kernel)

        # Buscar grupos de hojas
        contornos, _ = cv2.findContours(mascara_limpia, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        maleza_detectada = False

        for contorno in contornos:
            area = cv2.contourArea(contorno)
            x, y, w, h = cv2.boundingRect(contorno)
            
            # Condiciones para la maleza: area mínima menor que la del tronco y descartar áreas gigantes 
            if 500 < area < (cv_image.shape[0] * cv_image.shape[1] * 0.4):
                maleza_detectada = True
                
                # Dibujar bounding box
                cv2.rectangle(cv_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(cv_image, "Maleza", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        msg_bool = Bool()
        msg_bool.data = maleza_detectada
        self.pub_confirmado.publish(msg_bool)

        cv2.imshow("Mascara Maleza", mascara_limpia)
        cv2.imshow("Deteccion MAPIR", cv_image)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    nodo = ProcesadorMalezaNIR()
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
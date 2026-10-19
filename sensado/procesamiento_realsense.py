#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image, CameraInfo
from std_msgs.msg import Bool, Float32
from cv_bridge import CvBridge
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
import pyransac3d as pyr
from rclpy.qos import qos_profile_sensor_data
import cv2

class DetectorCilindroVisual(Node):
    def __init__(self):
        super().__init__('detector_cilindro_visual')
        self.bridge = CvBridge()
        
        self.ultima_imagen = None
        self.matriz_intrinseca = None
        self.procesando = False
        
        # 1. Suscripción a la Nube de Puntos (Geometría)
        self.sub_pcl = self.create_subscription(
            PointCloud2, 
            '/camera/camera/depth/color/points', 
            self.pcl_callback, 
            qos_profile_sensor_data
        )
        
        # 2. Suscripción a la Imagen Color (Para dibujar)
        self.sub_color = self.create_subscription(
            Image, 
            '/camera/camera/color/image_raw', 
            self.color_callback, 
            qos_profile_sensor_data
        )

        # 3. Suscripción a CameraInfo (Para saber cómo proyectar el 3D al 2D)
        self.sub_info = self.create_subscription(
            CameraInfo,
            '/camera/camera/color/camera_info',
            self.info_callback,
            qos_profile_sensor_data
        )
        
        self.pub_confirmado = self.create_publisher(Bool, '/rutina/tronco_confirmado', 10)
        self.pub_distancia = self.create_publisher(Float32, '/rutina/distancia_confirmada', 10)
        
        self.get_logger().info("Nodo RANSAC+Visual iniciado. Esperando tópicos...")

    def color_callback(self, msg):
        try:
            self.ultima_imagen = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception:
            pass

    def info_callback(self, msg):
        # Guardamos la matriz K (Intrínsecas de la cámara) necesaria para proyectar 3D a píxeles
        if self.matriz_intrinseca is None:
            self.matriz_intrinseca = np.array(msg.k).reshape((3, 3))

    def pcl_callback(self, msg):
        if self.procesando or self.ultima_imagen is None or self.matriz_intrinseca is None:
            return
            
        self.procesando = True
        img_dibujo = self.ultima_imagen.copy()
        
        # Extracción segura a NumPy (el fix que hicimos antes)
        generador = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        datos = np.array(list(generador))
        
        if datos.size == 0:
            self.fallo_visual(img_dibujo)
            return

        if datos.ndim == 1:
            puntos = np.vstack([datos['x'], datos['y'], datos['z']]).T
        else:
            puntos = datos

        # Filtro Passthrough
        mascara_z = (puntos[:, 2] > 0.3) & (puntos[:, 2] < 1.0)
        mascara_x = (puntos[:, 0] > -0.5) & (puntos[:, 0] < 0.5)
        mascara_y = (puntos[:, 1] > -0.5) & (puntos[:, 1] < 0.5)
        
        puntos_filtrados = puntos[mascara_z & mascara_x & mascara_y]
        
        if puntos_filtrados.shape[0] < 200:
            self.fallo_visual(img_dibujo)
            return

        # Ajuste RANSAC
        cilindro = pyr.Cylinder()
        try:
            centro, eje, radio, inliers = cilindro.fit(puntos_filtrados, thresh=0.015, maxIteration=100)
            
            # Evaluación Geométrica
            if (0.025 < radio < 0.070) and len(inliers) > 150:
                distancia = float(centro[2])
                
                # --- MAGIA: PROYECCIÓN 3D A PÍXELES 2D ---
                # Extraemos los puntos 3D que RANSAC dijo que pertenecen al cilindro (inliers)
                puntos_cilindro_3d = puntos_filtrados[inliers]
                
                # Proyectamos las coordenadas espaciales (X,Y,Z) al plano de imagen de la cámara (U,V)
                puntos_2d_homogeneos = np.dot(self.matriz_intrinseca, puntos_cilindro_3d.T).T
                
                # Obtenemos los píxeles reales dividiendo por Z
                u = (puntos_2d_homogeneos[:, 0] / puntos_2d_homogeneos[:, 2]).astype(int)
                v = (puntos_2d_homogeneos[:, 1] / puntos_2d_homogeneos[:, 2]).astype(int)
                
                # Calculamos el Bounding Box en base a esos píxeles
                min_x, max_x = np.min(u), np.max(u)
                min_y, max_y = np.min(v), np.max(v)
                
                # Dibujamos en OpenCV
                cv2.rectangle(img_dibujo, (min_x, min_y), (max_x, max_y), (0, 255, 0), 3)
                texto = f"Cilindro (R={radio*100:.1f}cm, D={distancia:.2f}m)"
                cv2.putText(img_dibujo, texto, (min_x, min_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Publicación a ROS
                msg_bool = Bool()
                msg_bool.data = True
                self.pub_confirmado.publish(msg_bool)
                
                msg_dist = Float32()
                msg_dist.data = distancia
                self.pub_distancia.publish(msg_dist)
                
                self.get_logger().info(f"TRUE - {texto}", throttle_duration_sec=0.5)
            else:
                self.fallo_visual(img_dibujo)
                
        except Exception:
            self.fallo_visual(img_dibujo)

        # Mostramos la ventana gráfica
        cv2.imshow("Deteccion RANSAC", img_dibujo)
        cv2.waitKey(1)
        self.procesando = False

    def fallo_visual(self, img_dibujo):
        # Publica Falso y refresca la pantalla aunque no haya detectado nada
        msg_bool = Bool()
        msg_bool.data = False
        self.pub_confirmado.publish(msg_bool)
        self.get_logger().info("FALSE - Buscando cilindro...", throttle_duration_sec=1.0)
        cv2.imshow("Deteccion RANSAC", img_dibujo)
        cv2.waitKey(1)
        self.procesando = False

def main(args=None):
    rclpy.init(args=args)
    nodo = DetectorCilindroVisual()
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
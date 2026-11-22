#!/usr/bin/env python3
"""
Herramienta de calibración en terreno para la cámara MAPIR (`ros2 run sensado mapir_calibrador`).

Mide en tiempo real la distribución de intensidades NIR dentro de la Región de Interés (ROI):
- Mediana del fondo (suelo / corteza sin maleza)
- Percentil 90, 95 y 99 (hojas brillantes en NIR)
- Calcula y recomienda automáticamente los parámetros óptimos (`umbral_nir_min`,
  `umbral_nir_fijo` y `delta_contraste_fondo`) para usar en `real_rutina.launch.py`.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np


class MapirCalibrador(Node):
    def __init__(self):
        super().__init__('mapir_calibrador')

        self.declare_parameter('roi_x_min', 0.10)
        self.declare_parameter('roi_x_max', 0.90)
        self.declare_parameter('roi_y_min', 0.10)
        self.declare_parameter('roi_y_max', 0.90)
        self.declare_parameter('canal_nir', 'gris')

        self.roi_x_min = float(self.get_parameter('roi_x_min').value)
        self.roi_x_max = float(self.get_parameter('roi_x_max').value)
        self.roi_y_min = float(self.get_parameter('roi_y_min').value)
        self.roi_y_max = float(self.get_parameter('roi_y_max').value)
        self.canal_nir = str(self.get_parameter('canal_nir').value).strip().lower()

        self.bridge = CvBridge()
        self.sub = self.create_subscription(Image, '/image_raw', self.image_callback, 10)

        self.get_logger().info(
            "Calibrador MAPIR iniciado. Escuchando /image_raw y calculando percentiles NIR..."
        )

    def image_callback(self, msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception:
            return

        alto, ancho = cv_image.shape[:2]
        x1 = int(np.clip(self.roi_x_min * ancho, 0, ancho - 1))
        x2 = int(np.clip(self.roi_x_max * ancho, x1 + 1, ancho))
        y1 = int(np.clip(self.roi_y_min * alto, 0, alto - 1))
        y2 = int(np.clip(self.roi_y_max * alto, y1 + 1, alto))

        roi = cv_image[y1:y2, x1:x2]
        if self.canal_nir == 'rojo_nir':
            nir = roi[:, :, 2]
        elif self.canal_nir == 'max_canal':
            nir = np.max(roi, axis=2)
        else:
            nir = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        nir_suave = cv2.GaussianBlur(nir, (5, 5), 0)

        mediana = float(np.median(nir_suave))
        p90 = float(np.percentile(nir_suave, 90))
        p95 = float(np.percentile(nir_suave, 95))
        p99 = float(np.percentile(nir_suave, 99))
        max_val = int(np.max(nir_suave))

        # Si hay objetos muy brillantes respecto a la mediana (posible maleza a la vista),
        # el umbral ideal se ubica a mitad de camino entre el percentil 90 del fondo y el P99 de la maleza.
        contraste = p99 - mediana
        if contraste >= 30.0:
            umbral_sugerido = int(np.clip(mediana + 0.55 * contraste, 120, 235))
            delta_sugerido = int(np.clip(0.55 * contraste, 20, 80))
            estado_escena = "ESCENA CON CONTRASTE (posible maleza en vista)"
        else:
            umbral_sugerido = int(np.clip(p95 + 25.0, 140, 235))
            delta_sugerido = int(np.clip((p95 - mediana) + 25.0, 25, 80))
            estado_escena = "ESCENA UNIFORME (solo suelo/fondo)"

        self.get_logger().info(
            f"\n--- [CALIBRACIÓN MAPIR - {estado_escena}] ---\n"
            f"  Intensidad NIR -> Mediana(fondo): {mediana:.1f} | P90: {p90:.1f} | "
            f"P95: {p95:.1f} | P99: {p99:.1f} | Máx: {max_val}\n"
            f"  Parámetros recomendados:\n"
            f"    umbral_nir_min:={umbral_sugerido}  umbral_nir_fijo:={umbral_sugerido}  "
            f"delta_contraste_fondo:={delta_sugerido}",
            throttle_duration_sec=1.5,
        )


def main(args=None):
    rclpy.init(args=args)
    nodo = MapirCalibrador()
    try:
        rclpy.spin(nodo)
    except KeyboardInterrupt:
        pass
    finally:
        nodo.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

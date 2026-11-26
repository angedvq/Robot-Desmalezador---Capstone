#!/usr/bin/env python3
"""
Herramienta de calibración en terreno para la cámara MAPIR (`ros2 run sensado mapir_calibrador`).

Mide en tiempo real la distribución de intensidades NIR dentro de la Región de Interés (ROI):
- Mediana del fondo (suelo / corteza sin maleza)
- Percentil 90, 95 y 99 (hojas brillantes en NIR)
- Calcula y recomienda automáticamente los parámetros óptimos (`umbral_nir_min`,
  `umbral_nir_fijo` y `delta_contraste_fondo`) para usar en `real_rutina.launch.py`.
- Dibuja los bounding boxes de maleza (>= 5x5 cm) sobre la imagen y:
  1. Publica `/mapir/calibrador_image` (`sensor_msgs/Image`) para poder visualizarlo
     en RViz2 / rqt_image_view desde el PC remoto.
  2. Si se ejecuta con `gui:=true` (ej. `ros2 run sensado mapir_calibrador --ros-args -p gui:=true`),
     abre una ventana de OpenCV con la cámara y los bounding boxes (funciona desde el PC
     conectándose por SSH con `ssh -X` o `ssh -Y`).
"""

import os
import sys
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np


def _detectar_flag_gui_cli() -> bool:
    """Permite pasar `gui:=true` o `gui=true` directamente como argumento en terminal."""
    for arg in sys.argv[1:]:
        limpio = arg.strip().lower()
        if limpio in ('gui:=true', 'gui=true', 'gui:=1', 'gui=1', '--gui'):
            return True
    return False


class MapirCalibrador(Node):
    def __init__(self):
        super().__init__('mapir_calibrador')

        self.declare_parameter('roi_ancho_cm', 15.0)
        self.declare_parameter('roi_alto_cm', 15.0)
        self.declare_parameter('roi_x_min', 0.35)
        self.declare_parameter('roi_x_max', 0.65)
        self.declare_parameter('roi_y_min', 0.30)
        self.declare_parameter('roi_y_max', 0.70)
        self.declare_parameter('canal_nir', 'gris')
        self.declare_parameter('min_ancho_cm', 3.0)
        self.declare_parameter('min_alto_cm', 3.0)
        self.declare_parameter('px_por_cm', 16.0)
        self.declare_parameter('area_min_px', 250)
        self.declare_parameter('area_max_ratio', 0.85)
        self.declare_parameter('gui', _detectar_flag_gui_cli())

        self.roi_ancho_cm = float(self.get_parameter('roi_ancho_cm').value)
        self.roi_alto_cm = float(self.get_parameter('roi_alto_cm').value)
        self.roi_x_min = float(self.get_parameter('roi_x_min').value)
        self.roi_x_max = float(self.get_parameter('roi_x_max').value)
        self.roi_y_min = float(self.get_parameter('roi_y_min').value)
        self.roi_y_max = float(self.get_parameter('roi_y_max').value)
        self.canal_nir = str(self.get_parameter('canal_nir').value).strip().lower()
        self.min_ancho_cm = float(self.get_parameter('min_ancho_cm').value)
        self.min_alto_cm = float(self.get_parameter('min_alto_cm').value)
        self.px_por_cm = max(1.0, float(self.get_parameter('px_por_cm').value))
        self.area_min_px = int(self.get_parameter('area_min_px').value)
        self.area_max_ratio = float(self.get_parameter('area_max_ratio').value)
        self.kernel = np.ones((5, 5), np.uint8)

        solicita_gui = bool(self.get_parameter('gui').value) or _detectar_flag_gui_cli()
        tiene_display = bool(os.environ.get('DISPLAY'))
        self.gui = solicita_gui and tiene_display
        self.ventana_inicializada = False

        if solicita_gui and not tiene_display:
            self.get_logger().warn(
                "Se solicitó gui=True pero no hay DISPLAY activo en la sesión SSH.\n"
                "  -> Para abrir la ventana directamente en tu PC por SSH, conéctate con: ssh -Y usuario@ip_rpi\n"
                "  -> Alternativamente, puedes ver la imagen con bounding boxes en RViz2 en el tópico /mapir/calibrador_image"
            )

        self.bridge = CvBridge()
        self.sub = self.create_subscription(Image, '/image_raw', self.image_callback, 10)
        self.pub_img = self.create_publisher(Image, '/mapir/calibrador_image', 5)

        self.get_logger().info(
            f"Calibrador MAPIR iniciado (gui={'ACTIVO' if self.gui else 'INACTIVO'}). "
            "Escuchando /image_raw y publicando vista con bounding boxes en /mapir/calibrador_image..."
        )

    def image_callback(self, msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception:
            return

        alto, ancho = cv_image.shape[:2]
        if alto == 0 or ancho == 0:
            return

        px_por_cm_efectivo = self.px_por_cm * (float(ancho) / 1280.0)
        if self.roi_ancho_cm > 0.0 and self.roi_alto_cm > 0.0:
            roi_w_px = min(ancho, int(round(self.roi_ancho_cm * px_por_cm_efectivo)))
            roi_h_px = min(alto, int(round(self.roi_alto_cm * px_por_cm_efectivo)))
            cx = ancho // 2
            cy = alto // 2
            x1 = max(0, cx - roi_w_px // 2)
            x2 = min(ancho, x1 + roi_w_px)
            y1 = max(0, cy - roi_h_px // 2)
            y2 = min(alto, y1 + roi_h_px)
        else:
            x1 = int(np.clip(self.roi_x_min * ancho, 0, ancho - 1))
            x2 = int(np.clip(self.roi_x_max * ancho, x1 + 1, ancho))
            y1 = int(np.clip(self.roi_y_min * alto, 0, alto - 1))
            y2 = int(np.clip(self.roi_y_max * alto, y1 + 1, alto))

        roi = cv_image[y1:y2, x1:x2]
        roi_area = float((x2 - x1) * (y2 - y1))

        if self.canal_nir == 'rojo_nir':
            nir = roi[:, :, 2]
            nir_global = cv_image[:, :, 2]
        elif self.canal_nir == 'max_canal':
            nir = np.max(roi, axis=2)
            nir_global = np.max(cv_image, axis=2)
        else:
            nir = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            nir_global = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        nir_suave = cv2.GaussianBlur(nir, (5, 5), 0)

        # Si una maleza cubre >50% del ROI de 15x15 cm, la mediana simple del ROI subiría
        # al valor de la hoja; anclamos la referencia del suelo al mínimo entre P25 del ROI
        # y la mediana global de la imagen.
        mediana = float(min(np.percentile(nir_suave, 25), np.median(nir_global)))
        p90 = float(np.percentile(nir_suave, 90))
        p95 = float(np.percentile(nir_suave, 95))
        p99 = float(np.percentile(nir_suave, 99))
        max_val = int(np.max(nir_suave))

        # Si hay objetos muy brillantes respecto a la mediana (posible maleza a la vista),
        # el umbral ideal se ubica al 55% entre la mediana del fondo y el P99 de la maleza.
        contraste = p99 - mediana
        if contraste >= 30.0:
            umbral_sugerido = int(np.clip(mediana + 0.55 * contraste, 120, 235))
            delta_sugerido = int(np.clip(0.55 * contraste, 20, 80))
            estado_escena = "ESCENA CON CONTRASTE (posible maleza en vista)"
        else:
            umbral_sugerido = int(np.clip(p95 + 25.0, 140, 235))
            delta_sugerido = int(np.clip((p95 - mediana) + 25.0, 25, 80))
            estado_escena = "ESCENA UNIFORME (solo suelo/fondo)"

        # Segmentar con el umbral sugerido y extraer bounding boxes como en mapir_procesamiento_nir
        _, mascara_brillante = cv2.threshold(nir_suave, umbral_sugerido, 255, cv2.THRESH_BINARY)
        mascara_limpia = cv2.morphologyEx(mascara_brillante, cv2.MORPH_OPEN, self.kernel)
        mascara_limpia = cv2.morphologyEx(mascara_limpia, cv2.MORPH_CLOSE, self.kernel)

        contornos, _ = cv2.findContours(mascara_limpia, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_ancho_px = int(round(self.min_ancho_cm * px_por_cm_efectivo))
        min_alto_px = int(round(self.min_alto_cm * px_por_cm_efectivo))
        area_max_permitida = roi_area * self.area_max_ratio

        vista_cal = cv_image.copy()
        # Recuadro de la ROI (15x15 cm) en celeste/amarillo
        cv2.rectangle(vista_cal, (x1, y1), (x2, y2), (255, 200, 0), 2)
        cv2.putText(
            vista_cal,
            f"ROI {self.roi_ancho_cm:.0f}x{self.roi_alto_cm:.0f}cm",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 200, 0),
            2,
        )

        num_bboxes_validos = 0
        for contorno in contornos:
            area = cv2.contourArea(contorno)
            bx, by, bw, bh = cv2.boundingRect(contorno)
            if bw < 10 or bh < 10:
                continue

            ancho_cm = bw / px_por_cm_efectivo
            alto_cm = bh / px_por_cm_efectivo
            gx, gy = x1 + bx, y1 + by

            # Condición de tamaño mínimo: ancho >= 3 cm Y alto >= 3 cm (no necesita ser cuadrada)
            cumple_bbox = (bw >= min_ancho_px) and (bh >= min_alto_px)
            cumple_area = self.area_min_px < area < area_max_permitida

            if cumple_bbox and cumple_area:
                num_bboxes_validos += 1
                # Nuevo bounding box válido (>= 3x3 cm) en verde intenso dentro del ROI
                cv2.rectangle(vista_cal, (gx, gy), (gx + bw, gy + bh), (0, 255, 0), 3)
                cv2.putText(
                    vista_cal,
                    f"MALEZA {ancho_cm:.1f}x{alto_cm:.1f}cm ({int(area)}px)",
                    (gx, max(22, gy - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                )
            elif area >= 80:
                # Contorno detectado pero descartado por tener ancho o alto menor a 3 cm
                cv2.rectangle(vista_cal, (gx, gy), (gx + bw, gy + bh), (0, 140, 255), 1)
                cv2.putText(
                    vista_cal,
                    f"<{self.min_ancho_cm:.0f}cm ({ancho_cm:.1f}x{alto_cm:.1f}cm)",
                    (gx, max(22, gy - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 140, 255),
                    1,
                )

        # Superponer resumen de calibración en la parte superior de la imagen
        color_titulo = (0, 255, 0) if num_bboxes_validos > 0 else (0, 215, 255)
        cv2.putText(
            vista_cal,
            f"CALIBRADOR MAPIR | Malezas validas (>={self.min_ancho_cm:.0f}x{self.min_alto_cm:.0f}cm): {num_bboxes_validos}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            color_titulo,
            2,
        )
        cv2.putText(
            vista_cal,
            f"Fondo(Med): {mediana:.0f} | P95: {p95:.0f} | P99: {p99:.0f} | "
            f"Umbral sug: {umbral_sugerido} | Delta sug: +{delta_sugerido}",
            (15, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2,
        )

        # Publicar imagen anotada en /mapir/calibrador_image
        if self.pub_img.get_subscription_count() > 0:
            msg_out = self.bridge.cv2_to_imgmsg(vista_cal, encoding='bgr8')
            msg_out.header = msg.header
            self.pub_img.publish(msg_out)

        # Mostrar ventana OpenCV si gui=True
        if self.gui:
            if not self.ventana_inicializada:
                cv2.namedWindow("Calibrador MAPIR (Bounding Boxes)", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Calibrador MAPIR (Bounding Boxes)", 960, 540)
                self.ventana_inicializada = True
            cv2.imshow("Calibrador MAPIR (Bounding Boxes)", vista_cal)
            cv2.waitKey(1)

        self.get_logger().info(
            f"\n--- [CALIBRACIÓN MAPIR - {estado_escena}] ---\n"
            f"  Intensidad NIR -> Mediana(fondo): {mediana:.1f} | P90: {p90:.1f} | "
            f"P95: {p95:.1f} | P99: {p99:.1f} | Máx: {max_val} | BBoxes >={self.min_ancho_cm:.0f}x{self.min_alto_cm:.0f}cm: {num_bboxes_validos}\n"
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
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        nodo.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

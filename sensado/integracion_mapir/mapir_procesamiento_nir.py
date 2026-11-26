#!/usr/bin/env python3
"""
Nodo de procesamiento y detección de maleza con cámara MAPIR (NIR).

Mejoras respecto al nodo básico preliminar (`procesamiento_nir.py`):
1. Región de Interés (ROI) configurable para enfocar únicamente la zona del suelo/base
   del tronco e ignorar bordes del chasis o fondo lejano.
2. Umbralización Híbrida Calibrada (`modo_umbral = 'hibrido'` o `'fijo'`):
   - Combina un umbral absoluto mínimo de reflectancia NIR (`umbral_nir_min`) con un
     margen de contraste sobre la mediana del fondo local (`fondo_mediana + delta_contraste_fondo`).
   - Esto permite detectar maleza tanto en sombra/interior como en luz intensa sin que
     el suelo iluminado dispare falsos positivos.
3. Filtrado morfológico (`MORPH_OPEN` + `MORPH_CLOSE`), validación geométrica de contornos
   y confirmación temporal por ventana deslizante (`frames_confirmacion` de `ventana_frames`).
4. Ejecución segura en Raspberry Pi por SSH (`mostrar_ventana = False` por defecto para no
   abortar si no hay servidor gráfico X11), con publicación de `/mapir/debug_image` y
   barras deslizantes (trackbars) cuando se ejecuta con pantalla (`mostrar_ventana:=true`).
"""

import os
from collections import deque
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32
from cv_bridge import CvBridge
import cv2
import numpy as np


class MapirProcesadorMalezaNIR(Node):
    def __init__(self):
        super().__init__('mapir_procesamiento_nir')

        # ---------------------------------------------------------------------
        # Parámetros de calibración y umbralización NIR
        # ---------------------------------------------------------------------
        # Modo de umbralización:
        # - 'hibrido': adapta el umbral según la iluminación del fondo acotado en [umbral_nir_min, umbral_nir_max]
        # - 'fijo':    usa directamente `umbral_nir_fijo` (igual al método clásico pero dentro de la ROI)
        self.declare_parameter('modo_umbral', 'hibrido')
        self.declare_parameter('umbral_nir_fijo', 180)
        self.declare_parameter('umbral_nir_min', 150)
        self.declare_parameter('umbral_nir_max', 235)
        self.declare_parameter('delta_contraste_fondo', 35)

        # Canal a analizar:
        # - 'gris': promedio ponderado BGR -> Gray
        # - 'rojo_nir': canal R (índice 2 en BGR), donde las cámaras MAPIR RGN/OCN concentran el NIR
        # - 'max_canal': máximo entre canales para máxima sensibilidad NIR
        self.declare_parameter('canal_nir', 'gris')

        # Región de Interés (ROI) normalizada [0.0 .. 1.0]
        self.declare_parameter('roi_x_min', 0.10)
        self.declare_parameter('roi_x_max', 0.90)
        self.declare_parameter('roi_y_min', 0.10)
        self.declare_parameter('roi_y_max', 0.90)

        # Filtros geométricos de contornos, tamaño mínimo de bounding box (5x5 cm) y área
        self.declare_parameter('min_ancho_cm', 2.0)
        self.declare_parameter('min_alto_cm', 2.0)
        self.declare_parameter('px_por_cm', 10.0)  # Escala a 1280x720 (~16 px/cm -> 5 cm = 80 px)
        self.declare_parameter('area_min_px', 500)
        self.declare_parameter('area_total_min_px', 600)
        self.declare_parameter('area_max_ratio', 0.40)
        self.declare_parameter('kernel_morfologico', 5)

        # Confirmación temporal (estabilidad frente a destellos aislados)
        self.declare_parameter('ventana_frames', 5)
        self.declare_parameter('frames_confirmacion', 3)

        # Visualización y telemetría de calibración
        self.declare_parameter('mostrar_ventana', False)
        self.declare_parameter('log_calibracion', False)

        # Lectura de parámetros
        self.modo_umbral = str(self.get_parameter('modo_umbral').value).strip().lower()
        self.umbral_nir_fijo = int(self.get_parameter('umbral_nir_fijo').value)
        self.umbral_nir_min = int(self.get_parameter('umbral_nir_min').value)
        self.umbral_nir_max = int(self.get_parameter('umbral_nir_max').value)
        self.delta_contraste_fondo = int(self.get_parameter('delta_contraste_fondo').value)
        self.canal_nir = str(self.get_parameter('canal_nir').value).strip().lower()

        self.roi_x_min = float(self.get_parameter('roi_x_min').value)
        self.roi_x_max = float(self.get_parameter('roi_x_max').value)
        self.roi_y_min = float(self.get_parameter('roi_y_min').value)
        self.roi_y_max = float(self.get_parameter('roi_y_max').value)

        self.min_ancho_cm = float(self.get_parameter('min_ancho_cm').value)
        self.min_alto_cm = float(self.get_parameter('min_alto_cm').value)
        self.px_por_cm = max(1.0, float(self.get_parameter('px_por_cm').value))
        self.area_min_px = int(self.get_parameter('area_min_px').value)
        self.area_total_min_px = int(self.get_parameter('area_total_min_px').value)
        self.area_max_ratio = float(self.get_parameter('area_max_ratio').value)
        k_size = max(3, int(self.get_parameter('kernel_morfologico').value))
        if k_size % 2 == 0:
            k_size += 1
        self.kernel = np.ones((k_size, k_size), np.uint8)

        self.ventana_frames = max(1, int(self.get_parameter('ventana_frames').value))
        self.frames_confirmacion = max(
            1, min(self.ventana_frames, int(self.get_parameter('frames_confirmacion').value))
        )

        tiene_display = bool(os.environ.get('DISPLAY'))
        solicita_ventana = bool(self.get_parameter('mostrar_ventana').value)
        self.mostrar_ventana = solicita_ventana and tiene_display
        if solicita_ventana and not tiene_display:
            self.get_logger().warn(
                "Se solicitó mostrar_ventana=True pero no hay DISPLAY activo (sesión SSH). "
                "Desactivando cv2.imshow para evitar caída del nodo."
            )

        self.log_calibracion = bool(self.get_parameter('log_calibracion').value)
        self.trackbars_creados = False

        # Historial temporal de detecciones (True/False por cada frame reciente)
        self.historial_deteccion = deque(maxlen=self.ventana_frames)

        # Suscriptores y publicadores
        self.sub = self.create_subscription(Image, '/image_raw', self.image_callback, 10)
        self.pub_confirmado = self.create_publisher(Bool, '/rutina/maleza_detectada', 10)
        self.pub_cobertura = self.create_publisher(Float32, '/mapir/porcentaje_maleza', 10)
        self.pub_debug_img = self.create_publisher(Image, '/mapir/debug_image', 5)
        self.bridge = CvBridge()

        self.get_logger().info(
            f"mapir_procesamiento_nir iniciado | Modo: {self.modo_umbral} | "
            f"Umbral min/fijo: {self.umbral_nir_min}/{self.umbral_nir_fijo} | "
            f"Delta fondo: +{self.delta_contraste_fondo} | "
            f"BBox mín: {self.min_ancho_cm:.1f}x{self.min_alto_cm:.1f}cm "
            f"({int(self.min_ancho_cm * self.px_por_cm)}x{int(self.min_alto_cm * self.px_por_cm)}px) | "
            f"Confirmación: {self.frames_confirmacion}/{self.ventana_frames} frames"
        )

    def _inicializar_trackbars(self):
        """Crea barras deslizantes para calibrar en vivo cuando mostrar_ventana=True."""
        if self.trackbars_creados:
            return
        cv2.namedWindow("Deteccion MAPIR", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Mascara Maleza", cv2.WINDOW_NORMAL)
        cv2.createTrackbar("Umbral Min", "Mascara Maleza", self.umbral_nir_min, 255, lambda _x: None)
        cv2.createTrackbar("Umbral Fijo", "Mascara Maleza", self.umbral_nir_fijo, 255, lambda _x: None)
        cv2.createTrackbar("Delta Fondo", "Mascara Maleza", self.delta_contraste_fondo, 120, lambda _x: None)
        cv2.createTrackbar("Area Min", "Mascara Maleza", self.area_min_px, 5000, lambda _x: None)
        self.trackbars_creados = True

    def _extraer_canal_nir(self, bgr_roi: np.ndarray) -> np.ndarray:
        """Extrae la banda NIR a partir del recorte BGR según la configuración."""
        if self.canal_nir == 'rojo_nir':
            return bgr_roi[:, :, 2]
        elif self.canal_nir == 'max_canal':
            return np.max(bgr_roi, axis=2)
        return cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2GRAY)

    def image_callback(self, msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Error convirtiendo imagen MAPIR: {e}")
            return

        alto, ancho = cv_image.shape[:2]
        if alto == 0 or ancho == 0:
            return

        # Actualizar valores desde trackbars si la ventana interactiva está activa
        if self.mostrar_ventana:
            self._inicializar_trackbars()
            self.umbral_nir_min = cv2.getTrackbarPos("Umbral Min", "Mascara Maleza")
            self.umbral_nir_fijo = cv2.getTrackbarPos("Umbral Fijo", "Mascara Maleza")
            self.delta_contraste_fondo = cv2.getTrackbarPos("Delta Fondo", "Mascara Maleza")
            self.area_min_px = max(50, cv2.getTrackbarPos("Area Min", "Mascara Maleza"))

        # 1. Recorte de la Región de Interés (ROI)
        x1 = int(np.clip(self.roi_x_min * ancho, 0, ancho - 1))
        x2 = int(np.clip(self.roi_x_max * ancho, x1 + 1, ancho))
        y1 = int(np.clip(self.roi_y_min * alto, 0, alto - 1))
        y2 = int(np.clip(self.roi_y_max * alto, y1 + 1, alto))

        roi_bgr = cv_image[y1:y2, x1:x2]
        roi_area = float((x2 - x1) * (y2 - y1))

        # 2. Extracción de banda NIR y suavizado Gaussiano contra ruido de sensor
        nir_raw = self._extraer_canal_nir(roi_bgr)
        nir_suave = cv2.GaussianBlur(nir_raw, (5, 5), 0)

        # 3. Estadísticos de la escena para calibración / umbral dinámico
        fondo_mediana = float(np.median(nir_suave))
        p95_nir = float(np.percentile(nir_suave, 95))

        if self.modo_umbral == 'fijo':
            umbral_efectivo = int(np.clip(self.umbral_nir_fijo, 0, 255))
        else:
            # Modo híbrido: exige superar tanto el mínimo absoluto NIR como el fondo + delta
            umbral_dinamico = fondo_mediana + float(self.delta_contraste_fondo)
            umbral_efectivo = int(
                np.clip(
                    max(float(self.umbral_nir_min), umbral_dinamico),
                    self.umbral_nir_min,
                    self.umbral_nir_max,
                )
            )

        # 4. Binarización y limpieza morfológica
        _, mascara_brillante = cv2.threshold(nir_suave, umbral_efectivo, 255, cv2.THRESH_BINARY)
        mascara_limpia = cv2.morphologyEx(mascara_brillante, cv2.MORPH_OPEN, self.kernel)
        mascara_limpia = cv2.morphologyEx(mascara_limpia, cv2.MORPH_CLOSE, self.kernel)

        # 5. Búsqueda y filtrado geométrico de contornos de maleza (mínimo 5x5 cm de bounding box)
        contornos, _ = cv2.findContours(mascara_limpia, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Escalar px_por_cm proporcionalmente si la resolución difiere de 1280 de ancho
        px_por_cm_efectivo = self.px_por_cm * (float(ancho) / 1280.0)
        min_ancho_px = int(round(self.min_ancho_cm * px_por_cm_efectivo))
        min_alto_px = int(round(self.min_alto_cm * px_por_cm_efectivo))

        area_max_permitida = roi_area * self.area_max_ratio
        area_maleza_total = 0.0
        num_manchas_validas = 0

        # Dibujar recuadro de la ROI en la imagen de diagnóstico
        vista_debug = cv_image.copy()
        cv2.rectangle(vista_debug, (x1, y1), (x2, y2), (255, 200, 0), 2)

        for contorno in contornos:
            area = cv2.contourArea(contorno)
            bx, by, bw, bh = cv2.boundingRect(contorno)

            # Condición de tamaño mínimo: tanto el ancho como el alto del bounding box
            # deben ser de al menos min_ancho_cm x min_alto_cm (por defecto 5.0 cm x 5.0 cm),
            # además de cumplir el rango de área válida.
            cumple_bbox_minimo = (bw >= min_ancho_px) and (bh >= min_alto_px)
            cumple_area = self.area_min_px < area < area_max_permitida

            if cumple_bbox_minimo and cumple_area:
                area_maleza_total += area
                num_manchas_validas += 1

                ancho_cm = bw / px_por_cm_efectivo
                alto_cm = bh / px_por_cm_efectivo

                # Coordenadas globales en la imagen completa
                gx, gy = x1 + bx, y1 + by
                cv2.rectangle(vista_debug, (gx, gy), (gx + bw, gy + bh), (0, 255, 0), 2)
                cv2.putText(
                    vista_debug,
                    f"Maleza {ancho_cm:.1f}x{alto_cm:.1f}cm",
                    (gx, max(20, gy - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                )

        # Si al menos un bounding box superó el tamaño mínimo (5x5 cm), es válido en este frame
        maleza_frame_actual = num_manchas_validas > 0
        self.historial_deteccion.append(maleza_frame_actual)

        # 6. Confirmación temporal (al menos M de los últimos N frames)
        votos_positivos = sum(1 for v in self.historial_deteccion if v)
        maleza_confirmada = votos_positivos >= self.frames_confirmacion

        porcentaje_maleza = (area_maleza_total / roi_area) * 100.0 if roi_area > 0 else 0.0

        # Publicar tópicos ROS 2
        msg_bool = Bool()
        msg_bool.data = bool(maleza_confirmada)
        self.pub_confirmado.publish(msg_bool)

        msg_pct = Float32()
        msg_pct.data = float(porcentaje_maleza)
        self.pub_cobertura.publish(msg_pct)

        # Telemetría en consola si se activa log_calibracion
        if self.log_calibracion:
            self.get_logger().info(
                f"[CAL MAPIR] Fondo(med): {fondo_mediana:.1f} | P95: {p95_nir:.1f} | "
                f"Umbral: {umbral_efectivo} | Área: {area_maleza_total:.0f}px "
                f"({porcentaje_maleza:.1f}%) | Votos: {votos_positivos}/{len(self.historial_deteccion)} "
                f"-> MALEZA: {'SI' if maleza_confirmada else 'NO'}",
                throttle_duration_sec=1.0,
            )

        # 7. Superponer telemetría en la imagen de diagnóstico (/mapir/debug_image)
        color_estado = (0, 255, 0) if maleza_confirmada else (0, 165, 255)
        texto_estado = (
            f"MALEZA: {'SI' if maleza_confirmada else 'NO'} "
            f"(votos {votos_positivos}/{self.ventana_frames} | area={int(area_maleza_total)}px)"
        )
        texto_cal = (
            f"Modo:{self.modo_umbral} | Fondo:{fondo_mediana:.0f} | P95:{p95_nir:.0f} | "
            f"Umbral:{umbral_efectivo}"
        )
        cv2.putText(vista_debug, texto_estado, (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_estado, 2)
        cv2.putText(vista_debug, texto_cal, (15, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        if self.pub_debug_img.get_subscription_count() > 0:
            msg_dbg = self.bridge.cv2_to_imgmsg(vista_debug, encoding='bgr8')
            msg_dbg.header = msg.header
            self.pub_debug_img.publish(msg_dbg)

        if self.mostrar_ventana:
            cv2.imshow("Mascara Maleza", mascara_limpia)
            cv2.imshow("Deteccion MAPIR", vista_debug)
            cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    nodo = MapirProcesadorMalezaNIR()
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

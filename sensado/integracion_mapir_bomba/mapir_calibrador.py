#!/usr/bin/env python3
"""
Herramienta de calibración en terreno y pasillo para la cámara MAPIR (`ros2 run sensado mapir_calibrador`).

Utiliza exactamente el mismo motor de detección multi-escenario (`analizar_roi_planta`) que
`mapir_procesamiento_nir.py`, garantizando que los bounding boxes que se ven en el calibrador
coincidan 1:1 con los de la rutina autónoma en:
- Pasillo sin iluminación artificial (con luz tenue de ventanas) y planta en maceta
- Pasillo con iluminación artificial
- Exterior en sombra
- Exterior al sol (filtrando reflejos solares especulares del piso)

Controles interactivos cuando se abre con `gui:=true`:
- Trackbars para ajustar en vivo `Umbral Min`, `Delta Fondo`, `Anti Reflejo` y `Sensibilidad %`.
- Tecla `s`: guarda la calibración actual en `~/.config/sensado_mapir_calibracion.json` para que
  `mapir_procesamiento_nir` la cargue automáticamente al lanzar `real_rutina.launch.py`.
- Tecla `r`: restaura los valores adaptativos por defecto.
"""

import json
import os
import sys
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

from sensado.integracion_mapir_bomba.mapir_procesamiento_nir import (
    CALIB_FILE_PATH,
    analizar_roi_planta,
    es_reflejo_solar_contorno,
)


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
        self.declare_parameter('modo_umbral', 'hibrido')
        self.declare_parameter('umbral_nir_min', 25)
        self.declare_parameter('umbral_nir_max', 232)
        self.declare_parameter('delta_contraste_fondo', 14)
        self.declare_parameter('umbral_reflejo_solar', 238)
        self.declare_parameter('sensibilidad_follaje', 1.0)
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
        self.modo_umbral = str(self.get_parameter('modo_umbral').value).strip().lower()
        self.umbral_nir_min = int(self.get_parameter('umbral_nir_min').value)
        self.umbral_nir_max = int(self.get_parameter('umbral_nir_max').value)
        self.delta_contraste_fondo = int(self.get_parameter('delta_contraste_fondo').value)
        self.umbral_reflejo_solar = int(self.get_parameter('umbral_reflejo_solar').value)
        self.sensibilidad_follaje = float(self.get_parameter('sensibilidad_follaje').value)
        self.min_ancho_cm = float(self.get_parameter('min_ancho_cm').value)
        self.min_alto_cm = float(self.get_parameter('min_alto_cm').value)
        self.px_por_cm = max(1.0, float(self.get_parameter('px_por_cm').value))
        self.area_min_px = int(self.get_parameter('area_min_px').value)
        self.area_max_ratio = float(self.get_parameter('area_max_ratio').value)
        self.kernel = np.ones((5, 5), np.uint8)

        self._cargar_calibracion_guardada()

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

    def _cargar_calibracion_guardada(self):
        if not os.path.isfile(CALIB_FILE_PATH):
            return
        try:
            with open(CALIB_FILE_PATH, 'r', encoding='utf-8') as f:
                datos = json.load(f)
            self.umbral_nir_min = int(datos.get('umbral_nir_min', self.umbral_nir_min))
            self.delta_contraste_fondo = int(datos.get('delta_contraste_fondo', self.delta_contraste_fondo))
            self.umbral_reflejo_solar = int(datos.get('umbral_reflejo_solar', self.umbral_reflejo_solar))
            self.sensibilidad_follaje = float(datos.get('sensibilidad_follaje', self.sensibilidad_follaje))
            self.get_logger().info(f"Calibración previa cargada desde {CALIB_FILE_PATH}")
        except Exception:
            pass

    def _guardar_calibracion(self):
        try:
            os.makedirs(os.path.dirname(CALIB_FILE_PATH), exist_ok=True)
            datos = {
                'umbral_nir_min': self.umbral_nir_min,
                'delta_contraste_fondo': self.delta_contraste_fondo,
                'umbral_reflejo_solar': self.umbral_reflejo_solar,
                'sensibilidad_follaje': round(self.sensibilidad_follaje, 2),
            }
            with open(CALIB_FILE_PATH, 'w', encoding='utf-8') as f:
                json.dump(datos, f, indent=2)
            self.get_logger().info(
                f"[GUARDADO] Calibración guardada en {CALIB_FILE_PATH}: {datos}. "
                "mapir_procesamiento_nir la usará automáticamente."
            )
        except Exception as e:
            self.get_logger().error(f"No se pudo guardar la calibración: {e}")

    def _inicializar_gui(self):
        if self.ventana_inicializada:
            return
        nombre_win = "Calibrador MAPIR (Bounding Boxes)"
        cv2.namedWindow(nombre_win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(nombre_win, 960, 600)
        cv2.createTrackbar("Umbral Min", nombre_win, self.umbral_nir_min, 220, lambda _x: None)
        cv2.createTrackbar("Delta Fondo", nombre_win, self.delta_contraste_fondo, 80, lambda _x: None)
        cv2.createTrackbar("Anti Reflejo", nombre_win, self.umbral_reflejo_solar, 255, lambda _x: None)
        cv2.createTrackbar(
            "Sensibilidad %",
            nombre_win,
            int(round(self.sensibilidad_follaje * 100)),
            200,
            lambda _x: None,
        )
        self.ventana_inicializada = True

    def image_callback(self, msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception:
            return

        alto, ancho = cv_image.shape[:2]
        if alto == 0 or ancho == 0:
            return

        nombre_win = "Calibrador MAPIR (Bounding Boxes)"
        if self.gui:
            self._inicializar_gui()
            self.umbral_nir_min = cv2.getTrackbarPos("Umbral Min", nombre_win)
            self.delta_contraste_fondo = max(4, cv2.getTrackbarPos("Delta Fondo", nombre_win))
            self.umbral_reflejo_solar = max(180, cv2.getTrackbarPos("Anti Reflejo", nombre_win))
            self.sensibilidad_follaje = max(40, cv2.getTrackbarPos("Sensibilidad %", nombre_win)) / 100.0

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

        roi_area = float((x2 - x1) * (y2 - y1))

        mascara_limpia, info = analizar_roi_planta(
            cv_image=cv_image,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            canal_nir=self.canal_nir,
            modo_umbral=self.modo_umbral,
            umbral_nir_fijo=self.umbral_nir_min,
            umbral_nir_min=self.umbral_nir_min,
            umbral_nir_max=self.umbral_nir_max,
            delta_contraste_fondo=self.delta_contraste_fondo,
            umbral_reflejo_solar=self.umbral_reflejo_solar,
            sensibilidad_follaje=self.sensibilidad_follaje,
            kernel=self.kernel,
        )

        piso_med = info['piso_anillo_med']
        fondo_base = info['fondo_mediana']
        p95 = info['p95_nir']
        p99 = info['p99_nir']
        umbral_efectivo = info['umbral_efectivo']
        regimen_luz = info['regimen_luz']

        contornos, _ = cv2.findContours(mascara_limpia, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_ancho_px = int(round(self.min_ancho_cm * px_por_cm_efectivo))
        min_alto_px = int(round(self.min_alto_cm * px_por_cm_efectivo))
        area_max_permitida = roi_area * self.area_max_ratio

        vista_cal = cv_image.copy()
        # Recuadro de la ROI (15x15 cm) en celeste
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
        num_reflejos = 0
        for contorno in contornos:
            area = cv2.contourArea(contorno)
            bx, by, bw, bh = cv2.boundingRect(contorno)
            if bw < 10 or bh < 10:
                continue

            ancho_cm = bw / px_por_cm_efectivo
            alto_cm = bh / px_por_cm_efectivo
            gx, gy = x1 + bx, y1 + by

            if es_reflejo_solar_contorno(
                contorno=contorno,
                nir_suave=info['nir_suave'],
                densidad_bordes=info['densidad_bordes'],
                mascara_reflejo=info['mascara_reflejo'],
                piso_anillo_med=piso_med,
                grad_piso_med=info['grad_piso_med'],
            ):
                num_reflejos += 1
                cv2.rectangle(vista_cal, (gx, gy), (gx + bw, gy + bh), (200, 0, 255), 1)
                cv2.putText(
                    vista_cal,
                    "Reflejo piso (ignorado)",
                    (gx, max(22, gy - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (200, 0, 255),
                    1,
                )
                continue

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
            f"CALIBRADOR MAPIR [{regimen_luz}] | Malezas (>={self.min_ancho_cm:.0f}x{self.min_alto_cm:.0f}cm): {num_bboxes_validos}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            color_titulo,
            2,
        )
        cv2.putText(
            vista_cal,
            f"Piso: {piso_med:.0f} | Base: {fondo_base:.0f} | P95: {p95:.0f} | "
            f"Umbral: {umbral_efectivo} | Sens: {int(self.sensibilidad_follaje * 100)}% | "
            "Tecla 's': Guardar | 'r': Reset",
            (15, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )

        # Miniatura de la máscara binaria en la esquina inferior derecha para inspección visual
        thumb_h, thumb_w = 140, 140
        mascara_bgr = cv2.cvtColor(cv2.resize(mascara_limpia, (thumb_w, thumb_h)), cv2.COLOR_GRAY2BGR)
        cv2.rectangle(mascara_bgr, (0, 0), (thumb_w - 1, thumb_h - 1), (0, 255, 0), 1)
        vista_cal[alto - thumb_h - 10 : alto - 10, ancho - thumb_w - 10 : ancho - 10] = mascara_bgr

        # Publicar imagen anotada en /mapir/calibrador_image
        if self.pub_img.get_subscription_count() > 0:
            msg_out = self.bridge.cv2_to_imgmsg(vista_cal, encoding='bgr8')
            msg_out.header = msg.header
            self.pub_img.publish(msg_out)

        # Mostrar ventana OpenCV si gui=True
        if self.gui:
            cv2.imshow(nombre_win, vista_cal)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                self._guardar_calibracion()
            elif key == ord('r'):
                cv2.setTrackbarPos("Umbral Min", nombre_win, 25)
                cv2.setTrackbarPos("Delta Fondo", nombre_win, 14)
                cv2.setTrackbarPos("Anti Reflejo", nombre_win, 238)
                cv2.setTrackbarPos("Sensibilidad %", nombre_win, 100)
                self._guardar_calibracion()

        self.get_logger().info(
            f"\n--- [CALIBRACIÓN MAPIR - {regimen_luz}] ---\n"
            f"  Piso(anillo): {piso_med:.1f} | Base(ROI): {fondo_base:.1f} | "
            f"P95: {p95:.1f} | P99: {p99:.1f} | Umbral efectivo: {umbral_efectivo}\n"
            f"  Malezas >={self.min_ancho_cm:.0f}x{self.min_alto_cm:.0f}cm: {num_bboxes_validos} | "
            f"Reflejos piso descartados: {num_reflejos}\n"
            f"  Parámetros activos: umbral_nir_min:={self.umbral_nir_min} "
            f"delta_contraste_fondo:={self.delta_contraste_fondo} "
            f"sensibilidad_follaje:={self.sensibilidad_follaje:.2f}",
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

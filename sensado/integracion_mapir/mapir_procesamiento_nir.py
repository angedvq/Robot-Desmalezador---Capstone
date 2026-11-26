#!/usr/bin/env python3
"""
Nodo de procesamiento y detección de maleza con cámara MAPIR (NIR) basado en:
1. Calibración inicial con el Panel de Reflectancia MAPIR (QR + parches 87%R, 51%R, 23%R):
   - Al iniciar la rutina (cuando `mapir_rutina_navegacion` solicita calibración), abre la GUI
     de visualización y espera a que el usuario apunte el panel de reflectancia MAPIR a la cámara.
   - Detecta automáticamente el código QR del panel y mide la intensidad en escala de grises
     de los 3 parches situados sobre el QR:
       * `23%R` (derecha): reflectancia de referencia para suelo/fondo oscuro.
       * `51%R` (centro):  reflectancia NIR de referencia para vegetación/maleza.
       * `87%R` (izquierda): reflectancia alta antes de saturación/reflejo especular.
   - Fija automáticamente el umbral NIR entre el parche de suelo (23%R) y el de vegetación (51%R):
       `umbral_nir = I_23 + factor_panel * (I_51 - I_23)`
     y envía `/mapir/bloquear_exposicion = True` al driver para congelar la auto-exposición.
2. Lógica directa de detección estilo `procesamiento_nir.py`:
   - Una vez calibrado, mantiene la ventana GUI abierta durante toda la rutina para monitoreo
     y ajuste en vivo mediante barras deslizantes (trackbars).
   - Aplica umbralización directa sobre la imagen en gris (`gris >= umbral_nir`), limpieza
     morfológica (`MORPH_OPEN` + `MORPH_CLOSE` con kernel 5x5) y validación de tamaño mínimo
     de bounding box (`>= 3x3 cm`, sin exigir que sea cuadrada) dentro de la ROI (`15x15 cm`).
"""

import json
import os
from collections import deque
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32
from cv_bridge import CvBridge
import cv2
import numpy as np

CALIB_FILE_PATH = os.path.expanduser("~/.config/sensado_mapir_calibracion.json")


def detectar_panel_mapir(gris: np.ndarray, vista_gui: np.ndarray = None):
    """
    Detecta el panel de calibración MAPIR mediante su código QR y extrae las medianas
    de intensidad (0..255) de los 3 parches de reflectancia ubicados sobre el QR:
      - Izquierda (87%R): center = p0 + 0.14 * u - 0.22 * v
      - Centro    (51%R): center = p0 + 0.50 * u - 0.22 * v
      - Derecha   (23%R): center = p0 + 0.86 * u - 0.22 * v
    donde `p0, p1, p2, p3` son las 4 esquinas del QR entregadas por `cv2.QRCodeDetector`.

    Retorna `(exito, dict_medidas)`:
      `exito`: True si el QR fue detectado y las intensidades cumplen `I_87 > I_51 > I_23`.
      `dict_medidas`: `{'i23': float, 'i51': float, 'i87': float}`
    """
    alto, ancho = gris.shape[:2]
    detector = cv2.QRCodeDetector()

    ok, pts = detector.detect(gris)
    if (not ok or pts is None) and max(alto, ancho) >= 900:
        # Reintentar a mitad de escala por si el panel está muy cerca de la cámara
        gris_small = cv2.resize(gris, (0, 0), fx=0.5, fy=0.5)
        ok_s, pts_s = detector.detect(gris_small)
        if ok_s and pts_s is not None:
            ok = True
            pts = pts_s * 2.0

    if not ok or pts is None or len(pts) == 0:
        return False, None

    quad = pts[0].astype(np.float32)
    if quad.shape != (4, 2):
        return False, None

    p0, p1, p2, p3 = quad[0], quad[1], quad[2], quad[3]
    u = p1 - p0  # Vector superior del QR (izquierda -> derecha)
    v = p3 - p0  # Vector lateral del QR (arriba -> abajo)
    ancho_qr = float(np.linalg.norm(u))
    alto_qr = float(np.linalg.norm(v))

    if ancho_qr < 40.0 or alto_qr < 40.0:
        return False, None

    # Dibujar contorno del QR detectado en la GUI
    if vista_gui is not None:
        pts_int = quad.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(vista_gui, [pts_int], True, (255, 220, 0), 2)

    radio_parche = max(6, int(round(0.055 * min(ancho_qr, alto_qr))))
    definicion_parches = [
        ('87%R', 0.14, (0, 255, 255)),
        ('51%R', 0.50, (0, 255, 0)),
        ('23%R', 0.86, (255, 180, 0)),
    ]

    medidas = {}
    for etiqueta, alpha, color in definicion_parches:
        centro = p0 + alpha * u - 0.22 * v
        cx = int(round(float(centro[0])))
        cy = int(round(float(centro[1])))

        x_a = max(0, cx - radio_parche)
        x_b = min(ancho, cx + radio_parche)
        y_a = max(0, cy - radio_parche)
        y_b = min(alto, cy + radio_parche)

        if (x_b - x_a) < 6 or (y_b - y_a) < 6:
            return False, None

        recorte = gris[y_a:y_b, x_a:x_b]
        val_med = float(np.median(recorte))
        medidas[etiqueta] = val_med

        if vista_gui is not None:
            cv2.rectangle(vista_gui, (x_a, y_a), (x_b, y_b), color, 2)
            cv2.putText(
                vista_gui,
                f"{etiqueta}:{val_med:.0f}",
                (max(5, x_a - 10), max(20, y_a - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                color,
                2,
            )

    i87 = medidas['87%R']
    i51 = medidas['51%R']
    i23 = medidas['23%R']

    # Validar orden físico de reflectancia: el parche de 87%R debe ser más brillante
    # que el de 51%R, y el de 51%R más brillante que el de 23%R con margen suficiente.
    orden_valido = (i87 > (i51 + 5.0)) and (i51 > (i23 + 5.0)) and ((i87 - i23) >= 18.0)
    return orden_valido, {'i23': i23, 'i51': i51, 'i87': i87}


def medir_recuadros_guia_manual(gris: np.ndarray, vista_gui: np.ndarray = None, dibujar: bool = True):
    """
    Dibuja 3 recuadros guía en la parte superior-central de la imagen para permitir
    calibrar manualmente con la tecla 'c' o Espacio alineando los 3 parches (87%R, 51%R, 23%R)
    incluso si el código QR no entra en la imagen.
    """
    alto, ancho = gris.shape[:2]
    cy = int(alto * 0.35)
    lado = max(36, int(min(ancho, alto) * 0.08))
    r = lado // 2
    separacion = int(lado * 1.85)
    cx_mid = ancho // 2

    posiciones = [
        ('87%R', cx_mid - separacion, (0, 255, 255)),
        ('51%R', cx_mid, (0, 255, 0)),
        ('23%R', cx_mid + separacion, (255, 180, 0)),
    ]
    medidas = {}
    for etiqueta, cx, color in posiciones:
        x_a, x_b = max(0, cx - r), min(ancho, cx + r)
        y_a, y_b = max(0, cy - r), min(alto, cy + r)
        val = float(np.median(gris[y_a:y_b, x_a:x_b]))
        medidas[etiqueta] = val
        if vista_gui is not None and dibujar:
            cv2.rectangle(vista_gui, (x_a, y_a), (x_b, y_b), color, 1)
            cv2.putText(
                vista_gui,
                f"Guia {etiqueta}:{val:.0f}",
                (x_a - 10, max(20, y_a - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
            )

    return {'i23': medidas['23%R'], 'i51': medidas['51%R'], 'i87': medidas['87%R']}


class MapirProcesadorMalezaNIR(Node):
    def __init__(self):
        super().__init__('mapir_procesamiento_nir')

        # ---------------------------------------------------------------------
        # Parámetros de calibración con panel MAPIR y umbralización directa NIR
        # ---------------------------------------------------------------------
        self.declare_parameter('requiere_calibracion_panel', True)
        self.declare_parameter('umbral_nir', 180)
        self.declare_parameter('umbral_max', 255)
        self.declare_parameter('factor_panel_pct', 60)  # 60% del camino entre 23%R (suelo) y 51%R (planta)
        self.declare_parameter('frames_panel_requeridos', 6)
        self.declare_parameter('canal_nir', 'gris')

        # Región de Interés (ROI) física centrada
        self.declare_parameter('roi_ancho_cm', 18.0)
        self.declare_parameter('roi_alto_cm', 10.0)

        # Filtros geométricos de contornos, tamaño mínimo de bounding box (3x3 cm) y área
        self.declare_parameter('min_ancho_cm', 3.0)
        self.declare_parameter('min_alto_cm', 3.0)
        self.declare_parameter('px_por_cm', 16.0)  # ~16 px/cm a 1280x720 -> 3 cm = 48 px
        self.declare_parameter('area_min_px', 250)
        self.declare_parameter('area_max_ratio', 0.85)
        self.declare_parameter('kernel_morfologico', 5)

        # Confirmación temporal
        self.declare_parameter('ventana_frames', 5)
        self.declare_parameter('frames_confirmacion', 3)

        # GUI abierta por defecto para apuntar el panel y monitorear malezas en vivo
        self.declare_parameter('gui', True)
        self.declare_parameter('log_calibracion', True)

        # Lectura de parámetros
        self.requiere_calibracion_panel = bool(self.get_parameter('requiere_calibracion_panel').value)
        self.umbral_nir = int(self.get_parameter('umbral_nir').value)
        self.umbral_max = int(self.get_parameter('umbral_max').value)
        if self.umbral_max < 240:
            self.umbral_max = 255
        self.factor_panel_pct = int(self.get_parameter('factor_panel_pct').value)
        self.frames_panel_requeridos = max(2, int(self.get_parameter('frames_panel_requeridos').value))
        self.canal_nir = str(self.get_parameter('canal_nir').value).strip().lower()

        self.roi_ancho_cm = float(self.get_parameter('roi_ancho_cm').value)
        self.roi_alto_cm = float(self.get_parameter('roi_alto_cm').value)

        self.min_ancho_cm = float(self.get_parameter('min_ancho_cm').value)
        self.min_alto_cm = float(self.get_parameter('min_alto_cm').value)
        self.px_por_cm = max(1.0, float(self.get_parameter('px_por_cm').value))
        self.area_min_px = int(self.get_parameter('area_min_px').value)
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
        solicita_ventana = bool(self.get_parameter('gui').value)
        self.mostrar_ventana = solicita_ventana and tiene_display
        if solicita_ventana and not tiene_display:
            self.get_logger().warn(
                "Se solicitó gui=True pero no hay DISPLAY activo en la sesión SSH.\n"
                "  -> Para ver la ventana GUI desde tu PC, entra por SSH con: ssh -Y usuario@ip_rpi\n"
                "  -> Igual puedes apuntar el panel QR frente a la cámara MAPIR para calibrar."
            )

        self.log_calibracion = bool(self.get_parameter('log_calibracion').value)
        self.trackbars_creados = False
        self.nombre_win = "Deteccion MAPIR (Panel + Debug)"
        self.nombre_win_mask = "Mascara Maleza"

        # Estado de calibración con el panel de reflectancia MAPIR
        self.calibrado = not self.requiere_calibracion_panel
        self.calibracion_habilitada = True  # La rutina puede pausarla durante calibración de bomba
        self.muestras_panel_i23 = []
        self.muestras_panel_i51 = []
        self.muestras_panel_i87 = []
        self.ultimo_i23 = 0.0
        self.ultimo_i51 = 0.0
        self.ultimo_i87 = 0.0

        # Historial temporal de detecciones
        self.historial_deteccion = deque(maxlen=self.ventana_frames)

        # Suscriptores y publicadores
        self.sub = self.create_subscription(Image, '/image_raw', self.image_callback, 10)
        self.sub_hab_cal = self.create_subscription(
            Bool, '/mapir/habilitar_calibracion', self._habilitar_cal_callback, 10
        )
        self.pub_confirmado = self.create_publisher(Bool, '/rutina/maleza_detectada', 10)
        self.pub_cobertura = self.create_publisher(Float32, '/mapir/porcentaje_maleza', 10)
        self.pub_calibrado = self.create_publisher(Bool, '/mapir/calibrado', 10)
        self.pub_lock_exp = self.create_publisher(Bool, '/mapir/bloquear_exposicion', 10)
        self.pub_debug_img = self.create_publisher(Image, '/mapir/debug_image', 5)
        self.bridge = CvBridge()

        # Publicar estado de calibración a 5 Hz para que la rutina de navegación esté sincronizada
        self.create_timer(0.20, self._publicar_estado_calibracion)

        self.get_logger().info(
            f"mapir_procesamiento_nir iniciado | "
            f"Calibración con panel QR: {'ACTIVA (esperando panel)' if not self.calibrado else 'OMITIDA'} | "
            f"Umbral inicial: {self.umbral_nir} | ROI: {self.roi_ancho_cm:.0f}x{self.roi_alto_cm:.0f}cm | "
            f"BBox mín: {self.min_ancho_cm:.1f}x{self.min_alto_cm:.1f}cm | "
            f"GUI: {'ABIERTA' if self.mostrar_ventana else 'SIN DISPLAY'}"
        )

    def _habilitar_cal_callback(self, msg: Bool):
        """Permite que la rutina indique cuándo ya terminó la bomba y está lista para el panel."""
        nuevo = bool(msg.data)
        if nuevo and not self.calibracion_habilitada:
            self.muestras_panel_i23.clear()
            self.muestras_panel_i51.clear()
            self.muestras_panel_i87.clear()
        self.calibracion_habilitada = nuevo

    def _publicar_estado_calibracion(self):
        msg = Bool()
        msg.data = bool(self.calibrado)
        self.pub_calibrado.publish(msg)

    def _guardar_calibracion_json(self):
        try:
            os.makedirs(os.path.dirname(CALIB_FILE_PATH), exist_ok=True)
            datos = {
                'umbral_nir': int(self.umbral_nir),
                'umbral_max': int(self.umbral_max),
                'factor_panel_pct': int(self.factor_panel_pct),
                'i23': round(self.ultimo_i23, 1),
                'i51': round(self.ultimo_i51, 1),
                'i87': round(self.ultimo_i87, 1),
            }
            with open(CALIB_FILE_PATH, 'w', encoding='utf-8') as f:
                json.dump(datos, f, indent=2)
        except Exception:
            pass

    def _inicializar_trackbars(self):
        """Crea las ventanas GUI y barras deslizantes que permanecen abiertas toda la rutina."""
        if self.trackbars_creados:
            return
        cv2.namedWindow(self.nombre_win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.nombre_win, 1024, 620)
        cv2.namedWindow(self.nombre_win_mask, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.nombre_win_mask, 480, 480)

        cv2.createTrackbar("Umbral NIR", self.nombre_win, self.umbral_nir, 255, lambda _x: None)
        cv2.createTrackbar("Umbral Max", self.nombre_win, self.umbral_max, 255, lambda _x: None)
        cv2.createTrackbar("Factor Panel %", self.nombre_win, self.factor_panel_pct, 120, lambda _x: None)
        cv2.createTrackbar("ROI Ancho cm", self.nombre_win, int(round(self.roi_ancho_cm)), 45, lambda _x: None)
        cv2.createTrackbar("ROI Alto cm", self.nombre_win, int(round(self.roi_alto_cm)), 45, lambda _x: None)
        cv2.createTrackbar(
            "Min Maleza mm", self.nombre_win, int(round(self.min_ancho_cm * 10.0)), 100, lambda _x: None
        )
        cv2.createTrackbar("Area Min px", self.nombre_win, self.area_min_px, 3000, lambda _x: None)
        self.trackbars_creados = True

    def _extraer_gris(self, bgr: np.ndarray) -> np.ndarray:
        if self.canal_nir == 'rojo_nir':
            return bgr[:, :, 2]
        elif self.canal_nir == 'max_canal':
            return np.max(bgr, axis=2)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    def _aplicar_calibracion_parches(self, i23: float, i51: float, i87: float, origen: str):
        """Calcula y fija el umbral NIR a partir de las lecturas 23%R, 51%R y 87%R del panel."""
        self.ultimo_i23 = i23
        self.ultimo_i51 = i51
        self.ultimo_i87 = i87

        factor = float(self.factor_panel_pct) / 100.0
        nuevo_umbral = int(round(np.clip(i23 + factor * (i51 - i23), 15.0, 245.0)))
        self.umbral_nir = nuevo_umbral
        self.calibrado = True

        if self.mostrar_ventana and self.trackbars_creados:
            cv2.setTrackbarPos("Umbral NIR", self.nombre_win, self.umbral_nir)

        # Bloquear auto-exposición en mapir_driver para que la cámara mantenga fija esta ganancia
        msg_lock = Bool()
        msg_lock.data = True
        self.pub_lock_exp.publish(msg_lock)

        self._publicar_estado_calibracion()
        self._guardar_calibracion_json()

        self.get_logger().info(
            f"\n============================================================\n"
            f" [CALIBRACIÓN MAPIR COMPLETADA - {origen}]\n"
            f"   Parche 23%R (Suelo)      : {i23:.1f}\n"
            f"   Parche 51%R (Vegetación) : {i51:.1f}\n"
            f"   Parche 87%R (Ref. Alta)  : {i87:.1f}\n"
            f"   -> Umbral NIR fijado en  : {self.umbral_nir} (Factor {self.factor_panel_pct}%)\n"
            f"   -> Auto-exposición       : BLOQUEADA\n"
            f"============================================================"
        )

    def image_callback(self, msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Error convirtiendo imagen MAPIR: {e}")
            return

        alto, ancho = cv_image.shape[:2]
        if alto == 0 or ancho == 0:
            return

        if self.mostrar_ventana:
            self._inicializar_trackbars()
            self.umbral_nir = cv2.getTrackbarPos("Umbral NIR", self.nombre_win)
            self.umbral_max = max(self.umbral_nir + 5, cv2.getTrackbarPos("Umbral Max", self.nombre_win))
            self.factor_panel_pct = max(10, cv2.getTrackbarPos("Factor Panel %", self.nombre_win))
            self.roi_ancho_cm = float(max(5, cv2.getTrackbarPos("ROI Ancho cm", self.nombre_win)))
            self.roi_alto_cm = float(max(5, cv2.getTrackbarPos("ROI Alto cm", self.nombre_win)))
            min_mm_gui = max(10, cv2.getTrackbarPos("Min Maleza mm", self.nombre_win))
            self.min_ancho_cm = float(min_mm_gui) / 10.0
            self.min_alto_cm = float(min_mm_gui) / 10.0
            self.area_min_px = max(50, cv2.getTrackbarPos("Area Min px", self.nombre_win))

        gris_completo = self._extraer_gris(cv_image)
        vista_debug = cv_image.copy()

        # =====================================================================
        # ETAPA 1: Calibración inicial con el Panel de Reflectancia MAPIR
        # =====================================================================
        if not self.calibrado:
            if self.calibracion_habilitada:
                ok_qr, med_qr = detectar_panel_mapir(gris_completo, vista_gui=vista_debug)
                med_guia = medir_recuadros_guia_manual(
                    gris_completo, vista_gui=vista_debug, dibujar=(not ok_qr)
                )

                if ok_qr and med_qr is not None:
                    self.muestras_panel_i23.append(med_qr['i23'])
                    self.muestras_panel_i51.append(med_qr['i51'])
                    self.muestras_panel_i87.append(med_qr['i87'])

                    n_ok = len(self.muestras_panel_i51)
                    cv2.putText(
                        vista_debug,
                        f"PANEL QR DETECTADO ({n_ok}/{self.frames_panel_requeridos}) | "
                        f"23%R={med_qr['i23']:.0f}  51%R={med_qr['i51']:.0f}  87%R={med_qr['i87']:.0f}",
                        (15, 95),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.70,
                        (0, 255, 0),
                        2,
                    )

                    if n_ok >= self.frames_panel_requeridos:
                        i23_med = float(np.median(self.muestras_panel_i23))
                        i51_med = float(np.median(self.muestras_panel_i51))
                        i87_med = float(np.median(self.muestras_panel_i87))
                        self._aplicar_calibracion_parches(
                            i23_med, i51_med, i87_med, origen="Panel QR MAPIR Automático"
                        )
            else:
                med_guia = {'i23': 0.0, 'i51': 0.0, 'i87': 0.0}

            # Banner de instrucciones en la GUI durante la espera del panel
            cv2.rectangle(vista_debug, (0, 0), (ancho, 72), (20, 20, 20), -1)
            estado_txt = (
                "ESPERANDO CALIBRACION - APUNTAR PANEL QR MAPIR"
                if self.calibracion_habilitada
                else "ESPERANDO FIN DE CALIBRACION DE BOMBA..."
            )
            cv2.putText(
                vista_debug,
                estado_txt,
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.78,
                (0, 215, 255),
                2,
            )
            cv2.putText(
                vista_debug,
                "Apunte el panel QR a la camara o presione 'c' (usar parches guia) / Espacio (usar Umbral NIR actual)",
                (15, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 255),
                1,
            )

        # =====================================================================
        # ETAPA 2: Lógica directa de `procesamiento_nir.py` con el umbral calibrado
        # =====================================================================
        px_por_cm_efectivo = self.px_por_cm * (float(ancho) / 1280.0)

        # Recorte de la ROI centrada (15x15 cm por defecto)
        roi_w_px = min(ancho, int(round(self.roi_ancho_cm * px_por_cm_efectivo)))
        roi_h_px = min(alto, int(round(self.roi_alto_cm * px_por_cm_efectivo)))
        cx = ancho // 2
        cy = alto // 2
        x1 = max(0, cx - roi_w_px // 2)
        x2 = min(ancho, x1 + roi_w_px)
        y1 = max(0, cy - roi_h_px // 2)
        y2 = min(alto, y1 + roi_h_px)

        gris_roi = gris_completo[y1:y2, x1:x2]
        roi_area = float((x2 - x1) * (y2 - y1))

        # Umbralización directa estilo procesamiento_nir.py:
        # Todo píxel con intensidad >= self.umbral_nir (y <= self.umbral_max) es candidato a maleza
        _, mascara_brillante = cv2.threshold(gris_roi, self.umbral_nir, 255, cv2.THRESH_BINARY)
        if self.umbral_max < 255:
            mascara_brillante[gris_roi > self.umbral_max] = 0

        # Limpieza morfológica idéntica a procesamiento_nir.py (MORPH_OPEN + MORPH_CLOSE)
        mascara_limpia = cv2.morphologyEx(mascara_brillante, cv2.MORPH_OPEN, self.kernel)
        mascara_limpia = cv2.morphologyEx(mascara_limpia, cv2.MORPH_CLOSE, self.kernel)

        contornos, _ = cv2.findContours(mascara_limpia, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_ancho_px = int(round(self.min_ancho_cm * px_por_cm_efectivo))
        min_alto_px = int(round(self.min_alto_cm * px_por_cm_efectivo))
        area_max_permitida = roi_area * self.area_max_ratio

        area_maleza_total = 0.0
        num_manchas_validas = 0

        # Dibujar recuadro de la ROI en celeste
        cv2.rectangle(vista_debug, (x1, y1), (x2, y2), (255, 200, 0), 2)
        cv2.putText(
            vista_debug,
            f"ROI {self.roi_ancho_cm:.0f}x{self.roi_alto_cm:.0f}cm",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 200, 0),
            2,
        )

        for contorno in contornos:
            area = cv2.contourArea(contorno)
            bx, by, bw, bh = cv2.boundingRect(contorno)
            if bw < 8 or bh < 8:
                continue

            ancho_cm = bw / px_por_cm_efectivo
            alto_cm = bh / px_por_cm_efectivo
            gx, gy = x1 + bx, y1 + by

            cumple_bbox_minimo = (bw >= min_ancho_px) and (bh >= min_alto_px)
            cumple_area = self.area_min_px < area < area_max_permitida

            if cumple_bbox_minimo and cumple_area:
                area_maleza_total += area
                num_manchas_validas += 1

                # Nuevo bounding box verde grueso para cada maleza válida detectada
                cv2.rectangle(vista_debug, (gx, gy), (gx + bw, gy + bh), (0, 255, 0), 3)
                cv2.putText(
                    vista_debug,
                    f"MALEZA {ancho_cm:.1f}x{alto_cm:.1f}cm",
                    (gx, max(20, gy - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                )
            elif area >= 80:
                # Contorno menor a 3x3 cm en naranjo delgado para diagnóstico
                cv2.rectangle(vista_debug, (gx, gy), (gx + bw, gy + bh), (0, 140, 255), 1)
                cv2.putText(
                    vista_debug,
                    f"<{self.min_ancho_cm:.1f}cm ({ancho_cm:.1f}x{alto_cm:.1f}cm)",
                    (gx, max(20, gy - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 140, 255),
                    1,
                )

        maleza_frame_actual = (num_manchas_validas > 0) and self.calibrado
        self.historial_deteccion.append(maleza_frame_actual)

        votos_positivos = sum(1 for v in self.historial_deteccion if v)
        maleza_confirmada = self.calibrado and (votos_positivos >= self.frames_confirmacion)
        porcentaje_maleza = (area_maleza_total / roi_area) * 100.0 if roi_area > 0 else 0.0

        # Publicar tópicos ROS 2
        msg_bool = Bool()
        msg_bool.data = bool(maleza_confirmada)
        self.pub_confirmado.publish(msg_bool)

        msg_pct = Float32()
        msg_pct.data = float(porcentaje_maleza)
        self.pub_cobertura.publish(msg_pct)

        mediana_roi = float(np.median(gris_roi)) if gris_roi.size > 0 else 0.0
        p95_roi = float(np.percentile(gris_roi, 95)) if gris_roi.size > 0 else 0.0

        if self.calibrado:
            color_estado = (0, 255, 0) if maleza_confirmada else (0, 165, 255)
            texto_estado = (
                f"MALEZA: {'SI' if maleza_confirmada else 'NO'} "
                f"(votos {votos_positivos}/{self.ventana_frames} | area={int(area_maleza_total)}px)"
            )
            info_panel = (
                f"Panel(23%={self.ultimo_i23:.0f}, 51%={self.ultimo_i51:.0f}, 87%={self.ultimo_i87:.0f}) | "
                if self.ultimo_i51 > 0
                else ""
            )
            texto_cal = (
                f"{info_panel}Umbral NIR: {self.umbral_nir} | "
                f"ROI Med:{mediana_roi:.0f} P95:{p95_roi:.0f} | Tecla 'r': Re-calibrar panel"
            )
            cv2.putText(vista_debug, texto_estado, (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_estado, 2)
            cv2.putText(vista_debug, texto_cal, (15, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            if self.log_calibracion:
                self.get_logger().info(
                    f"[MAPIR] Umbral:{self.umbral_nir} | ROI Med:{mediana_roi:.0f} P95:{p95_roi:.0f} | "
                    f"Área:{area_maleza_total:.0f}px ({porcentaje_maleza:.1f}%) | "
                    f"Votos:{votos_positivos}/{len(self.historial_deteccion)} -> "
                    f"MALEZA: {'SI' if maleza_confirmada else 'NO'}",
                    throttle_duration_sec=1.5,
                )

        if self.pub_debug_img.get_subscription_count() > 0:
            msg_dbg = self.bridge.cv2_to_imgmsg(vista_debug, encoding='bgr8')
            msg_dbg.header = msg.header
            self.pub_debug_img.publish(msg_dbg)

        if self.mostrar_ventana:
            cv2.imshow(self.nombre_win_mask, mascara_limpia)
            cv2.imshow(self.nombre_win, vista_debug)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('c'), ord('C')):
                # Calibrar usando los 3 recuadros guía manuales o las muestras del panel
                med_g = medir_recuadros_guia_manual(gris_completo, dibujar=False)
                if med_g['i51'] > med_g['i23'] + 3.0:
                    self._aplicar_calibracion_parches(
                        med_g['i23'], med_g['i51'], med_g['i87'], origen="Tecla 'c' (Recuadros Guía)"
                    )
                else:
                    self.calibrado = True
                    msg_lock = Bool()
                    msg_lock.data = True
                    self.pub_lock_exp.publish(msg_lock)
                    self._publicar_estado_calibracion()
                    self.get_logger().info(
                        f"[CALIBRACIÓN MAPIR] Confirmada manualmente con Umbral NIR = {self.umbral_nir}"
                    )
            elif key in (32, 13):  # Espacio o Enter: confirmar umbral actual de la barra deslizante
                self.calibrado = True
                msg_lock = Bool()
                msg_lock.data = True
                self.pub_lock_exp.publish(msg_lock)
                self._publicar_estado_calibracion()
                self._guardar_calibracion_json()
                self.get_logger().info(
                    f"[CALIBRACIÓN MAPIR] Confirmada desde GUI con Umbral NIR = {self.umbral_nir}"
                )
            elif key in (ord('r'), ord('R')):
                # Volver a modo calibración con panel y desbloquear auto-exposición temporalmente
                self.calibrado = False
                self.muestras_panel_i23.clear()
                self.muestras_panel_i51.clear()
                self.muestras_panel_i87.clear()
                msg_lock = Bool()
                msg_lock.data = False
                self.pub_lock_exp.publish(msg_lock)
                self._publicar_estado_calibracion()
                self.get_logger().info(
                    "[CALIBRACIÓN MAPIR] Re-calibración activada. Apunte el panel QR MAPIR a la cámara..."
                )


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

#!/usr/bin/env python3
"""
Nodo de procesamiento y detección robusta de maleza con cámara MAPIR (NIR).

Diseñado para funcionar de forma consistente en 4 escenarios de iluminación:
1. Pasillo sin iluminación artificial (poca luz natural proveniente de ventanas),
   detectando plantas en maceta o maleza real con bajo nivel de luz.
2. Pasillo con iluminación artificial (LED / fluorescente, donde casi no hay NIR
   y la detección se apoya en contraste adaptativo local, textura de follaje y color).
3. Exterior en sombra (baja reflectancia absoluta pero contraste relativo positivo).
4. Exterior con sol directo (rechazando reflejos solares especulares del piso
   que saturan la imagen pero carecen de estructura/textura de hojas).

Características principales:
- Región de Interés (ROI) centrada de 15x15 cm (`roi_ancho_cm = 15.0`, `roi_alto_cm = 15.0`).
- Tamaño mínimo de maleza válida de 3x3 cm (`min_ancho_cm = 3.0`, `min_alto_cm = 3.0`,
  sin exigir que sea cuadrada: basta que ancho >= 3 cm y alto >= 3 cm).
- Anillo de referencia de suelo alrededor de la ROI para estimar el brillo, ruido y color
  del piso sin que una planta grande dentro de la ROI distorsione el umbral.
- Ecualización adaptativa CLAHE en baja luz + filtro anti-reflejos solares especulares +
  validación de textura/bordes de follaje para no confundir reflejos del piso o sombras planas
  con plantas.
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


def extraer_canal_nir(bgr: np.ndarray, canal_nir: str) -> np.ndarray:
    """Extrae la banda NIR o luminancia según la configuración."""
    if canal_nir == 'rojo_nir':
        return bgr[:, :, 2]
    elif canal_nir == 'max_canal':
        return np.max(bgr, axis=2)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def analizar_roi_planta(
    cv_image: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    canal_nir: str = 'gris',
    modo_umbral: str = 'hibrido',
    umbral_nir_fijo: int = 180,
    umbral_nir_min: int = 25,
    umbral_nir_max: int = 232,
    delta_contraste_fondo: int = 14,
    umbral_reflejo_solar: int = 238,
    sensibilidad_follaje: float = 1.0,
    kernel: np.ndarray = None,
):
    """
    Analiza la ROI centrada de 15x15 cm comparándola con el anillo de piso circundante.
    Retorna la máscara binaria limpia de plantas, máscara de reflejos descartados y
    estadísticos de diagnóstico para telemetría/GUI.
    """
    if kernel is None:
        kernel = np.ones((5, 5), np.uint8)

    alto, ancho = cv_image.shape[:2]
    roi_bgr = cv_image[y1:y2, x1:x2]
    roi_h, roi_w = roi_bgr.shape[:2]

    # -------------------------------------------------------------------------
    # 1. Anillo exterior de referencia (el piso que rodea al ROI de 15x15 cm)
    # -------------------------------------------------------------------------
    pad = max(40, int(round(0.45 * max(roi_w, roi_h))))
    ax1 = max(0, x1 - pad)
    ay1 = max(0, y1 - pad)
    ax2 = min(ancho, x2 + pad)
    ay2 = min(alto, y2 + pad)

    parche_contexto = cv_image[ay1:ay2, ax1:ax2]
    nir_contexto = extraer_canal_nir(parche_contexto, canal_nir)
    nir_contexto_suave = cv2.GaussianBlur(nir_contexto, (5, 5), 0)

    # Máscara que selecciona únicamente el anillo fuera del ROI (piso de referencia)
    mascara_anillo = np.ones(nir_contexto_suave.shape, dtype=bool)
    rx1, ry1 = x1 - ax1, y1 - ay1
    rx2, ry2 = rx1 + roi_w, ry1 + roi_h
    mascara_anillo[ry1:ry2, rx1:rx2] = False

    pixeles_anillo = nir_contexto_suave[mascara_anillo]
    nir_roi_raw = extraer_canal_nir(roi_bgr, canal_nir)
    nir_suave = cv2.GaussianBlur(nir_roi_raw, (5, 5), 0)

    if pixeles_anillo.size < 100:
        pixeles_anillo = nir_suave.ravel()

    # -------------------------------------------------------------------------
    # 2. Detección temprana de reflejos solares / brillos especulares del piso
    # -------------------------------------------------------------------------
    hsv_roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    sat_roi = hsv_roi[:, :, 1]
    val_roi = hsv_roi[:, :, 2]

    # Gradiente morfológico local (estructura fina / bordes de hojas)
    k3 = np.ones((3, 3), np.uint8)
    grad_raw = cv2.morphologyEx(nir_suave, cv2.MORPH_GRADIENT, k3)
    densidad_bordes = cv2.GaussianBlur(grad_raw, (15, 15), 0)

    # Un reflejo solar especular tiene brillo extremo (cerca de saturación) y baja saturación
    # de color, o es una mancha plana muy brillante sin micro-textura de hojas.
    p50_anillo_bruto = float(np.median(pixeles_anillo))
    umbral_glare = int(np.clip(umbral_reflejo_solar, 210, 254))
    if p50_anillo_bruto > 150.0:
        # En exterior muy soleado, ser más estricto contra brillos blancos del piso
        umbral_glare = min(umbral_glare, 226)

    mascara_reflejo = (
        (nir_suave >= umbral_glare)
        | ((val_roi >= (umbral_glare - 10)) & (sat_roi < 25) & (nir_suave > (p50_anillo_bruto + 45.0)))
    )

    # Dilatar ligeramente la máscara de reflejo para cubrir halos de saturación
    mascara_reflejo_u8 = cv2.dilate(mascara_reflejo.astype(np.uint8) * 255, k3, iterations=1)
    mascara_reflejo = mascara_reflejo_u8 > 0

    # -------------------------------------------------------------------------
    # 3. Estadísticos robustos del piso (excluyendo reflejos solares saturados)
    # -------------------------------------------------------------------------
    pixeles_anillo_limpios = pixeles_anillo[pixeles_anillo < umbral_glare]
    if pixeles_anillo_limpios.size < 50:
        pixeles_anillo_limpios = pixeles_anillo

    piso_anillo_med = float(np.median(pixeles_anillo_limpios))
    piso_anillo_mad = float(np.median(np.abs(pixeles_anillo_limpios - piso_anillo_med)) * 1.4826)
    piso_anillo_mad = max(2.0, piso_anillo_mad)

    # Para el contraste positivo NIR (hoja más brillante que la tierra de la maceta o el suelo),
    # usamos como base oscura el mínimo entre la mediana del piso y el percentil 25 de la ROI.
    piso_oscuro_ref = float(min(piso_anillo_med, np.percentile(nir_suave, 25)))
    fondo_mediana = piso_oscuro_ref
    p95_nir = float(np.percentile(nir_suave[~mascara_reflejo], 95)) if np.any(~mascara_reflejo) else float(np.percentile(nir_suave, 95))
    p99_nir = float(np.percentile(nir_suave[~mascara_reflejo], 99)) if np.any(~mascara_reflejo) else float(np.percentile(nir_suave, 99))

    # Clasificar nivel de iluminación de la escena
    if piso_anillo_med < 75.0:
        regimen_luz = "BAJA (Pasillo/Sombra)"
    elif piso_anillo_med < 155.0:
        regimen_luz = "MEDIA (Interior/Nublado)"
    else:
        regimen_luz = "ALTA (Exterior/Sol)"

    # -------------------------------------------------------------------------
    # 4. Cálculo del umbral adaptativo y pistas complementarias (NIR + Textura + Color)
    # -------------------------------------------------------------------------
    escala_sens = max(0.4, min(2.5, float(sensibilidad_follaje)))

    if modo_umbral == 'fijo':
        umbral_efectivo = int(np.clip(umbral_nir_fijo, 5, 250))
        _, mascara_candidata = cv2.threshold(nir_suave, umbral_efectivo, 255, cv2.THRESH_BINARY)
        mascara_candidata[mascara_reflejo] = 0
    else:
        # Escalar el delta de contraste según la luz disponible:
        # En pasillo oscuro o sombra (fondo ~ 25..70), la diferencia absoluta entre hoja y suelo
        # es menor (8..16 niveles), mientras que en sol pleno se exige mayor separación.
        factor_luz = np.clip(piso_oscuro_ref / 130.0, 0.32, 1.15)
        delta_adaptativo = max(
            7.0,
            min(float(delta_contraste_fondo) * factor_luz, float(delta_contraste_fondo)),
            2.4 * piso_anillo_mad,
        ) / escala_sens

        piso_min = max(15, min(int(umbral_nir_min), int(piso_oscuro_ref + 6)))
        umbral_efectivo = int(
            np.clip(
                piso_oscuro_ref + delta_adaptativo,
                piso_min,
                min(int(umbral_nir_max), umbral_glare - 3),
            )
        )

        # Pista A1: Reflectancia NIR superior al fondo oscuro (tierra de maceta / suelo en sombra)
        mascara_nir_pos = (nir_suave >= umbral_efectivo) & (~mascara_reflejo)

        # Pista A2: Realce CLAHE para pasillos sin iluminación (solo luz tenue de ventana) o sombra
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(6, 6))
        nir_clahe_ctx = clahe.apply(nir_contexto_suave)
        nir_clahe_roi = nir_clahe_ctx[ry1:ry2, rx1:rx2]
        nir_clahe_anillo = nir_clahe_ctx[mascara_anillo]
        med_clahe = float(min(np.median(nir_clahe_anillo), np.percentile(nir_clahe_roi, 30)))
        mad_clahe = max(4.0, float(np.median(np.abs(nir_clahe_anillo - np.median(nir_clahe_anillo))) * 1.4826))
        umbral_clahe = med_clahe + max(14.0, 2.5 * mad_clahe) / escala_sens
        mascara_clahe_pos = (nir_clahe_roi >= umbral_clahe) & (~mascara_reflejo)

        # Pista B: Contraste de objeto + textura de follaje respecto al piso del pasillo.
        # En un pasillo con piso claro/reflectante (o luz artificial), una planta en maceta
        # puede verse más oscura o de distinto tono que la baldosa del piso, pero con alta
        # densidad de bordes de hojas/tallos (a diferencia de una sombra plana en el piso).
        grad_ctx = cv2.morphologyEx(nir_contexto_suave, cv2.MORPH_GRADIENT, k3)
        grad_ctx_suave = cv2.GaussianBlur(grad_ctx, (15, 15), 0)
        grad_anillo = grad_ctx_suave[mascara_anillo]
        grad_piso_med = float(np.median(grad_anillo))
        grad_piso_mad = max(1.5, float(np.median(np.abs(grad_anillo - grad_piso_med)) * 1.4826))
        umbral_textura = grad_piso_med + max(4.0, 2.6 * grad_piso_mad) / escala_sens

        diff_abs_piso = np.abs(nir_suave.astype(np.float32) - piso_anillo_med)
        umbral_diff_piso = max(10.0, 2.6 * piso_anillo_mad) / escala_sens

        # Pista C: Diferencia cromática/espectral entre canales respecto al piso (si existe)
        parche_float = parche_contexto.astype(np.float32)
        suma_ctx = np.sum(parche_float, axis=2, keepdims=True) + 15.0
        norm_ctx = parche_float / suma_ctx
        color_piso = np.median(norm_ctx[mascara_anillo], axis=0)
        norm_roi = norm_ctx[ry1:ry2, rx1:rx2]
        dist_espectral = np.linalg.norm(norm_roi - color_piso, axis=2) * 255.0
        dist_anillo = np.linalg.norm(norm_ctx[mascara_anillo] - color_piso, axis=1) * 255.0
        umbral_espectral = float(np.median(dist_anillo)) + max(
            6.0, 3.0 * float(np.median(np.abs(dist_anillo - np.median(dist_anillo))) * 1.4826)
        ) / escala_sens

        tiene_textura_hoja = densidad_bordes >= umbral_textura
        tiene_contraste_piso = (diff_abs_piso >= umbral_diff_piso) | (dist_espectral >= umbral_espectral)

        # Para evitar que sombras planas o reflejos suaves del piso disparen falsos positivos:
        # - La detección positiva NIR (mascara_nir_pos o mascara_clahe_pos) requiere que la zona
        #   no sea un reflejo solar plano.
        # - La detección por silueta/contraste sobre piso claro requiere obligatoriamente
        #   textura real de follaje (tiene_textura_hoja).
        mascara_follaje = (
            ((mascara_nir_pos | mascara_clahe_pos) & (tiene_textura_hoja | (nir_suave >= (umbral_efectivo + 6))))
            | (tiene_contraste_piso & tiene_textura_hoja)
        ) & (~mascara_reflejo)

        mascara_candidata = (mascara_follaje.astype(np.uint8)) * 255

    # -------------------------------------------------------------------------
    # 5. Limpieza morfológica para unir hojas cercanas de una misma planta
    # -------------------------------------------------------------------------
    k_open = np.ones((3, 3), np.uint8)
    k_close = np.ones((9, 9), np.uint8)
    mascara_limpia = cv2.morphologyEx(mascara_candidata, cv2.MORPH_OPEN, k_open)
    mascara_limpia = cv2.morphologyEx(mascara_limpia, cv2.MORPH_CLOSE, k_close)
    mascara_limpia = cv2.morphologyEx(mascara_limpia, cv2.MORPH_OPEN, kernel)

    info_escena = {
        'fondo_mediana': fondo_mediana,
        'piso_anillo_med': piso_anillo_med,
        'piso_anillo_mad': piso_anillo_mad,
        'p95_nir': p95_nir,
        'p99_nir': p99_nir,
        'umbral_efectivo': umbral_efectivo,
        'umbral_glare': umbral_glare,
        'regimen_luz': regimen_luz,
        'nir_suave': nir_suave,
        'densidad_bordes': densidad_bordes,
        'grad_piso_med': float(np.median(densidad_bordes)),
        'mascara_reflejo': mascara_reflejo,
    }
    return mascara_limpia, info_escena


def es_reflejo_solar_contorno(
    contorno: np.ndarray,
    nir_suave: np.ndarray,
    densidad_bordes: np.ndarray,
    mascara_reflejo: np.ndarray,
    piso_anillo_med: float,
    grad_piso_med: float,
) -> bool:
    """
    Verifica si un contorno candidato corresponde en realidad a un reflejo solar
    o brillo especular plano del piso en vez de una planta.
    """
    bx, by, bw, bh = cv2.boundingRect(contorno)
    recorte_nir = nir_suave[by : by + bh, bx : bx + bw]
    recorte_grad = densidad_bordes[by : by + bh, bx : bx + bw]
    recorte_glare = mascara_reflejo[by : by + bh, bx : bx + bw]

    if recorte_nir.size == 0:
        return False

    mascara_c = np.zeros(recorte_nir.shape, dtype=np.uint8)
    contorno_local = contorno - np.array([[bx, by]])
    cv2.drawContours(mascara_c, [contorno_local], -1, 255, -1)
    idx = mascara_c > 0
    if not np.any(idx):
        return False

    vals_nir = recorte_nir[idx]
    vals_grad = recorte_grad[idx]

    # Dilatar un poco el recorte de glare para evaluar solapamiento con bordes de reflejo
    fraccion_glare = float(np.mean(recorte_glare))
    p90_contorno = float(np.percentile(vals_nir, 90))
    media_grad = float(np.mean(vals_grad))

    # Si el contorno está saturado de brillo (>230) y su textura interna es baja (superficie plana)
    if p90_contorno >= 232.0 and media_grad < (grad_piso_med + 6.5):
        return True
    if fraccion_glare >= 0.30 and media_grad < (grad_piso_med + 9.0):
        return True
    # En exterior soleado, una mancha brillante en el piso con muy poca textura interna es reflejo
    if piso_anillo_med > 145.0 and p90_contorno > 218.0 and media_grad < (grad_piso_med + 5.0):
        return True

    return False


class MapirProcesadorMalezaNIR(Node):
    def __init__(self):
        super().__init__('mapir_procesamiento_nir')

        # ---------------------------------------------------------------------
        # Parámetros de calibración y umbralización NIR
        # ---------------------------------------------------------------------
        self.declare_parameter('modo_umbral', 'hibrido')
        self.declare_parameter('umbral_nir_fijo', 180)
        self.declare_parameter('umbral_nir_min', 25)
        self.declare_parameter('umbral_nir_max', 232)
        self.declare_parameter('delta_contraste_fondo', 14)
        self.declare_parameter('umbral_reflejo_solar', 238)
        self.declare_parameter('sensibilidad_follaje', 1.0)
        self.declare_parameter('canal_nir', 'gris')

        # Región de Interés (ROI): centrada y de 15x15 cm físicos
        self.declare_parameter('roi_ancho_cm', 15.0)
        self.declare_parameter('roi_alto_cm', 15.0)
        self.declare_parameter('roi_x_min', 0.35)
        self.declare_parameter('roi_x_max', 0.65)
        self.declare_parameter('roi_y_min', 0.30)
        self.declare_parameter('roi_y_max', 0.70)

        # Filtros geométricos de contornos, tamaño mínimo de bounding box (3x3 cm) y área
        self.declare_parameter('min_ancho_cm', 3.0)
        self.declare_parameter('min_alto_cm', 3.0)
        self.declare_parameter('px_por_cm', 16.0)  # Escala a 1280x720 (~16 px/cm -> 15 cm = 240 px, 3 cm = 48 px)
        self.declare_parameter('area_min_px', 250)
        self.declare_parameter('area_total_min_px', 300)
        self.declare_parameter('area_max_ratio', 0.85)
        self.declare_parameter('kernel_morfologico', 5)

        # Confirmación temporal (estabilidad frente a destellos aislados)
        self.declare_parameter('ventana_frames', 5)
        self.declare_parameter('frames_confirmacion', 3)

        # Visualización y telemetría de calibración (acepta mostrar_ventana:=true o gui:=true)
        self.declare_parameter('mostrar_ventana', False)
        self.declare_parameter('gui', False)
        self.declare_parameter('log_calibracion', False)

        # Lectura de parámetros
        self.modo_umbral = str(self.get_parameter('modo_umbral').value).strip().lower()
        self.umbral_nir_fijo = int(self.get_parameter('umbral_nir_fijo').value)
        self.umbral_nir_min = int(self.get_parameter('umbral_nir_min').value)
        self.umbral_nir_max = int(self.get_parameter('umbral_nir_max').value)
        self.delta_contraste_fondo = int(self.get_parameter('delta_contraste_fondo').value)
        self.umbral_reflejo_solar = int(self.get_parameter('umbral_reflejo_solar').value)
        self.sensibilidad_follaje = float(self.get_parameter('sensibilidad_follaje').value)
        self.canal_nir = str(self.get_parameter('canal_nir').value).strip().lower()

        # Si el launch pasó los valores antiguos (150 / 35) por defecto, adaptarlos al rango
        # multi-escenario (pasillo sin luz / sombra / exterior) a menos que se guarde calibración
        if self.umbral_nir_min == 150 and self.delta_contraste_fondo == 35:
            self.umbral_nir_min = 25
            self.delta_contraste_fondo = 14

        self._cargar_calibracion_guardada()

        self.roi_ancho_cm = float(self.get_parameter('roi_ancho_cm').value)
        self.roi_alto_cm = float(self.get_parameter('roi_alto_cm').value)
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
        solicita_ventana = bool(self.get_parameter('mostrar_ventana').value) or bool(
            self.get_parameter('gui').value
        )
        self.mostrar_ventana = solicita_ventana and tiene_display
        if solicita_ventana and not tiene_display:
            self.get_logger().warn(
                "Se solicitó gui/mostrar_ventana=True pero no hay DISPLAY activo (sesión SSH). "
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
            f"Umbral min: {self.umbral_nir_min} | Delta base: +{self.delta_contraste_fondo} | "
            f"Anti-reflejo: >={self.umbral_reflejo_solar} | "
            f"ROI: {self.roi_ancho_cm:.0f}x{self.roi_alto_cm:.0f}cm | "
            f"BBox mín: {self.min_ancho_cm:.1f}x{self.min_alto_cm:.1f}cm "
            f"({int(self.min_ancho_cm * self.px_por_cm)}x{int(self.min_alto_cm * self.px_por_cm)}px)"
        )

    def _cargar_calibracion_guardada(self):
        """Carga parámetros guardados por `mapir_calibrador` (tecla 's') si existen."""
        if not os.path.isfile(CALIB_FILE_PATH):
            return
        try:
            with open(CALIB_FILE_PATH, 'r', encoding='utf-8') as f:
                datos = json.load(f)
            self.umbral_nir_min = int(datos.get('umbral_nir_min', self.umbral_nir_min))
            self.delta_contraste_fondo = int(datos.get('delta_contraste_fondo', self.delta_contraste_fondo))
            self.umbral_reflejo_solar = int(datos.get('umbral_reflejo_solar', self.umbral_reflejo_solar))
            self.sensibilidad_follaje = float(datos.get('sensibilidad_follaje', self.sensibilidad_follaje))
            self.get_logger().info(
                f"Calibración MAPIR cargada desde {CALIB_FILE_PATH}: "
                f"min={self.umbral_nir_min}, delta={self.delta_contraste_fondo}, "
                f"anti_reflejo={self.umbral_reflejo_solar}, sens={self.sensibilidad_follaje:.2f}"
            )
        except Exception as e:
            self.get_logger().warn(f"No se pudo leer {CALIB_FILE_PATH}: {e}")

    def _inicializar_trackbars(self):
        """Crea barras deslizantes para calibrar en vivo cuando mostrar_ventana/gui=True."""
        if self.trackbars_creados:
            return
        cv2.namedWindow("Deteccion MAPIR", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Mascara Maleza", cv2.WINDOW_NORMAL)
        cv2.createTrackbar("Umbral Min", "Mascara Maleza", self.umbral_nir_min, 220, lambda _x: None)
        cv2.createTrackbar("Delta Fondo", "Mascara Maleza", self.delta_contraste_fondo, 80, lambda _x: None)
        cv2.createTrackbar("Anti Reflejo", "Mascara Maleza", self.umbral_reflejo_solar, 255, lambda _x: None)
        cv2.createTrackbar(
            "Sensibilidad %",
            "Mascara Maleza",
            int(round(self.sensibilidad_follaje * 100)),
            200,
            lambda _x: None,
        )
        self.trackbars_creados = True

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
            self.umbral_nir_min = cv2.getTrackbarPos("Umbral Min", "Mascara Maleza")
            self.delta_contraste_fondo = max(4, cv2.getTrackbarPos("Delta Fondo", "Mascara Maleza"))
            self.umbral_reflejo_solar = max(180, cv2.getTrackbarPos("Anti Reflejo", "Mascara Maleza"))
            self.sensibilidad_follaje = max(40, cv2.getTrackbarPos("Sensibilidad %", "Mascara Maleza")) / 100.0

        # Escalar px_por_cm proporcionalmente si la resolución difiere de 1280 de ancho
        px_por_cm_efectivo = self.px_por_cm * (float(ancho) / 1280.0)

        # 1. Recorte de la Región de Interés (ROI de 15x15 cm centrada)
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

        # 2. Segmentación robusta multi-escenario (pasillo sin luz / sombra / exterior con sol)
        mascara_limpia, info = analizar_roi_planta(
            cv_image=cv_image,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            canal_nir=self.canal_nir,
            modo_umbral=self.modo_umbral,
            umbral_nir_fijo=self.umbral_nir_fijo,
            umbral_nir_min=self.umbral_nir_min,
            umbral_nir_max=self.umbral_nir_max,
            delta_contraste_fondo=self.delta_contraste_fondo,
            umbral_reflejo_solar=self.umbral_reflejo_solar,
            sensibilidad_follaje=self.sensibilidad_follaje,
            kernel=self.kernel,
        )

        fondo_mediana = info['fondo_mediana']
        piso_anillo_med = info['piso_anillo_med']
        p95_nir = info['p95_nir']
        umbral_efectivo = info['umbral_efectivo']
        regimen_luz = info['regimen_luz']

        # 3. Búsqueda y filtrado geométrico de contornos de maleza (mínimo 3 cm de ancho y 3 cm de alto)
        contornos, _ = cv2.findContours(mascara_limpia, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_ancho_px = int(round(self.min_ancho_cm * px_por_cm_efectivo))
        min_alto_px = int(round(self.min_alto_cm * px_por_cm_efectivo))

        area_max_permitida = roi_area * self.area_max_ratio
        area_maleza_total = 0.0
        num_manchas_validas = 0
        num_reflejos_descartados = 0

        # Dibujar recuadro de la ROI (15x15 cm) en la imagen de diagnóstico
        vista_debug = cv_image.copy()
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
            if bw < 10 or bh < 10:
                continue

            ancho_cm = bw / px_por_cm_efectivo
            alto_cm = bh / px_por_cm_efectivo
            gx, gy = x1 + bx, y1 + by

            # Descartar si el contorno tiene firma de reflejo solar plano en el piso
            if es_reflejo_solar_contorno(
                contorno=contorno,
                nir_suave=info['nir_suave'],
                densidad_bordes=info['densidad_bordes'],
                mascara_reflejo=info['mascara_reflejo'],
                piso_anillo_med=piso_anillo_med,
                grad_piso_med=info['grad_piso_med'],
            ):
                num_reflejos_descartados += 1
                cv2.rectangle(vista_debug, (gx, gy), (gx + bw, gy + bh), (200, 0, 255), 1)
                cv2.putText(
                    vista_debug,
                    "Reflejo piso",
                    (gx, max(20, gy - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (200, 0, 255),
                    1,
                )
                continue

            # Condición de tamaño mínimo: ancho >= 3.0 cm y alto >= 3.0 cm (no necesita ser cuadrada)
            cumple_bbox_minimo = (bw >= min_ancho_px) and (bh >= min_alto_px)
            cumple_area = self.area_min_px < area < area_max_permitida

            if cumple_bbox_minimo and cumple_area:
                area_maleza_total += area
                num_manchas_validas += 1

                # Nuevo bounding box verde para cada maleza válida detectada dentro del ROI
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
                # Contorno descartado por no alcanzar el mínimo de 3x3 cm (en naranjo delgado)
                cv2.rectangle(vista_debug, (gx, gy), (gx + bw, gy + bh), (0, 140, 255), 1)
                cv2.putText(
                    vista_debug,
                    f"<{self.min_ancho_cm:.0f}cm ({ancho_cm:.1f}x{alto_cm:.1f}cm)",
                    (gx, max(20, gy - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 140, 255),
                    1,
                )

        # Si al menos un bounding box superó el tamaño mínimo (3x3 cm), es válido en este frame
        maleza_frame_actual = num_manchas_validas > 0
        self.historial_deteccion.append(maleza_frame_actual)

        # 4. Confirmación temporal (al menos M de los últimos N frames)
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

        if self.log_calibracion:
            self.get_logger().info(
                f"[MAPIR {regimen_luz}] Piso:{piso_anillo_med:.0f} | Base:{fondo_mediana:.0f} | "
                f"P95:{p95_nir:.0f} | Umbral:{umbral_efectivo} | Área:{area_maleza_total:.0f}px "
                f"({porcentaje_maleza:.1f}%) | Reflejos ign:{num_reflejos_descartados} | "
                f"Votos:{votos_positivos}/{len(self.historial_deteccion)} -> "
                f"MALEZA: {'SI' if maleza_confirmada else 'NO'}",
                throttle_duration_sec=1.0,
            )

        # 5. Superponer telemetría en la imagen de diagnóstico (/mapir/debug_image)
        color_estado = (0, 255, 0) if maleza_confirmada else (0, 165, 255)
        texto_estado = (
            f"MALEZA: {'SI' if maleza_confirmada else 'NO'} "
            f"(votos {votos_positivos}/{self.ventana_frames} | area={int(area_maleza_total)}px)"
        )
        texto_cal = (
            f"Luz:{regimen_luz} | Piso:{piso_anillo_med:.0f} | P95:{p95_nir:.0f} | "
            f"Umbral:{umbral_efectivo}"
        )
        cv2.putText(vista_debug, texto_estado, (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_estado, 2)
        cv2.putText(vista_debug, texto_cal, (15, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)

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

#!/usr/bin/env python3
"""
Nodo driver de captura para la cámara multiespectral/NIR MAPIR en el robot real.

Características principales:
1. Búsqueda automática de puerto V4L2: intenta primero el puerto configurado (`video_port`,
   por defecto 2) y, si no está disponible en ese índice (por ejemplo si la RPi lo enumeró
   en `/dev/video0`), prueba automáticamente los puertos candidatos `[video_port, 0, 1, 2, 3, 4]`.
2. Minimización de latencia (buffer size = 1): evita acumular frames antiguos en el buffer de
   OpenCV/V4L2 para que, cuando el robot se detenga frente a un tronco, la imagen evaluada
   corresponda exactamente al instante actual.
3. Bloqueo de Auto-Exposición y Balance de Blancos (`bloquear_auto_exposicion`):
   Permite congelar la exposición de la cámara tras los primeros frames de adaptación a la luz
   ambiente o exactamente en el instante en que se calibra con el panel de reflectancia MAPIR
   (`/mapir/bloquear_exposicion`), evitando que la cámara cambie el brillo sola durante la rutina.
4. Reconexión automática ante micro-cortes USB.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge
import cv2


class MapirDriverReal(Node):
    def __init__(self):
        super().__init__('mapir_driver')

        self.declare_parameter('video_port', 2)
        self.declare_parameter('frame_width', 1280)
        self.declare_parameter('frame_height', 720)
        self.declare_parameter('fps', 15.0)
        self.declare_parameter('auto_detect_port', True)
        self.declare_parameter('bloquear_auto_exposicion', True)
        self.declare_parameter('exposicion_fija', -1.0)

        self.puerto_preferido = int(self.get_parameter('video_port').value)
        self.frame_width = int(self.get_parameter('frame_width').value)
        self.frame_height = int(self.get_parameter('frame_height').value)
        self.fps = max(1.0, float(self.get_parameter('fps').value))
        self.auto_detect_port = bool(self.get_parameter('auto_detect_port').value)
        self.bloquear_auto_exposicion = bool(self.get_parameter('bloquear_auto_exposicion').value)
        self.exposicion_fija = float(self.get_parameter('exposicion_fija').value)

        self.publisher_ = self.create_publisher(Image, '/image_raw', 10)
        self.sub_lock = self.create_subscription(
            Bool, '/mapir/bloquear_exposicion', self._bloquear_exposicion_callback, 10
        )
        self.bridge = CvBridge()

        self.cap = None
        self.puerto_activo = None
        self.fallos_consecutivos = 0
        self.frames_capturados = 0
        self.exposicion_bloqueada = False

        self._conectar_camara()

        periodo = 1.0 / self.fps
        self.timer = self.create_timer(periodo, self.timer_callback)

    def _aplicar_bloqueo_exposicion(self, motivo: str = ""):
        if self.cap is None or not self.cap.isOpened():
            return
        try:
            exp_actual = self.cap.get(cv2.CAP_PROP_EXPOSURE)
            # En V4L2: 1 = Manual Mode, 3 = Aperture Priority (Auto)
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            if self.exposicion_fija > 0.0:
                self.cap.set(cv2.CAP_PROP_EXPOSURE, self.exposicion_fija)
            elif exp_actual > 0.0:
                self.cap.set(cv2.CAP_PROP_EXPOSURE, exp_actual)

            self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)
            self.exposicion_bloqueada = True
            self.get_logger().info(
                f"[MAPIR Driver] Auto-exposición y Auto-WB BLOQUEADOS ({motivo}) | "
                f"Exposición fijada: {self.cap.get(cv2.CAP_PROP_EXPOSURE):.1f}"
            )
        except Exception as e:
            self.get_logger().warn(f"No se pudo bloquear auto-exposición V4L2: {e}")

    def _bloquear_exposicion_callback(self, msg: Bool):
        if self.cap is None or not self.cap.isOpened():
            return
        if bool(msg.data):
            self._aplicar_bloqueo_exposicion(motivo="calibración con panel MAPIR completada")
        else:
            try:
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
                self.exposicion_bloqueada = False
                self.get_logger().info("[MAPIR Driver] Auto-exposición reactivada para re-calibrar.")
            except Exception:
                pass

    def _intentar_abrir_puerto(self, indice: int):
        cap = cv2.VideoCapture(indice, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            return None

        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        ret, frame = cap.read()
        if not ret or frame is None:
            cap.release()
            return None

        return cap

    def _conectar_camara(self) -> bool:
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        candidatos = [self.puerto_preferido]
        if self.auto_detect_port:
            for p in (0, 1, 2, 3, 4):
                if p not in candidatos:
                    candidatos.append(p)

        for idx in candidatos:
            cap = self._intentar_abrir_puerto(idx)
            if cap is not None:
                self.cap = cap
                self.puerto_activo = idx
                self.fallos_consecutivos = 0
                self.frames_capturados = 0
                self.exposicion_bloqueada = False
                ancho_real = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                alto_real = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.get_logger().info(
                    f"Cámara MAPIR conectada en /dev/video{idx} "
                    f"({ancho_real}x{alto_real} @ {self.fps:.1f} FPS)"
                )
                return True

        self.get_logger().error(
            f"No se pudo abrir la cámara MAPIR en los puertos {candidatos}. "
            "Reintentando automáticamente..."
        )
        return False

    def timer_callback(self):
        if self.cap is None or not self.cap.isOpened():
            self._conectar_camara()
            return

        ret, frame = self.cap.read()
        if ret and frame is not None:
            self.fallos_consecutivos = 0
            self.frames_capturados += 1

            if (
                self.bloquear_auto_exposicion
                and not self.exposicion_bloqueada
                and self.frames_capturados == 20
            ):
                self._aplicar_bloqueo_exposicion(motivo="estabilización inicial de 20 frames")

            msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'mapir_link'
            self.publisher_.publish(msg)
        else:
            self.fallos_consecutivos += 1
            self.get_logger().warn(
                f"Fallo al capturar frame de la cámara MAPIR (intento {self.fallos_consecutivos})",
                throttle_duration_sec=2.0,
            )
            if self.fallos_consecutivos >= 15:
                self.get_logger().warn("Reiniciando conexión V4L2 con la cámara MAPIR...")
                self._conectar_camara()

    def cerrar(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None


def main(args=None):
    rclpy.init(args=args)
    nodo = MapirDriverReal()
    try:
        rclpy.spin(nodo)
    except KeyboardInterrupt:
        pass
    finally:
        nodo.cerrar()
        nodo.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

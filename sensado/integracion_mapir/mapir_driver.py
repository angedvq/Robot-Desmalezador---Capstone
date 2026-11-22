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
3. Reconexión automática ante micro-cortes USB.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


class MapirDriverReal(Node):
    def __init__(self):
        super().__init__('mapir_driver')

        # ---------------------------------------------------------------------
        # Parámetros configurables por launch o línea de comandos
        # ---------------------------------------------------------------------
        self.declare_parameter('video_port', 2)
        self.declare_parameter('frame_width', 1280)
        self.declare_parameter('frame_height', 720)
        self.declare_parameter('fps', 15.0)
        self.declare_parameter('auto_detect_port', True)

        self.puerto_preferido = int(self.get_parameter('video_port').value)
        self.frame_width = int(self.get_parameter('frame_width').value)
        self.frame_height = int(self.get_parameter('frame_height').value)
        self.fps = max(1.0, float(self.get_parameter('fps').value))
        self.auto_detect_port = bool(self.get_parameter('auto_detect_port').value)

        self.publisher_ = self.create_publisher(Image, '/image_raw', 10)
        self.bridge = CvBridge()

        self.cap = None
        self.puerto_activo = None
        self.fallos_consecutivos = 0

        # Abrir cámara MAPIR al iniciar
        self._conectar_camara()

        # Temporizador de captura según los FPS configurados
        periodo = 1.0 / self.fps
        self.timer = self.create_timer(periodo, self.timer_callback)

    def _intentar_abrir_puerto(self, indice: int):
        """Intenta abrir `/dev/video{indice}` con V4L2 y verifica que entregue un frame válido."""
        cap = cv2.VideoCapture(indice, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            return None

        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Leer un frame de prueba para descartar nodos de metadatos V4L2
        ret, frame = cap.read()
        if not ret or frame is None:
            cap.release()
            return None

        return cap

    def _conectar_camara(self) -> bool:
        """Conecta con la cámara MAPIR en el puerto indicado o busca en puertos alternativos."""
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

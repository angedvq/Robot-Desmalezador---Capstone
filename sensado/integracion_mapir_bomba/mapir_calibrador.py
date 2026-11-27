#!/usr/bin/env python3
"""
Herramienta interactiva de calibración con Panel de Reflectancia MAPIR y prueba en vivo
(`ros2 run sensado mapir_calibrador`).

Ejecuta la misma ventana GUI con detección automática del panel QR MAPIR (parches 87%R, 51%R, 23%R)
y la lógica directa de `procesamiento_nir.py` con barras deslizantes ajustables en tiempo real.
"""

import rclpy
from sensado.integracion_mapir_bomba.mapir_procesamiento_nir import MapirProcesadorMalezaNIR
import cv2


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

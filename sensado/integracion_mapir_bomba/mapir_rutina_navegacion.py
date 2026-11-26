#!/usr/bin/env python3
"""
Nodo `mapir_rutina_navegacion`:
Ejecuta la rutina de navegación con cámara MAPIR activa por defecto (`usar_mapir=True`)
y soporte configurable de bomba (`usar_bomba=True` o `False`).
"""

import rclpy
from sensado.integracion_mapir_bomba.bomba_rutina_navegacion import BombaRutinaNavegacion


class MapirRutinaNavegacion(BombaRutinaNavegacion):
    def __init__(self):
        super().__init__(
            nombre_nodo='mapir_rutina_navegacion',
            default_usar_mapir=True,
            default_usar_bomba=True,
        )


def main(args=None):
    rclpy.init(args=args)
    node = MapirRutinaNavegacion()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        if rclpy.ok():
            node.detener_todo_emergencia()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

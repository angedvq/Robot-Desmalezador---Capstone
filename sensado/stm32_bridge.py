import struct
import time
from collections import deque

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Float32
import serial


def trama(orden, valor=0.0):
    raw = max(-32767, min(32767, round(valor * 100)))
    cuerpo = struct.pack('<Bh', orden, raw)
    chk = cuerpo[0] ^ cuerpo[1] ^ cuerpo[2] ^ 0xA5
    return cuerpo + bytes([chk])


class STM32Bridge(Node):
    def __init__(self):
        super().__init__('stm32_bridge')
        self.declare_parameter('port', '/dev/serial0')
        self.uart = serial.Serial(self.get_parameter('port').value, 115200)

        self.velocidad = 0.0
        self.ultimo_cmd = 0.0
        self.bomba_encendida = False
        # Arranque: motores y bomba apagados, referencia 0° y seguimiento activo.
        self.pendientes = deque([(0, 0), (6, 0), (1, 0), (2, 0)])

        self.create_subscription(Twist, '/cmd_vel', self.recibir_velocidad, 10)
        self.create_subscription(Bool, '/bomba', self.recibir_bomba, 10)
        self.create_subscription(Float32, '/brazo_mm', self.recibir_brazo, 10)
        # Enviar una sola trama por ciclo, para separarlas en la UART.
        self.create_timer(0.05, self.enviar_pendiente)

    def recibir_velocidad(self, msg):
        self.velocidad = float(msg.linear.x)
        self.ultimo_cmd = time.monotonic()

    def recibir_bomba(self, msg):
        self.bomba_encendida = bool(msg.data)
        self.pendientes.append((5 if msg.data else 6, 0))

    def recibir_brazo(self, msg):
        self.pendientes.append((4, float(msg.data)))

    def enviar_pendiente(self):
        # Si dejan de llegar comandos ROS 2, pedir detención.
        if time.monotonic() - self.ultimo_cmd > 0.5:
            self.velocidad = 0.0
            if self.bomba_encendida:
                self.bomba_encendida = False
                # Descartar un encendido que pudiera estar todavía en cola.
                self.pendientes = deque(cmd for cmd in self.pendientes if cmd[0] != 5)
                self.pendientes.appendleft((6, 0))

        orden, valor = (self.pendientes.popleft() if self.pendientes
                        else (0, self.velocidad))
        self.uart.write(trama(orden, valor))

    def cerrar(self):
        # Detención solicitada al cerrar normalmente; no reemplaza un E-stop.
        self.uart.write(trama(0, 0))
        time.sleep(0.02)
        self.uart.write(trama(6, 0))
        self.uart.close()


def main(args=None):
    rclpy.init(args=args)
    nodo = None
    try:
        nodo = STM32Bridge()
        rclpy.spin(nodo)
    except KeyboardInterrupt:
        pass
    finally:
        if nodo is not None:
            nodo.cerrar()
            nodo.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

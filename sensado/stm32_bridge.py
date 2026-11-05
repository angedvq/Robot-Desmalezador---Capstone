import struct
import time
from collections import deque

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Float32
import serial


# Códigos de orden según "Protocolo RPi -> STM32 (USART2) V2"
ORD_SET_VEL = 0         # velocidad (m/s)
ORD_SET_HEADING = 1     # rumbo objetivo (grados, +-180)
ORD_RESET_TRACKING = 2  # reinicia el tracking de la IMU
ORD_START_TRACKING = 3  # enciende la IMU y empieza a medir
ORD_STOP_TRACKING = 4   # apaga la IMU mientras está detenido
ORD_MOVE_STEPPER = 5    # desplazamiento stepper (cm)
ORD_STOP_STEPPER = 6    # detiene el stepper
ORD_START_BOMBA = 7     # enciende la bomba
ORD_STOP_BOMBA = 8      # apaga la bomba
ORD_STOP_ALL = 9        # detiene todo: movimiento, stepper y bomba
ORD_START_MOV = 10      # inicia movimiento (m/s), fija rumbo actual y enciende tracking
ORD_STOP_MOV = 11       # detiene movimiento, apaga tracking y pone velocidad en 0

ERRORES_STM32 = {
    0: "Error de comunicación",
    1: "Caída de voltaje",
    2: "Fallo bomba",
}


def trama(orden, valor=0.0):
    # Multiplica por 100 (ej. 0.20 m/s -> 20) y empaqueta en binario little-endian
    raw = max(-32767, min(32767, round(valor * 100)))
    cuerpo = struct.pack('<Bh', orden, raw)
    # Checksum XOR para integridad de datos
    chk = cuerpo[0] ^ cuerpo[1] ^ cuerpo[2] ^ 0xA5
    return cuerpo + bytes([chk])


class STM32Bridge(Node):
    def __init__(self):
        super().__init__('stm32_bridge')
        self.declare_parameter('port', '/dev/ttyAMA1')
        self.uart = serial.Serial(self.get_parameter('port').value, 115200, timeout=0.0)
        self.rx_buf = bytearray()

        self.velocidad = 0.0
        self.ultimo_rumbo = 0.0
        self.ultimo_cmd = 0.0
        self.bomba_encendida = False
        self.stop_solicitado = False
        self.en_movimiento = False
        self.primer_arranque = True

        # Arranque seguro en V2: todo detenido, IMU reseteada en 0° y referencia inicial 0.0°
        self.pendientes = deque([
            (ORD_STOP_ALL, 0.0),
            (ORD_RESET_TRACKING, 0.0),
            (ORD_SET_HEADING, 0.0),
        ])

        # Suscriptores
        self.create_subscription(Twist, '/cmd_vel', self.recibir_velocidad, 10)
        self.create_subscription(Bool, '/navegacion/stop', self.recibir_stop, 10)
        self.create_subscription(Float32, '/navegacion/corregir_grados', self.recibir_correccion, 10)
        self.create_subscription(Bool, '/bomba', self.recibir_bomba, 10)
        self.create_subscription(Float32, '/brazo_mm', self.recibir_brazo, 10)

        # Enviar una sola trama por ciclo (20 Hz)
        self.create_timer(0.05, self.enviar_pendiente)

        self.get_logger().info("Bridge STM32 (Protocolo V2) iniciado en modo binario.")

    def recibir_velocidad(self, msg):
        self.velocidad = float(msg.linear.x)
        self.ultimo_cmd = time.monotonic()

    def recibir_stop(self, msg):
        self.stop_solicitado = bool(msg.data)
        self.ultimo_cmd = time.monotonic()
        if self.stop_solicitado:
            self.velocidad = 0.0

    def recibir_correccion(self, msg):
        self.ultimo_rumbo = max(-180.0, min(180.0, float(msg.data)))
        if self.en_movimiento and not self.stop_solicitado:
            self.pendientes.append((ORD_SET_HEADING, self.ultimo_rumbo))

    def recibir_bomba(self, msg):
        self.bomba_encendida = bool(msg.data)
        self.pendientes.append((ORD_START_BOMBA if self.bomba_encendida else ORD_STOP_BOMBA, 0.0))

    def recibir_brazo(self, msg):
        # En Protocolo V2 el desplazamiento del stepper va en cm (10 mm = 1 cm)
        self.pendientes.append((ORD_MOVE_STEPPER, float(msg.data) / 10.0))

    def _leer_errores_stm32(self):
        try:
            waiting = self.uart.in_waiting
            if waiting > 0:
                self.rx_buf.extend(self.uart.read(waiting))
        except Exception:
            return

        while len(self.rx_buf) >= 2:
            b0, b1 = self.rx_buf[0], self.rx_buf[1]
            if (b0 ^ 0xA5) == b1 and b0 in ERRORES_STM32:
                self.get_logger().error(
                    f"STM32 reportó STOP_All por error ({b0}): {ERRORES_STM32[b0]}"
                )
                del self.rx_buf[:2]
            else:
                del self.rx_buf[:1]

    def enviar_pendiente(self):
        self._leer_errores_stm32()

        # Watchdog: Si dejan de llegar comandos ROS 2 (> 0.5 s), detener todo
        if self.ultimo_cmd > 0.0 and (time.monotonic() - self.ultimo_cmd) > 0.5:
            self.velocidad = 0.0
            self.bomba_encendida = False

        debe_parar = self.stop_solicitado or abs(self.velocidad) < 1e-4

        if debe_parar:
            if self.en_movimiento:
                self.en_movimiento = False
                # Limpiar órdenes de movimiento/rumbo en cola y enviar STOP_All de inmediato
                ordenes_mov = (ORD_SET_VEL, ORD_SET_HEADING, ORD_START_TRACKING, ORD_START_MOV)
                self.pendientes = deque(cmd for cmd in self.pendientes if cmd[0] not in ordenes_mov)
                self.pendientes.appendleft((ORD_STOP_ALL, 0.0))

            orden, valor = (self.pendientes.popleft() if self.pendientes
                            else (ORD_STOP_ALL, 0.0))
        else:
            if not self.en_movimiento:
                self.en_movimiento = True
                if self.primer_arranque:
                    # Primer arranque desde reposo alineado a la pared: START_Mov fija 0° y enciende IMU
                    self.primer_arranque = False
                    self.pendientes.append((ORD_START_MOV, self.velocidad))
                    if abs(self.ultimo_rumbo) >= 0.01:
                        self.pendientes.append((ORD_SET_HEADING, self.ultimo_rumbo))
                else:
                    # Reanudación tras obstáculo: encender IMU sin reiniciar la referencia de 0° original
                    self.pendientes.append((ORD_START_TRACKING, 0.0))
                    self.pendientes.append((ORD_SET_HEADING, self.ultimo_rumbo))
                    self.pendientes.append((ORD_SET_VEL, self.velocidad))

            orden, valor = (self.pendientes.popleft() if self.pendientes
                            else (ORD_SET_VEL, self.velocidad))

        try:
            self.uart.write(trama(orden, valor))
        except Exception as e:
            self.get_logger().error(f"Error UART: {e}")

    def cerrar(self):
        try:
            self.uart.write(trama(ORD_STOP_ALL, 0.0))
            time.sleep(0.02)
            self.uart.close()
        except Exception:
            pass


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
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

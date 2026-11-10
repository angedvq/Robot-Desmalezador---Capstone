import struct
import time
from collections import deque

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Float32
import serial

# ===============================================
# Códigos de orden 
# ===============================================
SET_VEL = 0         # velocidad (m/s)
SET_HEADING = 1     # rumbo objetivo (grados, +-180)
RESET_TRACKING = 2  # reinicia el tracking de la IMU
START_TRACKING = 3  # enciende la IMU y empieza a medir
STOP_TRACKING = 4   # apaga la IMU mientras está detenido
MOVE_STEPPER = 5    # desplazamiento stepper (cm)
STOP_STEPPER = 6    # detiene el stepper
START_BOMBA = 7     # enciende la bomba
STOP_BOMBA = 8      # apaga la bomba
STOP_ALL = 9        # detiene todo: movimiento, stepper y bomba
START_MOV = 10      # inicia movimiento (m/s), fija rumbo actual y enciende tracking
STOP_MOV = 11       # detiene movimiento, apaga tracking y pone velocidad en 0

ERRORES_STM32 = {
    0: "Error de comunicación",
    1: "Caída de voltaje",
    2: "Fallo bomba", # falta implementar
}


def trama(orden, valor=0.0):
    '''Convierte una orden y un numero decimal en un paquete de 4 bytes segun el protocolo de comunicacion'''

    # Multiplica por 100 y empaqueta en binario little-endian
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

        # Buffer para guardar temporalmente las respuestas que lleguen desde la stm32
        self.rx_buf = bytearray()

        # Variables de estado
        self.velocidad = 0.0
        self.ultimo_rumbo = 0.0
        self.ultimo_cmd = 0.0   # timestamp del ultimo comando recibido
        self.bomba_encendida = False
        self.stop_solicitado = False
        self.en_movimiento = False
        self.primer_arranque = True

        # Cola de tareas pendientes - deque() permite enviar tareas de forma consecutiva y ordenada para no saturar el puerto serial
        # Arranque seguro: todo detenido, IMU reseteada en 0° y referencia inicial 0.0°
        self.pendientes = deque([
            (STOP_ALL, 0.0),
            (RESET_TRACKING, 0.0),
            (SET_HEADING, 0.0),
        ])

        # Suscriptores
        self.create_subscription(Twist, '/cmd_vel', self.recibir_velocidad, 10)
        self.create_subscription(Bool, '/navegacion/stop', self.recibir_stop, 10)
        self.create_subscription(Float32, '/navegacion/corregir_grados', self.recibir_correccion, 10)
        self.create_subscription(Bool, '/bomba', self.recibir_bomba, 10)
        self.create_subscription(Float32, '/brazo_mm', self.recibir_brazo, 10)

        # Enviar una sola trama por ciclo. Ejecuta enviar_pendientes cada 0.05 segundos (20 Hz)
        self.create_timer(0.05, self.enviar_pendiente)

        self.get_logger().info("Bridge STM32 iniciado en modo binario")

    # ===============================================
    # Callbacks
    # ===============================================
    def recibir_velocidad(self, msg):
        # Extrae la velocidad de avance frontal desde /cmd/vel
        self.velocidad = float(msg.linear.x)
        self.ultimo_cmd = time.monotonic()

    def recibir_stop(self, msg):
        # Si se envia un Stop desde /navegacion/stop, se fuerza la velocidad a 0
        self.stop_solicitado = bool(msg.data)
        self.ultimo_cmd = time.monotonic()

        if self.stop_solicitado:
            self.velocidad = 0.0

    def recibir_correccion(self, msg):
        # Recibe los grados que el robot debe corregir desde /navegacion/corregir_grados (con un max de 180° para no saturar)
        self.ultimo_rumbo = max(-180.0, min(180.0, float(msg.data)))

        # Se agrega la correccion de grados a la cola de tareas pendientes solo si el robot está en movimiento
        if self.en_movimiento and not self.stop_solicitado:
            self.pendientes.append((SET_HEADING, self.ultimo_rumbo))

    def recibir_bomba(self, msg):
        # Si recibe la orden de encender la bomba desde /bomba, la enciende. Caso contrario la manteniene apagada
        self.bomba_encendida = bool(msg.data)
        self.pendientes.append((START_BOMBA if self.bomba_encendida else STOP_BOMBA, 0.0))

    def recibir_brazo(self, msg):
        # Mueve el brazo segun la orden desde /brazo_mm. Se divide en 10 para que sea en cms
        self.pendientes.append((MOVE_STEPPER, float(msg.data) / 10.0))

    # ===============================================
    # Checkeo de errores
    # ===============================================
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

    # ===============================================
    # Bucle Principal
    # ===============================================
    def enviar_pendiente(self):
        self._leer_errores_stm32()

        # Watchdog: Si dejan de llegar comandos de ROS (> 0.5 s), detener todo
        if self.ultimo_cmd > 0.0 and (time.monotonic() - self.ultimo_cmd) > 0.5:
            self.velocidad = 0.0
            self.bomba_encendida = False

        # Flag de detencion del robot. Ya sea por comando de stop recibido o por una caida hacia 0 en velocidad
        debe_parar = self.stop_solicitado or abs(self.velocidad) < 1e-4

        # Si el robot debe parar y está en movimiento
        if debe_parar:
            if self.en_movimiento:
                self.en_movimiento = False
                # Limpiar órdenes de movimiento/rumbo en cola y enviar STOP_All 
                ordenes_mov = (SET_VEL, SET_HEADING, START_TRACKING, START_MOV)
                self.pendientes = deque(cmd for cmd in self.pendientes if cmd[0] not in ordenes_mov)

                # Agregar al principio de la cola de tareas STOP_ALL (máx prioridad)
                self.pendientes.appendleft((STOP_ALL, 0.0))

            # Si la cola tiene tareas, sacamos la primera. Si esta vacia se envia STOP_ALL
            orden, valor = (self.pendientes.popleft() if self.pendientes else (STOP_ALL, 0.0))

        # Si el robot debe moverse
        else:
            if not self.en_movimiento:
                self.en_movimiento = True
                if self.primer_arranque:
                    # Primer arranque desde reposo alineado a la pared: START_Mov fija 0° y enciende IMU
                    self.primer_arranque = False
                    self.pendientes.append((START_MOV, self.velocidad))

                    if abs(self.ultimo_rumbo) >= 0.01:
                        self.pendientes.append((SET_HEADING, self.ultimo_rumbo))
                else:
                    # Reanudación tras obstáculo: encender IMU sin reiniciar la referencia de 0° original
                    self.pendientes.append((START_TRACKING, 0.0))
                    self.pendientes.append((SET_HEADING, self.ultimo_rumbo))
                    self.pendientes.append((SET_VEL, self.velocidad))

            # Sacamos la tarea de la fila. Si esta vacia enviamos la velocidad actual
            orden, valor = (self.pendientes.popleft() if self.pendientes else (SET_VEL, self.velocidad))

        # Enviar orden a STM32
        try:
            self.uart.write(trama(orden, valor))
        except Exception as e:
            self.get_logger().error(f"Error UART: {e}")

    def cerrar(self):
        try:
            self.uart.write(trama(STOP_ALL, 0.0))
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

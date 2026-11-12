#!/usr/bin/env python3
"""
===============================================================================
NODO DE SIMULACIÓN: sim_stm32_bridge.py
===============================================================================
Ubicación: `sensado/simulaciones/sim_stm32_bridge.py`
Equivalente en simulación de `real_stm32_bridge.py`.

PROPÓSITO:
1. Ejecutar exactamente la misma lógica de colas y empaquetado binario de 4
   bytes (Protocolo RPi -> STM32 V2) que `real_stm32_bridge.py`:
   - `STOP_Mov` (Orden 11) al detenerse frente a un árbol para permitir mover
     el brazo stepper sin abortarlo.
   - `MOVE_Stepper` (Orden 5) con `+X cm` para extender el brazo y `-X cm` para
     retraerlo.
   - `START_Mov` (Orden 10) en el primer arranque y `START_Tracking` (Orden 3)
     al reanudar marcha tras cada árbol para conservar la referencia IMU de 0°.
   - `STOP_All` (Orden 9) ante obstáculos frontales, watchdogs o fin de rutina.
2. En una PC de simulación donde no existe el puerto físico `/dev/ttyAMA1`,
   crea automáticamente un puerto serie virtual (PTY) y drena las tramas en un
   hilo secundario para que la transmisión UART simulada funcione sin errores.
3. Además, traduce las órdenes `MOVE_Stepper` y `STOP_All` al tópico de Gazebo
   `/brazo/cmd_pos` (`std_msgs/msg/Float64`), moviendo físicamente la
   articulación prismática `brazo_stepper_joint` del brazo trasero en Gazebo.
===============================================================================
"""

import os
import pty
import struct
import threading
import time
from collections import deque

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Float32, Float64
import serial

# =============================================================================
# Códigos de orden del Protocolo RPi -> STM32 (USART2) V2
# =============================================================================
SET_VEL = 0         # Fija velocidad de avance de referencia (m/s)
SET_HEADING = 1     # Fija rumbo objetivo del robot (grados, +-180°)
RESET_TRACKING = 2  # Reinicia el tracking de la IMU
START_TRACKING = 3  # Enciende la IMU y empieza a medir
STOP_TRACKING = 4   # Apaga la IMU mientras está detenido
MOVE_STEPPER = 5    # Mueve el brazo stepper (cm: + extiende, - retrae)
STOP_STEPPER = 6    # Detiene el stepper
START_BOMBA = 7     # Enciende la bomba
STOP_BOMBA = 8      # Apaga la bomba
STOP_ALL = 9        # Detiene todo: ruedas, stepper y bomba
START_MOV = 10      # Inicia movimiento (m/s), fija rumbo actual como 0° y enciende IMU
STOP_MOV = 11       # Detiene solo las ruedas, mantiene rumbo actual y apaga IMU


def trama(orden: int, valor: float = 0.0) -> bytes:
    """Empaqueta una orden y valor flotante en los 4 bytes del Protocolo V2."""
    raw = max(-32767, min(32767, round(float(valor) * 100.0)))
    cuerpo = struct.pack('<Bh', int(orden), int(raw))
    chk = cuerpo[0] ^ cuerpo[1] ^ cuerpo[2] ^ 0xA5
    return cuerpo + bytes([chk])


class SimSTM32Bridge(Node):
    """
    Puente STM32 para simulación en Gazebo Fortress: empaqueta las tramas reales
    del protocolo V2 sobre un PTY virtual y acciona la articulación del brazo
    trasero en Gazebo (`/brazo/cmd_pos`).
    """

    def __init__(self):
        super().__init__('sim_stm32_bridge')

        self.declare_parameter('port', '/dev/ttyAMA1')
        self.declare_parameter('baudrate', 115200)
        self.declare_parameter('recorrido_max_brazo_m', 0.22)

        port = str(self.get_parameter('port').value)
        baudrate = int(self.get_parameter('baudrate').value)
        self.recorrido_max_brazo_m = float(
            self.get_parameter('recorrido_max_brazo_m').value
        )

        # Si el puerto físico no existe en la PC de simulación, creamos un PTY virtual
        self._pty_master_fd = None
        self._pty_slave_fd = None
        if not os.path.exists(port):
            self._pty_master_fd, self._pty_slave_fd = pty.openpty()
            port = os.ttyname(self._pty_slave_fd)
            threading.Thread(
                target=self._drenar_pty, args=(self._pty_master_fd,), daemon=True
            ).start()

        self.uart = serial.Serial(port, baudrate, timeout=0.0)

        # Variables de estado idénticas a `real_stm32_bridge.py`
        self.velocidad = 0.0
        self.ultimo_rumbo = 0.0
        self.ultimo_cmd = 0.0
        self.bomba_encendida = False
        self.stop_solicitado = False
        self.en_movimiento = False
        self.primer_arranque = True

        self.pendientes = deque([
            (STOP_ALL, 0.0),
            (RESET_TRACKING, 0.0),
            (SET_HEADING, 0.0),
        ])

        # Publicador hacia el controlador prismático del brazo trasero en Gazebo
        self.pub_brazo_gz = self.create_publisher(Float64, '/brazo/cmd_pos', 10)

        # Suscriptores desde `sim_rutina_navegacion`
        self.create_subscription(Twist, '/cmd_vel', self.recibir_velocidad, 10)
        self.create_subscription(Bool, '/navegacion/stop', self.recibir_stop, 10)
        self.create_subscription(
            Float32, '/navegacion/corregir_grados', self.recibir_correccion, 10
        )
        self.create_subscription(Bool, '/bomba', self.recibir_bomba, 10)
        self.create_subscription(Float32, '/brazo_mm', self.recibir_brazo, 10)

        # Ciclo de transmisión a 20 Hz
        self.create_timer(0.05, self.enviar_pendiente)

        self.get_logger().info(
            f"sim_stm32_bridge iniciado en puerto virtual {port} | "
            "Protocolo V2 + Actuador de brazo en Gazebo (/brazo/cmd_pos) ACTIVOS"
        )

    @staticmethod
    def _drenar_pty(fd: int):
        while True:
            try:
                data = os.read(fd, 1024)
                if not data:
                    break
            except OSError:
                break

    def _mover_brazo_gazebo(self, posicion_m: float):
        """Envía la posición objetivo (m) al controlador prismático de Gazebo."""
        msg = Float64()
        msg.data = float(posicion_m)
        self.pub_brazo_gz.publish(msg)

    # =========================================================================
    # Callbacks de tópicos ROS 2
    # =========================================================================
    def recibir_velocidad(self, msg: Twist):
        self.velocidad = float(msg.linear.x)
        self.ultimo_cmd = time.monotonic()

    def recibir_stop(self, msg: Bool):
        nuevo_stop = bool(msg.data)
        self.ultimo_cmd = time.monotonic()

        if nuevo_stop and not self.stop_solicitado:
            self.velocidad = 0.0
            self.en_movimiento = False
            self.pendientes.clear()
            self.pendientes.appendleft((STOP_ALL, 0.0))

        self.stop_solicitado = nuevo_stop

    def recibir_correccion(self, msg: Float32):
        self.ultimo_rumbo = max(-180.0, min(180.0, float(msg.data)))
        self.ultimo_cmd = time.monotonic()

        if self.en_movimiento and not self.stop_solicitado:
            self.pendientes.append((SET_HEADING, self.ultimo_rumbo))

    def recibir_bomba(self, msg: Bool):
        self.bomba_encendida = bool(msg.data)
        self.ultimo_cmd = time.monotonic()
        self.pendientes.append((START_BOMBA if self.bomba_encendida else STOP_BOMBA, 0.0))

    def recibir_brazo(self, msg: Float32):
        self.ultimo_cmd = time.monotonic()
        desplazamiento_mm = float(msg.data)

        if abs(desplazamiento_mm) < 1e-3:
            self.pendientes.append((STOP_STEPPER, 0.0))
        else:
            desplazamiento_cm = desplazamiento_mm / 10.0
            self.pendientes.append((MOVE_STEPPER, desplazamiento_cm))
            accion = "EXTENDER" if desplazamiento_cm > 0.0 else "RETRAER"
            self.get_logger().info(
                f"[STM32-SIM] Orden brazo encolada: {accion} "
                f"(MOVE_Stepper = {desplazamiento_cm:+.1f} cm)"
            )

    # =========================================================================
    # Ciclo de transmisión de tramas (20 Hz)
    # =========================================================================
    def enviar_pendiente(self):
        sin_comandos_ros = (
            self.ultimo_cmd > 0.0 and (time.monotonic() - self.ultimo_cmd) > 0.6
        )
        if sin_comandos_ros:
            self.velocidad = 0.0
            self.stop_solicitado = True

        if self.stop_solicitado:
            if self.en_movimiento:
                self.en_movimiento = False
                self.pendientes.clear()
                self.pendientes.appendleft((STOP_ALL, 0.0))

            orden, valor = (
                self.pendientes.popleft() if self.pendientes else (STOP_ALL, 0.0)
            )

        elif abs(self.velocidad) < 1e-4:
            if self.en_movimiento:
                self.en_movimiento = False
                ordenes_mov = (SET_VEL, SET_HEADING, START_TRACKING, START_MOV)
                self.pendientes = deque(
                    cmd for cmd in self.pendientes if cmd[0] not in ordenes_mov
                )
                self.pendientes.appendleft((STOP_MOV, 0.0))

            orden, valor = (
                self.pendientes.popleft() if self.pendientes else (STOP_MOV, 0.0)
            )

        else:
            if not self.en_movimiento:
                self.en_movimiento = True
                if self.primer_arranque:
                    self.primer_arranque = False
                    self.pendientes.append((START_MOV, self.velocidad))
                    if abs(self.ultimo_rumbo) >= 0.01:
                        self.pendientes.append((SET_HEADING, self.ultimo_rumbo))
                else:
                    self.pendientes.append((START_TRACKING, 0.0))
                    self.pendientes.append((SET_HEADING, self.ultimo_rumbo))
                    self.pendientes.append((SET_VEL, self.velocidad))

            orden, valor = (
                self.pendientes.popleft()
                if self.pendientes
                else (SET_VEL, self.velocidad)
            )

        # Reflejar las órdenes del stepper en la articulación prismática de Gazebo
        if orden == MOVE_STEPPER:
            if valor > 0.0:
                self._mover_brazo_gazebo(self.recorrido_max_brazo_m)
            elif valor < 0.0:
                self._mover_brazo_gazebo(0.0)
        elif orden == STOP_ALL:
            self._mover_brazo_gazebo(0.0)

        try:
            self.uart.write(trama(orden, valor))
        except Exception as e:
            self.get_logger().error(f"Error en UART virtual STM32: {e}")

    def cerrar(self):
        try:
            self.uart.write(trama(STOP_ALL, 0.0))
            self.uart.close()
        except Exception:
            pass
        for fd in (self._pty_master_fd, self._pty_slave_fd):
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass


def main(args=None):
    rclpy.init(args=args)
    nodo = None
    try:
        nodo = SimSTM32Bridge()
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

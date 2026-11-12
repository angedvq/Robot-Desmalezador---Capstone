#!/usr/bin/env python3
import struct
import time
from collections import deque

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Float32
import serial

# =============================================================================
# Códigos de orden 
# =============================================================================

SET_VEL = 0         # Fija velocidad de avance de referencia (m/s)
SET_HEADING = 1     # Fija rumbo objetivo del robot (grados, rango +-180°)
RESET_TRACKING = 2  # Reinicia el tracking de la IMU
START_TRACKING = 3  # Enciende la IMU y empieza a medir sin borrar referencia 0°
STOP_TRACKING = 4   # Apaga la IMU para no integrar deriva mientras está detenido
MOVE_STEPPER = 5    # Mueve el brazo stepper (desplazamiento en cm: + extiende, - retrae)
STOP_STEPPER = 6    # Detiene el stepper inmediatamente
START_BOMBA = 7     # Enciende la bomba
STOP_BOMBA = 8      # Apaga la bomba
STOP_ALL = 9        # Detiene todo: ruedas, stepper y bomba
START_MOV = 10      # Inicia movimiento (m/s), fija rumbo actual como 0° y enciende IMU
STOP_MOV = 11       # Detiene solo las ruedas, mantiene rumbo actual y apaga IMU

ERRORES_STM32 = {
    0: "Error de comunicación",
    1: "Caída de voltaje",
    2: "Fallo bomba",
}


def trama(orden: int, valor: float = 0.0) -> bytes:
    '''Convierte una orden y un numero decimal en un paquete de 4 bytes segun el protocolo de comunicacion'''
    raw = max(-32767, min(32767, round(float(valor) * 100.0)))
    cuerpo = struct.pack('<Bh', int(orden), int(raw))
    chk = cuerpo[0] ^ cuerpo[1] ^ cuerpo[2] ^ 0xA5
    return cuerpo + bytes([chk])


class RealSTM32Bridge(Node):
    def __init__(self):
        super().__init__('real_stm32_bridge')

        self.declare_parameter('port', '/dev/ttyAMA1')
        self.declare_parameter('baudrate', 115200)

        port = str(self.get_parameter('port').value)
        baudrate = int(self.get_parameter('baudrate').value)
        self.uart = serial.Serial(port, baudrate, timeout=0.0)

        # Buffer de recepción para tramas de error 
        self.rx_buf = bytearray()

        # Estado interno
        self.velocidad = 0.0
        self.ultimo_rumbo = 0.0
        self.ultimo_cmd = 0.0
        self.bomba_encendida = False
        self.stop_solicitado = False
        self.en_movimiento = False
        self.primer_arranque = True

        # Cola FIFO de órdenes pendientes prioritarias
        # Al iniciar el nodo, nos aseguramos de que todo esté detenido y la IMU quede inicializada en 0.0°.
        # deque() permite enviar tareas de forma consecutiva y ordenada para no saturar el puerto serial
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

        self.create_timer(0.05, self.enviar_pendiente) 

        self.get_logger().info(
            f"real_stm32_bridge iniciado en {port} ({baudrate} bps) | "
            "Soporte STOP_Mov (árboles/brazo) + STOP_All (emergencia)"
        )

    # =========================================================================
    # Callbacks
    # =========================================================================
    def recibir_velocidad(self, msg: Twist):
        """Actualiza la velocidad lineal de referencia recibida en /cmd_vel."""
        self.velocidad = float(msg.linear.x)
        self.ultimo_cmd = time.monotonic()

    def recibir_stop(self, msg: Bool):
        """Recibe bandera de parada total /navegacion/stop"""
        nuevo_stop = bool(msg.data)
        self.ultimo_cmd = time.monotonic()

        # Si se activa el STOP de emergencia, encolar STOP_All al frente
        if nuevo_stop and not self.stop_solicitado:
            self.velocidad = 0.0
            self.en_movimiento = False
            self.pendientes.clear()
            self.pendientes.appendleft((STOP_ALL, 0.0))

        self.stop_solicitado = nuevo_stop

    def recibir_correccion(self, msg: Float32):
        """ Recibe el rumbo objetivo desde /navegacion/corregir_grados"""

        # max de 180° para no saturar
        self.ultimo_rumbo = max(-180.0, min(180.0, float(msg.data)))
        self.ultimo_cmd = time.monotonic()

        if self.en_movimiento and not self.stop_solicitado:
            self.pendientes.append((SET_HEADING, self.ultimo_rumbo))

    def recibir_bomba(self, msg: Bool):
        """Enciende o apaga la bomba (falta implementar la MAPIR)"""
        self.bomba_encendida = bool(msg.data)
        self.ultimo_cmd = time.monotonic()
        self.pendientes.append((START_BOMBA if self.bomba_encendida else STOP_BOMBA, 0.0))

    def recibir_brazo(self, msg: Float32):
        """
        Recibe orden de movimiento del brazo desde /brazo_mm y la convierte a centímetros 
        - Valor positivo (+X mm -> +X/10 cm): Extiende el brazo hasta fin de carrera.
        - Valor negativo (-X mm -> -X/10 cm): Retrae el brazo hasta fin de carrera.
        - Valor 0.0: Envía `STOP_Stepper` (Orden 6).
        """
        self.ultimo_cmd = time.monotonic()
        desplazamiento_mm = float(msg.data)

        # Si las ruedas seguían en movimiento al llegar la orden de brazo,
        # asegurar que STOP_Mov se envíe ANTES que MOVE_Stepper
        if self.en_movimiento:
            self.velocidad = 0.0
            self.en_movimiento = False
            ordenes_mov = (SET_VEL, SET_HEADING, START_TRACKING, START_MOV)
            self.pendientes = deque(cmd for cmd in self.pendientes if cmd[0] not in ordenes_mov)
            self.pendientes.appendleft((STOP_MOV, 0.0))

        # Si el valor de extension del brazo se acerca a 0, abortar movimiento
        if abs(desplazamiento_mm) < 1e-3:
            self.pendientes.append((STOP_STEPPER, 0.0))
        else:
            desplazamiento_cm = desplazamiento_mm / 10.0
            self.pendientes.append((MOVE_STEPPER, desplazamiento_cm))
            accion = "EXTENDER" if desplazamiento_cm > 0.0 else "RETRAER"
            self.get_logger().info(
                f"Orden brazo añadida: {accion} (MOVE_Stepper = {desplazamiento_cm:+.1f} cm)"
            )

    # =========================================================================
    # Lectura de diagnósticos enviados por la STM32
    # =========================================================================
    def _leer_errores_stm32(self):
        """Lee paquetes de error desde la STM32."""
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

    # =========================================================================
    # Loop principal
    # =========================================================================
    def enviar_pendiente(self):
        self._leer_errores_stm32()

        # Watchdog: Si dejan de llegar comandos de ROS (> 0.5 s), detener todo
        sin_comandos = (self.ultimo_cmd > 0.0 and (time.monotonic() - self.ultimo_cmd) > 0.6)

        if sin_comandos:
            self.velocidad = 0.0
            self.stop_solicitado = True

        # ---------------------------------------------------------------------
        # Parada total por obstáculo, fin o watchdog
        # ---------------------------------------------------------------------
        if self.stop_solicitado:
            if self.en_movimiento:
                self.en_movimiento = False
                self.pendientes.clear()
                self.pendientes.appendleft((STOP_ALL, 0.0))

            orden, valor = (self.pendientes.popleft() if self.pendientes else (STOP_ALL, 0.0))

        # ---------------------------------------------------------------------
        # Parada de movimiento frente un árbol
        # ---------------------------------------------------------------------
        elif abs(self.velocidad) < 1e-4:
            if self.en_movimiento:
                self.en_movimiento = False

                # Eliminar órdenes de avance pendientes pero conservar órdenes de brazo/bomba
                ordenes_mov = (SET_VEL, SET_HEADING, START_TRACKING, START_MOV)
                self.pendientes = deque(cmd for cmd in self.pendientes if cmd[0] not in ordenes_mov)

                # Detener las ruedas y pausar IMU con STOP_Mov
                self.pendientes.appendleft((STOP_MOV, 0.0))

            # Si hay órdenes en cola las enviamos; si la cola está vacía, mantenemos STOP_Mov
            orden, valor = (self.pendientes.popleft() if self.pendientes else (STOP_MOV, 0.0))

        # ---------------------------------------------------------------------
        # Robot avanzando hacia el siguiente árbol
        # ---------------------------------------------------------------------
        else:
            if not self.en_movimiento:
                self.en_movimiento = True
                if self.primer_arranque:
                    # Primer arranque del recorrido: fija 0° paralelo a la hilera y enciende IMU
                    self.primer_arranque = False
                    self.pendientes.append((START_MOV, self.velocidad))
                    if abs(self.ultimo_rumbo) >= 0.01:
                        self.pendientes.append((SET_HEADING, self.ultimo_rumbo))
                else:
                    # Reanudación tras árbol u obstáculo: conserva la referencia 0° original
                    self.pendientes.append((START_TRACKING, 0.0))
                    self.pendientes.append((SET_HEADING, self.ultimo_rumbo))
                    self.pendientes.append((SET_VEL, self.velocidad))

            orden, valor = (self.pendientes.popleft() if self.pendientes else (SET_VEL, self.velocidad))

        try:
            self.uart.write(trama(orden, valor))
        except Exception as e:
            self.get_logger().error(f"Error escribiendo en UART STM32: {e}")

    def cerrar(self):
        """Detiene todos los actuadores y cierra el puerto serial al apagar el nodo."""
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
        nodo = RealSTM32Bridge()
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

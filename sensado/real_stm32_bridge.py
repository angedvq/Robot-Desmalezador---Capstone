#!/usr/bin/env python3
"""
real_stm32_bridge_v2: misma lógica de envío que real_stm32_bridge, con la
recepción reescrita.

Protocolo STM32 -> RPi: tramas de 2 bytes (uint8 cada uno)
    [codigo, codigo ^ 0xA5]

Problemas del receptor anterior que se corrigen aquí:
  1. Eco de las propias tramas TX (4 bytes). Si el UART tiene loopback / eco
     (TX y RX puenteados, adaptador half-duplex o la STM reenviando lo que
     recibe), por RX llegan de vuelta las tramas de 4 bytes que manda la RPi.
     El parser viejo las recorría byte a byte y podía confundir pedazos de
     ellas con mensajes válidos, p. ej.:
        trama(SET_HEADING, 0.01) = 01 01 00 A5  -> "00 A5" = Error comunicación
        trama(SET_HEADING, 1.64) = 01 A4 00 ..  -> "01 A4" = Caída de voltaje
     Ahora se reconocen las tramas propias enviadas recientemente y se descartan
     completas (4 bytes) antes de buscar mensajes de la STM32.
  2. Los bytes descartados se perdían en silencio (y el print() inundaba la
     consola). Ahora se acumulan y se reportan en hexadecimal, con throttle,
     para ver exactamente qué está llegando.
  3. Mensajes duplicados (la STM enviando la misma trama 2 veces = 4 bytes)
     se filtran con una ventana de tiempo configurable.
  4. La lectura del UART corre en su propio timer (200 Hz), independiente del
     envío a 20 Hz.

Parámetros:
  port               (str)   puerto serie                 [/dev/ttyAMA1]
  baudrate           (int)                                [115200]
  filtrar_eco        (bool)  descarta eco de tramas TX    [True]
  ventana_duplicado  (float) s, ignora mismo código repetido dentro de esta ventana [0.15]
  debug_rx           (bool)  imprime en hex todo lo que llega por RX [False]
"""
import struct
import time
from collections import deque

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Float32, Int32
import serial

# =============================================================================
# Códigos de orden RPi -> STM32 (tramas de 4 bytes: [orden, int16 LE, chk])
# =============================================================================
SET_VEL = 0
SET_HEADING = 1
RESET_TRACKING = 2
START_TRACKING = 3
STOP_TRACKING = 4
MOVE_STEPPER = 5
STOP_STEPPER = 6
START_BOMBA = 7
STOP_BOMBA = 8
STOP_ALL = 9
START_MOV = 10
STOP_MOV = 11

# =============================================================================
# Códigos STM32 -> RPi (tramas de 2 bytes: [codigo, codigo ^ 0xA5])
# =============================================================================
CHK_XOR = 0xA5
BRAZO_RETRAIDO = 3
BRAZO_EXTENDIDO = 4

MENSAJES_STM32 = {
    0: "Error de comunicación",
    1: "Caída de voltaje",
    2: "Fallo bomba",
    BRAZO_RETRAIDO: "Brazo totalmente retraído (Brazo home)",
    BRAZO_EXTENDIDO: "Brazo totalmente extendido (Brazo end)",
}


def trama(orden: int, valor: float = 0.0) -> bytes:
    """Orden + valor decimal -> paquete de 4 bytes [orden, int16 LE (valor*100), chk]."""
    raw = max(-32767, min(32767, round(float(valor) * 100.0)))
    cuerpo = struct.pack('<Bh', int(orden), int(raw))
    chk = cuerpo[0] ^ cuerpo[1] ^ cuerpo[2] ^ CHK_XOR
    return cuerpo + bytes([chk])


def hexstr(data) -> str:
    return ' '.join(f'{b:02X}' for b in data)


# =============================================================================
# Parser de recepción (sin dependencias de ROS, testeable por separado)
# =============================================================================
class ReceptorSTM32:
    """
    Máquina de sincronización para tramas de 2 bytes [codigo, codigo ^ 0xA5].

    - Si filtrar_eco=True, cualquier trama TX (4 bytes) enviada hace menos de
      `ventana_eco` s que aparezca al inicio del buffer se descarta completa.
    - Si el buffer contiene sólo el comienzo de una posible trama eco, espera
      hasta `espera_parcial` s a que lleguen los bytes restantes antes de
      intentar interpretarlo como mensaje de la STM32.
    - Todo byte que no forme una trama válida se acumula en `basura`.
    """

    LARGO_TX = 4
    LARGO_RX = 2

    def __init__(self, codigos_validos, filtrar_eco=True,
                 ventana_eco=0.5, espera_parcial=0.02):
        self.codigos = set(codigos_validos)
        self.filtrar_eco = filtrar_eco
        self.ventana_eco = ventana_eco
        self.espera_parcial = espera_parcial

        self.buf = bytearray()
        self.tx_recientes = deque()  # (timestamp, bytes)
        self._t_parcial = None

        # Estadísticas / diagnóstico
        self.basura = bytearray()
        self.n_validos = 0
        self.n_ecos = 0
        self.n_basura = 0

    def registrar_tx(self, frame: bytes, t: float):
        if self.filtrar_eco:
            self.tx_recientes.append((t, bytes(frame)))
            self._purgar_tx(t)

    def _purgar_tx(self, t: float):
        while self.tx_recientes and (t - self.tx_recientes[0][0]) > self.ventana_eco:
            self.tx_recientes.popleft()
        # Límite duro por si el timer TX corre y no llega nada
        while len(self.tx_recientes) > 64:
            self.tx_recientes.popleft()

    def _estado_eco(self):
        """'eco' si el buffer empieza con una trama TX reciente, 'parcial' si
        podría serlo pero faltan bytes, None si no."""
        cabeza = bytes(self.buf[:self.LARGO_TX])
        parcial = False
        for i, (_, frame) in enumerate(self.tx_recientes):
            if len(cabeza) >= self.LARGO_TX:
                if cabeza == frame:
                    # Cada trama enviada produce a lo más un eco
                    del self.tx_recientes[i]
                    return 'eco'
            elif frame.startswith(cabeza):
                parcial = True
        return 'parcial' if parcial else None

    def alimentar(self, data: bytes, t: float):
        """Agrega bytes recibidos y devuelve la lista de códigos válidos."""
        if data:
            self.buf.extend(data)
        self._purgar_tx(t)

        codigos = []
        while len(self.buf) >= self.LARGO_RX:
            if self.filtrar_eco and self.tx_recientes:
                estado = self._estado_eco()
                if estado == 'eco':
                    del self.buf[:self.LARGO_TX]
                    self.n_ecos += 1
                    self._t_parcial = None
                    continue
                if estado == 'parcial':
                    if self._t_parcial is None:
                        self._t_parcial = t
                    if (t - self._t_parcial) < self.espera_parcial:
                        break  # esperar bytes restantes del posible eco
                self._t_parcial = None

            b0, b1 = self.buf[0], self.buf[1]
            if b0 in self.codigos and b1 == (b0 ^ CHK_XOR):
                codigos.append(b0)
                self.n_validos += 1
                del self.buf[:self.LARGO_RX]
            else:
                self.basura.append(b0)
                self.n_basura += 1
                del self.buf[:1]

        # Evita crecer sin límite si llega ruido continuo
        if len(self.basura) > 256:
            del self.basura[:-256]
        return codigos

    def sacar_basura(self) -> bytes:
        out = bytes(self.basura)
        self.basura.clear()
        return out


# =============================================================================
# Nodo ROS
# =============================================================================
class RealSTM32BridgeV2(Node):
    def __init__(self):
        super().__init__('real_stm32_bridge')

        self.declare_parameter('port', '/dev/ttyAMA1')
        self.declare_parameter('baudrate', 115200)
        self.declare_parameter('filtrar_eco', True)
        self.declare_parameter('ventana_duplicado', 0.15)
        self.declare_parameter('debug_rx', False)

        port = str(self.get_parameter('port').value)
        baudrate = int(self.get_parameter('baudrate').value)
        filtrar_eco = bool(self.get_parameter('filtrar_eco').value)
        self.ventana_duplicado = float(self.get_parameter('ventana_duplicado').value)
        self.debug_rx = bool(self.get_parameter('debug_rx').value)

        self.uart = serial.Serial(
            port, baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=0.0,
        )
        # Descarta lo que haya quedado en el buffer del kernel de una sesión anterior
        try:
            self.uart.reset_input_buffer()
        except Exception:
            pass

        self.rx = ReceptorSTM32(MENSAJES_STM32.keys(), filtrar_eco=filtrar_eco)
        self._ultimo_codigo = None
        self._t_ultimo_codigo = 0.0
        self._t_reporte_basura = 0.0

        # Estado interno
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

        # Suscriptores
        self.create_subscription(Twist, '/cmd_vel', self.recibir_velocidad, 10)
        self.create_subscription(Bool, '/navegacion/stop', self.recibir_stop, 10)
        self.create_subscription(Float32, '/navegacion/corregir_grados', self.recibir_correccion, 10)
        self.create_subscription(Bool, '/bomba', self.recibir_bomba, 10)
        self.create_subscription(Float32, '/brazo_mm', self.recibir_brazo, 10)

        # Publicadores
        self.pub_estado_brazo = self.create_publisher(Int32, '/stm32/estado_brazo', 10)
        self.pub_mensaje = self.create_publisher(Int32, '/stm32/mensaje', 10)

        # Timers: RX a 200 Hz, TX a 20 Hz (mismo executor -> sin carreras)
        self.create_timer(0.005, self.leer_uart)
        self.create_timer(0.05, self.enviar_pendiente)

        self.get_logger().info(
            f"real_stm32_bridge_v2 iniciado en {port} ({baudrate} bps, 8N1) | "
            f"filtrar_eco={filtrar_eco} ventana_duplicado={self.ventana_duplicado}s "
            f"debug_rx={self.debug_rx}"
        )

    # =========================================================================
    # Callbacks
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
                f"Orden brazo añadida: {accion} (MOVE_Stepper = {desplazamiento_cm:+.1f} cm)"
            )

    # =========================================================================
    # Recepción STM32 -> RPi
    # =========================================================================
    def leer_uart(self):
        ahora = time.monotonic()
        data = b''
        try:
            n = self.uart.in_waiting
            if n > 0:
                data = self.uart.read(n)
        except Exception as e:
            self.get_logger().error(f"Error leyendo UART STM32: {e}",
                                    throttle_duration_sec=2.0)
            return

        if data and self.debug_rx:
            self.get_logger().info(f"[RX raw] {len(data)} B: {hexstr(data)}")

        for codigo in self.rx.alimentar(data, ahora):
            self._procesar_codigo(codigo, ahora)

        # Reporte de bytes descartados (máx. 1 vez por segundo)
        if self.rx.basura and (ahora - self._t_reporte_basura) > 1.0:
            self._t_reporte_basura = ahora
            self.get_logger().warn(
                f"[RX] Bytes descartados (no forman trama [cod, cod^0xA5]): "
                f"{hexstr(self.rx.sacar_basura())} | "
                f"válidos={self.rx.n_validos} ecos={self.rx.n_ecos} basura={self.rx.n_basura}"
            )

    def _procesar_codigo(self, codigo: int, ahora: float):
        # Filtra la misma trama repetida (p. ej. STM enviándola 2 veces)
        if (codigo == self._ultimo_codigo
                and (ahora - self._t_ultimo_codigo) < self.ventana_duplicado):
            self._t_ultimo_codigo = ahora
            return
        self._ultimo_codigo = codigo
        self._t_ultimo_codigo = ahora

        msg = Int32()
        msg.data = int(codigo)
        self.pub_mensaje.publish(msg)

        print(codigo)
        if codigo in (BRAZO_RETRAIDO, BRAZO_EXTENDIDO):
            self.get_logger().info(f"[STM32 -> RPi] ({codigo}) {MENSAJES_STM32[codigo]}")
            self.pub_estado_brazo.publish(msg)
        else:
            self.get_logger().error(
                f"[STM32 -> RPi] Error ({codigo}): {MENSAJES_STM32[codigo]} (STOP_All)"
            )

    # =========================================================================
    # Envío RPi -> STM32 (misma lógica que real_stm32_bridge)
    # =========================================================================
    def _escribir(self, orden: int, valor: float):
        frame = trama(orden, valor)
        self.uart.write(frame)
        self.rx.registrar_tx(frame, time.monotonic())

    def enviar_pendiente(self):
        sin_comandos = (self.ultimo_cmd > 0.0 and (time.monotonic() - self.ultimo_cmd) > 0.6)
        if sin_comandos:
            self.velocidad = 0.0
            self.stop_solicitado = True

        if self.stop_solicitado:
            if self.en_movimiento:
                self.en_movimiento = False
                self.pendientes.clear()
                self.pendientes.appendleft((STOP_ALL, 0.0))
            orden, valor = (self.pendientes.popleft() if self.pendientes else (STOP_ALL, 0.0))

        elif abs(self.velocidad) < 1e-4:
            if self.en_movimiento:
                self.en_movimiento = False
                ordenes_mov = (SET_VEL, SET_HEADING, START_TRACKING, START_MOV)
                self.pendientes = deque(cmd for cmd in self.pendientes if cmd[0] not in ordenes_mov)
                self.pendientes.appendleft((STOP_MOV, 0.0))
            orden, valor = (self.pendientes.popleft() if self.pendientes else (STOP_MOV, 0.0))

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
            orden, valor = (self.pendientes.popleft() if self.pendientes else (SET_VEL, self.velocidad))

        try:
            self._escribir(orden, valor)
        except Exception as e:
            self.get_logger().error(f"Error escribiendo en UART STM32: {e}")

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
        nodo = RealSTM32BridgeV2()
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

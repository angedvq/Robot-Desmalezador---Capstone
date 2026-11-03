import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import serial

def parse_tfmini_distance(pkt, ultima_dist_valida=float('inf')):
    """Convierte una trama válida de 9 bytes del TF-Mini en distancia (m) y actualiza memoria."""
    dist_cm = pkt[2] | (pkt[3] << 8)
    strength = pkt[4] | (pkt[5] << 8)

    # 1. Saturación o zona ciega a quemarropa (< 10 cm):
    # - 65534 (-2): saturación de señal reportada por firmware TFmini/TFmini-S
    # - strength == 65535 o >= 1000 con lectura inválida (0 o >= 65532): lente tapado / muy cerca
    # - 0 < dist_cm <= 10: objeto pegado al sensor
    # - Objeto que venía acercándose (< 35 cm) y entró en zona ciega manteniendo eco (strength >= 100)
    if (
        dist_cm == 65534
        or strength == 65535
        or (0 < dist_cm <= 10)
        or ((dist_cm == 0 or dist_cm >= 65532) and strength >= 1000)
        or ((dist_cm == 0 or dist_cm >= 65532) and ultima_dist_valida < 0.35 and strength >= 100)
    ):
        return 0.01, 0.01

    # 2. Espacio despejado (> 12 m) o eco demasiado débil sin saturación:
    if dist_cm == 0 or dist_cm >= 1200 or strength < 20:
        return float('inf'), float('inf')

    # 3. Lectura normal válida (0.11 m a 11.99 m)
    distancia_m = dist_cm / 100.0
    return distancia_m, distancia_m


class TFMiniDriver(Node):
    def __init__(self):
        super().__init__('tfmini_driver')
        self.declare_parameter('port', '/dev/serial0')
        self.pub = self.create_publisher(LaserScan, '/tf_mini/scan', 10)

        self.port_name = self.get_parameter('port').value
        self.baud_rate = 115200
        self.ser = None
        self.rx_buf = bytearray()
        self.ultima_dist_valida = float('inf')

        self._conectar_serial()

        # Leer y publicar a 20Hz
        self.timer = self.create_timer(0.05, self.read_and_publish)

    def _conectar_serial(self):
        try:
            self.ser = serial.Serial(self.port_name, self.baud_rate, timeout=0.0)
            self.rx_buf.clear()
            self.get_logger().info(f"Conectado exitosamente al TF-Mini en {self.port_name}")
        except Exception as e:
            self.ser = None
            self.get_logger().error(
                f"Fallo al abrir TF-Mini en {self.port_name}: {e}",
                throttle_duration_sec=5.0
            )

    def read_and_publish(self):
        if self.ser is None or not self.ser.is_open:
            self._conectar_serial()
            return

        try:
            waiting = self.ser.in_waiting
            if waiting > 0:
                # Limitar tamaño en caso de ráfaga acumulada
                if waiting > 180:
                    self.ser.reset_input_buffer()
                    self.rx_buf.clear()
                    return
                self.rx_buf.extend(self.ser.read(waiting))
        except Exception as e:
            self.get_logger().error(f"Error leyendo TF-Mini: {e}")
            try:
                self.ser.close()
            except Exception:
                pass
            self.ser = None
            return

        # Buscar la trama válida (cabecera 0x59 0x59 + checksum) más reciente en el buffer
        ultimo_pkt = None
        consumido = 0
        i = 0
        limite = len(self.rx_buf) - 9
        while i <= limite:
            if self.rx_buf[i] == 0x59 and self.rx_buf[i + 1] == 0x59:
                pkt = self.rx_buf[i:i + 9]
                if (sum(pkt[:8]) & 0xFF) == pkt[8]:
                    ultimo_pkt = pkt
                    i += 9
                    consumido = i
                    continue
            i += 1

        if consumido > 0:
            del self.rx_buf[:consumido]
        elif len(self.rx_buf) > 18:
            del self.rx_buf[:-8]

        if ultimo_pkt is None:
            return

        distancia_m, self.ultima_dist_valida = parse_tfmini_distance(
            ultimo_pkt, self.ultima_dist_valida
        )

        scan = LaserScan()
        scan.header.stamp = self.get_clock().now().to_msg()
        scan.header.frame_id = 'tfmini_link'
        scan.angle_min = 0.0
        scan.angle_max = 0.0
        scan.angle_increment = 0.0
        scan.range_min = 0.0
        scan.range_max = 12.0
        scan.ranges = [distancia_m]

        self.pub.publish(scan)

def main(args=None):
    rclpy.init(args=args)
    node = TFMiniDriver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.ser is not None and node.ser.is_open:
            node.ser.close()
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()

if __name__ == '__main__':
    main()
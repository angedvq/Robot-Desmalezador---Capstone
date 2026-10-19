#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32, String
import serial
import time

class DriverSTM32UART(Node):
    def __init__(self):
        super().__init__('driver_stm32_uart')

        # ----------------------------------------------------------------------
        # Configuración del Puerto Serie (UART)
        # ----------------------------------------------------------------------
        # USB: '/dev/ttyACM0' o '/dev/ttyUSB0'
        # Pines GPIO RX/TX directos: '/dev/serial0' o '/dev/ttyAMA0'
        self.puerto_serial = '/dev/ttyACM0' 
        self.baudrate = 115200  # debe ser el mismo baudrate en la STM32

        try:
            self.ser = serial.Serial(self.puerto_serial, self.baudrate, timeout=0.1)
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
            self.get_logger().info(f"Conectado a STM32 en {self.puerto_serial} a {self.baudrate} de baudrate")
        except Exception as e:
            self.get_logger().error(f"No se pudo abrir el puerto serie {self.puerto_serial}: {e}")
            self.ser = None

        # Suscriptor de velocidad lineal y angular
        self.sub_cmd_vel = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10
        )

        # Suscriptor opcional para correcciones de orientación directa (Header en grados)
        # Permite enviar órdenes como "corregir 10 grados" -> Tópico: /navegacion/corregir_grados
        self.sub_corregir_grados = self.create_subscription(
            Float32,
            '/navegacion/corregir_grados',
            self.corregir_grados_callback,
            10
        )

    def cmd_vel_callback(self, msg: Twist):
        """
        Callback para el envío continuo de velocidades (Navegación general).
        Trama enviada a STM32: #V,velocidad_lineal,velocidad_angular\n
        Ejemplo: #V,0.20,0.05\n
        """
        if self.ser is None or not self.ser.is_open:
            return

        v_lineal = msg.linear.x   # m/s
        v_angular = msg.angular.z # rad/s

        # Formatear la trama de texto con 2 decimales de precisión
        trama = f"#V,{v_lineal:.2f},{v_angular:.2f}\n"
        self.enviar_trama_uart(trama)

    def corregir_grados_callback(self, msg: Float32):
        """
        Callback para enviar instrucciones de corrección de ángulo relativas.
        Ejemplo: Si msg.data = 10.0 -> Le dice a la STM32 que corrija +10 grados.
        Trama enviada a STM32: #H,10.0\n
        """
        if self.ser is None or not self.ser.is_open:
            return

        grados = msg.data
        trama = f"#H,{grados:.1f}\n"
        self.get_logger().info(f"Enviando corrección de orientación a STM32: {grados}°")
        self.enviar_trama_uart(trama)

    def enviar_trama_uart(self, trama: str):
        """Convierte el string a bytes codificados en ASCII y lo envía por UART."""
        try:
            self.ser.write(trama.encode('ascii'))
            self.get_logger().debug(f"UART Tx -> {trama.strip()}")
        except Exception as e:
            self.get_logger().error(f"Error escribiendo en UART: {e}")

    def stop(self):
        """Envía comando de parada de emergencia al apagar el nodo."""
        if self.ser and self.ser.is_open:
            try:
                trama_stop = "#S\n"
                self.ser.write(trama_stop.encode('ascii'))
                self.get_logger().info("Comando de PARADA (#S) enviado a la STM32.")
            except Exception:
                pass

def main(args=None):
    rclpy.init(args=args)
    node = DriverSTM32UART()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
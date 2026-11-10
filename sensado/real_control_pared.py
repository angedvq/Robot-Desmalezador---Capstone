#!/usr/bin/env python3
import math
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Float32


class RealControlPared(Node):
    def __init__(self):
        super().__init__('real_control_pared')

        # ============================================
        # Parámetros de seguimiento de pared
        # ============================================
        self.declare_parameter('distancia_meta', 0.70)      # Metros de separacion con la pared
        self.declare_parameter('distancia_frenado', 0.50)   # Metros de deteccion de obstaculo
        self.declare_parameter('velocidad_avance', 0.20)    # Velocidad

        # Ganancias de giro por cada metro de error
        self.declare_parameter('kp', 35.0)          # Giro suave para seguir la pared
        self.declare_parameter('kp_evasion', 75.0)  # Giro brusco para esquivar un pilar

        # Limites de giro para no volcar o ir en U
        self.declare_parameter('max_grados', 12.0)          # Max giro normal
        self.declare_parameter('max_grados_evasion', 45.0) # Max giro para evitar choque inminente

        # Rapidez de giro por segundo
        self.declare_parameter('slew_rate_deg_s', 6.0)      # Giro despacio
        self.declare_parameter('slew_evasion_deg_s', 30.0)  # Giro brusco

        # Umbrales y filtros
        self.declare_parameter('umbral_evasion', 0.12)  # Si estamos 12cms cerca de un obstaculo (a punto de chocar), se activan los parametros bruscos
        self.declare_parameter('alpha_dist', 0.35)      # Filtro de ruido

        self.declare_parameter('usar_tfmini', True)     # Activa/desactiva deteccion de obstaculos
        self.declare_parameter('lidar_timeout', 0.8)    
        self.declare_parameter('tfmini_timeout', 0.8)

        # Extraer valores de cada parametro al iniciar el nodo
        self.distancia_meta = float(self.get_parameter('distancia_meta').value)
        self.distancia_frenado = float(self.get_parameter('distancia_frenado').value)
        self.velocidad_avance = float(self.get_parameter('velocidad_avance').value)
        self.kp = float(self.get_parameter('kp').value)
        self.kp_evasion = float(self.get_parameter('kp_evasion').value)
        self.max_grados = float(self.get_parameter('max_grados').value)
        self.max_grados_evasion = float(self.get_parameter('max_grados_evasion').value)
        self.slew_rate_deg_s = float(self.get_parameter('slew_rate_deg_s').value)
        self.slew_evasion_deg_s = float(self.get_parameter('slew_evasion_deg_s').value)
        self.umbral_evasion = float(self.get_parameter('umbral_evasion').value)
        self.alpha_dist = float(self.get_parameter('alpha_dist').value)
        self.usar_tfmini = bool(self.get_parameter('usar_tfmini').value)
        self.lidar_timeout = float(self.get_parameter('lidar_timeout').value)
        self.tfmini_timeout = float(self.get_parameter('tfmini_timeout').value)

        # Suscriptores de sensores reales
        self.sub_dist = self.create_subscription(Float32, '/lidar/distancia_lateral', self.dist_callback, 10)
        self.sub_obstacle = self.create_subscription(Bool, '/obstaculo_frontal', self.obstacle_callback, 10)
        self.sub_tfmini = self.create_subscription(LaserScan, '/tf_mini/scan', self.tfmini_callback, 10)

        # Publicadores hacia stm32_bridge
        self.pub_vel = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_stop = self.create_publisher(Bool, '/navegacion/stop', 10)
        self.pub_grados = self.create_publisher(Float32, '/navegacion/corregir_grados', 10)

        # El rumbo por defecto arranca siempre en 0.0°
        self.distancia_actual = None
        self.rumbo_suavizado = 0.0          # Rumbo que estamos pidiendo girar ahora
        self.ultimo_rumbo_enviado = 0.0     # Ultimo rumbo mandado a stm32
        self.ultimo_envio_grados_ts = 0.0   # Timestamp de ultimo rumbo mandado

        self.obstaculo_topic = False
        self.obstaculo_scan = False
        self.en_pausa_por_obstaculo = False
        self.en_pausa_por_watchdog = True

        # Timestamps de sensores
        self.ultimo_lidar_ts = None
        self.ultimo_tfmini_ts = None

        # Bucle de control a 20 Hz (0.05 s)
        self.dt = 0.05
        self.timer = self.create_timer(self.dt, self.control_loop)

        self.get_logger().info(
            f"real_control_pared iniciado | meta={self.distancia_meta:.2f}m, "
            f"suave=(kp={self.kp:.0f}, max={self.max_grados:.0f}°), "
            f"evasion=(kp={self.kp_evasion:.0f}, max={self.max_grados_evasion:.0f}°)"
        )

    # ============================================
    # Callbacks
    # ============================================
    def dist_callback(self, msg):
        if math.isfinite(msg.data) and msg.data > 0.0:
            nueva = float(msg.data)
            if self.distancia_actual is None:
                self.distancia_actual = nueva # Primera lectura

            elif nueva < (self.distancia_actual - 0.08):
                # Caída repentina de distancia (por algun saliente): respuesta rápida
                self.distancia_actual = 0.80 * nueva + 0.20 * self.distancia_actual
            else:
                # Seguimiento normal: filtrado suave
                self.distancia_actual = (self.alpha_dist * nueva + (1.0 - self.alpha_dist) * self.distancia_actual)
            self.ultimo_lidar_ts = time.monotonic()

    def obstacle_callback(self, msg):
        # Leer si el tfmini detectó un obstaculo
        self.obstaculo_topic = bool(msg.data)

    def tfmini_callback(self, msg):
        self.ultimo_tfmini_ts = time.monotonic()
        detectado = False
        for r in msg.ranges:
            if math.isnan(r) or (math.isinf(r) and r > 0.0):
                continue

            # Si se detecta un obstaculo dentro del umbral, dar aviso de frenar
            if r < self.distancia_frenado:
                detectado = True
                break
        self.obstaculo_scan = detectado

    # ============================================
    # Loop principal
    # ============================================
    def detener_robot(self):
        msg_stop_flag = Bool()
        msg_stop_flag.data = True
        self.pub_stop.publish(msg_stop_flag)

        msg_stop_vel = Twist()
        msg_stop_vel.linear.x = 0.0
        msg_stop_vel.angular.z = 0.0
        self.pub_vel.publish(msg_stop_vel)

        self.rumbo_suavizado = 0.0

    def control_loop(self):
        now = time.monotonic()

        # Watchdogs de sensores. Revisa si pasó mucho tiempo desde la ultima recepcion de datos de sensores
        lidar_ok = (self.ultimo_lidar_ts is not None and (now - self.ultimo_lidar_ts) <= self.lidar_timeout)
        tfmini_ok = ((not self.usar_tfmini) or (self.ultimo_tfmini_ts is not None and (now - self.ultimo_tfmini_ts) <= self.tfmini_timeout))

        # Si algun sensor no está funcionando, freno de emergencia
        if not lidar_ok or not tfmini_ok:
            self.en_pausa_por_watchdog = True
            self.detener_robot()

            faltantes = []
            if not lidar_ok:
                faltantes.append('LiDAR lateral')
            if not tfmini_ok:
                faltantes.append('TF-Mini frontal')
            self.get_logger().warn(f"Watchdog sensores: sin datos de {', '.join(faltantes)}. STOP enviado.",throttle_duration_sec=2.0)

            return

        # Si no hay problemas, quitar la pausa
        elif self.en_pausa_por_watchdog:
            self.en_pausa_por_watchdog = False
            self.get_logger().info("Sensores operativos. Control autónomo habilitado")

        # Detección de obstáculo frontal
        if self.usar_tfmini and (self.obstaculo_topic or self.obstaculo_scan):
            if not self.en_pausa_por_obstaculo:
                self.en_pausa_por_obstaculo = True
            self.detener_robot()
            self.get_logger().warn(f"Obstáculo frontal (< {self.distancia_frenado:.2f}m). Esperando despeje de ruta...", throttle_duration_sec=2.0)
            return

        # Continuar ruta
        elif self.en_pausa_por_obstaculo:
            self.en_pausa_por_obstaculo = False
            self.get_logger().info("Ruta despejada. Reanudando movimiento...")

        # Quitar STOP_ALL
        msg_run = Bool()
        msg_run.data = False
        self.pub_stop.publish(msg_run)

        # Cálculo de rumbo:
        # Seguimiento normal: suave (kp=35, máx +12°)
        # Pilar/saliente: evasión agresiva (kp=75, máx -25°)
        error = self.distancia_actual - self.distancia_meta
        en_evasion = error < -self.umbral_evasion # si se está demasiado cerca de la meta de distancia, puede haber un choque, se activa el modo evasion

        # Asignacion del giro objetivo
        if abs(error) < 0.03:
            # Si estamos a +-3cms de la meta, seguimos derecho
            rumbo_objetivo = 0.0

        # Si tenemos que evadir, utilizar parametros bruscos
        elif en_evasion:
            rumbo_objetivo = max(-self.max_grados_evasion, error * self.kp_evasion)

        # En otro caso, utilizar parametros suaves
        else:
            rumbo_objetivo = max(-self.max_grados, min(self.max_grados, error * self.kp))

        # Rampa de giro: rápida al abrirse para esquivar un pilar, suave al volver a la pared
        abriendose_por_peligro = en_evasion and (rumbo_objetivo < self.rumbo_suavizado)

        # Velocidad de giro segun situacion
        tasa_deg_s = self.slew_evasion_deg_s if abriendose_por_peligro else self.slew_rate_deg_s

        # Grados maximos a sumar al volante
        max_paso = tasa_deg_s * self.dt

        delta = rumbo_objetivo - self.rumbo_suavizado
        if abs(delta) <= max_paso:
            self.rumbo_suavizado = rumbo_objetivo
        else:
            self.rumbo_suavizado += math.copysign(max_paso, delta)

        # Publicar SET_HeadingRef
        diferencia_angular = abs(self.rumbo_suavizado - self.ultimo_rumbo_enviado)
        retorno_a_cero = (self.rumbo_suavizado == 0.0 and self.ultimo_rumbo_enviado != 0.0)
        requiere_refresco = (now - self.ultimo_envio_grados_ts) >= 1.0

        # Enviar orden solo si el angulo cambió mas de 0.5 grados para no saturar
        if diferencia_angular >= 0.5 or retorno_a_cero or requiere_refresco:
            msg_grados = Float32()
            msg_grados.data = float(self.rumbo_suavizado)
            self.pub_grados.publish(msg_grados)

            if diferencia_angular >= 0.5 or retorno_a_cero:
                modo = "EVASION" if abriendose_por_peligro else "SUAVE"
                self.get_logger().info(
                    f"[{modo}] Dist: {self.distancia_actual:.2f}m -> SET_Heading: {self.rumbo_suavizado:.1f}° "
                    f"(obj: {rumbo_objetivo:.1f}°)"
                )

            # Actualizar variables de estado
            self.ultimo_rumbo_enviado = self.rumbo_suavizado
            self.ultimo_envio_grados_ts = now

        # Enviar velocidad lineal constante
        msg_vel = Twist()
        msg_vel.linear.x = self.velocidad_avance
        self.pub_vel.publish(msg_vel)


def main(args=None):
    rclpy.init(args=args)
    node = RealControlPared()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.detener_robot()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

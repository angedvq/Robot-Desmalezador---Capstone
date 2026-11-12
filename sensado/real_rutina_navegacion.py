#!/usr/bin/env python3
import math
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseArray
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Float32


class RealRutinaNavegacion(Node):
    def __init__(self):
        super().__init__('real_rutina_navegacion')

        # ---------------------------------------------------------------------
        # Declaración de parámetros 
        # ---------------------------------------------------------------------
        # Parámetros geométricos y de avance
        self.declare_parameter('distancia_meta', 0.70)        # Distancia a la hilera (m)
        self.declare_parameter('distancia_frenado', 0.50)     # Umbral obstáculo TF-Mini (m)
        self.declare_parameter('velocidad_avance', 0.20)      # Velocidad lineal (m/s)

        # Parámetros de corrección suave de rumbo (LiDAR + IMU)
        self.declare_parameter('kp_rumbo', 30.0)              # Ganancia proporcional. Fuerza de correcion del rumbo
        self.declare_parameter('max_grados_rumbo', 10.0)      # Límite de ángulo suave. Tope maximo de giro
        self.declare_parameter('slew_rate_deg_s', 6.0)        # Rampa máxima de giro. Velocidad de giro
        self.declare_parameter('alpha_dist', 0.40)            # Filtro distancia tronco

        # Parámetros de la rutina de árboles y brazo
        self.declare_parameter('meta_arboles', 4)      
        self.declare_parameter('timeout_sin_arbol', 20.0)    
        self.declare_parameter('tiempo_delay_brazo', 1.5)    
        self.declare_parameter('tiempo_brazo_extendido', 8.0) 
        self.declare_parameter('tiempo_brazo_retraido', 8.0)  
        self.declare_parameter('tiempo_enfriamiento', 3.0)    # Ceguera temporal para avanzar a sgte tronco (s)
        self.declare_parameter('desplazamiento_brazo_mm', 410.0)  # +200 mm (+20 cm) extiende, -200 mm retrae REVISAR

        # Watchdogs de sensores reales
        self.declare_parameter('usar_tfmini', True)
        self.declare_parameter('lidar_timeout', 2.0)
        self.declare_parameter('tfmini_timeout', 2.0)

        # Lectura de parámetros
        self.distancia_meta = float(self.get_parameter('distancia_meta').value)
        self.distancia_frenado = float(self.get_parameter('distancia_frenado').value)
        self.velocidad_avance = float(self.get_parameter('velocidad_avance').value)
        self.kp_rumbo = float(self.get_parameter('kp_rumbo').value)
        self.max_grados_rumbo = float(self.get_parameter('max_grados_rumbo').value)
        self.slew_rate_deg_s = float(self.get_parameter('slew_rate_deg_s').value)
        self.alpha_dist = float(self.get_parameter('alpha_dist').value)

        self.meta_arboles = int(self.get_parameter('meta_arboles').value)
        self.timeout_sin_arbol = float(self.get_parameter('timeout_sin_arbol').value)
        self.tiempo_delay_brazo = float(self.get_parameter('tiempo_delay_brazo').value)
        self.tiempo_brazo_extendido = float(self.get_parameter('tiempo_brazo_extendido').value)
        self.tiempo_brazo_retraido = float(self.get_parameter('tiempo_brazo_retraido').value)
        self.tiempo_enfriamiento = float(self.get_parameter('tiempo_enfriamiento').value)
        self.desplazamiento_brazo_mm = abs(float(self.get_parameter('desplazamiento_brazo_mm').value))

        self.usar_tfmini = bool(self.get_parameter('usar_tfmini').value)
        self.lidar_timeout = float(self.get_parameter('lidar_timeout').value)
        self.tfmini_timeout = float(self.get_parameter('tfmini_timeout').value)

        # ---------------------------------------------------------------------
        # Publicadores hacia stm32
        # ---------------------------------------------------------------------
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.stop_pub = self.create_publisher(Bool, '/navegacion/stop', 10)
        self.grados_pub = self.create_publisher(Float32, '/navegacion/corregir_grados', 10)
        self.brazo_pub = self.create_publisher(Float32, '/brazo_mm', 10)

        # ---------------------------------------------------------------------
        # Suscriptores de percepción
        # ---------------------------------------------------------------------
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_watchdog_callback, 10)
        self.tree_sub = self.create_subscription(PoseArray, '/arboles_detectados', self.tree_callback, 10)
        self.dist_sub = self.create_subscription(Float32, '/lidar/distancia_tronco_lateral', self.dist_tronco_callback, 10)
        self.obs_sub = self.create_subscription(Bool, '/obstaculo_frontal', self.obs_callback, 10)
        self.tfmini_sub = self.create_subscription(LaserScan, '/tf_mini/scan', self.tfmini_callback, 10)

        # ---------------------------------------------------------------------
        # Variables de estado y temporizadores
        # ---------------------------------------------------------------------
        self.estado_actual = 'AVANZANDO'
        self.arboles_procesados = 0
        self.arbol_en_mira = False
        self.brazo_extendido = False

        # Variables de guiado lateral y rumbo IMU
        self.distancia_tronco_actual = None
        self.ultimo_tronco_guia_ts = None
        self.rumbo_suavizado = 0.0
        self.ultimo_rumbo_enviado = 0.0
        self.ultimo_envio_grados_ts = 0.0

        # Watchdogs y estado de obstáculo frontal
        self.ultimo_lidar_ts = None
        self.ultimo_tfmini_ts = None
        self.obstaculo_topic = False
        self.obstaculo_scan = False
        self.en_pausa_por_watchdog = True
        self.en_pausa_por_obstaculo = False

        # Timestamps
        now = time.monotonic()
        self.tiempo_inicio_estado = now
        self.tiempo_ultimo_arbol = now
        self.tiempo_transcurrido_guardado = 0.0
        self.tiempo_sin_ver_arbol_guardado = 0.0

        # Bucle principal a 20 Hz 
        self.dt = 0.05
        self.control_timer = self.create_timer(self.dt, self.control_loop)

        self.get_logger().info(
            f"real_rutina_navegacion iniciado | Meta: {self.meta_arboles} árboles | "
            f"Distancia hilera: {self.distancia_meta:.2f}m | Delay brazo: {self.tiempo_delay_brazo:.1f}s | "
            f"Extensión: {self.tiempo_brazo_extendido:.1f}s | Retracción: {self.tiempo_brazo_retraido:.1f}s | "
            f"Recorrido brazo: {self.desplazamiento_brazo_mm / 10.0:.1f}cm | "
            f"Timeout sin árbol: {self.timeout_sin_arbol:.0f}s"
        )

    # =========================================================================
    # Callbacks
    # =========================================================================
    def scan_watchdog_callback(self, _msg: LaserScan):
        """Monitorea directamente /scan para el watchdog del RPLidar."""
        self.ultimo_lidar_ts = time.monotonic()

    def tree_callback(self, msg: PoseArray):
        """Recibe las detecciones de troncos en la ventana lateral"""
        self.ultimo_lidar_ts = time.monotonic()
        self.arbol_en_mira = len(msg.poses) > 0

    def dist_tronco_callback(self, msg: Float32):
        """Recibe la distancia perpendicular al tronco más cercano en diagonal frontal-izquierdo. 
        Si recibe NaN, significa que no hay tronco (espacio entre árboles) """
        now = time.monotonic()
        self.ultimo_lidar_ts = now

        if math.isfinite(msg.data) and msg.data > 0.0:
            nueva = float(msg.data)
            if self.distancia_tronco_actual is None:
                self.distancia_tronco_actual = nueva
            else:
                # Filtrar y suavizar conduccion
                self.distancia_tronco_actual = (
                    self.alpha_dist * nueva + (1.0 - self.alpha_dist) * self.distancia_tronco_actual
                )
            self.ultimo_tronco_guia_ts = now
        else:
            # En el claro entre árboles no hay lectura lateral
            self.distancia_tronco_actual = None

    def obs_callback(self, msg: Bool):
        """Recibe flag de obstáculo frontal desde procesamiento_tfmini"""
        self.obstaculo_topic = bool(msg.data)

    def tfmini_callback(self, msg: LaserScan):
        """
        Monitorea /tf_mini/scan para actualizar el watchdog del
        sensor frontal y verificar distancias menores a distancia_frenado
        """
        self.ultimo_tfmini_ts = time.monotonic()
        detectado = False
        for r in msg.ranges:
            if math.isnan(r) or (math.isinf(r) and r > 0.0):
                continue
            if r < self.distancia_frenado:
                detectado = True
                break
        self.obstaculo_scan = detectado

    # =========================================================================
    # Funciones auxiliares de actuación y seguridad
    # =========================================================================
    def comandar_brazo(self, accion: str):
        """
        Envía el comando de brazo hacia la stm32 a través de /brazo_mm:
        - 'EXTENDER': envía +desplazamiento_brazo_mm
        - 'RETRAER':  envía -desplazamiento_brazo_mm
        """
        msg_brazo = Float32()
        if accion == 'EXTENDER':
            msg_brazo.data = +self.desplazamiento_brazo_mm
            self.brazo_extendido = True
            self.get_logger().info(
                f"[BRAZO] Extendiendo brazo (+{self.desplazamiento_brazo_mm / 10.0:.1f} cm)..."
            )
        elif accion == 'RETRAER':
            msg_brazo.data = -self.desplazamiento_brazo_mm
            self.brazo_extendido = False
            self.get_logger().info(
                f"[BRAZO] Retrayendo brazo (-{self.desplazamiento_brazo_mm / 10.0:.1f} cm)..."
            )
        else:
            msg_brazo.data = 0.0
            self.brazo_extendido = False

        self.brazo_pub.publish(msg_brazo)

    def detener_todo_emergencia(self):
        """
        Envía STOP_All y velocidad 0.0 m/s
        """
        msg_stop = Bool()
        msg_stop.data = True
        self.stop_pub.publish(msg_stop)

        msg_vel = Twist()
        msg_vel.linear.x = 0.0
        msg_vel.angular.z = 0.0
        self.cmd_pub.publish(msg_vel)

        self.rumbo_suavizado = 0.0

    def _pausar_relojes(self, now: float):
        """Congela los tiempos transcurridos durante una pausa por obstáculo o watchdog"""
        self.tiempo_transcurrido_guardado = now - self.tiempo_inicio_estado
        self.tiempo_sin_ver_arbol_guardado = now - self.tiempo_ultimo_arbol

    def _restaurar_relojes(self, now: float):
        """Restaura los tiempos transcurridos al reanudarse la operación"""
        self.tiempo_inicio_estado = now - self.tiempo_transcurrido_guardado
        self.tiempo_ultimo_arbol = now - self.tiempo_sin_ver_arbol_guardado

    def _actualizar_y_publicar_rumbo(self, now: float, permitir_correccion_lateral: bool):
        """
        Calcula y publica el rumbo objetivo hacia la STM32 por /navegacion/corregir_grados:
        - Si `permitir_correccion_lateral` es True y hay un tronco guía reciente (< 0.5 s),
          calcula una corrección suave proporcional a `(distancia_tronco - 0.70 m)`.
        - Si no hay tronco a la vista (claro entre árboles) o estamos alineando/enfriando,
          el rumbo objetivo es `0.0°` para que la IMU de la STM32 mantenga al robot recto.
        """
        tronco_guia_vigente = (
            permitir_correccion_lateral
            and self.distancia_tronco_actual is not None
            and self.ultimo_tronco_guia_ts is not None
            and (now - self.ultimo_tronco_guia_ts) <= 0.5
        )

        # Si tenemos un tronco guia vigente, aplica una correccion suave 
        if tronco_guia_vigente:
            error = self.distancia_tronco_actual - self.distancia_meta
            if abs(error) < 0.03: # margen de error
                rumbo_objetivo = 0.0
            else:
                rumbo_objetivo = max(-self.max_grados_rumbo, min(self.max_grados_rumbo, error * self.kp_rumbo))
        else:
            # En el espacio entre árboles (o durante el delay para el brazo), mantener 0.0° con la IMU
            rumbo_objetivo = 0.0

        # Limitador de pendiente (slew-rate) para que todo giro sea suave
        max_paso = self.slew_rate_deg_s * self.dt
        delta = rumbo_objetivo - self.rumbo_suavizado
        if abs(delta) <= max_paso:
            self.rumbo_suavizado = rumbo_objetivo
        else:
            self.rumbo_suavizado += math.copysign(max_paso, delta)

        # Se envia orden a stm32 solo si hubo un cambio significante (0.5°) para no saturar
        diferencia = abs(self.rumbo_suavizado - self.ultimo_rumbo_enviado)
        retorno_a_cero = (self.rumbo_suavizado == 0.0 and self.ultimo_rumbo_enviado != 0.0)
        requiere_refresco = (now - self.ultimo_envio_grados_ts) >= 1.0

        if diferencia >= 0.5 or retorno_a_cero or requiere_refresco:
            msg_grados = Float32()
            msg_grados.data = float(self.rumbo_suavizado)
            self.grados_pub.publish(msg_grados)

            if diferencia >= 0.5 or retorno_a_cero:
                ref_str = (f"Tronco a {self.distancia_tronco_actual:.2f}m" if tronco_guia_vigente else "Claro IMU 0°")
                self.get_logger().info(
                    f"[RUMBO] {ref_str} -> SET_Heading: {self.rumbo_suavizado:.1f}° "
                    f"(obj: {rumbo_objetivo:.1f}°)")
            self.ultimo_rumbo_enviado = self.rumbo_suavizado
            self.ultimo_envio_grados_ts = now

    # =========================================================================
    # Bucle Principal de Control 
    # =========================================================================
    def control_loop(self):
        now = time.monotonic()

        # Si la rutina ya terminó, mantener todo apagado con STOP_All
        if self.estado_actual == 'FINALIZADO':
            self.detener_todo_emergencia()
            return

        # Verificación de Watchdogs de Sensores. Revisa si pasó mucho tiempo desde la ultima recepcion de datos de sensores
        operando_brazo_detenido = self.estado_actual in ('EXTENDIENDO_BRAZO', 'RETRAYENDO_BRAZO')
        lidar_ok = (self.ultimo_lidar_ts is not None and (now - self.ultimo_lidar_ts) <= self.lidar_timeout)
        tfmini_ok = ((not self.usar_tfmini) or (self.ultimo_tfmini_ts is not None and (now - self.ultimo_tfmini_ts) <= self.tfmini_timeout))

        # Si algun sensor no está funcionando mientras el robot se desplaza, freno de emergencia.
        # Nota: Si ya está detenido accionando el brazo, no enviamos STOP_All por un parpadeo
        # de sensor para no cortar el recorrido del motor stepper a mitad de camino.
        if (not lidar_ok or not tfmini_ok) and not operando_brazo_detenido:
            if not self.en_pausa_por_watchdog:
                self._pausar_relojes(now)
                self.en_pausa_por_watchdog = True
            self.detener_todo_emergencia()

            faltantes = []
            if not lidar_ok:
                faltantes.append('LiDAR (/scan)')
            if not tfmini_ok:
                faltantes.append('TF-Mini (/tf_mini/scan)')
            self.get_logger().warn(
                f"Watchdog sensores: esperando datos de {', '.join(faltantes)}. STOP_All activo.",
                throttle_duration_sec=2.0,
            )
            return

        # Si todo esta ok, quitar la pausa
        elif self.en_pausa_por_watchdog and lidar_ok and tfmini_ok:
            self._restaurar_relojes(now)
            self.en_pausa_por_watchdog = False
            self.get_logger().info("Sensores ok. Iniciando/reanudando rutina de árboles")
            if self.estado_actual == 'EXTENDIENDO_BRAZO':
                self.comandar_brazo('EXTENDER')
            elif self.estado_actual == 'RETRAYENDO_BRAZO':
                self.comandar_brazo('RETRAER')

        # ---------------------------------------------------------------------
        # Obstáculo Frontal (solo detiene cuando el robot pretende avanzar)
        # ---------------------------------------------------------------------
        hay_obstaculo = self.usar_tfmini and (self.obstaculo_topic or self.obstaculo_scan)
        if hay_obstaculo and not operando_brazo_detenido:
            if not self.en_pausa_por_obstaculo:
                self._pausar_relojes(now)
                self.en_pausa_por_obstaculo = True
            self.detener_todo_emergencia()
            self.get_logger().warn(
                f"Obstáculo frontal (< {self.distancia_frenado:.2f}m). "
                "Esperando despeje de la ruta...",
                throttle_duration_sec=2.0,
            )
            return

        # Continuar ruta
        elif self.en_pausa_por_obstaculo and not hay_obstaculo:
            self._restaurar_relojes(now)
            self.en_pausa_por_obstaculo = False
            self.get_logger().info("Ruta despejada. Reanudando...")
            # Si estaba en medio de la extensión del brazo cuando ocurrió el STOP_All,
            # reenviar la orden de extensión al reanudar
            if self.estado_actual == 'EXTENDIENDO_BRAZO':
                self.comandar_brazo('EXTENDER')
            elif self.estado_actual == 'RETRAYENDO_BRAZO':
                self.comandar_brazo('RETRAER')

        # Operación normal 
        msg_stop = Bool()
        msg_stop.data = False
        self.stop_pub.publish(msg_stop)

        # Tiempos activos en el estado actual y desde el último árbol visto
        tiempo_transcurrido = now - self.tiempo_inicio_estado
        tiempo_sin_ver_arbol = now - self.tiempo_ultimo_arbol

        msg_vel = Twist()

        # ---------------------------------------------------------------------
        # Máquina de Estados de Navegación y Accionamiento de Brazo
        # ---------------------------------------------------------------------
        if self.estado_actual == 'AVANZANDO':
            msg_vel.linear.x = self.velocidad_avance
            self._actualizar_y_publicar_rumbo(now, permitir_correccion_lateral=True)

            # Criterio de parada por timeout de 20 segundos sin detectar árboles
            if tiempo_sin_ver_arbol >= self.timeout_sin_arbol:
                self.estado_actual = 'FINALIZADO'
                self.detener_todo_emergencia()
                self.get_logger().error(
                    f"TIMEOUT: {tiempo_sin_ver_arbol:.1f}s sin detectar ningún árbol "
                    f"Deteniendo robot"
                )
                return

            # Detección de tronco confirmada a la altura del LiDAR
            if self.arbol_en_mira:
                self.arboles_procesados += 1
                self.tiempo_inicio_estado = now
                self.tiempo_ultimo_arbol = now
                self.estado_actual = 'ALINEANDO_BRAZO'
                self.get_logger().info(
                    f"Tronco {self.arboles_procesados}/{self.meta_arboles} detectado. "
                    f"Alineando brazo por {self.tiempo_delay_brazo:.1f}s"
                )

        elif self.estado_actual == 'ALINEANDO_BRAZO':
            # Sigue avanzando durante tiempo_delay_brazo para que el brazo quede alineado frente al tronco detectado
            if tiempo_transcurrido < self.tiempo_delay_brazo:
                msg_vel.linear.x = self.velocidad_avance
                self._actualizar_y_publicar_rumbo(now, permitir_correccion_lateral=False)
            else:
                # Primero publicamos velocidad 0.0 (STOP_Mov) y luego accionamos el brazo (MOVE_Stepper)
                msg_vel.linear.x = 0.0
                self.cmd_pub.publish(msg_vel)
                self.estado_actual = 'EXTENDIENDO_BRAZO'
                self.tiempo_inicio_estado = now
                self.get_logger().info(
                    f"Brazo alineado con tronco {self.arboles_procesados}. "
                    f"Deteniendo ruedas y extendiendo brazo por {self.tiempo_brazo_extendido:.1f}s..."
                )
                self.comandar_brazo('EXTENDER')
                return

        elif self.estado_actual == 'EXTENDIENDO_BRAZO':
            # Ruedas detenidas, brazo extendiéndose/extendido durante tiempo_brazo_extendido
            msg_vel.linear.x = 0.0

            if tiempo_transcurrido >= self.tiempo_brazo_extendido:
                self.estado_actual = 'RETRAYENDO_BRAZO'
                self.tiempo_inicio_estado = now
                self.get_logger().info(
                    f"Extensión completada ({tiempo_transcurrido:.1f}s). "
                    f"Retrayendo brazo por {self.tiempo_brazo_retraido:.1f}s..."
                )
                self.comandar_brazo('RETRAER')

        elif self.estado_actual == 'RETRAYENDO_BRAZO':
            # Ruedas detenidas mientras el brazo termina de contraerse durante tiempo_brazo_retraido
            msg_vel.linear.x = 0.0

            if tiempo_transcurrido >= self.tiempo_brazo_retraido:
                if self.arboles_procesados >= self.meta_arboles:
                    self.estado_actual = 'FINALIZADO'
                    self.detener_todo_emergencia()
                    self.get_logger().info(
                        f"Meta alcanzada ({self.arboles_procesados}/{self.meta_arboles} árboles procesados). "
                        "Brazo retraído y robot detenido por completo."
                    )
                    return
                else:
                    self.estado_actual = 'ENFRIAMIENTO'
                    self.tiempo_inicio_estado = now
                    self.tiempo_ultimo_arbol = now
                    self.get_logger().info(
                        f"Árbol {self.arboles_procesados}/{self.meta_arboles} completado ({tiempo_transcurrido:.1f}s retracción). "
                        f"Reanudando marcha (enfriamiento de {self.tiempo_enfriamiento:.1f}s)..."
                    )

        elif self.estado_actual == 'ENFRIAMIENTO':
            # Avanza ignorando el tronco que acaba de procesar para no re-detectarlo
            msg_vel.linear.x = self.velocidad_avance
            self._actualizar_y_publicar_rumbo(now, permitir_correccion_lateral=False)

            if tiempo_transcurrido >= self.tiempo_enfriamiento:
                self.estado_actual = 'AVANZANDO'
                self.tiempo_inicio_estado = now
                self.tiempo_ultimo_arbol = now
                self.get_logger().info(
                    f"Buscando árbol {self.arboles_procesados + 1}/{self.meta_arboles}..."
                )

        self.cmd_pub.publish(msg_vel)


def main(args=None):
    rclpy.init(args=args)
    node = RealRutinaNavegacion()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.detener_todo_emergencia()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

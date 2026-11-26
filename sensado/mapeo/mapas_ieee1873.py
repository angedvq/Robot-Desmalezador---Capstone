
#!/usr/bin/env python3
import os
import math
import json
import time
import rclpy
import xml.etree.ElementTree as ET

from datetime import datetime, timezone
from rclpy.node import Node
from rclpy.time import Time
from geometry_msgs.msg import PoseArray
from tf2_ros import Buffer, TransformListener, TransformException


def _detectar_carpeta_mapeo_fuente() -> str:
    """
    Devuelve la ruta de la carpeta sensado/mapeo en el código
    fuente si existe, o el directorio del archivo actual.
    """
    dir_actual = os.path.dirname(os.path.abspath(__file__))

    candidato_ws = os.path.expanduser('~/sensado/mapeo')

    if os.path.isdir(candidato_ws):
        return candidato_ws

    return dir_actual


class GeneradorMapa(Node):
    def __init__(self):
        super().__init__('mapas_ieee1873')

        # ============================================================
        # 1. CONFIGURACIÓN DEL MAPA
        # ============================================================
        self.declare_parameter('meta_arboles', 4)
        self.declare_parameter('distancia_minima', 0.45)
        self.declare_parameter(
            'ruta_xml',
            os.path.join(
                _detectar_carpeta_mapeo_fuente(),
                'mapa_arboles.xml'
            ),
        )

        self.marco_mapa = 'odom'
        self.distancia_minima = float(
            self.get_parameter('distancia_minima').value
        )
        self.meta_arboles = int(
            self.get_parameter('meta_arboles').value
        )
        self.ruta_xml = os.path.abspath(
            os.path.expanduser(
                str(self.get_parameter('ruta_xml').value)
            )
        )

        self.arboles = []
        self.fecha_creacion = datetime.now(timezone.utc).isoformat()

        # ============================================================
        # 2. TRANSFORMACIONES TF
        # ============================================================
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(
            self.tf_buffer, self
        )

        # ============================================================
        # 3. SUSCRIPCIÓN A LOS ÁRBOLES DETECTADOS
        # ============================================================
        self.tree_sub = self.create_subscription(
            PoseArray,
            '/arboles_detectados',
            self.tree_callback,
            10
        )

        # ============================================================
        # 4. CONFIGURACIÓN DEL REGISTRO DE TRAYECTORIA
        # ============================================================
        self.t_inicio = time.monotonic()
        self.trayectoria = []

        self.ruta_json = os.path.join(
            os.path.dirname(self.ruta_xml),
            'mapa_complementario.json'
        )

        # Registrar posición cada 0.5 segundos
        self.create_timer(0.5, self.registrar_trayectoria)

        self.get_logger().info(
            f'Generador de mapa IEEE 1873 iniciado '
            f'(meta: {self.meta_arboles} árboles). '
            f'Archivo destino: {self.ruta_xml}'
        )

    # ================================================================
    # 5. TRANSFORMAR Y REGISTRAR LOS ÁRBOLES
    # ================================================================
    def tree_callback(self, msg: PoseArray):

        if not msg.poses:
            return

        if len(self.arboles) >= self.meta_arboles:
            return

        # Obtener la última transformación disponible
        # desde base_footprint hacia odom.
        try:
            transformacion = self.tf_buffer.lookup_transform(
                self.marco_mapa,
                'base_footprint',
                Time()
            )

        except TransformException as error:
            self.get_logger().warn(
                f'Transformación TF no disponible: {error}',
                throttle_duration_sec=5.0
            )
            return

        tx = float(transformacion.transform.translation.x)
        ty = float(transformacion.transform.translation.y)

        q = transformacion.transform.rotation

        # Obtener yaw a partir del cuaternión
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)

        yaw = math.atan2(siny_cosp, cosy_cosp)

        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        for pose in msg.poses:

            # Las coordenadas publicadas por el detector real
            # ya están expresadas en los ejes del robot.
            px = float(pose.position.x)
            py = float(pose.position.y)

            # Transformación rígida 2D hacia el marco odom
            x = tx + (px * cos_yaw - py * sin_yaw)
            y = ty + (px * sin_yaw + py * cos_yaw)

            # Comprobar si el árbol ya estaba registrado
            repetido = False

            for ax, ay in self.arboles:
                distancia = math.hypot(x - ax, y - ay)

                if distancia < self.distancia_minima:
                    repetido = True
                    break

            if repetido:
                continue

            # Guardar solamente árboles nuevos
            self.arboles.append((x, y))

            self.get_logger().info(
                f'Árbol {len(self.arboles)}/{self.meta_arboles} '
                f'registrado en mapa: '
                f'x={x:.3f} m, y={y:.3f} m'
            )

            self.guardar_xml()

            if len(self.arboles) >= self.meta_arboles:
                self.get_logger().info(
                    f'{len(self.arboles)}/{self.meta_arboles} '
                    f'árboles registrados. Mapa completado.'
                )
                break

    # ================================================================
    # 6. REGISTRAR LA TRAYECTORIA DEL ROBOT
    # ================================================================
    def registrar_trayectoria(self):

        try:
            transformacion = self.tf_buffer.lookup_transform(
                self.marco_mapa,
                'base_footprint',
                Time()
            )

        except TransformException:
            return

        # Obtener posición estimada del robot
        x = float(transformacion.transform.translation.x)
        y = float(transformacion.transform.translation.y)

        # Tiempo transcurrido desde el inicio del nodo
        t = time.monotonic() - self.t_inicio

        # Registrar posición y tiempo
        self.trayectoria.append({
            't': round(t, 3),
            'x': round(x, 4),
            'y': round(y, 4)
        })

        self.guardar_trayectoria()

    # ================================================================
    # 7. GUARDAR TRAYECTORIA EN ARCHIVO JSON
    # ================================================================
    def guardar_trayectoria(self):

        datos = {
            'marco': self.marco_mapa,
            'trayectoria': self.trayectoria
        }

        # Asegurar que exista la carpeta destino
        os.makedirs(
            os.path.dirname(self.ruta_json),
            exist_ok=True
        )

        # Escribir primero en un archivo temporal
        temporal = self.ruta_json + '.tmp'

        with open(temporal, 'w', encoding='utf-8') as archivo:
            json.dump(datos, archivo, indent=4)

        # Reemplazar el JSON anterior
        os.replace(temporal, self.ruta_json)

    # ================================================================
    # 8. GENERAR ARCHIVO XML DEL MAPA GEOMÉTRICO
    # ================================================================
    def guardar_xml(self):

        raiz = ET.Element('mdr:maps', {
            'xmlns:mdr': 'http://www.example.org/mdr'
        })

        mapa = ET.SubElement(raiz, 'geometric_map', {
            'id': 'MapaArboles',
            'maptype': '2'
        })

        # Metadatos del mapa
        metadata = ET.SubElement(mapa, 'metadata')

        autores = ET.SubElement(metadata, 'authors')
        ET.SubElement(autores, 'author').text = 'Equipo Capstone'

        ET.SubElement(metadata, 'description').text = (
            'Mapa de árboles detectados mediante LiDAR. '
            f'Coordenadas en el marco ROS {self.marco_mapa}.'
        )

        ET.SubElement(
            metadata, 'creation_date'
        ).text = self.fecha_creacion

        ET.SubElement(
            metadata, 'last_modified'
        ).text = datetime.now(timezone.utc).isoformat()

        # Sistema de coordenadas local
        ET.SubElement(mapa, 'coordinate_system')

        # Elementos geométricos: centros de los troncos
        elementos = ET.SubElement(mapa, 'elements')

        for x, y in self.arboles:
            ET.SubElement(elementos, 'point', {
                'x': f'{x:.4f}',
                'y': f'{y:.4f}'
            })

        # Asegurar que exista la carpeta destino
        dir_destino = os.path.dirname(self.ruta_xml)

        if dir_destino:
            os.makedirs(dir_destino, exist_ok=True)

        # Escribir archivo XML
        arbol_xml = ET.ElementTree(raiz)
        ET.indent(arbol_xml, space='    ')

        arbol_xml.write(
            self.ruta_xml,
            encoding='utf-8',
            xml_declaration=True
        )

        self.get_logger().info(
            f'Mapa guardado: {self.ruta_xml}'
        )


# ====================================================================
# 9. EJECUCIÓN DEL NODO ROS 2
# ====================================================================
def main(args=None):
    rclpy.init(args=args)
    node = GeneradorMapa()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.guardar_trayectoria()
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

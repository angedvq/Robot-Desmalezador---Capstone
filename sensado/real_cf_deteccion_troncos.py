#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseArray, Pose, Point
from std_msgs.msg import Float32
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
from sklearn.cluster import DBSCAN


def circfit_HyperSVD(XY):
    """
    Ajusta una circunferencia a una nube de puntos 2D HyperSVD (sin sesgo hiperbólico en arcos parciales).
    """
    XY = np.asarray(XY, dtype=float)
    if len(XY) < 4:
        raise ValueError("Se requieren al menos 4 puntos para el ajuste HyperSVD")

    centroid = np.mean(XY, axis=0)
    X = XY[:, 0] - centroid[0]
    Y = XY[:, 1] - centroid[1]
    Z = X**2 + Y**2
    ZXY1 = np.column_stack((Z, X, Y, np.ones(len(Z))))
    _, S, Vt = np.linalg.svd(ZXY1, full_matrices=False)
    V = Vt.T

    if S[3] / S[0] < 1e-12:
        A = V[:, 3]
    else:
        R_mean = np.mean(ZXY1, axis=0)
        N = np.array(
            [
                [8.0 * R_mean[0], 4.0 * R_mean[1], 4.0 * R_mean[2], 2.0],
                [4.0 * R_mean[1], 1.0, 0.0, 0.0],
                [4.0 * R_mean[2], 0.0, 1.0, 0.0],
                [2.0, 0.0, 0.0, 0.0],
            ],
            dtype=float,
        )
        W = V @ np.diag(S) @ V.T
        N_inv = np.linalg.pinv(N)
        D, E = np.linalg.eig(W @ N_inv @ W)
        E = np.real(E)
        D = np.real(D)
        ID = np.argsort(D)
        Astar = E[:, ID[1]]
        A, _, _, _ = np.linalg.lstsq(W, Astar, rcond=None)

    if np.isclose(A[0], 0.0):
        raise ValueError("Coeficiente cuadrático A[0] muy cercano a cero (línea recta)")

    rad_inner = A[1]**2 + A[2]**2 - 4.0 * A[0] * A[3]
    if rad_inner <= 0.0:
        raise ValueError("Discriminante de radio negativo")

    center_x = -(A[1]) / (2.0 * A[0]) + centroid[0]
    center_y = -(A[2]) / (2.0 * A[0]) + centroid[1]
    radius = np.sqrt(rad_inner) / (2.0 * np.abs(A[0]))

    return float(center_x), float(center_y), float(radius)


def detectar_troncos(angles,ranges,range_min_msg,range_max_msg,fov_busqueda_min,fov_busqueda_max,fov_disparo_min,
                     fov_disparo_max,roi_min_dist,roi_max_dist,radio_min,radio_max,rumbo_imu_rad=0.0,
                     dbscan_eps=0.15,dbscan_min_samples=4):
    """
    Procesa dos zonas de deteccion:
    1. Zona diagonal/lookahead:
       Permite ver el próximo tronco mientras el robot se aproxima a él y medir su
       distancia perpendicular para corregir el rumbo hacia 0.70 m.
    2. Zona lateral:
       Detecta cuando el tronco pasa justo al costado izquierdo del LiDAR
    """
    # Normalizar ángulos del LiDAR a [-pi, pi]
    angles = np.arctan2(np.sin(angles), np.cos(angles))
    min_r = max(float(range_min_msg), roi_min_dist, 0.15)
    max_r = min(float(range_max_msg), roi_max_dist)

    # Filtrar lecturas dentro de una distancia y angulos validos
    valid_ranges = (ranges >= min_r) & (ranges <= max_r) & np.isfinite(ranges)
    valid_angles = (angles >= fov_busqueda_min) & (angles <= fov_busqueda_max)
    valid = valid_ranges & valid_angles

    resultado = {
        'tronco_en_disparo': None,
        'tronco_proximo': None,
        'distancia_lateral_guia': None,
        'todos_los_troncos': [],
    }

    if np.count_nonzero(valid) < dbscan_min_samples:
        return resultado

    ang_v = angles[valid]
    rng_v = ranges[valid]

    # Coordenadas cartesianas en el marco del LiDAR (para RViz y DBSCAN)
    X_lidar = rng_v * np.cos(ang_v)
    Y_lidar = rng_v * np.sin(ang_v)
    XY_lidar = np.column_stack((X_lidar, Y_lidar))

    # Agrupa puntos separados por menos de 0.15 mts. Un tronco es valido si hay al menos 4 puntos.
    clustering = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit(XY_lidar)
    labels = clustering.labels_

    candidatos_disparo = []
    candidatos_guia = []

    # Analizar cada cluster detectado
    for label in set(labels):
        if label == -1: # filtrar ruido
            continue

        cluster_points = XY_lidar[labels == label]
        if not (dbscan_min_samples <= len(cluster_points) < 80): # filtrar numero de muestras 
            continue

        try:
            # Ajustar circulo a los clusters
            cx_l, cy_l, r = circfit_HyperSVD(cluster_points)
        except ValueError:
            continue

        # Filtro de radio físico del tronco
        if not (radio_min <= r <= radio_max):
            continue

        # Filtro de distancia radial desde el LiDAR al centro del tronco (ROI)
        dist_to_center = math.hypot(cx_l, cy_l)
        if not (roi_min_dist <= dist_to_center <= roi_max_dist):
            continue

        # La cara iluminada por el láser debe estar más cerca del sensor que el centro geométrico calculado
        mean_pts_dist = float(np.mean(np.linalg.norm(cluster_points, axis=1)))
        if dist_to_center < mean_pts_dist:
            continue

        # Ajustar coordenadas del lidar al marco del robot
        ang_centro = math.atan2(cy_l, cx_l)
        x_robot = -cx_l  # +X_robot es hacia adelante (-X_lidar)
        y_robot = -cy_l  # +Y_robot es hacia la izquierda (-Y_lidar)

        # Solo considerar troncos a la izquierda del robot
        if y_robot <= 0.10:
            continue

        # Clasificar tronco hallado si está dentro de la zona de deteccion
        en_disparo = (fov_disparo_min <= ang_centro <= fov_disparo_max)
        resultado['todos_los_troncos'].append((cx_l, cy_l, r, en_disparo))
        
        if en_disparo:
            candidatos_disparo.append((dist_to_center, x_robot, y_robot, r, cx_l, cy_l))

        # Proyección compensada con el ángulo actual de la IMU (rumbo_imu_rad) respecto
        # a la línea de referencia 0° de la hilera de árboles:
        #   x_hilera: distancia longitudinal real hacia adelante a lo largo de la hilera
        #   d_perp:   distancia perpendicular real a la hilera (invariante al giro del chasis)
        cos_t = math.cos(rumbo_imu_rad)
        sin_t = math.sin(rumbo_imu_rad)
        x_hilera = x_robot * cos_t - y_robot * sin_t
        d_perp = x_robot * sin_t + y_robot * cos_t

        # Para el guiado lateral a 0.70 m solo usamos el lookahead mientras el próximo
        # tronco está entre 0.45 m y 2.00 m por delante. Al bajar de 0.45 m, deja de guiar
        # para que la IMU de la STM32 enderece el robot a 0.0° antes de que el tronco
        # entre al cono lateral de detección [-100°, -80°].
        if 0.45 <= x_hilera <= 2.0 and d_perp > 0.10:
            candidatos_guia.append((dist_to_center, x_robot, d_perp, r, cx_l, cy_l))

    if candidatos_disparo:
        # Si hay más de un candidato en la ventana de disparo, elegir el más cercano
        candidatos_disparo.sort(key=lambda item: item[0])
        _, x_r, y_r, r_t, cx_l, cy_l = candidatos_disparo[0]
        resultado['tronco_en_disparo'] = (x_r, y_r, r_t, cx_l, cy_l)

    if candidatos_guia:
        # Elegir el tronco válido más cercano para guiar la distancia lateral a la hilera
        candidatos_guia.sort(key=lambda item: item[0])
        _, x_r, d_p, r_t, cx_l, cy_l = candidatos_guia[0]
        resultado['tronco_proximo'] = (x_r, d_p, r_t, cx_l, cy_l)
        resultado['distancia_lateral_guia'] = float(d_p)

    return resultado


class RealPercepcionTroncos(Node):
    def __init__(self):
        super().__init__('real_cf_deteccion_troncos')

        # ---------------------------------------------------------------------
        # Declaración de parámetros 
        # ---------------------------------------------------------------------
        # FOV de disparo lateral
        self.declare_parameter('fov_disparo_min_deg', -100.0)
        self.declare_parameter('fov_disparo_max_deg', -80.0)

        # Ventana amplia de búsqueda y guiado lateral hacia la hilera de árboles
        # cubre desde la diagonal frontal-izquierda hasta el lateral
        self.declare_parameter('fov_busqueda_min_deg', -160.0)
        self.declare_parameter('fov_busqueda_max_deg', -60.0)

        # Distancias y radios válidos de tronco (m)
        self.declare_parameter('roi_min_dist', 0.15)
        self.declare_parameter('roi_max_dist', 2.50)
        self.declare_parameter('radio_tronco_min', 0.14)
        self.declare_parameter('radio_tronco_max', 0.22)

        # Confirmación por escaneos consecutivos (evita falsos positivos)
        self.declare_parameter('required_detections', 4)

        # Extraer valores de parámetros
        self.fov_disparo_min = np.deg2rad(float(self.get_parameter('fov_disparo_min_deg').value))
        self.fov_disparo_max = np.deg2rad(float(self.get_parameter('fov_disparo_max_deg').value))
        self.fov_busqueda_min = np.deg2rad(float(self.get_parameter('fov_busqueda_min_deg').value))
        self.fov_busqueda_max = np.deg2rad(float(self.get_parameter('fov_busqueda_max_deg').value))
        self.roi_min_dist = float(self.get_parameter('roi_min_dist').value)
        self.roi_max_dist = float(self.get_parameter('roi_max_dist').value)
        self.radio_min = float(self.get_parameter('radio_tronco_min').value)
        self.radio_max = float(self.get_parameter('radio_tronco_max').value)
        self.required_detections = int(self.get_parameter('required_detections').value)

        self.consecutive_detections = 0
        self.rumbo_imu_rad = 0.0

        # ---------------------------------------------------------------------
        # Suscriptores y Publicadores
        # ---------------------------------------------------------------------
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.rumbo_sub = self.create_subscription(
            Float32, '/navegacion/corregir_grados', self.rumbo_callback, 10
        )

        # Publica PoseArray al confirmar un tronco en el costado izquierdo 
        self.pose_pub = self.create_publisher(PoseArray, '/arboles_detectados', 10)

        # Publica la distancia perpendicular (m) al tronco para mantener los 0.7 m
        self.dist_lat_pub = self.create_publisher(Float32, '/lidar/distancia_tronco_lateral', 10)

        # Marcadores 3D para RViz
        self.marker_pub = self.create_publisher(MarkerArray, '/tree_markers', 10)

        self.get_logger().info(
            f"real_cf_deteccion_troncos iniciado | Disparo: "
            f"[{np.rad2deg(self.fov_disparo_min):.0f}°, {np.rad2deg(self.fov_disparo_max):.0f}°] | "
            f"Guiado: [{np.rad2deg(self.fov_busqueda_min):.0f}°, {np.rad2deg(self.fov_busqueda_max):.0f}°] | "
            f"Radio tronco: [{self.radio_min:.2f}, {self.radio_max:.2f}] m"
        )

    def rumbo_callback(self, msg: Float32):
        """Actualiza el ángulo de referencia enviado a la IMU de la STM32 (en radianes)."""
        if math.isfinite(msg.data):
            self.rumbo_imu_rad = math.radians(float(msg.data))

    def scan_callback(self, msg: LaserScan):
        angles = msg.angle_min + np.arange(len(msg.ranges)) * msg.angle_increment
        ranges = np.array(msg.ranges, dtype=float)

        info = detectar_troncos(
            angles=angles,
            ranges=ranges,
            range_min_msg=msg.range_min,
            range_max_msg=msg.range_max,
            fov_busqueda_min=self.fov_busqueda_min,
            fov_busqueda_max=self.fov_busqueda_max,
            fov_disparo_min=self.fov_disparo_min,
            fov_disparo_max=self.fov_disparo_max,
            roi_min_dist=self.roi_min_dist,
            roi_max_dist=self.roi_max_dist,
            radio_min=self.radio_min,
            radio_max=self.radio_max,
            rumbo_imu_rad=self.rumbo_imu_rad,
        )

        # Publicar distancia lateral al tronco (o NaN si estamos en un espacio vacio)
        msg_dist = Float32()
        if info['distancia_lateral_guia'] is not None:
            msg_dist.data = float(info['distancia_lateral_guia'])
        else:
            msg_dist.data = float('nan')
        self.dist_lat_pub.publish(msg_dist)

        # Conteo de confirmaciones consecutivas en la ventana lateral
        tronco_disparo = info['tronco_en_disparo']
        if tronco_disparo is not None:
            self.consecutive_detections += 1
        else:
            self.consecutive_detections = 0

        # Publicar /arboles_detectados
        pose_array_msg = PoseArray()
        pose_array_msg.header.frame_id = msg.header.frame_id
        pose_array_msg.header.stamp = self.get_clock().now().to_msg()

        # Si detectamos un arbol valido, publicar posicion
        confirmado_lateral = (
            tronco_disparo is not None
            and self.consecutive_detections >= self.required_detections
        )
        if confirmado_lateral:
            x_robot, y_robot, r_tronco, _, _ = tronco_disparo
            pose = Pose()
            pose.position.x = float(x_robot)
            pose.position.y = float(y_robot)
            pose.position.z = float(r_tronco)
            pose_array_msg.poses.append(pose)

        self.pose_pub.publish(pose_array_msg)

        # Publicar Markers 3D para RViz2 (/tree_markers) en el mismo frame_id del LiDAR
        frame_id = msg.header.frame_id if msg.header.frame_id else 'laser_link'
        self._publicar_markers_rviz(
            frame_id=frame_id,
            tronco_proximo=info['tronco_proximo'],
            tronco_disparo=tronco_disparo,
            confirmado_lateral=confirmado_lateral,
        )

    def _publicar_markers_rviz(self, frame_id: str, tronco_proximo, tronco_disparo, confirmado_lateral: bool):
        """
        Publica en `/tree_markers` (visualization_msgs/MarkerArray) los conos de visión
        y los cilindros/etiquetas del árbol en lookahead y del árbol lateral para RViz2.
        """
        stamp = self.get_clock().now().to_msg()
        marker_array = MarkerArray()

        clear_marker = Marker()
        clear_marker.header.frame_id = frame_id
        clear_marker.header.stamp = stamp
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)

        # 1. Conos visuales de las dos zonas de detección del LiDAR
        marker_array.markers.append(
            self._crear_cono_fov_rviz(
                stamp, frame_id, m_id=100,
                ang_min=self.fov_busqueda_min, ang_max=self.fov_busqueda_max,
                radio=2.0, r=0.0, g=0.75, b=1.0, a=0.35, ns='fov_guiado_amplio'
            )
        )
        marker_array.markers.append(
            self._crear_cono_fov_rviz(
                stamp, frame_id, m_id=101,
                ang_min=self.fov_disparo_min, ang_max=self.fov_disparo_max,
                radio=1.5, r=0.1, g=1.0, b=0.2, a=0.60, ns='fov_disparo_lateral'
            )
        )

        # 2. Marker del ÁRBOL PRÓXIMO / LOOKAHEAD (Zona de guiado: CIAN + Línea Amarilla)
        if tronco_proximo is not None:
            _, d_perp, r_t, cx_l, cy_l = tronco_proximo
            m_prox = Marker()
            m_prox.header.frame_id = frame_id
            m_prox.header.stamp = stamp
            m_prox.ns = 'arbol_proximo_guia'
            m_prox.id = 1
            m_prox.type = Marker.CYLINDER
            m_prox.action = Marker.ADD
            m_prox.pose.position.x = float(cx_l)
            m_prox.pose.position.y = float(cy_l)
            m_prox.pose.position.z = 0.4
            m_prox.scale.x = float(r_t * 2.4)
            m_prox.scale.y = float(r_t * 2.4)
            m_prox.scale.z = 1.0
            m_prox.color.r, m_prox.color.g, m_prox.color.b, m_prox.color.a = (0.0, 0.85, 1.0, 0.65)
            marker_array.markers.append(m_prox)

            m_line = Marker()
            m_line.header.frame_id = frame_id
            m_line.header.stamp = stamp
            m_line.ns = 'arbol_proximo_guia'
            m_line.id = 2
            m_line.type = Marker.LINE_STRIP
            m_line.action = Marker.ADD
            m_line.scale.x = 0.03
            m_line.color.r, m_line.color.g, m_line.color.b, m_line.color.a = (1.0, 0.9, 0.0, 0.9)
            m_line.points = [
                Point(x=0.0, y=0.0, z=0.0),
                Point(x=float(cx_l), y=float(cy_l), z=0.0),
            ]
            marker_array.markers.append(m_line)

            m_txt = Marker()
            m_txt.header.frame_id = frame_id
            m_txt.header.stamp = stamp
            m_txt.ns = 'arbol_proximo_guia'
            m_txt.id = 3
            m_txt.type = Marker.TEXT_VIEW_FACING
            m_txt.action = Marker.ADD
            m_txt.pose.position.x = float(cx_l)
            m_txt.pose.position.y = float(cy_l)
            m_txt.pose.position.z = 1.1
            m_txt.scale.z = 0.18
            m_txt.color.r, m_txt.color.g, m_txt.color.b, m_txt.color.a = (0.0, 0.95, 1.0, 1.0)
            m_txt.text = f"LOOKAHEAD (d_perp={d_perp:.2f}m | R={r_t*100:.1f}cm)"
            marker_array.markers.append(m_txt)

        # 3. Marker del ÁRBOL LATERAL (Zona de disparo [-100°, -80°]: NARANJA -> VERDE)
        if tronco_disparo is not None:
            _, y_r, r_t, cx_l, cy_l = tronco_disparo
            m_lat = Marker()
            m_lat.header.frame_id = frame_id
            m_lat.header.stamp = stamp
            m_lat.ns = 'arbol_lateral_disparo'
            m_lat.id = 10
            m_lat.type = Marker.CYLINDER
            m_lat.action = Marker.ADD
            m_lat.pose.position.x = float(cx_l)
            m_lat.pose.position.y = float(cy_l)
            m_lat.pose.position.z = 0.5
            m_lat.scale.x = float(r_t * 2.8)
            m_lat.scale.y = float(r_t * 2.8)
            m_lat.scale.z = 1.2
            if confirmado_lateral:
                m_lat.color.r, m_lat.color.g, m_lat.color.b, m_lat.color.a = (0.0, 1.0, 0.1, 0.85)
            else:
                m_lat.color.r, m_lat.color.g, m_lat.color.b, m_lat.color.a = (1.0, 0.6, 0.0, 0.75)
            marker_array.markers.append(m_lat)

            m_lat_txt = Marker()
            m_lat_txt.header.frame_id = frame_id
            m_lat_txt.header.stamp = stamp
            m_lat_txt.ns = 'arbol_lateral_disparo'
            m_lat_txt.id = 11
            m_lat_txt.type = Marker.TEXT_VIEW_FACING
            m_lat_txt.action = Marker.ADD
            m_lat_txt.pose.position.x = float(cx_l)
            m_lat_txt.pose.position.y = float(cy_l)
            m_lat_txt.pose.position.z = 1.3
            m_lat_txt.scale.z = 0.20
            m_lat_txt.color.r, m_lat_txt.color.g, m_lat_txt.color.b, m_lat_txt.color.a = (0.1, 1.0, 0.2, 1.0)
            estado_txt = "CONFIRMADO!" if confirmado_lateral else f"verificando {self.consecutive_detections}/{self.required_detections}"
            m_lat_txt.text = f"ARBOL LATERAL ({estado_txt} | Y={y_r:.2f}m)"
            marker_array.markers.append(m_lat_txt)

        self.marker_pub.publish(marker_array)

    def _crear_cono_fov_rviz(self, stamp, frame_id, m_id, ang_min, ang_max, radio, r, g, b, a, ns):
        """Dibuja en RViz2 los bordes angulares de un sector del LiDAR."""
        m = Marker()
        m.header.frame_id = frame_id
        m.header.stamp = stamp
        m.ns = ns
        m.id = m_id
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = 0.02
        m.color.r, m.color.g, m_color_b, m.color.a = (float(r), float(g), float(b), float(a))
        m.color.b = float(b)

        pts = [Point(x=0.0, y=0.0, z=0.0)]
        for ang in np.linspace(ang_min, ang_max, 12):
            pts.append(Point(x=float(radio * math.cos(ang)), y=float(radio * math.sin(ang)), z=0.0))
        pts.append(Point(x=0.0, y=0.0, z=0.0))
        m.points = pts
        return m

def main(args=None):
    rclpy.init(args=args)
    node = RealPercepcionTroncos()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
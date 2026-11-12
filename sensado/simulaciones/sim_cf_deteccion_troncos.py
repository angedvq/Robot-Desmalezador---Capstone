#!/usr/bin/env python3
"""
===============================================================================
NODO DE SIMULACIÓN: sim_cf_deteccion_troncos.py
===============================================================================
Ubicación: `sensado/simulaciones/sim_cf_deteccion_troncos.py`
Equivalente en simulación de `real_cf_deteccion_troncos.py`.

PROPÓSITO:
1. Ejecutar exactamente la misma lógica matemática de detección de troncos que
   el robot real (`DBSCAN` + `circfit_HyperSVD` + filtros de radio, distancia y
   convexidad) considerando el RPLidar A1 montado en la esquina superior
   izquierda del robot con su orientación real (-90° hacia la hilera izquierda).
2. Detectar y publicar las dos zonas de interés a la izquierda del robot:
   - ZONA DE BÚSQUEDA / GUIADO AMPLIO (`[-160°, -60°]`):
     Detecta el **árbol próximo** mientras el robot se acerca en diagonal y
     publica su distancia perpendicular en `/lidar/distancia_tronco_lateral`
     (`Float32`) para mantener al robot a 0.70 m de la hilera.
   - ZONA DE DISPARO LATERAL (`[-100°, -80°]`, es decir `-90° +- 10°`):
     Detecta el **árbol lateral** justo al costado del LiDAR y lo publica en
     `/arboles_detectados` (`PoseArray`) para iniciar el delay de 1.0 s que
     alinea el brazo trasero con el tronco.
3. VISUALIZACIÓN DE MARKERS FUNCIONAL EN GAZEBO FORTRESS Y RVIZ2:
   En versiones anteriores los Markers nunca funcionaron por dos motivos:
   a) `ros_gz_bridge` en ROS 2 Humble no soporta puentear `/marker` como tópico;
      Gazebo Fortress renderiza marcadores 3D únicamente mediante el servicio de
      Ignition Transport `/marker` (`ignition.msgs.Marker`) y exige coordenadas
      globales del marco `world` (con `visibility: GUI` para que el láser no
      rebote contra el propio marcador).
   b) En RViz2, `/scan` traía el `frame_id` interno de Gazebo en vez de
      `lidar_link`.
   Este nodo soluciona ambos problemas simultáneamente:
   - Publica `MarkerArray` en `/tree_markers` con `frame_id = 'lidar_link'` para
     RViz2 (incluyendo cilindros, líneas guía, textos y conos de visión).
   - Transforma las posiciones detectadas al marco global `world` usando `/odom`
     y actualiza en un hilo asíncrono el servicio `/marker` de Gazebo Fortress,
     mostrando directamente en la ventana 3D de Gazebo:
       * ÁRBOL PRÓXIMO (Zona de guiado): Cilindro CIAN/AMARILLO + línea guía
         desde el LiDAR + etiqueta de distancia.
       * ÁRBOL LATERAL (Zona de disparo): Cilindro VERDE BRILLANTE + etiqueta
         cuando el tronco está al costado del LiDAR.
===============================================================================
"""

import math
import os
import subprocess
import threading
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, Pose, PoseArray
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32
from tf2_msgs.msg import TFMessage
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
from sklearn.cluster import DBSCAN


def circfit_HyperSVD(XY):
    """
    Ajusta una circunferencia a una nube de puntos 2D (N x 2) mediante el
    algoritmo algebraico HyperSVD.

    Parámetros:
        XY (np.ndarray): Matriz (N, 2) con coordenadas [x, y] en metros.

    Retorna:
        tuple(float, float, float): (center_x, center_y, radius) en metros.
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
        raise ValueError("Coeficiente A[0] cercano a cero")

    rad_inner = A[1]**2 + A[2]**2 - 4.0 * A[0] * A[3]
    if rad_inner <= 0.0:
        raise ValueError("Discriminante de radio negativo")

    center_x = -(A[1]) / (2.0 * A[0]) + centroid[0]
    center_y = -(A[2]) / (2.0 * A[0]) + centroid[1]
    radius = np.sqrt(rad_inner) / (2.0 * np.abs(A[0]))

    return float(center_x), float(center_y), float(radius)


def detectar_troncos_en_scan(
    angles,
    ranges,
    range_min_msg,
    range_max_msg,
    fov_busqueda_min,
    fov_busqueda_max,
    fov_disparo_min,
    fov_disparo_max,
    roi_min_dist,
    roi_max_dist,
    radio_min,
    radio_max,
    dbscan_eps=0.15,
    dbscan_min_samples=4,
):
    """
    Analiza un barrido LaserScan y clasifica los troncos detectados en:
    - `tronco_proximo`: tronco válido más cercano en la zona de búsqueda/guiado
      amplio (`[fov_busqueda_min, fov_busqueda_max]`, ej. `[-160°, -60°]`).
    - `tronco_en_disparo`: tronco válido dentro de la ventana lateral estrecha
      (`[fov_disparo_min, fov_disparo_max]`, ej. `[-100°, -80°]`).
    """
    angles = np.arctan2(np.sin(angles), np.cos(angles))
    min_r = max(float(range_min_msg), roi_min_dist, 0.15)
    max_r = min(float(range_max_msg), roi_max_dist)

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

    X_lidar = rng_v * np.cos(ang_v)
    Y_lidar = rng_v * np.sin(ang_v)
    XY_lidar = np.column_stack((X_lidar, Y_lidar))

    clustering = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit(XY_lidar)
    labels = clustering.labels_

    candidatos_disparo = []
    candidatos_guia = []

    for label in set(labels):
        if label == -1:
            continue

        cluster_points = XY_lidar[labels == label]
        if not (dbscan_min_samples <= len(cluster_points) < 80):
            continue

        try:
            cx_l, cy_l, r = circfit_HyperSVD(cluster_points)
        except ValueError:
            continue

        if not (radio_min <= r <= radio_max):
            continue

        dist_to_center = math.hypot(cx_l, cy_l)
        if not (roi_min_dist <= dist_to_center <= roi_max_dist):
            continue

        mean_pts_dist = float(np.mean(np.linalg.norm(cluster_points, axis=1)))
        if dist_to_center < mean_pts_dist:
            continue

        ang_centro = math.atan2(cy_l, cx_l)
        # Conversión del marco del LiDAR (rotado pi rad) al marco del robot:
        # +X_robot = adelante (-X_lidar), +Y_robot = izquierda (-Y_lidar)
        x_robot = -cx_l
        y_robot = -cy_l

        if y_robot <= 0.10:
            continue

        en_disparo = (fov_disparo_min <= ang_centro <= fov_disparo_max)
        resultado['todos_los_troncos'].append((cx_l, cy_l, r, en_disparo))

        if en_disparo:
            candidatos_disparo.append((dist_to_center, x_robot, y_robot, r, cx_l, cy_l))

        if -0.25 <= x_robot <= 2.20:
            candidatos_guia.append((dist_to_center, x_robot, y_robot, r, cx_l, cy_l))

    if candidatos_disparo:
        candidatos_disparo.sort(key=lambda item: item[0])
        _, x_r, y_r, r_t, cx_l, cy_l = candidatos_disparo[0]
        resultado['tronco_en_disparo'] = (x_r, y_r, r_t, cx_l, cy_l)

    if candidatos_guia:
        candidatos_guia.sort(key=lambda item: item[0])
        _, x_r, y_r, r_t, cx_l, cy_l = candidatos_guia[0]
        resultado['tronco_proximo'] = (x_r, y_r, r_t, cx_l, cy_l)
        resultado['distancia_lateral_guia'] = float(y_r)

    return resultado


class GazeboMarkerRenderer:
    """
    Cliente asíncrono no bloqueante para dibujar Markers 3D directamente en la
    ventana de Gazebo Fortress invocando el servicio `/marker_array` del plugin
    `MarkerManager` de Ignition GUI (`ign-transport-service`).

    Nota técnica importante:
    1. En Ignition Gazebo 6 (`libMarkerManager.so`), el servicio `/marker` es
       unidireccional (`Empty` reply), mientras que `/marker_array` es el
       servicio bidireccional (`ignition.msgs.Marker_V` -> `ignition.msgs.Boolean`)
       compatible con `ign-transport-service` y permite actualizar todos los
       marcadores en una sola llamada IPC por ciclo.
    2. Todos los marcadores se envían con `visibility: GUI` para que sean visibles
       en la cámara 3D de Gazebo pero INVISIBLES para los rayos del sensor
       `gpu_lidar` (evitando falsos rebotes del LiDAR contra el propio marcador).
    """

    IGN_SERVICE_BIN = '/usr/libexec/gz/transport11/ign-transport-service'

    def __init__(self, enabled=True):
        self.enabled = enabled and os.path.exists(self.IGN_SERVICE_BIN)
        self._lock = threading.Lock()
        self._latest_active = {}
        self._pending_deletes = {}
        self._event = threading.Event()
        self._stop = False
        if self.enabled:
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()

    def actualizar_marcadores(self, activos_dict, deletes_dict):
        """
        Actualiza el estado de marcadores activos (`{(ns, id): req_str}`) y
        acumula eliminaciones pendientes (`{(ns, id): req_str}`) para que
        ningún `DELETE_MARKER` se pierda entre barridos del LiDAR.
        """
        if not self.enabled:
            return
        with self._lock:
            for key, req in deletes_dict.items():
                self._pending_deletes[key] = req
            for key in activos_dict:
                self._pending_deletes.pop(key, None)
            self._latest_active = dict(activos_dict)
            hay_trabajo = bool(self._latest_active or self._pending_deletes)
        if hay_trabajo:
            self._event.set()

    def _worker(self):
        while not self._stop:
            self._event.wait(timeout=0.5)
            self._event.clear()
            with self._lock:
                items = list(self._pending_deletes.values()) + list(
                    self._latest_active.values()
                )
                self._pending_deletes.clear()
                self._latest_active.clear()

            if not items or self._stop:
                continue

            req_str = ' '.join(f'marker {{ {m} }}' for m in items)
            try:
                subprocess.run(
                    [
                        self.IGN_SERVICE_BIN,
                        '-s', '/marker_array',
                        '--reqtype', 'ignition.msgs.Marker_V',
                        '--reptype', 'ignition.msgs.Boolean',
                        '--timeout', '200',
                        '--req', req_str,
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=0.35,
                )
            except Exception:
                pass

    def cerrar(self):
        self._stop = True
        self._event.set()


class SimPercepcionTroncos(Node):
    """
    Nodo de percepción de troncos en simulación Gazebo con visualización dual
    de Markers (Gazebo Fortress 3D + RViz2) para el árbol próximo y el árbol lateral.
    """

    # Offset físico del RPLidar A1 respecto a base_link en robot_desmalezador.urdf.xacro
    LIDAR_OFFSET_X = 0.15
    LIDAR_OFFSET_Y = 0.12
    LIDAR_OFFSET_Z = 0.33

    def __init__(self):
        super().__init__('sim_cf_deteccion_troncos')

        # Parámetros de detección (idénticos a real_cf_deteccion_troncos.py)
        self.declare_parameter('fov_disparo_min_deg', -100.0)
        self.declare_parameter('fov_disparo_max_deg', -80.0)
        self.declare_parameter('fov_busqueda_min_deg', -160.0)
        self.declare_parameter('fov_busqueda_max_deg', -60.0)
        self.declare_parameter('roi_min_dist', 0.15)
        self.declare_parameter('roi_max_dist', 2.50)
        self.declare_parameter('radio_tronco_min', 0.05)
        self.declare_parameter('radio_tronco_max', 0.18)
        self.declare_parameter('required_detections', 4)

        # Posición inicial de spawn en Gazebo para transformar `/odom` al marco `world`
        self.declare_parameter('spawn_x', -6.0)
        self.declare_parameter('spawn_y', -1.82)
        self.declare_parameter('spawn_yaw', 0.0)
        self.declare_parameter('usar_gazebo_markers', True)

        self.fov_disparo_min = np.deg2rad(float(self.get_parameter('fov_disparo_min_deg').value))
        self.fov_disparo_max = np.deg2rad(float(self.get_parameter('fov_disparo_max_deg').value))
        self.fov_busqueda_min = np.deg2rad(float(self.get_parameter('fov_busqueda_min_deg').value))
        self.fov_busqueda_max = np.deg2rad(float(self.get_parameter('fov_busqueda_max_deg').value))
        self.roi_min_dist = float(self.get_parameter('roi_min_dist').value)
        self.roi_max_dist = float(self.get_parameter('roi_max_dist').value)
        self.radio_min = float(self.get_parameter('radio_tronco_min').value)
        self.radio_max = float(self.get_parameter('radio_tronco_max').value)
        self.required_detections = int(self.get_parameter('required_detections').value)

        self.spawn_x = float(self.get_parameter('spawn_x').value)
        self.spawn_y = float(self.get_parameter('spawn_y').value)
        self.spawn_yaw = float(self.get_parameter('spawn_yaw').value)
        usar_gz_markers = bool(self.get_parameter('usar_gazebo_markers').value)

        self.consecutive_detections = 0
        self._gz_prox_activo = False
        self._gz_lat_activo = False

        # Pose actual del robot en el mundo de Gazebo (actualizada por
        # `/world/cultivo_arboles_world/dynamic_pose/info` o `/odom`)
        self.robot_world_x = self.spawn_x
        self.robot_world_y = self.spawn_y
        self.robot_world_yaw = self.spawn_yaw
        self._tiene_pose_mundo_exacta = False

        # Suscriptores
        self.subscription = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10
        )
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10
        )
        self.world_pose_sub = self.create_subscription(
            TFMessage,
            '/world/cultivo_arboles_world/dynamic_pose/info',
            self.world_pose_callback,
            10,
        )

        # Publicadores de control y visualización
        self.pose_pub = self.create_publisher(PoseArray, '/arboles_detectados', 10)
        self.dist_lat_pub = self.create_publisher(
            Float32, '/lidar/distancia_tronco_lateral', 10
        )
        self.marker_pub = self.create_publisher(MarkerArray, '/tree_markers', 10)

        # Renderizador asíncrono de Markers 3D en Gazebo Fortress
        self.gz_renderer = GazeboMarkerRenderer(enabled=usar_gz_markers)

        self.get_logger().info(
            f"sim_cf_deteccion_troncos iniciado | Disparo lateral: "
            f"[{np.rad2deg(self.fov_disparo_min):.0f}°, {np.rad2deg(self.fov_disparo_max):.0f}°] | "
            f"Guiado amplio: [{np.rad2deg(self.fov_busqueda_min):.0f}°, {np.rad2deg(self.fov_busqueda_max):.0f}°] | "
            f"Markers Gazebo 3D + RViz2 ACTIVOS"
        )

    def world_pose_callback(self, msg: TFMessage):
        """
        Recibe la pose global exacta de Gazebo (`dynamic_pose/info`) para que los
        Markers 3D en Gazebo se dibujen exactamente sobre los árboles incluso
        cuando las ruedas patinan sobre el terreno irregular.
        """
        for tf_item in msg.transforms:
            if 'robot_desmalezador' in tf_item.child_frame_id and (
                tf_item.child_frame_id == 'robot_desmalezador'
                or tf_item.child_frame_id.endswith('/robot_desmalezador')
            ):
                t = tf_item.transform.translation
                q = tf_item.transform.rotation
                siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
                cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
                self.robot_world_x = float(t.x)
                self.robot_world_y = float(t.y)
                self.robot_world_yaw = math.atan2(siny_cosp, cosy_cosp)
                self._tiene_pose_mundo_exacta = True
                break

    def odom_callback(self, msg: Odometry):
        """
        Respaldo de pose mediante odometría relativa del plugin DiffDrive cuando
        no está activo `dynamic_pose/info` (ej. pruebas unitarias sin Gazebo).
        """
        if self._tiene_pose_mundo_exacta:
            return
        ox = float(msg.pose.pose.position.x)
        oy = float(msg.pose.pose.position.y)
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        oyaw = math.atan2(siny_cosp, cosy_cosp)

        cos_s = math.cos(self.spawn_yaw)
        sin_s = math.sin(self.spawn_yaw)
        self.robot_world_x = self.spawn_x + (ox * cos_s - oy * sin_s)
        self.robot_world_y = self.spawn_y + (ox * sin_s + oy * cos_s)
        self.robot_world_yaw = self.spawn_yaw + oyaw

    def _robot_a_mundo(self, x_robot: float, y_robot: float):
        """
        Transforma un punto `(x_robot, y_robot)` medido desde el LiDAR (esquina
        superior izquierda) a coordenadas globales `(X_world, Y_world)` de Gazebo.
        """
        x_base = self.LIDAR_OFFSET_X + x_robot
        y_base = self.LIDAR_OFFSET_Y + y_robot
        c = math.cos(self.robot_world_yaw)
        s = math.sin(self.robot_world_yaw)
        xw = self.robot_world_x + (x_base * c - y_base * s)
        yw = self.robot_world_y + (x_base * s + y_base * c)
        return xw, yw

    def scan_callback(self, msg: LaserScan):
        angles = msg.angle_min + np.arange(len(msg.ranges)) * msg.angle_increment
        ranges = np.array(msg.ranges, dtype=float)

        info = detectar_troncos_en_scan(
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
        )

        # 1. Publicar distancia perpendicular al árbol próximo (zona de guiado amplio)
        msg_dist = Float32()
        if info['distancia_lateral_guia'] is not None:
            msg_dist.data = float(info['distancia_lateral_guia'])
        else:
            msg_dist.data = float('nan')
        self.dist_lat_pub.publish(msg_dist)

        # 2. Evaluar detección del árbol lateral (zona de disparo -90° +- 10°)
        tronco_disparo = info['tronco_en_disparo']
        if tronco_disparo is not None:
            self.consecutive_detections += 1
        else:
            self.consecutive_detections = 0

        confirmado_lateral = (
            tronco_disparo is not None
            and self.consecutive_detections >= self.required_detections
        )

        # 3. Publicar `/arboles_detectados` hacia `sim_rutina_navegacion`
        pose_array_msg = PoseArray()
        pose_array_msg.header.frame_id = 'lidar_link'
        pose_array_msg.header.stamp = self.get_clock().now().to_msg()

        if confirmado_lateral:
            x_robot, y_robot, r_tronco, _, _ = tronco_disparo
            pose = Pose()
            pose.position.x = float(x_robot)
            pose.position.y = float(y_robot)
            pose.position.z = float(r_tronco)
            pose_array_msg.poses.append(pose)

        self.pose_pub.publish(pose_array_msg)

        # 4. Publicar Markers para RViz2 (`/tree_markers`) y Gazebo 3D (`/marker`)
        self._publicar_markers_rviz(info['tronco_proximo'], tronco_disparo, confirmado_lateral)
        self._publicar_markers_gazebo(info['tronco_proximo'], tronco_disparo, confirmado_lateral)

    # =========================================================================
    # Construcción de Markers para RViz2 (`/tree_markers`, frame `lidar_link`)
    # =========================================================================
    def _publicar_markers_rviz(self, tronco_proximo, tronco_disparo, confirmado_lateral: bool):
        stamp = self.get_clock().now().to_msg()
        frame_id = 'lidar_link'
        marker_array = MarkerArray()

        clear_marker = Marker()
        clear_marker.header.frame_id = frame_id
        clear_marker.header.stamp = stamp
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)

        # A) Conos visuales de las dos zonas de detección del LiDAR
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

        # B) Marker del ÁRBOL PRÓXIMO (Zona de búsqueda/guiado amplio: CIAN / AMARILLO)
        if tronco_proximo is not None:
            _, y_r, r_t, cx_l, cy_l = tronco_proximo
            # Cilindro alrededor del árbol próximo
            m_prox = Marker()
            m_prox.header.frame_id = frame_id
            m_prox.header.stamp = stamp
            m_prox.ns = 'arbol_proximo_guia'
            m_prox.id = 1
            m_prox.type = Marker.CYLINDER
            m_prox.action = Marker.ADD
            m_prox.pose.position.x = float(cx_l)
            m_prox.pose.position.y = float(cy_l)
            m_prox.pose.position.z = 0.8
            m_prox.scale.x = float(r_t * 2.4)
            m_prox.scale.y = float(r_t * 2.4)
            m_prox.scale.z = 2.1
            m_prox.color.r, m_prox.color.g, m_prox.color.b, m_prox.color.a = (0.0, 0.85, 1.0, 0.65)
            marker_array.markers.append(m_prox)

            # Línea guía desde el LiDAR hasta el árbol próximo
            m_line = Marker()
            m_line.header.frame_id = frame_id
            m_line.header.stamp = stamp
            m_line.ns = 'arbol_proximo_guia'
            m_line.id = 2
            m_line.type = Marker.LINE_STRIP
            m_line.action = Marker.ADD
            m_line.scale.x = 0.03
            m_line.color.r, m_line.color.g, m_line.color.b, m_line.color.a = (1.0, 0.9, 0.0, 0.9)
            p0 = Point(x=0.0, y=0.0, z=0.0)
            p1 = Point(x=float(cx_l), y=float(cy_l), z=0.0)
            m_line.points = [p0, p1]
            marker_array.markers.append(m_line)

            # Texto con la distancia perpendicular medida hacia el árbol próximo
            m_txt = Marker()
            m_txt.header.frame_id = frame_id
            m_txt.header.stamp = stamp
            m_txt.ns = 'arbol_proximo_guia'
            m_txt.id = 3
            m_txt.type = Marker.TEXT_VIEW_FACING
            m_txt.action = Marker.ADD
            m_txt.pose.position.x = float(cx_l)
            m_txt.pose.position.y = float(cy_l)
            m_txt.pose.position.z = 2.1
            m_txt.scale.z = 0.22
            m_txt.color.r, m_txt.color.g, m_txt.color.b, m_txt.color.a = (0.0, 0.95, 1.0, 1.0)
            m_txt.text = f"ARBOL PROXIMO (Y={y_r:.2f}m | R={r_t*100:.1f}cm)"
            marker_array.markers.append(m_txt)

        # C) Marker del ÁRBOL LATERAL (Zona de disparo [-100°, -80°]: VERDE BRILLANTE)
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
            m_lat.pose.position.z = 0.9
            m_lat.scale.x = float(r_t * 2.8)
            m_lat.scale.y = float(r_t * 2.8)
            m_lat.scale.z = 2.3
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
            m_lat_txt.pose.position.z = 2.45
            m_lat_txt.scale.z = 0.25
            m_lat_txt.color.r, m_lat_txt.color.g, m_lat_txt.color.b, m_lat_txt.color.a = (0.1, 1.0, 0.2, 1.0)
            m_lat_txt.text = "ARBOL LATERAL CONFIRMADO!" if confirmado_lateral else "ARBOL LATERAL (verificando)"
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
        m.color.r, m.color.g, m.color.b, m.color.a = (float(r), float(g), float(b), float(a))

        pts = [Point(x=0.0, y=0.0, z=0.0)]
        for ang in np.linspace(ang_min, ang_max, 12):
            pts.append(Point(x=float(radio * math.cos(ang)), y=float(radio * math.sin(ang)), z=0.0))
        pts.append(Point(x=0.0, y=0.0, z=0.0))
        m.points = pts
        return m

    # =========================================================================
    # Construcción de Markers 3D nativos para la ventana de Gazebo Fortress
    # =========================================================================
    def _publicar_markers_gazebo(self, tronco_proximo, tronco_disparo, confirmado_lateral: bool):
        activos = {}
        deletes = {}
        lidar_xw, lidar_yw = self._robot_a_mundo(0.0, 0.0)

        # 1. ÁRBOL PRÓXIMO en Gazebo (Cilindro Cian id=1 + Línea Guía Amarilla id=2)
        if tronco_proximo is not None:
            x_r, y_r, r_t, _, _ = tronco_proximo
            tx_w, ty_w = self._robot_a_mundo(x_r, y_r)
            diam = max(r_t * 2.6, 0.26)
            activos[('arbol_proximo', 1)] = (
                f'ns: "arbol_proximo" id: 1 action: ADD_MODIFY type: CYLINDER visibility: GUI '
                f'pose {{ position {{ x: {tx_w:.3f} y: {ty_w:.3f} z: 1.05 }} }} '
                f'scale {{ x: {diam:.3f} y: {diam:.3f} z: 2.15 }} '
                f'material {{ ambient {{ r: 0.0 g: 0.85 b: 1.0 a: 0.65 }} '
                f'diffuse {{ r: 0.0 g: 0.85 b: 1.0 a: 0.65 }} '
                f'emissive {{ r: 0.0 g: 0.55 b: 0.75 a: 0.65 }} }}'
            )
            activos[('arbol_proximo', 2)] = (
                f'ns: "arbol_proximo" id: 2 action: ADD_MODIFY type: LINE_STRIP visibility: GUI '
                f'point {{ x: {lidar_xw:.3f} y: {lidar_yw:.3f} z: {self.LIDAR_OFFSET_Z:.3f} }} '
                f'point {{ x: {tx_w:.3f} y: {ty_w:.3f} z: {self.LIDAR_OFFSET_Z:.3f} }} '
                f'material {{ ambient {{ r: 1.0 g: 0.9 b: 0.0 a: 1.0 }} '
                f'diffuse {{ r: 1.0 g: 0.9 b: 0.0 a: 1.0 }} '
                f'emissive {{ r: 1.0 g: 0.9 b: 0.0 a: 1.0 }} }}'
            )
            self._gz_prox_activo = True
        elif self._gz_prox_activo:
            deletes[('arbol_proximo', 1)] = 'ns: "arbol_proximo" id: 1 action: DELETE_MARKER'
            deletes[('arbol_proximo', 2)] = 'ns: "arbol_proximo" id: 2 action: DELETE_MARKER'
            self._gz_prox_activo = False

        # 2. ÁRBOL LATERAL en Gazebo (Cilindro Verde Brillante id=10 + Línea id=11 al entrar en [-100°, -80°])
        if tronco_disparo is not None:
            x_r, y_r, r_t, _, _ = tronco_disparo
            lx_w, ly_w = self._robot_a_mundo(x_r, y_r)
            diam_lat = max(r_t * 3.2, 0.32)
            if confirmado_lateral:
                cr, cg, cb = (0.0, 1.0, 0.1)
            else:
                cr, cg, cb = (1.0, 0.55, 0.0)
            activos[('arbol_lateral', 10)] = (
                f'ns: "arbol_lateral" id: 10 action: ADD_MODIFY type: CYLINDER visibility: GUI '
                f'pose {{ position {{ x: {lx_w:.3f} y: {ly_w:.3f} z: 1.15 }} }} '
                f'scale {{ x: {diam_lat:.3f} y: {diam_lat:.3f} z: 2.40 }} '
                f'material {{ ambient {{ r: {cr} g: {cg} b: {cb} a: 0.85 }} '
                f'diffuse {{ r: {cr} g: {cg} b: {cb} a: 0.85 }} '
                f'emissive {{ r: {cr} g: {cg} b: {cb} a: 0.85 }} }}'
            )
            activos[('arbol_lateral', 11)] = (
                f'ns: "arbol_lateral" id: 11 action: ADD_MODIFY type: LINE_STRIP visibility: GUI '
                f'point {{ x: {lidar_xw:.3f} y: {lidar_yw:.3f} z: {self.LIDAR_OFFSET_Z + 0.03:.3f} }} '
                f'point {{ x: {lx_w:.3f} y: {ly_w:.3f} z: {self.LIDAR_OFFSET_Z + 0.03:.3f} }} '
                f'material {{ ambient {{ r: {cr} g: {cg} b: {cb} a: 1.0 }} '
                f'diffuse {{ r: {cr} g: {cg} b: {cb} a: 1.0 }} '
                f'emissive {{ r: {cr} g: {cg} b: {cb} a: 1.0 }} }}'
            )
            self._gz_lat_activo = True
        elif self._gz_lat_activo:
            deletes[('arbol_lateral', 10)] = 'ns: "arbol_lateral" id: 10 action: DELETE_MARKER'
            deletes[('arbol_lateral', 11)] = 'ns: "arbol_lateral" id: 11 action: DELETE_MARKER'
            self._gz_lat_activo = False

        if activos or deletes:
            self.gz_renderer.actualizar_marcadores(activos, deletes)

    def destroy_node(self):
        self.gz_renderer.cerrar()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SimPercepcionTroncos()
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

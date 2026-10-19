import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
from sklearn.cluster import DBSCAN
import psutil
import os

def circfit_HyperSVD(XY):
    XY = np.asarray(XY)
    if len(XY) < 4: raise ValueError("Se requieren al menos 4 puntos para el ajuste SVD")
    centroid = np.mean(XY, axis=0)
    X = XY[:, 0] - centroid[0]
    Y = XY[:, 1] - centroid[1]
    Z = X**2 + Y**2
    ZXY1 = np.column_stack((Z, X, Y, np.ones(len(Z))))
    U, S, Vt = np.linalg.svd(ZXY1, full_matrices=False)
    V = Vt.T
    
    if (S[3] / S[0] < 1e-12): A = V[:, 3]
    else:
        R_mean = np.mean(ZXY1, axis=0)
        N = np.array([[8*R_mean[0], 4*R_mean[1], 4*R_mean[2], 2], [4*R_mean[1], 1, 0, 0], [4*R_mean[2], 0, 1, 0], [2, 0, 0, 0]], dtype=float)
        W = V @ np.diag(S) @ V.T
        N_inv = np.linalg.pinv(N)
        D, E = np.linalg.eig(W @ N_inv @ W)
        E = np.real(E)
        D = np.real(D)
        ID = np.argsort(D)
        Astar = E[:, ID[1]]
        A, _, _, _ = np.linalg.lstsq(W, Astar, rcond=None)
        
    if np.isclose(A[0], 0): 
        raise ValueError("A[0] es muy cercano a cero")

    rad_inner = A[1]**2 + A[2]**2 - 4*A[0]*A[3]

    if rad_inner <= 0: 
        raise ValueError("Raíz cuadrada negativa")
    
    center_x = -(A[1]) / (2 * A[0]) + centroid[0]
    center_y = -(A[2]) / (2 * A[0]) + centroid[1]
    radius = np.sqrt(rad_inner) / (2 * np.abs(A[0]))

    return center_x, center_y, radius


class LidarTreeDetector(Node):
    def __init__(self):
        super().__init__('lidar_tree_detector')
        self.subscription = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/tree_markers', 10)

        # CORRECCIÓN DE FALSOS POSITIVOS: Reducir FOV a +- 10 grados
        self.fov_min = np.deg2rad(-10.0) 
        self.fov_max = np.deg2rad(10.0)
        self.roi_min_dist = 0.1
        self.roi_max_dist = 2.5  
        self.required_detections = 6      
        self.consecutive_detections = 0    

        # Máquina de estados
        self.estado_actual = 'AVANZANDO'
        self.arboles_rociados = 0
        self.meta_arboles = 4
        self.velocidad_avance = 0.2 # m/s
        
        # Tiempos de la rutina
        self.tiempo_detencion = 5.0  
        self.tiempo_enfriamiento = 3.0  
        self.timeout_sin_arbol = 15.0   

        self.tiempo_estado = self.get_clock().now()
        self.tiempo_ultimo_arbol = self.get_clock().now()

        self.control_timer = self.create_timer(0.05, self.control_loop)

    def control_loop(self):
        msg = Twist()
        now = self.get_clock().now()
        tiempo_transcurrido = (now - self.tiempo_estado).nanoseconds / 1e9

        if self.estado_actual == 'AVANZANDO':
            msg.linear.x = self.velocidad_avance
            # Timeout de 15s
            tiempo_sin_ver_arbol = (now - self.tiempo_ultimo_arbol).nanoseconds / 1e9
            if tiempo_sin_ver_arbol >= self.timeout_sin_arbol:
                self.estado_actual = 'FINALIZADO'
                self.get_logger().warn(f'{self.timeout_sin_arbol} segundos sin detección. Rutina abortada.')
            
        elif self.estado_actual == 'DETENIDO':
            msg.linear.x = 0.0
            if tiempo_transcurrido >= self.tiempo_detencion:
                # Evitar nueva detección del mismo árbol
                self.estado_actual = 'ENFRIAMIENTO'
                self.tiempo_estado = now
                self.consecutive_detections = 0
                self.get_logger().info(f'Reanudando. Árboles procesados: {self.arboles_rociados}')
                
        elif self.estado_actual == 'ENFRIAMIENTO':
            msg.linear.x = self.velocidad_avance
            if tiempo_transcurrido >= self.tiempo_enfriamiento:
                self.estado_actual = 'AVANZANDO'
                self.tiempo_ultimo_arbol = now # Resetear timeout al volver a escanear
                
        elif self.estado_actual == 'FINALIZADO':
            msg.linear.x = 0.0

        self.cmd_pub.publish(msg)

    def scan_callback(self, msg):
        angles = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))
        ranges = np.array(msg.ranges)

        valid_ranges = (ranges > msg.range_min) & (ranges < msg.range_max) & ~np.isinf(ranges) & ~np.isnan(ranges)
        valid_angles = (angles >= self.fov_min) & (angles <= self.fov_max)
        valid = valid_ranges & valid_angles
        
        X = ranges[valid] * np.cos(angles[valid])
        Y = ranges[valid] * np.sin(angles[valid])
        XY = np.column_stack((X, Y))
        
        marker_array = MarkerArray()
        clear_marker = Marker()
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)
        
        marker_id = 1
        valid_tree_in_this_scan = False

        if self.estado_actual == 'AVANZANDO' and len(XY) >= 4:
            clustering = DBSCAN(eps=0.15, min_samples=4).fit(XY) 
            labels = clustering.labels_ 
            
            for label in set(labels):
                if label == -1: continue 

                cluster_points = XY[labels == label]
                if 4 <= len(cluster_points) < 80:
                    try:
                        cx, cy, r = circfit_HyperSVD(cluster_points)
                        # filtro de tamaño
                        if not (0.07 < r < 0.14): 
                            continue
                        
                        dist_to_center = np.hypot(cx, cy)

                        if not (self.roi_min_dist < dist_to_center < self.roi_max_dist): 
                            continue
                        mean_pts_dist = np.mean(np.linalg.norm(cluster_points, axis=1))

                        if dist_to_center < mean_pts_dist: 
                            continue
                            
                        valid_tree_in_this_scan = True
                        marker = self.create_marker(cx, cy, r, marker_id, msg.header.frame_id)
                        marker_array.markers.append(marker)
                        marker_id += 1

                    except: continue 

        if valid_tree_in_this_scan:
            self.consecutive_detections += 1
        else:
            self.consecutive_detections = 0 

        # Transición lógica
        if self.estado_actual == 'AVANZANDO' and self.consecutive_detections >= self.required_detections:
            self.arboles_rociados += 1
            self.tiempo_estado = self.get_clock().now()
            self.tiempo_ultimo_arbol = self.get_clock().now() 

            if self.arboles_rociados >= self.meta_arboles:
                self.estado_actual = 'FINALIZADO'
                self.get_logger().info(f'{self.meta_arboles} árboles detectados. Rutina finalizada')
            else:
                self.estado_actual = 'DETENIDO'
                self.get_logger().info(f'Tronco {self.arboles_rociados} detectado. Detención por {self.tiempo_detencion}s...') # IMPLEMENTAR MAPIR/BRAZO

        self.marker_pub.publish(marker_array)

    # Visualizacion de detección para RViz
    def create_marker(self, x, y, r, m_id, frame_id):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "trees"
        marker.id = m_id
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        marker.pose.position.x = float(x)
        marker.pose.position.y = float(y)
        marker.pose.position.z = 0.5 
        marker.scale.x = float(r * 2) 
        marker.scale.y = float(r * 2)
        marker.scale.z = 1.0 
        marker.color.r = 0.0
        marker.color.g = 1.0 if self.consecutive_detections >= self.required_detections else 0.5
        marker.color.b = 0.0
        marker.color.a = 0.8
        marker.lifetime.sec = 0
        marker.lifetime.nanosec = 300000000 
        return marker

def main(args=None):
    rclpy.init(args=args)
    node = LidarTreeDetector()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()

if __name__ == '__main__':
    main()
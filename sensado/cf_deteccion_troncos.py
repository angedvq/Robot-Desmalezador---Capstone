import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
from sklearn.cluster import DBSCAN
import psutil
import os

def circfit_HyperSVD(XY):
    XY = np.asarray(XY)
    if len(XY) < 4:
        raise ValueError("Se requieren al menos 4 puntos para el ajuste SVD")
        
    centroid = np.mean(XY, axis=0)
    X = XY[:, 0] - centroid[0]
    Y = XY[:, 1] - centroid[1]
    Z = X**2 + Y**2
    ZXY1 = np.column_stack((Z, X, Y, np.ones(len(Z))))
    U, S, Vt = np.linalg.svd(ZXY1, full_matrices=False)
    V = Vt.T
    
    if (S[3] / S[0] < 1e-12):
        A = V[:, 3]
    else:
        R_mean = np.mean(ZXY1, axis=0)
        N = np.array([
            [8*R_mean[0], 4*R_mean[1], 4*R_mean[2], 2],
            [4*R_mean[1], 1, 0, 0],
            [4*R_mean[2], 0, 1, 0],
            [2, 0, 0, 0]
        ], dtype=float)
        
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
        self.marker_pub = self.create_publisher(MarkerArray, '/tree_markers', 10)

        # Angulos de fov de deteccion
        self.fov_min = np.deg2rad(-30.0) 
        self.fov_max = np.deg2rad(30.0)

        # Distancia/profundidad minima y maxima para la deteccion
        self.roi_min_dist = 0.1
        self.roi_max_dist = 2.5  

        # Frames consecutivos requeridos para dar un positivo
        self.required_detections = 6      
        self.consecutive_detections = 0    

        # Monitorización de recursos 
        self.process = psutil.Process(os.getpid())
        self.process.cpu_percent() 
        self.cpu_records = []
        self.ram_records = []
        self.threads_records = []
        self.total_system_cores = psutil.cpu_count(logical=True)
        self.resource_timer = self.create_timer(0.5, self.track_resources)

    def track_resources(self):
        ram_mb = self.process.memory_info().rss / (1024 * 1024)
        cpu_perc = self.process.cpu_percent() 
        num_threads = self.process.num_threads()
        self.ram_records.append(ram_mb)
        self.cpu_records.append(cpu_perc)
        self.threads_records.append(num_threads)

    def print_resource_summary(self):
        if not self.cpu_records or not self.ram_records:
            return

        avg_cpu_perc = sum(self.cpu_records) / len(self.cpu_records)
        peak_cpu_perc = max(self.cpu_records)
        avg_cores_used = avg_cpu_perc / 100.0
        peak_cores_used = peak_cpu_perc / 100.0
        avg_ram = sum(self.ram_records) / len(self.ram_records)
        peak_ram = max(self.ram_records)
        peak_threads = max(self.threads_records)

        print("\n" + "="*60)
        print(" RESUMEN DE CONSUMO")
        print("="*60)
        print(f" CPU Total:   {self.total_system_cores} núcleos disponibles")
        print(f" Hilos Activos:    {peak_threads} hilos (Peak en este nodo)")
        print("-" * 60)
        print(f" NÚCLEOS USADOS:   {avg_cores_used:.2f} núcleos (Promedio)")
        print(f" NÚCLEOS USADOS:   {peak_cores_used:.2f} núcleos (Peak)")
        print(f" Carga bruta CPU:  {avg_cpu_perc:.2f} %")
        print("-" * 60)
        print(f" RAM (Promedio):   {avg_ram:.2f} MB")
        print(f" RAM (Peak):       {peak_ram:.2f} MB")
        print("="*60 + "\n")

    def scan_callback(self, msg):
        angles = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))
        ranges = np.array(msg.ranges)

        valid_ranges = (ranges > msg.range_min) & (ranges < msg.range_max) & ~np.isinf(ranges) & ~np.isnan(ranges)
        valid_angles = (angles >= self.fov_min) & (angles <= self.fov_max)
        
        valid = valid_ranges & valid_angles
        angles = angles[valid]
        ranges = ranges[valid]

        X = ranges * np.cos(angles)
        Y = ranges * np.sin(angles)
        XY = np.column_stack((X, Y))
        
        marker_array = MarkerArray()
        clear_marker = Marker()
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)
        
        marker_id = 1
        valid_tree_in_this_scan = False

        # Si hay puntos suficientes, procesar DBSCAN y ajuste
        if len(XY) >= 4:
            clustering = DBSCAN(eps=0.15, min_samples=4).fit(XY) 
            labels = clustering.labels_ 
            
            for label in set(labels):
                if label == -1: 
                    continue 

                cluster_points = XY[labels == label]
                
                if 4 <= len(cluster_points) < 80:
                    try:
                        cx, cy, r = circfit_HyperSVD(cluster_points)
                        
                        # Filtro de tamaño del tronco (mts)
                        if not (0.03 < r < 0.05):
                            continue
                            
                        # Validación de ROI
                        dist_to_center = np.hypot(cx, cy)
                        if not (self.roi_min_dist < dist_to_center < self.roi_max_dist):
                            continue
                            
                        # Validación de Centroide
                        mean_pts_dist = np.mean(np.linalg.norm(cluster_points, axis=1))
                        if dist_to_center < mean_pts_dist:
                            continue
                            
                        # Confirmación del clúster actual
                        valid_tree_in_this_scan = True
                        marker = self.create_marker(cx, cy, r, marker_id, msg.header.frame_id)
                        marker_array.markers.append(marker)
                        marker_id += 1

                    except ValueError:
                        continue 
                    except Exception as e:
                        self.get_logger().warn(f"Error: {e}")

        # Actualizacion frames consecutivos
        if valid_tree_in_this_scan:
            self.consecutive_detections += 1
        else:
            self.consecutive_detections = 0 

        # Comando de parada, solo si se cumplen los criterios anteriores
        stable_tree_detected = (self.consecutive_detections >= self.required_detections)

        print(f"STOP PLATAFORMA: {stable_tree_detected}")
        
        self.marker_pub.publish(marker_array)

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
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.print_resource_summary()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
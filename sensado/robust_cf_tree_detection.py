import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.optimize import least_squares
import psutil
import os

def circfit_HyperSVD(XY):
    """Ajuste algebraico inicial."""
    XY = np.asarray(XY)
    if len(XY) < 4:
        raise ValueError("Se requieren al menos 4 puntos.")
        
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
        
        ID = np.argsort(np.real(D))
        Astar = np.real(E)[:, ID[1]]
        A, _, _, _ = np.linalg.lstsq(W, Astar, rcond=None)
        
    if np.isclose(A[0], 0):
        raise ValueError("Puntos colineales.")
        
    rad_inner = A[1]**2 + A[2]**2 - 4*A[0]*A[3]
    if rad_inner <= 0:
        raise ValueError("Geometría inválida.")
        
    cx = -(A[1]) / (2 * A[0]) + centroid[0]
    cy = -(A[2]) / (2 * A[0]) + centroid[1]
    r = np.sqrt(rad_inner) / (2 * np.abs(A[0]))
    
    return np.array([cx, cy, r])

def circfit_TR(XY, ParIni):
    """
    Ajuste por Trust Region (Equivalente exacto a circfit_TR.m).
    Minimiza la distancia geométrica iterativamente.
    """
    def residual(vars, x, y):
        cx, cy, R = vars
        # Distancia del punto al borde del círculo
        return np.sqrt((x - cx)**2 + (y - cy)**2) - R
    
    # Optimizador Trust Region Reflective nativo de SciPy
    res = least_squares(
        residual, 
        ParIni, 
        args=(XY[:, 0], XY[:, 1]), 
        method='trf', 
        loss='linear'
    )
    return res.x # Retorna [cx, cy, r] refinado

def robust_circle_fit(XY):
    """
    Adaptación integrada de robustCircleFit.m y RobCircFit.m
    """
    k = len(XY)
    if k < 4:
        raise ValueError("Insuficientes puntos para RANSAC.")
        
    h0 = 4 # Mínimo de puntos para formular un HyperSVD inicial
    h = max(h0, int(0.75 * k)) # Cantidad de inliers (descartamos el 25% más lejano)
    
    pr = 0.99
    eps = 0.25
    
    # Cálculo estadístico de iteraciones necesarias
    denom = np.log(1 - (1 - eps)**h0)
    if denom == 0:
        In = 15
    else:
        In = int(np.floor(np.log(1 - pr) / denom))
        
    In = max(5, min(In, 30)) # Limitamos entre 5 y 30 iteraciones por seguridad de CPU
    
    best_error = float('inf')
    best_par = None
    
    for _ in range(In):
        # 1. Muestreo aleatorio
        idx = np.random.choice(k, h0, replace=False)
        sample = XY[idx]
        
        # 2. Modelo inicial con SVD
        try:
            Par0 = circfit_HyperSVD(sample)
        except ValueError:
            continue
            
        # 3. Filtrado (Evaluar todos los puntos del clúster contra Par0)
        dists = np.abs(np.linalg.norm(XY - Par0[:2], axis=1) - Par0[2])
        idx_sorted = np.argsort(dists)
        inliers = XY[idx_sorted[:h]] # Tomamos los 'h' con menor error
        
        # 4. Refinamiento geométrico sobre los inliers limpios
        try:
            Par_ref = circfit_TR(inliers, Par0)
        except ValueError:
            continue
            
        # 5. Cálculo del error final de esta hipótesis
        dists_ref = np.abs(np.linalg.norm(inliers - Par_ref[:2], axis=1) - Par_ref[2])
        e_i = np.sum(dists_ref**2)
        
        if e_i < best_error:
            best_error = e_i
            best_par = Par_ref
            
    if best_par is None:
        raise ValueError("RANSAC falló en encontrar un modelo válido.")
        
    return best_par[0], best_par[1], best_par[2]

class LidarTreeDetector(Node):
    def __init__(self):
        super().__init__('lidar_tree_detector')
        self.subscription = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/tree_markers', 10)
        
        self.fov_min = np.deg2rad(-60.0) 
        self.fov_max = np.deg2rad(60.0)

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
        print(" RESUMEN DE RENDIMIENTO (VERSIÓN ROBUSTA RANSAC)")
        print("="*60)
        print(f" CPU Host Total:   {self.total_system_cores} núcleos")
        print(f" Hilos Activos:    {peak_threads}")
        print("-" * 60)
        print(f" NÚCLEOS USADOS:   {avg_cores_used:.2f} (Promedio) | {peak_cores_used:.2f} (Peak)")
        print(f" Carga bruta CPU:  {avg_cpu_perc:.2f} %")
        print("-" * 60)
        print(f" RAM (Promedio):   {avg_ram:.2f} MB | Peak: {peak_ram:.2f} MB")
        print("="*60 + "\n")

    def scan_callback(self, msg):
        angles = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))
        ranges = np.array(msg.ranges)

        valid_ranges = (ranges > msg.range_min) & (ranges < msg.range_max) & ~np.isinf(ranges) & ~np.isnan(ranges)
        valid_angles = (angles >= self.fov_min) & (angles <= self.fov_max)
        
        valid = valid_ranges & valid_angles
        angles = angles[valid]
        ranges = ranges[valid]

        XY = np.column_stack((ranges * np.cos(angles), ranges * np.sin(angles)))

        if len(XY) < 4:
            return
            
        clustering = DBSCAN(eps=0.15, min_samples=4).fit(XY) 
        labels = clustering.labels_ 
        
        marker_array = MarkerArray()
        clear_marker = Marker()
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)
        
        marker_id = 1
        tree_detected = False
        
        for label in set(labels):
            if label == -1: 
                continue 

            cluster_points = XY[labels == label]
            
            if 4 <= len(cluster_points) < 80:
                try:
                    # IMPLEMENTACIÓN NUEVA
                    cx, cy, r = robust_circle_fit(cluster_points)
                    
                    if 0.3 < r < 0.5: # Rango para la botella de prueba
                        tree_detected = True
                        marker = self.create_marker(cx, cy, r, marker_id, msg.header.frame_id)
                        marker_array.markers.append(marker)
                        marker_id += 1

                except ValueError:
                    continue 
                except Exception as e:
                    self.get_logger().warn(f"Error en ajuste robusto: {e}")

        if len(marker_array.markers) > 1: 
            print(f"Tree Detected: {tree_detected}")
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
        marker.pose.position.z = 0.2 
        marker.scale.x = float(r * 2) 
        marker.scale.y = float(r * 2)
        marker.scale.z = 0.4 
        marker.color.r = 0.0
        marker.color.g = 1.0
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
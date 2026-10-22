import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseArray, Pose
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
from sklearn.cluster import DBSCAN

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

class PercepcionTroncos(Node):
    def __init__(self):
        super().__init__('percepcion_troncos')
        self.subscription = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        
        # Publicadores
        self.pose_pub = self.create_publisher(PoseArray, '/arboles_detectados', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/tree_markers', 10)
        self.gz_marker_pub = self.create_publisher(Marker, '/marker', 10)

        # Configuración
        self.fov_min = np.deg2rad(-10.0) 
        self.fov_max = np.deg2rad(10.0)
        self.roi_min_dist = 0.1
        self.roi_max_dist = 2.5  
        self.required_detections = 6      
        self.consecutive_detections = 0    

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
        self.gz_marker_pub.publish(clear_marker)
        
        valid_tree_in_this_scan = False
        detected_x, detected_y = 0.0, 0.0

        if len(XY) >= 4:
            clustering = DBSCAN(eps=0.15, min_samples=4).fit(XY) 
            for label in set(clustering.labels_):
                if label == -1: 
                    continue 

                cluster_points = XY[clustering.labels_ == label]
                if 4 <= len(cluster_points) < 80:
                    try:
                        cx, cy, r = circfit_HyperSVD(cluster_points)

                        if not (0.07 < r < 0.14): 
                            continue
                        
                        dist_to_center = np.hypot(cx, cy)

                        if not (self.roi_min_dist < dist_to_center < self.roi_max_dist): 
                            continue

                        mean_pts_dist = np.mean(np.linalg.norm(cluster_points, axis=1))

                        if dist_to_center < mean_pts_dist: 
                            continue
                            
                        valid_tree_in_this_scan = True
                        detected_x, detected_y = cx, cy
                        
                        # Publicar marcadores visuales
                        marker = self.create_marker(cx, cy, r, 1, msg.header.frame_id)
                        marker_array.markers.append(marker)
                        gz_marker = self.create_gazebo_marker(cx, cy, r, 1, msg.header.frame_id)
                        self.gz_marker_pub.publish(gz_marker)

                    except: continue 

        if valid_tree_in_this_scan:
            self.consecutive_detections += 1
        else:
            self.consecutive_detections = 0 

        # Comunicación al nodo de control 
        pose_array_msg = PoseArray()
        pose_array_msg.header.frame_id = msg.header.frame_id
        pose_array_msg.header.stamp = self.get_clock().now().to_msg()

        if self.consecutive_detections >= self.required_detections:
            pose = Pose()
            pose.position.x = float(detected_x)
            pose.position.y = float(detected_y)
            pose.position.z = 0.0
            pose_array_msg.poses.append(pose)

        self.pose_pub.publish(pose_array_msg)
        self.marker_pub.publish(marker_array)

    def create_marker(self, x, y, r, m_id, frame_id):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns, marker.id, marker.type, marker.action = "trees", m_id, Marker.CYLINDER, Marker.ADD
        marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = float(x), float(y), 0.5 
        marker.scale.x, marker.scale.y, marker.scale.z = float(r * 2), float(r * 2), 1.0 
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = 0.0, 1.0 if self.consecutive_detections >= self.required_detections else 0.5, 0.0, 0.8
        return marker

    def create_gazebo_marker(self, x, y, r, m_id, frame_id):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns, marker.id, marker.type, marker.action = "gazebo_trees", m_id, Marker.CYLINDER, Marker.ADD
        marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = float(x), float(y), 1.0 
        marker.scale.x, marker.scale.y, marker.scale.z = float(r * 2.5), float(r * 2.5), 2.5 
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = 0.0, 1.0 if self.consecutive_detections >= self.required_detections else 0.5, 0.0, 0.6
        return marker

def main(args=None):
    rclpy.init(args=args)
    node = PercepcionTroncos()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()

if __name__ == '__main__':
    main()
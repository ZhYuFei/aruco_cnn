#!/usr/bin/env python3
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Pose, Quaternion
from sensor_msgs.msg import Image

class ArucoDetector(Node):
    def __init__(self):
        super().__init__('aruco_detector')
        
        # 参数声明
        self.declare_parameter('marker_length', 0.00088)  # 默认5厘米
        self.declare_parameter('camera_frame', 'camera_optical_frame')
        
        # 获取参数
        self.marker_length = self.get_parameter('marker_length').value
        self.camera_frame = self.get_parameter('camera_frame').value
        
        # 初始化ArUco检测器
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)
        
        # 相机参数（需要根据实际校准填写）
        self.camera_matrix = np.array([
            [500.0, 0.0, 320.0],
            [0.0, 500.0, 240.0],
            [0.0, 0.0, 1.0]
        ])
        self.dist_coeffs = np.zeros((4,1))
        
        # ROS2初始化
        self.bridge = CvBridge()
        self.marker_pub = self.create_publisher(MarkerArray, 'aruco_markers', 10)
        self.image_pub = self.create_publisher(Image, 'aruco_debug_image', 10)
        
        # 初始化摄像头
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # 定时器
        self.timer = self.create_timer(0.033, self.process_frame)  # ~30Hz
        self.get_logger().info(f"ArUco检测节点已启动，标记尺寸: {self.marker_length}m")

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("摄像头读取失败", once=True)
            return
        
        # 检测ArUco标记
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        
        # 创建MarkerArray消息
        marker_array = MarkerArray()
        current_time = self.get_clock().now().to_msg()
        
        if ids is not None:
            # 估计所有标记的姿态
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, self.marker_length,
                self.camera_matrix, self.dist_coeffs
            )
            
            for i in range(len(ids)):
                # 创建单个Marker
                marker = Marker()
                marker.header.frame_id = self.camera_frame
                marker.header.stamp = current_time
                marker.ns = "aruco_markers"
                marker.id = int(ids[i][0])  # 使用ArUco ID作为marker ID
                
                marker.type = Marker.CUBE
                marker.action = Marker.ADD
                
                # 设置位置
                marker.pose.position.x = tvecs[i][0][0]*500
                marker.pose.position.y = tvecs[i][0][1]*500
                marker.pose.position.z = tvecs[i][0][2]*500
                
                # 设置方向（旋转向量转四元数）
                rot_matrix = cv2.Rodrigues(rvecs[i])[0]
                q = Quaternion()
                q.w = np.sqrt(1.0 + rot_matrix[0,0] + rot_matrix[1,1] + rot_matrix[2,2]) / 2.0
                q.x = (rot_matrix[2,1] - rot_matrix[1,2]) / (4*q.w)
                q.y = (rot_matrix[0,2] - rot_matrix[2,0]) / (4*q.w)
                q.z = (rot_matrix[1,0] - rot_matrix[0,1]) / (4*q.w)
                marker.pose.orientation = q
                
                # 设置尺寸
                marker.scale.x = self.marker_length*500.0
                marker.scale.y = self.marker_length*500.0
                marker.scale.z = 0.1  # 厚度1cm
                
                # 设置颜色（按ID区分）
                marker.color.r = float(ids[i][0] % 10) / 10.0
                marker.color.g = float((ids[i][0] + 5) % 10) / 10.0
                marker.color.b = float((ids[i][0] + 2) % 10) / 10.0
                marker.color.a = 0.8  # 透明度
                
                # 设置生命周期（300ms）
                marker.lifetime.sec = 0
                marker.lifetime.nanosec = 100000000
                
                # 添加到数组
                marker_array.markers.append(marker)
                
                # 在图像上绘制结果（可选）
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, 
                                 rvecs[i], tvecs[i], self.marker_length/2)
        
        # 一次性发布所有标记
        self.marker_pub.publish(marker_array)
        
        # 调试图像发布（可选）
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(frame, "bgr8"))

    def __del__(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()

def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
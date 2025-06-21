#!/usr/bin/env python3
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Quaternion
from sensor_msgs.msg import Image as ImageMsg
import onnxruntime as ort
import torch
from torchvision import transforms
from PIL import Image

# 加载训练好的模型
class ArucoClassifier(torch.nn.Module):
    def __init__(self, num_classes=50):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            
            torch.nn.Conv2d(32, 64, 3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            
            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        
        self.avgpool = torch.nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(128 * 6 * 6, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

class YoloOnnxCnnDetector(Node):
    def __init__(self):
        super().__init__('yolo_onnx_cnn_detector')
        
        # 参数声明
        self.declare_parameter('object_size', 0.00088)  # 默认物体尺寸(m)
        self.declare_parameter('camera_frame', 'camera_optical_frame')
        self.declare_parameter('yolo_onnx_model', '/home/zhouyufei/aruco_cnn/test_model/yolov8n.onnx')
        self.declare_parameter('cnn_model', '/home/zhouyufei/aruco_cnn/test_model/cnn.pth')
        self.declare_parameter('conf_threshold', 0.5)
        self.declare_parameter('iou_threshold', 0.5)
        
        # 获取参数
        self.object_size = self.get_parameter('object_size').value
        self.camera_frame = self.get_parameter('camera_frame').value
        yolo_model_path = self.get_parameter('yolo_onnx_model').value
        cnn_model_path = self.get_parameter('cnn_model').value
        self.conf_threshold = self.get_parameter('conf_threshold').value
        self.iou_threshold = self.get_parameter('iou_threshold').value
        
        # 初始化ONNX Runtime的YOLO模型
        self.yolo_session = ort.InferenceSession(yolo_model_path)
        self.yolo_input_name = self.yolo_session.get_inputs()[0].name
        self.yolo_output_name = self.yolo_session.get_outputs()[0].name
        self.yolo_input_shape = self.yolo_session.get_inputs()[0].shape
        self.get_logger().info(f"加载ONNX YOLO模型: {yolo_model_path}")
        
        # 初始化CNN分类模型
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cnn_model = ArucoClassifier().to(device)
        self.cnn_model.load_state_dict(torch.load(cnn_model_path, map_location=device))
        self.cnn_model.eval()
        self.get_logger().info(f"加载CNN模型: {cnn_model_path}")
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # 类别映射（示例）
        self.class_names = {
            0: "class_0",
            1: "class_1",
            2: "class_2"
        }
        
        # 相机参数（需要校准）
        self.camera_matrix = np.array([
            [520.9732, 0.0,     327.0129],
            [0.0,      519.2718,246.9725],
            [0.0, 0.0, 1.0]
        ])
        self.dist_coeffs = np.zeros((4,1))
        
        # ROS2初始化
        self.bridge = CvBridge()
        self.marker_pub = self.create_publisher(MarkerArray, 'detected_objects', 10)
        self.image_pub = self.create_publisher(ImageMsg, 'detection_debug_image', 10)
        self.roi_pub = self.create_publisher(ImageMsg, 'detection_debug_roi', 10)
        
        # 初始化摄像头
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # 定时器
        self.timer = self.create_timer(0.033, self.process_frame)
        self.get_logger().info("YOLO(ONNX)+CNN检测节点已启动")

    def preprocess_yolo(self, frame):
        """预处理图像以适应YOLO ONNX模型输入"""
        # 调整大小并归一化
        input_img = cv2.resize(frame, (self.yolo_input_shape[3], self.yolo_input_shape[2]))
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)  # HWC -> CHW
        input_img = np.expand_dims(input_img, axis=0).astype(np.float32)
        return input_img

    def postprocess_yolo(self, outputs, frame_shape):
        """后处理YOLO输出"""
        # 获取原始输出
        predictions = np.squeeze(outputs[0]).T
        
        # 过滤低置信度检测
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.conf_threshold]
        scores = scores[scores > self.conf_threshold]
        
        if len(predictions) == 0:
            return [], [], []
        
        # 获取类别ID
        class_ids = np.argmax(predictions[:, 4:], axis=1)
        
        # 提取边界框 (cx, cy, w, h) -> (x1, y1, x2, y2)
        boxes = predictions[:, :4]
        input_shape = np.array([self.yolo_input_shape[3], self.yolo_input_shape[2], 
                              self.yolo_input_shape[3], self.yolo_input_shape[2]])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes[:, 0] *= frame_shape[1]  # x1
        boxes[:, 1] *= frame_shape[0]  # y1
        boxes[:, 2] *= frame_shape[1]  # x2
        boxes[:, 3] *= frame_shape[0]  # y2
        
        # 应用NMS
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), 
                                  self.conf_threshold, self.iou_threshold)
        
        if len(indices) > 0:
            boxes = boxes[indices.flatten()]
            scores = scores[indices.flatten()]
            class_ids = class_ids[indices.flatten()]
        
        return boxes, scores, class_ids

    def process_roi(self, frame, box):
        """提取目标ROI并进行CNN分类"""
        x1, y1, x2, y2 = map(int, box)
        crop_img = frame[y1:y2, x1:x2]
        
        if crop_img.size == 0:
            return None, 0.0, 0
        
        # 转换为PIL图像并预处理
        pil_img = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
        input_tensor = self.transform(pil_img).unsqueeze(0)
        
        # CNN分类
        with torch.no_grad():
            output = self.cnn_model(input_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            top_prob, top_class = torch.max(probs, 1)
        
        return crop_img, top_prob.item(), top_class.item()

    def estimate_pose(self, box, img_width, img_height):
        """估计物体姿态"""
        x1, y1, x2, y2 = box
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1
        
        fx = self.camera_matrix[0,0]
        fy = self.camera_matrix[1,1]
        cx = self.camera_matrix[0,2]
        cy = self.camera_matrix[1,2]
        
        z = (fx * self.object_size) / width
        x = (center_x - cx) * z / fx
        y = (center_y - cy) * z / fy
        
        return np.array([x, y, z]), np.zeros(3)

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("摄像头读取失败", once=True)
            return
        
        # YOLO ONNX推理
        input_img = self.preprocess_yolo(frame)
        outputs = self.yolo_session.run([self.yolo_output_name], {self.yolo_input_name: input_img})
        boxes, scores, class_ids = self.postprocess_yolo(outputs, frame.shape)
        
        # 创建MarkerArray消息
        marker_array = MarkerArray()
        current_time = self.get_clock().now().to_msg()
        debug_frame = frame.copy()
        
        for i, (box, score, cls_id) in enumerate(zip(boxes, scores, class_ids)):
            # 提取ROI并进行CNN分类
            roi_img, cnn_prob, cnn_class = self.process_roi(frame, box)
            
            if roi_img is None:
                continue
            
            # 估计姿态
            tvec, rvec = self.estimate_pose(box, frame.shape[1], frame.shape[0])
            
            # 创建Marker
            marker = Marker()
            marker.header.frame_id = self.camera_frame
            marker.header.stamp = current_time
            marker.ns = "detected_objects"
            marker.id = int(cls_id) * 100 + i
            
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position.x = tvec[0]
            marker.pose.position.y = tvec[1]
            marker.pose.position.z = tvec[2]
            marker.pose.orientation.w = 1.0
            
            # 设置尺寸和颜色
            marker.scale.x = self.object_size
            marker.scale.y = self.object_size
            marker.scale.z = self.object_size
            marker.color.r = float(cls_id % 3) / 3.0
            marker.color.g = float((cls_id + 1) % 3) / 3.0
            marker.color.b = float((cls_id + 2) % 3) / 3.0
            marker.color.a = 0.8
            marker.lifetime.sec = 0
            marker.lifetime.nanosec = 100000000
            
            marker_array.markers.append(marker)
            
            # 绘制结果
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 显示标签
            yolo_class = f"class_{cls_id}"  # 替换为你的YOLO类别名称
            cnn_class_name = self.class_names.get(cnn_class, f"class_{cnn_class}")
            label = f"YOLO: {yolo_class}({score:.2f}) | CNN: {cnn_class_name}({cnn_prob:.2f})"
            cv2.putText(debug_frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 发布ROI图像
            self.roi_pub.publish(self.bridge.cv2_to_imgmsg(roi_img, "bgr8"))
        
        # 发布结果
        self.marker_pub.publish(marker_array)
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(debug_frame, "bgr8"))

    def __del__(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()

def main(args=None):
    rclpy.init(args=args)
    node = YoloOnnxCnnDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
import cv2
import cv2.aruco as aruco
import os
import numpy as np
from datetime import datetime

class ArucoDatasetGenerator:
    def __init__(self, output_dir="aruco_dataset"):
        """
        初始化ArUco标记数据集生成器
        
        参数:
            output_dir (str): 保存图像和标注文件的目录
        """
        self.output_dir = output_dir
        self.images_dir = os.path.join(output_dir, "images")
        self.labels_dir = os.path.join(output_dir, "labels")
        
        # 创建目录（如果不存在）
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)
        
        # 初始化ArUco字典和参数
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.parameters = aruco.DetectorParameters()
        
        # 初始化摄像头
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("无法打开摄像头设备")
            
        # 图像计数器
        self.image_count = 0
        
    def detect_aruco_markers(self, frame):
        """
        检测图像中的ArUco标记
        
        参数:
            frame: 输入图像帧
            
        返回:
            corners: 检测到的标记角点列表
            ids: 检测到的标记ID列表
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detector = aruco.ArucoDetector(self.aruco_dict, self.parameters)
        corners, ids, _ = detector.detectMarkers(gray)
        return corners, ids
    
    def convert_to_yolo_format(self, corners, frame_width, frame_height):
        """
        将ArUco标记转换为YOLO格式（归一化的中心x, 中心y, 宽度, 高度）
        
        参数:
            corners: 检测到的标记角点
            frame_width: 图像宽度
            frame_height: 图像高度
            
        返回:
            YOLO格式的标注列表
        """
        yolo_annotations = []
        
        for marker_corners in corners:
            # 计算边界框坐标
            x_min = min(marker_corners[0][:, 0])
            x_max = max(marker_corners[0][:, 0])
            y_min = min(marker_corners[0][:, 1])
            y_max = max(marker_corners[0][:, 1])
            
            # 计算YOLO格式（归一化）
            width = (x_max - x_min) / frame_width
            height = (y_max - y_min) / frame_height
            center_x = ((x_min + x_max) / 2) / frame_width
            center_y = ((y_min + y_max) / 2) / frame_height
            
            # 所有ArUco标记使用同一类别0
            yolo_annotations.append([0, center_x, center_y, width, height])
        
        return yolo_annotations
    
    def save_frame_and_annotations(self, frame, yolo_annotations):
        """
        保存图像帧和YOLO格式标注
        
        参数:
            frame: 要保存的图像
            yolo_annotations: YOLO格式的标注列表
        """
        # 生成基于时间戳的文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        image_filename = f"{timestamp}_{self.image_count}.jpg"
        label_filename = f"{timestamp}_{self.image_count}.txt"
        
        # 保存图像
        image_path = os.path.join(self.images_dir, image_filename)
        cv2.imwrite(image_path, frame)
        
        # 保存标注
        label_path = os.path.join(self.labels_dir, label_filename)
        with open(label_path, 'w') as f:
            for annotation in yolo_annotations:
                line = ' '.join(map(str, annotation)) + '\n'
                f.write(line)
        
        self.image_count += 1
        print(f"已保存 {image_path} 和 {label_path}")
    
    def run(self):
        """
        主循环：捕获帧、检测标记并保存数据集
        """
        print("开始ArUco标记检测。按's'保存当前帧，按'q'退出。")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("获取帧失败")
                break
            
            # 检测ArUco标记
            corners, ids = self.detect_aruco_markers(frame)
            
            # 绘制检测到的标记
            if ids is not None:
                aruco.drawDetectedMarkers(frame, corners, ids)
                
                # 显示标记位置（可选）
                for i, marker_id in enumerate(ids):
                    center = corners[i][0].mean(axis=0)
                    cv2.putText(frame, f"ID:{marker_id[0]}", 
                               (int(center[0]), int(center[1])), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 显示图像帧
            cv2.imshow('ArUco标记检测', frame)
            
            # 处理按键
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and ids is not None:
                # 仅在检测到标记时保存
                frame_height, frame_width = frame.shape[:2]
                yolo_annotations = self.convert_to_yolo_format(corners, frame_width, frame_height)
                self.save_frame_and_annotations(frame, yolo_annotations)
        
        # 清理资源
        self.cap.release()
        cv2.destroyAllWindows()
        
        print(f"数据集创建完成。共保存了 {self.image_count} 个样本到 {self.output_dir}")

if __name__ == "__main__":
    dataset_generator = ArucoDatasetGenerator()
    dataset_generator.run()
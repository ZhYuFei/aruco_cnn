import cv2
import torch
import numpy as np
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

# 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ArucoClassifier().to(device)
model.load_state_dict(torch.load("best_classifier.pth", map_location=device))
model.eval()

# 图像预处理
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# 初始化摄像头
cap = cv2.VideoCapture(0)  # 0 表示默认摄像头

# 创建ArUco字典
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 转换为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 检测ArUco标记
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    
    if ids is not None:
        for i in range(len(ids)):
            # 提取每个标记的区域
            marker_corners = corners[i][0]
            x, y, w, h = cv2.boundingRect(marker_corners)
            marker_roi = gray[y:y+h, x:x+w]
            
            try:
                # 预处理并分类
                marker_roi = cv2.resize(marker_roi, (200, 200))
                tensor = transform(marker_roi).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    outputs = model(tensor)
                    _, predicted = torch.max(outputs.data, 1)
                    aruco_id = predicted.item()
                
                # 在图像上显示结果
                cv2.putText(frame, f"ID: {aruco_id}", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                
            except Exception as e:
                print(f"处理标记时出错: {e}")
                continue
    
    # 显示结果
    cv2.imshow('ArUco Marker Classifier', frame)
    
    # 按'q'退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 清理资源
cap.release()
cv2.destroyAllWindows()
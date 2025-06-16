import cv2
import cv2.aruco as aruco
import os
import numpy as np
from datetime import datetime

# 配置参数
DATA_DIR = "aruco_classification"
DICT_TYPE = aruco.DICT_4X4_50
MARKER_SIZE = 200  # 像素

def setup_dirs():
    """创建按ID分类的目录结构"""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        for i in range(50):  # DICT_4X4_50有50个ID
            os.makedirs(os.path.join(DATA_DIR, str(i)))

def process_marker(frame, corners, marker_id):
    """提取标记ROI并保存"""
    # 透视校正
    dst_pts = np.array([[0,0], [MARKER_SIZE,0], [MARKER_SIZE,MARKER_SIZE], [0,MARKER_SIZE]], dtype="float32")
    M = cv2.getPerspectiveTransform(corners.astype("float32"), dst_pts)
    warped = cv2.warpPerspective(frame, M, (MARKER_SIZE, MARKER_SIZE))
    
    # 保存图像
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    cv2.imwrite(f"{DATA_DIR}/{marker_id}/{timestamp}.png", cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY))

def main():
    setup_dirs()
    detector = aruco.ArucoDetector(aruco.getPredefinedDictionary(DICT_TYPE), 
                                  aruco.DetectorParameters())
    
    cap = cv2.VideoCapture(0)
    print("Press 's' to save, 'q' to quit...")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # 检测标记
        corners, ids, _ = detector.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        
        # 显示检测结果
        debug_frame = frame.copy()
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(debug_frame, corners, ids)
            
        cv2.imshow("ArUco Collector", debug_frame)
        
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('s') and ids is not None:
            for i in range(len(ids)):
                process_marker(frame, corners[i][0], ids[i][0])
                print(f"Saved marker {ids[i][0]}")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
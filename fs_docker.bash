docker run -it \
  --gpus all \
  --name aruco_trainer \
  -v /home/zhouyufei/aruco_cnn/aruco_dataset:/workspace \
  --restart=unless-stopped \
  -p 8888:8888 \
  --shm-size=8G \
  pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime \
  /bin/bash
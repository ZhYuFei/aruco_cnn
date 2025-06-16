# rm_gimbal_description

> xacro xxx.xacro > xxx.urdf

## 坐标系定义

单位和方向请参考 https://www.ros.org/reps/rep-0103.html

odom: 以云台中心为原点的惯性系  
yaw_joint: 表述云台的 yaw 轴与惯性系的旋转关系  
pitch_joint: 表述云台的 pitch 轴与惯性系的旋转关系  
camera_joint: 表述相机到惯性系的变换关系  
camera_optical_joint: 表述以 z 轴为前方的相机坐标系转换为 x 轴为前方的相机坐标系的旋转关系


# Wheeltec 机器人参数

平台无关的硬件参考文档，任何项目部署在此硬件上均可直接引用。

---

## 机器人底盘

| 参数 | 数值 |
|------|------|
| 驱动方式 | 差速驱动 |
| 操作系统 / 中间件 | Ubuntu 20.04 / ROS2 Humble |
| 机载计算 | NVIDIA Jetson Orin / AGX Xavier |
| 自重 | 13.3 kg |
| 负载能力 | 12 kg |
| 最大速度（硬件极限） | 2.7 m/s |
| 尺寸（长×宽×高） | 486 × 523 × 542 mm |
| 驱动轮直径 | 152 mm（越野轮） |
| 轮距 | 312.7 mm |
| 电机 | MD36L 60W 直流有刷电机 |
| 编码器 | 500线 AB相高精度编码器 |

### 导航推荐速度

| 场景 | 推荐值 |
|------|--------|
| 直线速度 | 0.15 – 0.25 m/s |
| 曲线转弯线速度 | < 0.20 m/s |
| 转弯角速度 | < 30 deg/s（0.52 rad/s） |
| **原地旋转角速度** | **< 10 deg/s（0.175 rad/s）**，需极慢 |

---

## 相机：Orbbec Gemini 336L（推荐）

型号：G40155-180

### 规格

| 参数 | 数值 |
|------|------|
| 深度技术 | 双目立体 |
| 发射波长 | 850 nm |
| **深度分辨率（最大）** | **1280 × 800 @ 30fps** |
| **RGB 分辨率（最大）** | **1280 × 800 @ 60fps** |
| 深度测量范围 | 0.17 – 20 m |
| **最优深度范围** | **0.25 – 6 m** |
| 深度 FOV | H 90° × V 65° |
| RGB FOV | H 94° × V 68° |
| 深度精度 | ≤ 0.8%（2m）/ ≤ 1.6%（4m） |
| IMU | 支持 |
| 数据接口 | USB 3.0 Type-C |
| 功耗 | 平均 < 3W |
| 尺寸（W×H×D） | 124 × 29 × 27.7 mm |
| 重量 | 135 g |
| SDK | Orbbec SDK |

### 相机内参

> **注意：** 内参依赖实际标定结果和 SDK 输出分辨率配置，以下数值仅供参考。

**配置为 1280×800 输出（原生分辨率，由 FOV 推算）：**

```
Resolution : 1280 × 800
fx         ≈ 597  （= 640 / tan(47°)）
fy         ≈ 593  （= 400 / tan(34°)）
cx         = 640  （= width / 2）
cy         = 400  （= height / 2）

内参矩阵（4×4）：
[[597.0,   0.0,  640.0,  0.0],
 [  0.0, 593.0,  400.0,  0.0],
 [  0.0,   0.0,    1.0,  0.0],
 [  0.0,   0.0,    0.0,  1.0]]
```

**配置为 1280×720 输出（SDK 截图模式，InternNav 实测标定值）：**

```
Resolution : 1280 × 720
fx         = 607.45
fy         = 607.40
cx         = 639.19
cy         = 361.75

内参矩阵（4×4）：
[[607.45,   0.0,  639.19,  0.0],
 [  0.0,  607.40, 361.75,  0.0],
 [  0.0,    0.0,    1.0,   0.0],
 [  0.0,    0.0,    0.0,   1.0]]
```

部署前请使用 `ros2 topic echo /camera/camera_info` 确认实际内参。

### 深度处理

```
有效范围       : 0.25 – 6.00 m（最优），0.17 – 20 m（极限）
深度编码       : 16UC1，单位 mm
米制转换       : depth_m = raw_uint16 / 1000.0
无效值处理     : NaN / Inf → 0.0
```

---

## 相机：Orbbec Astra S（备选）

| 参数 | 数值 |
|------|------|
| 分辨率（RGB / 深度） | 640 × 480 |
| fx | 570.3 |
| fy | 570.3 |
| cx | 319.5 |
| cy | 239.5 |
| 深度有效范围 | 0.10 – 5.00 m |
| 深度编码 | 16UC1（mm） |

内参矩阵（4×4）：
```
[[570.3,   0.0,  319.5,  0.0],
 [  0.0,  570.3, 239.5,  0.0],
 [  0.0,    0.0,   1.0,  0.0],
 [  0.0,    0.0,   0.0,  1.0]]
```

---

## ROS2 接口

| 方向 | 话题 | 消息类型 | 编码 / 说明 |
|------|------|---------|------------|
| 订阅 | `/camera/color/image_raw` | `sensor_msgs/Image` | `rgb8` |
| 订阅 | `/camera/depth/image_raw` | `sensor_msgs/Image` | `16UC1`（mm） |
| 订阅 | `/odom` | `nav_msgs/Odometry` | 轮式里程计，方向为四元数 |
| 发布 | `/cmd_vel` | `geometry_msgs/Twist` | `linear.x`（m/s），`angular.z`（rad/s） |

### 里程计四元数提取偏航角

```python
import math

def yaw_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)
```

速度字段：`twist.twist.linear.x`（m/s）、`twist.twist.angular.z`（rad/s）

---

## 碰撞安全阈值

```
有效深度范围   : 0.17 – 20.0 m（Gemini 336L 全量程）
碰撞距离阈值   : 0.60 m（中心 ROI 最小深度 < 此值则停车）
检测区域       : 图像水平中三分之一、垂直中三分之一
```

---

## PID 增益（差速底盘，室内导航）

```
Kp_trans = 3.0
Kd_trans = 0.5
Kp_yaw   = 3.0
Kd_yaw   = 0.5

收敛阈值：位置误差 < 0.10 m，偏航误差 < 0.10 rad
```

---

## 相机驱动启动

```bash
# Gemini 336L
ros2 launch orbbec_camera gemini_336l.launch.py

# Astra S
ros2 launch orbbec_camera astra.launch.py

# 底盘驱动（发布 /odom，订阅 /cmd_vel）
ros2 launch wheeltec_robot_bringup bringup.launch.py
```

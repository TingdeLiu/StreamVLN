# StreamVLN × Wheeltec 实机部署指南

硬件参数参见 [`wheeltec_parameter.md`](wheeltec_parameter.md)。

---

## 系统架构

```
GPU 工作站（推理端）                    Jetson / 机器人端
┌──────────────────────────────┐      ┌──────────────────────────────────┐
│ http_realworld_server_       │      │ wheeltec_vln_client.py           │
│   wheeltec.py                │      │                                  │
│                              │      │  ROS2 话题订阅                   │
│  StreamVLNForCausalLM        │ HTTP │  /camera/color/image_raw (RGB)   │
│  VLNEvaluator.step()   ◄─────┼──────┤  /camera/depth/image_raw (Depth)  │
│                              │      │  /odom                           │
│  :8909 /eval_vln             │      │                                  │
│                              │      │  ROS2 话题发布                   │
└──────────────────────────────┘      │  /cmd_vel (Twist)                │
                                      │                                  │
                                      │  Planning 线程 ── 调 HTTP 服务器 │
                                      │  Control  线程 ── PID → Twist    │
                                      └──────────────────────────────────┘
```

---

## 与 Go2 部署的差异

| 方面 | Go2 (`go2_vln_client.py`) | Wheeltec (`wheeltec_vln_client.py`) |
|------|--------------------------|-------------------------------------|
| 运动控制 | Unitree SDK `/api/sport/request` | `geometry_msgs/Twist → /cmd_vel` |
| 里程计 | `SportModeState`（直接 RPY 欧拉角） | `nav_msgs/Odometry`（四元数） |
| RGB 话题 | `/camera/camera/color/image_raw` | `/camera/color/image_raw` |
| 深度话题 | 禁用 | `/camera/depth/image_raw`（启用，用于碰撞检测） |
| 图像同步 | 独立回调 | `ApproximateTimeSynchronizer` |
| 最大线速度 | 1.0 m/s | 0.25 m/s |
| 最大角速度 | 1.2 rad/s | 0.30 rad/s |
| 相机内参 | RealSense D400（代码未显式定义） | Gemini 336L / Astra S（显式传入服务器） |
| 相机高度 | 0.55 m（估算） | 0.35 m |
| 导航指令 | 服务器端硬编码 | 客户端 `--instruction` 参数传入 |

---

## 环境准备

### 服务器端（GPU 工作站）

```bash
conda activate streamvln
pip install flask
```

### 机器人端（Jetson）

```bash
pip install rclpy opencv-python-headless Pillow requests
# 确保已安装 cv_bridge、message_filters（ROS2 humble）
```

---

## 部署步骤

### 步骤 1：启动推理服务器（GPU 工作站）

```bash
cd /path/to/StreamVLN

python streamvln/http_realworld_server_wheeltec.py \
    --model_path checkpoints/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_real_world \
    --camera gemini_336l \
    --num_history 8 \
    --num_future_steps 4 \
    --num_frames 32 \
    --port 8909
```
切换相机只需改 `--camera astra_s`，服务器会自动使用对应内参。

### 步骤 1.5：用 curl 直接测试服务器：
```bash
curl -X POST http://115.190.160.32:8909/eval_vln \
  -F "image=@/home/wheeltec/VLN/StreamVLN/realworld/test.jpg" \
  -F 'json={"reset": true, "instruction": "Go to the chair."}'
```

### 步骤 2：启动机器人驱动（Jetson）

```bash
source /opt/ros/humble/setup.bash

# 相机驱动
ros2 launch turn_on_wheeltec_robot wheeltec_camera.launch.py   # 或 astra.launch.py

# 底盘驱动（发布 /odom，订阅 /cmd_vel）
ros2 launch turn_on_wheeltec_robot turn_on_wheeltec_robot.launch.py
```

### 步骤 3：启动 VLN 客户端（Jetson）

```bash
cd /home/wheeltec/VLN/StreamVLN/realworld

python wheeltec_vln_client.py \
    --server http://115.190.160.32:8909/eval_vln \
    --camera gemini_336l \
    --instruction "Go to the chair and stop."
```

---

## 关键参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--server` | `http://localhost:8909/eval_vln` | 推理服务器地址，局域网部署需改为工作站 IP |
| `--camera` | `gemini_336l` | 相机型号，决定传给服务器的内参矩阵 |
| `--instruction` | `"Navigate to the destination."` | 自然语言导航指令，实时传给 VLN 模型 |
| `--num_history` | 8 | SlowFast 记忆帧数（服务器侧） |
| `--num_future_steps` | 4 | 每次推理预测的动作步数（服务器侧） |
| `--num_frames` | 32 | 每步采样帧数（服务器侧） |

---

## 线程与控制逻辑

```
ROS spin 线程
  └─ rgb_depth_callback()   → 更新 rgb_image / depth_image（已同步）
  └─ odom_callback()        → 更新 homo_odom / vel（每 5 帧采样一次）

Planning 线程（should_plan = True 时触发）
  1. 碰撞检测：深度图中心区域 < 0.60 m → 暂停并等待
  2. 调用 eval_vln(rgb, instruction, server_url)
  3. incremental_change_goal(actions) 更新 homo_goal

Control 线程（10 Hz）
  1. pid.solve(homo_odom, homo_goal, vel) → (v, w)
  2. 发布 Twist 到 /cmd_vel
  3. 位置误差 < 0.10 m 且偏航误差 < 0.10 rad → 触发下一次 Planning
```

---

## 常见问题

**Q: 图像和深度不同步怎么办？**
调大 `ApproximateTimeSynchronizer` 的 `slop` 参数（默认 0.1 s）。

**Q: 小车运动抖动？**
调低 PID 增益，参考 `wheeltec_parameter.md` 中的默认值，建议先将 `Kp_trans` 从 3.0 降至 1.5。

**Q: 服务器推理太慢，动作滞后？**
- 确认模型在 GPU 上（`--device cuda:0`）
- 降低 `--num_history`（如从 8 改为 4）
- 检查网络延迟：`ping <工作站IP>`

**Q: 碰撞检测误触发？**
调高 `COLLISION_DIST`（默认 0.60 m），或缩小检测 ROI 区域（在 `wheeltec_vln_client.py` 中修改 `collision_check()`）。

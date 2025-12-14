#!/usr/bin/env python3
import rospy
import torch
import numpy as np
import cv2
import time
import os
import sys

# ROS 消息类型
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, Twist
from sensor_msgs.msg import Image, PointCloud2, PointField
from sensor_msgs import point_cloud2
import std_msgs.msg

# 数学工具
from scipy.spatial.transform import Rotation as R

# 引入你的项目模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.config import cfg
from policy.rgb_yopo_network import CMCL_YOPO_Network
from policy.state_transform import StateTransform

class YopoInferenceNode:
    def __init__(self):
        rospy.init_node('yopo_inference', anonymous=False)
        
        # --- 1. 参数配置 ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.height = 32   # 2D 场景高度
        self.width = 160
        self.max_dist = 20.0 # 深度图最大距离
        self.traj_time = cfg["sgm_time"]
        
        # 权重路径 (请修改为你训练好的模型路径)
        self.ckpt_path = rospy.get_param("~weights", "./saved/YOPO_5/epoch49.pth")
        
        # 目标点 (全局坐标)
        self.goal_world = np.array([10, 0.0]) # 默认前方10米
        self.has_odom = False
        
        # 车辆当前状态 (全局)
        self.cur_pos = np.zeros(2) # x, y
        self.cur_yaw = 0.0
        self.cur_vel = np.zeros(2) # vx, vy (Body Frame)
        
        # --- 2. 加载模型 ---
        print(f"Loading model from {self.ckpt_path}...")
        self.policy = CMCL_YOPO_Network().to(self.device)
        self.state_transform = StateTransform()
        
        try:
            # 加载权重
            state_dict = torch.load(self.ckpt_path, map_location=self.device)
            self.policy.load_state_dict(state_dict, strict=False) 
            self.policy.eval()
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            exit(1)

        # --- 3. ROS 通信 ---
        # Subscribers
        self.sub_odom = rospy.Subscriber("/odom", Odometry, self.cb_odom, queue_size=1)
        self.sub_depth = rospy.Subscriber("/camera/depth/image", Image, self.cb_depth, queue_size=1, tcp_nodelay=True)
        self.sub_goal = rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.cb_goal, queue_size=1)
        
        # Publishers
        self.pub_traj = rospy.Publisher("/yopo/traj_vis", PointCloud2, queue_size=1) # 可视化轨迹
        self.pub_cmd = rospy.Publisher("/cmd_vel", Twist, queue_size=1) # 控制指令

        print("🚀 YOPO 2D Inference Node Started!")
        rospy.spin()

    def cb_goal(self, msg):
        self.goal_world = np.array([msg.pose.position.x, msg.pose.position.y])
        print(f"New Goal Received: {self.goal_world}")

    def cb_odom(self, msg):
        # 提取位置
        self.cur_pos = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
        
        # 提取姿态 (Quaternion -> Yaw)
        q = msg.pose.pose.orientation
        rot = R.from_quat([q.x, q.y, q.z, q.w])
        self.cur_yaw = rot.as_euler('zyx')[0]
        
        # 提取速度 (Body Frame)
        # 注意: ROS Odometry 的 twist.linear 通常是 Body Frame (child_frame_id)
        self.cur_vel = np.array([msg.twist.twist.linear.x, msg.twist.twist.linear.y])
        
        self.has_odom = True

    @torch.inference_mode()
    def cb_depth(self, msg):
       
        if not self.has_odom: return
        t0 = time.time()

        # --- 1. 深度图预处理 ---
        # 假设输入是 32FC1 (浮点深度, 单位米)
        # 如果是 uint16 (毫米)，需要除以 1000.0
        if msg.encoding == "32FC1":
            depth_np = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
        elif msg.encoding == "16UC1":
            depth_np = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width) / 1000.0
        else:
            return

        # Resize 到网络输入尺寸 (160x32)
        depth_resized = cv2.resize(depth_np, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        
        # 归一化 & 截断
        depth_norm = np.clip(depth_resized / self.max_dist, 0.0, 1.0)
        
        # 堆叠成 3 通道 [3, 32, 160] (适配你的 Dataset 逻辑)
        depth_3ch = np.stack([depth_norm] * 3, axis=0)
        
        # 转 Tensor [1, 3, 32, 160]
        depth_tensor = torch.from_numpy(depth_3ch).float().unsqueeze(0).to(self.device)

        # --- 2. 状态向量预处理 (Body Frame) ---
        # 计算局部目标点 (Goal in Body Frame)
        # 向量 = Goal - Pos
        vec_w = self.goal_world - self.cur_pos
        # 旋转到 Body 系: R_inv * vec
        c, s = np.cos(self.cur_yaw), np.sin(self.cur_yaw)
        R_inv = np.array([[c, s], [-s, c]]) # 2D 旋转矩阵的逆
        goal_b = R_inv @ vec_w
        
        # 限制 Goal 距离 (防止过大数值)
        if np.linalg.norm(goal_b) > 5.0:
            goal_b = goal_b / np.linalg.norm(goal_b) * 5.0

        # 组装 Obs: [vx, vy, ax, ay, gx, gy]
        # 假设当前加速度 ax, ay 为 0 (或者你可以从 IMU 获取)
        acc_b = np.array([0.0, 0.0]) 
        
        obs_np = np.hstack([self.cur_vel, acc_b, goal_b]).astype(np.float32)
        obs_tensor = torch.from_numpy(obs_np).unsqueeze(0).to(self.device)
        
        # 归一化 Obs
        # obs_tensor = self.state_transform.normalize_obs(obs_tensor) (如果训练时用了这个)

        # --- 3. 网络推理 ---
        # endstate: [1, 6] (px, py, vx, vy, ax, ay)
        # score: [1, 1]
        endstate_norm, score = self.policy.inference(depth_tensor, obs_tensor)
        
        # --- 4. 生成轨迹 & 控制 ---
        # 还原物理状态 (Body Frame)
        # 注意: 这里的 pred_to_endstate_2d 需要是你最新修改过的那个单轨迹版本
        # 它返回的是 Body Frame 下的 [1, 3, 3] (Pos, Vel, Acc) 或者是 [1, 6]
        # 我们这里假设 inference 内部已经调用了 pred_to_endstate_2d，返回的是物理值 [1, 3, 3]
        # 如果 inference 返回的是 normalized 的，这里需要手动转换一下
        
        # 假设 inference 返回的是已经解算好的物理状态 (Body Frame)
        # endstate: [Batch, 3, 3] -> [Pos(x,y,z), Vel, Acc]
        
        # 如果你的 inference 没有做物理转换，在这里做：
        # endstate_phys = self.policy.pred_to_endstate_2d(endstate_norm)
        endstate_phys = endstate_norm # 假设 inference 里已经转好了
        
        # 生成多项式轨迹点
        traj_points = self.generate_poly_traj(self.cur_vel, acc_b, endstate_phys[0])
        
        # 发布可视化
        self.publish_traj(traj_points, score.item())
        
        # 发布控制 (简单的纯追踪 Pure Pursuit 或 PID)
        self.publish_control(traj_points)
        
        t_process = (time.time() - t0) * 1000
        
        # print(f"Inference Time: {t_process:.2f}ms | Score: {score.item():.4f}")

    def generate_poly_traj(self, start_vel, start_acc, end_state):
        """
        生成 5 阶多项式轨迹用于可视化和控制
        end_state: [3, 3] (Pos, Vel, Acc) (包含Z轴0) 或 [6]
        """
        # 构造 Start (Body Frame, Pos=0)
        p0 = np.zeros(2)
        v0 = start_vel
        a0 = start_acc
        
        # 解析 End
        # 如果 end_state 是 [3, 3] (Pos, Vel, Acc)
        if end_state.shape == (3, 3):
            p1 = end_state[0, :2].cpu().numpy()
            v1 = end_state[1, :2].cpu().numpy()
            a1 = end_state[2, :2].cpu().numpy()
        else:
            # 如果是 [6] (px, py, vx, vy, ax, ay)
            e = end_state.cpu().numpy()
            p1, v1, a1 = e[0:2], e[2:4], e[4:6]

        T = self.traj_time
        
        # 求解 5 阶多项式系数 (Quintic Polynomial)
        # p(t) = c0 + c1*t + c2*t^2 + c3*t^3 + c4*t^4 + c5*t^5
        # 已知边界条件求解
        # 这里用简化的矩阵形式求解 X 和 Y
        
        def solve_quintic(x0, v0, a0, x1, v1, a1, T):
            A = np.array([
                [0, 0, 0, 0, 0, 1], # p(0)
                [0, 0, 0, 0, 1, 0], # v(0)
                [0, 0, 0, 2, 0, 0], # a(0)
                [T**5, T**4, T**3, T**2, T, 1], # p(T)
                [5*T**4, 4*T**3, 3*T**2, 2*T, 1, 0], # v(T)
                [20*T**3, 12*T**2, 6*T, 2, 0, 0]  # a(T)
            ])
            b = np.array([x0, v0, a0, x1, v1, a1])
            # coeffs: [c5, c4, c3, c2, c1, c0]
            return np.linalg.solve(A, b)

        coeffs_x = solve_quintic(p0[0], v0[0], a0[0], p1[0], v1[0], a1[0], T)
        coeffs_y = solve_quintic(p0[1], v0[1], a0[1], p1[1], v1[1], a1[1], T)
        
        # 采样 20 个点
        t = np.linspace(0, T, 20)
        
        # 计算坐标 (Horners method or matrix)
        # p = c5*t^5 + ...
        poly = lambda c, t: c[0]*t**5 + c[1]*t**4 + c[2]*t**3 + c[3]*t**2 + c[4]*t + c[5]
        
        xs = poly(coeffs_x, t)
        ys = poly(coeffs_y, t)
        
        return np.stack([xs, ys], axis=1)

    def publish_traj(self, points, score):
        # 构造 PointCloud2 (Body Frame / base_link)
        # points: [N, 2]
        z = np.zeros((points.shape[0], 1))
        # intensity 用 score 填充
        i = np.full((points.shape[0], 1), score) 
        
        pc_data = np.hstack([points, z, i]).astype(np.float32)
        
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "base_link" # 关键：轨迹是在车身坐标系下的
        
        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('intensity', 12, PointField.FLOAT32, 1)
        ]
        
        msg = point_cloud2.create_cloud(header, fields, pc_data)
        self.pub_traj.publish(msg)

    def publish_control(self, points):
        # 简单的纯追踪 (Pure Pursuit) 或 取点控制
        # 取第 5 个点 (约 0.2s - 0.4s 处) 作为预瞄点
        lookahead_idx = min(5, len(points)-1)
        target = points[lookahead_idx] # [x, y]
        
        # 计算曲率 / 角速度
        # w = 2 * y / L^2 * v
        L2 = target[0]**2 + target[1]**2
        if L2 < 0.01: 
            w = 0
            v = 0
        else:
            # 期望速度 (可以根据 curvature 动态调整)
            v_cmd = 1.0 # 假设恒定速度，或者用网络预测的 v1
            w_cmd = 2 * target[1] / L2 * v_cmd
        
        cmd = Twist()
        cmd.linear.x = v_cmd
        cmd.angular.z = np.clip(w_cmd, -1.0, 1.0) # 限制角速度
        self.pub_cmd.publish(cmd)

if __name__ == "__main__":
    try:
        node = YopoInferenceNode()
    except rospy.ROSInterruptException:
        pass
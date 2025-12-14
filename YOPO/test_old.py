import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import sys

# 引入你的项目模块
from config.config import cfg
from policy.rgb_yopo_network import CMCL_YOPO_Network
from data.Yopo_dataset import YOPODataset
from torch.utils.data import DataLoader

# 【关键修改】引入 YOPOLoss
from loss.loss_function import YOPOLoss

# ================= 配置 =================
# 5  10
CHECKPOINT_PATH = "/home/zxx/YOPO-main/YOPO/YOPO/saved/YOPO_19/epoch49.pth" # 替换为你实际保存权重的路径
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAJ_TIME = cfg["sgm_time"]
def visualize_result(depth_img, traj_pred, score_pred, goal_body, start_state):
    """
    画图：左边是深度图，右边是 2D 俯视图（轨迹）
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # --- 1. 左图：深度图修正 ---
    # depth_img 输入可能是 Tensor [B, 3, H, W]
    if isinstance(depth_img, torch.Tensor):
        # 1. 取 Batch 第 0 个 -> [3, 32, 160]
        # 2. 取 Channel 第 0 个 -> [32, 160] (我们只需要单通道显示热力图)
        # 3. 转 Numpy
        if depth_img.dim() == 4:
            depth_show = depth_img[0, 0, :, :].cpu().numpy()
        elif depth_img.dim() == 3:
            depth_show = depth_img[0, :, :].cpu().numpy()
    else:
        # 如果是 Numpy [3, 32, 160] -> [32, 160]
        depth_show = depth_img[0]
        
    # 
    # 现在 depth_show 的 shape 是 (32, 160)，imshow 可以正确处理了
    axes[0].imshow(depth_show, cmap='plasma')
    axes[0].set_title(f"Input Depth (Score: {score_pred:.4f})")
    axes[0].axis('off')

    # --- 2. 右图：2D 轨迹 (保持不变) ---
    traj = traj_pred.cpu().numpy()
    
    axes[1].plot(0, 0, 'ro', markersize=10, label='Car')
    axes[1].arrow(0, 0, 1.0, 0, head_width=0.2, head_length=0.2, fc='k', ec='k')
    
    gx, gy = goal_body[0], goal_body[1]
    axes[1].plot(gx, gy, 'g*', markersize=15, label='Goal')
    
    axes[1].plot(traj[:, 0], traj[:, 1], 'b-', linewidth=3, label='Predicted Traj')
    
    axes[1].set_xlim(-1, 10)
    axes[1].set_ylim(-5, 5)
    axes[1].grid(True)
    axes[1].set_title("Trajectory (Top-Down View)")
    axes[1].legend()
    axes[1].set_xlabel("X (Forward)")
    axes[1].set_ylabel("Y (Left/Right)")
    
    plt.tight_layout()
    plt.show()

def test_one_batch():
    # 1. 加载模型
    print(f"Loading model from {CHECKPOINT_PATH}...")
    model = CMCL_YOPO_Network().to(DEVICE)
    
    try:
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(ckpt)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Loading strict failed, trying strict=False... ({e})")
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE), strict=False)
        
    model.eval()
    
    # 2. 准备数据
    val_dataset = YOPODataset(mode='valid')
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)
    
    # ==============================================================
    # 【关键修改】 3. 初始化工具
    # ==============================================================
    print("Initializing YOPOLoss to get SafetyLoss tool...")
    # 直接初始化 YOPOLoss，它内部会计算 L 并创建 safety_loss
    loss_module = YOPOLoss()
    # 取出里面的 safety_loss 对象，它包含了 get_coefficient 等工具函数
    safety_tool = loss_module.safety_loss
    
    # 确保工具在正确的设备上
    # 如果 YOPOLoss 初始化没放 device，这里手动赋值一下 L 的 device
    # (假设 SafetyLoss 内部用 self._L.device)
    if hasattr(safety_tool, '_L'):
        safety_tool._L = safety_tool._L.to(DEVICE)
        safety_tool.device = DEVICE

    print("Start Inference...")
    
    for i, batch in enumerate(val_loader):
        if i >= 5: break 
        map_id = batch['map_id'].to(DEVICE)
        depth = batch['depth'].to(DEVICE)
        obs = batch['obs'].to(DEVICE) # [B, 6]
        
        # --- 推理 ---
        with torch.no_grad():
            endstate, score = model.inference(depth, obs)
        
        # --- 还原轨迹点 ---
        start_pos = torch.zeros(1, 2).to(DEVICE)
        start_vel = obs[:, 0:2]
        start_acc = obs[:, 2:4]
        
        end_pos = endstate[:, 0:2]
        end_vel = endstate[:, 2:4]
        end_acc = endstate[:, 4:6]
        
        # 拼成 [B, 2, 3] (Dim1: x,y; Dim2: p,v,a)
        # 注意 SafetyLoss2D 的 coefficient 计算需要 [B, 2, 3]
        Df = torch.stack([start_pos, start_vel, start_acc], dim=2)
        Dp = torch.stack([end_pos, end_vel, end_acc], dim=2)
        
        # 调用 safety_tool 算轨迹点
        # 1. 扩充 L 矩阵 [1, 6, 6]
        L = safety_tool._L.unsqueeze(0).expand(1, -1, -1)
        
        # 2. 算系数
        zeros = torch.zeros_like(Df[:, :1, ...])
        Df_3d =  torch.cat([Df, zeros], dim=1) 
        Dp_3d =  torch.cat([Dp, zeros], dim=1)
        coe = safety_tool.get_coefficient_from_derivative(Dp_3d, Df_3d, L)
        
        # 3. 算点
        dt = cfg["sgm_time"] / 50
        t_list = torch.linspace(dt, cfg["sgm_time"], 50, device=DEVICE).view(1, -1, 1)
        pos_seq = safety_tool.get_position_from_coeff(coe, t_list) # [1, 50, 2]
        
        # --- 可视化 ---
        goal_b = obs[0, 4:6].cpu().numpy()
        start_v = obs[0, 0].item()
        
        print(f"Sample {i}: Vel={start_v:.2f}, Score={score.item():.4f}")
        visualize_result(depth, pos_seq[0], score.item(), goal_b, None)

if __name__ == "__main__":
    test_one_batch()
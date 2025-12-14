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
CHECKPOINT_PATH = "/home/zxx/YOPO-main/YOPO/YOPO/saved/YOPO_1/epoch50.pth" # 替换为你实际保存权重的路径
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAJ_TIME = cfg["sgm_time"]

def visualize_both_with_local_obstacles(depth_img, traj_pred, score_pred, goal_world, esdf_map, meta_info, vehicle_pos, current_vel=None):
    """
    新增参数:
    current_vel: [vx, vy] 当前车辆速度，用于计算车头朝向
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # --- 1. 左图：深度图 ---
    ax1 = axes[0]
    if isinstance(depth_img, torch.Tensor):
        if depth_img.dim() == 4: depth_show = depth_img[0, 0].cpu().numpy()
        elif depth_img.dim() == 3: depth_show = depth_img[0].cpu().numpy()
        else: depth_show = depth_img
    else: depth_show = depth_img
    
    ax1.imshow(depth_show, cmap='plasma') # 建议用 plasma 或 viridis，看起来更像深度
    ax1.set_title(f"Input Depth (Camera View)\nScore: {score_pred:.4f}")
    ax1.axis('off')
    
    # --- 2. 右图：局部 ESDF 地图 ---
    ax2 = axes[1]
    
    # 解析地图信息
    ox, oy, res, map_width, map_height = meta_info
    ox, oy, res = ox.item(), oy.item(), res.item()
    map_width, map_height = int(map_width.item()), int(map_height.item())
    esdf_data = esdf_map.cpu().numpy()
    
    # 车辆位置
    vx, vy = vehicle_pos[0], vehicle_pos[1]
    
    # --- 【关键调试步 1】计算车头朝向 Yaw ---
    # 如果传入了速度且速度模长足够大，用速度方向；否则默认 0 或上一帧方向
    yaw = 0.0
    if current_vel is not None:
        vel_norm = np.linalg.norm(current_vel)
        if vel_norm > 0.1:
            yaw = np.arctan2(current_vel[1], current_vel[0])
        else:
            print("Warning: Speed too low, assuming Yaw=0 (East)")
            
    # --- 裁剪局部地图 (20米范围) ---
    local_radius = 15.0 # 米
    
    # 画全图背景 (为了调试，先画大一点的范围，或者直接 crop)
    # 计算 Crop 边界
    v_px = int((vx - ox) / res)
    v_py = int((vy - oy) / res)
    r_px = int(local_radius / res)
    
    x_min_px, x_max_px = max(0, v_px - r_px), min(map_width, v_px + r_px)
    y_min_px, y_max_px = max(0, v_py - r_px), min(map_height, v_py + r_px)
    
    local_esdf = esdf_data[y_min_px:y_max_px, x_min_px:x_max_px]
    
    # 计算局部图的物理范围
    extent = [
        ox + x_min_px * res, ox + x_max_px * res,
        oy + y_min_px * res, oy + y_max_px * res
    ]
    
    # 显示地图 (注意 origin='lower')
    # RdYlGn: 红(低/危险) -> 绿(高/安全)。如果 ESDF<0 是障碍，那么障碍是红的。
    # 你的图中障碍物似乎是红色，所以不要用 _r，或者根据你的 ESDF 定义调整
    im = ax2.imshow(local_esdf, extent=extent, origin='lower', cmap='RdYlGn', alpha=0.8)
    plt.colorbar(im, ax=ax2, label='Distance to Obstacle')
    
    # --- 【关键调试步 2】画相机视野 (FOV) ---
    # 假设相机 FOV 是 90 度
    fov = np.deg2rad(90)
    view_len = 8.0 # 视野长度 8米
    
    # 左边界向量
    left_angle = yaw + fov / 2
    lx = vx + view_len * np.cos(left_angle)
    ly = vy + view_len * np.sin(left_angle)
    
    # 右边界向量
    right_angle = yaw - fov / 2
    rx = vx + view_len * np.cos(right_angle)
    ry = vy + view_len * np.sin(right_angle)
    
    # 画出 FOV 锥形 (黄色虚线)
    ax2.plot([vx, lx], [vy, ly], 'y--', linewidth=2, label='Camera FOV Left')
    ax2.plot([vx, rx], [vy, ry], 'y--', linewidth=2, label='Camera FOV Right')
    # 画车头箭头
    ax2.arrow(vx, vy, 2*np.cos(yaw), 2*np.sin(yaw), head_width=0.5, color='black', zorder=10)

    # 绘制车辆和目标
    ax2.plot(vx, vy, 'ro', markersize=10, label='Vehicle', zorder=10)
    ax2.plot(goal_world[0], goal_world[1], 'g*', markersize=15, label='Goal')
    
    # 绘制轨迹
    traj = traj_pred.cpu().numpy()
    ax2.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=2, label='Pred Traj')
    
    ax2.set_title(f"World Map (Vehicle Heading: {np.rad2deg(yaw):.1f}°)")
    ax2.legend(loc='upper right')
    ax2.set_aspect('equal') # 保证 X 和 Y 比例一致，这很重要！
    
    plt.tight_layout()
    plt.show()
    
def debug_map_alignment(esdf_map, meta_info, vehicle_pos, depth_img):
    """
    画出4种可能的地图方向，帮你一眼看出哪种是对的。
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # 1. 准备数据
    esdf_raw = esdf_map.cpu().numpy() # [H, W] or [W, H]
    ox, oy, res = meta_info[0].item(), meta_info[1].item(), meta_info[2].item()
    vx, vy = vehicle_pos[0].item(), vehicle_pos[1].item()
    
    # 2. 生成四种变换候选项
    candidates = [
        ("Original (No Change)", esdf_raw),
        ("Flip Up-Down (Reverses Y)", np.flipud(esdf_raw)),
        ("Transpose (Swap X/Y)", esdf_raw.T),
        ("Transpose + Flip UD", np.flipud(esdf_raw.T))
    ]

    # 3. 绘图
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # (0,0) 放深度图作为参考
    ax_ref = axes[0, 0]
    if isinstance(depth_img, torch.Tensor):
        d_show = depth_img[0, 0].cpu().numpy() if depth_img.dim()==4 else depth_img[0].cpu().numpy()
    else: d_show = depth_img
    ax_ref.imshow(d_show, cmap='plasma')
    ax_ref.set_title("REFERENCE: Depth View\n(Bottom should be clear)")
    
    # 后面4个格子放地图候选项
    map_axes = [axes[0, 1], axes[0, 2], axes[1, 1], axes[1, 2]]
    axes[1, 0].axis('off') # 空一个格子

    for ax, (name, map_data) in zip(map_axes, candidates):
        h, w = map_data.shape
        # 假设 map_data 的 row 对应 y, col 对应 x (或者反之，我们看图说话)
        # 强制使用 origin='lower'，模拟物理坐标系习惯
        extent = [ox, ox + w * res, oy, oy + h * res]
        
        im = ax.imshow(map_data, origin='lower', extent=extent, cmap='RdYlGn', alpha=0.8)
        
        # 画车
        ax.plot(vx, vy, 'ro', markersize=15, markeredgecolor='black', label='Car')
        # 画个小箭头表示“如果车向东开”
        ax.arrow(vx, vy, 2.0, 0, head_width=0.5, color='blue')
        
        # 局部裁剪 (方便看细节)
        margin = 15
        ax.set_xlim(vx - margin, vx + margin)
        ax.set_ylim(vy - margin, vy + margin)
        
        # 判定逻辑提示
        val_at_car = "Unknown"
        # 尝试读取一下车位置的数值
        px = int((vx - ox) / res)
        py = int((vy - oy) / res)
        if 0 <= px < w and 0 <= py < h:
            val = map_data[py, px]
            status = "COLLISION" if val <= 0.5 else "SAFE"
            val_str = f"{val:.1f} ({status})"
        else:
            val_str = "Out of Map"
            
        ax.set_title(f"{name}\nCar Value: {val_str}")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    input("Press Enter to continue inference...") # 暂停让你看图

def test_one_batch():
    # ... (加载模型代码保持不变) ...
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
    # 初始化 Loss 工具
    loss_module = YOPOLoss()
    safety_tool = loss_module.safety_loss
    if hasattr(safety_tool, '_L'):
        safety_tool._L = safety_tool._L.to(DEVICE)
        safety_tool.device = DEVICE

    print("Start Inference...")
    val_dataset = YOPODataset(mode='valid')
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)
    

    for i, batch in enumerate(val_loader):
        if i >= 5: break
            
        map_id = batch['map_id'].to(DEVICE)
        depth = batch['depth'].to(DEVICE)
        obs = batch['obs'].to(DEVICE) 
        # ==========================================
        # 【关键修复 1】 获取真实的世界坐标起点
        # ==========================================
        # 假设 batch['gt_traj'] 存在，形状通常是 [B, Points, 2]
        # 如果你的 dataset key 不叫 'gt_traj'，请查看 YOPODataset 的 __getitem__ 返回了什么
        if 'gt_traj' in batch:
            # 取 GT 轨迹的第一个点作为当前车辆的世界坐标
            real_start_pos = batch['gt_traj'][:, 0, :].to(DEVICE) # [B, 2]
        elif 'pos_w' in batch:
            real_start_pos = batch['pos_w'].to(DEVICE)
        else:
            # 如果 Dataset 真的没返回绝对坐标，这里只能报错或者临时用 0 (但这会导致地图不对)
            print("Warning: No global position found in batch! Visualization will be wrong.")
            real_start_pos = torch.zeros(1, 2).to(DEVICE)

        # --- 推理 (得到局部轨迹) ---
        with torch.no_grad():
            # endstate 是相对于车辆的局部状态
            local_endstate, score = model.inference(depth, obs)
        
        # --- 还原轨迹点 ---
        # 起点不再是 0，而是真实世界坐标
        start_pos = real_start_pos 
        
        # 速度加速度保持不变 (obs 里通常是局部系或者车身系，直接用)
        start_vel = obs[:, 0:2]
        start_acc = obs[:, 2:4]
        
        # 终点状态 = 局部预测 + 世界坐标起点
        end_pos = local_endstate[:, 0:2] + start_pos # 【关键】加上偏移量
        end_vel = local_endstate[:, 2:4]
        end_acc = local_endstate[:, 4:6]
        
        Df = torch.stack([start_pos, start_vel, start_acc], dim=2)
        Dp = torch.stack([end_pos, end_vel, end_acc], dim=2)
        
        # ... (计算系数 coe 和 pos_seq 逻辑不变) ...
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
        
        # --- 获取地图信息 ---
        map_idx = map_id[0].long().item()
        esdf_map = safety_tool.all_maps[map_idx, 0]
        meta_info = safety_tool.all_metas[map_idx]
        
        # --- 可视化 ---
        # 目标点 goal_b 是局部的，为了画图，我们也把它转成世界坐标
        local_goal = obs[0, 4:6]
        global_goal = local_goal + start_pos[0]
        # ... Inside loop ...
# obs: [B, 6] -> [vx, vy, ax, ay, gx, gy]
        current_velocity = obs[0, 0:2].cpu().numpy() # [vx, vy]
        debug_map_alignment(esdf_map, meta_info, start_pos[0].cpu().numpy(), depth)
        collision_points = visualize_both_with_local_obstacles(
            depth, 
            pos_seq[0], 
            score.item(), 
            global_goal.cpu().numpy(),
            esdf_map,
            meta_info,
            vehicle_pos=start_pos[0].cpu().numpy(),
            current_vel=current_velocity # <--- 新增
        )

if __name__ == "__main__":
    test_one_batch()
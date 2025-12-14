import os
import sys
import cv2
import time
import torch
import tomli
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from scipy.spatial.transform import Rotation as R
from sklearn.model_selection import train_test_split


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
from config.config import cfg




class YOPODataset(Dataset):
    def __init__(self, mode='train', val_ratio=0.1):
        super(YOPODataset, self).__init__()
        
        self.height = int(cfg["image_height"])
        self.width = int(cfg["image_width"])
        self.max_depth_range = 20.0
        
        # --- 随机状态参数 (2D) ---
        self.vel_max = cfg["vel_max_train"]
        self.acc_max = cfg["acc_max_train"]
        
        
        # 只需要 X, Y 的均值和方差 最终速度(Vx​)=动力天花板−阻力 (Noise)   
        target_v_ratio = cfg["vx_mean_unit"] #0.2
        target_std = cfg["vx_std_unit"] #1.2 波动
        noise_mean = 1.2 - target_v_ratio   #在2.4 -1.2波动
        sigma_sq = np.log(1 + target_std ** 2 / (noise_mean ** 2))

        self.vx_logmorm_sigma = np.sqrt(sigma_sq)  # 这里的 sigma 越大，尾巴越长！
        self.vx_lognorm_mean = np.log(noise_mean) - 0.5 * sigma_sq

        self.v_mean = np.array([cfg["vx_mean_unit"], cfg["vy_mean_unit"]])
        self.v_std = np.array([cfg["vx_std_unit"], cfg["vy_std_unit"]])


        self.a_mean = np.array([cfg["ax_mean_unit"], cfg["ay_mean_unit"]])
        self.a_std = np.array([cfg["ax_std_unit"], cfg["ay_std_unit"]])
        
        self.goal_length = cfg['goal_length']
        self.goal_yaw_std = cfg["goal_yaw_std"]

        # self.data_root = "/home/zxx/YOPO-Sim/TrainingData"
        self.data_root = "/home/zxx/YOPO-Sim/Test"
        # --- 扫描数据 ---
        self.samples = [] 
        self._scan_dataset(mode, val_ratio)
        
        if mode == 'train': self.print_data()

    def _scan_dataset(self, mode, val_ratio):
        if not os.path.exists(self.data_root):
            raise FileNotFoundError(f"Data root not found: {self.data_root}")
        

        scene_folders = sorted([
            f.path for f in os.scandir(self.data_root) if f.is_dir() and "Scene_" in f.name
        ], key=lambda x: int(os.path.basename(x).split("_")[-1]))

        all_data = []
        print(f"Scanning {len(scene_folders)} scenes...")
        
        for scene_path in scene_folders:
            toml_path = os.path.join(scene_path, "data.toml")
            textures_dir = os.path.join(scene_path, "Textures")
            meta_path = os.path.join(scene_path, "map_meta.npy")
            
            if not os.path.exists(meta_path) or not os.path.exists(toml_path):
                continue
                
            map_meta = np.load(meta_path) # [ox, oy, res, w, h]
            scene_idx = int(os.path.basename(scene_path).split("_")[-1])
            
            with open(toml_path, "rb") as f:
                conf = tomli.load(f)
            
            if "dataArray" not in conf: continue

            for entry in conf["dataArray"]:
                files = entry.get("imageFileNameList", [])
                depth_name = next((f for f in files if "depth" in f and f.endswith(".exr")), None)

                if depth_name:
                    full_depth_path = os.path.join(textures_dir, depth_name)
                    if not os.path.exists(full_depth_path): continue
                    
                    pos_2d = entry.get("posStart", [0.0, 0.0])
                    yaw_deg = entry.get("yawStart", 0.0)
                    # 将角度转为弧度
                    yaw_rad = np.deg2rad(yaw_deg)
                    c = np.cos(yaw_rad)
                    s = np.sin(yaw_rad)
                    # 构造 2D 旋转矩阵 (Loss计算必须用)
                    rot_matrix = np.array([[c, -s], 
                       [s,  c]], dtype=np.float32)
                    
                    # 构造 2D 位置 (Loss计算必须用) 传原始 transform  
                    pos_2d = np.array([pos_2d[0], pos_2d[1]], dtype=np.float32)
                    all_data.append({
                        "depth_path": full_depth_path,
                        "pos_w": pos_2d,
                        "rot_w": rot_matrix,
                        "map_meta": map_meta,
                        "map_id": scene_idx
                    })

        # train_data, val_data = train_test_split(all_data, test_size=0.1, random_state=42)
        if len(all_data) < 2:
            print("⚠️ Warning: Only 1 sample found. Using it for both Train and Val.")
            train_data = all_data
            val_data = all_data 
        else:
            train_data, val_data = train_test_split(all_data, test_size=0.1, random_state=42)
            
        self.samples = train_data if mode == 'train' else val_data
        print(f"[{mode.capitalize()}] Loaded {len(self.samples)} samples.")

    def __getitem__(self, idx):
        item = self.samples[idx]
        
        # 1. Depth Image
        depth_img = cv2.imread(item["depth_path"], cv2.IMREAD_UNCHANGED)
       
        if depth_img is None: 
            depth_img = np.zeros((self.height, self.width), dtype=np.float32)
        
        # 【核心修复】 处理多通道 EXR (例如 (H, W, 4) -> (H, W))
        if len(depth_img.shape) == 3:
            # 只取第一个通道作为深度
            depth_img = depth_img[:, :, 0]
        
        depth_img = cv2.resize(depth_img, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        depth_img = depth_img.astype(np.float32) / self.max_depth_range
        depth_img = np.clip(depth_img, 0.0, 1.0)
        # 三通道

        depth_tensor = np.stack([depth_img] * 3, axis=0)  # [3, H, W]

        # 2. Random State (2D)
        vel_b, acc_b = self._get_random_state() # [vx, vy], [ax, ay]

        # 3. Random Goal (2D Body Frame)
        goal_b = self._get_random_goal_body() # [gx, gy]
        
        # 4. 组装输入 Observation (6维)
        # [vx, vy, ax, ay, gx, gy]# vel & acc & goal are in body frame, NWU, and no-normalization
        obs = np.hstack((vel_b, acc_b, goal_b)).astype(np.float32)

        return {
            "depth": torch.from_numpy(depth_tensor),
            # --- 辅助数据 (2D, 用于 Loss) ---
            "pos_w": item["pos_w"],      # [3]
            "rot_w": item["rot_w"],      # [3, 3]
            "obs": torch.from_numpy(obs),
            "map_meta": item["map_meta"],
            "map_id": item["map_id"]
        }

    def _get_random_state(self):
        """生成 2D 随机速度和加速度"""
        while True:
            # 2D 高斯分布
            vel = self.vel_max * (self.v_mean + self.v_std * np.random.randn(2))
            
            # X轴速度修正：对数正态分布 (保证主要向前)
            right_skewed_vx = -1
            while right_skewed_vx < 0:
                right_skewed_vx = self.vel_max * np.random.lognormal(
                    mean=self.vx_lognorm_mean, sigma=self.vx_logmorm_sigma
                )
                # 翻转逻辑：让大部分速度集中在 vel_max 附近，且为正
                right_skewed_vx = -right_skewed_vx + 1.2 * self.vel_max
            
            # 简单的截断，防止负速度过多
            vel[0] = max(right_skewed_vx, 0.0)
            
            if np.linalg.norm(vel) < 1.2 * self.vel_max: 
                break

        while True:
            acc = self.acc_max * (self.a_mean + self.a_std * np.random.randn(2))
            if np.linalg.norm(acc) < 1.2 * self.acc_max: 
                break
        
        return vel, acc

    def _get_random_goal_body(self):
        """生成 2D 随机目标 (Body Frame)"""
        # Yaw 角度正态分布 (0度代表正前方)
        yaw_deg = np.random.normal(0.0, self.goal_yaw_std)
        yaw_rad = np.radians(yaw_deg)

        # 极坐标 -> 笛卡尔坐标
        direction = np.array([np.cos(yaw_rad), np.sin(yaw_rad)])
        
        # 10% 概率生成近处目标
        scale = self.goal_length * (np.random.rand() * 0.8 + 0.2 if np.random.rand() < 0.1 else 1.0)
        
        return scale * direction

    def print_data(self):
        print(f"Dataset Info: {self.width}x{self.height}, MaxVel={self.vel_max}, MaxDist={self.max_depth_range}")

    def __len__(self):
        return len(self.samples)

# ================= 验证环节 =================
if __name__ == '__main__':
    # 1. 实例化
    print("Initializing Dataset...")
    ds = YOPODataset(mode='train')
    
    if len(ds) == 0:
        print("❌ Dataset is empty. Check path.")
        sys.exit()

    # 2. DataLoader 测试
    dl = DataLoader(ds, batch_size=16, shuffle=True)
    batch = next(iter(dl))
    
    depth = batch['depth']
    obs = batch['obs']
    pos = batch['pos_w']
    rot = batch['rot_w']
    
    print("\n✅ Batch Data Check:")
    print(f"  Depth Shape: {depth.shape} (Expect [B, 1, 32, 160])")
    print(f"  Obs Shape:   {obs.shape}   (Expect [B, 6]) -> [vx, vy, ax, ay, gx, gy]")
    print(f"  Pos Shape:   {pos.shape}   (Expect [B, 3]) -> 2D World Pos")
    print(f"  Rot Shape:   {rot.shape}   (Expect [B, 3, 3]) -> 2D Rotation Matrix")

    # 3. 采样分布可视化验证
    print("\n📊 Generating Sampling Distribution Plot...")
    N = 10000
    vels, accs, goals = [], [], []
    
    for _ in range(N):
        v, a = ds._get_random_state()
        g = ds._get_random_goal_body()
        vels.append(v)
        accs.append(a)
        goals.append(g)
    
    vels = np.array(vels)
    accs = np.array(accs)
    goals = np.array(goals)
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # Velocity Plot
    axs[0].scatter(vels[:, 0], vels[:, 1], s=2, alpha=0.5)
    axs[0].set_title(f"Velocity Distribution (Max={ds.vel_max})")
    axs[0].set_xlabel("Vx (Forward)")
    axs[0].set_ylabel("Vy (Lateral)")
    axs[0].grid(True)
    axs[0].axis('equal')
    
    # Acceleration Plot
    axs[1].scatter(accs[:, 0], accs[:, 1], s=2, alpha=0.5, c='orange')
    axs[1].set_title(f"Acceleration Distribution (Max={ds.acc_max})")
    axs[1].set_xlabel("Ax")
    axs[1].set_ylabel("Ay")
    axs[1].grid(True)
    axs[1].axis('equal')
    
    # Goal Plot
    axs[2].scatter(goals[:, 0], goals[:, 1], s=2, alpha=0.5, c='green')
    axs[2].set_title(f"Goal Distribution (Len={ds.goal_length})")
    axs[2].set_xlabel("Gx")
    axs[2].set_ylabel("Gy")
    axs[2].grid(True)
    axs[2].axis('equal')
    
    print("Plotting done. Please check the popup window.")
    plt.show()
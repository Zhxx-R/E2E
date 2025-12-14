import os
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from config.config import cfg

class SafetyLoss(nn.Module):
    def __init__(self, L):
        super(SafetyLoss, self).__init__()
        self.traj_num = cfg['traj_num']
        self.d0 = cfg["d0"]
        self.r = cfg["r"]
        self._L = L
        self.sgm_time = cfg["sgm_time"]
        self.eval_points = 30
        self.device = self._L.device
        self.truncate_cost = False 

        # SDF
        self.voxel_size = 0.2
        print("Loading 2D ESDF maps...")
        self.all_maps, self.all_metas = self._load_maps_stack()

    def _load_maps_stack(self):
        data_dir = "/home/zxx/YOPO-Sim/Test"
        # data_dir = "/home/zxx/YOPO-Sim/TrainingData" # 请确认路径
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data root not found: {data_dir}")
            
        scene_folders = sorted([
            f.path for f in os.scandir(data_dir) if f.is_dir() and "Scene_" in f.name
        ], key=lambda x: int(os.path.basename(x).split("_")[-1]))

        map_list = []
        meta_list = []

        for scene_path in scene_folders:
            npy_path = os.path.join(scene_path, "esdf_2d.npy")
            meta_path = os.path.join(scene_path, "map_meta.npy")
            
            if os.path.exists(npy_path):
                # 原始: [Width, Height] -> [x, y]
                esdf_np = np.load(npy_path).astype(np.float32)
                
               
                # 原版 3D: (X, Y, Z) -> Permute -> (Z, Y, X)
                # 目标 2D: (X, Y)    -> Transpose -> (Y, X)
                # 这样 grid_sample(x, y) 里的 x 对应 最后一维(X), y 对应 倒数第二维(Y)
                map_tensor = th.from_numpy(esdf_np.T).unsqueeze(0) # [1, H, W]
               
                
                map_list.append(map_tensor)
                meta_list.append(th.from_numpy(np.load(meta_path).astype(np.float32)))
            else:
                raise FileNotFoundError(f"Missing map in {scene_path}")

        all_maps = th.stack(map_list).to(self.device)
        all_metas = th.stack(meta_list).to(self.device)
        return all_maps, all_metas

    def forward(self, Df, Dp, map_id):

        batch_size = Dp.shape[0]
        L = self._L.unsqueeze(0).expand(batch_size, -1, -1)
        zeros = th.zeros_like(Df[:, :1, ...])
        Df_3d = th.cat([Df, zeros], dim=1) 
        Dp_3d = th.cat([Dp, zeros], dim=1)
        coe = self.get_coefficient_from_derivative(Dp_3d, Df_3d, L)

        dt = self.sgm_time / self.eval_points
        t_list = th.linspace(dt, self.sgm_time, self.eval_points, device=self.device)
        t_list = t_list.view(1, -1, 1).expand(batch_size, -1, -1)
        pos_coe = self.get_position_from_coeff(coe, t_list)
        
        # 计算 Cost
        cost, dist = self.get_distance_cost_batch(pos_coe, map_id, Df)
        min_dist = dist.min().item()
        max_dist = dist.max().item()
        avg_cost = cost.mean().item()
        print(f"[DEBUG] Dist Range: [{min_dist:.4f}, {max_dist:.4f}] | Avg Cost: {avg_cost:.4f}")

        if not self.truncate_cost:
            cost_colli = (cost * dt).sum(dim=-1)
        else:
            N = dist.shape[1]
            mask = dist <= 0 
            arange = th.arange(N, device=self.device).unsqueeze(0).expand(batch_size, N)
            index = th.where(mask, arange, th.full_like(arange, N - 1))
            first_colli_idx = index.min(dim=1).values
            valid_mask = arange <= first_colli_idx.unsqueeze(1)
            masked_cost = cost * valid_mask
            valid_count = first_colli_idx + 1
            cost_colli = self.sgm_time * masked_cost.sum(dim=-1) / valid_count

        return cost_colli

    def get_distance_cost_batch(self, pos, map_id, Df=None):
        # 在 get_distance_cost_batch 里
    # 打印起点坐标和当前地图的 Origin
        B, N, _ = pos.shape
        batch_maps = self.all_maps[map_id.long()] 
        batch_metas = self.all_metas[map_id.long()] 
        
        ox = batch_metas[:, 0].view(B, 1, 1)
        oy = batch_metas[:, 1].view(B, 1, 1)
        res = batch_metas[:, 2].view(B, 1, 1)

        # batch_maps: [B, 1, H, W] -> 对应 [Batch, 1, Y_dim, X_dim]
        H, W = batch_maps.shape[2], batch_maps.shape[3]

        px = pos[:, :, 0].unsqueeze(1) 
        py = pos[:, :, 1].unsqueeze(1)
        
        # 原版逻辑: grid = (pos - min) / span * 2 - 1
        # align_corners=False (PyTorch默认) 的标准归一化:
        norm_x = (px - ox) / (W * res) * 2.0 - 1.0
        norm_y = (py - oy) / (H * res) * 2.0 - 1.0


        grid = th.stack([norm_x, norm_y], dim=-1) # [B, 1, N, 2]
       
       

        # 采样
        # padding_mode='border': 防止出界后 Cost 突变导致梯度消失
        dists = F.grid_sample(batch_maps, grid, align_corners=False, padding_mode='border')
        dists = dists.view(B, N)
        
        cost_obstacle = self.cost_function(dists)
        return cost_obstacle, dists

    def cost_function(self, d):
        return th.exp(-(d - self.d0) / self.r)

 
    def get_coefficient_from_derivative(self, Dp, Df, L):   
        coefficient = th.zeros(Dp.shape[0], 18, device=self.device)
        for i in range(3):
            d = th.cat([Df[:, i, :], Dp[:, i, :]], dim=1).unsqueeze(-1)
            coe = (L @ d).squeeze()
            coefficient[:, 6 * i: 6 * (i + 1)] = coe
        return coefficient

    def get_position_from_coeff(self, coe, t):
        t_power = th.stack([th.ones_like(t), t, t ** 2, t ** 3, t ** 4, t ** 5], dim=-1).squeeze(-2)
        coe_x, coe_y, coe_z = coe[:, 0: 6], coe[:, 6:12], coe[:, 12:18]
        x = th.sum(t_power * coe_x.unsqueeze(1), dim=-1)
        y = th.sum(t_power * coe_y.unsqueeze(1), dim=-1)
        return th.stack([x, y], dim=-1)
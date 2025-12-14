import os
import glob
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from scipy.stats import binned_statistic_2d
from tqdm import tqdm

# ================= 配置区域 (无人车专用) =================
# DATASET_ROOT = "/home/zxx/YOPO-Sim/TrainingData"
DATASET_ROOT = "/home/zxx/maybe_trush"
OUTPUT_FILENAME = "esdf_2d.npy"  # 输出文件名

RESOLUTION = 0.1  

# 无人车障碍物判定高度 (相对于地面)
# 忽略地面的小起伏和树根
ROBOT_REL_HEIGHT_MIN = 0.05 
ROBOT_REL_HEIGHT_MAX = 1.3  

# 边距 (Padding)在地图边缘多留几米，防止ESDF边界截断
MAP_PADDING = 1.0 

def process_scene(scene_path):
    terrain_path = os.path.join(scene_path, "terrain.ply")
    tree_path = os.path.join(scene_path, "tree.ply")
    
    if not os.path.exists(terrain_path):
        return

    # 1. 读取点云
    pcd_terrain = o3d.io.read_point_cloud(terrain_path)
    pcd_tree = o3d.io.read_point_cloud(tree_path)
    
    pts_terrain = np.asarray(pcd_terrain.points)
    pts_tree = np.asarray(pcd_tree.points)

    if len(pts_terrain) == 0: return

    # --- 调试：检查坐标轴 ---
    # Unity 通常是 Y 轴向上。如果你的 Z 值范围很小，而 Y 值范围很大，说明轴搞反了。
    # 这里我们打印一下范围，请你在控制台确认一下高度轴是否正确。
    print(f"\nProcessing {os.path.basename(scene_path)}")
    print(f"Terrain Bounds: X[{pts_terrain[:,0].min():.1f}, {pts_terrain[:,0].max():.1f}] "
          f"Y[{pts_terrain[:,1].min():.1f}, {pts_terrain[:,1].max():.1f}] "
          f"Z[{pts_terrain[:,2].min():.1f}, {pts_terrain[:,2].max():.1f}]")
    
    # 假设 Z 是高度 (如果发现 Y 范围很小，请交换 Y 和 Z)
    # X, Y 是平面
    
    # 2. 确定地图范围
    min_x, min_y = pts_terrain[:, 0].min(), pts_terrain[:, 1].min()
    max_x, max_y = pts_terrain[:, 0].max(), pts_terrain[:, 1].max()
    
    origin_x = min_x - MAP_PADDING
    origin_y = min_y - MAP_PADDING
    end_x = max_x + MAP_PADDING
    end_y = max_y + MAP_PADDING
    print(origin_x, origin_y, end_x, end_y)
    width = int((end_x - origin_x) / RESOLUTION)
    height = int((end_y - origin_y) / RESOLUTION)
    
    # 3. 构建 DEM (只用来计算相对高度)
    stat = binned_statistic_2d(
        pts_terrain[:, 0], pts_terrain[:, 1], values=pts_terrain[:, 2],
        statistic='mean', 
        bins=[width, height], 
        range=[[origin_x, end_x], [origin_y, end_y]]
    )
    
    # 用最低点填充 NaN，防止计算相对高度时出错
    # 【改动】不要根据 NaN 设置障碍物！只用它来做基准面。
    ground_elevation_map = np.nan_to_num(stat.statistic, nan=np.nanmin(pts_terrain[:, 2]))

    # --- 初始化 Occupancy Grid ---
    occupancy_grid = np.zeros((width, height), dtype=bool)

    # 4. 过滤障碍物 (树木)
    if len(pts_tree) > 0:
        tree_idx_x = ((pts_tree[:, 0] - origin_x) / RESOLUTION).astype(int)
        tree_idx_y = ((pts_tree[:, 1] - origin_y) / RESOLUTION).astype(int)
        
        # 边界检查
        valid_mask = (tree_idx_x >= 0) & (tree_idx_x < width) & \
                     (tree_idx_y >= 0) & (tree_idx_y < height)
        
        tree_idx_x = tree_idx_x[valid_mask]
        tree_idx_y = tree_idx_y[valid_mask]
        valid_tree_z = pts_tree[valid_mask, 2] # 假设 Z 是高度
        
        # 查表获取地面高度
        ground_z = ground_elevation_map[tree_idx_x, tree_idx_y]
        relative_height = valid_tree_z - ground_z
        
        # 高度过滤
        obstacle_mask = (relative_height > ROBOT_REL_HEIGHT_MIN) & \
                        (relative_height < ROBOT_REL_HEIGHT_MAX)
        
        final_obs_x = tree_idx_x[obstacle_mask]
        final_obs_y = tree_idx_y[obstacle_mask]
        
        # 标记障碍物
        occupancy_grid[final_obs_x, final_obs_y] = True

    # 5. 计算 ESDF
    if np.sum(occupancy_grid) > 0:
        # ~occupancy_grid 中: 障碍物=False(0), 空地=True(1)
        # edt 计算离 0 最近的距离 -> 即计算离障碍物最近的距离
        dist_outside = distance_transform_edt(~occupancy_grid)
        
        # occupancy_grid 中: 障碍物=True(1), 空地=False(0)
        # edt 计算离 0 最近的距离 -> 即计算离空地最近的距离
        dist_inside = distance_transform_edt(occupancy_grid)
        
        esdf_2d = (dist_outside - dist_inside) * RESOLUTION

    # 6. 保存数据
    # 直接保存 [Width, Height] 矩阵，对应 [x, y]
    np.save(os.path.join(scene_path, OUTPUT_FILENAME), esdf_2d.astype(np.float32))
    
    # 保存元数据: [ox, oy, res, w, h]
    np.save(os.path.join(scene_path, "map_meta.npy"), 
            np.array([origin_x, origin_y, RESOLUTION, width, height]))
    print(origin_x, origin_y, RESOLUTION, width, height)
    # 7. 可视化 (Debug)
    # vmin=0 (蓝), vmax=3.0 (红)
    # 如果还是全蓝，说明 max 真的很小；如果全红，说明没有障碍物
    plt.imsave(os.path.join(scene_path, "debug_esdf.png"), esdf_2d.T, cmap='jet', vmin=-1.0, vmax=3.0)    
    # 打印一些统计信息帮助调试
    print(f"  Shape: {esdf_2d.shape}")
    print(f"  Range: [{esdf_2d.min():.2f}m, {esdf_2d.max():.2f}m]")
    if esdf_2d.min() >= 0:
        print("  [WARNING] No negative values found! Check logic if obstacles exist.")
    else:
        print("  [OK] Signed Distance Field generated (contains negative values).")

def main():
    search_pattern = os.path.join(DATASET_ROOT, "Scene_*")
    scene_folders = sorted(glob.glob(search_pattern))
    
    for folder in tqdm(scene_folders):
        process_scene(folder)

if __name__ == "__main__":
    main()
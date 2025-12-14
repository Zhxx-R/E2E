import os
import glob
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from data.learning_dataset import RD4Dataset
from policy.rgb_yopo_network import CMCL_YOPO_Network

# ----------------------------------------------------------
# 配置参数
# ----------------------------------------------------------
# 【修改点】你的真实数据根目录
DATA_ROOT_PATH = "/home/zxx/YOPO-Sim/TrainingData"  

EPOCHS = 1000
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
IMAGE_SIZE_W = 160 
IMAGE_SIZE_H = 90
IMAGE_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 【重要】开启 OpenCV 对 EXR 格式的支持
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


def alignment_loss_mse(z_rgb: torch.Tensor, z_depth: torch.Tensor):
    """
    最小化两个特征向量之间的欧氏距离
    """
    return nn.MSELoss()(z_rgb, z_depth)


def train_cmcl_stage():
    dataset = RD4Dataset(DATA_ROOT_PATH, size_h = IMAGE_SIZE_H,size_w = IMAGE_SIZE_W)
    
    if len(dataset) == 0:
        print("错误: 未找到任何数据。请检查：")
        print("1. 路径是否正确")
        print("2. Scene_XX/Textures/ 下是否有 rgb_0.png 和 *depth*.exr 文件")
        return

    # num_workers > 0 利用多进程加速文件读取 (建议设为 CPU 核心数的一半，如 4 或 8)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    
    # 2. 初始化网络
    model = CMCL_YOPO_Network().to(DEVICE)
    
    # 3. 优化器：只训练两个 Backbone
    optimizer = optim.Adam(
        list(model.rgb_backbone.parameters()) + list(model.depth_backbone.parameters()), 
        lr=LEARNING_RATE
    )
    
    model.train()
    
    # 4. 训练循环
    for epoch in range(EPOCHS):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for rgb, depth in pbar:
            rgb = rgb.to(DEVICE)
            depth = depth.to(DEVICE)
            
            # --- 前向传播 ---
            # 使用 forward_cmcl 接口，它返回归一化后的特征 z
            z_rgb, z_depth = model.forward_cmcl(rgb, depth)
            
            # --- 计算损失 ---
            loss = alignment_loss_mse(z_rgb, z_depth)
            
            # --- 反向传播 ---
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(dataloader)
        print(f"\nEpoch {epoch+1} 完成. 平均对齐损失: {avg_loss:.4f}")

    # 5. 保存权重
    # 保存名字加上了 cmcl 前缀，方便区分
    torch.save(model.rgb_backbone.state_dict(), "cmcl_rgb_backbone.pth")
    torch.save(model.depth_backbone.state_dict(), "cmcl_depth_backbone.pth")
    print("训练结束，权重已保存为 cmcl_rgb_backbone.pth 和 cmcl_depth_backbone.pth")

if __name__ == "__main__":
    train_cmcl_stage()
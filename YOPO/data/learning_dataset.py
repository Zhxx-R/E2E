import glob
import os
import cv2
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
#32 160 -> 1 * 5
class RD4Dataset(Dataset):
    def __init__(self, root_dir, size_h=32, size_w=160):
        self.root_dir = root_dir
        self.height = size_h
        self.width = size_w
        self.data_pairs = []

        # 扫描文件
        self._scan_dataset()
        
        # 初始化 Transform
        self.transform = RD4ImageTransform(height=self.height, width=self.width)

    def _scan_dataset(self):
        print(f"正在扫描数据: {self.root_dir} ...")
        scene_dirs = sorted(glob.glob(os.path.join(self.root_dir, "Scene_*")))

        for scene_path in scene_dirs:
            tex_dir = os.path.join(scene_path, "Textures")
            if not os.path.isdir(tex_dir):
                continue

            # 找到 RGB 文件并排序
            rgb_files = glob.glob(os.path.join(tex_dir, "rgb_*.png"))
            # 按文件名中的数字排序，防止 rgb_10 排在 rgb_2 前面
            rgb_files.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))

            for rgb_path in rgb_files:
                filename = os.path.basename(rgb_path)
                file_id = filename.replace("rgb_", "").replace(".png", "")
                
                # 匹配 EXR
                depth_name = f"depth_{file_id}.exr"
                depth_path = os.path.join(tex_dir, depth_name)

                if os.path.exists(depth_path):
                    self.data_pairs.append((rgb_path, depth_path))
                # 这里的 else print 建议注释掉，否则如果只有部分数据会刷屏
                # else:
                #    print(f"警告: 缺失 {depth_name}")

        print(f"扫描完成。共加载 {len(self.data_pairs)} 对图像。")
        if len(self.data_pairs) == 0:
            raise RuntimeError("数据集为空！")

    def __len__(self):
        # [修正] 之前写成了 self.rgb_imgs (不存在)
        return len(self.data_pairs)

    def __getitem__(self, idx):
        rgb_path, depth_path = self.data_pairs[idx]

        # --- RGB 读取 ---
        rgb_image = cv2.imread(rgb_path)
        if rgb_image is None:
            raise ValueError(f"无法读取图片: {rgb_path}")
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB) # [H, W, 3] uint8
        # ToTensor() 会自动把 [H, W, C] (0-255) 转成 [C, H, W] (0.0-1.0)
        rgb_tensor = transforms.ToTensor()(rgb_image)

        # --- Depth 读取 [重要修正] ---
        depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

        if depth_image is None:
            raise ValueError(f"无法读取深度图: {depth_path}")

        if len(depth_image.shape) == 3:
            depth_image = depth_image[:, :, 0] # 取第一通道
        # depth_image: [H, W] (numpy)
        # 复制成 3 通道: [H, W, 3]  change？？？？？axis
        depth_3ch = np.stack([depth_image] * 3, axis=-1) 

        depth_tensor = transforms.ToTensor()(depth_3ch)
        depth_tensor = depth_tensor.float()
       

        # --- Transform ---
        if self.transform:
            rgb_tensor, depth_tensor = self.transform(rgb_tensor, depth_tensor)

        return rgb_tensor, depth_tensor

class RD4ImageTransform:
    def __init__(self, height, width):
        self.height = height # 96
        self.width = width   # 160
        
        # 颜色增强 (仅针对 RGB Tensor)
        self.color_jitter = transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5
        )

    def __call__(self, rgb_tensor, depth_tensor):
        """
        输入:
            rgb_tensor:   [3, 32, 160] (Tensor)
            depth_tensor: [3, 32, 160] (Tensor)
        """
        
        # RGB 使用双线性插值 (平滑)
        # Depth 使用最近邻插值 (保持深度值真实性，防止边缘插值错误)
        rgb_crop = TF.resize(rgb_tensor, [self.height, self.width], interpolation=transforms.InterpolationMode.BILINEAR, antialias=True)
        depth_crop = TF.resize(depth_tensor, [self.height, self.width], interpolation=transforms.InterpolationMode.NEAREST)
        # 1. Padding: 90 -> 96
        # TF.pad 对 Tensor 同样有效。参数 (left, top, right, bottom)
        # rgb_tensor = TF.pad(rgb_tensor, (0, 3, 0, 3), fill=0)
        # depth_tensor = TF.pad(depth_tensor, (0, 3, 0, 3), fill=0)
        # # 2. 同步随机裁剪
        # # get_params 也支持 Tensor 输入
        # i, j, h, w = transforms.RandomResizedCrop.get_params(
        #     rgb_tensor, scale=[0.5, 1.0], ratio=[1.5, 1.8]
        # )

        # # 对 Tensor 进行裁剪缩放
        # # antialias=True 是新版 PyTorch 推荐的，防止缩放锯齿
        # rgb_crop = TF.resized_crop(rgb_tensor, i, j, h, w, (self.height, self.width), antialias=True)
        # depth_crop = TF.resized_crop(depth_tensor, i, j, h, w, (self.height, self.width), antialias=True)

        # 3. 同步水平翻转
        if random.random() > 0.5:
            rgb_crop = TF.hflip(rgb_crop)
            depth_crop = TF.hflip(depth_crop)

        # 4. 颜色增强 (仅 RGB)
        # ColorJitter 可以直接作用于 [3, H, W] 的 Tensor
        if random.random() < 0.8:
            rgb_crop = self.color_jitter(rgb_crop)
        
        # 灰度化 (保持3通道)
        if random.random() < 0.3:
            rgb_crop = TF.to_grayscale(rgb_crop, num_output_channels=3)

        # 5. 高斯模糊
        if random.random() < 0.5:
            sigma = random.uniform(0.1, 2.0)
            k_size = int(0.1 * self.height) 
            if k_size % 2 == 0: k_size += 1
            k_size = max(k_size, 1)

            rgb_crop = TF.gaussian_blur(rgb_crop, [k_size, k_size], [sigma, sigma])
            # Depth 同步模糊
            depth_crop = TF.gaussian_blur(depth_crop, [k_size, k_size], [sigma, sigma])

        return rgb_crop, depth_crop

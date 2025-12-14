# policy/models/backbone.py
import torch
import torch.nn as nn
from policy.models.resnet import resnet18
# 32 160
class YopoBackbone(nn.Module):
    def __init__(self, output_dim: int, input_channels: int = 3):
        super(YopoBackbone, self).__init__()
       
        self.cnn = resnet18(pretrained=False) 
        #rgb 3
        self.cnn.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # 1*1 cnn : 512 -> 64
        self.cnn.output_layer = torch.nn.Conv2d(512, output_dim, kernel_size=1, stride=1, padding=0, bias=False)
       

    def forward(self, x):
        # 输入: [Batch, 3, 32, 160]
        # 输出: output: [B, 64, 1, 5]
        return self.cnn(x)
    





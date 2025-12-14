import torch.nn as nn


class OneYopoHead(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim = 256, num_anchors = 5):
        super(OneYopoHead, self).__init__()
        # 特征提取层：保持卷积结构，处理每个 Anchor 的局部特征
        # 输入: [B, 70, 1, 5]
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_dim, 256, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=1, stride=1),
            nn.ReLU(),
            # nn.Conv2d(256, output_dim, kernel_size=1, stride=1)
        )

        flatten_dim = hidden_dim * num_anchors  #[B, 1280]
        
        self.decision_mlp = nn.Sequential(
            # nn.Linear(hidden_dim, flatten_dim),
            nn.Linear(flatten_dim, 512),
            nn.ReLU(),
            nn.Linear(512, hidden_dim), #256 * 5 , 256
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)    # 256, 7   [B,7]
        )


    def forward(self, x):
        # x: [B, 70, 1, 5]
        x = self.conv_layers(x)  # [B, 256, 1, 5]
        x = x.flatten(start_dim=1) # [B, 1280]

        return self.decision_mlp(x) # -> [B, 7]
    


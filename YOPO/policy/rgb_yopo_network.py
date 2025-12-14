import torch
from torch import nn
import torch.nn.functional as F

from policy.models.RD_backbone import YopoBackbone 
from policy.models.head import YopoHead
from policy.models.Onehead import OneYopoHead
from policy.state_transform import StateTransform 

class CMCL_YOPO_Network(nn.Module):
    def __init__(
            self,
            observation_dim=6,   # vx, vy, ax, ay, gx, gy
            output_dim=7,        # x_pva, y_pva,  score
            hidden_state=64,     # 内部特征通道数 (Policy Head的输入通道)
            feature_dim=64,     # Backbone 输出通道数
    ):
        super(CMCL_YOPO_Network, self).__init__()
        self.state_transform = StateTransform()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_backbone = nn.Sequential()
        in_channels = feature_dim + observation_dim
        # backbone_output: [B, 64, 1, 5]
        self.rgb_backbone = YopoBackbone(output_dim=feature_dim, input_channels=3)
        self.depth_backbone = YopoBackbone(output_dim=feature_dim, input_channels=3) 
        # self.yopo_head = YopoHead(hidden_state + observation_dim, output_dim)
        self.one_yopo_head = OneYopoHead(hidden_state + observation_dim, output_dim)  # 70, 7 -> [B, 7]


    def forward_cmcl(self, rgb: torch.Tensor, depth: torch.Tensor):
        """
        [阶段 A] 感知对齐训练接口
        """
        # grid_rgb: [B, 64, 1, 5] depth已做3通道：data
        grid_rgb = self.rgb_backbone(rgb)
        grid_depth = self.depth_backbone(depth)
        
        # 展平网格
        # [B, 64, 1, 5] -> [B, 64, 5] -> [B, 5, 64] -> [B*5, 64]
        flat_rgb = grid_rgb.flatten(2).permute(0, 2, 1).reshape(-1, grid_rgb.size(1))
        flat_depth = grid_depth.flatten(2).permute(0, 2, 1).reshape(-1, grid_depth.size(1))
        # 归一化 
        z_rgb_norm = F.normalize(flat_rgb, dim=1)
        z_depth_norm = F.normalize(flat_depth, dim=1)

        return z_rgb_norm, z_depth_norm


    def _run_policy_head(self, visual_grid: torch.Tensor, obs: torch.Tensor):
        """
        visual_grid: [B, 64, 1, 5] 
        obs:         [B, 6]
        """
        B, C, H, W = visual_grid.shape
        # obs: [B, 6] (vx, vy, ax, ay, gx, gy) in body frame
        obs = obs.view(B, -1, 1, 1).expand(-1, -1, H, W)
        obs = self.state_backbone(obs)
        # 拼接: [B, 70, 1, 5]
        combined_feat = torch.cat([visual_grid, obs ], dim=1)
        # 预测: [B, 7]
        output = self.one_yopo_head(combined_feat)
        
       
        # [B, 6, 1, 5]
        # endstate = torch.tanh(output[:, :6, :, :]) # [batch, 6, vertical_num, horizon_num]
        # score = F.softplus(output[:, 6, :, :])
        # 使用 OneYopoHead 出的，[B , 7] 
        endstate = torch.tanh(output[:, :6]) # [batch, 6, vertical_num, horizon_num]
        score = F.softplus(output[:, 6])
        return endstate, score

    
    def forward_policy(self, rgb_img: torch.Tensor, obs: torch.Tensor, is_training: bool = True):
        """
        [阶段 B] 策略训练接口 (Policy Training)
        """
        # 修改backbone  depth
        # 决定是否启用梯度 (取决于外部调用，用于实现全训练或冻结)
        if not is_training:
            with torch.no_grad(): 
                z_rgb = self.depth_backbone(rgb_img)
        else:
            z_rgb = self.depth_backbone(rgb_img) 
            
        # 2. 核心预测 (使用 CMCL 特征)
        # output : 
        endstate_pred, score_pred = self._run_policy_head(z_rgb, obs)

        # 返回预测结果，外部计算 YOPO Loss (指导梯度)
        return endstate_pred, score_pred


    # @torch.no_grad()
    def inference(self, rgb: torch.Tensor, obs: torch.Tensor):   
        """
        [部署接口] 推理模式，只运行 RGB 分支，无梯度计算
        """
        obs = self.state_transform.normalize_obs(obs)
        endstate_pred, score_pred = self.forward_policy(rgb, obs) #[B,6] output:[b,6] [b,1]
        endstate = self.state_transform.pred_to_endstate_2d(endstate_pred)
        return endstate, score_pred
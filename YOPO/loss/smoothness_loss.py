import torch.nn as nn
import torch as th


class SmoothnessLoss(nn.Module):
    def __init__(self, R):
        super(SmoothnessLoss, self).__init__()
        self._R = R

    def forward(self, Df, Dp):
        """
        Args:
            Dp: decision parameters: (batch_size, 2, 3) → [px, vx, ax; py, vy, ay]
            Df: fixed parameters: (batch_size, 2, 3) → [px, vx, ax; py, vy, ay]
        Returns:
            cost_smooth: (batch_size) → smoothness loss
        """
        R = self._R.unsqueeze(0).expand(Dp.shape[0], -1, -1)   # 2* 3 *3
        D_all = th.cat([Df, Dp], dim=2)  # dx, dy will be rows  (batch, 2, 6)

        # Reshape for matmul: (batch, 6, 1)
        dx, dy = D_all[:, 0].unsqueeze(2), D_all[:, 1].unsqueeze(2)

        # Compute smoothness loss: dxᵀ R dx + ...
        # (batch, 1, 6) 
        cost_smooth = dx.transpose(1, 2) @ R @ dx + dy.transpose(1, 2) @ R @ dy

        return cost_smooth.squeeze()
import os
import torch.nn as nn
import torch as th
from ruamel.yaml import YAML


class GuidanceLoss(nn.Module):
    def __init__(self):
        super(GuidanceLoss, self).__init__()
        base_dir = os.path.dirname(os.path.abspath(__file__))
        cfg = YAML().load(open(os.path.join(base_dir, "../config/traj_opt.yaml"), 'r'))
        self.max_similarity = 2.0 * cfg['radio_range']

    def forward(self, Df, Dp, goal):
        """
        Args:
            Dp: decision parameters: (batch_size, 3, 3) → [px, vx, ax; py, vy, ay; pz, vz, az]
            Df: fixed parameters: (batch_size, 3, 3) → [px, vx, ax; py, vy, ay; pz, vz, az]
            goal: (batch_size, 3)
        Returns:
            similarity: (batch_size) → guidance loss
        """
        cur_pos = Df[:, :, 0]
        end_pos = Dp[:, :, 0]

        traj_dir = end_pos - cur_pos  # [B, 3]
        goal_dir = goal - cur_pos  # [B, 3]
        goal_dir = goal_dir / (goal_dir.norm(dim=1, keepdim=True) + 1e-8)  # [B, 3]

        similarity = self.max_similarity - th.sum(traj_dir * goal_dir, dim=1)  # [B]

        return similarity

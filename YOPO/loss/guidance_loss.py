import os
import torch.nn as nn
import torch as th
import torch.nn.functional as F
from ruamel.yaml import YAML


class GuidanceLoss(nn.Module):
    def __init__(self):
        super(GuidanceLoss, self).__init__()
        base_dir = os.path.dirname(os.path.abspath(__file__))
        cfg = YAML().load(open(os.path.join(base_dir, "../config/traj_opt.yaml"), 'r'))
        self.goal_length = 2.0 * cfg['radio_range']

    def forward(self, Df, Dp, goal):
        """
        Args:
            Dp: decision parameters: (batch_size, 3, 3) → [px, vx, ax; py, vy, ay; pz, vz, az]
            Df: fixed parameters: (batch_size, 3, 3) → [px, vx, ax; py, vy, ay; pz, vz, az]
            goal: (batch_size, 3)
        Returns:
            similarity: (batch_size) → guidance loss

        GuidanceLoss: Projection length of the trajectory onto the goal direction: higher cosine similarity and longer trajectory are preferred
        """
        cur_pos = Df[:, :, 0]
        end_pos = Dp[:, :, 0]

        traj_dir = end_pos - cur_pos  # [B, 3]
        goal_dir = goal - cur_pos  # [B, 3]

        guidance_loss = self.terminal_aware_similarity_loss(traj_dir, goal_dir)
        return guidance_loss

    def distance_loss(self, traj_dir, goal_dir):
        """
        Returns:
            l1_distance: (batch_size) → guidance loss

        L1Loss: L1 distance (same scale as the similarity loss) to the normalized goal (for numerical stability).
                closer to the goal is preferred.
        Better near the goal, but slightly inferior to the similarity cost in general situations.
        """
        l1_distance = F.smooth_l1_loss(traj_dir, goal_dir, reduction='none')  # shape: (B, 3)
        l1_distance = l1_distance.sum(dim=1)  # (B)
        return l1_distance

    def similarity_loss(self, traj_dir, goal_dir):
        """
        Returns:
            similarity: (batch_size) → guidance loss

        SimilarityLoss: Projection length of the trajectory onto the goal direction:
                        higher cosine similarity and longer trajectory are preferred.
        Performs better in general by allowing longer lateral avoidance without slowing down, but less precise near the goal.
        """
        goal_length = goal_dir.norm(dim=1)

        goal_dir_norm = goal_dir / (goal_length.unsqueeze(1) + 1e-8)  # [B, 3]
        similarity = th.sum(traj_dir * goal_dir_norm, dim=1)  # [B]

        similarity_loss = th.abs(goal_length - similarity)
        return similarity_loss

    def terminal_aware_similarity_loss(self, traj_dir, goal_dir):
        """
        Returns:
            similarity: (batch_size) → guidance loss

        SimilarityLoss: Projection length of the trajectory onto the goal direction:
                        higher cosine similarity and longer trajectory are preferred.
        Reduce perpendicular deviation when approaching the goal, and apply dynamic weighting to ensure loss continuity.
        """
        goal_length = goal_dir.norm(dim=1)

        goal_dir_norm = goal_dir / (goal_length.unsqueeze(1) + 1e-8)  # [B, 3]
        similarity = th.sum(traj_dir * goal_dir_norm, dim=1)  # [B]

        traj_dir_proj = similarity.unsqueeze(1) * goal_dir_norm  # [B, 3]
        perp_component = (traj_dir - traj_dir_proj).norm(dim=1)  # [B]

        perp_weight = 2 * (self.goal_length - goal_length) / self.goal_length   # [B]
        perp_weight[perp_weight.abs() < 1e-4] = 0.0  # eliminate tiny numerical errors for stability
        similarity_loss = th.abs(goal_length - similarity) + perp_weight * perp_component
        return similarity_loss
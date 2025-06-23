import torch.nn as nn
import torch as th


class GuidanceLoss(nn.Module):
    def __init__(self):
        super(GuidanceLoss, self).__init__()

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

        goal_length = goal_dir.norm(dim=1)

        goal_dir_norm = goal_dir / (goal_length.unsqueeze(1) + 1e-8)  # [B, 3]
        similarity = th.sum(traj_dir * goal_dir_norm, dim=1)  # [B]

        similarity_loss = th.abs(goal_length - similarity)
        return similarity_loss

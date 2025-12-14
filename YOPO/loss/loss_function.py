import math
import torch as th
import torch.nn as nn
from config.config import cfg
from loss.safety_loss import SafetyLoss
from loss.smoothness_loss import SmoothnessLoss
from loss.guidance_loss import GuidanceLoss


class YOPOLoss(nn.Module):
    def __init__(self):
        """
        Compute the cost: including smoothness, safety, guidance, goal cost, etc.
        Currently, keeping multi-segment polynomial support (not yet verified), but only using a single-segment polynomial (m = 1) for now.
        dp: decision parameters
        df: fixed parameters
        """
        super(YOPOLoss, self).__init__()
        self.sgm_time = cfg["sgm_time"]
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        self._C, self._B, self._L, self._R = self.qp_generation()
        self._R = self._R.to(self.device)
        self._L = self._L.to(self.device)
        vel_scale = cfg["vel_max_train"] / 1.0
        self.smoothness_weight = cfg["ws"]
        self.safety_weight = cfg["wc"]
        self.goal_weight = cfg["wg"]
        self.denormalize_weight(vel_scale)
        self.smoothness_loss = SmoothnessLoss(self._R)
        self.safety_loss = SafetyLoss(self._L)
        self.goal_loss = GuidanceLoss()
        print("------ Actual Loss ------")
        print(f"| {'smooth':<12} = {self.smoothness_weight:6.4f} |")
        print(f"| {'safety':<12} = {self.safety_weight:6.4f} |")
        print(f"| {'goal':<12} = {self.goal_weight:6.4f} |")
        print("-------------------------")

    def qp_generation(self):
        # 论文中的映射矩阵 d = A @ c：c=[c0​,c1​,c2​,c3​,c4​,c5​]T
        A = th.zeros((6, 6))
        for i in range(3):
            A[2 * i, i] = math.factorial(i)
            for j in range(i, 6):
                A[2 * i + 1, j] = math.factorial(j) / math.factorial(j - i) * (self.sgm_time ** (j - i))
        #位置 p(t)=c0​+c1​t+c2​t2+c3​t3+c4​t4+c5​t5

        # 速度 v(t)=c1​+2c2​t+3c3​t2+4c4​t3+5c5​t4

        # 加速 a(t)=2c2​+6c3​t+12c4​t2+20c5​t3

        # tensor([[ 1.,  0.,  0.,  0.,  0.,  0.],                   p(0)
                # [ 1.,  1.,  1.,  1.,  1.,  1.],   * t(j-i)        p(t)
                # [ 0.,  1.,  0.,  0.,  0.,  0.],
                # [ 0.,  1.,  2.,  3.,  4.,  5.],
                # [ 0.,  0.,  2.,  0.,  0.,  0.],
                # [ 0.,  0.,  2.,  6., 12., 20.]])

        # tensor([[  0.,   0.,   0.,   0.,   0.,   0.],
        #         [  0.,   0.,   0.,   0.,   0.,   0.],
        #         [  0.,   0.,   0.,   0.,   0.,   0.],   * t(j+i-5) 
        #         [  0.,   0.,   0.,  36.,  72., 120.],
        #         [  0.,   0.,   0.,  72., 192., 360.],
        #         [  0.,   0.,   0., 120., 360., 720.]])
        # H海森矩阵，论文中的矩阵Q # Minimum Jerk 的积分项公式
        H = th.zeros((6, 6))
        for i in range(3, 6):
            for j in range(3, 6):
                H[i, j] = i * (i - 1) * (i - 2) * j * (j - 1) * (j - 2) / (i + j - 5) * (self.sgm_time ** (i + j - 5))

        return self.stack_opt_dep(A, H)

    def stack_opt_dep(self, A, Q):
        # 矩阵 Ct (和 _C)：调序矩阵  dsorted​=[p0​,v0​,a0​,pT​,vT​,aT​]
        Ct = th.zeros((6, 6))
        # 在特定位置赋值为1 行 and 列
        Ct[[0, 2, 4, 1, 3, 5], [0, 1, 2, 3, 4, 5]] = 1

        _C = th.transpose(Ct, 0, 1)

        B = th.inverse(A)

        B_T = th.transpose(B, 0, 1)

        _L = B @ Ct  #c = _L * d sort  _L - > M-1C -> safety_LP

        _R = _C @ (B_T) @ Q @ B @ Ct     #loss: CT M-T Q M-1 C   左右* d = j_smooth

        return _C, B, _L, _R

    def denormalize_weight(self, vel_scale):
        """
        Denormalize the cost weight to ensure consistency across different speeds to simplify parameter tuning.
        smoothness cost: time integral of jerk² is used as a smoothness cost.
                         If the speed is scaled by n, the cost is scaled by n⁵ (because jerk * n⁶ and time * 1/n).
        safety cost:     time integral of the distance from trajectory to obstacles.
                         If the speed is scaled by n, the cost is scaled by 1/n (because time * 1/n).
        goal cost:       projection of the trajectory onto goal direction.
                         Independent of speed.
        """
        self.smoothness_weight = self.smoothness_weight / vel_scale ** 5
        self.safety_weight = self.safety_weight * vel_scale
        self.goal_weight = self.goal_weight

    def forward(self, state, prediction, goal, map_id):
        """
        Args:

            prediction: (batch_size, 3, 2) → [px, py; vx, vy; ax, ay] in world frame
            state: (batch_size, 3, 2) → [px, py; vx, vy; ax, ay] in world frame
            map_id: (batch_size) which ESDF map to query

        Returns:
            cost: (batch_size) → weighted cost
        """
        # Fixed part: initial pos, vel, acc → (batch_size, 2，3) [px, vx, ax; py, vy, ay]
        Df = state.permute(0, 2, 1)

        # Decision parameters (local frame) → (batch_size, 2， 3) [px, vx, ax; py, vy, ay] 交换维度
        Dp = prediction.permute(0, 2, 1)
       
        smoothness_cost = th.tensor(0.0, device=self.device, requires_grad=True)
        safety_cost = th.tensor(0.0, device=self.device, requires_grad=True)
        goal_cost = th.tensor(0.0, device=self.device, requires_grad=True)

        if self.smoothness_weight > 0:
            smoothness_cost = self.smoothness_loss(Df, Dp)
        if self.safety_weight > 0:
            safety_cost = self.safety_loss(Df, Dp, map_id)
        if self.goal_weight > 0:
            goal_cost = self.goal_loss(Df, Dp, goal)

        return self.smoothness_weight * smoothness_cost, self.safety_weight * safety_cost, self.goal_weight * goal_cost
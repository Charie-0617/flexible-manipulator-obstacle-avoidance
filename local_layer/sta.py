import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatioTemporalAttentionGRU(nn.Module):
    """
    时空注意力 GRU 网络。

    通过空间注意力自适应关注高风险障碍物，使用 GRU 建模障碍物运动
    的时间演化，并行输出预测位移序列和碰撞风险评分。
    """

    def __init__(
        self,
        hidden_dim=128,
        obs_feat_dim=8,
        num_layers=1,
        future_steps=10
    ):
        super().__init__()

        self.future_steps = future_steps

        # 障碍物特征编码。
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 空间注意力打分。
        self.attn_score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        # 时序 GRU 编码。
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        # 前向轨迹解码器（三层 MLP）。
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, future_steps * 2)
        )

        # 风险预测头。
        self.risk_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, robot_pos, robot_vel, obs_pos, obs_vel, goal_rel):
        """
        Parameters
        ----------
        robot_pos : [B, T, 2]  末端执行器历史位置。
        robot_vel : [B, T, 2]  末端执行器历史速度。
        obs_pos   : [B, T, N, 2]  障碍物历史位置。
        obs_vel   : [B, T, N, 2]  障碍物历史速度。
        goal_rel  : [B, T, 2]  当前帧相对目标向量。

        Returns
        -------
        future : [B, future_steps, 2]  预测的末端位置。
        risk   : [B, 1]  碰撞风险概率。
        """
        B, T, N, _ = obs_pos.shape

        # 障碍物相对状态。
        rel_pos = obs_pos - robot_pos.unsqueeze(2)
        rel_vel = obs_vel - robot_vel.unsqueeze(2)

        distance = torch.norm(rel_pos, dim=-1, keepdim=True)
        approaching_speed = torch.sum(rel_pos * rel_vel, dim=-1, keepdim=True) \
                             / (distance + 1e-6)

        goal_dir = goal_rel.unsqueeze(2).expand(-1, -1, N, -1)

        # 拼接障碍物特征。
        obs_feat = torch.cat([
            rel_pos, rel_vel, distance, approaching_speed, goal_dir
        ], dim=-1)

        # 空间注意力加权融合。
        h = self.obs_encoder(obs_feat)
        score = self.attn_score(h).squeeze(-1)
        attn = F.softmax(score, dim=-1)

        context = torch.sum(attn.unsqueeze(-1) * h, dim=2)

        # 时序 GRU 编码。
        gru_out, h_n = self.gru(context)

        final_state = gru_out[:, -1]

        future = self.decoder(final_state)
        future = future.view(B, self.future_steps, 2)

        risk = self.risk_head(final_state)

        return future, risk

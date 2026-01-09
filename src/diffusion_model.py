"""
基于GNN+注意力+扩散模型的分子对齐模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class SinusoidalPositionEmbeddings(nn.Module):
    """正弦位置编码用于时间步"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        """
        time: (B,) 时间步
        返回: (B, dim) 时间编码
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ImprovedGNN(nn.Module):
    """改进的GNN编码器（继承自improved_model）"""
    def __init__(self, node_dim, hidden_dim, num_layers, num_heads, dropout=0.1):
        super().__init__()
        self.node_dim = node_dim

        # 原子类型嵌入（118种元素）
        self.atom_embedding = nn.Embedding(120, node_dim)

        # 坐标编码
        self.coord_encoder = nn.Sequential(
            nn.Linear(3, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, node_dim)
        )

        # GNN层
        self.gnn_layers = nn.ModuleList([
            GNNLayer(node_dim, hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, coords, atom_types, mask):
        """
        coords: (B, N, 3)
        atom_types: (B, N) - 原子序数
        mask: (B, N) - 有效原子mask
        """
        # 嵌入
        atom_feat = self.atom_embedding(atom_types)  # (B, N, D)
        coord_feat = self.coord_encoder(coords)      # (B, N, D)

        # 组合特征
        x = atom_feat + coord_feat

        # GNN编码
        for layer in self.gnn_layers:
            x = layer(x, mask)

        return x


class GNNLayer(nn.Module):
    """GNN层（继承自improved_model）"""
    def __init__(self, node_dim, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = node_dim // num_heads

        assert node_dim % num_heads == 0

        # 多头自注意力
        self.q_proj = nn.Linear(node_dim, node_dim)
        self.k_proj = nn.Linear(node_dim, node_dim)
        self.v_proj = nn.Linear(node_dim, node_dim)
        self.out_proj = nn.Linear(node_dim, node_dim)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, node_dim)
        )

        self.norm1 = nn.LayerNorm(node_dim)
        self.norm2 = nn.LayerNorm(node_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        B, N, D = x.shape

        # 多头自注意力
        residual = x
        x = self.norm1(x)

        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        mask_2d = mask.unsqueeze(1).unsqueeze(2) * mask.unsqueeze(1).unsqueeze(3)
        attn = attn.masked_fill(~mask_2d.bool(), float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.out_proj(out)

        x = residual + self.dropout(out)

        # FFN
        residual = x
        x = self.norm2(x)
        x = residual + self.dropout(self.ffn(x))

        return x


class CrossAttentionLayer(nn.Module):
    """交叉注意力层（继承自improved_model）"""
    def __init__(self, node_dim, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = node_dim // num_heads

        assert node_dim % num_heads == 0

        self.q_proj_a = nn.Linear(node_dim, node_dim)
        self.k_proj_b = nn.Linear(node_dim, node_dim)

        self.q_proj_b = nn.Linear(node_dim, node_dim)
        self.k_proj_a = nn.Linear(node_dim, node_dim)

    def forward(self, feat_a, feat_b, mask_a, mask_b):
        B, N, D = feat_a.shape

        q_a = self.q_proj_a(feat_a).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k_b = self.k_proj_b(feat_b).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn_ab = torch.matmul(q_a, k_b.transpose(-2, -1)) / math.sqrt(self.head_dim)

        mask_ab = mask_a.unsqueeze(1).unsqueeze(3) * mask_b.unsqueeze(1).unsqueeze(2)
        # 使用0而不是-inf填充无效位置，避免产生nan
        attn_ab = attn_ab * mask_ab

        q_b = self.q_proj_b(feat_b).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k_a = self.k_proj_a(feat_a).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn_ba = torch.matmul(q_b, k_a.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # 使用0而不是-inf填充无效位置，避免产生nan
        attn_ba = attn_ba * mask_ab.transpose(-2, -1)

        attn = (attn_ab + attn_ba.transpose(-2, -1)) / 2.0
        attn_mean = attn.mean(dim=1)

        return attn_mean


class DiffusionAlignmentModel(nn.Module):
    """扩散模型版本的分子对齐模型"""
    def __init__(self, node_dim=768, hidden_dim=2048, num_gnn_layers=12,
                 num_cross_layers=8, num_heads=16, dropout=0.1,
                 num_timesteps=1000):
        super().__init__()

        self.node_dim = node_dim
        self.num_timesteps = num_timesteps

        # 编码器（复用improved架构）
        self.encoder = ImprovedGNN(node_dim, hidden_dim, num_gnn_layers, num_heads, dropout)

        # 交叉注意力层
        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionLayer(node_dim, num_heads, dropout)
            for _ in range(num_cross_layers)
        ])

        # 时间步编码
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(node_dim),
            nn.Linear(node_dim, node_dim * 4),
            nn.GELU(),
            nn.Linear(node_dim * 4, node_dim)
        )

        # 噪声预测头（预测添加的噪声）
        # 输入: feat_A + feat_B + condition_scores + noisy_alignment = 2*node_dim + 2
        self.noise_head = nn.Sequential(
            nn.Linear(node_dim * 2 + 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        # 设置扩散schedule（线性）
        self.register_buffer('betas', self._linear_beta_schedule(num_timesteps))
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - self.alphas_cumprod))

    def _linear_beta_schedule(self, timesteps, beta_start=1e-4, beta_end=0.02):
        """线性beta schedule"""
        return torch.linspace(beta_start, beta_end, timesteps)

    def forward(self, coords_A, types_A, coords_B, types_B, mask, noisy_alignment, timesteps):
        """
        训练时的前向传播

        coords_A, coords_B: (B, N, 3)
        types_A, types_B: (B, N)
        mask: (B, N)
        noisy_alignment: (B, N, N) 加噪后的对齐矩阵
        timesteps: (B,) 时间步

        返回: (B, N, N) 预测的噪声
        """
        B, N = coords_A.shape[0], coords_A.shape[1]

        # 编码分子
        feat_A = self.encoder(coords_A, types_A, mask)  # (B, N, D)
        feat_B = self.encoder(coords_B, types_B, mask)  # (B, N, D)

        # 时间步编码
        t_emb = self.time_embed(timesteps)  # (B, D)

        # 将时间步编码加到特征上
        feat_A = feat_A + t_emb.unsqueeze(1)
        feat_B = feat_B + t_emb.unsqueeze(1)

        # 交叉注意力得到条件信息
        cross_scores = []
        for layer in self.cross_attention_layers:
            cross_score = layer(feat_A, feat_B, mask, mask)
            cross_scores.append(cross_score)

        # 平均所有交叉注意力分数作为条件
        condition_scores = torch.stack(cross_scores, dim=0).mean(dim=0)  # (B, N, N)

        # 构建成对特征：结合条件和噪声对齐矩阵
        # 扩展特征到N×N
        feat_A_expanded = feat_A.unsqueeze(2).expand(B, N, N, self.node_dim)  # (B, N, N, D)
        feat_B_expanded = feat_B.unsqueeze(1).expand(B, N, N, self.node_dim)  # (B, N, N, D)

        # 将交叉注意力分数和噪声对齐矩阵作为额外特征
        # condition_scores: (B, N, N) -> (B, N, N, 1)
        # noisy_alignment: (B, N, N) -> (B, N, N, 1)
        condition_feat = torch.stack([condition_scores, noisy_alignment], dim=-1)  # (B, N, N, 2)

        # 拼接所有特征: 分子特征 + 条件信息 + 噪声对齐
        pairwise_feat = torch.cat([feat_A_expanded, feat_B_expanded, condition_feat], dim=-1)  # (B, N, N, 2D+2)

        # 预测噪声
        noise_pred = self.noise_head(pairwise_feat).squeeze(-1)  # (B, N, N)

        return noise_pred

    def add_noise(self, alignment, timesteps):
        """
        前向扩散：向对齐矩阵添加噪声

        alignment: (B, N, N) 干净的对齐矩阵
        timesteps: (B,) 时间步

        返回: noisy_alignment, noise
        """
        B, N, _ = alignment.shape
        device = alignment.device

        # 生成噪声
        noise = torch.randn_like(alignment)

        # 根据时间步添加噪声
        sqrt_alpha_t = self.sqrt_alphas_cumprod[timesteps].view(B, 1, 1)
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[timesteps].view(B, 1, 1)

        noisy_alignment = sqrt_alpha_t * alignment + sqrt_one_minus_alpha_t * noise

        return noisy_alignment, noise

    @torch.no_grad()
    def sample(self, coords_A, types_A, coords_B, types_B, mask, num_steps=50):
        """
        采样：从噪声逐步去噪得到对齐矩阵（DDIM采样）

        num_steps: 采样步数（可以少于训练步数）
        """
        B, N = coords_A.shape[0], coords_A.shape[1]
        device = coords_A.device

        # 从纯噪声开始
        alignment = torch.randn(B, N, N, device=device)

        # DDIM采样步骤
        timesteps = torch.linspace(self.num_timesteps - 1, 0, num_steps, dtype=torch.long, device=device)

        for i, t in enumerate(timesteps):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)

            # 预测噪声
            noise_pred = self.forward(coords_A, types_A, coords_B, types_B, mask, alignment, t_batch)

            # DDIM更新
            alpha_t = self.alphas_cumprod[t]

            if i < len(timesteps) - 1:
                alpha_t_prev = self.alphas_cumprod[timesteps[i + 1]]
            else:
                alpha_t_prev = torch.tensor(1.0, device=device)

            # 预测x0
            pred_x0 = (alignment - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)

            # 更新
            if i < len(timesteps) - 1:
                alignment = torch.sqrt(alpha_t_prev) * pred_x0 + torch.sqrt(1 - alpha_t_prev) * noise_pred
            else:
                alignment = pred_x0

        return alignment


if __name__ == '__main__':
    # 测试模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = DiffusionAlignmentModel(
        node_dim=768,
        hidden_dim=2048,
        num_gnn_layers=12,
        num_cross_layers=8,
        num_heads=16,
        num_timesteps=1000
    ).to(device)

    # 测试数据
    B, N = 4, 50
    coords_A = torch.randn(B, N, 3).to(device)
    types_A = torch.randint(1, 10, (B, N)).to(device)
    coords_B = torch.randn(B, N, 3).to(device)
    types_B = torch.randint(1, 10, (B, N)).to(device)
    mask = torch.ones(B, N).to(device)
    mask[:, 30:] = 0

    alignment_gt = torch.eye(N).unsqueeze(0).repeat(B, 1, 1).to(device)

    # 测试训练
    timesteps = torch.randint(0, 1000, (B,)).to(device)
    noisy_alignment, noise = model.add_noise(alignment_gt, timesteps)
    noise_pred = model(coords_A, types_A, coords_B, types_B, mask, noisy_alignment, timesteps)

    print(f"Noise pred shape: {noise_pred.shape}")
    print(f"Loss: {F.mse_loss(noise_pred, noise).item():.4f}")

    # 测试采样
    sampled_alignment = model.sample(coords_A, types_A, coords_B, types_B, mask, num_steps=50)
    print(f"Sampled alignment shape: {sampled_alignment.shape}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params / 1e6:.2f}M")

import torch
import torch.nn.functional as F


def temporal_info_nce(z_a: torch.Tensor, z_pos: torch.Tensor,
                      z_neg_noise: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    """L_temporal (式2):
    锚点: 原序列第 t 个时间片(patch)的表示 r_{i,t}
    正样本: 频率自适应采样序列同一时间片的表示
    负样本: 同一序列其他时间片的表示 + 加噪序列同时间片的表示

    z_a, z_pos, z_neg_noise: (B, P, D)  (P 个 patch)
    """
    a = F.normalize(z_a, dim=-1)
    p = F.normalize(z_pos, dim=-1)
    n = F.normalize(z_neg_noise, dim=-1)

    sim_pos = (a * p).sum(-1, keepdim=True) / tau           # (B, P, 1)
    sim_noise = (a * n).sum(-1, keepdim=True) / tau         # (B, P, 1)
    sim_intra = torch.bmm(a, a.transpose(1, 2)) / tau       # (B, P, P) 同序列各时间片
    P = a.shape[1]
    eye = torch.eye(P, dtype=torch.bool, device=a.device)
    sim_intra = sim_intra.masked_fill(eye, float("-inf"))   # 去掉自身

    denom = torch.cat([sim_pos, sim_noise, sim_intra], dim=-1)  # (B, P, P+2)
    log_prob = sim_pos.squeeze(-1) - torch.logsumexp(denom, dim=-1)
    return -log_prob.mean()


def instance_info_nce(v: torch.Tensor, labels: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    """L_instance (式3):
    锚点: 公司 i 的整条序列表示 r_i (时间维 mean-pooling 后的投影)
    正样本: 同一行业其他公司的表示; 负样本: 不同行业公司的表示
    多正样本 InfoNCE: -log( sum_pos exp(s) / sum_{j!=i} exp(s) )

    v: (B, D), labels: (B,) 行业标签 (编码为整数)
    """
    v = F.normalize(v, dim=-1)
    sim = torch.mm(v, v.t()) / tau                          # (B, B)
    B = v.shape[0]
    self_mask = torch.eye(B, dtype=torch.bool, device=v.device)
    pos_mask = (labels.view(-1, 1) == labels.view(1, -1)) & ~self_mask

    sim_all = sim.masked_fill(self_mask, float("-inf"))
    log_denom = torch.logsumexp(sim_all, dim=-1)            # (B,)
    sim_pos = sim.masked_fill(~pos_mask, float("-inf"))
    log_num = torch.logsumexp(sim_pos, dim=-1)              # (B,) 无正样本时为 -inf

    has_pos = pos_mask.any(dim=-1)
    if not has_pos.any():
        return v.new_zeros(())
    return (log_denom[has_pos] - log_num[has_pos]).mean()

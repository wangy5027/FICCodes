import torch


def frequency_adaptive_sampling(x: torch.Tensor, keep_ratio: float = 0.3) -> torch.Tensor:
    """频率自适应采样: 沿最后一维(时间)保留幅值最大的 top-k 个频率成分 (含直流)。

    x: (B, L) 或 (B, C, L) 原始序列
    return: 同形状重构的低噪声序列 (实数)
    """
    L = x.shape[-1]
    X = torch.fft.rfft(x, dim=-1)                       # (B, L//2+1) 复数
    n_freq = X.shape[-1]
    k = max(2, int(round(n_freq * keep_ratio)))         # 至少保留直流+一个主频
    mag = X.abs()
    # 直流分量(均值)强制保留: 将其幅值设为最大, 使其必入 top-k
    mag = mag.clone()
    mag[..., 0] = mag.max(dim=-1).values + 1.0
    idx = mag.topk(k, dim=-1).indices                   # (..., k)
    mask = torch.zeros_like(X, dtype=torch.bool)
    mask.scatter_(-1, idx, True)
    X_filtered = X * mask
    return torch.fft.irfft(X_filtered, n=L, dim=-1)


def _unit_std_lognormal(shape, device):
    """生成零均值、单位 std 的对数正态噪声"""
    z = torch.randn(shape, device=device)
    s = torch.exp(z)
    return (s - s.mean()) / s.std().clamp_min(1e-8)


def mix_noise_injection(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """混合噪声注入: 每个样本(及通道)随机选 {高斯, 均匀, 对数正态} 之一,
    噪声按最后一维(时间)的 std 缩放 (乘 scale) 后叠加 (与原序列同尺度)。

    x: (B, L) 或 (B, C, L)
    return: 同形状加噪序列
    """
    std = x.std(dim=-1, keepdim=True).clamp_min(1e-6)      # (..., 1)
    g = torch.randn_like(x)                                # 高斯 N(0,1)
    u = (torch.rand_like(x) * 2 - 1) * (3 ** 0.5)          # 均匀, 单位方差
    ln = _unit_std_lognormal(tuple(x.shape), x.device)     # 对数正态, 单位方差
    choice = torch.randint(0, 3, (*x.shape[:-1], 1), device=x.device)
    noise = torch.where(choice == 0, g, torch.where(choice == 1, u, ln))
    return x + noise * std * scale

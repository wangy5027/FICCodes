import torch
import torch.nn as nn

PATCH_LEN = 3
D_MODEL = 64
N_LAYERS = 2
N_HEADS = 4
FFN = 128
DROPOUT = 0.1
N_COV = 8


class PatchEncoder(nn.Module):
    def __init__(self, seq_len: int, patch_len: int = PATCH_LEN, d_model: int = D_MODEL,
                 n_layers: int = N_LAYERS, n_heads: int = N_HEADS, ffn: int = FFN,
                 dropout: float = DROPOUT, in_channels: int = 1):
        super().__init__()
        assert seq_len % patch_len == 0
        self.n_patch = seq_len // patch_len
        self.patch_len = patch_len
        self.in_channels = in_channels
        # 多变量 (ETT, C=7): 每个 patch 取全部通道展平 (p*C 维) 后线性投影 ——
        # 实现最简单且通道信息完整共享; returns (C=1) 时退化为原单变量形式
        self.patch_embed = nn.Linear(patch_len * in_channels, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.n_patch, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, ffn, dropout, batch_first=True,
            activation="gelu")  # 内部 Norm 为 LayerNorm (见模块 docstring 说明)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L) 或 (B, L, C) -> z: (B, P, D)"""
        if x.dim() == 2:
            x = x.unsqueeze(-1)                                   # (B, L, 1)
        patches = x.unfold(1, self.patch_len, self.patch_len)     # (B, P, C, p)
        patches = patches.permute(0, 1, 3, 2).flatten(2)          # (B, P, p*C)
        z = self.patch_embed(patches) + self.pos_emb
        return self.encoder(z)


class QueryDecoder(nn.Module):
    def __init__(self, horizon: int, d_model: int = D_MODEL, n_layers: int = N_LAYERS,
                 n_heads: int = N_HEADS, ffn: int = FFN, dropout: float = DROPOUT,
                 n_cov: int = N_COV, out_channels: int = 1):
        super().__init__()
        self.out_channels = out_channels
        self.queries = nn.Parameter(torch.zeros(1, horizon, d_model))  # H 个可学习 query
        self.cov_proj = nn.Linear(n_cov, d_model)                       # 协变量 c -> D
        layer = nn.TransformerDecoderLayer(
            d_model, n_heads, ffn, dropout, batch_first=True, activation="gelu")
        self.decoder = nn.TransformerDecoder(layer, n_layers)
        self.head = nn.Linear(d_model, out_channels)   # ETT: C=7; returns: C=1
        nn.init.trunc_normal_(self.queries, std=0.02)

    def forward(self, z: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
        """z: (B, P, D), cov: (B, n_cov) -> pred: (B, H) 或 (B, H, C)"""
        B = z.shape[0]
        q = self.queries.expand(B, -1, -1) + self.cov_proj(cov).unsqueeze(1)
        out = self.decoder(q, z)                    # cross-attention 到 encoder 输出
        pred = self.head(out)                       # (B, H, C)
        return pred.squeeze(-1) if self.out_channels == 1 else pred


class TimesFNP(nn.Module):
    """完整模型: encoder-decoder 预测 + 对比学习表示头。
    use_cl 仅控制训练时是否计算对比损失, 网络结构不变。"""

    def __init__(self, seq_len: int = 36, horizon: int = 1, use_cl: bool = True, **kw):
        super().__init__()
        self.use_cl = use_cl
        self.encoder = PatchEncoder(seq_len, **{k: v for k, v in kw.items()
                                                if k in ("patch_len", "d_model", "n_layers",
                                                         "n_heads", "ffn", "dropout",
                                                         "in_channels")})
        self.decoder = QueryDecoder(horizon, **{k: v for k, v in kw.items()
                                                if k in ("d_model", "n_layers", "n_heads",
                                                         "ffn", "dropout", "n_cov",
                                                         "out_channels")})
        # 对比学习投影头 (per-patch 投影, 实例表示由 patch 表示 mean-pooling 得到)
        self.proj = nn.Sequential(
            nn.Linear(kw.get("d_model", D_MODEL), kw.get("d_model", D_MODEL)),
            nn.ReLU(),
            nn.Linear(kw.get("d_model", D_MODEL), kw.get("d_model", D_MODEL)),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def project(self, z: torch.Tensor) -> torch.Tensor:
        """per-patch 投影表示, 用于 L_temporal; (B, P, D)"""
        return self.proj(z)

    def instance_repr(self, z: torch.Tensor) -> torch.Tensor:
        """整条序列表示: 时间维 mean-pooling + 投影, 用于 L_instance; (B, D)"""
        return self.proj(z.mean(dim=1))

    def forward(self, x: torch.Tensor, cov: torch.Tensor):
        """x: (B, L), cov: (B, n_cov) -> (pred (B,H), z (B,P,D))"""
        z = self.encode(x)
        pred = self.decoder(z, cov)
        return pred, z

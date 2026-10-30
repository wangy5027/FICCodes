import os

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.normpath(os.path.join(HERE, "..", "data", "raw"))

N_CH = 7


class ETTH1Data:
    def __init__(self, horizon: int = 96, seq_len: int = 96, device: str = "cuda"):
        df = pd.read_csv(os.path.join(RAW, "ETTh1.csv"))
        vals = df.iloc[:, 1:].values.astype(np.float32)          # (T, 7)
        self.T = len(vals)
        self.L = seq_len
        self.H = horizon

        # 标准协议: 12/4/4 个月切分 (每月按 30 天计), border 与 Informer 等一致
        n_train = 12 * 30 * 24
        n_val = 4 * 30 * 24
        n_test = 4 * 30 * 24
        borders = [(0, n_train),
                   (n_train - self.L, n_train + n_val),
                   (n_train + n_val - self.L, n_train + n_val + n_test)]

        # train 段统计量标准化 (按通道)
        mean = vals[:n_train].mean(axis=0)
        std = vals[:n_train].std(axis=0) + 1e-8
        z = torch.from_numpy((vals - mean) / std).to(device)     # (T, 7)
        self.data = z
        self.mean = torch.from_numpy(mean).to(device)
        self.std = torch.from_numpy(std).to(device)
        self.device = device
        self.N = N_CH

        # 每个 split 内合法的窗口起点 s: 输入 z[s:s+L], 目标 z[s+L:s+L+H]
        self.s_ranges = {
            name: range(b0, b1 - self.L - self.H + 1)
            for name, (b0, b1) in zip(("train", "val", "test"), borders)
        }

    def _fetch(self, starts):
        s = torch.tensor(starts, device=self.device)
        idx_in = s.view(-1, 1) + torch.arange(self.L, device=self.device)   # (B, L)
        idx_out = s.view(-1, 1) + self.L + torch.arange(self.H, device=self.device)
        x = self.data[idx_in]                                    # (B, L, 7)
        y = self.data[idx_out]                                   # (B, H, 7)
        cov = x.new_zeros(len(starts), 8)      # 无协变量, 零向量占位 (见模块 docstring)
        lab = torch.zeros(len(starts), dtype=torch.long, device=self.device)  # 无行业标签
        return x, y, cov, lab

    def train_batch(self, batch_size: int = 32):
        starts = np.random.choice(list(self.s_ranges["train"]), size=batch_size,
                                  replace=len(self.s_ranges["train"]) < batch_size)
        return self._fetch(starts.tolist())

    def eval_batches(self, split: str, batch_size: int = 512):
        starts = list(self.s_ranges[split])
        for i in range(0, len(starts), batch_size):
            yield self._fetch(starts[i:i + batch_size])

    def n_windows(self, split: str) -> int:
        return len(self.s_ranges[split])

    def destandardize(self, y: torch.Tensor) -> torch.Tensor:
        return y * self.std + self.mean

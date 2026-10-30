import json
import os

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.normpath(os.path.join(HERE, "..", "data", "processed"))

TARGETS = ("ret", "rank", "path_winz", "path_minmax", "path_z")


class GKXPanelData:
    def __init__(self, horizon: int, lookback: int = 36, device: str = "cuda",
                 scale: str = "winz", target: str = "ret"):
        d = np.load(os.path.join(PROC, "gkx_panel.npz"), allow_pickle=True)
        with open(os.path.join(PROC, "gkx_meta.json"), encoding="utf-8") as f:
            self.meta = json.load(f)
        raw = d["returns"]                                        # (T, N)
        self.macro = torch.from_numpy(d["macro"]).to(device)      # (T, 8)
        self.dates = d["dates"].tolist()
        self.first_idx = d["first_idx"]
        self.last_idx = d["last_idx"]
        self.T, self.N = raw.shape
        self.H = horizon
        self.L = lookback
        self.device = device
        self.scale = scale
        self.target = target
        assert scale in ("raw", "pct", "z", "winz") and target in TARGETS

        # 切分: train 1957-1989 / val 1990-1999 / test 2000-2016 (按目标起始月)
        train_end = max(i for i, dt in enumerate(self.dates)
                        if dt <= self.meta["train_end"])
        val_end = max(i for i, dt in enumerate(self.dates)
                      if dt <= self.meta["val_end"])
        self.t_ranges = {
            "train": range(self.L, train_end + 1),
            "val": range(train_end + 1, val_end + 1 - self.H + 1),
            "test": range(val_end + 1, self.T - self.H + 1),
        }
        tr_last = list(self.t_ranges["train"])[-1]

        # 活跃期掩码 (T, N)
        ti = np.arange(self.T)[:, None]
        active = (ti >= self.first_idx[None, :]) & (ti <= self.last_idx[None, :])

        # ---- target 变换 (panel 级, 在加窗前完成) ----
        if target == "rank":
            # 截面 rank -> [-1,1], 只在当月活跃股票内排名 (GKX 特征标准做法)
            ranked = np.zeros_like(raw)
            for t in range(self.T):
                m = active[t]
                v = raw[t, m]
                n = len(v)
                if n > 1:
                    order = v.argsort().argsort()          # 0..n-1
                    ranked[t, m] = 2.0 * order / (n - 1) - 1.0
            self.panel = torch.from_numpy(ranked.astype(np.float32))
        elif target in ("path_winz", "path_minmax", "path_z"):
            self.panel = torch.from_numpy(
                np.cumsum(raw, axis=0).astype(np.float32))  # 价格路径 cumsum
        else:
            self.panel = torch.from_numpy(raw)

        # 全局统计量 (z: ret 用收益; path_z: 用路径; 均取 train 段活跃观测)
        if target == "path_z":
            src = self.panel.numpy()
        else:
            src = raw
        vals = src[:tr_last + 1]
        act = active[:tr_last + 1]
        m = vals[act]
        self.g_mean = float(m.mean())
        self.g_std = float(m.std()) + 1e-8

        # 行业标签 (sic2 直接用 2 位码分组); 缺失行业(-1)的股票各占一个独立标签,
        # 避免它们在 instance-wise 对比中构成伪正样本对
        sic2 = d["sic2"].astype(str)
        uniq = {s: i for i, s in enumerate(sorted(set(sic2) - {"-1"}))}
        next_id = len(uniq)
        lab_list = []
        for s in sic2:
            if s == "-1":
                lab_list.append(next_id)
                next_id += 1
            else:
                lab_list.append(uniq[s])
        self.labels_np = np.array(lab_list)
        self.labels = torch.from_numpy(self.labels_np).to(device)
        ind2stocks = {}
        for i, s in enumerate(sic2):
            if s != "-1":
                ind2stocks.setdefault(s, []).append(i)
        self.strata = [np.array(v) for v in ind2stocks.values() if len(v) >= 2]
        # 每个行业 -> 每股活跃区间 (供快速筛选)
        self._eligible_cache = {}

    def _eligible(self, t: int) -> np.ndarray:
        """t 月有完整输入窗口+目标区间的股票"""
        if t not in self._eligible_cache:
            ok = (self.first_idx <= t - self.L) & (self.last_idx >= t + self.H - 1)
            self._eligible_cache[t] = np.nonzero(ok)[0]
            if len(self._eligible_cache) > 800:     # 简单容量控制
                self._eligible_cache.clear()
        return self._eligible_cache[t]

    def _fetch(self, stocks, t):
        s_np = np.asarray(stocks)
        s = torch.from_numpy(s_np).to(self.device)
        x = self.panel[t - self.L:t, s_np].t().contiguous().to(self.device)  # (B, L)
        y = self.panel[t:t + self.H, s_np].t().contiguous().to(self.device)  # (B, H)
        mu = sig = None
        if self.target == "rank":
            pass                                    # 已在 [-1,1], 不再缩放
        elif self.target == "path_winz":
            mu = x.mean(dim=1, keepdim=True)
            sig = x.std(dim=1, keepdim=True).clamp_min(0.02)   # 防僵尸股平坦窗口
            x = (x - mu) / sig
            y = (y - mu) / sig
        elif self.target == "path_minmax":
            mu = x.amin(dim=1, keepdim=True)                   # 窗口 min
            sig = (x.amax(dim=1, keepdim=True) - mu).clamp_min(0.02)
            x = (x - mu) / sig                               # [0, 1]
            y = (y - mu) / sig                               # 可超出 [0,1]
        elif self.target == "path_z":
            x = (x - self.g_mean) / self.g_std
            y = (y - self.g_mean) / self.g_std
        elif self.scale == "pct":
            x, y = x * 100.0, y * 100.0
        elif self.scale == "z":
            x = (x - self.g_mean) / self.g_std
            y = (y - self.g_mean) / self.g_std
        elif self.scale == "winz":
            mu = x.mean(dim=1, keepdim=True)                       # (B, 1)
            # σ 下限 0.02 (约窗口 σ 的 1% 分位以下): 防止近零收益僵尸股
            # (停牌期填 0 导致窗口近乎平坦) 归一化后目标爆炸
            sig = x.std(dim=1, keepdim=True).clamp_min(0.02)
            x = (x - mu) / sig
            y = (y - mu) / sig
        cov = self.macro[t - 1].unsqueeze(0).expand(len(stocks), -1)  # (B, 8)
        lab = self.labels[s]
        return x, y, cov, lab, mu, sig

    def train_batch(self, batch_size: int = 64):
        t = int(np.random.choice(list(self.t_ranges["train"])))
        elig = self._eligible(t)
        # sic2 分层: 8 行业 x 8 股, 保证同行业配对
        lab_e = self.labels_np[elig]
        stocks = []
        per_ind = 8
        uniq = np.unique(lab_e)
        # 只取在 elig 中 >=2 只的行业
        cnt = {u: (lab_e == u).sum() for u in uniq}
        ok_ind = np.array([u for u in uniq if cnt[u] >= 2])
        n_ind = max(1, batch_size // per_ind)
        chosen = np.random.choice(ok_ind, size=min(n_ind, len(ok_ind)), replace=False)
        for u in chosen:
            members = elig[lab_e == u]
            k = min(per_ind, len(members))
            stocks.extend(np.random.choice(members, size=k, replace=False).tolist())
        while len(stocks) < batch_size:
            stocks.append(int(elig[np.random.randint(len(elig))]))
        return self._fetch(stocks[:batch_size], t)

    def eval_batches(self, split: str, batch_size: int = 1024, max_windows: int = 30000):
        """遍历 split 窗口; 窗口过多时在时间×股票两个维度等距子抽样到 ~max_windows"""
        ts = list(self.t_ranges[split])
        avg_s = float(np.mean([len(self._eligible(t))
                               for t in ts[: min(5, len(ts))]]))
        total = max(1.0, len(ts) * avg_s)
        ratio = total / max_windows
        import math
        stride_t = max(1, int(round(math.sqrt(ratio))))
        stride_s = max(1, int(round(ratio / stride_t)))
        for t in ts[::stride_t]:
            elig = self._eligible(t)[::stride_s]
            for i in range(0, len(elig), batch_size):
                yield self._fetch(elig[i:i + batch_size].tolist(), t)

    def n_windows(self, split: str) -> int:
        ts = list(self.t_ranges[split])
        return sum(len(self._eligible(t)) for t in ts)

    def denorm_batch(self, pred, y, mu, sig):
        """还原到原始超额收益口径 (仅 ret 口径有意义; 变换口径下原样返回,
        指标在各自变换空间计算)"""
        if self.target != "ret":
            return pred, y
        if self.scale == "winz":
            return pred * sig + mu, y * sig + mu
        if self.scale == "pct":
            return pred / 100.0, y / 100.0
        if self.scale == "z":
            return pred * self.g_std + self.g_mean, y * self.g_std + self.g_mean
        return pred, y

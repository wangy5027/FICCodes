import json
import os
import zipfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")
OUT = os.path.join(HERE, "processed")
os.makedirs(OUT, exist_ok=True)

START = "1957-03-01"
END = "2016-12-31"
MISS_THRESH = 0.30
TRAIN_END = "1989-12-31"
VAL_END = "1999-12-31"

MACRO_COLS = ["dp", "ep", "bm", "ntis", "tbl", "tms", "dfy", "svar"]  # Welch-Goyal 8 变量


def main():
    # ---- 低内存读取: 只取 4 列 ----
    zpath = os.path.join(RAW, "gkx_datashare.zip")
    with zipfile.ZipFile(zpath) as zf:
        name = [n for n in zf.namelist() if n.endswith(".csv")][0]
        with zf.open(name) as f:
            df = pd.read_csv(f, usecols=["permno", "DATE", "mom1m", "sic2"],
                             dtype={"permno": np.int32, "mom1m": np.float32,
                                    "sic2": "Int64"})
    print("rows:", len(df))
    df["month"] = pd.to_datetime(df["DATE"], format="%Y%m%d") + pd.offsets.MonthEnd(0)
    df = df[(df["month"] >= START) & (df["month"] <= END)]
    df = df.drop_duplicates(subset=["month", "permno"], keep="first")

    # ---- 面板: month x permno ----
    panel = df.pivot(index="month", columns="permno", values="mom1m").sort_index()
    print("panel:", panel.shape)
    sic2 = df.drop_duplicates("permno").set_index("permno")["sic2"]

    # ---- 无风险利率: 超额收益 = mom1m - rf ----
    rf = pd.read_csv(os.path.join(RAW, "riskfree.csv"), parse_dates=["date"])
    rf.index = rf["date"] + pd.offsets.MonthEnd(0)
    rf = rf["rf"].reindex(panel.index).ffill().bfill()
    excess = panel.sub(rf, axis=0)          # NaN 保持 NaN

    # ---- 活跃期与缺失处理 ----
    valid = excess.notna().values           # (T, N)
    first_idx = valid.argmax(axis=0)        # 首个有效月
    last_idx = len(valid) - 1 - valid[::-1].argmax(axis=0)
    span = (last_idx - first_idx + 1).astype(np.float64)
    n_valid = valid.sum(axis=0)
    has_data = n_valid > 0
    miss_in_span = np.where(has_data, 1.0 - n_valid / np.maximum(span, 1), 1.0)
    keep = has_data & (miss_in_span <= MISS_THRESH)
    cols = excess.columns[keep]
    print(f"kept {keep.sum()} / {len(keep)} stocks (active-span missing<={MISS_THRESH:.0%})")

    excess = excess[cols].fillna(0.0)       # 活跃期内残余缺失按 0 填充 (见 docstring)
    first_idx = first_idx[keep]
    last_idx = last_idx[keep]
    sic2 = sic2.reindex(cols).fillna(-1).astype(int).astype(str)

    # ---- 宏观协变量: Welch-Goyal 8 变量, train 段统计量标准化 ----
    mac = pd.read_csv(os.path.join(RAW, "macro_gw.csv"), parse_dates=["date"])
    mac.index = mac["date"] + pd.offsets.MonthEnd(0)
    mac = mac[MACRO_COLS].reindex(panel.index).ffill().bfill()
    train_mask = panel.index <= TRAIN_END
    m_mean = mac.values[train_mask].mean(axis=0)
    m_std = mac.values[train_mask].std(axis=0) + 1e-8
    mac_z = (mac - m_mean) / m_std

    dates = panel.index
    np.savez_compressed(
        os.path.join(OUT, "gkx_panel.npz"),
        returns=excess.values.astype(np.float32),        # (T, N) 超额收益, 已填充
        first_idx=first_idx.astype(np.int32),            # 每股活跃期首月
        last_idx=last_idx.astype(np.int32),
        macro=mac_z.values.astype(np.float32),           # (T, 8)
        dates=np.array([d.strftime("%Y-%m-%d") for d in dates]),
        permnos=excess.columns.values.astype(np.int32),
        sic2=sic2.values,
    )
    meta = {
        "miss_thresh": MISS_THRESH, "fill": "0 (active-span residual)",
        "train_end": TRAIN_END, "val_end": VAL_END, "test_end": END,
        "macro_cols": MACRO_COLS, "macro_mean": m_mean.tolist(),
        "macro_std": m_std.tolist(),
        "n_stocks": int(len(cols)), "n_months": int(len(dates)),
        "date_start": str(dates[0].date()), "date_end": str(dates[-1].date()),
        "note": "mom1m == CRSP monthly total return (Green-Hand-Zhang def); "
                "excess = mom1m - rf; industry = 2-digit SIC",
    }
    with open(os.path.join(OUT, "gkx_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    v = excess.values[train_mask]
    print(f"excess ret: months {dates[0].date()}~{dates[-1].date()}, "
          f"mean {v.mean():.5f}, std {v.std():.5f} (train段)")
    top = pd.Series(sic2.values).value_counts().head(5)
    print("sic2 groups:", sic2.nunique(), "| top5:", dict(top))
    print("saved to", OUT)


if __name__ == "__main__":
    main()

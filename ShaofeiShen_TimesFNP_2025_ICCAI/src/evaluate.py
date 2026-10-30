import argparse
import glob
import json
import os

import torch
import torch.nn.functional as F

from dataset_ett import ETTH1Data
from model import TimesFNP

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
CKPT = os.path.join(ROOT, "checkpoints")
LOGS = os.path.join(ROOT, "results", "logs")
RESULTS = os.path.join(ROOT, "results")


def make_tag(model_name, horizon, dataset, lambda1, lambda2):
    if dataset == "ett":
        lam_tag = f"_l{str(lambda2).replace('.', '')}" \
            if model_name == "timesfnp" and lambda2 != 0 else ""
        return f"ett_{model_name}{lam_tag}_H{horizon}"
    lam_tag = "" if (lambda1 == 1.0 and lambda2 == 1.0) \
        else f"_l{str(lambda1).replace('.', '')}"
    return f"{model_name}{lam_tag}_H{horizon}"


@torch.no_grad()
def evaluate(model_name: str, horizon: int, device: str = "cuda",
             dataset: str = "ett", lambda1: float = 1.0, lambda2: float = 1.0):
    tag = make_tag(model_name, horizon, dataset, lambda1, lambda2)
    ckpt = torch.load(os.path.join(CKPT, f"{tag}.pt"), map_location=device,
                      weights_only=False)
    if dataset == "ett":
        data = ETTH1Data(horizon=horizon, seq_len=96, device=device)
        kw = dict(in_channels=data.N, out_channels=data.N)
    else:
        raise ValueError("gkx 数据请使用 --tag 按 checkpoint 评估")
    model = TimesFNP(seq_len=data.L, horizon=horizon,
                     use_cl=(model_name == "timesfnp"), **kw).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    se, ae, n = 0.0, 0.0, 0
    for x, y, cov, _ in data.eval_batches("test"):
        pred, _ = model(x, cov)
        pred, y_raw = data.destandardize(pred), data.destandardize(y)
        se += F.mse_loss(pred, y_raw, reduction="sum").item()
        ae += F.l1_loss(pred, y_raw, reduction="sum").item()
        n += y_raw.numel()
    return se / n, ae / n


@torch.no_grad()
def evaluate_tag(tag: str, device: str = "cuda"):
    """按 checkpoint tag 评估 (gkx 从 ckpt 内保存的 args 重建数据/模型配置)"""
    ckpt = torch.load(os.path.join(CKPT, f"{tag}.pt"), map_location=device,
                      weights_only=False)
    a = ckpt.get("args", {})
    dataset = a.get("dataset", "gkx")
    horizon = a.get("horizon", 1)
    model_name = a.get("model", "timesfnp" if "timesfnp" in tag else "base")
    if dataset == "ett":
        data = ETTH1Data(horizon=horizon, seq_len=96, device=device)
        kw = dict(in_channels=data.N, out_channels=data.N)
    elif dataset == "gkx":
        from dataset_gkx import GKXPanelData
        data = GKXPanelData(horizon, lookback=a.get("lookback", 36), device=device,
                            scale=a.get("scale", "z"), target=a.get("target", "ret"))
        kw = dict(patch_len=a.get("patch", 3), d_model=a.get("dmodel", 64))
    else:
        raise ValueError(f"unknown dataset in checkpoint args: {dataset}")
    model = TimesFNP(seq_len=data.L, horizon=horizon,
                     use_cl=(model_name == "timesfnp"), **kw).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    se, ae, se_r, ae_r, n = 0.0, 0.0, 0.0, 0.0, 0
    for batch in data.eval_batches("test"):
        x, y, cov = batch[0], batch[1], batch[2]
        mu, sig = (batch[4], batch[5]) if len(batch) == 6 else (None, None)
        pred, _ = model(x, cov)
        se += F.mse_loss(pred, y, reduction="sum").item()
        ae += F.l1_loss(pred, y, reduction="sum").item()
        if hasattr(data, "denorm_batch"):
            p_raw, y_raw = data.denorm_batch(pred, y, mu, sig)
        else:
            p_raw, y_raw = data.destandardize(pred), data.destandardize(y)
        se_r += F.mse_loss(p_raw, y_raw, reduction="sum").item()
        ae_r += F.l1_loss(p_raw, y_raw, reduction="sum").item()
        n += y.numel()
    return se / n, ae / n, se_r / n, ae_r / n


def aggregate():
    recs = []
    for p in sorted(glob.glob(os.path.join(LOGS, "*.json"))):
        with open(p, encoding="utf-8") as f:
            r = json.load(f)
        r.setdefault("lambda1", 1.0)   # 旧记录无 lambda 字段, 默认 1.0
        r.setdefault("lambda2", 1.0)
        r.setdefault("dataset", "gkx")
        recs.append(r)
    if not recs:
        print("no result json found")
        return
    datasets = sorted({r["dataset"] for r in recs})

    lines = ["dataset,model,lambda1,lambda2,horizon,test_mse,test_mae,"
             "best_val_mse,best_epoch,seconds"]
    for r in sorted(recs, key=lambda r: (r["dataset"], r["model"], r["lambda1"],
                                         r["lambda2"], r["horizon"])):
        lines.append(f"{r['dataset']},{r['model']},{r['lambda1']},{r['lambda2']},"
                     f"{r['horizon']},{r['test_mse']:.6f},{r['test_mae']:.6f},"
                     f"{r['best_val_mse']:.6f},{r['best_epoch']},{r['seconds']:.1f}")
    with open(os.path.join(RESULTS, "results.csv"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    md = ["# TimesFNP 实验结果 (test 段)",
          "",
          "Improv. = (base - timesfnp) / base，正值表示 +CL 优于 base。",
          ""]

    # ---- ett: ETTh1 多变量 ----
    ett = [r for r in recs if r["dataset"] == "ett"]
    if ett:
        by_e = {(r["model"], r["lambda2"], r["horizon"]): r for r in ett}
        horizons = sorted({r["horizon"] for r in ett})
        lam2s = sorted({l for (m, l, h) in by_e if m == "timesfnp"}, reverse=True)
        md += ["## ett: ETTh1 多变量 (seq_len=96, pred_len=96, 7 通道)",
               "",
               "ETT 无公司/行业概念, 仅使用 temporal-wise 对比损失 (λ1=0)。",
               ""]
        for metric, key in (("MSE", "test_mse"), ("MAE", "test_mae")):
            header = f"| pred_len | base {metric} |"
            sep = "|---|---|"
            for lam in lam2s:
                header += f" +CL λ2={lam:g} {metric} | Improv. |"
                sep += "---|---|"
            md += [f"### {metric}", "", header, sep]
            for h in horizons:
                b = by_e.get(("base", 0.0, h)) or by_e.get(("base", 1.0, h))
                if not b:
                    continue
                row = f"| {h} | {b[key]:.6f} |"
                for lam in lam2s:
                    t = by_e.get(("timesfnp", lam, h))
                    if t:
                        imp = (b[key] - t[key]) / b[key] * 100
                        row += f" {t[key]:.6f} | {imp:+.2f}% |"
                    else:
                        row += " - | - |"
                md.append(row)
            md.append("")

    # ---- gkx: CRSP 源个股月度数据 ----
    gkx = [r for r in recs if r["dataset"] == "gkx"]
    if gkx:
        md += ["## gkx: 原始 CRSP 源数据 (GKX datashare, 1957-2016)",
               "",
               "主口径 = 全局 z-score (train 段统计量)。以下为全部 gkx 训练记录 "
               "(完整配置见 `results/tuning_log.csv`):",
               "",
               "| tag | H | val MSE | test MSE | test MAE | test MSE(raw) | test MAE(raw) |",
               "|---|---|---|---|---|---|---|"]
        for r in sorted(gkx, key=lambda r: (r["horizon"], r["model"],
                                            r["lambda1"], r["lambda2"])):
            md.append(f"| {r['tag']} | {r['horizon']} | {r['best_val_mse']:.4f} "
                      f"| {r['test_mse']:.4f} | {r['test_mae']:.4f} "
                      f"| {r.get('test_mse_raw', float('nan')):.5f} "
                      f"| {r.get('test_mae_raw', float('nan')):.5f} |")
        md.append("")

    md.append("单次训练时长、早停与 GPU 温度信息见 `results/logs/*.json` 与 `*.log`。")
    with open(os.path.join(RESULTS, "results.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["base", "timesfnp"])
    ap.add_argument("--horizon", type=int)
    ap.add_argument("--dataset", choices=["ett", "gkx"], default="gkx")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--lambda1", type=float, default=1.0)
    ap.add_argument("--lambda2", type=float, default=1.0)
    ap.add_argument("--tag", help="直接按 checkpoint tag 评估 (gkx 推荐)")
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args()
    if args.aggregate:
        aggregate()
    elif args.tag:
        mse, mae, mse_r, mae_r = evaluate_tag(args.tag, args.device)
        print(f"{args.tag}: test MSE {mse:.6f} MAE {mae:.6f} "
              f"(raw-scale MSE {mse_r:.6f} MAE {mae_r:.6f})")
    else:
        mse, mae = evaluate(args.model, args.horizon, args.device,
                            args.dataset, args.lambda1, args.lambda2)
        print(f"{args.dataset}/{args.model} H={args.horizon}: "
              f"test MSE {mse:.6f} MAE {mae:.6f}")


if __name__ == "__main__":
    main()

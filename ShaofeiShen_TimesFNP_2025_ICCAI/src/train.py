import argparse
import json
import os
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F

from augment import frequency_adaptive_sampling, mix_noise_injection
from dataset_ett import ETTH1Data
from dataset_gkx import GKXPanelData
from losses import instance_info_nce, temporal_info_nce
from model import TimesFNP

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
CKPT = os.path.join(ROOT, "checkpoints")
LOGS = os.path.join(ROOT, "results", "logs")
TUNING_LOG = os.path.join(ROOT, "results", "tuning_log.csv")
os.makedirs(CKPT, exist_ok=True)
os.makedirs(LOGS, exist_ok=True)

TAU = 0.1
STEPS_PER_EPOCH = 300   # ETT 数据规模小, 300 step/epoch 已足够覆盖
GKX_TRAIN_WINDOWS_PER_EPOCH = 20000   # GKX 数据量大, 每 epoch 子采样窗口数


def gpu_stat():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu,utilization.gpu,memory.used",
             "--format=csv,noheader"], text=True).strip()
        return out
    except Exception as e:
        return f"nvidia-smi failed: {e}"


def gpu_temp():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            text=True).strip()
        return int(out)
    except Exception:
        return -1


def unpack(batch):
    """兼容 4 元组 (ett) 与 6 元组 (gkx, 带 winz 的 mu/sig)"""
    if len(batch) == 6:
        return batch
    return (*batch, None, None)


def denorm(data, pred, y, mu, sig):
    if hasattr(data, "denorm_batch"):
        return data.denorm_batch(pred, y, mu, sig)
    return data.destandardize(pred), data.destandardize(y)


@torch.no_grad()
def eval_mse(model, data, split):
    model.eval()
    se, ae, n = 0.0, 0.0, 0
    for batch in data.eval_batches(split):
        x, y, cov, _, _, _ = unpack(batch)
        pred, _ = model(x, cov)
        se += F.mse_loss(pred, y, reduction="sum").item()
        ae += F.l1_loss(pred, y, reduction="sum").item()
        n += y.numel()
    return se / n, ae / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["base", "timesfnp"], required=True)
    ap.add_argument("--horizon", type=int, required=True,
                    help="gkx: {1,3,6,12}; ett: 96")
    ap.add_argument("--dataset", choices=["ett", "gkx"], default="gkx")
    ap.add_argument("--scale", choices=["raw", "pct", "z", "winz"], default="winz",
                    help="仅 gkx target=ret: 量纲口径 (raw / x100 / 全局z / 每窗口z)")
    ap.add_argument("--target", choices=["ret", "rank", "path_winz", "path_minmax",
                                         "path_z"], default="ret",
                    help="仅 gkx: 目标变换")
    ap.add_argument("--lookback", type=int, default=36)
    ap.add_argument("--patch", type=int, default=3)
    ap.add_argument("--dmodel", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lambda1", type=float, default=1.0)
    ap.add_argument("--lambda2", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=0.1, help="InfoNCE 温度")
    ap.add_argument("--keep-ratio", type=float, default=0.3, dest="keep_ratio",
                    help="频率自适应采样保留的能量成分比例")
    ap.add_argument("--noise-scale", type=float, default=1.0, dest="noise_scale",
                    help="混合噪声尺度倍数")
    ap.add_argument("--patience", type=int, default=5, help="早停 patience")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.dataset == "ett":
        assert args.horizon == 96, "ETT 协议固定 pred_len=96"
        assert args.batch <= 32, "ETT 实验 batch<=32"
    else:
        assert args.horizon in (1, 3, 6, 12)
        assert args.epochs <= 60 and args.batch <= 64

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    use_cl = args.model == "timesfnp"
    if args.dataset == "ett":
        # ETT 无行业标签, 只用 temporal-wise 损失 (lambda2), tag 以 lambda2 区分
        lam_tag = f"_l{str(args.lambda2).replace('.', '')}" \
            if use_cl and args.lambda2 != 0 else ""
        tag = f"ett_{args.model}{lam_tag}_H{args.horizon}"
    elif args.dataset == "gkx":
        lam_tag = "" if (args.lambda1 == 1.0 and args.lambda2 == 1.0) \
            else f"_l{str(args.lambda1).replace('.', '')}-{str(args.lambda2).replace('.', '')}"
        scale_tag = args.scale if args.target == "ret" else args.target
        tag = (f"gkx_{args.model}{lam_tag}_L{args.lookback}p{args.patch}"
               f"D{args.dmodel}_{scale_tag}_H{args.horizon}")
        # 非默认的 CL 超参追加进 tag, 防止覆盖已有结果
        extra = ""
        if args.tau != 0.1:
            extra += f"_t{str(args.tau).replace('.', '')}"
        if args.keep_ratio != 0.3:
            extra += f"_k{str(args.keep_ratio).replace('.', '')}"
        if args.noise_scale != 1.0:
            extra += f"_n{str(args.noise_scale).replace('.', '')}"
        if args.patience != 5:
            extra += f"_p{args.patience}"
        tag += extra
    log_path = os.path.join(LOGS, f"{tag}.log")
    logf = open(log_path, "w", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    log(f"[{tag}] args: {vars(args)}")
    stat = gpu_stat()
    log(f"[gpu before] {stat}")
    temp0 = gpu_temp()
    if 0 <= temp0 >= 83:
        log(f"[FATAL] GPU temperature {temp0}C >= 83C before training, abort.")
        return

    if args.dataset == "ett":
        data = ETTH1Data(horizon=args.horizon, seq_len=96, device=args.device)
        model_kw = dict(in_channels=data.N, out_channels=data.N)
        log(f"data: ETTh1 {data.T} hours x {data.N} channels | "
            f"train windows {data.n_windows('train')}, val {data.n_windows('val')}, "
            f"test {data.n_windows('test')}")
        steps_cap = STEPS_PER_EPOCH
    elif args.dataset == "gkx":
        data = GKXPanelData(args.horizon, lookback=args.lookback, device=args.device,
                            scale=args.scale, target=args.target)
        model_kw = dict(patch_len=args.patch, d_model=args.dmodel)
        n_tr = data.n_windows("train")
        log(f"data: GKX {data.N} stocks x {data.T} months | target={args.target} "
            f"scale={args.scale} L={data.L} | train windows {n_tr}, "
            f"val {data.n_windows('val')}, "
            f"test {data.n_windows('test')} | global mean {data.g_mean:.5f} "
            f"std {data.g_std:.5f}")
        steps_cap = max(1, GKX_TRAIN_WINDOWS_PER_EPOCH // args.batch)

    model = TimesFNP(seq_len=data.L, horizon=args.horizon, use_cl=use_cl,
                     **model_kw).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    steps_per_epoch = min(steps_cap,
                          max(1, -(-data.n_windows("train") // args.batch)))

    ckpt_path = os.path.join(CKPT, f"{tag}.pt")
    best_val, best_epoch, bad = float("inf"), -1, 0
    t0 = time.time()
    stop_reason = "max_epochs"

    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_mse = ep_ti = ep_tt = 0.0
        for _ in range(steps_per_epoch):
            x, y, cov, lab, _, _ = unpack(data.train_batch(args.batch))
            pred, z = model(x, cov)
            loss_mse = F.mse_loss(pred, y)
            loss = loss_mse
            l_inst = l_temp = x.new_zeros(())
            if use_cl:
                with torch.no_grad():
                    # 增强沿时间维进行: 多变量输入 (B,L,C) 先转置为 (B,C,L)
                    xt = x.transpose(1, 2) if x.dim() == 3 else x
                    x_freq = frequency_adaptive_sampling(xt, args.keep_ratio)  # 正样本
                    x_noise = mix_noise_injection(xt, args.noise_scale)        # 负样本
                    if x.dim() == 3:
                        x_freq = x_freq.transpose(1, 2)
                        x_noise = x_noise.transpose(1, 2)
                z_pos = model.project(model.encode(x_freq))
                z_noise = model.project(model.encode(x_noise))
                z_a = model.project(z)
                l_temp = temporal_info_nce(z_a, z_pos, z_noise, tau=args.tau)
                loss = loss_mse + args.lambda2 * l_temp
                if args.lambda1 != 0:   # ETT (无行业标签) 时 lambda1=0, 跳过实例损失
                    l_inst = instance_info_nce(model.instance_repr(z), lab,
                                               tau=args.tau)
                    loss = loss + args.lambda1 * l_inst
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_mse += loss_mse.item()
            ep_ti += l_inst.item()
            ep_tt += l_temp.item()

        val_mse, val_mae = eval_mse(model, data, "val")
        improved = val_mse < best_val
        if improved:
            best_val, best_epoch, bad = val_mse, epoch, 0
            torch.save({"model": model.state_dict(), "args": vars(args)}, ckpt_path)
        else:
            bad += 1
        log(f"epoch {epoch:02d} | train mse {ep_mse/steps_per_epoch:.5f} "
            f"L_inst {ep_ti/steps_per_epoch:.4f} L_temp {ep_tt/steps_per_epoch:.4f} | "
            f"val mse {val_mse:.5f} mae {val_mae:.5f} | best@{best_epoch}"
            + (" *" if improved else ""))

        temp = gpu_temp()
        if 0 <= temp >= 83:
            log(f"[FATAL] GPU temperature {temp}C >= 83C, stopping training.")
            stop_reason = f"gpu_overheat_{temp}C"
            break
        if bad >= args.patience:
            stop_reason = f"early_stop@{epoch}"
            break

    # 测试集评估 (模型口径 + 还原到原始超额收益口径)
    ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    test_mse, test_mae = 0.0, 0.0
    test_mse_r, test_mae_r, n = 0.0, 0.0, 0
    model.eval()
    with torch.no_grad():
        for batch in data.eval_batches("test"):
            x, y, cov, _, mu, sig = unpack(batch)
            pred, _ = model(x, cov)
            test_mse += F.mse_loss(pred, y, reduction="sum").item()
            test_mae += F.l1_loss(pred, y, reduction="sum").item()
            p_raw, y_raw = denorm(data, pred, y, mu, sig)
            test_mse_r += F.mse_loss(p_raw, y_raw, reduction="sum").item()
            test_mae_r += F.l1_loss(p_raw, y_raw, reduction="sum").item()
            n += y.numel()
    test_mse, test_mae = test_mse / n, test_mae / n
    test_mse_r, test_mae_r = test_mse_r / n, test_mae_r / n
    elapsed = time.time() - t0
    log(f"[result] {tag}: test MSE {test_mse:.6f} MAE {test_mae:.6f} "
        f"(raw-scale MSE {test_mse_r:.6f} MAE {test_mae_r:.6f}) "
        f"| best val {best_val:.6f}@ep{best_epoch} | {elapsed:.1f}s | stop={stop_reason}")
    log(f"[gpu after] {gpu_stat()}")
    logf.close()

    rec = {"model": args.model, "dataset": args.dataset, "horizon": args.horizon,
           "lambda1": args.lambda1,
           "lambda2": args.lambda2, "tau": args.tau, "keep_ratio": args.keep_ratio,
           "noise_scale": args.noise_scale, "target": args.target,
           "tag": tag, "test_mse": test_mse,
           "test_mae": test_mae, "test_mse_raw": test_mse_r,
           "test_mae_raw": test_mae_r, "best_val_mse": best_val,
           "best_epoch": best_epoch,
           "epochs_run": epoch, "seconds": elapsed, "stop_reason": stop_reason,
           "gpu_before": stat, "gpu_after": gpu_stat()}
    with open(os.path.join(LOGS, f"{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)

    if args.dataset == "gkx":
        # 调参日志: 每次 gkx 训练追加一行
        header = ("tag,model,horizon,scale,lookback,patch,dmodel,lambda1,lambda2,"
                  "tau,keep_ratio,noise_scale,patience,"
                  "lr,batch,epochs_run,best_val_mse,test_mse,test_mae,"
                  "test_mse_raw,test_mae_raw,seconds,stop_reason\n")
        row = (f"{tag},{args.model},{args.horizon},{args.scale},{args.lookback},"
               f"{args.patch},{args.dmodel},{args.lambda1},{args.lambda2},"
               f"{args.tau},{args.keep_ratio},{args.noise_scale},{args.patience},"
               f"{args.lr},{args.batch},{epoch},{best_val:.6f},{test_mse:.6f},"
               f"{test_mae:.6f},{test_mse_r:.6f},{test_mae_r:.6f},"
               f"{elapsed:.1f},{stop_reason}\n")
        if not os.path.exists(TUNING_LOG):
            with open(TUNING_LOG, "w", encoding="utf-8") as f:
                f.write(header)
        else:   # 旧表头无新列则升级
            with open(TUNING_LOG, encoding="utf-8") as f:
                first = f.readline()
            if "tau" not in first:
                import pandas as _pd
                _df = _pd.read_csv(TUNING_LOG)
                for c in ("tau", "keep_ratio", "noise_scale", "patience"):
                    _df[c] = {"tau": 0.1, "keep_ratio": 0.3,
                              "noise_scale": 1.0, "patience": 5}[c]
                _df.to_csv(TUNING_LOG, index=False)
        with open(TUNING_LOG, "a", encoding="utf-8") as f:
            f.write(row)


if __name__ == "__main__":
    main()

"""
最终对比实验入口脚本（只保留 FS_LLM_model_CON）。

特点：
- 预设用于 autoimmune 数据集：Data_set_190503.xlsx
- 任务：H vs 其他类别（二分类）
- 5 折（可配置）StratifiedKFold
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
from sklearn.metrics import f1_score, precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from dataloader import TabularDataset, load_dataset, prepare_numeric_features
from fsnet import FS_LLM_model_CON, get_data_AD_SF
from metric_helpper import comp_sensitivity, comp_specificity


@dataclass(frozen=True)
class FoldResult:
    fold: int
    auc: float
    acc: float
    f1: float
    precision: float
    sensitivity: float
    specificity: float


def _summarize_loaded_data(
    *,
    data_path: str,
    x_all: np.ndarray,
    column_names: list[str],
    label_s,
    y_all: np.ndarray,
    log_path: Path,
    head_n: int = 3,
    head_cols: int = 12,
) -> None:
    label_counts = label_s.astype(str).value_counts(dropna=False).to_dict()
    finite_ratio = float(np.isfinite(x_all).mean()) if x_all.size else 0.0
    neg1_ratio = float((x_all == -1.0).mean()) if x_all.size else 0.0
    x_min = float(np.min(x_all)) if x_all.size else 0.0
    x_max = float(np.max(x_all)) if x_all.size else 0.0
    x_mean = float(np.mean(x_all)) if x_all.size else 0.0
    x_std = float(np.std(x_all)) if x_all.size else 0.0

    preview_cols = column_names[: min(len(column_names), head_cols)]
    x_preview = x_all[: min(len(x_all), head_n), : min(x_all.shape[1], head_cols)] if x_all.ndim == 2 else x_all
    y_preview = y_all[: min(len(y_all), head_n)]

    lines = []
    lines.append(f"data_path: {data_path}")
    lines.append(f"raw_label_counts: {label_counts}")
    lines.append(f"x_shape: {tuple(x_all.shape)}  n_features: {len(column_names)}")
    lines.append(
        f"x_stats: mean={x_mean:.6g} std={x_std:.6g} min={x_min:.6g} max={x_max:.6g} "
        f"finite_ratio={finite_ratio:.6g} neg1_ratio={neg1_ratio:.6g}"
    )
    lines.append(f"binary_label_counts: pos(H)= {int(y_all.sum())}  neg(other)= {int((y_all == 0).sum())}")
    lines.append(f"preview_cols(<= {head_cols}): {preview_cols}")
    lines.append(f"preview_y(<= {head_n}): {y_preview.tolist()}")
    lines.append(f"preview_x(<= {head_n} rows, <= {head_cols} cols):\n{np.array2string(x_preview, precision=4)}")

    text = "\n".join(lines)
    print(text)
    with open(log_path, "a", encoding="utf-8") as f:
        print(text, file=f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FS_LLM_model_CON comparative experiment")

    parser.add_argument("--epochs", type=int, default=100, help="每折训练的 epoch 数")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率（Adam）")
    parser.add_argument("--reg", type=float, default=1e-4, help="权重衰减（weight_decay）")
    parser.add_argument("--seed", type=int, default=2022, help="随机种子（numpy/torch/cudnn）")
    parser.add_argument("--bs", type=int, default=128, help="batch size")
    parser.add_argument("--n_splits", type=int, default=5, help="StratifiedKFold 折数")

    parser.add_argument("--run_dir", type=str, default="./runs/autoimmune_H_binary", help="输出目录（日志/最优权重）")
    parser.add_argument("--json_file", type=str, default="setting.json", help="配置文件（读取 filename_process[0] 作为数据）")
    parser.add_argument(
        "--data_csv",
        type=str,
        default=r"./data/autoimmune/Data_set_190503.xlsx",
        help="数据路径（支持 csv/xlsx）；若提供则优先于 json_file",
    )
    parser.add_argument("--excel_sheet", type=str, default="", help="Excel sheet 名称/索引（为空表示第 0 个）")
    parser.add_argument("--label_col", type=str, default="Status", help="标签列名（该数据集为 Status）")
    parser.add_argument("--positive_label", type=str, default="H", help="二分类正类标签（H vs others）")
    parser.add_argument(
        "--pretrained_dir",
        type=str,
        default=r"./pretrained/clinicbert",
        help="预训练模型目录（ClinicBERT/RoBERTa 权重所在目录）",
    )

    parser.add_argument(
        "--sampler_type",
        type=str,
        default="weight",
        choices=["weight", "basic", "under"],
        help="重采样策略（与原版一致）：weight=WeightedRandomSampler；under=下采样多数类；basic=不重采样",
    )
    parser.add_argument("--sampler_pos_weight", type=float, default=1.0, help="weight 采样时正类(标签=1)的采样权重（负类固定为 1）")
    parser.add_argument("--sampler", type=float, default=None, help="兼容原版参数：等价于 --sampler_pos_weight（仅在 weight 时生效）")
    parser.add_argument(
        "--under_sampling_strategy",
        type=float,
        default=0.33,
        help="under 下采样目标比例（与原版一致）：minority/majority（例如 0.33 表示最终少数类:多数类≈1:3）",
    )

    parser.add_argument("--d_token", type=int, default=768, help="token embedding 维度（需与预训练 encoder hidden_size 匹配）")
    parser.add_argument("--pad_numerical_to", type=int, default=512, help="数值特征统一 padding/truncate 到的维度（与旧版一致）")
    parser.add_argument("--use_fs", type=int, default=0, help="是否启用 FS 阶段（0: 关闭，按旧版 BASELINE；1: 开启）")
    parser.add_argument("--aug_mask", type=float, default=0.0, help="数值 token mask 比例（仅训练时生效）")
    parser.add_argument("--beta", type=float, default=0.0, help="全局权重 global_w 与当前 batch 权重的融合系数（0 表示不融合）")
    parser.add_argument("--lam_fea", type=float, default=0.8, help="FS 阶段稀疏强度（越大越倾向压低平均权重）")
    parser.add_argument("--kminc", type=float, default=0.8, help="FS 阶段 pre_GCN 的稀疏系数（越大越稀疏）")
    parser.add_argument("--knn_k", type=int, default=30, help="构建特征图时的 KNN 邻居数（在特征维度上建图）")

    parser.add_argument("--cuda_visible_devices", type=str, default="", help="可选：设置 CUDA_VISIBLE_DEVICES")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader num_workers（Windows 建议 0）")

    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_data_path(args: argparse.Namespace) -> str:
    if args.data_csv:
        return args.data_csv

    json_path = Path(args.json_file)
    if not json_path.exists():
        raise FileNotFoundError("未提供 --data_csv 且 --json_file 不存在")

    with open(json_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    filename_list = cfg.get("filename_process", [])
    if not filename_list:
        raise ValueError("json 中缺少 filename_process 或为空")
    return str(filename_list[0])


def weighted_sampler_from_labels(y: np.ndarray, *, pos_weight: float) -> torch.utils.data.Sampler:
    weights = [1.0 if int(x) == 0 else float(pos_weight) for x in y.reshape(-1)]
    weights_t = torch.FloatTensor(np.array(weights, dtype=np.float32))
    return torch.utils.data.sampler.WeightedRandomSampler(weights_t, len(weights_t), replacement=True)


def _random_under_sample(
    x: np.ndarray,
    y: np.ndarray,
    *,
    sampling_strategy: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    与 RandomUnderSampler(sampling_strategy=..., random_state=...) 等价的核心逻辑（仅二分类）。
    sampling_strategy 定义为：resampled_minority / resampled_majority。
    """
    y = y.reshape(-1).astype(np.int64)
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    if len(idx0) == 0 or len(idx1) == 0:
        return x, y

    if len(idx0) <= len(idx1):
        minority_idx = idx0
        majority_idx = idx1
    else:
        minority_idx = idx1
        majority_idx = idx0

    strategy = float(sampling_strategy)
    if strategy <= 0.0 or strategy > 1.0:
        raise ValueError(f"under_sampling_strategy 必须在 (0, 1]，实际为 {strategy}")

    target_majority = int(round(len(minority_idx) / strategy))
    target_majority = max(target_majority, len(minority_idx))
    if len(majority_idx) <= target_majority:
        keep = np.concatenate([minority_idx, majority_idx])
        rng = np.random.RandomState(seed)
        rng.shuffle(keep)
        return x[keep], y[keep]

    rng = np.random.RandomState(seed)
    majority_keep = rng.choice(majority_idx, size=target_majority, replace=False)
    keep = np.concatenate([minority_idx, majority_keep])
    rng.shuffle(keep)
    return x[keep], y[keep]


def train_one_epoch(
    model: FS_LLM_model_CON,
    *,
    s: torch.Tensor,
    loader: data_utils.DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    lam_fea: float,
    global_w: torch.Tensor,
) -> tuple[float, torch.Tensor]:
    model.train()
    total_loss = 0.0
    n = 0

    for batch in loader:
        x = batch["data_num"].to(device)
        y = batch["label"].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits, _, global_w, _ = model(s, x, lam_fea, global_w)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()

        bs = int(y.shape[0])
        total_loss += float(loss.item()) * bs
        n += bs

    return total_loss / max(n, 1), global_w


@torch.no_grad()
def evaluate(
    model: FS_LLM_model_CON,
    *,
    s: torch.Tensor,
    loader: data_utils.DataLoader,
    device: torch.device,
    lam_fea: float,
    global_w: torch.Tensor,
) -> FoldResult:
    model.eval()

    all_y = []
    all_pred = []
    all_prob = []

    for batch in loader:
        x = batch["data_num"].to(device)
        y = batch["label"].to(device)
        logits, _, global_w, _ = model(s, x, lam_fea, global_w)
        if not torch.isfinite(logits).all():
            raise RuntimeError("non-finite logits detected in evaluate()")
        prob = torch.softmax(logits, dim=1)[:, 1]
        pred = logits.argmax(dim=1)

        all_y.append(y.detach().cpu().numpy())
        all_pred.append(pred.detach().cpu().numpy())
        all_prob.append(prob.detach().cpu().numpy())

    if len(all_y) == 0:
        ds_len = len(getattr(loader, "dataset", []))
        raise ValueError(
            f"evaluate() 没有拿到任何 batch（all_y 为空）。"
            f"常见原因：val_loader.drop_last=True 且验证集样本数({ds_len}) < batch_size。"
            f"请把 val_loader.drop_last 设为 False 或调小 --bs。"
        )

    y_true = np.concatenate(all_y, axis=0)
    y_pred = np.concatenate(all_pred, axis=0)
    y_prob = np.concatenate(all_prob, axis=0)
    if np.isnan(y_prob).any() or np.isinf(y_prob).any():
        raise RuntimeError("non-finite probabilities detected in evaluate()")

    auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0
    acc = float((y_pred == y_true).mean())
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    precision = float(precision_score(y_true, y_pred, zero_division=0))

    y_true_bool = (y_true == 1).tolist()
    y_pred_bool = (y_pred == 1).tolist()
    sensitivity = float(comp_sensitivity(y_true_bool, y_pred_bool))
    specificity = float(comp_specificity(y_true_bool, y_pred_bool))

    return FoldResult(
        fold=-1,
        auc=auc,
        acc=acc,
        f1=f1,
        precision=precision,
        sensitivity=sensitivity,
        specificity=specificity,
    )


def main() -> None:
    args = parse_args()

    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "log.txt"

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = resolve_data_path(args)
    sheet = None
    if args.excel_sheet.strip() != "":
        try:
            sheet = int(args.excel_sheet)
        except ValueError:
            sheet = args.excel_sheet

    data_df, label_s = load_dataset(data_path, sheet_name=sheet, label_col=args.label_col)
    x_all, column_names = prepare_numeric_features(data_df)
    y_all = (label_s.astype(str) == args.positive_label).to_numpy(dtype=np.int64)
    n_classes = 2

    _summarize_loaded_data(
        data_path=data_path,
        x_all=x_all,
        column_names=column_names,
        label_s=label_s,
        y_all=y_all,
        log_path=log_path,
    )

    with open(log_path, "a", encoding="utf-8") as f:
        print("start_time:", datetime.now(), file=f)
        print("data_path:", data_path, file=f)
        print("x_shape:", x_all.shape, file=f)
        print("positive_label:", args.positive_label, file=f)
        print("pos_count:", int(y_all.sum()), file=f)
        print("neg_count:", int((y_all == 0).sum()), file=f)
        print("columns:", len(column_names), file=f)
        for k, v in vars(args).items():
            print(f"{k}: {v}", file=f)

    kf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    fold_results: list[FoldResult] = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(x_all, y_all), start=1):
        x_train = x_all[train_idx]
        y_train = y_all[train_idx]
        x_val = x_all[val_idx]
        y_val = y_all[val_idx]

        fold_dist = (
            f"fold={fold} split_counts: "
            f"train_pos={int(y_train.sum())} train_neg={int((y_train == 0).sum())} "
            f"val_pos={int(y_val.sum())} val_neg={int((y_val == 0).sum())}"
        )
        print(fold_dist)
        with open(log_path, "a", encoding="utf-8") as f:
            print(fold_dist, file=f)

        if args.sampler_type == "under":
            x_train, y_train = _random_under_sample(
                x_train,
                y_train,
                sampling_strategy=args.under_sampling_strategy,
                seed=args.seed + fold,
            )
            after_dist = (
                f"fold={fold} after_under: "
                f"train_pos={int(y_train.sum())} train_neg={int((y_train == 0).sum())} "
                f"(under_sampling_strategy={args.under_sampling_strategy})"
            )
            print(after_dist)
            with open(log_path, "a", encoding="utf-8") as f:
                print(after_dist, file=f)

        s = get_data_AD_SF(x_train, args.seed, knn_k=args.knn_k).to(device)#虽然计算了，但最后模型并没有用到

        train_ds = TabularDataset(x_train, y_train)
        val_ds = TabularDataset(x_val, y_val)

        if args.sampler_type == "weight":
            pos_weight = args.sampler_pos_weight if args.sampler is None else float(args.sampler)
            sampler = weighted_sampler_from_labels(y_train, pos_weight=pos_weight)
            train_loader = data_utils.DataLoader(
                train_ds,
                batch_size=args.bs,
                sampler=sampler,
                drop_last=True,
                num_workers=args.num_workers,
            )
        else:
            train_loader = data_utils.DataLoader(
                train_ds,
                batch_size=args.bs,
                drop_last=True,
                num_workers=args.num_workers,
            )

        val_loader = data_utils.DataLoader(
            val_ds,
            batch_size=args.bs,
            shuffle=False,
            drop_last=False,
            num_workers=args.num_workers,
        )

        # FS_LLM_model_CON 接口说明：
        # - d_numerical: 表格数值特征维度（prepare_numeric_features 后的列数）
        # - fs_input_dim: FS 阶段采用 pre_GCN（动态图）并按 batch_size 作为输入维度设计，因此这里应与 bs 一致；
        #   若最后一个 batch 小于 bs，模型内部会做 padding 以保证维度匹配
        # - d_token: token embedding 维度（与 encoder hidden_size 对齐）
        # - aug_mask: 数值 token 随机 mask 比例（训练增强，可设 0 关闭）
        # - beta: global_w 融合系数（0 关闭，>0 使用全局信息平滑权重）
        model = FS_LLM_model_CON(
            d_numerical=args.pad_numerical_to,
            fs_input_dim=args.bs,
            d_token=args.d_token,
            token_bias=True,
            aug_mask=args.aug_mask,
            beta=args.beta,
            kminc=args.kminc,
            pad_numerical_to=args.pad_numerical_to,
            use_fs=bool(args.use_fs),
            pretrained_dir=args.pretrained_dir,
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.reg)
        global_w = torch.ones(args.pad_numerical_to, device=device)

        best_auc = -1.0
        best_result: FoldResult | None = None

        for epoch in range(1, args.epochs + 1):
            train_loss, global_w = train_one_epoch(
                model,
                s=s,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                lam_fea=args.lam_fea,
                global_w=global_w,
            )

            result = evaluate(
                model,
                s=s,
                loader=val_loader,
                device=device,
                lam_fea=args.lam_fea,
                global_w=global_w,
            )

            if result.auc > best_auc:
                best_auc = result.auc
                best_result = FoldResult(
                    fold=fold,
                    auc=result.auc,
                    acc=result.acc,
                    f1=result.f1,
                    precision=result.precision,
                    sensitivity=result.sensitivity,
                    specificity=result.specificity,
                )
                torch.save(model.state_dict(), run_dir / f"best_fold_{fold}.pth")

            with open(log_path, "a", encoding="utf-8") as f:
                print(
                    f"fold={fold} epoch={epoch} train_loss={train_loss:.6f} "
                    f"val_auc={result.auc:.4f} val_acc={result.acc:.4f} val_f1={result.f1:.4f}",
                    file=f,
                )

        if best_result is not None:
            fold_line = (
                f"fold={best_result.fold} best_auc={best_result.auc:.4f} best_acc={best_result.acc:.4f} "
                f"best_f1={best_result.f1:.4f} best_precision={best_result.precision:.4f} "
                f"best_sensitivity={best_result.sensitivity:.4f} best_specificity={best_result.specificity:.4f}"
            )
            print(fold_line)
            with open(log_path, "a", encoding="utf-8") as f:
                print(fold_line, file=f)

        if best_result is not None:
            fold_results.append(best_result)

    if fold_results:
        aucs = [r.auc for r in fold_results]
        accs = [r.acc for r in fold_results]
        f1s = [r.f1 for r in fold_results]
        pres = [r.precision for r in fold_results]
        sens = [r.sensitivity for r in fold_results]
        spes = [r.specificity for r in fold_results]

        summary = (
            f"mean_auc={np.mean(aucs):.4f} mean_acc={np.mean(accs):.4f} mean_f1={np.mean(f1s):.4f} "
            f"mean_precision={np.mean(pres):.4f} mean_sensitivity={np.mean(sens):.4f} mean_specificity={np.mean(spes):.4f}"
        )
        print(summary)
        with open(log_path, "a", encoding="utf-8") as f:
            print(summary, file=f)


if __name__ == "__main__":
    main()

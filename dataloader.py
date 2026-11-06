"""
最终对比实验用的最小 dataloader。

约定：
- 支持 CSV / Excel（xlsx/xls）
- 训练/验证仅使用数值特征（会做 to_numeric + 缺失值填充）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

try:
    import torch.utils.data as data_utils
except Exception:  # pragma: no cover
    data_utils = None


def load_dataset(
    file_path: str,
    *,
    sheet_name: str | int | None = None,
    label_col: str = "Status",
) -> tuple[pd.DataFrame, pd.Series]:
    suffix = str(file_path).lower()
    if suffix.endswith((".xlsx", ".xls")):
        df = pd.read_excel(file_path, sheet_name=0 if sheet_name is None else sheet_name)
    else:
        df = pd.read_csv(file_path)

    df.columns = [str(c).strip() for c in df.columns]

    if label_col not in df.columns and df.shape[1] >= 2:
        key_col = df.columns[0]
        first_col = df[key_col].astype(str).str.strip()
        if (first_col == label_col).any():
            df_t = df.set_index(key_col).T
            df_t.index.name = "id1"
            df = df_t.reset_index()

    if label_col not in df.columns:
        raise ValueError(f"标签列不存在: {label_col}（如果 Status 在某一行，请确保第 1 列包含该行名）")

    label = df[label_col].astype(str)
    data = df.drop(columns=[label_col])
    return data, label


def prepare_numeric_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """
    将 DataFrame 变为纯数值特征矩阵。
    - 会删除 id1
    - 尽量保持与原始 process_data 版本一致：特征应当本来就是数值
    - 如果某些列不是数值：会尝试严格转为数值；若无法转换则直接报错（不静默转为 NaN）
    - 若存在缺失值 NaN：用 -1 填充（与原始 tabular_data 的数值列空值处理一致）
    """
    for id_col in ["id1", "Sample ID", "sample_id", "SampleID"]:
        if id_col in df.columns:
            df = df.drop(columns=[id_col])
    column_names = list(df.columns)

    df_num = df.copy()
    for col in df_num.columns:
        if pd.api.types.is_numeric_dtype(df_num[col]):
            continue
        try:
            df_num[col] = pd.to_numeric(df_num[col], errors="raise")
        except Exception as e:
            sample_values = df_num[col].dropna().astype(str).unique()[:5].tolist()
            raise ValueError(f"特征列不是数值且无法转换: {col}, 示例={sample_values}") from e

    if df_num.isna().any().any():
        df_num = df_num.fillna(-1.0)
    x = df_num.to_numpy(dtype=np.float32)
    return x, column_names


@dataclass(frozen=True)
class TabularBatch:
    data_num: Any
    label: Any


if data_utils is not None:

    class TabularDataset(data_utils.Dataset):
        def __init__(self, x_num: np.ndarray, y: np.ndarray) -> None:
            import torch

            self.x_num = torch.from_numpy(x_num)
            self.y = torch.from_numpy(y.astype(np.int64))

        def __len__(self) -> int:
            return int(self.y.shape[0])

        def __getitem__(self, index: int):
            return {"data_num": self.x_num[index], "label": self.y[index]}

else:

    class TabularDataset:  # pragma: no cover
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("torch 未正确安装，无法创建 TabularDataset")

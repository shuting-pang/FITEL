"""
FS_LLM_model_CON 的最小实现（用于最终对比实验）。

目标：
- 只保留 FS_LLM_model_CON 及其必需组件
- 去掉无效依赖（zero/lib/ViT 等）与硬编码路径
- 在同一批训练数据上构建“特征图”(feature graph) 并学习特征权重，再送入 LLM encoder 做分类
"""

import math
import typing as ty
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as nn_init
from torch import Tensor
from transformers import AutoModel

from utils_FSNET import create_graph_from_embedding, preprocess_adj


def _process_adj_sf(a: np.ndarray, *, knn_k: int) -> torch.Tensor:
    adj = create_graph_from_embedding(a, name="knn", n=knn_k)
    adj, _ = preprocess_adj(adj)
    adj = adj.todense()
    return torch.from_numpy(np.asarray(adj)).to(torch.float32)


def get_data_AD_SF(features: np.ndarray, seed: int, *, knn_k: int = 30) -> torch.Tensor:
    """
    根据训练集数值特征构建“特征-特征”邻接矩阵 S。

    features: shape = [n_samples, n_features]
    返回: S, shape = [n_features, n_features]
    """
    rng = np.random.RandomState(seed)
    perm = rng.permutation(features.shape[0])
    features = features[perm]
    s = _process_adj_sf(features.T, knn_k=knn_k)
    return s


class GraphConvolution(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, *, use_bias: bool = True) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        self.bias = nn.Parameter(torch.Tensor(output_dim)) if use_bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn_init.kaiming_uniform_(self.weight)
        if self.bias is not None:
            nn_init.zeros_(self.bias)

    def forward(self, input_feature: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        support = torch.mm(input_feature, self.weight)
        output = torch.mm(adjacency, support)
        if self.bias is not None:
            output = output + self.bias
        return output


class GraphConvolutionLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x_w = torch.mm(x, self.weight)
        out = torch.spmm(adj, x_w)
        return out + self.bias if self.bias is not None else out


class GCN_patch(nn.Module):
    def __init__(self, channels: int, hid_features: int, out_features: int) -> None:
        super().__init__()
        self.gcn1 = GraphConvolutionLayer(channels, hid_features)
        self.gcn2 = GraphConvolutionLayer(hid_features, out_features)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x = F.dropout(x, p=0.4, training=self.training)
        x = F.relu(self.gcn1(x, adj))
        x = F.dropout(x, p=0.4, training=self.training)
        x = self.gcn2(x, adj)
        return x


class pre_GCN(nn.Module):
    def __init__(self, dim: int, kminc: float = 0.8) -> None:
        super().__init__()
        self.kminc = float(kminc)
        self.temp_matrix = nn.Parameter(torch.ones(dim, dim))
        self.v0 = nn.Linear(dim, dim)
        self.v1 = nn.Linear(dim, 128)
        self.act = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(negative_slope=5e-2)
        self.gcn = GCN_patch(dim, dim, dim)

    def process_adj(self, adj: torch.Tensor) -> torch.Tensor:
        adj = (adj + adj.t()) / 2
        rowsum = adj.sum(1) + 1e-12
        r_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
        r_inv_sqrt[torch.isinf(r_inv_sqrt)] = 0.0
        r_mat_inv_sqrt = torch.diag(r_inv_sqrt)
        adj = torch.matmul(adj, r_mat_inv_sqrt)
        adj = torch.transpose(adj, 0, 1)
        adj = torch.matmul(adj, r_mat_inv_sqrt)
        return adj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.squeeze(0)
        z = torch.mm(x, self.v1(self.act(self.v0(self.temp_matrix))))
        q = torch.mm(self.leaky_relu(z), self.leaky_relu(z).t())
        q = q - self.kminc * torch.mean(q)
        q = F.relu(q)
        q = F.normalize(q, dim=0, eps=1e-12)
        q = self.process_adj(q + (1e-7) * torch.eye(q.shape[0], device=q.device))
        out = self.gcn(x, q)
        return out.unsqueeze(0)


class FS_model(nn.Module):
    def __init__(self, *, input_dim: int, beta: float = 0.0, kminc: float = 0.8) -> None:
        super().__init__()
        self.beta = beta
        self.input_dim = int(input_dim)
        self.fsnet_gcn = pre_GCN(self.input_dim, kminc=kminc)

    def _produce_m(self, hidden_feature: torch.Tensor, lam: float, batch_size: int) -> torch.Tensor:
        """
        根据 feature embedding 的 L2 范数生成每个样本共享的特征权重，并扩展到 batch 维度。
        输出 shape = [batch_size, n_features]。
        """
        m1 = torch.sqrt(torch.sum(hidden_feature * hidden_feature, dim=1))
        m2 = m1 - lam * torch.mean(m1)
        m = F.relu(m2.repeat(batch_size, 1))
        return m

    def forward(self, s: torch.Tensor, feature: torch.Tensor, lam_fea: float) -> torch.Tensor:
        batch_size = int(feature.shape[0])
        if batch_size < self.input_dim:
            padded = torch.zeros(self.input_dim, feature.shape[1], dtype=feature.dtype, device=feature.device)
            padded[:batch_size] = feature
            feature_for_fs = padded
        elif batch_size == self.input_dim:
            feature_for_fs = feature
        else:
            raise ValueError(f"batch_size({batch_size}) > FS.input_dim({self.input_dim})，请调大 --bs 或开启 drop_last")

        x = feature_for_fs.unsqueeze(0).permute(0, 2, 1)
        y = self.fsnet_gcn(x.float()).squeeze(0)
        m_feature = self._produce_m(y, lam_fea, batch_size)
        return m_feature


class TabularTokenizer(nn.Module):
    """
    将数值表格特征映射为 transformer 可用的 token embedding。

    - token 0 为 [CLS]
    - 其余 token 对应每个数值特征
    """
    def __init__(self, *, d_numerical: int, d_token: int, bias: bool, aug_mask: float) -> None:
        super().__init__()
        self.d_numerical = d_numerical
        self.d_token = d_token
        self.aug_mask = aug_mask

        self.weight = nn.Parameter(Tensor(d_numerical + 1, d_token))
        self.bias = nn.Parameter(Tensor(d_numerical, d_token)) if bias else None
        self.mask_token = nn.Parameter(Tensor(1, 1, d_token))

        nn_init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn_init.kaiming_uniform_(self.mask_token, a=math.sqrt(5))
        if self.bias is not None:
            nn_init.kaiming_uniform_(self.bias, a=math.sqrt(5))

    def _random_masking(self, x: Tensor, mask_ratio: float) -> Tensor:
        if mask_ratio <= 0.0:
            return x
        b, n, d = x.shape
        mask = torch.bernoulli(torch.ones(b, n - 1, device=x.device) * mask_ratio)
        mask = torch.cat((torch.zeros(b, 1, device=x.device), mask), dim=1)
        mask = mask.unsqueeze(-1).to(torch.float).repeat(1, 1, d)
        mask_tokens = self.mask_token.repeat(b, n, 1)
        return x * (1 - mask) + mask_tokens * mask

    def forward(self, x_num: Tensor) -> Tensor:
        if x_num.shape[1] != self.d_numerical:
            raise ValueError(f"x_num.shape[1] 必须等于 d_numerical={self.d_numerical}，实际为 {x_num.shape[1]}")

        x_num = torch.cat([torch.ones(len(x_num), 1, device=x_num.device), x_num], dim=1)
        x = self.weight[None] * x_num[:, :, None]

        if self.bias is not None:
            bias = torch.cat([torch.zeros(1, self.bias.shape[1], device=x.device), self.bias])
            x = x + bias[None]

        x = self._random_masking(x, self.aug_mask if self.training else 0.0)
        return x


class LLM_model(nn.Module):
    """
    基于本地预训练模型（pre_model/clinicbert）encoder 的分类器。
    """
    def __init__(
        self,
        *,
        d_numerical: int,
        d_token: int,
        token_bias: bool,
        aug_mask: float,
        pretrained_dir: Path,
    ) -> None:
        super().__init__()
        self.tokenizer = TabularTokenizer(d_numerical=d_numerical, d_token=d_token, bias=token_bias, aug_mask=aug_mask)
        self.pt_model = AutoModel.from_pretrained(str(pretrained_dir))

        hidden_size = int(getattr(self.pt_model.config, "hidden_size", d_token))
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
        )

    def forward(self, feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.tokenizer(feature.float())
        if not hasattr(self.pt_model, "encoder"):
            raise AttributeError("当前预训练模型不包含 encoder 属性，无法按旧版方式绕过 embedding")

        encoder_outputs = self.pt_model.encoder(x)
        hidden_states = encoder_outputs[0] if isinstance(encoder_outputs, (tuple, list)) else encoder_outputs.last_hidden_state
        cls_feature = hidden_states[:, 0]
        logits = self.classifier(cls_feature)
        return logits, cls_feature


class FS_LLM_model_CON(nn.Module):
    """
    最终使用的模型：FS（特征选择/加权） + LLM encoder 分类。
    """
    def __init__(
        self,
        *,
        d_numerical: int,
        fs_input_dim: int,
        d_token: int = 768,
        token_bias: bool = True,
        aug_mask: float = 0.0,
        beta: float = 0.0,
        kminc: float = 0.8,
        pad_numerical_to: int = 512,
        use_fs: bool = False,
        pretrained_dir: ty.Optional[str] = None,
    ) -> None:
        super().__init__()
        self.beta = float(beta)
        self.pad_numerical_to = int(pad_numerical_to)
        self.use_fs = bool(use_fs)

        if pretrained_dir is None:
            base_dir = Path(__file__).resolve().parent
            pretrained_dir = str((base_dir.parent / "pre_model" / "clinicbert").resolve())

        self.fs_model = FS_model(input_dim=fs_input_dim, beta=self.beta, kminc=kminc)
        self.llm_model = LLM_model(
            d_numerical=d_numerical,
            d_token=d_token,
            token_bias=token_bias,
            aug_mask=aug_mask,
            pretrained_dir=Path(pretrained_dir),
        )

    def _pad_or_truncate(self, x: torch.Tensor) -> torch.Tensor:
        target = self.pad_numerical_to
        if x.shape[1] == target:
            return x
        if x.shape[1] > target:
            return x[:, :target]
        out = torch.zeros(x.shape[0], target, dtype=x.dtype, device=x.device)
        out[:, : x.shape[1]] = x
        return out

    def forward(
        self,
        s: torch.Tensor,
        feature: torch.Tensor,
        lam_fea: float,
        global_w: torch.Tensor,
    ) -> tuple[torch.Tensor, None, torch.Tensor, int]:
        feature = self._pad_or_truncate(feature)
        if not self.use_fs:
            logits, _ = self.llm_model(feature)
            new_global_w = torch.ones(feature.shape[1], device=feature.device, dtype=feature.dtype)
            return logits, None, new_global_w, 0

        if global_w.ndim == 1 and global_w.numel() != feature.shape[1]:
            global_w = torch.ones(feature.shape[1], device=feature.device, dtype=feature.dtype)

        m_feature = self.fs_model(s, feature, lam_fea)

        if global_w.ndim == 1:
            global_w = global_w.unsqueeze(0).expand_as(m_feature)
        if self.beta != 0.0:
            m_feature = self.beta * global_w + (1.0 - self.beta) * m_feature
        feature_selected = feature * m_feature
        logits, _ = self.llm_model(feature_selected)

        new_global_w = m_feature.detach().mean(dim=0)
        zero_count = int((m_feature <= 0).sum().item())
        return logits, None, new_global_w, zero_count

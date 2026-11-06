# FITEL 二分类实验说明（H vs 其他）

本文件夹是用于 autoimmune 数据集的二分类（`H` vs 其他）的整理版本，入口脚本为 [main_comparative_experiement.py](../main_comparative_experiement.py)，模型为 [FS_LLM_model_CON](../fsnet.py)。

## 1. 预训练模型来源（ClinicBERT）

- 模型来源（HuggingFace）：https://huggingface.co/tdobrxl/ClinicBERT
- 说明：该模型与 RoBERTa 架构一致，适合做 encoder 特征抽取。
- 代码里通过 `AutoModel.from_pretrained(pretrained_dir)` 读取本地目录权重。
- 运行时你可能会看到类似提示：
  - `Some weights ... pooler ... newly initialized`
  - 这是常见 warning（很多 RoBERTa checkpoint 没有对应 pooler 权重），不影响继续训练/分类。

## 2. 数据集与标签

### 2.1 数据路径（默认）

入口脚本默认读取：
- `./data/autoimmune/Data_set_190503.xlsx`

也可通过参数覆盖：
- `--data_csv <xlsx/csv 路径>`

### 2.2 数据内容与“旋转/转置”处理（关键点）

该数据集可能存在一种布局：`Status` 不在表头列名里，而是在“某一行”的第一列中（即第一列是字段名/变量名，其中某行是 `Status`）。

对应处理逻辑在 [dataloader.py:load_dataset](../dataloader.py)：
- 如果读入后找不到 `label_col=Status` 这列：
  - 检查“第 1 列”是否存在值等于 `Status`
  - 若存在则自动执行 `set_index(第1列).T` 完成转置，使 `Status` 变成标准的标签列

### 2.3 二分类定义

- 标签列：默认 `--label_col Status`
- 正类：默认 `--positive_label H`
- 二分类映射：`y = (Status == "H")`

## 3. 数值特征处理（关键点）

数值特征提取在 [dataloader.py:prepare_numeric_features](../dataloader.py)：
- 会移除常见 ID 列（如 `id1`, `Sample ID`）
- 对非数值列：尝试严格 `to_numeric(errors="raise")`
  - 转不了会直接报错（避免静默变成 NaN 导致后续训练异常）
- 只有当原数据本身存在缺失值时才会 `fillna(-1.0)`

## 4. “关键 trick/对齐点”（复现最重要的差异点）

### 4.1 数值特征强制 padding/truncate 到 512（对齐旧版）

在 [fsnet.py:FS_LLM_model_CON](../fsnet.py) 中：
- forward 开始会把输入特征 `feature` pad/truncate 到 `--pad_numerical_to`（默认 512）
- 这是为了对齐旧版逻辑：LLM 侧看到的数值 token 序列长度固定（+1 CLS）

### 4.2 LLM 侧按旧版方式调用 encoder（绕过 embedding）

在 [fsnet.py:LLM_model.forward](../fsnet.py) 中：
- 使用 `self.pt_model.encoder(x)`（而不是 `pt_model(inputs_embeds=...)` 或 `pt_model(input_ids=...)`）
- 目的是对齐旧版“直接喂 embedding 到 encoder”路径

### 4.3 FS 阶段输入维度与 batch_size 强绑定

在 [fsnet.py:FS_model](../fsnet.py) 中：
- FS 的图卷积按 `fs_input_dim == batch_size` 设计
- 最后一个 batch 若小于 `bs`，FS 内部会 padding 到 `bs` 再做图卷积

因此在 [main_comparative_experiement.py](../main_comparative_experiement.py) 构造模型时固定：
- `fs_input_dim = --bs`

### 4.4 训练/验证 drop_last 设置（避免 batch 太小导致不稳定）

- 训练集：`drop_last=True`
  - 主要是避免训练过程中出现极小 batch（例如 BN 统计不稳定）
- 验证集：`drop_last=False`
  - 否则当验证集样本数 < batch_size 时会直接出现 0 个 batch，导致 `np.concatenate` 报错

### 4.5 重采样策略与原版对齐

入口脚本提供与原版对应的策略（见 [main_comparative_experiement.py](../main_comparative_experiement.py)）：
- `--sampler_type weight`
  - `WeightedRandomSampler`
  - 权重规则与原版一致：label=0 权重 1；label=1 权重 `--sampler`（兼容原版）或 `--sampler_pos_weight`
- `--sampler_type under`
  - 多数类下采样（等价 RandomUnderSampler 的核心逻辑）
  - `--under_sampling_strategy` 表示 `minority/majority`（默认 0.33）
- `--sampler_type basic`
  - 不重采样

### 4.6 FS 是否启用（用于对照原版“CON 覆盖 FS”的现象）

参数：
- `--use_fs 0/1`（默认 0）

说明：
- `use_fs=0`：走 BASELINE（不做 FS 加权），用于复现旧版“看起来 FS 计算了但实际未生效”的行为
- `use_fs=1`：启用 FS 权重 `m_feature` 对特征加权，再送入 LLM 分类

## 5. 运行方式

在仓库根目录执行（Bash / zsh 示例）：

```bash
python main_comparative_experiement.py \
  --data_csv ./data/autoimmune/Data_set_190503.xlsx \
  --pretrained_dir ./pretrained/clinicbert \
  --label_col Status \
  --positive_label H \
  --n_splits 5 \
  --bs 63 \
  --epochs 100 \
  --sampler_type weight \
  --sampler 1.0
```

以上保留代码默认的 `--use_fs 0`；若要启用特征选择/加权分支，请额外传入 `--use_fs 1`。Windows PowerShell 中可将命令写成一行；CMD 多行命令需要使用 `^` 作为续行符。

输出：
- 控制台与 `--run_dir` 下的 `log.txt` 会打印：
  - 原始 `Status` 分布
  - 二分类后正/负计数
  - 每折 train/val 的正负分布（以及 under 下采样后的分布）
  - 每个 epoch 的验证指标与每折 best 指标

## 6. 常见问题

### 6.1 `need at least one array to concatenate`

含义：验证集 DataLoader 没有产出任何 batch（常见原因：`val_samples < batch_size` 且 `drop_last=True`）。

本工程中验证集已固定为 `drop_last=False`，若你改过这里，请保持该设置或调小 `--bs`。

### 6.2 实验仍然可能导致“不完全一致”的因素（尤其是 GPU）

- GPU 上部分算子可能存在非确定性实现（即使 seed 一致，最后几位也可能漂移）。
- 想要更“强”的可复现性：
  - 固定 `CUDA_VISIBLE_DEVICES`，避免换卡/换驱动
  - 尽量固定依赖版本（torch/transformers/numpy/sklearn）

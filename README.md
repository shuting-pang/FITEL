# FITEL

Code accompanying **Feature-interpretable disease prediction from tabular data via dynamic GCN and LLMs**, *Pattern Recognition* (2026).

[Paper](https://www.sciencedirect.com/science/article/pii/S0031320326011027) · [中文实验说明](docs/REPRODUCTION_zh.md)

FITEL combines feature-graph learning with a pretrained language-model encoder for tabular disease prediction. This release contains the supplied **H-versus-other binary-classification experiment**, built around `FS_LLM_model_CON`. It does not include the full set of experiments or an automated raw-EHR preprocessing pipeline.

## What is included

```text
FITEL/
├── main_comparative_experiement.py  # Training, stratified cross-validation and evaluation
├── fsnet.py                        # Feature graph, feature weighting and pretrained encoder
├── dataloader.py                   # CSV/Excel loading and numeric feature preparation
├── con_loss.py                     # Auxiliary loss modules
├── metric_helpper.py               # Evaluation helpers
├── utils_FSNET.py                  # Graph construction and processing
├── utils_LLM.py                    # Ramp-up utility
├── requirements.txt
└── docs/REPRODUCTION_zh.md
```

The original entry-point spelling, `main_comparative_experiement.py`, is retained so existing commands remain compatible. Datasets, pretrained model weights and training outputs are not bundled.

## Setup

Use Python 3.10 or newer and install the dependencies:

```bash
pip install -r requirements.txt
```

The requirements list is inferred from imports in the supplied code; it is not a locked or fully validated training environment.

Place compatible pretrained encoder files in a local directory such as `./pretrained/clinicbert`. The provided experiment loads them through `AutoModel.from_pretrained`. The original model reference is [ClinicBERT](https://huggingface.co/tdobrxl/ClinicBERT), a RoBERTa-based encoder. Set `--d_token` to match the encoder hidden size; the script default is 768.

## Data format

Provide a CSV or Excel table with:

- A label column named `Status`, or another name selected with `--label_col`.
- The positive label `H`, or another value selected with `--positive_label`; all remaining labels map to the negative class.
- Numeric feature columns. Numeric strings are converted strictly; missing numeric values are filled with `-1.0`.

The loader removes common sample-ID columns and supports the transposed layout described in the [Chinese experiment notes](docs/REPRODUCTION_zh.md). Non-numeric features must be processed before running this experiment.

## Run

From the repository root:

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
  --sampler 1.0 \
  --use_fs 1
```

**Feature selection is opt-in:** the script default is `--use_fs 0`, which runs the baseline without feature weighting. The command above explicitly enables it. Use `--use_fs 0` for that baseline comparison.

Other settings retained from the supplied experiment:

- `--pad_numerical_to 512` pads or truncates the numeric feature vector.
- `--sampler_type` supports `weight`, `basic` and `under`.
- The feature-selection input dimension is tied to `--bs`; training uses `drop_last=True`, validation uses `drop_last=False`.
- Graph construction uses the training portion of each fold.

## Outputs and evaluation

The script writes `log.txt` and `best_fold_<fold>.pth` under `--run_dir` (default: `./runs/autoimmune_H_binary`). Logs contain fold metrics and aggregate means for AUROC, accuracy, F1, precision, sensitivity and specificity.

Each fold's checkpoint is selected using the highest validation AUROC. The reported fold summaries therefore use the same validation folds for checkpoint selection and evaluation; they are not results from a separate held-out test set.

Only syntax and data-loader checks were performed while preparing this repository. End-to-end training and paper-result reproduction require the corresponding dataset, pretrained weights and a suitable PyTorch environment.

# GRPO 奖励函数使用指南

## 概述

`train_grpo.py` 现在支持两种奖励函数：

1. **MDM 评估器奖励函数** (`reward_model.py`) - 基于 MDM 项目的评估器
2. **TMR 预训练模型奖励函数** (`reward_model_tmr.py`) - 基于 TMR 预训练权重

## 命令行参数

### 基础参数

- `--reward_model_type`: 选择奖励模型类型
  - `mdm` (默认): 使用 MDM 评估器奖励函数
  - `tmr`: 使用 TMR 预训练模型奖励函数

- `--reward_type`: 奖励类型
  - 对于 MDM: `matching`, `r_precision`, `combined`
  - 对于 TMR: `matching`, `cosine`

### TMR 特定参数（仅当 `--reward_model_type=tmr` 时使用）

- `--tmr_checkpoint_path`: TMR 预训练权重路径（必需）
- `--tmr_similarity_type`: 相似度类型 (`cosine` 或 `euclidean`，默认: `cosine`)
- `--tmr_normalization`: 归一化方式 (`linear`, `exponential`, `sigmoid`，默认: `linear`)
- `--tmr_max_distance`: 最大距离（用于线性归一化，默认: `10.0`）
- `--tmr_scale`: 缩放因子（用于指数/Sigmoid 归一化，默认: `2.0`）

## 使用示例

### 1. 使用 MDM 评估器奖励函数（默认）

```bash
# 使用匹配分数（默认）
python -m train.train_grpo \
    --model_path ./save/pretrained_model/model000200000.pt \
    --save_dir ./save/grpo_mdm \
    --dataset humanml \
    --batch_size 2 \
    --group_size 4 \
    --num_steps 10000 \
    --reward_model_type mdm \
    --reward_type matching

# 使用 R-Precision
python -m train.train_grpo \
    --model_path ./save/pretrained_model/model000200000.pt \
    --save_dir ./save/grpo_mdm_rprecision \
    --dataset humanml \
    --batch_size 2 \
    --group_size 4 \
    --num_steps 10000 \
    --reward_model_type mdm \
    --reward_type r_precision

# 使用组合奖励
python -m train.train_grpo \
    --model_path ./save/pretrained_model/model000200000.pt \
    --save_dir ./save/grpo_mdm_combined \
    --dataset humanml \
    --batch_size 2 \
    --group_size 4 \
    --num_steps 10000 \
    --reward_model_type mdm \
    --reward_type combined
```

### 2. 使用 TMR 预训练模型奖励函数

#### 2.1 使用余弦相似度（推荐）

```bash
python -m train.train_grpo \
    --model_path ./save/pretrained_model/model000200000.pt \
    --save_dir ./save/grpo_tmr_cosine \
    --dataset humanml \
    --batch_size 2 \
    --group_size 4 \
    --num_steps 10000 \
    --reward_model_type tmr \
    --reward_type cosine \
    --tmr_checkpoint_path ./path/to/tmr/checkpoint.pth
```

#### 2.2 使用匹配分数（可配置）

```bash
# 使用余弦相似度 + 线性归一化
python -m train.train_grpo \
    --model_path ./save/pretrained_model/model000200000.pt \
    --save_dir ./save/grpo_tmr_matching_cosine \
    --dataset humanml \
    --batch_size 2 \
    --group_size 4 \
    --num_steps 10000 \
    --reward_model_type tmr \
    --reward_type matching \
    --tmr_checkpoint_path ./path/to/tmr/checkpoint.pth \
    --tmr_similarity_type cosine

# 使用欧氏距离 + 线性归一化
python -m train.train_grpo \
    --model_path ./save/pretrained_model/model000200000.pt \
    --save_dir ./save/grpo_tmr_matching_euclidean_linear \
    --dataset humanml \
    --batch_size 2 \
    --group_size 4 \
    --num_steps 10000 \
    --reward_model_type tmr \
    --reward_type matching \
    --tmr_checkpoint_path ./path/to/tmr/checkpoint.pth \
    --tmr_similarity_type euclidean \
    --tmr_normalization linear \
    --tmr_max_distance 10.0

# 使用欧氏距离 + 指数归一化
python -m train.train_grpo \
    --model_path ./save/pretrained_model/model000200000.pt \
    --save_dir ./save/grpo_tmr_matching_euclidean_exp \
    --dataset humanml \
    --batch_size 2 \
    --group_size 4 \
    --num_steps 10000 \
    --reward_model_type tmr \
    --reward_type matching \
    --tmr_checkpoint_path ./path/to/tmr/checkpoint.pth \
    --tmr_similarity_type euclidean \
    --tmr_normalization exponential \
    --tmr_scale 2.0

# 使用欧氏距离 + Sigmoid 归一化
python -m train.train_grpo \
    --model_path ./save/pretrained_model/model000200000.pt \
    --save_dir ./save/grpo_tmr_matching_euclidean_sigmoid \
    --dataset humanml \
    --batch_size 2 \
    --group_size 4 \
    --num_steps 10000 \
    --reward_model_type tmr \
    --reward_type matching \
    --tmr_checkpoint_path ./path/to/tmr/checkpoint.pth \
    --tmr_similarity_type euclidean \
    --tmr_normalization sigmoid \
    --tmr_scale 2.0
```

## 参数说明

### MDM 奖励函数参数

| 参数 | 选项 | 说明 |
|------|------|------|
| `--reward_model_type` | `mdm` | 使用 MDM 评估器 |
| `--reward_type` | `matching` | 基于文本-动作匹配分数（欧氏距离） |
| | `r_precision` | 基于 R-Precision 检索精度 |
| | `combined` | 组合匹配分数和 R-Precision |

### TMR 奖励函数参数

| 参数 | 选项 | 说明 |
|------|------|------|
| `--reward_model_type` | `tmr` | 使用 TMR 预训练模型 |
| `--reward_type` | `cosine` | 余弦相似度（最简单，推荐） |
| | `matching` | 匹配分数（可配置相似度和归一化） |
| `--tmr_checkpoint_path` | 路径 | TMR 预训练权重路径（必需） |
| `--tmr_similarity_type` | `cosine` | 余弦相似度（推荐） |
| | `euclidean` | 欧氏距离 |
| `--tmr_normalization` | `linear` | 线性归一化 |
| | `exponential` | 指数衰减归一化 |
| | `sigmoid` | Sigmoid 归一化 |
| `--tmr_max_distance` | 浮点数 | 最大距离（用于线性归一化，默认: 10.0） |
| `--tmr_scale` | 浮点数 | 缩放因子（用于指数/Sigmoid，默认: 2.0） |

## 推荐配置

### 快速开始（推荐）

**使用 MDM 评估器（无需额外权重）**:
```bash
python -m train.train_grpo \
    --model_path ./save/pretrained_model/model000200000.pt \
    --save_dir ./save/grpo_mdm \
    --dataset humanml \
    --batch_size 2 \
    --group_size 4 \
    --num_steps 10000 \
    --reward_model_type mdm \
    --reward_type matching
```

**使用 TMR（如果有预训练权重）**:
```bash
python -m train.train_grpo \
    --model_path ./save/pretrained_model/model000200000.pt \
    --save_dir ./save/grpo_tmr \
    --dataset humanml \
    --batch_size 2 \
    --group_size 4 \
    --num_steps 10000 \
    --reward_model_type tmr \
    --reward_type cosine \
    --tmr_checkpoint_path ./path/to/tmr/checkpoint.pth
```

## 注意事项

1. **TMR 权重文件**: 使用 TMR 奖励函数时，必须提供 `--tmr_checkpoint_path` 参数
2. **参数兼容性**: 
   - `--tmr_similarity_type`, `--tmr_normalization` 等参数仅在 `--reward_type=matching` 时生效
   - 当 `--reward_type=cosine` 时，这些参数会被忽略
3. **数据集支持**: 两种奖励函数都支持 `humanml` 和 `kit` 数据集
4. **性能**: 
   - MDM 评估器：使用项目内置的评估器，无需额外下载
   - TMR：需要下载预训练权重，但可能提供更好的文本-动作对齐

## 故障排除

### 问题 1: TMR 权重文件未找到

**错误信息**:
```
ValueError: 使用 TMR 奖励模型时，必须提供 --tmr_checkpoint_path 参数
```

**解决方案**: 确保提供了 `--tmr_checkpoint_path` 参数，并且路径正确。

### 问题 2: TMR 权重加载失败

**错误信息**:
```
FileNotFoundError: TMR 权重文件不存在: ...
```

**解决方案**: 
1. 检查权重文件路径是否正确
2. 确保文件格式正确（.pth 或 .tar）
3. 参考 `TMR_REWARD_README.md` 了解权重文件格式要求

### 问题 3: 奖励值异常

**问题**: 奖励值全为 0 或全为 1

**解决方案**:
1. 检查奖励函数是否正确加载
2. 对于 TMR，尝试不同的归一化方式
3. 调整 `--tmr_max_distance` 或 `--tmr_scale` 参数

---

*更多详细信息，请参考 `GRPO_README.md` 和 `TMR_REWARD_README.md`*


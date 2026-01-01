# TMR 奖励模型使用指南

## 概述

`reward_model_tmr.py` 提供了基于 **TMR (Text-to-Motion Retrieval)** 预训练模型的奖励函数实现。TMR 是一个用于文本-动作检索的模型，通过对比学习将文本和动作映射到共同的嵌入空间。

## 特点

- ✅ 使用 TMR 预训练权重，提供更好的文本-动作对齐
- ✅ 支持多种相似度计算方式（余弦相似度、欧氏距离）
- ✅ 支持多种归一化方式（线性、指数、Sigmoid）
- ✅ 与现有 GRPO 训练流程完全兼容

## 安装要求

1. 确保已安装项目依赖
2. 下载 TMR 预训练权重（需要提供权重文件路径）
3. 确保 GloVe 词向量已下载（用于文本处理）

## 使用方法

### 1. 基本使用

```python
from model.GRPO import create_tmr_reward_function

# 创建 TMR 奖励函数
reward_fn = create_tmr_reward_function(
    tmr_checkpoint_path='./path/to/tmr/checkpoint.pth',  # TMR 预训练权重路径
    reward_type='cosine',  # 或 'matching'
    dataset_name='humanml',
    device='cuda',
)

# 在 GRPO 训练中使用
from model.GRPO import create_grpo_trainer

trainer = create_grpo_trainer(
    model=model,
    ref_model=ref_model,
    diffusion=diffusion,
    reward_fn=reward_fn,  # 使用 TMR 奖励函数
    group_size=4,
    ...
)
```

### 2. 使用匹配分数奖励（可配置）

```python
reward_fn = create_tmr_reward_function(
    tmr_checkpoint_path='./path/to/tmr/checkpoint.pth',
    reward_type='matching',
    similarity_type='cosine',  # 或 'euclidean'
    normalization='linear',  # 或 'exponential', 'sigmoid'
    max_distance=10.0,  # 用于线性归一化（仅当 similarity_type='euclidean' 时）
    scale=2.0,  # 用于指数/Sigmoid 归一化（仅当 similarity_type='euclidean' 时）
    dataset_name='humanml',
    device='cuda',
)
```

### 3. 在训练脚本中使用

修改 `train/train_grpo.py`：

```python
from model.GRPO import create_tmr_reward_function

# 创建 TMR 奖励函数
reward_fn = create_tmr_reward_function(
    tmr_checkpoint_path=args.tmr_checkpoint_path,  # 从命令行参数获取
    reward_type=args.reward_type,
    dataset_name=args.dataset,
    device=args.device,
)

# 创建 GRPO 训练器
trainer = create_grpo_trainer(
    model=model,
    ref_model=ref_model,
    diffusion=diffusion,
    reward_fn=reward_fn,
    ...
)
```

### 4. 命令行参数示例

```bash
python -m train.train_grpo \
    --model_path ./save/pretrained_model/model000200000.pt \
    --save_dir ./save/grpo_tmr \
    --dataset humanml \
    --tmr_checkpoint_path ./path/to/tmr/checkpoint.pth \
    --reward_type cosine \
    --batch_size 2 \
    --group_size 4 \
    --use_lora
```

## 奖励函数类型

### 1. `TMRCosineSimilarityReward`

使用余弦相似度计算奖励，最简单直接。

- **相似度范围**: [-1, 1]
- **奖励范围**: [0, 1]（归一化后）
- **优点**: 计算简单，对向量长度不敏感
- **适用场景**: 大多数情况下的默认选择

### 2. `TMRMatchingScoreReward`

可配置的匹配分数奖励，支持多种相似度计算和归一化方式。

**相似度类型**:
- `'cosine'`: 余弦相似度（推荐）
- `'euclidean'`: 欧氏距离

**归一化方式**（仅当 `similarity_type='euclidean'` 时）:
- `'linear'`: 线性归一化 `rewards = 1.0 - clamp(distances / max_distance, 0, 1)`
- `'exponential'`: 指数衰减 `rewards = exp(-distances / scale)`
- `'sigmoid'`: Sigmoid 归一化 `rewards = sigmoid(-distances / scale)`

## TMR 权重文件格式

TMR 权重文件可以是以下格式之一：

1. **标准格式**（字典，包含各个组件）:
```python
{
    'text_encoder': {...},
    'motion_encoder': {...},
    'movement_encoder': {...},
}
```

2. **带状态键的格式**:
```python
{
    'text_encoder_state_dict': {...},
    'motion_encoder_state_dict': {...},
    'movement_encoder_state_dict': {...},
}
```

3. **完整模型格式**（包含前缀）:
```python
{
    'model': {
        'text_encoder.layer1.weight': ...,
        'motion_encoder.layer1.weight': ...,
        ...
    }
}
```

代码会自动尝试不同的格式，如果加载失败，请检查权重文件格式。

## 与 MDM 奖励模型的对比

| 特性 | MDM 奖励模型 | TMR 奖励模型 |
|------|-------------|-------------|
| **基础模型** | MDM 评估器 | TMR 预训练模型 |
| **文本编码** | WordVectorizer + BiGRU | TMR Text Encoder |
| **动作编码** | Movement Encoder + BiGRU | TMR Motion Encoder |
| **相似度计算** | 欧氏距离 | 余弦相似度/欧氏距离 |
| **归一化方式** | 线性（硬编码） | 多种可选 |
| **预训练权重** | MDM 项目提供 | 需要单独下载 |

## 故障排除

### 问题 1: 权重文件加载失败

**错误信息**:
```
FileNotFoundError: TMR 权重文件不存在: ...
```

**解决方案**:
1. 检查权重文件路径是否正确
2. 确保文件格式正确（.pth 或 .tar）
3. 如果权重文件格式不标准，可能需要手动调整加载逻辑

### 问题 2: 模块导入错误

**错误信息**:
```
ImportError: TMR 相关模块未找到
```

**解决方案**:
1. 确保已安装所有依赖
2. 检查 `data_loaders/humanml/networks/modules.py` 是否存在
3. 确保 GloVe 词向量已下载

### 问题 3: 奖励值异常

**问题**: 奖励值全为 0 或全为 1

**解决方案**:
1. 检查 TMR 模型是否正确加载
2. 尝试不同的归一化方式
3. 调整 `max_distance` 或 `scale` 参数
4. 检查文本和动作嵌入是否在合理范围内

## 性能建议

1. **使用余弦相似度**: 通常比欧氏距离更稳定
2. **批量处理**: TMR 模型支持批量处理，可以提高效率
3. **缓存嵌入**: 如果同一个文本被多次使用，可以考虑缓存文本嵌入
4. **GPU 加速**: 确保 TMR 模型在 GPU 上运行

## 示例代码

完整的使用示例请参考 `reward_model_tmr.py` 文件末尾的 `if __name__ == '__main__'` 部分。

## 注意事项

1. **权重文件路径**: 确保 TMR 预训练权重文件路径正确
2. **数据集兼容性**: 当前支持 'humanml' 和 'kit' 数据集
3. **设备一致性**: 确保 TMR 模型和训练模型在同一设备上
4. **内存占用**: TMR 模型会占用额外的 GPU 内存，注意调整批次大小

## 参考

- TMR 论文: [Text-to-Motion Retrieval]（如果可用）
- MDM 项目: 当前项目的 README
- GRPO 训练: `train/GRPO_README.md`




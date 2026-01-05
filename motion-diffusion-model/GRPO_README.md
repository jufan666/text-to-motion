# GRPO (Group Relative Policy Optimization) for Motion Diffusion Models

## 概述

GRPO 是一种无需 Critic 网络的强化学习算法，适用于微调文本生成动作的扩散模型。它通过组内相对优势来替代传统的绝对优势估计，从而避免了价值函数（Critic）的需求，大幅降低显存占用。

## 核心思想

### 1. 组采样 (Group Sampling)
对于每个文本提示 $x$，我们生成 $G$ 个样本（组大小，例如 $G=4$ 或 $8$）：
- 从当前策略 $\pi_\theta$ 采样动作序列 $\{y_1, y_2, ..., y_G\}$
- 所有样本使用相同的初始噪声和条件，确保可比较性

### 2. 奖励计算 (Reward Computation)
使用奖励模型（Reward Model）计算每个生成动作的奖励：
- **奖励模型** (`reward_model.py`): 基于 MDM 评估器，计算文本-动作匹配分数
- 对于每个生成的动作 $y_i$，奖励模型输出标量奖励 $r_i$
- 奖励值通常归一化到 [0, 1] 范围

### 3. 优势估计 (Advantage Estimation)
利用组内统计计算相对优势，无需 Critic：
$$A_i = \frac{r_i - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r}) + \epsilon}$$

**关键**：优势不是由 Critic 网络计算的，而是通过组内归一化得到的相对优势。

### 4. 目标函数
$$\mathcal{L}_{\text{GRPO}} = \frac{1}{G} \sum_{i=1}^G \left[ \min \left( \text{ratio}_i \cdot A_i, \text{clip}(\text{ratio}_i, 1-\epsilon, 1+\epsilon) \cdot A_i \right) - \beta D_{KL}(\pi_\theta || \pi_{\text{ref}}) \right]$$

其中 $\text{ratio}_i = \frac{\pi_\theta(y_i|x)}{\pi_{\text{ref}}(y_i|x)}$。

## 训练流程详解

### 完整训练步骤

1. **Rollout 阶段**：使用当前模型为每个 prompt 生成 $G$ 个样本
2. **Log Prob 计算**：计算当前模型和参考模型对生成轨迹的 log probability
3. **奖励计算**：使用奖励模型计算每个生成动作的奖励值
4. **优势计算**：将奖励转换为组内相对优势
5. **损失计算**：使用优势、ratio 和 KL 惩罚计算 GRPO 损失
6. **梯度更新**：反向传播并更新模型参数（仅 LoRA 参数）

### Log Probability 计算（基于 DanceGRPO 论文）

根据 DanceGRPO 论文的公式 (7) 和 (8)，策略 $\pi_\theta(z_{t-1}|z_t)$ 被建模为高斯分布。

**核心方法 `get_batch_log_prob`**：

1. **轨迹保存**：在采样过程中保存完整轨迹 `[z_T, z_{T-1}, ..., z_0]`
2. **高斯 Log Prob 计算**：对于每个时间步 $t$：
   - 模型预测均值 $\mu_\theta$ 和方差 $\sigma^2$
   - 实际采样点 $z_{t-1}$ 在 $N(\mu_\theta, \sigma^2)$ 下的 log probability：
     $$\log p(z_{t-1}) = -\frac{1}{2}\left(\frac{z_{t-1} - \mu_\theta}{\sigma}\right)^2 - \log(\sigma) - \frac{1}{2}\log(2\pi)$$
3. **累积求和**：对所有时间步的 log prob 求和，得到整条轨迹的累积 log probability

**关键优势**：
- 直接计算连续空间的高斯概率，无需离散化
- 使用实际采样轨迹，计算更准确
- 支持梯度回传，可用于策略优化

### 组采样处理

- 每个 prompt 重复 $G$ 次，形成 `batch_size * group_size` 的扩展批次
- 使用相同的噪声种子确保可比较性
- 组内优势归一化确保训练稳定性

## 项目结构

GRPO 相关代码位于 `model/GRPO/` 目录下：

```
model/
  GRPO/
    __init__.py          # 模块导出
    grpo_trainer.py      # GRPO 训练器核心实现
    reward_model.py      # 基于 MDM 评估器的奖励模型
```

### 使用 MDM 评估器作为奖励模型

项目提供了基于 MDM 评估器的奖励函数实现（`model/GRPO/reward_model.py`）：

```python
from model.GRPO.reward_model import create_mdm_reward_function

# 创建基于匹配分数的奖励函数
reward_fn = create_mdm_reward_function(
    reward_type='matching',  # 'matching', 'r_precision', 或 'combined'
    dataset_name='humanml',
    device='cuda',
)

# 使用
rewards = reward_fn(motions, prompts)  # [B*G]
```

**可用的奖励类型**：
- `'matching'`: 基于文本-动作匹配分数（欧氏距离）
- `'r_precision'`: 基于 R-Precision 检索精度
- `'combined'`: 组合多种指标

#### 为什么只使用 Matching Score 和 R-Precision？

MDM 项目定义了 5 种评估指标，但只有部分适合作为 GRPO 的奖励函数：

| 评估指标 | 是否适合作为奖励 | 原因 |
|---------|----------------|------|
| **Matching Score** | ✅ **适合** | 可以为每个样本单独计算，直接衡量文本-动作匹配度 |
| **R-Precision** | ✅ **适合** | 可以为每个样本单独计算，衡量检索精度 |
| **FID** | ❌ **不适合** | 需要计算整个数据集的统计特征（均值和协方差），是批量指标，不能为单个样本计算 |
| **Diversity** | ❌ **不适合** | 需要多个样本才能计算多样性，是批量指标 |
| **MultiModality** | ❌ **不适合** | 需要为每个文本生成多个样本，然后计算样本间差异，是批量指标 |

**GRPO 对奖励函数的要求**：
1. **单样本计算**：必须能为每个生成的动作序列 $y_i$ 计算一个标量奖励 $r_i$
2. **质量反映**：奖励应该反映该样本的质量（特别是与文本的匹配度）
3. **梯度友好**：奖励值应该能够指导策略优化（通过优势函数）

**推荐使用**：
- **Matching Score**（推荐）：最直接、最常用，计算效率高，直接反映文本-动作对齐质量
- **R-Precision**：可以作为辅助指标，但计算相对复杂
- **Combined**：组合 Matching Score 和 R-Precision，可能获得更好的效果

**注意**：FID、Diversity 和 MultiModality 虽然不适合作为单个样本的奖励，但它们仍然可以作为**训练后的评估指标**，用于评估整体模型性能。

## 参数说明

### GRPOTrainer 参数

- `group_size` (int, default=4): 每个 prompt 的采样数量 $G$
- `clip_epsilon` (float, default=0.2): PPO 风格的裁剪参数
- `kl_penalty` (float, default=0.1): KL 散度惩罚权重 $\beta$
- `advantage_eps` (float, default=1e-8): 优势归一化的数值稳定性参数
- `use_checkpointing` (bool, default=False): 是否使用梯度检查点节省显存

### 训练参数建议

- **Group Size**: 4-8 通常效果较好。更大的组可以提供更稳定的优势估计，但会增加计算成本
- **Learning Rate**: 1e-5 到 1e-4，通常比标准训练更小
- **KL Penalty**: 0.01-0.1，控制策略偏离参考模型的程度
- **Batch Size**: 根据显存调整，注意实际批次大小是 `batch_size * group_size`

## 显存优化

1. **使用 LoRA**：只训练低秩适配器，大幅减少可训练参数
2. **梯度检查点**：设置 `use_checkpointing=True`
3. **减少组大小**：较小的 $G$ 可以减少显存占用
4. **混合精度训练**：可以进一步优化（需要额外实现）

## 注意事项

1. **Log Probability 计算**：当前实现使用轨迹重建方法，对于非常长的序列可能不够精确。对于更精确的计算，需要跟踪实际的采样轨迹。

2. **奖励函数设计**：奖励函数的质量直接影响训练效果。确保奖励函数：
   - 能够区分好样本和坏样本
   - 数值范围合理（建议归一化到 [0, 1] 或 [-1, 1]）
   - 计算效率高（会被频繁调用）

3. **组大小选择**：
   - 太小的 $G$（如 2）可能导致优势估计不稳定
   - 太大的 $G$（如 16+）会增加计算成本，但可能不会带来显著提升

4. **KL 惩罚**：
   - 太小的 $\beta$ 可能导致策略偏离参考模型太快
   - 太大的 $\beta$ 可能限制策略改进

## 故障排除

### 问题 1: 显存不足 (CUDA out of memory)

**错误信息示例**：
```
torch.cuda.OutOfMemoryError: CUDA out of memory. 
Tried to allocate 20.00 MiB (GPU 0; 23.65 GiB total capacity; 
12.84 GiB already allocated; 17.56 MiB free)
```

**原因分析**：
GRPO 训练需要保存完整的扩散轨迹（所有时间步的中间状态），内存占用非常大：
- 每个样本需要保存 `T` 个中间状态（T 是扩散步数，通常 50-1000）
- 实际批次大小 = `batch_size * group_size`
- 总内存 ≈ `batch_size * group_size * T * latent_size`

**解决方案（按优先级）**：

1. **减小批次大小**（最有效）
   ```bash
   # 将 batch_size 从 2 减小到 1
   --batch_size 1
   
   # 或将 group_size 从 4 减小到 2
   --group_size 2
   ```

2. **减小扩散步数**（如果使用 respace）
   - 如果使用 `RespaceDiffusion`，可以减少实际采样步数
   - 注意：这可能会影响生成质量

3. **使用梯度检查点**
   ```python
   trainer = create_grpo_trainer(
       ...,
       use_checkpointing=True,  # 启用梯度检查点
   )
   ```

4. **使用 LoRA**（减少模型参数）
   ```bash
   --use_lora --lora_r 4 --lora_alpha 8  # 使用更小的 LoRA 参数
   ```

5. **清理 GPU 缓存**
   - 代码已自动在计算 log prob 后清理轨迹
   - 如果仍有问题，可以在训练循环中添加：
     ```python
     torch.cuda.empty_cache()
     ```

6. **分批处理**（高级）
   - 如果必须使用大批次，可以实现分批处理逻辑
   - 将 `batch_size * group_size` 分成多个小批次处理

**推荐配置（24GB GPU）**：
- `batch_size=1`, `group_size=4`（总批次 4）
- `batch_size=2`, `group_size=2`（总批次 4）
- 使用 LoRA (`--use_lora`)

**推荐配置（12GB GPU）**：
- `batch_size=1`, `group_size=2`（总批次 2）
- 必须使用 LoRA
- 考虑减小扩散步数

### 问题 2: 训练不稳定
- 检查奖励函数是否合理
- 调整 `kl_penalty` 和 `clip_epsilon`
- 减小学习率

### 问题 3: Log Prob 计算错误
- 确保使用相同的噪声种子
- 检查模型输出是否在合理范围内
- 验证扩散过程的参数设置

## 代码结构说明

### `model/GRPO/grpo_trainer.py`
- `GRPOTrainer`: GRPO 训练器主类
  - `get_batch_log_prob()`: 计算批量 log probability（基于 DanceGRPO 论文）
  - `sample_with_trajectory()`: 采样并保存完整轨迹
  - `compute_group_advantages()`: 计算组相对优势
  - `compute_grpo_loss()`: 计算 GRPO 损失
  - `step()`: 执行一步训练
- `create_grpo_trainer()`: 工厂函数，创建 GRPO 训练器

### `model/GRPO/reward_model.py`
- `MDMRewardFunction`: 奖励函数基类，提供文本和动作预处理功能
- `MatchingScoreReward`: 基于匹配分数的奖励函数
  - 使用 `EvaluatorMDMWrapper` 计算文本和动作嵌入
  - 通过欧氏距离衡量文本-动作匹配度
  - 距离越小，奖励越大
- `RPrecisionReward`: 基于 R-Precision 的奖励函数
  - 衡量在 top-k 检索中正确匹配的比例
- `CombinedMDMReward`: 组合多种指标的奖励函数
  - 可组合匹配分数和 R-Precision
- `create_mdm_reward_function()`: 工厂函数，创建奖励函数

**奖励模型在训练中的使用**：
在 `GRPOTrainer.step()` 方法中，奖励模型被调用来计算生成动作的奖励：
```python
# 在 grpo_trainer.py 的 step() 方法中
rewards = self.reward_fn(motions, expanded_prompts)  # [B*G]
# 然后计算优势
advantages = self.compute_group_advantages(rewards)  # [B*G]
# 最后用于计算 GRPO loss
loss = compute_grpo_loss(log_prob_current, log_prob_ref, advantages)
```





# GRPO Reward 函数设计分析

## 当前实现概述

### 1. 奖励函数架构

GRPO 的奖励函数基于 **MDM 评估器**（`EvaluatorMDMWrapper`），该评估器将文本和动作映射到共同的嵌入空间。

**核心流程**：

```
文本提示 → 文本嵌入 (512维)
动作序列 → 动作嵌入 (512维)
↓
计算嵌入距离 → 转换为奖励值
```

### 2. 三种奖励函数实现

#### (1) MatchingScoreReward（默认，最常用）

**实现逻辑**：

```python
# 1. 获取文本和动作嵌入
text_embeddings, motion_embeddings = evaluator.get_co_embeddings(...)

# 2. 计算欧氏距离
distances = torch.norm(text_embeddings - motion_embeddings, dim=-1)  # [B]

# 3. 线性归一化到 [0, 1]
max_distance = 10.0  # 硬编码
rewards = 1.0 - torch.clamp(distances / max_distance, 0, 1)
```

**特点**：

- ✅ 简单直接，计算效率高
- ✅ 奖励范围固定为 [0, 1]
- ❌ `max_distance=10.0` 硬编码，可能不适合所有情况
- ❌ 距离 > 10 时奖励为 0，可能丢失信息

#### (2) RPrecisionReward

**实现逻辑**：

```python
# 计算距离矩阵
dist_mat = euclidean_distance_matrix(text_emb, motion_emb)

# 对于每个样本，检查是否在 top-k 中
if i in top_k_indices:
    reward = 1.0
else:
    reward = 1.0 / (1.0 + distances[i])  # 距离倒数
```

**特点**：

- ✅ 考虑了相对排名
- ❌ 计算复杂度较高（需要计算距离矩阵）
- ❌ 奖励分布可能不够平滑

#### (3) CombinedMDMReward

**实现逻辑**：

```python
combined_rewards = (
    matching_weight * matching_rewards +
    r_precision_weight * r_precision_rewards
)
```

**特点**：

- ✅ 结合多种指标，可能更全面
- ❌ 计算成本更高
- ❌ 需要调优权重

---

## ✅ 设计合理性分析

### 优点

1. **基于成熟的评估器**
   - 使用 MDM 项目已有的评估器，经过验证
   - 文本和动作嵌入在联合空间中，语义对齐良好

2. **单样本可计算**
   - 满足 GRPO 的要求：可以为每个样本单独计算奖励
   - 不依赖批量统计（如 FID、Diversity）

3. **数值范围合理**
   - 奖励归一化到 [0, 1]，便于优势计算
   - 避免了奖励尺度问题

4. **与评估指标一致**
   - Matching Score 是 MDM 论文中的主要评估指标
   - 奖励函数与评估指标对齐，训练目标明确

### 潜在问题

#### 🔴 问题 1: 硬编码的 max_distance

**当前实现**：

```python
max_distance = 10.0  # 硬编码
rewards = 1.0 - torch.clamp(distances / max_distance, 0, 1)
```

**问题**：

- 如果实际距离分布不在 [0, 10] 范围内，奖励会饱和
- 例如：如果距离通常在 [0, 5]，那么大部分奖励在 [0.5, 1.0]，区分度不够
- 如果距离经常 > 10，奖励会被截断为 0，丢失信息

**建议改进**：

```python
# 方法 1: 自适应归一化（基于历史统计）
# 在训练开始时，采样一批样本，计算距离分布
# 使用分位数（如 95% 分位数）作为 max_distance

# 方法 2: 使用指数衰减（更平滑）
scale = 2.0  # 可调参数
rewards = torch.exp(-distances / scale)

# 方法 3: 使用分位数归一化
# 计算当前批次的距离分位数，动态调整
```

#### 🔴 问题 2: 奖励分布可能不够敏感

**问题**：

- 线性归一化可能导致奖励分布集中在某个范围
- 对于距离差异小的样本，奖励差异可能不够明显

**建议**：

- 使用非线性变换（如 sigmoid、tanh）增强区分度
- 或者使用排名归一化（rank normalization）

#### 🔴 问题 3: 没有考虑动作质量的其他维度

**当前设计只考虑文本-动作匹配度**，但动作质量还包括：

- **流畅性**：动作是否自然、连贯
- **多样性**：动作是否过于单调
- **物理合理性**：是否符合物理规律（如足部接触）

**建议**：

- 可以添加额外的奖励项（需要额外的评估器）
- 或者使用组合奖励函数

#### 🔴 问题 4: RPrecisionReward 的实现可能有问题

**当前实现**：

```python
for i in range(batch_size):
    distances = dist_mat[i]
    top_k_indices = np.argsort(distances)[:self.top_k]
    if i in top_k_indices:
        reward = 1.0
    else:
        reward = 1.0 / (1.0 + distances[i])
```

**问题**：

- 在 GRPO 中，每个 prompt 生成 G 个样本，这些样本应该与同一个文本比较
- 但当前实现中，`dist_mat[i]` 是第 i 个文本与所有动作的距离，这可能不是我们想要的
- 应该计算：对于第 i 个文本，它在所有动作中的排名

**建议修正**：

```python
# 对于每个文本-动作对，计算该动作在所有动作中的排名
for i in range(batch_size):
    # 获取第 i 个文本对应的动作距离
    distances = dist_mat[i]  # [batch_size]
    # 计算排名（距离越小，排名越靠前）
    rank = (distances < distances[i]).sum()  # 有多少个动作距离更小
    # 如果排名在 top-k 中，给予高奖励
    if rank < self.top_k:
        reward = 1.0
    else:
        # 使用排名倒数作为奖励
        reward = 1.0 / (1.0 + rank)
```

---

## 💡 改进建议

### 改进 1: 自适应归一化

```python
class AdaptiveMatchingScoreReward(MDMRewardFunction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.distance_history = []
        self.max_history_size = 1000
        self.percentile = 95  # 使用 95% 分位数
    
    def __call__(self, motions, prompts, lengths=None):
        # ... 计算距离 ...
        distances = torch.norm(text_embeddings - motion_embeddings, dim=-1)
        
        # 更新历史
        self.distance_history.extend(distances.cpu().tolist())
        if len(self.distance_history) > self.max_history_size:
            self.distance_history = self.distance_history[-self.max_history_size:]
        
        # 计算自适应阈值
        if len(self.distance_history) > 100:
            max_distance = np.percentile(self.distance_history, self.percentile)
        else:
            max_distance = 10.0  # 初始值
        
        # 归一化
        rewards = 1.0 - torch.clamp(distances / max_distance, 0, 1)
        return rewards
```

### 改进 2: 使用指数衰减（更平滑）

```python
class ExponentialMatchingScoreReward(MDMRewardFunction):
    def __init__(self, scale=2.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scale = scale  # 控制衰减速度
    
    def __call__(self, motions, prompts, lengths=None):
        # ... 计算距离 ...
        distances = torch.norm(text_embeddings - motion_embeddings, dim=-1)
        
        # 指数衰减：距离越大，奖励越小
        rewards = torch.exp(-distances / self.scale)
        
        # 可选：归一化到 [0, 1]
        # rewards = (rewards - rewards.min()) / (rewards.max() - rewards.min() + 1e-8)
        
        return rewards
```

### 改进 3: 组合奖励（考虑多个维度）

```python
class MultiDimensionalReward(MDMRewardFunction):
    def __init__(
        self,
        matching_weight=0.6,
        smoothness_weight=0.2,
        diversity_weight=0.2,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.matching_weight = matching_weight
        self.smoothness_weight = smoothness_weight
        self.diversity_weight = diversity_weight
    
    def _compute_smoothness(self, motions):
        """计算动作流畅性（速度变化）"""
        # 计算速度
        velocities = motions[..., 1:] - motions[..., :-1]
        # 计算加速度
        accelerations = velocities[..., 1:] - velocities[..., :-1]
        # 流畅性 = 负的加速度变化（变化越小越流畅）
        smoothness = -torch.norm(accelerations, dim=-1).mean(dim=-1)
        return smoothness
    
    def __call__(self, motions, prompts, lengths=None):
        # 匹配度奖励
        matching_rewards = self._compute_matching(motions, prompts, lengths)
        
        # 流畅性奖励
        smoothness_rewards = self._compute_smoothness(motions)
        smoothness_rewards = torch.sigmoid(smoothness_rewards)  # 归一化
        
        # 组合
        rewards = (
            self.matching_weight * matching_rewards +
            self.smoothness_weight * smoothness_rewards
        )
        return rewards
```

### 改进 4: 组内归一化奖励（更适合 GRPO）

```python
class GroupNormalizedReward(MDMRewardFunction):
    """
    在组内归一化奖励，确保组内奖励分布合理
    """
    def __call__(self, motions, prompts, lengths=None, group_size=None):
        # 计算原始奖励
        raw_rewards = self._compute_raw_rewards(motions, prompts, lengths)
        
        if group_size is not None:
            # 在组内归一化
            batch_size = raw_rewards.shape[0] // group_size
            rewards_reshaped = raw_rewards.view(batch_size, group_size)
            
            # 组内归一化到 [0, 1]
            group_min = rewards_reshaped.min(dim=1, keepdim=True)[0]
            group_max = rewards_reshaped.max(dim=1, keepdim=True)[0]
            group_range = group_max - group_min + 1e-8
            
            normalized_rewards = (rewards_reshaped - group_min) / group_range
            return normalized_rewards.view(-1)
        else:
            return raw_rewards
```





# GRPO 奖励函数使用

## 概述

`train_grpo.py` 现在支持两种奖励函数：

1. **MDM 评估器奖励函数** (`reward_model.py`) - 基于 MDM 项目的评估器
2. **TMR 预训练模型奖励函数** (`reward_model_tmr.py`) - 基于 TMR 预训练权重

## 使用示例(见训练使用)

### 1. 使用 MDM 评估器奖励函数

### 2. 使用 TMR 预训练模型奖励函数

#### 2.1 使用余弦相似度（推荐）

#### 2.2 使用匹配分数（可配置）

## 故障排除

### 问题 1: TMR 权重文件未找到

**错误信息**:

```
ValueError: 使用 TMR 奖励模型时，必须提供 --tmr_text_encoder_path 参数
ValueError: 使用 TMR 奖励模型时，必须提供 --tmr_motion_encoder_path 参数
ValueError: 使用 TMR 奖励模型时，必须提供 --tmr_movement_encoder_path 参数
```

**解决方案**: 确保提供了三个独立的权重文件路径：
- `--tmr_text_encoder_path`: text_encoder.pt
- `--tmr_motion_encoder_path`: motion_encoder.pt
- `--tmr_movement_encoder_path`: motion_decoder.pt

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







# 训练使用

### 使用训练脚本()

- 不挂lora的状态下

  ```bash
  python -m train.train_grpo --model_path ./save/official_humanml_enc_512_50steps/model000750000.pt --save_dir ./save/grpo_finetuned --dataset humanml --batch_size 1 --group_size 4 --num_steps 10000 --learning_rate 1e-6 --reward_model_type mdm  --reward_type matching --device 0 
  ```

- 挂lora的状态

  ```bash
  python -m train.train_grpo --model_path ./save/pretrained_model/model000200000.pt --save_dir ./save/grpo_finetuned --dataset humanml --batch_size 1 --group_size 4 --num_steps 10000 --learning_rate 5e-7 --use_lora --lora_r 8 lora_alpha 16 --reward_model_type mdm \ --reward_type matching --device 0 
  ```

- 使用mdm评估函数作为reward_model

  ```bash
  # 使用匹配分数（默认）
  python -m train.train_grpo --model_path ./save/official_humanml_enc_512_50steps/model000750000.pt --save_dir ./save/grpo_finetuned_humanml_enc_512_50steps_750000_mdm_matching --dataset humanml --batch_size 1 --group_size 4 --learning_rate 1e-6 --num_steps 15000 --reward_model_type mdm --reward_type matching --device 3
  
  # 使用 R-Precision
  python -m train.train_grpo --model_path ./save/official_humanml_enc_512_50steps/model000750000.pt --save_dir ./save/grpo_finetuned_humanml_enc_512_50steps_750000_r_precision --dataset humanml --batch_size 1 --group_size 4 --learning_rate 1e-6 --num_steps 15000 --reward_model_type mdm --reward_type r_precision --device 4  
  
  # 使用组合奖励
  python -m train.train_grpo --model_path ./save/official_humanml_enc_512_50steps/model000750000.pt --save_dir ./save/grpo_finetuned_humanml_enc_512_50steps_750000_mdm_combined --dataset humanml --batch_size 1 --group_size 4 --learning_rate 1e-6 --num_steps 15000 --reward_model_type mdm  --reward_type combined  --device 1  
  ```

- 使用 TMR 预训练模型奖励函数

  余弦相似度

  ```bash
  python -m train.train_grpo --model_path ./save/official_humanml_enc_512_50steps/model000750000.pt --save_dir ./save/grpo_finetuned_humanml_enc_512_50steps_750000_tmr_cosine --dataset humanml  --batch_size 1 --group_size 4 --learning_rate 1e-6  --num_steps 15000 --reward_model_type tmr --reward_type cosine --tmr_text_encoder_path ./model/GRPO/tmr_weights/text_encoder.pt --tmr_motion_encoder_path ./model/GRPO/tmr_weights/motion_encoder.pt --tmr_movement_encoder_path ./model/GRPO/tmr_weights/motion_decoder.pt --device 2
  ```

  匹配分数

  ```bash
  # 使用余弦相似度 + 线性归一化
  python -m train.train_grpo --model_path ./save/official_humanml_enc_512_50steps/model000750000.pt --save_dir ./save/grpo_finetuned_humanml_enc_512_50steps_750000_tmr_matching_cosine --dataset humanml --batch_size 1 --group_size 4 --num_steps 10000 --reward_model_type tmr  --reward_type matching  --tmr_text_encoder_path ./model/GRPO/tmr_weights/text_encoder.pt --tmr_motion_encoder_path ./model/GRPO/tmr_weights/motion_encoder.pt --tmr_movement_encoder_path ./model/GRPO/tmr_weights/motion_decoder.pt  --tmr_similarity_type cosine --device 1
  
  # 使用欧氏距离 + 线性归一化
  python -m train.train_grpo --model_path ./save/official_humanml_enc_512_50steps/model000750000.pt  --save_dir ./save/grpo_finetuned_humanml_enc_512_50steps_750000_tmr_matching_euclidean_linear --dataset humanml --batch_size 1 --group_size 4 --num_steps 10000 --reward_model_type tmr --reward_type matching --tmr_text_encoder_path ./model/GRPO/tmr_weights/text_encoder.pt --tmr_motion_encoder_path ./model/GRPO/tmr_weights/motion_encoder.pt --tmr_movement_encoder_path ./model/GRPO/tmr_weights/motion_decoder.pt --tmr_similarity_type euclidean --tmr_normalization linear --tmr_max_distance 10.0 --device 0
  
  # 使用欧氏距离 + 指数归一化
  python -m train.train_grpo --model_path ./save/official_humanml_enc_512_50steps/model000750000.pt --save_dir ./save/grpo_finetuned_humanml_enc_512_50steps_750000_tmr_matching_euclidean_exp --dataset humanml --batch_size 1 --group_size 4 --num_steps 10000 --reward_model_type tmr --reward_type matching 
  --tmr_text_encoder_path ./model/GRPO/tmr_weights/text_encoder.pt --tmr_motion_encoder_path ./model/GRPO/tmr_weights/motion_encoder.pt --tmr_movement_encoder_path ./model/GRPO/tmr_weights/motion_decoder.pt --tmr_similarity_type euclidean --tmr_normalization exponential --tmr_scale 2.0 --device 1
  
  # 使用欧氏距离 + Sigmoid 归一化
  python -m train.train_grpo --model_path ./save/official_humanml_enc_512_50steps/model000750000.pt --save_dir /save/grpo_finetuned_humanml_enc_512_50steps_750000_tmr_matching_euclidean_sigmoid --dataset humanml --batch_size 1 --group_size 4 --num_steps 10000 --reward_model_type tmr --reward_type matching 
  --tmr_text_encoder_path ./model/GRPO/tmr_weights/text_encoder.pt --tmr_motion_encoder_path ./model/GRPO/tmr_weights/motion_encoder.pt --tmr_movement_encoder_path ./model/GRPO/tmr_weights/motion_decoder.pt --tmr_similarity_type euclidean --tmr_normalization sigmoid --tmr_scale 2.0 --device 2
  ```
  
  

**奖励类型选项**：
- `--reward_model_type`: 选择奖励模型类型
  - `mdm` (默认): 使用 MDM 评估器奖励函数
  - `tmr`: 使用 TMR 预训练模型奖励函数

- `--reward_type`: 奖励类型
  - 对于 MDM: `matching`, `r_precision`, `combined`
  - 对于 TMR: `matching`, `cosine`

### MDM 奖励函数参数

| 参数                  | 选项          | 说明                              |
| --------------------- | ------------- | --------------------------------- |
| `--reward_model_type` | `mdm`         | 使用 MDM 评估器                   |
| `--reward_type`       | `matching`    | 基于文本-动作匹配分数（欧氏距离） |
|                       | `r_precision` | 基于 R-Precision 检索精度         |
|                       | `combined`    | 组合匹配分数和 R-Precision        |

### TMR 奖励函数参数

| 参数                          | 选项          | 说明                                             |
| ----------------------------- | ------------- | ------------------------------------------------ |
| `--reward_model_type`         | `tmr`         | 使用 TMR 预训练模型                              |
| `--reward_type`               | `cosine`      | 余弦相似度（最简单，推荐）                       |
|                               | `matching`    | 匹配分数（可配置相似度和归一化）                 |
| `--tmr_text_encoder_path`     | 路径          | TMR 文本编码器权重路径 (text_encoder.pt，必需)   |
| `--tmr_motion_encoder_path`   | 路径          | TMR 动作编码器权重路径 (motion_encoder.pt，必需) |
| `--tmr_movement_encoder_path` | 路径          | TMR 动作解码器权重路径 (motion_decoder.pt，必需) |
| `--tmr_similarity_type`       | `cosine`      | 余弦相似度（推荐）                               |
|                               | `euclidean`   | 欧氏距离                                         |
| `--tmr_normalization`         | `linear`      | 线性归一化                                       |
|                               | `exponential` | 指数衰减归一化                                   |
|                               | `sigmoid`     | Sigmoid 归一化                                   |
| `--tmr_max_distance`          | 浮点数        | 最大距离（用于线性归一化，默认: 10.0）           |
| `--tmr_scale`                 | 浮点数        | 缩放因子（用于指数/Sigmoid，默认: 2.0）          |

**TMR特定参数**

- `--tmr_text_encoder_path`: TMR 文本编码器权重路径 (text_encoder.pt，必需)
- `--tmr_motion_encoder_path`: TMR 动作编码器权重路径 (motion_encoder.pt，必需)
- `--tmr_movement_encoder_path`: TMR 动作解码器权重路径 (motion_decoder.pt，必需)
- `--tmr_similarity_type`: 相似度类型 (`cosine` 或 `euclidean`，默认: `cosine`)
- `--tmr_normalization`: 归一化方式 (`linear`, `exponential`, `sigmoid`，默认: `linear`)
- `--tmr_max_distance`: 最大距离（用于线性归一化，默认: `10.0`）
- `--tmr_scale`: 缩放因子（用于指数/Sigmoid 归一化，默认: `2.0`）

## 注意事项

1. **TMR 权重文件**: 使用 TMR 奖励函数时，必须提供三个独立的权重文件路径：
   - `--tmr_text_encoder_path`: 文本编码器权重 (text_encoder.pt)
   - `--tmr_motion_encoder_path`: 动作编码器权重 (motion_encoder.pt)
   - `--tmr_movement_encoder_path`: 动作解码器权重 (motion_decoder.pt)
2. **参数兼容性**: 
   - `--tmr_similarity_type`, `--tmr_normalization` 等参数仅在 `--reward_type=matching` 时生效
   - 当 `--reward_type=cosine` 时，这些参数会被忽略
3. **数据集支持**: 两种奖励函数都支持 `humanml` 和 `kit` 数据集
4. **性能**: 
   - MDM 评估器：使用项目内置的评估器，无需额外下载
   - TMR：需要下载三个预训练权重文件，但可能提供更好的文本-动作对齐

**可视化平台选项**：

- `--train_platform_type NoPlatform`: 不使用可视化（默认）
- `--train_platform_type TensorboardPlatform`: 使用 TensorBoard
- `--train_platform_type WandBPlatform`: 使用 Weights & Biases
- `--train_platform_type ClearmlPlatform`: 使用 ClearML

**使用 TensorBoard 查看训练进度**：

```bash
# 训练时使用 TensorBoard
python -m train.train_grpo ... --train_platform_type TensorboardPlatform

# 在另一个终端启动 TensorBoard
tensorboard --logdir ./save/grpo_finetuned
```

**记录的训练指标**：

- **Loss**: `loss`, `policy_loss`, `kl_penalty`
- **Reward**: `mean_reward`, `std_reward`, `min_reward`, `max_reward`
- **Advantage**: `mean_advantage`, `std_advantage`
- **LogProb**: `mean_log_prob_current`, `mean_log_prob_ref`, `mean_ratio`
- **Training**: `grad_norm`, `learning_rate`

- 评估命令
python -m eval.eval_humanml --model_path ./save/humanml_trans_enc_512/model000475000.pt
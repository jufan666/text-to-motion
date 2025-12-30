# GRPO Reward 函数设计分析

## 📊 当前实现概述

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

---

## 🎯 总结与建议

### 当前设计的合理性评分：**7/10**

**优点**：
- ✅ 基于成熟的评估器，可靠
- ✅ 实现简单，计算高效
- ✅ 与评估指标对齐

**需要改进的地方**：
- ⚠️ 硬编码的 `max_distance` 应该改为自适应
- ⚠️ 奖励分布可能不够敏感
- ⚠️ 可以考虑添加其他质量维度

### 推荐方案

**短期改进**（快速实施）：
1. 将 `max_distance` 改为可配置参数，并在训练脚本中提供
2. 添加指数衰减选项作为替代方案
3. 修复 `RPrecisionReward` 的实现

**中期改进**（需要更多开发）：
1. 实现自适应归一化
2. 添加流畅性奖励项
3. 实现组内归一化奖励

**长期改进**（研究性）：
1. 学习奖励函数（使用神经网络）
2. 多目标奖励优化
3. 人类反馈强化学习（RLHF）

---

## 📝 使用建议

### 当前使用方式（推荐）

```python
# 使用默认的 MatchingScoreReward
reward_fn = create_mdm_reward_function('matching', device='cuda')
```

### 如果遇到问题

1. **奖励区分度不够**：
   - 尝试使用 `'combined'` 类型
   - 或手动调整 `max_distance` 参数

2. **训练不稳定**：
   - 检查奖励分布（应该在 [0, 1] 范围内）
   - 确保组内奖励有足够差异（std > 0.1）

3. **奖励饱和**：
   - 如果大部分奖励接近 1.0，说明 `max_distance` 太大
   - 如果大部分奖励接近 0.0，说明 `max_distance` 太小

---

*本文档基于代码分析生成，建议结合实际训练效果进行调整。*


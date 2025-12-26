"""
GRPO (Group Relative Policy Optimization) 训练器，用于动作扩散模型

本实现将 GRPO 应用于使用 LoRA 微调的文本生成动作扩散模型，
无需 Critic 网络（价值函数）。

参考：受 DDPO (Differentiable Diffusion Policy Optimization) 和
语言模型的 GRPO 启发，参考 DanceGRPO 论文 (arXiv:2505.07818v4)。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Optional, Tuple, Callable
from tqdm import tqdm
import copy

from diffusion.nn import mean_flat
from diffusion.losses import normal_kl
from diffusion.gaussian_diffusion import _extract_into_tensor


class GRPOTrainer:
    
    def __init__(
        self,
        model: nn.Module,
        ref_model: nn.Module,
        diffusion: 'GaussianDiffusion',
        optimizer: torch.optim.Optimizer,
        reward_fn: Callable[[torch.Tensor, List[str]], torch.Tensor],
        group_size: int = 4,
        clip_epsilon: float = 0.2,
        kl_penalty: float = 0.1,
        advantage_eps: float = 1e-8,
        device: str = 'cuda',
        use_checkpointing: bool = False,
    ):
        """
        初始化 GRPO 训练器。
        
        参数:
            model: 可训练的模型（带 LoRA）用于优化
            ref_model: 冻结的参考模型，用于 KL 惩罚
            diffusion: 扩散调度器
            optimizer: 优化器
            reward_fn: 计算奖励的函数: (motions, prompts) -> rewards
            group_size: 每个 prompt 的采样数量（论文中的 G）
            clip_epsilon: PPO 风格的裁剪参数
            kl_penalty: KL 散度惩罚权重（beta）
            advantage_eps: 优势归一化的数值稳定性小量
            device: 运行设备
            use_checkpointing: 是否使用梯度检查点以节省显存
        """
        self.model = model
        self.ref_model = ref_model
        self.diffusion = diffusion
        self.optimizer = optimizer
        self.reward_fn = reward_fn
        self.group_size = group_size
        self.clip_epsilon = clip_epsilon
        self.kl_penalty = kl_penalty
        self.advantage_eps = advantage_eps
        self.device = device
        self.use_checkpointing = use_checkpointing
        
        # 确保参考模型被冻结
        for param in self.ref_model.parameters():
            param.requires_grad = False
        self.ref_model.eval()
    
    def get_batch_log_prob(
        self,
        model: nn.Module,
        latents_sequence: List[torch.Tensor],
        timesteps_sequence: List[torch.Tensor],
        model_kwargs: Dict,
    ) -> torch.Tensor:
        """
        根据 DanceGRPO 论文公式 (7) 和 (8) 计算批量 log probability。
        
        策略 π_θ(z_{t-1}|z_t) 被建模为高斯分布。对于每个时间步 t，
        我们计算实际采样点 z_{t-1} 在模型预测的高斯分布下的 log probability。
        
        高斯分布的 log probability 公式：
        log p(x) = -0.5 * ((x - μ) / σ)² - log(σ) - 0.5 * log(2π)
        
        参数:
            model: 用于计算的模型
            latents_sequence: 采样轨迹序列 [z_T, z_{T-1}, ..., z_0]
                - 每个元素形状为 [B, C, H, W]
                - 长度为 num_timesteps + 1
            timesteps_sequence: 对应的时间步序列 [T, T-1, ..., 0]
                - 每个元素形状为 [B]
            model_kwargs: 条件信息（如文本嵌入）
            
        返回:
            log_prob: 累积 log probability [B]
        """
        batch_size = latents_sequence[0].shape[0]
        device = latents_sequence[0].device
        
        # 累积所有时间步的 log probability
        total_log_prob = torch.zeros(batch_size, device=device)
        
        # 遍历每个时间步（从 T 到 1）
        # latents_sequence[0] = z_T (初始噪声)
        # latents_sequence[1] = z_{T-1} (从 z_T 采样得到)
        # latents_sequence[i] 是 z_{T-i+1}，latents_sequence[i+1] 是 z_{T-i}
        # timesteps_sequence[i] 对应从 latents_sequence[i] 到 latents_sequence[i+1] 的转换时间步
        for i in range(len(latents_sequence) - 1):
            z_t = latents_sequence[i]  # 当前状态 z_t [B, C, H, W]
            z_t_minus_1 = latents_sequence[i + 1]  # 实际采样的下一步 z_{t-1} [B, C, H, W]
            t = timesteps_sequence[i]  # 从 z_t 到 z_{t-1} 的时间步 [B]
            
            # 1. 模型预测：获取噪声预测或 x_0 预测
            with torch.set_grad_enabled(model.training):
                out = self.diffusion.p_mean_variance(
                    model,
                    z_t,
                    t,
                    clip_denoised=False,  # 计算 log prob 时不裁剪
                    model_kwargs=model_kwargs,
                )
            
            # 2. 获取高斯分布的参数（均值和方差）
            # out["mean"] 是模型预测的均值 μ_θ
            # out["log_variance"] 是对数方差
            dist_mean = out["mean"]  # [B, C, H, W] - 模型预测的"理想下一步"均值
            dist_log_variance = out["log_variance"]  # [B, C, H, W]
            dist_variance = torch.exp(dist_log_variance)  # [B, C, H, W]
            dist_std = torch.exp(0.5 * dist_log_variance)  # [B, C, H, W] - 标准差
            
            # 3. 计算高斯 log probability
            # 实际采取的下一步是 z_{t-1}（在 Rollout 阶段采样保存的）
            # 计算 z_{t-1} 在 N(μ_θ, σ²) 下的 log probability
            
            # 标准化误差
            squared_error = (z_t_minus_1 - dist_mean) ** 2  # [B, C, H, W]
            
            # 高斯 log probability: log p(x) = -0.5 * ((x-μ)/σ)² - log(σ) - 0.5*log(2π)
            log_prob_per_element = (
                -0.5 * (squared_error / dist_variance)
                - dist_log_variance * 0.5  # log(σ) = 0.5 * log(σ²)
                - 0.5 * math.log(2 * math.pi)
            )  # [B, C, H, W]
            
            # 对所有空间维度求和，得到每个样本的 log prob
            # 根据 DanceGRPO，我们对所有维度求和
            log_prob_per_timestep = log_prob_per_element.sum(dim=[1, 2, 3])  # [B]
            
            # 累积到总 log probability
            total_log_prob = total_log_prob + log_prob_per_timestep
        
        return total_log_prob
    
    def sample_with_trajectory(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        model_kwargs: Dict,
        noise: Optional[torch.Tensor] = None,
        save_trajectory: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        从模型采样并保存完整轨迹，用于后续 log probability 计算。
        
        参数:
            model: 用于采样的模型
            shape: 采样形状 [B, C, H, W]
            model_kwargs: 条件信息
            noise: 可选的固定噪声（用于可重现性和 log prob 计算）
            save_trajectory: 是否保存完整轨迹
            
        返回:
            字典包含:
                - 'samples': 生成的样本 [B, C, H, W]
                - 'latents_sequence': 完整轨迹序列（如果 save_trajectory=True）
                - 'timesteps_sequence': 时间步序列（如果 save_trajectory=True）
        """
        batch_size = shape[0]
        device = next(model.parameters()).device
        
        if noise is None:
            noise = torch.randn(*shape, device=device)
        else:
            noise = noise.to(device)
        
        # 初始化
        x_t = noise.clone()
        timesteps = list(range(self.diffusion.num_timesteps))[::-1]
        
        # 保存轨迹
        # latents_sequence[0] = z_T (初始噪声)
        # latents_sequence[1] = z_{T-1}
        # ...
        # latents_sequence[T] = z_0 (最终样本)
        latents_sequence = [x_t.clone()] if save_trajectory else []
        timesteps_sequence = []
        
        # 渐进式采样
        for i, t_val in enumerate(timesteps):
            t = torch.full((batch_size,), t_val, device=device, dtype=torch.long)
            
            # 保存当前状态和时间步（用于后续 log prob 计算）
            if save_trajectory and i > 0:
                # 保存当前状态 z_t（在采样之前）
                # 注意：第一次循环时，x_t 已经是 z_T，已在上面保存
                pass  # 当前 x_t 已在上一轮保存
            
            # 获取模型预测
            out = self.diffusion.p_mean_variance(
                model,
                x_t,
                t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )
            
            pred_xstart = out["pred_xstart"]
            model_mean = out["mean"]
            model_log_variance = out["log_variance"]
            
            # 采样下一步 z_{t-1}
            nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
            noise_step = torch.randn_like(x_t)
            x_t = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise_step
            
            # 保存采样后的状态和时间步
            if save_trajectory:
                latents_sequence.append(x_t.clone())
                timesteps_sequence.append(t.clone())
        
        result = {
            'samples': x_t,
        }
        
        if save_trajectory:
            result['latents_sequence'] = latents_sequence
            result['timesteps_sequence'] = timesteps_sequence
        
        return result
    
    def compute_group_advantages(
        self,
        rewards: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算组相对优势。
        
        对于每个组（来自同一 prompt 的 G 个样本），计算：
        A_i = (r_i - mean(r_group)) / (std(r_group) + eps)
        
        参数:
            rewards: 奖励张量 [B*G]，其中 B 是批次大小，G 是组大小
            
        返回:
            advantages: 组相对优势 [B*G]
        """
        batch_size = rewards.shape[0] // self.group_size
        rewards_reshaped = rewards.view(batch_size, self.group_size)
        
        # 计算组统计量
        group_mean = rewards_reshaped.mean(dim=1, keepdim=True)  # [B, 1]
        group_std = rewards_reshaped.std(dim=1, keepdim=True)  # [B, 1]
        
        # 在每个组内归一化
        advantages = (rewards_reshaped - group_mean) / (group_std + self.advantage_eps)
        
        return advantages.view(-1)  # [B*G]
    
    def compute_kl_penalty(
        self,
        log_prob_current: torch.Tensor,
        log_prob_ref: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算 KL 散度惩罚: D_KL(π_θ || π_ref)。
        
        对于扩散模型，我们近似为：
        D_KL ≈ E[log π_θ(y|x) - log π_ref(y|x)]
        
        参数:
            log_prob_current: 当前模型下的 log probs [B*G]
            log_prob_ref: 参考模型下的 log probs [B*G]
            
        返回:
            kl_penalty: 每个样本的 KL 散度 [B*G]
        """
        return log_prob_current - log_prob_ref
    
    def compute_grpo_loss(
        self,
        log_prob_current: torch.Tensor,
        log_prob_ref: torch.Tensor,
        advantages: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算 GRPO 损失。
        
        L_GRPO = (1/G) * Σ_i [min(ratio_i * A_i, clip(ratio_i, 1-ε, 1+ε) * A_i) - β * KL_i]
        
        参数:
            log_prob_current: 当前模型下的 log probs [B*G]
            log_prob_ref: 参考模型下的 log probs [B*G]
            advantages: 组相对优势 [B*G]
            
        返回:
            loss: 标量损失值
            stats: 用于记录的统计信息字典
        """
        # 计算 ratio
        ratio = torch.exp(log_prob_current - log_prob_ref)  # [B*G]
        
        # 计算 KL 惩罚
        kl = self.compute_kl_penalty(log_prob_current, log_prob_ref)
        
        # PPO 风格的裁剪目标
        ratio_clipped = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
        
        # 策略损失: min(ratio * A, clipped_ratio * A)
        policy_loss_1 = ratio * advantages
        policy_loss_2 = ratio_clipped * advantages
        policy_loss = torch.min(policy_loss_1, policy_loss_2)
        
        # 总损失: 策略损失 - KL 惩罚
        loss = -(policy_loss.mean() - self.kl_penalty * kl.mean())
        
        # 计算统计信息
        stats = {
            'loss': loss.item(),
            'policy_loss': -policy_loss.mean().item(),
            'kl_penalty': kl.mean().item(),
            'mean_ratio': ratio.mean().item(),
            'mean_advantage': advantages.mean().item(),
            'mean_log_prob_current': log_prob_current.mean().item(),
            'mean_log_prob_ref': log_prob_ref.mean().item(),
        }
        
        return loss, stats
    
    def step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        执行一步 GRPO 训练。
        参数:
            batch: 批次数据，包含:
                - 'text': 文本提示列表 [B]
                - 其他条件信息
                
        返回:
            stats: 训练统计信息字典
        """
        self.model.train()
        
        # 提取 prompts
        prompts = batch.get('text', [])
        if isinstance(prompts, torch.Tensor):
            prompts = prompts.tolist()
        batch_size = len(prompts)
        
        # 扩展 prompts 用于组采样：每个 prompt 重复 G 次
        expanded_prompts = []
        for prompt in prompts:
            expanded_prompts.extend([prompt] * self.group_size)
        
        # 准备扩展批次的 model kwargs
        expanded_batch_size = batch_size * self.group_size
        model_kwargs = self._prepare_model_kwargs(batch, expanded_batch_size)
        
        # ========== 阶段 1: Rollout ==========
        # 使用当前模型生成样本并计算 log probs
        # 确定动作长度
        if 'lengths' in batch and isinstance(batch['lengths'], torch.Tensor):
            motion_length = int(batch['lengths'].max().item()) if len(batch['lengths']) > 0 else 196
        else:
            motion_length = 196  # 默认长度
        
        shape = (
            expanded_batch_size,
            self.model.njoints,
            self.model.nfeats,
            motion_length
        )
        
        # 使用固定噪声进行采样（用于可重现性和 log prob 计算）
        # 我们将使用相同的噪声用于当前模型和参考模型
        noise = torch.randn(*shape, device=self.device)
        
        # 使用当前模型采样（可训练，保存轨迹）
        with torch.set_grad_enabled(True):
            current_result = self.sample_with_trajectory(
                self.model,
                shape,
                model_kwargs,
                noise=noise,
                save_trajectory=True,
            )
        
        motions = current_result['samples']  # [B*G, C, H, W]
        latents_sequence_current = current_result['latents_sequence']  # 完整轨迹
        timesteps_sequence = current_result['timesteps_sequence']  # 时间步序列
        
        # 使用 get_batch_log_prob 计算当前模型的 log probability
        log_prob_current = self.get_batch_log_prob(
            self.model,
            latents_sequence_current,
            timesteps_sequence,
            model_kwargs,
        )  # [B*G]
        
        # 使用参考模型计算 log probability（冻结，相同噪声和条件）
        with torch.no_grad():
            ref_result = self.sample_with_trajectory(
                self.ref_model,
                shape,
                model_kwargs,
                noise=noise,
                save_trajectory=True,
            )
            
            latents_sequence_ref = ref_result['latents_sequence']
            
            log_prob_ref = self.get_batch_log_prob(
                self.ref_model,
                latents_sequence_ref,
                timesteps_sequence,
                model_kwargs,
            )  # [B*G]
        
        # ========== 阶段 2: 奖励计算 ==========
        # 计算生成动作的奖励
        # 注意: motions 需要转换为适合奖励函数的格式
        rewards = self.reward_fn(motions, expanded_prompts)  # [B*G]
        rewards = rewards.to(self.device)
        
        # ========== 阶段 3: 优势计算 ==========
        advantages = self.compute_group_advantages(rewards)  # [B*G]
        
        # ========== 阶段 4: 损失计算和更新 ==========
        loss, stats = self.compute_grpo_loss(
            log_prob_current,
            log_prob_ref,
            advantages,
        )
        
        # 添加奖励统计信息
        stats['mean_reward'] = rewards.mean().item()
        stats['std_reward'] = rewards.std().item()
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪（可选但推荐）
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return stats
    
    def _prepare_model_kwargs(
        self,
        batch: Dict[str, torch.Tensor],
        expanded_batch_size: int,
    ) -> Dict:
        """
        为扩展批次（组采样）准备 model kwargs。
        
        参数:
            batch: 原始批次
            expanded_batch_size: 新的批次大小（B * G）
            
        返回:
            model_kwargs: 模型参数字典
        """
        model_kwargs = {'y': {}}
        
        # 处理文本 prompts
        if 'text' in batch:
            prompts = batch['text']
            if isinstance(prompts, list):
                # 扩展 prompts
                expanded_prompts = []
                for prompt in prompts:
                    expanded_prompts.extend([prompt] * self.group_size)
                model_kwargs['y']['text'] = expanded_prompts
            else:
                # 张量情况：沿批次维度重复
                model_kwargs['y']['text'] = prompts.repeat_interleave(self.group_size, dim=0)
        
        # 处理其他条件
        for key in ['lengths', 'mask']:
            if key in batch:
                value = batch[key]
                if isinstance(value, torch.Tensor):
                    model_kwargs['y'][key] = value.repeat_interleave(self.group_size, dim=0)
                else:
                    model_kwargs['y'][key] = value
        
        return model_kwargs


def create_grpo_trainer(
    model: nn.Module,
    ref_model: nn.Module,
    diffusion: 'GaussianDiffusion',
    reward_fn: Callable,
    learning_rate: float = 1e-5,
    group_size: int = 4,
    clip_epsilon: float = 0.2,
    kl_penalty: float = 0.1,
    **kwargs,
) -> GRPOTrainer:
    """
    创建 GRPO 训练器的工厂函数。
    
    参数:
        model: 可训练模型
        ref_model: 参考模型
        diffusion: 扩散调度器
        reward_fn: 奖励函数
        learning_rate: 优化器学习率
        group_size: GRPO 的组大小
        clip_epsilon: 裁剪参数
        kl_penalty: KL 惩罚权重
        **kwargs: GRPOTrainer 的额外参数
        
    返回:
        GRPOTrainer 实例
    """
    # 创建优化器（仅针对可训练参数，例如 LoRA）
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)
    
    return GRPOTrainer(
        model=model,
        ref_model=ref_model,
        diffusion=diffusion,
        optimizer=optimizer,
        reward_fn=reward_fn,
        group_size=group_size,
        clip_epsilon=clip_epsilon,
        kl_penalty=kl_penalty,
        **kwargs,
    )

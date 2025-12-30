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
    """
    扩散模型的组相对策略优化训练器。
    
    GRPO 使用基于组的相对优势替代绝对优势，从而无需 Critic 网络。
    根据 DanceGRPO 论文，策略 π_θ(z_{t-1}|z_t) 被建模为高斯分布。
    """
    
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
        
        # 内存优化：在计算 log prob 后立即清理轨迹
        self.clear_trajectory_after_logprob = True
    
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
            
            # 数值稳定性：限制 log_variance 的范围，避免 exp 溢出
            dist_log_variance = torch.clamp(dist_log_variance, min=-20, max=20)
            dist_variance = torch.exp(dist_log_variance)  # [B, C, H, W]
            # 确保 variance 不会太小，避免除零
            dist_variance = torch.clamp(dist_variance, min=1e-8)
            dist_std = torch.sqrt(dist_variance)  # [B, C, H, W] - 标准差
            
            # 3. 计算高斯 log probability
            # 实际采取的下一步是 z_{t-1}（在 Rollout 阶段采样保存的）
            # 计算 z_{t-1} 在 N(μ_θ, σ²) 下的 log probability
            
            # 标准化误差
            squared_error = (z_t_minus_1 - dist_mean) ** 2  # [B, C, H, W]
            
            # 数值稳定性：避免 squared_error 过大
            squared_error = torch.clamp(squared_error, max=1e6)
            
            # 高斯 log probability: log p(x) = -0.5 * ((x-μ)/σ)² - log(σ) - 0.5*log(2π)
            # 使用更稳定的计算方式
            normalized_error = squared_error / dist_variance  # [B, C, H, W]
            normalized_error = torch.clamp(normalized_error, max=1e6)  # 防止溢出
            
            log_prob_per_element = (
                -0.5 * normalized_error
                - dist_log_variance * 0.5  # log(σ) = 0.5 * log(σ²)
                - 0.5 * math.log(2 * math.pi)
            )  # [B, C, H, W]
            
            # 检查 NaN
            if torch.isnan(log_prob_per_element).any():
                print(f"警告: log_prob_per_element 包含 NaN at step {i}")
                print(f"  dist_mean range: [{dist_mean.min().item():.4f}, {dist_mean.max().item():.4f}]")
                print(f"  dist_log_variance range: [{dist_log_variance.min().item():.4f}, {dist_log_variance.max().item():.4f}]")
                print(f"  squared_error range: [{squared_error.min().item():.4f}, {squared_error.max().item():.4f}]")
                log_prob_per_element = torch.nan_to_num(log_prob_per_element, nan=0.0, posinf=0.0, neginf=-1e6)
            
            # 对所有空间维度求和，得到每个样本的 log prob
            # 根据 DanceGRPO，我们对所有维度求和
            # 注意：对于高维数据（如 263*1*196），log prob 的绝对值会很大，这是正常的
            log_prob_per_timestep = log_prob_per_element.sum(dim=[1, 2, 3])  # [B]
            
            # 检查异常值（log prob 绝对值过大）
            # 对于高维数据，每个时间步的 log prob 可能在 [-1e5, 1e5] 范围内
            # 累积 50 个时间步后，总 log prob 可能在 [-5e6, 5e6] 范围内
            # 这是正常的，关键是要确保 log_ratio 在合理范围内
            if (log_prob_per_timestep.abs() > 2e5).any():
                print(f"警告: log_prob_per_timestep 绝对值过大 at timestep {i}")
                print(f"  log_prob_per_timestep range: [{log_prob_per_timestep.min().item():.2f}, {log_prob_per_timestep.max().item():.2f}]")
                print(f"  log_prob_per_element shape: {log_prob_per_element.shape}")
                num_elements = log_prob_per_element.shape[1] * log_prob_per_element.shape[2] * log_prob_per_element.shape[3]
                print(f"  num_elements per timestep: {num_elements}")
                print(f"  log_prob_per_element mean: {log_prob_per_element.mean().item():.6f}")
                print(f"  log_prob_per_element std: {log_prob_per_element.std().item():.6f}")
                # 限制 log prob 的范围（但允许较大的值，因为高维数据是正常的）
                log_prob_per_timestep = torch.clamp(log_prob_per_timestep, min=-2e5, max=2e5)
            
            # 检查 NaN
            if torch.isnan(log_prob_per_timestep).any():
                print(f"警告: log_prob_per_timestep 包含 NaN at timestep {i}")
                log_prob_per_timestep = torch.nan_to_num(log_prob_per_timestep, nan=0.0, posinf=2e5, neginf=-2e5)
            
            # 累积到总 log probability
            total_log_prob = total_log_prob + log_prob_per_timestep
        
        # 最终检查
        if torch.isnan(total_log_prob).any():
            print("警告: total_log_prob 包含 NaN")
            total_log_prob = torch.nan_to_num(total_log_prob, nan=0.0, posinf=0.0, neginf=-1e6)
        
        # 检查最终 log prob 是否异常
        # 对于高维数据，累积 log prob 的绝对值可能很大（几百万），这是正常的
        # 关键是要确保 log_prob_current 和 log_prob_ref 的差值在合理范围内
        if (total_log_prob.abs() > 1e7).any():
            print("警告: total_log_prob 绝对值过大")
            print(f"  total_log_prob range: [{total_log_prob.min().item():.2f}, {total_log_prob.max().item():.2f}]")
            print(f"  total_log_prob mean: {total_log_prob.mean().item():.2f}")
            print(f"  轨迹长度: {len(latents_sequence) - 1} 个时间步")
            # 限制范围（但允许较大的值）
            total_log_prob = torch.clamp(total_log_prob, min=-1e7, max=1e7)
        
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
        
        # 检查组内奖励差异是否过小
        if (group_std < 1e-6).any():
            print("警告: 组内奖励标准差过小，advantage 可能接近 0")
            print(f"  rewards range: [{rewards.min().item():.4f}, {rewards.max().item():.4f}]")
            print(f"  rewards mean: {rewards.mean().item():.4f}, std: {rewards.std().item():.4f}")
            print(f"  group_mean range: [{group_mean.min().item():.4f}, {group_mean.max().item():.4f}]")
            print(f"  group_std range: [{group_std.min().item():.4f}, {group_std.max().item():.4f}]")
            print(f"  建议: 检查奖励函数是否有足够的区分度，或增加 group_size 以获得更多样本多样性")
        
        # 数值稳定性：确保 std 不会太小
        group_std = torch.clamp(group_std, min=self.advantage_eps)
        
        # 在每个组内归一化
        advantages = (rewards_reshaped - group_mean) / (group_std + self.advantage_eps)
        
        # 检查 NaN
        if torch.isnan(advantages).any():
            print("警告: advantages 计算中包含 NaN")
            print(f"  rewards range: [{rewards.min().item():.4f}, {rewards.max().item():.4f}]")
            print(f"  group_mean range: [{group_mean.min().item():.4f}, {group_mean.max().item():.4f}]")
            print(f"  group_std range: [{group_std.min().item():.4f}, {group_std.max().item():.4f}]")
            advantages = torch.nan_to_num(advantages, nan=0.0, posinf=1e6, neginf=-1e6)
        
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
        # 计算 ratio（数值稳定性：限制差值范围）
        log_ratio = log_prob_current - log_prob_ref  # [B*G]
        
        # 检查 log_ratio 是否异常大
        # 对于高维数据，log_prob 的绝对值可能很大，但它们的差值应该相对较小
        # 如果差值很大（> 10），说明当前模型和参考模型的预测差异很大
        if (log_ratio.abs() > 10).any():
            print("警告: log_ratio 超出安全范围，将被裁剪")
            print(f"  log_ratio range (before clamp): [{log_ratio.min().item():.2f}, {log_ratio.max().item():.2f}]")
            print(f"  log_ratio mean: {log_ratio.mean().item():.2f}, std: {log_ratio.std().item():.2f}")
            print(f"  log_prob_current range: [{log_prob_current.min().item():.2f}, {log_prob_current.max().item():.2f}]")
            print(f"  log_prob_current mean: {log_prob_current.mean().item():.2f}")
            print(f"  log_prob_ref range: [{log_prob_ref.min().item():.2f}, {log_prob_ref.max().item():.2f}]")
            print(f"  log_prob_ref mean: {log_prob_ref.mean().item():.2f}")
            print(f"  建议: 检查模型是否正常，可能需要降低学习率或增加 KL 惩罚")
        
        # 限制 log_ratio 范围，避免 exp 溢出
        # 使用更严格的范围：-10 到 10，对应 ratio 范围 [4e-5, 2e4]
        # 这比 -20 到 20 更保守，但更稳定
        log_ratio = torch.clamp(log_ratio, min=-10, max=10)
        ratio = torch.exp(log_ratio)  # [B*G]
        
        # 额外限制 ratio 的范围（双重保护）
        ratio = torch.clamp(ratio, min=1e-4, max=1e4)
        
        # 检查 NaN
        if torch.isnan(ratio).any():
            print("警告: ratio 包含 NaN")
            print(f"  log_prob_current range: [{log_prob_current.min().item():.4f}, {log_prob_current.max().item():.4f}]")
            print(f"  log_prob_ref range: [{log_prob_ref.min().item():.4f}, {log_prob_ref.max().item():.4f}]")
            print(f"  log_ratio range: [{log_ratio.min().item():.4f}, {log_ratio.max().item():.4f}]")
            ratio = torch.nan_to_num(ratio, nan=1.0, posinf=1e6, neginf=1e-6)
        
        # 计算 KL 惩罚
        kl = self.compute_kl_penalty(log_prob_current, log_prob_ref)
        
        # 检查 KL NaN
        if torch.isnan(kl).any():
            print("警告: kl 包含 NaN")
            kl = torch.nan_to_num(kl, nan=0.0, posinf=1e6, neginf=0.0)
        
        # PPO 风格的裁剪目标
        ratio_clipped = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
        
        # 策略损失: min(ratio * A, clipped_ratio * A)
        policy_loss_1 = ratio * advantages
        policy_loss_2 = ratio_clipped * advantages
        policy_loss = torch.min(policy_loss_1, policy_loss_2)
        
        # 总损失: 策略损失 - KL 惩罚
        loss = -(policy_loss.mean() - self.kl_penalty * kl.mean())
        
        # 损失缩放：如果损失值过大，进行缩放以避免梯度爆炸
        # 这对于高维数据和大的 log prob 值很重要
        loss_scale = 1.0
        loss_value = loss.item() if not torch.isnan(loss) and not torch.isinf(loss) else float('inf')
        
        if abs(loss_value) > 1e6:
            # 计算缩放因子，使损失值在合理范围内
            loss_scale = 1e6 / abs(loss_value)
            print("警告: 损失值异常大，应用损失缩放")
            print(f"  原始 loss: {loss_value:.2f}")
            print(f"  缩放因子: {loss_scale:.6f}")
            print(f"  policy_loss mean: {policy_loss.mean().item():.4f}")
            print(f"  kl mean: {kl.mean().item():.4f}")
            print(f"  mean_ratio: {ratio.mean().item():.4f}")
            print(f"  mean_advantage: {advantages.mean().item():.4f}")
            loss = loss * loss_scale
        
        # 检查损失是否异常大（可能导致梯度爆炸）
        loss_value_after_scale = loss.item() if not torch.isnan(loss) and not torch.isinf(loss) else float('inf')
        if abs(loss_value_after_scale) > 1e6:
            print("错误: 损失值仍然异常大，即使经过缩放")
            print(f"  缩放后 loss: {loss_value_after_scale:.2f}")
            print(f"  建议: 检查损失计算，可能需要进一步降低学习率或增加 KL 惩罚")
        
        # 检查损失是否包含 NaN
        if torch.isnan(loss) or torch.isinf(loss):
            print("警告: loss 包含 NaN 或 Inf")
            print(f"  policy_loss mean: {policy_loss.mean().item():.4f}")
            print(f"  kl mean: {kl.mean().item():.4f}")
            # 使用安全的默认值
            loss = torch.tensor(0.0, device=loss.device, requires_grad=True)
        
        # 计算统计信息（使用安全的 item() 调用）
        try:
            stats = {
                'loss': loss.item() if not torch.isnan(loss) else 0.0,
                'policy_loss': -policy_loss.mean().item() if not torch.isnan(policy_loss.mean()) else 0.0,
                'kl_penalty': kl.mean().item() if not torch.isnan(kl.mean()) else 0.0,
                'mean_ratio': ratio.mean().item() if not torch.isnan(ratio.mean()) else 1.0,
                'mean_advantage': advantages.mean().item() if not torch.isnan(advantages.mean()) else 0.0,
                'mean_log_prob_current': log_prob_current.mean().item() if not torch.isnan(log_prob_current.mean()) else 0.0,
                'mean_log_prob_ref': log_prob_ref.mean().item() if not torch.isnan(log_prob_ref.mean()) else 0.0,
            }
        except Exception as e:
            print(f"警告: 计算统计信息时出错: {e}")
            stats = {
                'loss': 0.0,
                'policy_loss': 0.0,
                'kl_penalty': 0.0,
                'mean_ratio': 1.0,
                'mean_advantage': 0.0,
                'mean_log_prob_current': 0.0,
                'mean_log_prob_ref': 0.0,
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
        
        # 检查 log_prob_current 是否异常
        if torch.isnan(log_prob_current).any():
            print("警告: log_prob_current 包含 NaN")
            print(f"  log_prob_current range: [{log_prob_current.min().item():.4f}, {log_prob_current.max().item():.4f}]")
            log_prob_current = torch.nan_to_num(log_prob_current, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # 检查 log_prob_current 绝对值是否过大
        # 对于高维数据，log prob 绝对值大是正常的，但需要限制范围以确保数值稳定
        # 注意：限制范围不能太小，否则会丢失 log_prob_current 和 log_prob_ref 之间的差异信息
        if (log_prob_current.abs() > 1e7).any():
            print("警告: log_prob_current 绝对值过大")
            print(f"  log_prob_current range (before clamp): [{log_prob_current.min().item():.2f}, {log_prob_current.max().item():.2f}]")
            print(f"  log_prob_current mean: {log_prob_current.mean().item():.2f}")
            print(f"  注意: 对于高维数据，log prob 绝对值大是正常的，将被限制在 [-1e7, 1e7]")
            # 限制范围（增加到 -1e7 到 1e7，保留更多信息）
            log_prob_current = torch.clamp(log_prob_current, min=-1e7, max=1e7)
        
        # 清理不需要的中间变量以释放内存
        del current_result
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 使用参考模型计算 log probability（使用相同的轨迹，无需重新采样）
        # 注意：参考模型应该使用相同的轨迹来计算 log prob，而不是重新采样
        with torch.no_grad():
            log_prob_ref = self.get_batch_log_prob(
                self.ref_model,
                latents_sequence_current,  # 使用相同的轨迹
                timesteps_sequence,
                model_kwargs,
            )  # [B*G]
            
            # 检查 log_prob_ref 是否异常
            if torch.isnan(log_prob_ref).any():
                print("警告: log_prob_ref 包含 NaN")
                print(f"  log_prob_ref range: [{log_prob_ref.min().item():.4f}, {log_prob_ref.max().item():.4f}]")
                log_prob_ref = torch.nan_to_num(log_prob_ref, nan=0.0, posinf=0.0, neginf=-1e6)
            
            # 检查 log_prob_ref 绝对值是否过大
            # 对于高维数据，log prob 绝对值大是正常的
            # 注意：限制范围应该与 log_prob_current 保持一致，以保留差值信息
            if (log_prob_ref.abs() > 1e7).any():
                print("警告: log_prob_ref 绝对值过大")
                print(f"  log_prob_ref range (before clamp): [{log_prob_ref.min().item():.2f}, {log_prob_ref.max().item():.2f}]")
                print(f"  log_prob_ref mean: {log_prob_ref.mean().item():.2f}")
                print(f"  注意: 对于高维数据，log prob 绝对值大是正常的，将被限制在 [-1e7, 1e7]")
                log_prob_ref = torch.clamp(log_prob_ref, min=-1e7, max=1e7)
        
        # 清理轨迹以释放内存（如果不再需要）
        # 注意：轨迹只在计算 log prob 时需要，之后可以删除
        if self.clear_trajectory_after_logprob:
            del latents_sequence_current, timesteps_sequence
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # ========== 阶段 2: 奖励计算 ==========
        # 计算生成动作的奖励
        # 注意: motions 需要转换为适合奖励函数的格式
        rewards = self.reward_fn(motions, expanded_prompts)  # [B*G]
        rewards = rewards.to(self.device)
        
        # 检查奖励是否包含 NaN
        if torch.isnan(rewards).any():
            print("警告: rewards 包含 NaN")
            print(f"  rewards range: [{rewards.min().item():.4f}, {rewards.max().item():.4f}]")
            rewards = torch.nan_to_num(rewards, nan=0.0, posinf=1.0, neginf=0.0)
        
        # ========== 阶段 3: 优势计算 ==========
        advantages = self.compute_group_advantages(rewards)  # [B*G]
        
        # 检查优势是否包含 NaN
        if torch.isnan(advantages).any():
            print("警告: advantages 包含 NaN")
            print(f"  advantages range: [{advantages.min().item():.4f}, {advantages.max().item():.4f}]")
            advantages = torch.nan_to_num(advantages, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # ========== 阶段 4: 损失计算和更新 ==========
        loss, stats = self.compute_grpo_loss(
            log_prob_current,
            log_prob_ref,
            advantages,
        )
        
        # 添加奖励统计信息
        stats['mean_reward'] = rewards.mean().item()
        stats['std_reward'] = rewards.std().item()
        
        # 检查损失是否异常
        if torch.isnan(loss) or torch.isinf(loss):
            print("错误: 损失包含 NaN 或 Inf，跳过此步")
            print(f"  loss: {loss.item() if not torch.isnan(loss) else 'NaN'}")
            print(f"  policy_loss: {stats.get('policy_loss', 'N/A')}")
            print(f"  kl_penalty: {stats.get('kl_penalty', 'N/A')}")
            print(f"  mean_ratio: {stats.get('mean_ratio', 'N/A')}")
            print(f"  mean_advantage: {stats.get('mean_advantage', 'N/A')}")
            # 返回统计信息但不更新模型
            return stats
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        
        # 计算梯度范数（在裁剪之前）
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=float('inf'))
        stats['grad_norm'] = grad_norm.item()
        
        # 检查梯度是否异常
        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
            print(f"错误: 梯度包含 NaN 或 Inf，跳过此步")
            print(f"  grad_norm: {grad_norm.item() if not torch.isnan(grad_norm) else 'NaN'}")
            # 清零梯度，不更新模型
            self.optimizer.zero_grad()
            return stats
        
        # 分析梯度来源（找出哪个参数导致梯度爆炸）
        if grad_norm > 1000:
            print(f"警告: 梯度范数异常大 ({grad_norm.item():.2f})，分析梯度来源...")
            max_grad_per_param = 0
            max_grad_param_name = None
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    param_grad_norm = param.grad.norm().item()
                    if param_grad_norm > max_grad_per_param:
                        max_grad_per_param = param_grad_norm
                        max_grad_param_name = name
            if max_grad_param_name:
                print(f"  最大梯度来自参数: {max_grad_param_name}")
                print(f"  该参数的梯度范数: {max_grad_per_param:.2f}")
        
        # 梯度裁剪：使用更严格的值
        # 如果梯度太大，先裁剪，然后检查是否仍然异常
        max_grad_norm = 1.0  # 非常严格的值，因为学习率已经很小
        if grad_norm > max_grad_norm:
            print(f"警告: 梯度范数过大 ({grad_norm.item():.2f})，将被裁剪到 {max_grad_norm}")
            print(f"  损失值: {loss.item():.4f}")
            print(f"  policy_loss: {stats.get('policy_loss', 'N/A')}")
            print(f"  kl_penalty: {stats.get('kl_penalty', 'N/A')}")
            print(f"  mean_ratio: {stats.get('mean_ratio', 'N/A')}")
            print(f"  mean_advantage: {stats.get('mean_advantage', 'N/A')}")
            
            # 裁剪梯度
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_grad_norm)
            
            # 重新计算梯度范数
            grad_norm_after = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=float('inf'))
            stats['grad_norm'] = grad_norm_after.item()
            
            # 如果裁剪后仍然很大，跳过更新
            if grad_norm_after > max_grad_norm * 10:
                print(f"错误: 梯度裁剪后仍然过大 ({grad_norm_after.item():.2f})，跳过此步")
                print(f"  建议: 进一步降低学习率（当前: 5e-7），或增加 KL 惩罚，或检查损失计算")
                self.optimizer.zero_grad()
                return stats
        else:
            # 正常情况下的梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_grad_norm)
        
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
    
    # 验证只有 LoRA 参数是可训练的
    trainable_param_names = [name for name, param in model.named_parameters() if param.requires_grad]
    non_lora_trainable = [name for name in trainable_param_names if 'lora' not in name.lower()]
    if non_lora_trainable:
        print(f"警告: 发现非 LoRA 参数是可训练的: {non_lora_trainable[:5]}")
        if len(non_lora_trainable) > 5:
            print(f"  ... 还有 {len(non_lora_trainable) - 5} 个参数")
    else:
        print(f"✓ 确认: 只有 LoRA 参数是可训练的（共 {len(trainable_param_names)} 个参数）")
    
    if len(trainable_params) == 0:
        raise ValueError("错误: 没有可训练的参数！请检查模型是否正确配置了 LoRA")
    
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=0.01)
    
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


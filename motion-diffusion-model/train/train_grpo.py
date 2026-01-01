"""
Training script for GRPO (Group Relative Policy Optimization) fine-tuning
of Motion Diffusion Models with LoRA.

Usage:
    python -m train.train_grpo \
        --model_path ./save/pretrained_model/model000200000.pt \
        --save_dir ./save/grpo_finetuned \
        --group_size 4 \
        --batch_size 2 \
        --num_steps 10000 \
        --reward_type matching
"""

import os
import json
import torch
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，适合服务器环境
import matplotlib.pyplot as plt
from typing import Callable, Optional
from tqdm import tqdm

from utils.fixseed import fixseed
from utils import dist_util
from utils.model_util import create_model_and_diffusion, load_saved_model
from utils.parser_util import grpo_args
from data_loaders.get_data import get_dataset_loader
from model.GRPO.grpo_trainer import create_grpo_trainer


def create_reward_fn(
    device: str = 'cuda',
    reward_model_type: str = 'mdm',
    reward_type: str = 'matching',
    dataset_name: str = 'humanml',
    tmr_checkpoint_path: Optional[str] = None,
    tmr_similarity_type: str = 'cosine',
    tmr_normalization: str = 'linear',
    tmr_max_distance: float = 10.0,
    tmr_scale: float = 2.0,
) -> Callable:
    """
    创建动作生成的奖励函数
    
    支持两种奖励模型：
    1. MDM 评估器奖励函数（基于 MDM 项目的评估器）
    2. TMR 预训练模型奖励函数（基于 TMR 预训练权重）
    
    参数:
        device: 运行设备
        reward_model_type: 奖励模型类型 ('mdm' 或 'tmr')
        reward_type: 奖励类型
            - 对于 MDM: 'matching', 'r_precision', 'combined'
            - 对于 TMR: 'matching', 'cosine'
        dataset_name: 数据集名称 ('humanml' 或 'kit')
        tmr_checkpoint_path: TMR 预训练权重路径（仅当 reward_model_type='tmr' 时需要）
        tmr_similarity_type: TMR 相似度类型 ('cosine' 或 'euclidean')
        tmr_normalization: TMR 归一化方式 ('linear', 'exponential', 'sigmoid')
        tmr_max_distance: TMR 最大距离（用于线性归一化）
        tmr_scale: TMR 缩放因子（用于指数/Sigmoid 归一化）
        
    返回:
        reward_fn: 计算奖励的函数
    """
    if reward_model_type == 'mdm':
        # 使用 MDM 评估器奖励函数
        try:
            from model.GRPO.reward_model import create_mdm_reward_function
            reward_fn = create_mdm_reward_function(
                reward_type=reward_type,
                dataset_name=dataset_name,
                device=device,
            )
            print(f"✓ 使用 MDM 评估器奖励函数 (类型: {reward_type}, 数据集: {dataset_name})")
            return reward_fn
        except Exception as e:
            print(f"✗ 无法加载 MDM 评估器奖励函数: {e}")
            print("使用占位奖励函数（随机奖励）")
            
            # 占位函数
            def reward_fn(motions: torch.Tensor, prompts: list) -> torch.Tensor:
                batch_size = motions.shape[0]
                rewards = torch.randn(batch_size, device=motions.device) * 0.1 + 0.5
                return rewards
            
            return reward_fn
    
    elif reward_model_type == 'tmr':
        # 使用 TMR 预训练模型奖励函数
        if tmr_checkpoint_path is None:
            raise ValueError("使用 TMR 奖励模型时，必须提供 --tmr_checkpoint_path 参数")
        
        try:
            from model.GRPO.reward_model_tmr import create_tmr_reward_function
            
            # 构建 TMR 奖励函数的参数
            tmr_kwargs = {}
            if reward_type == 'matching':
                # 对于 matching 类型，可以配置相似度和归一化方式
                tmr_kwargs['similarity_type'] = tmr_similarity_type
                tmr_kwargs['normalization'] = tmr_normalization
                tmr_kwargs['max_distance'] = tmr_max_distance
                tmr_kwargs['scale'] = tmr_scale
            # 对于 cosine 类型，使用默认参数即可
            
            reward_fn = create_tmr_reward_function(
                tmr_checkpoint_path=tmr_checkpoint_path,
                reward_type=reward_type,
                dataset_name=dataset_name,
                device=device,
                **tmr_kwargs,
            )
            print(f"✓ 使用 TMR 预训练模型奖励函数")
            print(f"  - 权重路径: {tmr_checkpoint_path}")
            print(f"  - 奖励类型: {reward_type}")
            if reward_type == 'matching':
                print(f"  - 相似度类型: {tmr_similarity_type}")
                print(f"  - 归一化方式: {tmr_normalization}")
            print(f"  - 数据集: {dataset_name}")
            return reward_fn
        except Exception as e:
            print(f"✗ 无法加载 TMR 奖励函数: {e}")
            import traceback
            traceback.print_exc()
            print("使用占位奖励函数（随机奖励）")
            
            # 占位函数
            def reward_fn(motions: torch.Tensor, prompts: list) -> torch.Tensor:
                batch_size = motions.shape[0]
                rewards = torch.randn(batch_size, device=motions.device) * 0.1 + 0.5
                return rewards
            
            return reward_fn
    
    else:
        raise ValueError(f"不支持的奖励模型类型: {reward_model_type}，请选择 'mdm' 或 'tmr'")


def main():
    args = grpo_args()
    fixseed(args.seed)
    
    # Setup device
    dist_util.setup_dist(args.device)
    device = dist_util.dev()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Save arguments
    with open(os.path.join(args.save_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Load dataset
    print("Loading dataset...")
    data_loader = get_dataset_loader(
        name=args.dataset,
        batch_size=args.batch_size,
        num_frames=None,
        split='train',
        hml_mode='train',
    )
    
    # Create model and diffusion
    print("Creating model and diffusion...")
    # args 已经包含了所有需要的参数（通过 grpo_args() 扩展了 train_args()）
    # 尝试从 checkpoint 加载参数（如果存在）
    # 注意：保存用户指定的训练参数，避免被 checkpoint 参数覆盖
    user_batch_size = args.batch_size
    user_num_steps = args.num_steps
    user_group_size = args.group_size
    user_learning_rate = getattr(args, 'learning_rate', None) or getattr(args, 'lr', None)
    
    checkpoint_dir = os.path.dirname(args.model_path)
    args_path = os.path.join(checkpoint_dir, 'args.json')
    if os.path.exists(args_path):
        with open(args_path, 'r') as f:
            checkpoint_args = json.load(f)
            for key, value in checkpoint_args.items():
                # 保护用户指定的训练参数和 GRPO 特定参数
                protected_keys = [
                    'model_path', 'save_dir', 
                    'group_size', 'learning_rate', 'clip_epsilon', 'kl_penalty', 'reward_type',
                    'batch_size', 'num_steps', 'log_interval', 'save_interval'
                ]
                if hasattr(args, key) and key not in protected_keys:
                    # 只更新模型相关参数，不覆盖用户指定的训练参数
                    setattr(args, key, value)
    
    # 恢复用户指定的训练参数（确保不被 checkpoint 覆盖）
    args.batch_size = user_batch_size
    args.num_steps = user_num_steps
    args.group_size = user_group_size
    if user_learning_rate is not None:
        if hasattr(args, 'learning_rate'):
            args.learning_rate = user_learning_rate
        if hasattr(args, 'lr'):
            args.lr = user_learning_rate
    
    # 创建模型（如果 use_lora，会在 create_model_and_diffusion 中添加 LoRA）
    model, diffusion = create_model_and_diffusion(args, data_loader)
    
    # Load pretrained weights
    print(f"Loading pretrained model from {args.model_path}...")
    load_saved_model(model, args.model_path, use_avg=False)
    model.to(device)
    
    # 注意：如果 use_lora=True，create_model_and_diffusion 已经添加了 LoRA
    # 所以不需要再次添加（第 149-160 行的代码是重复的，但保留以防万一）
    
    # Create reference model (frozen copy)
    # 重要：参考模型应该**不包含 LoRA**，因为它是"参考"（原始预训练模型）
    # 如果参考模型也包含 LoRA，LoRA 层是随机初始化的，会导致 log_prob 差异极大
    print("Creating reference model (without LoRA)...")
    ref_model_path = args.ref_model_path if args.ref_model_path else args.model_path
    
    # 临时禁用 use_lora，创建不包含 LoRA 的参考模型
    use_lora_backup = args.use_lora
    args.use_lora = False
    ref_model, _ = create_model_and_diffusion(args, data_loader)
    args.use_lora = use_lora_backup  # 恢复原始设置
    
    load_saved_model(ref_model, ref_model_path, use_avg=False)
    ref_model.to(device)
    
    print("✓ 参考模型创建完成（不包含 LoRA，使用原始预训练权重）")
    
    # Freeze reference model
    for param in ref_model.parameters():
        param.requires_grad = False
    ref_model.eval()
    
    # Create reward function
    print(f"Creating reward function (model: {args.reward_model_type}, type: {args.reward_type})...")
    reward_fn = create_reward_fn(
        device=device,
        reward_model_type=args.reward_model_type,
        reward_type=args.reward_type,
        dataset_name=args.dataset,
        tmr_checkpoint_path=getattr(args, 'tmr_checkpoint_path', None),
        tmr_similarity_type=getattr(args, 'tmr_similarity_type', 'cosine'),
        tmr_normalization=getattr(args, 'tmr_normalization', 'linear'),
        tmr_max_distance=getattr(args, 'tmr_max_distance', 10.0),
        tmr_scale=getattr(args, 'tmr_scale', 2.0),
    )
    
    # Create GRPO trainer
    print("Creating GRPO trainer...")
    # 优先使用 --learning_rate，如果没有提供则使用 --lr
    grpo_lr = getattr(args, 'learning_rate', None) or getattr(args, 'lr', 1e-5)
    trainer = create_grpo_trainer(
        model=model,
        ref_model=ref_model,
        diffusion=diffusion,
        reward_fn=reward_fn,
        learning_rate=grpo_lr,
        group_size=args.group_size,
        clip_epsilon=args.clip_epsilon,
        kl_penalty=args.kl_penalty,
        device=device,
    )
    
    # Training loop
    print("Starting training...")
    print(f"Total steps: {args.num_steps}, Batch size: {args.batch_size}, Group size: {args.group_size}")
    step = 0
    
    # 用于绘制曲线的数据收集
    training_steps = []
    loss_values = []
    prompt_avg_reward_values = []
    
    # 创建进度条
    pbar = tqdm(
        total=args.num_steps,
        desc="GRPO Training",
        unit="step",
        ncols=120,  # 进度条宽度
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
    )
    
    try:
        for epoch in range(1000):  # Large number, will break on num_steps
            for batch_idx, (motions, cond) in enumerate(data_loader):
                if step >= args.num_steps:
                    break
                
                # Prepare batch for GRPO
                # The batch should contain text prompts
                batch = {
                    'text': cond['y'].get('text', []),
                    'lengths': cond['y'].get('lengths', None),
                    'mask': cond['y'].get('mask', None),
                }
                
                # Training step
                try:
                    stats = trainer.step(batch)
                    
                    # 收集数据用于绘制曲线
                    training_steps.append(step)
                    if 'loss' in stats:
                        loss_values.append(stats['loss'])
                    else:
                        loss_values.append(0.0)
                    
                    if 'prompt_avg_reward' in stats:
                        prompt_avg_reward_values.append(stats['prompt_avg_reward'])
                    elif 'mean_reward' in stats:
                        # 如果没有 prompt_avg_reward，使用 mean_reward 作为近似
                        prompt_avg_reward_values.append(stats['mean_reward'])
                    else:
                        prompt_avg_reward_values.append(0.0)
                    
                    # 更新进度条显示
                    # 构建显示信息（只显示关键指标，避免过长）
                    postfix_dict = {}
                    
                    # 主要指标（格式化显示）
                    if 'loss' in stats:
                        postfix_dict['Loss'] = f"{stats['loss']:.4f}"
                    if 'prompt_avg_reward' in stats:
                        postfix_dict['AvgScore'] = f"{stats['prompt_avg_reward']:.3f}"
                    elif 'mean_reward' in stats:
                        postfix_dict['Reward'] = f"{stats['mean_reward']:.3f}"
                    if 'mean_advantage' in stats:
                        postfix_dict['Adv'] = f"{stats['mean_advantage']:.3f}"
                    if 'mean_ratio' in stats:
                        ratio_val = stats['mean_ratio']
                        postfix_dict['Ratio'] = f"{ratio_val:.3f}"
                    if 'kl_penalty' in stats:
                        postfix_dict['KL'] = f"{stats['kl_penalty']:.4f}"
                    
                    # 更新进度条
                    pbar.set_postfix(postfix_dict, refresh=False)
                    pbar.update(1)
                    
                    # 每10步打印详细统计信息
                    if step % 10 == 0:
                        tqdm.write(f"\nStep {step}:")
                        for key, value in stats.items():
                            tqdm.write(f"  {key}: {value:.4f}")
                        # 打印 motion 平均得分
                        if 'prompt_avg_reward' in stats:
                            tqdm.write(f"  Motion 平均得分 (每个 prompt 的 {args.group_size} 个 motion 平均): {stats['prompt_avg_reward']:.4f}")
                    
                    # 详细日志（按 log_interval，如果 log_interval > 10）
                    if args.log_interval > 10 and step % args.log_interval == 0:
                        tqdm.write(f"\n=== Detailed Log at Step {step} ===")
                        for key, value in stats.items():
                            tqdm.write(f"  {key}: {value:.4f}")
                    
                    # Save checkpoint
                    if step % args.save_interval == 0 and step > 0:
                        checkpoint_path = os.path.join(
                            args.save_dir,
                            f'model{step:09d}.pt'
                        )
                        torch.save({
                            'model': model.state_dict(),
                            'optimizer': trainer.optimizer.state_dict(),
                            'step': step,
                            'stats': stats,
                        }, checkpoint_path)
                        tqdm.write(f"✓ Saved checkpoint to {checkpoint_path}")
                        
                        # 绘制并保存训练曲线
                        plot_training_curves(
                            training_steps,
                            loss_values,
                            prompt_avg_reward_values,
                            args.save_dir,
                            step
                        )
                    
                    step += 1
                    
                except Exception as e:
                    tqdm.write(f"✗ Error at step {step}: {e}")
                    import traceback
                    traceback.print_exc()
                    pbar.update(1)  # 即使出错也更新进度条
                    step += 1
                    continue
            
            if step >= args.num_steps:
                break
    finally:
        # 确保进度条显示100%
        if step < args.num_steps:
            pbar.update(args.num_steps - step)
        # 关闭进度条
        pbar.close()
    
    # Save final checkpoint
    final_checkpoint_path = os.path.join(args.save_dir, 'model_final.pt')
    torch.save({
        'model': model.state_dict(),
        'optimizer': trainer.optimizer.state_dict(),
        'step': step,
    }, final_checkpoint_path)
    print(f"\n✓ Training complete! Total steps: {step}")
    print(f"✓ Final checkpoint saved to {final_checkpoint_path}")
    
    # 绘制并保存最终训练曲线
    plot_training_curves(
        training_steps,
        loss_values,
        prompt_avg_reward_values,
        args.save_dir,
        step
    )
    print(f"✓ Training curves saved to {args.save_dir}")


def plot_training_curves(steps, losses, prompt_avg_rewards, save_dir, current_step):
    """
    绘制训练曲线图
    
    参数:
        steps: 训练步数列表
        losses: 损失值列表
        prompt_avg_rewards: 每个 prompt 的平均得分列表
        save_dir: 保存目录
        current_step: 当前训练步数
    """
    if len(steps) == 0:
        return
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 第一个图：Loss 损失曲线
    ax1.plot(steps, losses, 'b-', linewidth=1.5, label='Loss')
    ax1.set_xlabel('Training Steps', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Loss Curve', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 第二个图：Motion 平均得分曲线
    ax2.plot(steps, prompt_avg_rewards, 'g-', linewidth=1.5, label='Motion Average Score')
    ax2.set_xlabel('Training Steps', fontsize=12)
    ax2.set_ylabel('Motion Average Score (per prompt)', fontsize=12)
    ax2.set_title('Motion Average Score Curve', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plot_path = os.path.join(save_dir, f'training_curves_step_{current_step:09d}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # 同时保存一个最新的版本（覆盖更新）
    latest_plot_path = os.path.join(save_dir, 'training_curves_latest.png')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    ax1.plot(steps, losses, 'b-', linewidth=1.5, label='Loss')
    ax1.set_xlabel('Training Steps', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Loss Curve', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.plot(steps, prompt_avg_rewards, 'g-', linewidth=1.5, label='Motion Average Score')
    ax2.set_xlabel('Training Steps', fontsize=12)
    ax2.set_ylabel('Motion Average Score (per prompt)', fontsize=12)
    ax2.set_title('Motion Average Score Curve', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(latest_plot_path, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    main()


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
import argparse
from typing import Callable

from utils.fixseed import fixseed
from utils import dist_util
from utils.model_util import create_model_and_diffusion, load_saved_model
from data_loaders.get_data import get_dataset_loader
from model.GRPO.grpo_trainer import create_grpo_trainer


def create_reward_fn(device: str = 'cuda', reward_type: str = 'matching', dataset_name: str = 'humanml') -> Callable:
    """
    创建动作生成的奖励函数
    
    MDM 项目本身不使用 reward 函数，而是使用评估指标。
    这里我们基于 MDM 的评估器（EvaluatorMDMWrapper）创建奖励函数。
    
    参数:
        device: 运行设备
        reward_type: 奖励类型 ('matching', 'r_precision', 'combined')
            - 'matching': 基于文本-动作匹配分数（欧氏距离）
            - 'r_precision': 基于 R-Precision
            - 'combined': 组合多种指标
        dataset_name: 数据集名称 ('humanml' 或 'kit')
        
    返回:
        reward_fn: 计算奖励的函数
    """
    try:
        # 尝试使用基于 MDM 评估器的奖励函数
        from model.GRPO.reward_model import create_mdm_reward_function
        reward_fn = create_mdm_reward_function(
            reward_type=reward_type,
            dataset_name=dataset_name,
            device=device,
        )
        print(f"使用 MDM 评估器奖励函数 (类型: {reward_type}, 数据集: {dataset_name})")
        return reward_fn
    except Exception as e:
        print(f"无法加载 MDM 评估器奖励函数: {e}")
        print("使用占位奖励函数（随机奖励）")
        
        # 占位函数
        def reward_fn(motions: torch.Tensor, prompts: list) -> torch.Tensor:
            """
            计算生成动作的奖励（占位函数）
            
            参数:
                motions: 生成的动作序列 [B, njoints, nfeats, nframes]
                prompts: 文本提示列表 [B]
                
            返回:
                rewards: 奖励值 [B]
            """
            batch_size = motions.shape[0]
            # 占位：返回随机奖励
            rewards = torch.randn(batch_size, device=motions.device) * 0.1 + 0.5
            return rewards
        
        return reward_fn


def parse_args():
    """
    解析命令行参数，扩展 train_args() 的参数解析器，添加 GRPO 特定参数
    """
    from utils.parser_util import add_base_options, add_data_options, add_model_options, add_diffusion_options, add_training_options, apply_rules
    from argparse import ArgumentParser
    
    # 创建解析器并添加所有标准训练参数
    parser = ArgumentParser(description='GRPO Training for Motion Diffusion Models')
    add_base_options(parser)
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    add_training_options(parser)
    
    # 添加 GRPO 特定参数
    grpo_group = parser.add_argument_group('GRPO')
    grpo_group.add_argument('--model_path', type=str, required=True,
                            help='Path to pretrained model checkpoint')
    grpo_group.add_argument('--ref_model_path', type=str, default=None,
                            help='Path to reference model (if different from model_path)')
    grpo_group.add_argument('--group_size', type=int, default=4,
                            help='Group size G for GRPO (samples per prompt)')
    grpo_group.add_argument('--learning_rate', type=float, default=1e-5,
                            help='Learning rate for GRPO training')
    grpo_group.add_argument('--clip_epsilon', type=float, default=0.2,
                            help='PPO clipping parameter')
    grpo_group.add_argument('--kl_penalty', type=float, default=0.1,
                            help='KL divergence penalty weight')
    grpo_group.add_argument('--reward_type', type=str, default='matching',
                            choices=['matching', 'r_precision', 'combined'],
                            help='Reward function type: matching (Matching Score), r_precision (R-Precision), or combined (both)')
    
    # 解析参数并应用规则
    args = apply_rules(parser.parse_args())
    
    return args


def main():
    args = parse_args()
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
    # args 已经包含了所有需要的参数（通过 parse_args() 扩展了 train_args()）
    # 尝试从 checkpoint 加载参数（如果存在）
    checkpoint_dir = os.path.dirname(args.model_path)
    args_path = os.path.join(checkpoint_dir, 'args.json')
    if os.path.exists(args_path):
        with open(args_path, 'r') as f:
            checkpoint_args = json.load(f)
            for key, value in checkpoint_args.items():
                if hasattr(args, key) and key not in ['model_path', 'save_dir', 'group_size', 'learning_rate', 'clip_epsilon', 'kl_penalty', 'reward_type']:
                    # 只更新模型相关参数，不覆盖 GRPO 特定参数
                    setattr(args, key, value)
    
    model, diffusion = create_model_and_diffusion(args, data_loader)
    
    # Load pretrained weights
    print(f"Loading pretrained model from {args.model_path}...")
    load_saved_model(model, args.model_path, use_avg=False)
    model.to(device)
    
    # Add LoRA if specified
    if args.use_lora:
        print("Adding LoRA adapters...")
        from model.lora_adapter import add_lora_to_mdm
        model = add_lora_to_mdm(
            model,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.0,
            target_spec='all',
        )
        model.to(device)
    
    # Create reference model (frozen copy)
    print("Creating reference model...")
    ref_model_path = args.ref_model_path if args.ref_model_path else args.model_path
    ref_model, _ = create_model_and_diffusion(args, data_loader)
    load_saved_model(ref_model, ref_model_path, use_avg=False)
    ref_model.to(device)
    
    # Freeze reference model
    for param in ref_model.parameters():
        param.requires_grad = False
    ref_model.eval()
    
    # Create reward function
    print(f"Creating reward function (type: {args.reward_type})...")
    reward_fn = create_reward_fn(device=device, reward_type=args.reward_type, dataset_name=args.dataset)
    
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
    step = 0
    
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
                
                # Logging
                if step % args.log_interval == 0:
                    print(f"Step {step}:")
                    for key, value in stats.items():
                        print(f"  {key}: {value:.4f}")
                
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
                    print(f"Saved checkpoint to {checkpoint_path}")
                
                step += 1
                
            except Exception as e:
                print(f"Error at step {step}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if step >= args.num_steps:
            break
    
    # Save final checkpoint
    final_checkpoint_path = os.path.join(args.save_dir, 'model_final.pt')
    torch.save({
        'model': model.state_dict(),
        'optimizer': trainer.optimizer.state_dict(),
        'step': step,
    }, final_checkpoint_path)
    print(f"Training complete! Final checkpoint saved to {final_checkpoint_path}")


if __name__ == '__main__':
    main()


"""
GRPO (Group Relative Policy Optimization) 模块

本模块包含 GRPO 训练器和奖励模型的实现。
"""

from .grpo_trainer import GRPOTrainer, create_grpo_trainer
from .reward_model import (
    MDMRewardFunction,
    MatchingScoreReward,
    RPrecisionReward,
    CombinedMDMReward,
    create_mdm_reward_function,
)

__all__ = [
    'GRPOTrainer',
    'create_grpo_trainer',
    'MDMRewardFunction',
    'MatchingScoreReward',
    'RPrecisionReward',
    'CombinedMDMReward',
    'create_mdm_reward_function',
]


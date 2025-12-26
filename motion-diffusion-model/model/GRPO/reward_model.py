"""
基于 MDM 评估器的奖励模型实现

MDM 项目本身不使用 reward 函数，而是使用评估指标来衡量生成质量。
主要的评估指标包括：
1. Matching Score - 文本和动作的匹配分数（欧氏距离）
2. R-precision - 检索精度
3. FID - Fréchet Inception Distance
4. Diversity - 多样性
5. MultiModality - 多模态性

本文件提供基于这些评估指标的 reward 模型实现，可用于 GRPO 训练。

【重要说明】为什么只实现 Matching Score 和 R-Precision？
- GRPO 需要为每个样本单独计算奖励值，因此只能使用支持单样本计算的指标
- Matching Score 和 R-Precision 可以为每个样本单独计算，适合作为奖励函数
- FID、Diversity 和 MultiModality 需要批量样本才能计算，是批量指标，不适合作为单个样本的奖励
- 这些批量指标仍然可以作为训练后的评估指标，用于评估整体模型性能
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Optional
from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper
from data_loaders.humanml.utils.word_vectorizer import WordVectorizer
from data_loaders.humanml.utils.metrics import euclidean_distance_matrix

# 尝试导入 spacy，如果不可用则使用简单处理
try:
    import spacy
    _spacy_available = True
except ImportError:
    _spacy_available = False
    print("警告: spacy 未安装，将使用简单的文本处理方式。建议安装 spacy: pip install spacy && python -m spacy download en_core_web_sm")


class MDMRewardFunction:
    """
    基于 MDM 评估器的奖励函数基类
    """
    
    def __init__(
        self,
        dataset_name: str = 'humanml',
        device: str = 'cuda',
        word_vectorizer: Optional[WordVectorizer] = None,
    ):
        """
        初始化 MDM 奖励函数
        
        参数:
            dataset_name: 数据集名称 ('humanml' 或 'kit')
            device: 设备
            word_vectorizer: 词向量化器（如果为 None，会尝试加载）
        """
        self.device = device
        self.dataset_name = dataset_name
        
        # 初始化评估器
        self.evaluator = EvaluatorMDMWrapper(dataset_name, device)
        
        # 初始化词向量化器
        if word_vectorizer is None:
            self.word_vectorizer = WordVectorizer('./glove', 'our_vab')
        else:
            self.word_vectorizer = word_vectorizer
        
        # 初始化 spacy（如果可用）
        if _spacy_available:
            try:
                self.nlp = spacy.load('en_core_web_sm')
            except OSError:
                print("警告: 无法加载 spacy 模型 'en_core_web_sm'，将使用简单文本处理")
                self.nlp = None
        else:
            self.nlp = None
    
    def _process_text(self, sentence: str) -> tuple:
        """
        处理文本，提取词和词性
        
        参数:
            sentence: 输入文本
            
        返回:
            word_list: 词列表
            pos_list: 词性列表
        """
        if self.nlp is not None:
            # 使用 spacy 处理（推荐方式）
            sentence = sentence.replace('-', '')
            doc = self.nlp(sentence)
            word_list = []
            pos_list = []
            for token in doc:
                word = token.text
                if not word.isalpha():
                    continue
                if (token.pos_ == 'NOUN' or token.pos_ == 'VERB') and (word != 'left'):
                    word_list.append(token.lemma_)
                else:
                    word_list.append(word)
                pos_list.append(token.pos_)
            return word_list, pos_list
        else:
            # 简单处理方式（如果 spacy 不可用）
            words = sentence.lower().split()
            # 简单假设所有词都是 NOUN（这不是最佳方式，但可以工作）
            pos_list = ['NOUN'] * len(words)
            return words, pos_list
    
    def _prepare_text_inputs(self, prompts: List[str]):
        """
        将文本提示转换为评估器所需的格式
        
        参数:
            prompts: 文本提示列表
            
        返回:
            word_embs: 词嵌入
            pos_ohot: 词性 one-hot
            cap_lens: 文本长度
        """
        batch_size = len(prompts)
        max_text_len = 20  # 不包括 SOS 和 EOS
        
        word_embs_list = []
        pos_ohot_list = []
        cap_lens_list = []
        
        for prompt in prompts:
            # 处理文本，获取词和词性
            word_list, pos_list = self._process_text(prompt)
            
            # 转换为 WordVectorizer 所需的格式: 'word/POS'
            tokens = ['%s/%s' % (word_list[i], pos_list[i]) for i in range(len(word_list))]
            tokens = tokens[:max_text_len]  # 截断到最大长度
            
            # 添加 SOS 和 EOS tokens（评估器期望这些 tokens）
            if len(tokens) < max_text_len:
                tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
                sent_len = len(tokens)
                tokens = tokens + ['unk/OTHER'] * (max_text_len + 2 - sent_len)
            else:
                # 如果太长，裁剪
                tokens = tokens[:max_text_len]
                tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
                sent_len = len(tokens)
            
            # 获取词嵌入和词性 one-hot
            word_embeddings = []
            pos_one_hots = []
            for token in tokens:
                word_emb, pos_oh = self.word_vectorizer[token]
                word_embeddings.append(word_emb[None, :])
                pos_one_hots.append(pos_oh[None, :])
            
            if len(word_embeddings) == 0:
                # 如果处理失败，使用 unk token
                word_emb, pos_oh = self.word_vectorizer['unk/OTHER']
                word_embeddings = [word_emb[None, :]]
                pos_one_hots = [pos_oh[None, :]]
            
            word_emb = np.concatenate(word_embeddings, axis=0)
            pos_ohot = np.concatenate(pos_one_hots, axis=0)
            
            word_embs_list.append(word_emb)
            pos_ohot_list.append(pos_ohot)
            cap_lens_list.append(sent_len)
        
        word_embs = torch.tensor(np.stack(word_embs_list), dtype=torch.float32).to(self.device)
        pos_ohot = torch.tensor(np.stack(pos_ohot_list), dtype=torch.float32).to(self.device)
        cap_lens = torch.tensor(cap_lens_list, dtype=torch.long).to(self.device)
        
        return word_embs, pos_ohot, cap_lens
    
    def _prepare_motion_inputs(self, motions: torch.Tensor, lengths: Optional[torch.Tensor] = None):
        """
        准备动作输入
        
        参数:
            motions: 动作序列 [B, njoints, nfeats, nframes]
            lengths: 动作长度（如果为 None，使用完整长度）
            
        返回:
            motions_processed: 处理后的动作
            m_lens: 动作长度
        """
        batch_size = motions.shape[0]
        
        if lengths is None:
            # 假设使用完整长度
            m_lens = torch.full((batch_size,), motions.shape[-1], dtype=torch.long, device=self.device)
        else:
            m_lens = lengths.to(self.device)
        
        # 转换格式：从 [B, njoints, nfeats, nframes] 到 [B, nframes, njoints*nfeats]
        # 注意：这里需要根据实际数据格式调整
        if len(motions.shape) == 4:
            # [B, njoints, nfeats, nframes] -> [B, nframes, njoints*nfeats]
            motions_processed = motions.permute(0, 3, 1, 2).reshape(batch_size, motions.shape[-1], -1)
        else:
            motions_processed = motions
        
        return motions_processed, m_lens


class MatchingScoreReward(MDMRewardFunction):
    """
    基于 Matching Score 的奖励函数
    
    Matching Score 使用文本和动作嵌入之间的欧氏距离来衡量匹配程度。
    距离越小，匹配度越高，奖励越大。
    """
    
    def __call__(self, motions: torch.Tensor, prompts: List[str], lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算基于匹配分数的奖励
        
        参数:
            motions: 生成的动作序列 [B, njoints, nfeats, nframes]
            prompts: 文本提示列表 [B]
            lengths: 动作长度（可选）
            
        返回:
            rewards: 奖励值 [B]，范围大约在 [0, 1]（经过归一化）
        """
        batch_size = motions.shape[0]
        
        # 准备文本输入
        word_embs, pos_ohot, cap_lens = self._prepare_text_inputs(prompts)
        
        # 准备动作输入
        motions_processed, m_lens = self._prepare_motion_inputs(motions, lengths)
        
        # 获取文本和动作嵌入
        with torch.no_grad():
            text_embeddings, motion_embeddings = self.evaluator.get_co_embeddings(
                word_embs=word_embs,
                pos_ohot=pos_ohot,
                cap_lens=cap_lens,
                motions=motions_processed,
                m_lens=m_lens,
            )
        
        # 计算欧氏距离（匹配分数）
        # 距离越小，匹配度越高
        distances = torch.norm(text_embeddings - motion_embeddings, dim=-1)  # [B]
        
        # 将距离转换为奖励（距离越小，奖励越大）
        # 使用负距离或指数衰减
        # 方法1: 负距离（需要归一化）
        # rewards = -distances
        
        # 方法2: 使用指数衰减（更稳定）
        # rewards = torch.exp(-distances / scale)
        
        # 方法3: 线性归一化（简单有效）
        # 假设距离范围大致在 [0, 10]，归一化到 [0, 1]
        max_distance = 10.0  # 可根据实际情况调整
        rewards = 1.0 - torch.clamp(distances / max_distance, 0, 1)
        
        return rewards


class RPrecisionReward(MDMRewardFunction):
    """
    基于 R-Precision 的奖励函数
    
    R-Precision 衡量在 top-k 检索中正确匹配的比例。
    这里我们使用与对应文本的相似度作为奖励。
    """
    
    def __init__(self, top_k: int = 1, *args, **kwargs):
        """
        参数:
            top_k: 用于计算 R-precision 的 k 值
        """
        super().__init__(*args, **kwargs)
        self.top_k = top_k
    
    def __call__(self, motions: torch.Tensor, prompts: List[str], lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算基于 R-Precision 的奖励
        
        参数:
            motions: 生成的动作序列 [B, njoints, nfeats, nframes]
            prompts: 文本提示列表 [B]
            lengths: 动作长度（可选）
            
        返回:
            rewards: 奖励值 [B]，范围 [0, 1]
        """
        batch_size = motions.shape[0]
        
        # 准备输入
        word_embs, pos_ohot, cap_lens = self._prepare_text_inputs(prompts)
        motions_processed, m_lens = self._prepare_motion_inputs(motions, lengths)
        
        # 获取嵌入
        with torch.no_grad():
            text_embeddings, motion_embeddings = self.evaluator.get_co_embeddings(
                word_embs=word_embs,
                pos_ohot=pos_ohot,
                cap_lens=cap_lens,
                motions=motions_processed,
                m_lens=m_lens,
            )
        
        # 计算距离矩阵
        text_emb = text_embeddings.cpu().numpy()
        motion_emb = motion_embeddings.cpu().numpy()
        dist_mat = euclidean_distance_matrix(text_emb, motion_emb)
        
        # 对于每个文本，找到最相似的 k 个动作
        # 如果对应的动作在 top-k 中，给予奖励
        rewards = []
        for i in range(batch_size):
            # 获取第 i 个文本对应的动作距离
            distances = dist_mat[i]
            # 排序，找到 top-k
            top_k_indices = np.argsort(distances)[:self.top_k]
            # 如果对应的动作（索引 i）在 top-k 中，奖励为 1，否则为 0
            if i in top_k_indices:
                reward = 1.0
            else:
                # 也可以使用距离的倒数作为奖励
                reward = 1.0 / (1.0 + distances[i])
            rewards.append(reward)
        
        return torch.tensor(rewards, dtype=torch.float32, device=self.device)


class CombinedMDMReward(MDMRewardFunction):
    """
    组合多种 MDM 评估指标的奖励函数
    """
    
    def __init__(
        self,
        matching_weight: float = 0.7,
        r_precision_weight: float = 0.3,
        *args,
        **kwargs,
    ):
        """
        参数:
            matching_weight: Matching Score 的权重
            r_precision_weight: R-Precision 的权重
        """
        super().__init__(*args, **kwargs)
        self.matching_reward = MatchingScoreReward(*args, **kwargs)
        self.r_precision_reward = RPrecisionReward(top_k=1, *args, **kwargs)
        self.matching_weight = matching_weight
        self.r_precision_weight = r_precision_weight
    
    def __call__(self, motions: torch.Tensor, prompts: List[str], lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算组合奖励
        
        参数:
            motions: 生成的动作序列 [B, njoints, nfeats, nframes]
            prompts: 文本提示列表 [B]
            lengths: 动作长度（可选）
            
        返回:
            rewards: 组合奖励值 [B]
        """
        matching_rewards = self.matching_reward(motions, prompts, lengths)
        r_precision_rewards = self.r_precision_reward(motions, prompts, lengths)
        
        combined_rewards = (
            self.matching_weight * matching_rewards +
            self.r_precision_weight * r_precision_rewards
        )
        
        return combined_rewards


def create_mdm_reward_function(
    reward_type: str = 'matching',
    dataset_name: str = 'humanml',
    device: str = 'cuda',
    **kwargs,
) -> MDMRewardFunction:
    """
    创建 MDM 奖励函数的工厂函数
    
    参数:
        reward_type: 奖励类型 ('matching', 'r_precision', 'combined')
        dataset_name: 数据集名称
        device: 设备
        **kwargs: 其他参数
        
    返回:
        reward_function: 奖励函数实例
    """
    if reward_type == 'matching':
        return MatchingScoreReward(dataset_name=dataset_name, device=device, **kwargs)
    elif reward_type == 'r_precision':
        return RPrecisionReward(dataset_name=dataset_name, device=device, **kwargs)
    elif reward_type == 'combined':
        return CombinedMDMReward(dataset_name=dataset_name, device=device, **kwargs)
    else:
        raise ValueError(f"Unknown reward type: {reward_type}")


# 使用示例
if __name__ == '__main__':
    # 创建奖励函数
    reward_fn = create_mdm_reward_function('matching', device='cuda')
    
    # 示例用法
    batch_size = 4
    motions = torch.randn(batch_size, 263, 1, 196)  # HumanML3D 格式
    prompts = [
        "a person walks forward",
        "someone jumps up",
        "a person sits down",
        "someone runs fast"
    ]
    
    rewards = reward_fn(motions, prompts)
    print(f"Rewards: {rewards}")


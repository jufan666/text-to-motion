
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
from typing import List, Optional, Tuple, Dict, Union
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
        # 新增参数
        use_dense_reward: bool = False,
        use_physics_reward: bool = False,
        k_segments: int = 1,
        max_motion_length: int = 196,
        alpha: float = 0.5,
        beta_s: float = 1.0,
        beta_p: float = 0.1,
        lambda_skate: float = 1.0,
        lambda_jerk: float = 1.0,
        fps: float = 20.0,  # 数据集帧率，HumanML=20, KIT=12.5
    ):
        """
        初始化 MDM 奖励函数
        
        参数:
            dataset_name: 数据集名称 ('humanml' 或 'kit')
            device: 设备
            word_vectorizer: 词向量化器（如果为 None，会尝试加载）
            use_dense_reward: 是否使用分段密集打分 (Segment-Dense)，False=整体打分 (Global)
            use_physics_reward: 是否计算物理正则化
            k_segments: 文本拼接数量（用于校验或默认处理）
            max_motion_length: 动作最大帧数限制
            alpha: 负向惩罚权重
            beta_s: 语义奖励权重
            beta_p: 物理奖励权重
            lambda_skate: 滑行惩罚权重
            lambda_jerk: 加速度突变惩罚权重
            fps: 数据集帧率（帧/秒），用于将 duration 转换为帧数
        """
        self.device = device
        self.dataset_name = dataset_name
        
        # 新增配置参数
        self.use_dense_reward = use_dense_reward
        self.use_physics_reward = use_physics_reward
        self.k_segments = k_segments
        self.max_motion_length = max_motion_length
        self.alpha = alpha
        self.beta_s = beta_s
        self.beta_p = beta_p
        self.lambda_skate = lambda_skate
        self.lambda_jerk = lambda_jerk
        self.fps = fps  # 数据集帧率
        
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
            sentence: 输入文本（str 或 List[str]，如果是列表则连接）
            
        返回:
            word_list: 词列表
            pos_list: 词性列表
        """
        # 处理输入：如果是列表，转换为字符串
        if isinstance(sentence, list):
            sentence = " ".join(sentence)
        
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
    
    def _truncate_motions(
        self,
        motions: torch.Tensor,
        segments: Optional[List[List[Tuple[int, int]]]] = None,
    ) -> Tuple[torch.Tensor, Optional[List[List[Tuple[int, int]]]]]:
        """
        截断动作到最大长度，并修正 segments
        
        参数:
            motions: 动作序列 [B, njoints, nfeats, nframes]
            segments: 分段信息 List[List[Tuple[int, int]]]，每个样本有 K 个 (start, end)
            
        返回:
            motions_truncated: 截断后的动作
            segments_corrected: 修正后的 segments（如果提供了 segments）
        """
        if motions.shape[-1] <= self.max_motion_length:
            return motions, segments
        
        print(f"警告: 动作长度 {motions.shape[-1]} 超过最大长度 {self.max_motion_length}，将截断")
        motions_truncated = motions[:, :, :, :self.max_motion_length]
        
        # 修正 segments
        segments_corrected = None
        if segments is not None:
            segments_corrected = []
            for seg_list in segments:
                corrected_seg = []
                for start, end in seg_list:
                    # 确保结束时间不超过最大帧数
                    end = min(end, self.max_motion_length)
                    start = min(start, end)  # 确保 start <= end
                    corrected_seg.append((start, end))
                segments_corrected.append(corrected_seg)
        
        return motions_truncated, segments_corrected
    
    def _extract_foot_contact(
        self,
        motions: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """
        从动作数据中提取脚部接触标签
        
        参数:
            motions: 动作序列 [B, njoints, nfeats, nframes] 或 [B, nframes, dim]
            
        返回:
            foot_contact: 脚部接触标签 [B, nframes, 2] (left, right)，如果无法提取则返回 None
        """
        # HumanML 格式：动作数据的最后 4 个维度是脚部接触信息
        # 格式: [B, nframes, dim]，最后 4 维是 [feet_l, feet_r, ...]
        # 或者 [B, njoints, nfeats, nframes] 需要转换
        
        try:
            if len(motions.shape) == 4:
                # [B, njoints, nfeats, nframes] -> [B, nframes, njoints*nfeats]
                B, njoints, nfeats, nframes = motions.shape
                motions_flat = motions.permute(0, 3, 1, 2).reshape(B, nframes, -1)
            else:
                motions_flat = motions
            
            # 检查是否有足够的维度（至少需要 4 个维度用于脚部接触）
            if motions_flat.shape[-1] < 4:
                return None
            
            # 提取最后 4 维中的前 2 维（左右脚）
            # 注意：实际格式可能是 [..., feet_l, feet_r, ...]，需要根据实际数据格式调整
            # 这里假设最后 4 维是 [feet_l, feet_r, ...]
            foot_contact = motions_flat[:, :, -4:-2]  # [B, nframes, 2]
            
            # 二值化（如果还不是二值）
            foot_contact = (foot_contact > 0.5).float()
            
            return foot_contact
        except Exception as e:
            print(f"警告: 无法提取脚部接触信息: {e}")
            return None
    
    def compute_physics_reward(
        self,
        motions: torch.Tensor,
        foot_contact_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        计算物理正则化奖励
        
        公式: R_phy = exp(-lambda_skate * L_skate - lambda_jerk * L_jerk)
        
        参数:
            motions: 动作序列 [B, njoints, nfeats, nframes] 或 [B, nframes, dim]
            foot_contact_labels: 脚部接触标签 [B, nframes, 2]（可选）
            
        返回:
            physics_rewards: 物理奖励 [B]
        """
        with torch.no_grad():
            # 转换格式
            if len(motions.shape) == 4:
                B, njoints, nfeats, nframes = motions.shape
                motions_flat = motions.permute(0, 3, 1, 2).reshape(B, nframes, -1)
            else:
                motions_flat = motions
                B, nframes = motions_flat.shape[:2]
            
            # 提取根节点速度（前 3 维通常是 root 信息）
            # 假设前 3 维是 [root_rot_vel, root_linear_vel_x, root_linear_vel_z]
            if motions_flat.shape[-1] >= 3:
                root_vel = motions_flat[:, :, :2]  # [B, nframes, 2] (x, z 方向速度)
            else:
                # 如果无法提取，使用零速度
                root_vel = torch.zeros(B, nframes, 2, device=motions.device)
            
            # 计算滑行惩罚 L_skate
            L_skate = torch.zeros(B, device=motions.device)
            if foot_contact_labels is not None:
                # foot_contact_labels: [B, nframes, 2]
                # 当脚接触地面时（contact=1），水平速度应该接近 0
                contact_mask = foot_contact_labels.sum(dim=-1) > 0  # [B, nframes] (任意脚接触)
                
                for b in range(B):
                    contact_frames = contact_mask[b]  # [nframes]
                    if contact_frames.any():
                        # 计算接触时的水平速度模长平方
                        contact_vel = root_vel[b, contact_frames]  # [N_contact, 2]
                        vel_squared = (contact_vel ** 2).sum(dim=-1)  # [N_contact]
                        L_skate[b] = vel_squared.mean()
            else:
                # 如果没有脚部接触信息，仅计算 Jerk
                L_skate = torch.zeros(B, device=motions.device)
            
            # 计算加速度突变惩罚 L_jerk
            # 计算加速度：a_t = v_t - v_{t-1}
            if nframes > 1:
                # 使用所有关节的速度变化
                velocities = motions_flat[:, 1:] - motions_flat[:, :-1]  # [B, nframes-1, dim]
                accelerations = velocities[:, 1:] - velocities[:, :-1]  # [B, nframes-2, dim]
                # 计算加速度的 L2 范数
                jerk = (accelerations ** 2).sum(dim=-1).mean(dim=-1)  # [B]
                L_jerk = jerk
            else:
                L_jerk = torch.zeros(B, device=motions.device)
            
            # 计算物理奖励
            physics_rewards = torch.exp(
                -self.lambda_skate * L_skate - self.lambda_jerk * L_jerk
            )
            
            return physics_rewards
    
    def compute_semantic_reward(
        self,
        motions: torch.Tensor,
        text_lists: List[List[str]],
        segments: Optional[List[List[Tuple[int, int]]]] = None,
        durations: Optional[List[List[float]]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算语义奖励
        
        参数:
            motions: 动作序列 [B, njoints, nfeats, nframes]
            text_lists: 文本列表 List[List[str]]，每个样本有 K 个子文本
            segments: 分段信息 List[List[Tuple[int, int]]]，每个样本有 K 个 (start, end)
            durations: 每个文本段对应的持续时间（秒）List[List[float]]，每个样本有 K 个 duration
            
        返回:
            semantic_rewards: 语义奖励 [B]
            components: 包含 R_pos, R_neg 的字典
        """
        batch_size = motions.shape[0]
        device = motions.device
        
        if not self.use_dense_reward:
            # ========== Global Mode ==========
            # 将每个样本的文本列表拼接
            combined_texts = []
            for text_list in text_lists:
                combined_text = " ".join(text_list)
                combined_texts.append(combined_text)
            
            # 准备文本输入
            word_embs, pos_ohot, cap_lens = self._prepare_text_inputs(combined_texts)
            
            # 准备动作输入
            motions_processed, m_lens = self._prepare_motion_inputs(motions)
            
            # 获取嵌入
            with torch.no_grad():
                text_embeddings, motion_embeddings = self.evaluator.get_co_embeddings(
                    word_embs=word_embs,
                    pos_ohot=pos_ohot,
                    cap_lens=cap_lens,
                    motions=motions_processed,
                    m_lens=m_lens,
                )
            
            # 计算余弦相似度
            text_emb_norm = F.normalize(text_embeddings, p=2, dim=-1)
            motion_emb_norm = F.normalize(motion_embeddings, p=2, dim=-1)
            R_pos = (text_emb_norm * motion_emb_norm).sum(dim=-1)  # [B]
            R_neg = torch.zeros(batch_size, device=device)
            
        else:
            # ========== Segment-Dense Mode ==========
            R_pos_list = []
            R_neg_list = []
            
            for b in range(batch_size):
                text_list = text_lists[b]
                K = len(text_list)
                
                if segments is not None and len(segments[b]) == K:
                    # 如果提供了 segments，直接使用
                    seg_list = segments[b]
                elif durations is not None and len(durations[b]) == K:
                    # 如果提供了 durations，根据 duration * fps 计算 segments
                    nframes = motions.shape[-1]
                    seg_list = []
                    current_frame = 0
                    
                    for k, duration in enumerate(durations[b]):
                        # 计算该段的帧数
                        seg_frames = int(duration * self.fps)
                        # 确保不超过总帧数
                        end_frame = min(current_frame + seg_frames, nframes)
                        seg_list.append((current_frame, end_frame))
                        current_frame = end_frame
                        
                        # 如果已经到达末尾，后续段都设为空
                        if current_frame >= nframes:
                            # 填充剩余的段
                            for remaining_k in range(k + 1, K):
                                seg_list.append((nframes, nframes))
                            break
                    
                    # 如果最后一个段没有到达末尾，将其延伸到末尾
                    if len(seg_list) > 0 and seg_list[-1][1] < nframes:
                        seg_list[-1] = (seg_list[-1][0], nframes)
                else:
                    # 如果没有提供 segments 或 durations，均匀分割
                    nframes = motions.shape[-1]
                    seg_len = nframes // K
                    seg_list = [(i * seg_len, (i + 1) * seg_len) for i in range(K)]
                    # 最后一个分段到末尾
                    seg_list[-1] = (seg_list[-1][0], nframes)
                
                # 提取动作片段
                motion_segments = []
                for start, end in seg_list:
                    motion_seg = motions[b:b+1, :, :, start:end]  # [1, njoints, nfeats, seg_len]
                    motion_segments.append(motion_seg)
                
                # 计算正向奖励 R_pos
                pos_scores = []
                for k in range(K):
                    text_k = text_lists[b][k]
                    # 确保 text_k 是字符串（如果是列表则连接）
                    if isinstance(text_k, list):
                        text_k = " ".join(text_k)
                    motion_k = motion_segments[k]
                    
                    # 准备输入
                    word_embs_k, pos_ohot_k, cap_lens_k = self._prepare_text_inputs([text_k])
                    motions_processed_k, m_lens_k = self._prepare_motion_inputs(motion_k)
                    
                    # 获取嵌入
                    with torch.no_grad():
                        text_emb_k, motion_emb_k = self.evaluator.get_co_embeddings(
                            word_embs=word_embs_k,
                            pos_ohot=pos_ohot_k,
                            cap_lens=cap_lens_k,
                            motions=motions_processed_k,
                            m_lens=m_lens_k,
                        )
                    
                    # 计算相似度
                    text_emb_k_norm = F.normalize(text_emb_k, p=2, dim=-1)
                    motion_emb_k_norm = F.normalize(motion_emb_k, p=2, dim=-1)
                    sim_k = (text_emb_k_norm * motion_emb_k_norm).sum(dim=-1)  # [1]
                    pos_scores.append(sim_k.item())
                
                R_pos_b = torch.tensor(pos_scores, device=device).mean()
                R_pos_list.append(R_pos_b)
                
                # 计算负向奖励 R_neg（自适应边界）
                if K > 1:
                    # 计算文本间相似度作为边界
                    text_embs_all = []
                    for k in range(K):
                        text_k = text_lists[b][k]
                        # 确保 text_k 是字符串（如果是列表则连接）
                        if isinstance(text_k, list):
                            text_k = " ".join(text_k)
                        word_embs_k, pos_ohot_k, cap_lens_k = self._prepare_text_inputs([text_k])
                        with torch.no_grad():
                            text_emb_k, _ = self.evaluator.get_co_embeddings(
                                word_embs=word_embs_k,
                                pos_ohot=pos_ohot_k,
                                cap_lens=cap_lens_k,
                                motions=motions_processed_k[:1],  # 占位
                                m_lens=m_lens_k[:1],
                            )
                        text_embs_all.append(text_emb_k)
                    
                    text_embs_all = torch.cat(text_embs_all, dim=0)  # [K, dim]
                    text_embs_norm = F.normalize(text_embs_all, p=2, dim=-1)
                    
                    # 计算文本间相似度矩阵
                    B_matrix = torch.mm(text_embs_norm, text_embs_norm.t())  # [K, K]
                    
                    # 计算干扰度
                    neg_penalties = []
                    for k in range(K):
                        motion_k = motion_segments[k]
                        motions_processed_k, m_lens_k = self._prepare_motion_inputs(motion_k)
                        
                        # 获取动作嵌入
                        with torch.no_grad():
                            # 使用占位文本嵌入
                            text_placeholder = text_lists[b][0]
                            # 确保 text_placeholder 是字符串（如果是列表则连接）
                            if isinstance(text_placeholder, list):
                                text_placeholder = " ".join(text_placeholder)
                            # 确保 text_placeholder 是字符串（如果是列表则连接）
                            if isinstance(text_placeholder, list):
                                text_placeholder = " ".join(text_placeholder)
                            word_embs_placeholder, pos_ohot_placeholder, cap_lens_placeholder = self._prepare_text_inputs([text_placeholder])
                            _, motion_emb_k = self.evaluator.get_co_embeddings(
                                word_embs=word_embs_placeholder,
                                pos_ohot=pos_ohot_placeholder,
                                cap_lens=cap_lens_placeholder,
                                motions=motions_processed_k,
                                m_lens=m_lens_k,
                            )
                        
                        motion_emb_k_norm = F.normalize(motion_emb_k, p=2, dim=-1)
                        
                        # 计算与所有文本的相似度
                        s_kj = torch.mm(motion_emb_k_norm, text_embs_norm.t())  # [1, K]
                        s_kj = s_kj.squeeze(0)  # [K]
                        
                        # 计算惩罚：ReLU(s_kj - B_kj)
                        penalties = F.relu(s_kj - B_matrix[k])
                        # 取最大违规项
                        max_penalty = penalties.max()
                        neg_penalties.append(max_penalty)
                    
                    R_neg_b = torch.stack(neg_penalties).mean()
                else:
                    R_neg_b = torch.tensor(0.0, device=device)
                
                R_neg_list.append(R_neg_b)
            
            R_pos = torch.stack(R_pos_list)  # [B]
            R_neg = torch.stack(R_neg_list)  # [B]
        
        # 最终语义奖励
        R_sem = R_pos - self.alpha * R_neg
        
        components = {
            'R_pos': R_pos,
            'R_neg': R_neg,
        }
        
        return R_sem, components
    
    def compute_logic_accuracy(
        self,
        motions: torch.Tensor,
        text_lists: List[List[str]],
        segments: Optional[List[List[Tuple[int, int]]]] = None,
        durations: Optional[List[List[float]]] = None,
    ) -> Dict[str, float]:
        """
        计算 Logic-Acc 指标：对于第 k 个片段，检查 Sim(hat{y}_{T_k}, x_k) 是否是该行相似度矩阵中的最大值
        
        参数:
            motions: 动作序列 [B, njoints, nfeats, nframes]
            text_lists: 文本列表 List[List[str]]，每个样本有 K 个子文本
            segments: 分段信息 List[List[Tuple[int, int]]]，每个样本有 K 个 (start, end)
            durations: 每个文本段对应的持续时间（秒）List[List[float]]
            
        返回:
            logic_acc_dict: 包含逻辑准确率的字典
                - 'logic_acc': 整体逻辑准确率
                - 'logic_acc_per_segment': 每个片段的准确率列表
        """
        if not self.use_dense_reward:
            # Global 模式不支持 Logic-Acc
            return {'logic_acc': 0.0, 'logic_acc_per_segment': []}
        
        batch_size = motions.shape[0]
        device = motions.device
        
        all_correct = []
        segment_accs = []
        
        for b in range(batch_size):
            text_list = text_lists[b]
            K = len(text_list)
            
            # 计算 segments（与 compute_semantic_reward 中的逻辑一致）
            if segments is not None and len(segments[b]) == K:
                seg_list = segments[b]
            elif durations is not None and len(durations[b]) == K:
                nframes = motions.shape[-1]
                seg_list = []
                current_frame = 0
                for k, duration in enumerate(durations[b]):
                    seg_frames = int(duration * self.fps)
                    end_frame = min(current_frame + seg_frames, nframes)
                    seg_list.append((current_frame, end_frame))
                    current_frame = end_frame
                    if current_frame >= nframes:
                        for remaining_k in range(k + 1, K):
                            seg_list.append((nframes, nframes))
                        break
                if len(seg_list) > 0 and seg_list[-1][1] < nframes:
                    seg_list[-1] = (seg_list[-1][0], nframes)
            else:
                nframes = motions.shape[-1]
                seg_len = nframes // K
                seg_list = [(i * seg_len, (i + 1) * seg_len) for i in range(K)]
                seg_list[-1] = (seg_list[-1][0], nframes)
            
            # 提取动作片段
            motion_segments = []
            for start, end in seg_list:
                motion_seg = motions[b:b+1, :, :, start:end]
                motion_segments.append(motion_seg)
            
            # 准备所有文本的嵌入
            text_embs_all = []
            for k in range(K):
                text_k = text_list[k]
                word_embs_k, pos_ohot_k, cap_lens_k = self._prepare_text_inputs([text_k])
                with torch.no_grad():
                    text_emb_k, _ = self.evaluator.get_co_embeddings(
                        word_embs=word_embs_k,
                        pos_ohot=pos_ohot_k,
                        cap_lens=cap_lens_k,
                        motions=motion_segments[k][:1],  # 占位
                        m_lens=torch.tensor([motion_segments[k].shape[-1]], device=device),
                    )
                text_embs_all.append(text_emb_k)
            
            text_embs_all = torch.cat(text_embs_all, dim=0)  # [K, dim]
            text_embs_norm = F.normalize(text_embs_all, p=2, dim=-1)
            
            # 对每个片段计算逻辑准确率
            sample_correct = []
            for k in range(K):
                motion_k = motion_segments[k]
                motions_processed_k, m_lens_k = self._prepare_motion_inputs(motion_k)
                
                # 获取动作嵌入
                with torch.no_grad():
                    word_embs_placeholder, pos_ohot_placeholder, cap_lens_placeholder = self._prepare_text_inputs([text_list[0]])
                    _, motion_emb_k = self.evaluator.get_co_embeddings(
                        word_embs=word_embs_placeholder,
                        pos_ohot=pos_ohot_placeholder,
                        cap_lens=cap_lens_placeholder,
                        motions=motions_processed_k,
                        m_lens=m_lens_k,
                    )
                
                motion_emb_k_norm = F.normalize(motion_emb_k, p=2, dim=-1)
                
                # 计算与所有文本的相似度
                similarities = torch.mm(motion_emb_k_norm, text_embs_norm.t())  # [1, K]
                similarities = similarities.squeeze(0)  # [K]
                
                # 检查第 k 个文本的相似度是否是最大值
                max_idx = similarities.argmax().item()
                is_correct = (max_idx == k)
                sample_correct.append(is_correct)
            
            all_correct.extend(sample_correct)
            segment_accs.append(sum(sample_correct) / K)  # 该样本的片段准确率
        
        # 计算整体逻辑准确率
        logic_acc = sum(all_correct) / len(all_correct) if len(all_correct) > 0 else 0.0
        avg_segment_acc = sum(segment_accs) / len(segment_accs) if len(segment_accs) > 0 else 0.0
        
        return {
            'logic_acc': logic_acc,
            'avg_segment_acc': avg_segment_acc,
            'logic_acc_per_segment': segment_accs,
        }


class MatchingScoreReward(MDMRewardFunction):
    """
    基于 Matching Score 的奖励函数
    
    Matching Score 使用文本和动作嵌入之间的欧氏距离来衡量匹配程度。
    距离越小，匹配度越高，奖励越大。
    """
    
    def __call__(
        self,
        motions: torch.Tensor,
        prompts: Union[List[str], List[List[str]]],
        lengths: Optional[torch.Tensor] = None,
        text_lists: Optional[List[List[str]]] = None,
        segments: Optional[List[List[Tuple[int, int]]]] = None,
        durations: Optional[List[List[float]]] = None,
        foot_contact_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        计算基于匹配分数的奖励
        
        参数:
            motions: 生成的动作序列 [B, njoints, nfeats, nframes]
            prompts: 文本提示列表 [B] 或 List[List[str]]（向后兼容）
            lengths: 动作长度（可选）
            text_lists: 文本列表 List[List[str]]，每个样本有 K 个子文本（新功能）
            segments: 分段信息 List[List[Tuple[int, int]]]，每个样本有 K 个 (start, end)（新功能）
            durations: 每个文本段对应的持续时间（秒）List[List[float]]（新功能）
            foot_contact_labels: 脚部接触标签 [B, nframes, 2]（新功能，可选）
            
        返回:
            rewards: 奖励值 [B]
        """
        # ========== 预处理：动作长度截断 ==========
        motions, segments = self._truncate_motions(motions, segments)
        
        # ========== 兼容性处理：转换 prompts 为 text_lists ==========
        if text_lists is None:
            # 向后兼容：将 prompts 转换为 text_lists
            if isinstance(prompts[0], str):
                text_lists = [[p] for p in prompts]
            else:
                text_lists = prompts
        
        batch_size = motions.shape[0]
        device = motions.device
        
        # ========== 计算语义奖励 ==========
        R_sem, sem_components = self.compute_semantic_reward(
            motions=motions,
            text_lists=text_lists,
            segments=segments,
            durations=durations,
        )
        
        # ========== 计算物理奖励 ==========
        if self.use_physics_reward:
            # 尝试提取脚部接触信息
            if foot_contact_labels is None:
                foot_contact_labels = self._extract_foot_contact(motions)
            
            R_phy = self.compute_physics_reward(
                motions=motions,
                foot_contact_labels=foot_contact_labels,
            )
        else:
            R_phy = torch.zeros(batch_size, device=device)
        
        # ========== 总分聚合 ==========
        R_total = self.beta_s * R_sem + self.beta_p * R_phy
        
        # 返回奖励和组件信息（用于绘制曲线）
        # 注意：为了兼容性，如果调用者期望单个张量，我们返回元组
        # 训练器需要处理这种情况
        return R_total, {
            'R_pos': sem_components.get('R_pos', torch.zeros(batch_size, device=device)),
            'R_neg': sem_components.get('R_neg', torch.zeros(batch_size, device=device)),
            'R_sem': R_sem,
            'R_phy': R_phy,
        }


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
    
    def __call__(
        self,
        motions: torch.Tensor,
        prompts: Union[List[str], List[List[str]]],
        lengths: Optional[torch.Tensor] = None,
        text_lists: Optional[List[List[str]]] = None,
        segments: Optional[List[List[Tuple[int, int]]]] = None,
        durations: Optional[List[List[float]]] = None,
        foot_contact_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        计算基于 R-Precision 的奖励
        
        参数:
            motions: 生成的动作序列 [B, njoints, nfeats, nframes]
            prompts: 文本提示列表 [B] 或 List[List[str]]（向后兼容）
            lengths: 动作长度（可选）
            text_lists: 文本列表 List[List[str]]（新功能）
            segments: 分段信息（新功能）
            durations: 每个文本段对应的持续时间（秒）List[List[float]]（新功能）
            foot_contact_labels: 脚部接触标签（新功能）
            
        返回:
            rewards: 奖励值 [B]
        """
        # 使用基类的语义奖励计算（简化版，仅使用 Global 模式）
        if text_lists is None:
            if isinstance(prompts[0], str):
                text_lists = [[p] for p in prompts]
            else:
                text_lists = prompts
        
        motions, segments = self._truncate_motions(motions, segments)
        
        # 计算语义奖励（使用 Global 模式）
        R_sem, _ = self.compute_semantic_reward(motions, text_lists, segments, durations)
        
        # 计算物理奖励
        if self.use_physics_reward:
            if foot_contact_labels is None:
                foot_contact_labels = self._extract_foot_contact(motions)
            R_phy = self.compute_physics_reward(motions, foot_contact_labels)
        else:
            R_phy = torch.zeros(motions.shape[0], device=motions.device)
        
        # 聚合
        rewards = self.beta_s * R_sem + self.beta_p * R_phy
        return rewards
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
    
    def __call__(
        self,
        motions: torch.Tensor,
        prompts: Union[List[str], List[List[str]]],
        lengths: Optional[torch.Tensor] = None,
        text_lists: Optional[List[List[str]]] = None,
        segments: Optional[List[List[Tuple[int, int]]]] = None,
        durations: Optional[List[List[float]]] = None,
        foot_contact_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        计算组合奖励
        
        参数:
            motions: 生成的动作序列 [B, njoints, nfeats, nframes]
            prompts: 文本提示列表 [B] 或 List[List[str]]（向后兼容）
            lengths: 动作长度（可选）
            text_lists: 文本列表（新功能）
            segments: 分段信息（新功能）
            durations: 每个文本段对应的持续时间（秒）List[List[float]]（新功能）
            foot_contact_labels: 脚部接触标签（新功能）
            
        返回:
            rewards: 组合奖励值 [B]
        """
        matching_rewards = self.matching_reward(
            motions, prompts, lengths, text_lists, segments, durations, foot_contact_labels
        )
        r_precision_rewards = self.r_precision_reward(
            motions, prompts, lengths, text_lists, segments, durations, foot_contact_labels
        )
        
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


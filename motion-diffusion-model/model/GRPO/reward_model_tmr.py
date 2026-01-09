"""
基于 TMR (Text-to-Motion Retrieval) 预训练模型的奖励模型实现

TMR 是一个用于文本-动作检索的模型，通过对比学习将文本和动作映射到共同的嵌入空间。
本文件提供基于 TMR 预训练权重的奖励函数实现，可用于 GRPO 训练。

TMR 模型通常包含：
1. 文本编码器（Text Encoder）- 将文本编码为嵌入向量
2. 动作编码器（Motion Encoder）- 将动作序列编码为嵌入向量
3. 可能包含 Movement Encoder - 用于动作的预处理

奖励计算方式：
- 使用文本和动作嵌入之间的相似度（余弦相似度或欧氏距离）作为奖励
- 相似度越高，奖励越大
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Dict, Union, Tuple
from os.path import join as pjoin
import os

# 尝试导入 TMR 相关的模块
try:
    from data_loaders.humanml.networks.modules import (
        TextEncoderBiGRUCo,
        MotionEncoderBiGRUCo,
        MovementConvEncoder,
    )
    from data_loaders.humanml.utils.word_vectorizer import WordVectorizer, POS_enumerator
    _tmr_modules_available = True
except ImportError:
    _tmr_modules_available = False
    print("警告: TMR 相关模块未找到，请确保已正确安装依赖")

# 尝试导入 spacy，如果不可用则使用简单处理
try:
    import spacy
    _spacy_available = True
except ImportError:
    _spacy_available = False
    print("警告: spacy 未安装，将使用简单的文本处理方式。建议安装 spacy: pip install spacy && python -m spacy download en_core_web_sm")


class TMRModelWrapper:
    """
    TMR 模型包装器，用于加载和使用 TMR 预训练权重
    """
    
    def __init__(
        self,
        text_encoder_path: str,
        motion_encoder_path: str,
        movement_encoder_path: str,
        dataset_name: str = 'humanml',
        device: str = 'cuda',
    ):
        """
        初始化 TMR 模型包装器
        
        参数:
            text_encoder_path: 文本编码器权重路径 (text_encoder.pt)
            motion_encoder_path: 动作编码器权重路径 (motion_encoder.pt)
            movement_encoder_path: 动作解码器/编码器权重路径 (motion_decoder.pt 或 movement_encoder.pt)
            dataset_name: 数据集名称 ('humanml' 或 'kit')
            device: 设备
        """
        if not _tmr_modules_available:
            raise ImportError("TMR 相关模块未找到，无法加载 TMR 模型")
        
        self.device = device
        self.dataset_name = dataset_name
        
        # 根据数据集设置维度
        if dataset_name == 'humanml' or dataset_name == 't2m':
            dim_pose = 263
        elif dataset_name == 'kit':
            dim_pose = 251
        else:
            raise ValueError(f"不支持的数据集: {dataset_name}")
        
        # 构建模型
        self.text_encoder = TextEncoderBiGRUCo(
            word_size=300,
            pos_size=len(POS_enumerator),
            hidden_size=512,
            output_size=512,
            device=device
        )
        
        self.motion_encoder = MotionEncoderBiGRUCo(
            input_size=512,
            hidden_size=1024,
            output_size=512,
            device=device
        )
        
        self.movement_encoder = MovementConvEncoder(
            dim_pose - 4,  # 排除根位置和旋转
            512,  # hidden size
            512   # latent size
        )
        
        # 加载预训练权重
        self._load_checkpoints(text_encoder_path, motion_encoder_path, movement_encoder_path)
        
        # 移动到设备并设置为评估模式
        self.text_encoder.to(device)
        self.motion_encoder.to(device)
        self.movement_encoder.to(device)
        
        self.text_encoder.eval()
        self.motion_encoder.eval()
        self.movement_encoder.eval()
    
    def _load_checkpoints(
        self,
        text_encoder_path: str,
        motion_encoder_path: str,
        movement_encoder_path: str,
    ):
        """
        分别加载三个组件的预训练权重
        
        参数:
            text_encoder_path: 文本编码器权重路径
            motion_encoder_path: 动作编码器权重路径
            movement_encoder_path: 动作解码器/编码器权重路径
        """
        # 检查文件是否存在
        if not os.path.exists(text_encoder_path):
            raise FileNotFoundError(f"文本编码器权重文件不存在: {text_encoder_path}")
        if not os.path.exists(motion_encoder_path):
            raise FileNotFoundError(f"动作编码器权重文件不存在: {motion_encoder_path}")
        if not os.path.exists(movement_encoder_path):
            raise FileNotFoundError(f"动作解码器权重文件不存在: {movement_encoder_path}")
        
        # 加载文本编码器
        print(f"加载文本编码器权重: {text_encoder_path}")
        text_checkpoint = torch.load(text_encoder_path, map_location=self.device)
        if isinstance(text_checkpoint, dict):
            # 尝试不同的键名
            if 'model' in text_checkpoint:
                self.text_encoder.load_state_dict(text_checkpoint['model'], strict=False)
            elif 'state_dict' in text_checkpoint:
                self.text_encoder.load_state_dict(text_checkpoint['state_dict'], strict=False)
            elif 'text_encoder' in text_checkpoint:
                self.text_encoder.load_state_dict(text_checkpoint['text_encoder'], strict=False)
            else:
                # 直接使用字典作为 state_dict
                self.text_encoder.load_state_dict(text_checkpoint, strict=False)
        else:
            # 直接是 state_dict
            self.text_encoder.load_state_dict(text_checkpoint, strict=False)
        print("  文本编码器加载完成")
        
        # 加载动作编码器
        print(f"加载动作编码器权重: {motion_encoder_path}")
        motion_checkpoint = torch.load(motion_encoder_path, map_location=self.device)
        if isinstance(motion_checkpoint, dict):
            # 尝试不同的键名
            if 'model' in motion_checkpoint:
                self.motion_encoder.load_state_dict(motion_checkpoint['model'], strict=False)
            elif 'state_dict' in motion_checkpoint:
                self.motion_encoder.load_state_dict(motion_checkpoint['state_dict'], strict=False)
            elif 'motion_encoder' in motion_checkpoint:
                self.motion_encoder.load_state_dict(motion_checkpoint['motion_encoder'], strict=False)
            else:
                # 直接使用字典作为 state_dict
                self.motion_encoder.load_state_dict(motion_checkpoint, strict=False)
        else:
            # 直接是 state_dict
            self.motion_encoder.load_state_dict(motion_checkpoint, strict=False)
        print("  动作编码器加载完成")
        
        # 加载动作解码器/编码器 (movement encoder)
        print(f"加载动作解码器权重: {movement_encoder_path}")
        movement_checkpoint = torch.load(movement_encoder_path, map_location=self.device)
        if isinstance(movement_checkpoint, dict):
            # 尝试不同的键名
            if 'model' in movement_checkpoint:
                self.movement_encoder.load_state_dict(movement_checkpoint['model'], strict=False)
            elif 'state_dict' in movement_checkpoint:
                self.movement_encoder.load_state_dict(movement_checkpoint['state_dict'], strict=False)
            elif 'movement_encoder' in movement_checkpoint:
                self.movement_encoder.load_state_dict(movement_checkpoint['movement_encoder'], strict=False)
            elif 'motion_decoder' in movement_checkpoint:
                self.movement_encoder.load_state_dict(movement_checkpoint['motion_decoder'], strict=False)
            else:
                # 直接使用字典作为 state_dict
                self.movement_encoder.load_state_dict(movement_checkpoint, strict=False)
        else:
            # 直接是 state_dict
            self.movement_encoder.load_state_dict(movement_checkpoint, strict=False)
        print("  动作解码器加载完成")
        
        print("TMR 模型所有组件加载完成")
    
    def encode_text(
        self,
        word_embs: torch.Tensor,
        pos_ohot: torch.Tensor,
        cap_lens: torch.Tensor,
    ) -> torch.Tensor:
        """
        编码文本
        
        参数:
            word_embs: 词嵌入 [B, max_len, 300]
            pos_ohot: 词性 one-hot [B, max_len, pos_dim]
            cap_lens: 文本长度 [B]
            
        返回:
            text_embeddings: 文本嵌入 [B, 512]
        """
        with torch.no_grad():
            text_embeddings = self.text_encoder(word_embs, pos_ohot, cap_lens)
        return text_embeddings
    
    def encode_motion(
        self,
        motions: torch.Tensor,
        m_lens: torch.Tensor,
    ) -> torch.Tensor:
        """
        编码动作序列
        
        参数:
            motions: 动作序列 [B, nframes, njoints*nfeats] 或 [B, njoints, nfeats, nframes]
            m_lens: 动作长度 [B]
            
        返回:
            motion_embeddings: 动作嵌入 [B, 512]
        """
        with torch.no_grad():
            # 如果输入是 [B, njoints, nfeats, nframes]，需要转换
            if len(motions.shape) == 4:
                B, njoints, nfeats, nframes = motions.shape
                motions = motions.permute(0, 3, 1, 2).reshape(B, nframes, -1)
            
            # Movement encoding
            movements = self.movement_encoder(motions[..., :-4])  # 排除根位置和旋转
            m_lens = m_lens // 4  # unit_length = 4
            
            # Motion encoding
            motion_embeddings = self.motion_encoder(movements, m_lens)
        
        return motion_embeddings
    
    def get_co_embeddings(
        self,
        word_embs: torch.Tensor,
        pos_ohot: torch.Tensor,
        cap_lens: torch.Tensor,
        motions: torch.Tensor,
        m_lens: torch.Tensor,
    ) -> tuple:
        """
        同时获取文本和动作嵌入
        
        参数:
            word_embs: 词嵌入 [B, max_len, 300]
            pos_ohot: 词性 one-hot [B, max_len, pos_dim]
            cap_lens: 文本长度 [B]
            motions: 动作序列 [B, nframes, njoints*nfeats] 或 [B, njoints, nfeats, nframes]
            m_lens: 动作长度 [B]
            
        返回:
            text_embeddings: 文本嵌入 [B, 512]
            motion_embeddings: 动作嵌入 [B, 512]
        """
        text_embeddings = self.encode_text(word_embs, pos_ohot, cap_lens)
        motion_embeddings = self.encode_motion(motions, m_lens)
        return text_embeddings, motion_embeddings


class TMRRewardFunction:
    """
    基于 TMR 模型的奖励函数基类
    """
    
    def __init__(
        self,
        text_encoder_path: str,
        motion_encoder_path: str,
        movement_encoder_path: str,
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
        初始化 TMR 奖励函数
        
        参数:
            text_encoder_path: 文本编码器权重路径 (text_encoder.pt)
            motion_encoder_path: 动作编码器权重路径 (motion_encoder.pt)
            movement_encoder_path: 动作解码器权重路径 (motion_decoder.pt)
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
        
        # 初始化 TMR 模型
        self.tmr_model = TMRModelWrapper(
            text_encoder_path=text_encoder_path,
            motion_encoder_path=motion_encoder_path,
            movement_encoder_path=movement_encoder_path,
            dataset_name=dataset_name,
            device=device,
        )
        
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
            pos_list = ['NOUN'] * len(words)
            return words, pos_list
    
    def _prepare_text_inputs(self, prompts: List[str]):
        """
        将文本提示转换为 TMR 模型所需的格式
        
        参数:
            prompts: 文本提示列表
            
        返回:
            word_embs: 词嵌入 [B, max_len, 300]
            pos_ohot: 词性 one-hot [B, max_len, pos_dim]
            cap_lens: 文本长度 [B]
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
            
            # 添加 SOS 和 EOS tokens
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
            motions_processed: 处理后的动作 [B, nframes, njoints*nfeats]
            m_lens: 动作长度 [B]
        """
        batch_size = motions.shape[0]
        
        if lengths is None:
            # 假设使用完整长度
            m_lens = torch.full((batch_size,), motions.shape[-1], dtype=torch.long, device=self.device)
        else:
            m_lens = lengths.to(self.device)
        
        # 转换格式：从 [B, njoints, nfeats, nframes] 到 [B, nframes, njoints*nfeats]
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
            if motions_flat.shape[-1] >= 3:
                root_vel = motions_flat[:, :, :2]  # [B, nframes, 2] (x, z 方向速度)
            else:
                root_vel = torch.zeros(B, nframes, 2, device=motions.device)
            
            # 计算滑行惩罚 L_skate
            L_skate = torch.zeros(B, device=motions.device)
            if foot_contact_labels is not None:
                contact_mask = foot_contact_labels.sum(dim=-1) > 0  # [B, nframes]
                
                for b in range(B):
                    contact_frames = contact_mask[b]  # [nframes]
                    if contact_frames.any():
                        contact_vel = root_vel[b, contact_frames]  # [N_contact, 2]
                        vel_squared = (contact_vel ** 2).sum(dim=-1)  # [N_contact]
                        L_skate[b] = vel_squared.mean()
            else:
                L_skate = torch.zeros(B, device=motions.device)
            
            # 计算加速度突变惩罚 L_jerk
            if nframes > 1:
                velocities = motions_flat[:, 1:] - motions_flat[:, :-1]  # [B, nframes-1, dim]
                accelerations = velocities[:, 1:] - velocities[:, :-1]  # [B, nframes-2, dim]
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
                text_embeddings, motion_embeddings = self.tmr_model.get_co_embeddings(
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
                        text_emb_k, motion_emb_k = self.tmr_model.get_co_embeddings(
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
                            text_emb_k, _ = self.tmr_model.get_co_embeddings(
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
                            text_placeholder = text_lists[b][0]
                            # 确保 text_placeholder 是字符串（如果是列表则连接）
                            if isinstance(text_placeholder, list):
                                text_placeholder = " ".join(text_placeholder)
                            word_embs_placeholder, pos_ohot_placeholder, cap_lens_placeholder = self._prepare_text_inputs([text_placeholder])
                            _, motion_emb_k = self.tmr_model.get_co_embeddings(
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


class TMRMatchingScoreReward(TMRRewardFunction):
    """
    基于 TMR 匹配分数的奖励函数
    
    使用文本和动作嵌入之间的相似度（余弦相似度或欧氏距离）作为奖励。
    """
    
    def __init__(
        self,
        text_encoder_path: str,
        motion_encoder_path: str,
        movement_encoder_path: str,
        similarity_type: str = 'cosine',  # 'cosine' 或 'euclidean'
        max_distance: float = 10.0,  # 用于欧氏距离归一化
        scale: float = 2.0,  # 用于指数衰减
        normalization: str = 'linear',  # 'linear', 'exponential', 'sigmoid'
        *args,
        **kwargs,
    ):
        """
        参数:
            text_encoder_path: 文本编码器权重路径 (text_encoder.pt)
            motion_encoder_path: 动作编码器权重路径 (motion_encoder.pt)
            movement_encoder_path: 动作解码器权重路径 (motion_decoder.pt)
            similarity_type: 相似度类型 ('cosine' 或 'euclidean')
            max_distance: 最大距离（用于线性归一化）
            scale: 缩放因子（用于指数衰减）
            normalization: 归一化方式 ('linear', 'exponential', 'sigmoid')
        """
        super().__init__(text_encoder_path, motion_encoder_path, movement_encoder_path, *args, **kwargs)
        self.similarity_type = similarity_type
        self.max_distance = max_distance
        self.scale = scale
        self.normalization = normalization
    
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
        计算基于 TMR 匹配分数的奖励
        
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


class TMRCosineSimilarityReward(TMRRewardFunction):
    """
    基于 TMR 余弦相似度的奖励函数（简化版本）
    """
    
    def __call__(
        self,
        motions: torch.Tensor,
        prompts: Union[List[str], List[List[str]]],
        lengths: Optional[torch.Tensor] = None,
        text_lists: Optional[List[List[str]]] = None,
        segments: Optional[List[List[Tuple[int, int]]]] = None,
        foot_contact_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        计算基于余弦相似度的奖励
        
        参数:
            motions: 生成的动作序列 [B, njoints, nfeats, nframes]
            prompts: 文本提示列表 [B] 或 List[List[str]]（向后兼容）
            lengths: 动作长度（可选）
            text_lists: 文本列表（新功能）
            segments: 分段信息（新功能）
            foot_contact_labels: 脚部接触标签（新功能）
            
        返回:
            rewards: 奖励值 [B]
        """
        # 使用基类的语义奖励计算
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


def create_tmr_reward_function(
    text_encoder_path: str,
    motion_encoder_path: str,
    movement_encoder_path: str,
    reward_type: str = 'matching',
    dataset_name: str = 'humanml',
    device: str = 'cuda',
    **kwargs,
) -> TMRRewardFunction:
    """
    创建 TMR 奖励函数的工厂函数
    
    参数:
        text_encoder_path: 文本编码器权重路径 (text_encoder.pt)
        motion_encoder_path: 动作编码器权重路径 (motion_encoder.pt)
        movement_encoder_path: 动作解码器权重路径 (motion_decoder.pt)
        reward_type: 奖励类型 ('matching', 'cosine')
        dataset_name: 数据集名称
        device: 设备
        **kwargs: 其他参数（如 similarity_type, max_distance, scale, normalization）
        
    返回:
        reward_function: 奖励函数实例
    """
    if reward_type == 'matching':
        return TMRMatchingScoreReward(
            text_encoder_path=text_encoder_path,
            motion_encoder_path=motion_encoder_path,
            movement_encoder_path=movement_encoder_path,
            dataset_name=dataset_name,
            device=device,
            **kwargs,
        )
    elif reward_type == 'cosine':
        return TMRCosineSimilarityReward(
            text_encoder_path=text_encoder_path,
            motion_encoder_path=motion_encoder_path,
            movement_encoder_path=movement_encoder_path,
            dataset_name=dataset_name,
            device=device,
            **kwargs,
        )
    else:
        raise ValueError(f"未知的奖励类型: {reward_type}")


# 使用示例
if __name__ == '__main__':
    # 示例：创建 TMR 奖励函数
    # 注意：需要提供三个 TMR 预训练权重文件路径
    text_encoder_path = './path/to/tmr/text_encoder.pt'
    motion_encoder_path = './path/to/tmr/motion_encoder.pt'
    movement_encoder_path = './path/to/tmr/motion_decoder.pt'
    
    # 创建奖励函数（使用余弦相似度）
    reward_fn = create_tmr_reward_function(
        text_encoder_path=text_encoder_path,
        motion_encoder_path=motion_encoder_path,
        movement_encoder_path=movement_encoder_path,
        reward_type='cosine',
        device='cuda',
    )
    
    # 或者使用匹配分数（可配置）
    reward_fn = create_tmr_reward_function(
        text_encoder_path=text_encoder_path,
        motion_encoder_path=motion_encoder_path,
        movement_encoder_path=movement_encoder_path,
        reward_type='matching',
        similarity_type='cosine',  # 或 'euclidean'
        normalization='linear',  # 或 'exponential', 'sigmoid'
        device='cuda',
    )
    
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


"""
复合数据集加载器

用于加载通过 construct_composite_dataset.py 生成的复合数据集
"""

import os
import numpy as np
import torch
from torch.utils import data
from os.path import join as pjoin
from typing import List, Dict, Optional
import random
from data_loaders.humanml.utils.word_vectorizer import WordVectorizer
import spacy


class CompositeDataset(data.Dataset):
    """
    复合数据集类
    
    从 .npy 文件加载复合样本，每个样本包含：
    - composite_prompt: 拼接后的长句子
    - sub_prompts: 原始的子文本描述列表
    - durations: 对应的持续时间（秒）
    - durations_frames: 对应的帧数
    - source_ids: 原始 HumanML3D 的 ID
    
    注意：B_matrix 不再预计算，将在训练时由奖励函数根据 text_lists 动态计算
    """
    
    def __init__(
        self,
        composite_data_path: str,
        mean: np.ndarray,
        std: np.ndarray,
        w_vectorizer: WordVectorizer,
        max_motion_length: int = 196,
        fps: float = 20.0,
        mode: str = 'eval',
        mean_for_eval: np.ndarray = None,
        std_for_eval: np.ndarray = None,
    ):
        """
        初始化复合数据集
        
        参数:
            composite_data_path: 复合数据集 .npy 文件路径
            mean: 动作数据的均值（用于归一化）
            std: 动作数据的标准差（用于归一化）
            w_vectorizer: 词向量化器
            max_motion_length: 最大动作长度（帧）
            fps: 帧率
            mode: 数据集模式（'train', 'eval', 'gt'等）
            mean_for_eval: 用于评估的均值（如果为None，使用mean）
            std_for_eval: 用于评估的标准差（如果为None，使用std）
        """
        self.mean = mean
        self.std = std
        self.max_motion_length = max_motion_length
        self.fps = fps
        self.w_vectorizer = w_vectorizer
        self.max_text_len = 20
        self.mode = mode
        
        # 用于评估的均值和标准差（兼容 comp_v6_model_dataset.py）
        self.mean_for_eval = mean_for_eval if mean_for_eval is not None else mean
        self.std_for_eval = std_for_eval if std_for_eval is not None else std
        
        # 创建一个简单的 t2m_dataset 包装器以兼容 comp_v6_model_dataset.py
        # 这个包装器只提供 inv_transform 方法
        class SimpleT2MDataset:
            def __init__(self, mean, std):
                self.mean = mean
                self.std = std
            
            def inv_transform(self, data):
                """逆变换：从归一化数据恢复到原始数据"""
                return data * self.std + self.mean
        
        self.t2m_dataset = SimpleT2MDataset(mean, std)
        
        # 初始化 spacy 用于文本处理（将普通文本转换为 word/POS 格式）
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            print("警告: 无法加载 spacy 模型 'en_core_web_sm'，将使用简单的文本处理")
            self.nlp = None
        
        # 加载复合数据集
        print(f"Loading composite dataset from {composite_data_path}...")
        data_dict = np.load(composite_data_path, allow_pickle=True)[None][0]
        
        self.samples = data_dict['samples']
        self.metadata = data_dict.get('metadata', {})
        
        # 从文件名推断 split（如果文件名包含 split 信息）
        # 例如：composite_k3_test.npy -> test
        file_split = 'unknown'
        if 'test' in composite_data_path.lower():
            file_split = 'test'
        elif 'val' in composite_data_path.lower() or 'validation' in composite_data_path.lower():
            file_split = 'val'
        elif 'train' in composite_data_path.lower():
            file_split = 'train'
        
        # 优先使用 metadata 中的 split，如果不存在则使用从文件名推断的
        actual_split = self.metadata.get('split', file_split)
        
        print(f"Loaded {len(self.samples)} composite samples")
        print(f"K={self.metadata.get('k_segments', 'unknown')}, "
              f"Split={actual_split} (from metadata: {self.metadata.get('split', 'N/A')}, from filename: {file_split})")
        
        # 添加 num_actions 属性以兼容评估代码（与 HumanML3D 数据集保持一致）
        self.num_actions = 1  # dummy placeholder
    
    def __len__(self):
        return len(self.samples)
    
    def _process_text_to_tokens(self, sentence):
        """将普通文本转换为 word/POS 格式的 tokens（参考 RawTextDataset）"""
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
        tokens = ['%s/%s' % (word_list[i], pos_list[i]) for i in range(len(word_list))]
        return tokens
    
    def __getitem__(self, idx):
        """
        获取一个复合样本
        
        返回格式与 Text2MotionDatasetV2 兼容：
        word_embeddings, pos_one_hots, caption, sent_len, motion, length, tokens
        """
        sample = self.samples[idx]
        
        # 提取信息
        composite_prompt = sample['composite_prompt']
        sub_prompts = sample['sub_prompts']  # List[str]
        durations = sample['durations']  # List[float] (秒)
        durations_frames = sample['durations_frames']  # List[int] (帧)
        source_ids = sample['source_ids']  # List[str]
        
        # 处理文本：将 composite_prompt 转换为 word/POS 格式的 tokens
        if self.nlp is not None:
            # 使用 spacy 处理文本，提取词性和词根
            tokens = self._process_text_to_tokens(composite_prompt)
        else:
            # 简单处理：假设所有词都是 OTHER 类型
            words = composite_prompt.split()
            tokens = [f'{word}/OTHER' for word in words]
        
        if len(tokens) < self.max_text_len:
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.max_text_len + 2 - sent_len)
        else:
            tokens = tokens[:self.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        
        # 准备词嵌入和 POS
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)
        
        # 注意：复合数据集不包含实际的 motion 数据
        # 这里返回一个占位符，实际 motion 会在训练时生成
        # 为了兼容性，返回一个零数组
        motion = np.zeros((self.max_motion_length, 263))  # HumanML3D 特征维度为 263
        length = self.max_motion_length
        
        # 将 sub_prompts 转换为 text_lists 格式（用于奖励函数）
        text_lists = [[prompt] for prompt in sub_prompts]
        
        # 返回格式与 Text2MotionDatasetV2 兼容
        return {
            'word_embeddings': word_embeddings,
            'pos_one_hots': pos_one_hots,
            'caption': composite_prompt,
            'sent_len': sent_len,
            'motion': motion,  # 占位符
            'length': length,
            'tokens': '_'.join(tokens),
            # 复合数据集特有字段
            'composite_prompt': composite_prompt,
            'sub_prompts': sub_prompts,
            'durations': durations,
            'durations_frames': durations_frames,
            'source_ids': source_ids,
            'text_lists': text_lists,
        }


def composite_collate_fn(batch):
    """
    复合数据集的 collate 函数
    
    返回格式与 t2m_collate 兼容
    """
    # 提取标准字段
    word_embeddings = [item['word_embeddings'] for item in batch]
    pos_one_hots = [item['pos_one_hots'] for item in batch]
    captions = [item['caption'] for item in batch]
    sent_lens = [item['sent_len'] for item in batch]
    motions = [item['motion'] for item in batch]
    lengths = [item['length'] for item in batch]
    tokens = [item['tokens'] for item in batch]
    
    # 提取复合数据集特有字段
    composite_prompts = [item['composite_prompt'] for item in batch]
    sub_prompts_list = [item['sub_prompts'] for item in batch]
    durations_list = [item['durations'] for item in batch]
    durations_frames_list = [item['durations_frames'] for item in batch]
    source_ids_list = [item['source_ids'] for item in batch]
    text_lists_list = [item['text_lists'] for item in batch]
    
    # 转换为标准格式（与 Text2MotionDatasetV2 兼容）
    standard_batch = []
    for i in range(len(batch)):
        standard_batch.append((
            word_embeddings[i],
            pos_one_hots[i],
            captions[i],
            sent_lens[i],
            motions[i],
            lengths[i],
            tokens[i],
        ))
    
    # 使用 t2m_collate 处理标准字段
    from data_loaders.tensors import t2m_collate
    motion_tensor, cond = t2m_collate(standard_batch, len(batch))
    
    # 添加复合数据集特有字段到 cond
    cond['y']['composite_prompts'] = composite_prompts
    cond['y']['sub_prompts'] = sub_prompts_list
    cond['y']['durations'] = durations_list
    cond['y']['durations_frames'] = durations_frames_list
    cond['y']['source_ids'] = source_ids_list
    cond['y']['text_lists'] = text_lists_list
    
    return motion_tensor, cond


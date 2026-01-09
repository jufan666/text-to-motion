"""
复合数据集构造脚本

功能：
1. 从 HumanML3D 数据集中选择 K=3/4/5 个动作，组合成复合样本
2. 计算每个片段的 duration（秒数）
3. 检查长度合法性（总长度接近 196 帧）
4. 保存为 .npy 文件

注意：B_matrix 不再预计算，将在训练时由奖励函数根据 text_lists 动态计算
"""

import os
import sys
import numpy as np
import argparse
from tqdm import tqdm
from os.path import join as pjoin
import codecs as cs
from typing import List, Dict, Optional
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loaders.humanml.utils.get_opt import get_opt


def load_dataset_info(opt, split_file):
    """加载数据集信息"""
    id_list = []
    with cs.open(split_file, 'r') as f:
        for line in f.readlines():
            id_list.append(line.strip())
    
    data_dict = {}
    min_motion_len = 40 if opt.dataset_name == 't2m' else 24
    
    print(f"Loading dataset from {split_file}...")
    for name in tqdm(id_list, desc="Loading motions"):
        try:
            motion_path = pjoin(opt.motion_dir, name + '.npy')
            if not os.path.exists(motion_path):
                continue
            
            motion = np.load(motion_path)
            if len(motion) < min_motion_len or len(motion) >= 200:
                continue
            
            text_data = []
            text_path = pjoin(opt.text_dir, name + '.txt')
            if not os.path.exists(text_path):
                continue
            
            with cs.open(text_path) as f:
                for line in f.readlines():
                    text_dict = {}
                    line_split = line.strip().split('#')
                    if len(line_split) < 4:
                        continue
                    
                    caption = line_split[0]
                    tokens = line_split[1].split(' ')
                    f_tag = float(line_split[2])
                    to_tag = float(line_split[3])
                    f_tag = 0.0 if np.isnan(f_tag) else f_tag
                    to_tag = 0.0 if np.isnan(to_tag) else to_tag
                    
                    text_dict['caption'] = caption
                    text_dict['tokens'] = tokens
                    text_dict['f_tag'] = f_tag
                    text_dict['to_tag'] = to_tag
                    text_data.append(text_dict)
            
            if len(text_data) > 0:
                data_dict[name] = {
                    'motion': motion,
                    'length': len(motion),
                    'text': text_data
                }
        except Exception as e:
            continue
    
    print(f"Loaded {len(data_dict)} samples")
    return data_dict, id_list


def create_composite_sample(sample_ids, data_dict, k_segments, fps=20.0, target_length=196, 
                           tolerance=20):
    """创建一个复合样本"""
    if len(sample_ids) != k_segments:
        return None
    
    for sample_id in sample_ids:
        if sample_id not in data_dict:
            return None
    
    sub_prompts = []
    durations_seconds = []
    durations_frames = []
    source_ids = []
    
    for sample_id in sample_ids:
        sample_data = data_dict[sample_id]
        text_list = sample_data['text']
        
        selected_text = None
        for text_dict in text_list:
            if text_dict['f_tag'] == 0.0 and text_dict['to_tag'] == 0.0:
                selected_text = text_dict
                break
        
        if selected_text is None:
            selected_text = text_list[0]
        
        caption = selected_text['caption']
        f_tag = selected_text['f_tag']
        to_tag = selected_text['to_tag']
        
        if f_tag == 0.0 and to_tag == 0.0:
            duration_seconds = sample_data['length'] / fps
            duration_frames = sample_data['length']
        else:
            duration_seconds = to_tag - f_tag
            duration_frames = int((to_tag - f_tag) * fps)
        
        sub_prompts.append(caption)
        durations_seconds.append(duration_seconds)
        durations_frames.append(duration_frames)
        source_ids.append(sample_id)
    
    total_frames = sum(durations_frames)
    # 确保总长度不超过 target_length（MDM 无法处理超过 196 帧的序列）
    # tolerance 用于允许一定的下界（避免过短），但上限必须严格不超过 target_length
    if total_frames > target_length or total_frames < target_length - tolerance:
        return None
    
    if k_segments == 3:
        composite_prompt = f"First {sub_prompts[0]}, then {sub_prompts[1]}, finally {sub_prompts[2]}"
    elif k_segments == 4:
        composite_prompt = f"First {sub_prompts[0]}, then {sub_prompts[1]}, then {sub_prompts[2]}, finally {sub_prompts[3]}"
    elif k_segments == 5:
        composite_prompt = f"First {sub_prompts[0]}, then {sub_prompts[1]}, then {sub_prompts[2]}, then {sub_prompts[3]}, finally {sub_prompts[4]}"
    else:
        parts = [f"{'First' if i == 0 else 'then' if i < k_segments - 1 else 'finally'} {sub_prompts[i]}" 
                 for i in range(k_segments)]
        composite_prompt = ", ".join(parts)
    
    composite_sample = {
        'composite_prompt': composite_prompt,
        'sub_prompts': sub_prompts,
        'durations': durations_seconds,  # 秒数
        'durations_frames': durations_frames,  # 帧数
        'source_ids': source_ids,
        'total_frames': total_frames,
    }
    
    return composite_sample


def construct_composite_dataset(dataset_name='humanml', split='train', k_segments=3,
                                output_dir='dataset/HumanML3D/composite', target_length=196,
                                tolerance=20, fps=20.0, max_samples=1000,
                                abs_path='.', cache_path='.'):
    """构造复合数据集"""
    print(f"=== 构造复合数据集 ===")
    print(f"数据集: {dataset_name}, 划分: {split}, K={k_segments}")
    print(f"目标长度: ≤{target_length} 帧（上限），≥{target_length - tolerance} 帧（下限）")
    print(f"最大样本数: {max_samples}")
    
    if dataset_name == 'humanml' or dataset_name == 't2m':
        datapath = pjoin(abs_path, 'dataset/humanml_opt.txt')
    elif dataset_name == 'kit':
        datapath = pjoin(abs_path, 'dataset/kit_opt.txt')
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")
    
    # 数据集构造不需要 GPU，使用 'cpu' 即可
    opt = get_opt(datapath, 'cpu')
    opt.motion_dir = pjoin(abs_path, opt.motion_dir)
    opt.text_dir = pjoin(abs_path, opt.text_dir)
    opt.data_root = pjoin(abs_path, opt.data_root)
    opt.cache_dir = cache_path
    
    split_file = pjoin(opt.data_root, f'{split}.txt')
    data_dict, id_list = load_dataset_info(opt, split_file)
    
    if len(data_dict) < k_segments:
        raise ValueError(f"数据集样本数 ({len(data_dict)}) 少于 K ({k_segments})")
    
    composite_samples = []
    id_list_shuffled = id_list.copy()
    random.shuffle(id_list_shuffled)
    
    # 用于去重，避免重复尝试相同的组合
    tried_combinations = set()
    
    print(f"\n构造复合样本 (K={k_segments})...")
    pbar = tqdm(total=max_samples, desc="Creating composite samples")
    
    idx = 0
    attempts = 0
    consecutive_failures = 0
    max_attempts = max_samples * 200  # 增加最大尝试次数
    max_consecutive_failures = len(id_list) * 10  # 如果连续失败太多次，重新洗牌
    
    while len(composite_samples) < max_samples and attempts < max_attempts:
        attempts += 1
        
        # 如果索引超出范围，重新洗牌
        if idx + k_segments > len(id_list_shuffled):
            random.shuffle(id_list_shuffled)
            idx = 0
            consecutive_failures = 0  # 重置连续失败计数
        
        sample_ids = id_list_shuffled[idx:idx+k_segments]
        idx += k_segments
        
        # 检查是否已经尝试过这个组合（去重）
        combination_key = tuple(sorted(sample_ids))
        if combination_key in tried_combinations:
            consecutive_failures += 1
            if consecutive_failures > max_consecutive_failures:
                # 如果连续失败太多次，重新洗牌并清空已尝试组合
                random.shuffle(id_list_shuffled)
                idx = 0
                consecutive_failures = 0
                tried_combinations.clear()
            continue
        
        tried_combinations.add(combination_key)
        
        composite_sample = create_composite_sample(
            sample_ids=sample_ids,
            data_dict=data_dict,
            k_segments=k_segments,
            fps=fps,
            target_length=target_length,
            tolerance=tolerance,
        )
        
        if composite_sample is not None:
            composite_samples.append(composite_sample)
            pbar.update(1)
            consecutive_failures = 0  # 重置连续失败计数
        else:
            consecutive_failures += 1
    
    pbar.close()
    
    if attempts >= max_attempts:
        print(f"\n⚠️  警告: 达到最大尝试次数 ({max_attempts})，停止构造")
        print(f"   成功生成 {len(composite_samples)}/{max_samples} 个样本")
        print(f"   成功率: {len(composite_samples)/attempts*100:.2f}%")
        print(f"   建议: 增大 --tolerance 参数或减小 --target_length 参数以提高成功率")
    
    if len(composite_samples) == 0:
        raise ValueError(
            f"未能创建任何复合样本。请检查参数：\n"
            f"  - 总长度必须 ≤ {target_length} 帧（MDM 无法处理超过此长度的序列）\n"
            f"  - 总长度建议 ≥ {target_length - tolerance} 帧（避免过短）\n"
            f"  - 尝试增大 --tolerance 参数以允许更短的长度"
        )
    
    print(f"\n✓ 成功创建 {len(composite_samples)} 个复合样本")
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = pjoin(output_dir, f'composite_k{k_segments}_{split}.npy')
    
    print(f"\n保存到 {output_path}...")
    np.save(output_path, {
        'samples': composite_samples,
        'metadata': {
            'dataset_name': dataset_name,
            'split': split,
            'k_segments': k_segments,
            'target_length': target_length,
            'tolerance': tolerance,
            'fps': fps,
            'num_samples': len(composite_samples),
        }
    })
    
    print(f"✓ 保存完成: {output_path}")
    
    total_frames_list = [s['total_frames'] for s in composite_samples]
    print(f"\n=== 统计信息 ===")
    print(f"总样本数: {len(composite_samples)}")
    print(f"总帧数范围: [{min(total_frames_list)}, {max(total_frames_list)}]")
    print(f"平均总帧数: {np.mean(total_frames_list):.1f}")
    print(f"标准差: {np.std(total_frames_list):.1f}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='构造复合数据集')
    parser.add_argument('--dataset', type=str, default='humanml', choices=['humanml', 'kit', 't2m'])
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'])
    parser.add_argument('--k_segments', type=int, default=3, choices=[3, 4, 5])
    parser.add_argument('--output_dir', type=str, default='dataset/HumanML3D/composite')
    parser.add_argument('--target_length', type=int, default=196)
    parser.add_argument('--tolerance', type=int, default=20)
    parser.add_argument('--fps', type=float, default=20.0)
    parser.add_argument('--max_samples', type=int, default=1000)
    parser.add_argument('--abs_path', type=str, default='.')
    parser.add_argument('--cache_path', type=str, default='.')
    
    args = parser.parse_args()
    
    construct_composite_dataset(
        dataset_name=args.dataset,
        split=args.split,
        k_segments=args.k_segments,
        output_dir=args.output_dir,
        target_length=args.target_length,
        tolerance=args.tolerance,
        fps=args.fps,
        max_samples=args.max_samples,
        abs_path=args.abs_path,
        cache_path=args.cache_path,
    )


if __name__ == '__main__':
    main()


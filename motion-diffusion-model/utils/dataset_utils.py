"""
数据集工具函数：用于限制训练样本数量（用于快速验证）
"""
import torch
from torch.utils.data import Subset, DataLoader


def limit_dataset_size(dataset, max_samples: int):
    """
    限制数据集的大小，用于快速验证。
    
    参数:
        dataset: PyTorch Dataset 实例或 DataLoader
        max_samples: 最大样本数
    
    返回:
        Subset 或修改后的 DataLoader
    """
    if max_samples <= 0:
        return dataset
    
    # 如果是 DataLoader，需要访问其 dataset 属性
    if isinstance(dataset, DataLoader):
        original_dataset = dataset.dataset
        dataset_size = len(original_dataset)
        
        if dataset_size <= max_samples:
            print(f"数据集大小 ({dataset_size}) <= 限制 ({max_samples})，无需截断")
            return dataset
        
        # 创建子集
        indices = list(range(min(max_samples, dataset_size)))
        subset = Subset(original_dataset, indices)
        print(f"限制数据集大小: {dataset_size} -> {len(indices)} 个样本")
        
        # 从原始 DataLoader 中提取参数
        # DataLoader 的关键参数可以通过其属性或 __dict__ 获取
        loader_kwargs = {
            'batch_size': dataset.batch_size,
            'num_workers': dataset.num_workers,
            'pin_memory': getattr(dataset, 'pin_memory', False),
            'drop_last': getattr(dataset, 'drop_last', False),
        }
        
        # collate_fn 从 DataLoader 的 collate_fn 属性获取
        if hasattr(dataset, 'collate_fn') and dataset.collate_fn is not None:
            loader_kwargs['collate_fn'] = dataset.collate_fn
        
        # shuffle 参数需要从 DataLoader 的 sampler 推断
        # DataLoader 如果 shuffle=True，会使用 RandomSampler
        from torch.utils.data import RandomSampler, SequentialSampler
        if isinstance(dataset.sampler, RandomSampler) or (hasattr(dataset, 'shuffle') and dataset.shuffle):
            loader_kwargs['shuffle'] = True
        else:
            loader_kwargs['shuffle'] = False
        
        # 创建新的 DataLoader，保持原有参数
        return DataLoader(subset, **loader_kwargs)
    else:
        # 如果是 Dataset
        dataset_size = len(dataset)
        if dataset_size <= max_samples:
            print(f"数据集大小 ({dataset_size}) <= 限制 ({max_samples})，无需截断")
            return dataset
        
        indices = list(range(min(max_samples, dataset_size)))
        print(f"限制数据集大小: {dataset_size} -> {len(indices)} 个样本")
        return Subset(dataset, indices)


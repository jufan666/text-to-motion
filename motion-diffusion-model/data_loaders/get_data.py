from torch.utils.data import DataLoader
from data_loaders.tensors import collate as all_collate
from data_loaders.tensors import t2m_collate, t2m_prefix_collate

def get_dataset_class(name):
    if name == "amass":
        from .amass import AMASS
        return AMASS
    elif name == "uestc":
        from .a2m.uestc import UESTC
        return UESTC
    elif name == "humanact12":
        from .a2m.humanact12poses import HumanAct12Poses
        return HumanAct12Poses
    elif name == "humanml":
        from data_loaders.humanml.data.dataset import HumanML3D
        return HumanML3D
    elif name == "kit":
        from data_loaders.humanml.data.dataset import KIT
        return KIT
    else:
        raise ValueError(f'Unsupported dataset name [{name}]')

def get_collate_fn(name, hml_mode='train', pred_len=0, batch_size=1):
    if hml_mode == 'gt':
        from data_loaders.humanml.data.dataset import collate_fn as t2m_eval_collate
        return t2m_eval_collate
    if name in ["humanml", "kit"]:
        if pred_len > 0:
            return lambda x: t2m_prefix_collate(x, pred_len=pred_len)
        return lambda x: t2m_collate(x, batch_size)
    else:
        return all_collate


def get_dataset(name, num_frames, split='train', hml_mode='train', abs_path='.', fixed_len=0, 
                device=None, autoregressive=False, cache_path=None, disable_random_crop=False,
                use_composite_dataset=False, composite_data_path=None, composite_k_segments=3): 
    # 如果使用复合数据集
    if use_composite_dataset and composite_data_path is not None:
        from data_loaders.humanml.data.composite_dataset import CompositeDataset
        from data_loaders.humanml.utils.word_vectorizer import WordVectorizer
        from os.path import join as pjoin
        import numpy as np
        
        # 加载均值和标准差
        if name == 'humanml' or name == 't2m':
            mean = np.load(pjoin(abs_path, 'dataset/HumanML3D/Mean.npy'))
            std = np.load(pjoin(abs_path, 'dataset/HumanML3D/Std.npy'))
            # 加载用于评估的均值和标准差（T2M 评估器使用的格式）
            try:
                from data_loaders.humanml.utils.get_opt import get_opt
                opt = get_opt(pjoin(abs_path, 'dataset/humanml_opt.txt'), 'cpu')
                mean_for_eval = np.load(pjoin(opt.meta_dir, f'{opt.dataset_name}_mean.npy'))
                std_for_eval = np.load(pjoin(opt.meta_dir, f'{opt.dataset_name}_std.npy'))
            except:
                # 如果无法加载，使用默认值
                mean_for_eval = mean
                std_for_eval = std
        elif name == 'kit':
            mean = np.load(pjoin(abs_path, 'dataset/KIT-ML/Mean.npy'))
            std = np.load(pjoin(abs_path, 'dataset/KIT-ML/Std.npy'))
            try:
                from data_loaders.humanml.utils.get_opt import get_opt
                opt = get_opt(pjoin(abs_path, 'dataset/kit_opt.txt'), 'cpu')
                mean_for_eval = np.load(pjoin(opt.meta_dir, f'{opt.dataset_name}_mean.npy'))
                std_for_eval = np.load(pjoin(opt.meta_dir, f'{opt.dataset_name}_std.npy'))
            except:
                mean_for_eval = mean
                std_for_eval = std
        else:
            raise ValueError(f"Composite dataset not supported for dataset: {name}")
        
        # 加载词向量化器
        cache_dir = cache_path if cache_path else pjoin(abs_path, '.')
        w_vectorizer = WordVectorizer(pjoin(cache_dir, 'glove'), 'our_vab')
        
        # 创建复合数据集
        dataset = CompositeDataset(
            composite_data_path=composite_data_path,
            mean=mean,
            std=std,
            w_vectorizer=w_vectorizer,
            max_motion_length=196,
            fps=20.0 if name == 'humanml' else 12.5,
            mode=hml_mode,  # 传递模式参数
            mean_for_eval=mean_for_eval,
            std_for_eval=std_for_eval,
        )
        return dataset
    
    # 使用标准数据集
    DATA = get_dataset_class(name)
    if name in ["humanml", "kit"]:
        dataset = DATA(split=split, num_frames=num_frames, mode=hml_mode, abs_path=abs_path, fixed_len=fixed_len, 
                       device=device, autoregressive=autoregressive, disable_random_crop=disable_random_crop)
    else:
        dataset = DATA(split=split, num_frames=num_frames)
    return dataset


def get_dataset_loader(name, batch_size, num_frames, split='train', hml_mode='train', fixed_len=0, pred_len=0, 
                       device=None, autoregressive=False, disable_random_crop=False,
                       use_composite_dataset=False, composite_data_path=None, composite_k_segments=3, cache_path=None, abs_path='.'):
    dataset = get_dataset(name, num_frames, split=split, hml_mode=hml_mode, fixed_len=fixed_len, 
                device=device, autoregressive=autoregressive, disable_random_crop=disable_random_crop,
                use_composite_dataset=use_composite_dataset, composite_data_path=composite_data_path,
                composite_k_segments=composite_k_segments, cache_path=cache_path, abs_path=abs_path)
    
    # 如果使用复合数据集，使用复合数据集的 collate 函数
    if use_composite_dataset and composite_data_path is not None:
        from data_loaders.humanml.data.composite_dataset import composite_collate_fn
        collate = composite_collate_fn
    else:
        collate = get_collate_fn(name, hml_mode, pred_len, batch_size)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=8, drop_last=True, collate_fn=collate
    )

    return loader
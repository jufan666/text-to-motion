"""
训练脚本修改示例：添加 LoRA 验证和数据量限制
"""


import os
import json
from utils.fixseed import fixseed
from utils.parser_util import train_args
from utils import dist_util
from data_loaders.get_data import get_dataset_loader
from train.training_loop import TrainLoop
from utils.model_util import create_model_and_diffusion
from train.train_platforms import WandBPlatform, ClearmlPlatform, TensorboardPlatform, NoPlatform  # required for the eval operation
from peft import PeftModel
from utils.dataset_utils import limit_dataset_size
import torch

def main():
    """
    这是示例代码，展示如何修改 train_mdm.py
    """
    args = train_args()
    fixseed(args.seed)
    train_platform_type = eval(args.train_platform_type)
    train_platform = train_platform_type(args.save_dir)
    train_platform.report_args(args, name='Args')

    if args.save_dir is None:
        raise FileNotFoundError('save_dir was not specified.')
    elif os.path.exists(args.save_dir) and not args.overwrite:
        raise FileExistsError('save_dir [{}] already exists.'.format(args.save_dir))
    elif not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args_path = os.path.join(args.save_dir, 'args.json')
    with open(args_path, 'w') as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)

    dist_util.setup_dist(args.device)

    
    # ... 原有的初始化代码 ...
    
    # ========== 步骤 1: 数据加载并限制大小 ==========
    print("="*60)
    print("加载训练数据...")
    print("="*60)
    
    # 限制训练数据量（如果指定了 max_train_samples）
    print("creating data loader...")
    data = get_dataset_loader(name=args.dataset, 
                              batch_size=args.batch_size, 
                              num_frames=args.num_frames, 
                              fixed_len=args.pred_len + args.context_len, 
                              pred_len=args.pred_len,
                              device=dist_util.dev(),)
    if hasattr(args, 'max_train_samples') and args.max_train_samples > 0:
        print(f"\n[限制数据量] 设置为最多使用 {args.max_train_samples} 个训练样本")
        # 如果 data 是 DataLoader:
        if hasattr(data, 'dataset'):
            data = limit_dataset_size(data, args.max_train_samples)
        # 如果 data 是 Dataset:
        else:
            data = limit_dataset_size(data, args.max_train_samples)
    
    # ========== 步骤 2: 创建模型并验证 LoRA ==========
    print("\n" + "="*60)
    print("创建模型...")
    print("="*60)
    
    model, diffusion = create_model_and_diffusion(args, data)
    model.to(dist_util.dev())
    
    # LoRA 验证代码
    print("\n" + "="*60)
    print("LoRA 挂载验证")
    print("="*60)
    print(f"模型类型: {type(model).__name__}")
    print(f"模型完整类型: {type(model)}")
    
    if isinstance(model, PeftModel):
        print("\n✓ LoRA 已成功挂载到 MDM 上")
        print("\n可训练参数统计:")
        model.print_trainable_parameters()
        
        # 统计 LoRA 层
        lora_layers = [name for name, param in model.named_parameters() if "lora_" in name]
        print(f"\nLoRA 层总数: {len(lora_layers)}")
        
        if len(lora_layers) > 0:
            print("\n前 10 个 LoRA 层示例:")
            for i, layer_name in enumerate(lora_layers[:10]):
                param = dict(model.named_parameters())[layer_name]
                print(f"  {i+1}. {layer_name}: shape={param.shape}, requires_grad={param.requires_grad}")
            
            if len(lora_layers) > 10:
                print(f"  ... 还有 {len(lora_layers) - 10} 个 LoRA 层")
            
            # 按类型分组统计
            attn_lora = [n for n in lora_layers if "out_proj" in n]
            ffn_lora = [n for n in lora_layers if "linear1" in n or "linear2" in n]
            text_lora = [n for n in lora_layers if "embed_text" in n]
            
            print("\nLoRA 层分类统计:")
            print(f"  - Attention (out_proj): {len(attn_lora)} 个")
            print(f"  - FFN (linear1/linear2): {len(ffn_lora)} 个")
            print(f"  - Text Embedding: {len(text_lora)} 个")
        else:
            print("未找到任何 LoRA 层！")
    else:
        print("\n LoRA 未挂载（当前不是 PeftModel）")
        print("提示: 请确保使用了 --use_lora 参数")
        
        # 检查是否真的没有 LoRA
        has_lora_params = any("lora_" in name for name, _ in model.named_parameters())
        if has_lora_params:
            print("模型参数中包含 'lora_' 但类型不是 PeftModel，可能有问题")
        else:
            print("确认: 模型参数中确实没有 LoRA 相关参数")
    
    print("="*60 + "\n")
    
    # ========== 步骤 3: 继续原有的训练流程 ==========
    print("="*60)
    print("开始训练...")
    print("="*60)
    
    # 原有的 TrainLoop 创建和运行代码
    model.rot2xyz.smpl_model.eval()

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters_wo_clip()) / 1000000.0))
    print("Training...")
    TrainLoop(args, train_platform, model, diffusion, data).run_loop()
    train_platform.close()


if __name__ == "__main__":
    main()

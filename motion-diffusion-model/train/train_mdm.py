# This code is based on https://github.com/openai/guided-diffusion
"""
Train a diffusion model on images.
"""

import os
import json
from utils.fixseed import fixseed
from utils.parser_util import train_args
from utils import dist_util
from train.training_loop import TrainLoop
from data_loaders.get_data import get_dataset_loader
from utils.model_util import create_model_and_diffusion
from train.train_platforms import WandBPlatform, ClearmlPlatform, TensorboardPlatform, NoPlatform  # required for the eval operation

import torch
from peft import PeftModel 

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

def main():
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

    print("creating data loader...")

    data = get_dataset_loader(name=args.dataset, 
                              batch_size=args.batch_size, 
                              num_frames=args.num_frames, 
                              fixed_len=args.pred_len + args.context_len, 
                              pred_len=args.pred_len,
                              device=dist_util.dev(),)

    print("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data)
    model.to(dist_util.dev())

    # === LoRA 挂载验证 ===
    print("Model class:", type(model))
    if isinstance(model, PeftModel):
        print("LoRA 已成功挂载到 MDM 上")
        print("\n可训练参数统计:")
        model.print_trainable_parameters()
        
        # 打印 LoRA 层的名称
        print("\nLoRA 层列表:")
        lora_layers = [name for name, param in model.named_parameters() if "lora_" in name]
        for layer_name in lora_layers[:10]:  # 只打印前10个，避免输出太长
            print(f"  - {layer_name}")
        if len(lora_layers) > 10:
            print(f"  ... 还有 {len(lora_layers) - 10} 个 LoRA 层")
    else:
        print("LoRA 未挂载（当前不是 PeftModel）")
        print("提示: 请确保使用了 --use_lora 参数")
    # =====================

    model.rot2xyz.smpl_model.eval()

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters_wo_clip()) / 1000000.0))
    print("Training...")
    TrainLoop(args, train_platform, model, diffusion, data).run_loop()
    train_platform.close()

if __name__ == "__main__":
    main()

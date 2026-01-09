from argparse import ArgumentParser
import argparse
import os
import json
import torch
import platform


def parse_and_load_from_model(parser):
    # args according to the loaded model
    # do not try to specify them from cmd line since they will be overwritten
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    args = parser.parse_args()
    args_to_overwrite = []
    for group_name in ['dataset', 'model', 'diffusion']:
        args_to_overwrite += get_args_per_group_name(parser, args, group_name)

    # load args from model
    if args.model_path != '':  # if not using external results file
        args = load_args_from_model(args, args_to_overwrite)

    if args.cond_mask_prob == 0:
        args.guidance_param = 1
    
    return apply_rules(args)

def load_args_from_model(args, args_to_overwrite):
    model_path = get_model_path_from_args()
    args_path = os.path.join(os.path.dirname(model_path), 'args.json')
    assert os.path.exists(args_path), 'Arguments json file was not found!'
    with open(args_path, 'r') as fr:
        model_args = json.load(fr)

    for a in args_to_overwrite:
        if a in model_args.keys():
            setattr(args, a, model_args[a])

        elif 'cond_mode' in model_args: # backward compitability
            unconstrained = (model_args['cond_mode'] == 'no_cond')
            setattr(args, 'unconstrained', unconstrained)

        else:
            print('Warning: was not able to load [{}], using default value [{}] instead.'.format(a, args.__dict__[a]))
    return args

def apply_rules(args):
    # For prefix completion
    if args.pred_len == 0:
        args.pred_len = args.context_len

    # For target conditioning
    if args.lambda_target_loc > 0.:
        args.multi_target_cond = True
    
    # Auto-detect device if 'auto' is specified
    if hasattr(args, 'device') and args.device == 'auto':
        args.device = auto_detect_device()
    elif hasattr(args, 'device') and isinstance(args.device, str) and args.device.isdigit():
        # Convert string number to int for backward compatibility
        args.device = int(args.device)
    
    return args


def auto_detect_device():
    """
    自动检测最佳可用设备
    
    优先级:
    1. CUDA (如果可用)
    2. MPS (macOS Metal, 如果可用)
    3. CPU (默认)
    
    返回:
        str: 设备字符串 ('cuda:0', 'mps', 'cpu')
    """
    if torch.cuda.is_available():
        return 'cuda:0'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


def get_args_per_group_name(parser, args, group_name):
    for group in parser._action_groups:
        if group.title == group_name:
            group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
            return list(argparse.Namespace(**group_dict).__dict__.keys())
    return ValueError('group_name was not found.')

def get_model_path_from_args():
    try:
        dummy_parser = ArgumentParser()
        dummy_parser.add_argument('--model_path')
        dummy_args, _ = dummy_parser.parse_known_args()
        return dummy_args.model_path
    except:
        raise ValueError('model_path argument must be specified.')


def add_base_options(parser):
    group = parser.add_argument_group('base')
    group.add_argument("--cuda", default=True, type=bool, help="Use cuda device, otherwise use CPU.")
    group.add_argument("--device", default='auto', type=str, 
                       help="Device to use. Options: 'auto' (auto-detect), 'cuda' or 'cuda:0' (CUDA GPU), 'mps' (macOS Metal), 'cpu', or integer (CUDA device ID for backward compatibility).")
    group.add_argument("--seed", default=10, type=int, help="For fixing random seed.")
    group.add_argument("--batch_size", default=64, type=int, help="Batch size during training.")
    group.add_argument("--train_platform_type", default='NoPlatform', choices=['NoPlatform', 'ClearmlPlatform', 'TensorboardPlatform', 'WandBPlatform'], type=str,
                       help="Choose platform to log results. NoPlatform means no logging.")
    group.add_argument("--external_mode", default=False, type=bool, help="For backward cometability, do not change or delete.")


def add_diffusion_options(parser):
    group = parser.add_argument_group('diffusion')
    group.add_argument("--noise_schedule", default='cosine', choices=['linear', 'cosine'], type=str,
                       help="Noise schedule type")
    group.add_argument("--diffusion_steps", default=1000, type=int,
                       help="Number of diffusion steps (denoted T in the paper)")
    group.add_argument("--sigma_small", default=True, type=bool, help="Use smaller sigma values.")


def add_model_options(parser):
    group = parser.add_argument_group('model')
    group.add_argument("--arch", default='trans_enc',
                       choices=['trans_enc', 'trans_dec', 'gru'], type=str,
                       help="Architecture types as reported in the paper.")
    group.add_argument("--text_encoder_type", default='clip',
                       choices=['clip', 'bert'], type=str, help="Text encoder type.")
    group.add_argument("--emb_trans_dec", action='store_true',
                       help="For trans_dec architecture only, if true, will inject condition as a class token"
                            " (in addition to cross-attention).")
    group.add_argument("--layers", default=8, type=int,
                       help="Number of layers.")
    group.add_argument("--latent_dim", default=512, type=int,
                       help="Transformer/GRU width.")
    group.add_argument("--cond_mask_prob", default=.1, type=float,
                       help="The probability of masking the condition during training."
                            " For classifier-free guidance learning.")
    group.add_argument("--mask_frames", action='store_true', help="If true, will fix Rotem's bug and mask invalid frames.")
    group.add_argument("--lambda_rcxyz", default=0.0, type=float, help="Joint positions loss.")
    group.add_argument("--lambda_vel", default=0.0, type=float, help="Joint velocity loss.")
    group.add_argument("--lambda_fc", default=0.0, type=float, help="Foot contact loss.")
    group.add_argument("--lambda_target_loc", default=0.0, type=float, help="For HumanML only, when . L2 with target location.")
    group.add_argument("--unconstrained", action='store_true',
                       help="Model is trained unconditionally. That is, it is constrained by neither text nor action. "
                            "Currently tested on HumanAct12 only.")
    group.add_argument("--pos_embed_max_len", default=5000, type=int,
                       help="Pose embedding max length.")
    group.add_argument("--use_ema", action='store_true',
                    help="If True, will use EMA model averaging.")
    # LoRA 相关参数（用于显存友好的高效微调）
    group.add_argument(
        "--use_lora",
        action="store_true",
        help="如果为 True，则在 MDM 的注意力和文本嵌入层上挂载 LoRA 适配器，仅训练低秩增量参数。",
    )
    group.add_argument(
        "--lora_r",
        default=8,
        type=int,
        help="LoRA rank（低秩维度），典型取值 8~16。",
    )
    group.add_argument(
        "--lora_alpha",
        default=16,
        type=int,
        help="LoRA scaling 系数 α，典型取值 16~32。",
    )
    group.add_argument(
        "--lora_dropout",
        default=0.0,
        type=float,
        help="LoRA dropout 概率。",
    )
    group.add_argument(
        "--lora_target_spec",
        default="all",
        type=str,
        help="LoRA 作用范围，逗号分隔。预设：attn,ffn,text,all；也可直接写模块名。",
    )
    

    group.add_argument("--multi_target_cond", action='store_true', help="If true, enable multi-target conditioning (aka Sigal's model).")
    group.add_argument("--multi_encoder_type", default='single', choices=['single', 'multi', 'split'], type=str, help="Specifies the encoder type to be used for the multi joint condition.")
    group.add_argument("--target_enc_layers", default=1, type=int, help="Num target encoder layers")


    # Prefix completion model
    group.add_argument("--context_len", default=0, type=int, help="If larger than 0, will do prefix completion.")
    group.add_argument("--pred_len", default=0, type=int, help="If context_len larger than 0, will do prefix completion. If pred_len will not be specified - will use the same length as context_len")
    



def add_data_options(parser):
    group = parser.add_argument_group('dataset')
    group.add_argument("--dataset", default='humanml', choices=['humanml', 'kit', 'humanact12', 'uestc'], type=str,
                       help="Dataset name (choose from list).")
    group.add_argument("--data_dir", default="", type=str,
                       help="If empty, will use defaults according to the specified dataset.")
    # 复合数据集选项
    group.add_argument("--use_composite_dataset", action='store_true',
                       help="Use composite dataset (constructed by construct_composite_dataset.py)")
    group.add_argument("--composite_data_path", type=str, default=None,
                       help="Path to composite dataset .npy file (required if --use_composite_dataset)")
    group.add_argument("--composite_k_segments", type=int, default=3, choices=[3, 4, 5],
                       help="Number of segments K in composite dataset (default: 3)")


def add_training_options(parser):
    group = parser.add_argument_group('training')
    group.add_argument("--save_dir", required=True, type=str,
                       help="Path to save checkpoints and results.")
    group.add_argument("--overwrite", action='store_true',
                       help="If True, will enable to use an already existing save_dir.")
    group.add_argument("--lr", default=1e-4, type=float, help="Learning rate.")
    group.add_argument("--weight_decay", default=0.0, type=float, help="Optimizer weight decay.")
    group.add_argument("--lr_anneal_steps", default=0, type=int, help="Number of learning rate anneal steps.")
    group.add_argument("--eval_batch_size", default=32, type=int,
                       help="Batch size during evaluation loop. Do not change this unless you know what you are doing. "
                            "T2m precision calculation is based on fixed batch size 32.")
    group.add_argument("--eval_split", default='test', choices=['val', 'test'], type=str,
                       help="Which split to evaluate on during training.")
    group.add_argument("--eval_during_training", action='store_true',
                       help="If True, will run evaluation during training.")
    group.add_argument("--eval_rep_times", default=3, type=int,
                       help="Number of repetitions for evaluation loop during training.")
    group.add_argument("--eval_num_samples", default=1_000, type=int,
                       help="If -1, will use all samples in the specified split.")
    group.add_argument("--log_interval", default=1_000, type=int,
                       help="Log losses each N steps")
    group.add_argument("--save_interval", default=50_000, type=int,
                       help="Save checkpoints and run evaluation each N steps")
    group.add_argument("--num_steps", default=600_000, type=int,
                       help="Training will stop after the specified number of steps.")
    group.add_argument("--num_frames", default=60, type=int,
                       help="Limit for the maximal number of frames. In HumanML3D and KIT this field is ignored.")
    group.add_argument("--resume_checkpoint", default="", type=str,
                       help="If not empty, will start from the specified checkpoint (path to model###.pt file).")
    group.add_argument("--max_train_samples", default=0, type=int,
                       help="限制训练样本数量（用于快速验证），0 表示不限制。例如设置为 100 用于快速验证 LoRA。")
    
    group.add_argument("--gen_during_training", action='store_true',
                       help="If True, will generate motions during training, on each save interval.")
    group.add_argument("--gen_num_samples", default=3, type=int,
                       help="Number of samples to sample while generating")
    group.add_argument("--gen_num_repetitions", default=2, type=int,
                       help="Number of repetitions, per sample (text prompt/action)")
    group.add_argument("--gen_guidance_param", default=2.5, type=float,
                       help="For classifier-free sampling - specifies the s parameter, as defined in the paper.")
    
    group.add_argument("--avg_model_beta", default=0.9999, type=float, help="Average model beta (for EMA).")
    group.add_argument("--adam_beta2", default=0.999, type=float, help="Adam beta2.")
    
    group.add_argument("--target_joint_names", default='DIMP_FINAL', type=str, help="Force single joint configuration by specifing the joints (coma separated). If None - will use the random mode for all end effectors.")
    group.add_argument("--autoregressive", action='store_true', help="If true, and we use a prefix model will generate motions in an autoregressive loop.")
    group.add_argument("--autoregressive_include_prefix", action='store_true', help="If true, include the init prefix in the output, otherwise, will drop it.")
    group.add_argument("--autoregressive_init", default='data', type=str, choices=['data', 'isaac'], 
                        help="Sets the source of the init frames, either from the dataset or isaac init poses.")


def add_sampling_options(parser):
    group = parser.add_argument_group('sampling')
    group.add_argument("--model_path", required=True, type=str,
                       help="Path to model####.pt file to be sampled.")
    group.add_argument("--output_dir", default='', type=str,
                       help="Path to results dir (auto created by the script). "
                            "If empty, will create dir in parallel to checkpoint.")
    group.add_argument("--num_samples", default=6, type=int,
                       help="Maximal number of prompts to sample, "
                            "if loading dataset from file, this field will be ignored.")
    group.add_argument("--num_repetitions", default=3, type=int,
                       help="Number of repetitions, per sample (text prompt/action)")
    group.add_argument("--guidance_param", default=2.5, type=float,
                       help="For classifier-free sampling - specifies the s parameter, as defined in the paper.")

    group.add_argument("--autoregressive", action='store_true', help="If true, and we use a prefix model will generate motions in an autoregressive loop.")
    group.add_argument("--autoregressive_include_prefix", action='store_true', help="If true, include the init prefix in the output, otherwise, will drop it.")
    group.add_argument("--autoregressive_init", default='data', type=str, choices=['data', 'isaac'], 
                        help="Sets the source of the init frames, either from the dataset or isaac init poses.")

def add_generate_options(parser):
    group = parser.add_argument_group('generate')
    group.add_argument("--motion_length", default=6.0, type=float,
                       help="The length of the sampled motion [in seconds]. "
                            "Maximum is 9.8 for HumanML3D (text-to-motion), and 2.0 for HumanAct12 (action-to-motion)")
    group.add_argument("--input_text", default='', type=str,
                       help="Path to a text file lists text prompts to be synthesized. If empty, will take text prompts from dataset.")
    group.add_argument("--dynamic_text_path", default='', type=str,
                       help="For the autoregressive mode only! Path to a text file lists text prompts to be synthesized. If empty, will take text prompts from dataset.")
    group.add_argument("--action_file", default='', type=str,
                       help="Path to a text file that lists names of actions to be synthesized. Names must be a subset of dataset/uestc/info/action_classes.txt if sampling from uestc, "
                            "or a subset of [warm_up,walk,run,jump,drink,lift_dumbbell,sit,eat,turn steering wheel,phone,boxing,throw] if sampling from humanact12. "
                            "If no file is specified, will take action names from dataset.")
    group.add_argument("--text_prompt", default='', type=str,
                       help="A text prompt to be generated. If empty, will take text prompts from dataset.")
    group.add_argument("--action_name", default='', type=str,
                       help="An action name to be generated. If empty, will take text prompts from dataset.")
    group.add_argument("--target_joint_names", default='DIMP_FINAL', type=str, help="Force single joint configuration by specifing the joints (coma separated). If None - will use the random mode for all end effectors.")


def add_edit_options(parser):
    group = parser.add_argument_group('edit')
    group.add_argument("--edit_mode", default='in_between', choices=['in_between', 'upper_body'], type=str,
                       help="Defines which parts of the input motion will be edited.\n"
                            "(1) in_between - suffix and prefix motion taken from input motion, "
                            "middle motion is generated.\n"
                            "(2) upper_body - lower body joints taken from input motion, "
                            "upper body is generated.")
    group.add_argument("--text_condition", default='', type=str,
                       help="Editing will be conditioned on this text prompt. "
                            "If empty, will perform unconditioned editing.")
    group.add_argument("--prefix_end", default=0.25, type=float,
                       help="For in_between editing - Defines the end of input prefix (ratio from all frames).")
    group.add_argument("--suffix_start", default=0.75, type=float,
                       help="For in_between editing - Defines the start of input suffix (ratio from all frames).")


def add_evaluation_options(parser):
    group = parser.add_argument_group('eval')
    group.add_argument("--model_path", required=True, type=str,
                       help="Path to model####.pt file to be sampled.")
    group.add_argument("--eval_mode", default='wo_mm', choices=['wo_mm', 'mm_short', 'debug', 'full'], type=str,
                       help="wo_mm (t2m only) - 20 repetitions without multi-modality metric; "
                            "mm_short (t2m only) - 5 repetitions with multi-modality metric; "
                            "debug - short run, less accurate results."
                            "full (a2m only) - 20 repetitions.")
    group.add_argument("--autoregressive", action='store_true', help="If true, and we use a prefix model will generate motions in an autoregressive loop.")
    group.add_argument("--autoregressive_include_prefix", action='store_true', help="If true, include the init prefix in the output, otherwise, will drop it.")
    group.add_argument("--autoregressive_init", default='data', type=str, choices=['data', 'isaac'], 
                        help="Sets the source of the init frames, either from the dataset or isaac init poses.")
    group.add_argument("--guidance_param", default=2.5, type=float,
                       help="For classifier-free sampling - specifies the s parameter, as defined in the paper.")
    # 注意：复合数据集选项在 add_data_options 中定义，避免重复


def get_cond_mode(args):
    if args.unconstrained:
        cond_mode = 'no_cond'
    elif args.dataset in ['kit', 'humanml']:
        cond_mode = 'text'
    else:
        cond_mode = 'action'
    return cond_mode


def train_args():
    parser = ArgumentParser()
    add_base_options(parser)
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    add_training_options(parser)
    return apply_rules(parser.parse_args())


def generate_args():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_sampling_options(parser)
    add_generate_options(parser)
    args = parse_and_load_from_model(parser)
    cond_mode = get_cond_mode(args)

    if (args.input_text or args.text_prompt) and cond_mode != 'text':
        raise Exception('Arguments input_text and text_prompt should not be used for an action condition. Please use action_file or action_name.')
    elif (args.action_file or args.action_name) and cond_mode != 'action':
        raise Exception('Arguments action_file and action_name should not be used for a text condition. Please use input_text or text_prompt.')

    return args


def grpo_args():
    """
    解析 GRPO 训练的命令行参数
    
    扩展 train_args() 的参数解析器，添加 GRPO 特定参数。
    用于 Group Relative Policy Optimization 训练。
    """
    parser = ArgumentParser(description='GRPO Training for Motion Diffusion Models')
    add_base_options(parser)
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    add_training_options(parser)
    
    # 添加 GRPO 特定参数
    grpo_group = parser.add_argument_group('GRPO')
    grpo_group.add_argument('--model_path', type=str, required=True,
                            help='Path to pretrained model checkpoint')
    grpo_group.add_argument('--ref_model_path', type=str, default=None,
                            help='Path to reference model (if different from model_path)')
    grpo_group.add_argument('--group_size', type=int, default=4,
                            help='Group size G for GRPO (samples per prompt)')
    grpo_group.add_argument('--learning_rate', type=float, default=5e-7,
                            help='Learning rate for GRPO training')
    grpo_group.add_argument('--clip_epsilon', type=float, default=0.2,
                            help='PPO clipping parameter')
    grpo_group.add_argument('--kl_penalty', type=float, default=1,
                            help='KL divergence penalty weight')
    grpo_group.add_argument('--grpo_type', type=str, default='normal_grpo',
                            choices=['normal_grpo', 'flow_grpo'],
                            help='GRPO trainer type: normal_grpo (standard GRPO) or flow_grpo (Flow-based GRPO with SDE)')
    grpo_group.add_argument('--reward_model_type', type=str, default='mdm',
                            choices=['mdm', 'tmr'],
                            help='Reward model type: mdm (MDM evaluator) or tmr (TMR pretrained model)')
    grpo_group.add_argument('--reward_type', type=str, default='matching',
                            choices=['matching', 'r_precision', 'combined', 'cosine'],
                            help='Reward function type: matching (Matching Score), r_precision (R-Precision), combined (both), or cosine (for TMR)')
    grpo_group.add_argument('--tmr_text_encoder_path', type=str, default=None,
                            help='Path to TMR text encoder checkpoint (text_encoder.pt, required if reward_model_type=tmr)')
    grpo_group.add_argument('--tmr_motion_encoder_path', type=str, default=None,
                            help='Path to TMR motion encoder checkpoint (motion_encoder.pt, required if reward_model_type=tmr)')
    grpo_group.add_argument('--tmr_movement_encoder_path', type=str, default=None,
                            help='Path to TMR movement encoder/decoder checkpoint (motion_decoder.pt, required if reward_model_type=tmr)')
    grpo_group.add_argument('--tmr_similarity_type', type=str, default='cosine',
                            choices=['cosine', 'euclidean'],
                            help='Similarity type for TMR reward (cosine or euclidean)')
    grpo_group.add_argument('--tmr_normalization', type=str, default='linear',
                            choices=['linear', 'exponential', 'sigmoid'],
                            help='Normalization method for TMR reward (linear, exponential, or sigmoid)')
    grpo_group.add_argument('--tmr_max_distance', type=float, default=10.0,
                            help='Max distance for TMR euclidean distance normalization')
    grpo_group.add_argument('--tmr_scale', type=float, default=2.0,
                            help='Scale factor for TMR exponential/sigmoid normalization')
    
    # 奖励函数高级参数（适用于 MDM 和 TMR 奖励模型）
    grpo_group.add_argument('--use_dense_reward', action='store_true',
                            help='Enable Segment-Dense reward mode. If False, use Global reward mode (default: False)')
    grpo_group.add_argument('--use_physics_reward', action='store_true',
                            help='Enable physics regularization reward (default: False)')
    grpo_group.add_argument('--k_segments', type=int, default=1,
                            help='Number of text segments for validation or default processing (default: 1)')
    grpo_group.add_argument('--max_motion_length', type=int, default=196,
                            help='Maximum motion length in frames. Motions exceeding this will be truncated (default: 196)')
    grpo_group.add_argument('--alpha', type=float, default=0.5,
                            help='Negative reward penalty weight for Segment-Dense mode (default: 0.5)')
    grpo_group.add_argument('--beta_s', type=float, default=1.0,
                            help='Semantic reward weight (default: 1.0)')
    grpo_group.add_argument('--beta_p', type=float, default=0.1,
                            help='Physics reward weight (default: 0.1)')
    grpo_group.add_argument('--lambda_skate', type=float, default=1.0,
                            help='Skating penalty weight for physics reward (default: 1.0)')
    grpo_group.add_argument('--lambda_jerk', type=float, default=1.0,
                            help='Jerk penalty weight for physics reward (default: 1.0)')
    grpo_group.add_argument('--fps', type=float, default=20.0,
                            help='Dataset frame rate (frames per second). HumanML=20.0, KIT=12.5 (default: 20.0)')
    grpo_group.add_argument('--disable_random_crop', action='store_true',
                            help='Disable random crop and offset augmentation in dataset loading. '
                                 'Required when using fixed durations for composite prompts.')
    
    # Flow-GRPO 特定参数（仅当 --grpo_type=flow_grpo 时使用）
    grpo_group.add_argument('--noise_scale', type=float, default=0.7,
                            help='SDE noise scale coefficient a for Flow-GRPO (default: 0.7)')
    grpo_group.add_argument('--train_timesteps', type=int, default=10,
                            help='Number of inference steps during training for Flow-GRPO (default: 10)')
    grpo_group.add_argument('--inference_timesteps', type=int, default=40,
                            help='Number of inference steps during inference for Flow-GRPO (default: 40)')
    
    # 解析参数并应用规则
    args = apply_rules(parser.parse_args())
    
    return args


def edit_args():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_sampling_options(parser)
    add_edit_options(parser)
    return parse_and_load_from_model(parser)


def evaluation_parser():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_evaluation_options(parser)
    return parse_and_load_from_model(parser)
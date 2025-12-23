import logging
from typing import Iterable, Optional, Sequence

from peft import LoraConfig, get_peft_model

LOGGER = logging.getLogger(__name__)


def _resolve_target_modules(target_modules: Optional[Iterable[str]], target_spec: Optional[str]) -> Sequence[str]:
    """
    根据用户指定的 target_modules 或 target_spec（逗号分隔的 preset）生成最终 LoRA 作用的模块名列表。
    
    预设关键词（peft 使用模糊匹配，会匹配所有包含该字符串的模块名）：
        - attn:   所有 attention 的 out_proj（包括 self-attn 和 cross-attn）
                  * trans_enc: seqTransEncoder.layers.*.self_attn.out_proj
                  * trans_dec: seqTransDecoder.layers.*.self_attn.out_proj (self-attn)
                              seqTransDecoder.layers.*.multihead_attn.out_proj (cross-attn)
        
        - ffn:    所有 FFN 的 linear1 / linear2
                  * trans_enc: seqTransEncoder.layers.*.linear1/linear2
                  * trans_dec: seqTransDecoder.layers.*.linear1/linear2
        
        - text:   文本条件的 embed_text 线性层
                  * 位置: embed_text (MDM 类下，将 CLIP/BERT 编码投影到 latent_dim)
        
        - all:    attn + ffn + text（所有上述模块）
    
    也可以直接传入具体模块名（如 out_proj, linear1），peft 会进行模糊匹配。
    """
    if target_modules is not None:
        return list(target_modules)

    # 默认为 all，兼顾时序 self-attn / cross-attn 以及文本投影
    if not target_spec:
        target_spec = "all"

    # PyTorch Transformer 模块命名说明：
    # - TransformerEncoderLayer: self_attn.out_proj, linear1, linear2
    # - TransformerDecoderLayer: self_attn.out_proj, multihead_attn.out_proj, linear1, linear2
    # - MDM 中: seqTransEncoder.layers.*.self_attn.out_proj 或 seqTransDecoder.layers.*.self_attn.out_proj / multihead_attn.out_proj
    # - MDM 中: seqTransEncoder.layers.*.linear1/linear2 或 seqTransDecoder.layers.*.linear1/linear2
    # - MDM 中: embed_text (直接在 MDM 类下，用于文本条件投影)
    # peft 使用模糊匹配，所以 "out_proj" 会匹配所有包含该字符串的模块名
    preset_map = {
        "attn": ["out_proj"],               # 匹配所有 attention 的 out_proj（包括 self-attn 和 cross-attn）
        # 位置: seqTransEncoder.layers.*.self_attn.out_proj (trans_enc)
        #       seqTransDecoder.layers.*.self_attn.out_proj (trans_dec, self-attn)
        #       seqTransDecoder.layers.*.multihead_attn.out_proj (trans_dec, cross-attn)
        
        "ffn": ["linear1", "linear2"],      # 匹配所有 FFN 的线性层
        # 位置: seqTransEncoder.layers.*.linear1/linear2 (trans_enc)
        #       seqTransDecoder.layers.*.linear1/linear2 (trans_dec)
        
        "text": ["embed_text"],             # 文本条件的线性投影层
        # 位置: embed_text (MDM 类下，将 CLIP/BERT 编码后的文本特征投影到 latent_dim)
        
        "all": ["out_proj", "linear1", "linear2", "embed_text"],  # 上述所有模块
    }

    targets = []
    for tok in target_spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok in preset_map:
            targets.extend(preset_map[tok] if tok != "all" else ["out_proj", "linear1", "linear2", "embed_text"])
        else:
            # 允许用户直接写模块名
            targets.append(tok)

    # 去重保持顺序
    seen = set()
    final = []
    for t in targets:
        if t not in seen:
            final.append(t)
            seen.add(t)
    return final


def add_lora_to_mdm(
    model,
    r: int = 128,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
    target_modules: Optional[Iterable[str]] = None,
    target_spec: Optional[str] = None,
):
    """
    在 MDM 上挂载 LoRA 适配器。
    """
    resolved_targets = _resolve_target_modules(target_modules, target_spec)

    # -----------------------------------------------------------
    # 手动给自定义模型添加一个 config 属性
    # PEFT 库需要通过 model.config 来判断一些模型属性，自定义模型通常没有。
    # 这里我们创建一个简单的字典或对象即可。
    # -----------------------------------------------------------
    if not hasattr(model, "config"):
        model.config = {"model_type": "custom_mdm"}

    # -----------------------------------------------------------
    # 移除 task_type="SEQ_2_SEQ_LM"
    # 对于非 HF Transformer 的自定义模型（如扩散模型），不要指定 task_type。
    # 指定 task_type 会导致 PEFT 尝试封装特定的 forward 逻辑，这与 MDM 的输入不兼容。
    # 不指定 task_type 时，PEFT 会默认为通用的 "MODEL" 模式，只注入权重而不改变 forward 行为。
    # -----------------------------------------------------------
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=list(resolved_targets),
        # task_type="SEQ_2_SEQ_LM",  <-- 删除这一行，或者显式设为 None
    )

    LOGGER.info(
        "Applying LoRA to MDM: r=%d, alpha=%d, dropout=%.3f, targets=%s",
        r,
        lora_alpha,
        lora_dropout,
        ",".join(resolved_targets),
    )

    model = get_peft_model(model, lora_config)
    
    # 【可选优化】打印可训练参数量，确认 LoRA 挂载成功
    # model.print_trainable_parameters() 
    
    return model
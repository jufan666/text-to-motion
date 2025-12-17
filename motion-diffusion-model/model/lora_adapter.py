import logging
from typing import Iterable, Optional, Sequence

from peft import LoraConfig, get_peft_model

LOGGER = logging.getLogger(__name__)


def _resolve_target_modules(target_modules: Optional[Iterable[str]], target_spec: Optional[str]) -> Sequence[str]:
    """
    根据用户指定的 target_modules 或 target_spec（逗号分隔的 preset）生成最终 LoRA 作用的模块名列表。
    预设关键词：
        - attn:   MultiheadAttention 的输出投影 out_proj
        - ffn:    Transformer FFN 的 linear1 / linear2
        - text:   文本条件的 embed_text 线性层
        - all:    attn + ffn + text
    也可以直接传入具体模块名（如 out_proj, linear1）。
    """
    if target_modules is not None:
        return list(target_modules)

    # 默认为 all，兼顾时序 self-attn / cross-attn 以及文本投影
    if not target_spec:
        target_spec = "all"

    preset_map = {
        "attn": ["out_proj"],               # attention 输出层（同时覆盖 self / cross）
        "ffn": ["linear1", "linear2"],      # FFN 两个线性层
        "text": ["embed_text"],             # 文本投影
        "all": ["out_proj", "linear1", "linear2", "embed_text"],
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
    在 MDM 上挂载 LoRA 适配器（用户可选择作用范围）。

    参数:
        model: 已构建好的 MDM 模型实例。
        r: LoRA rank（低秩维度），建议 128~256。
        lora_alpha: LoRA scaling 系数，建议 16~32。
        lora_dropout: LoRA dropout。
        target_modules: 直接给出模块名列表（模糊匹配）；若为空，则根据 target_spec 生成。
        target_spec: 逗号分隔的 preset/模块名，支持 attn, ffn, text, all。
    """

    resolved_targets = _resolve_target_modules(target_modules, target_spec)

    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=list(resolved_targets),
        task_type="SEQ_2_SEQ_LM",  # 序列到序列条件生成，最接近当前任务
    )

    LOGGER.info(
        "Applying LoRA to MDM: r=%d, alpha=%d, dropout=%.3f, targets=%s",
        r,
        lora_alpha,
        lora_dropout,
        ",".join(resolved_targets),
    )

    model = get_peft_model(model, lora_config)
    return model



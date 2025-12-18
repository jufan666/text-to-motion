- 只在注意力输出投影上挂 LoRA（包含 self / cross attention）：

  ```
  python -m train.train_mdm ... --use_lora --lora_target_spec attn
  ```

- 只在文本投影上挂：

  ```
  python -m train.train_mdm ... --use_lora --lora_target_spec text
  ```

- 在transformer FFN和注意力输出投影上面挂

  ```
  python -m train.train_mdm ... --use_lora --lora_target_spec attn,ffn
  ```

- 


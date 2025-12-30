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





- rl训练 50步扩散mdm（750000）不带lora

  ```
  python -m train.train_grpo     --model_path ./save/official_humanml_enc_512_50steps/model000750000.pt     --save_dir ./save/grpo_finetuned_humanml_enc_512_50steps_750000_2     --dataset humanml     --batch_size 1 --group_size 2 --num_steps 10000 --learning_rate 5e-7 --kl_penalty 1.0 --clip_epsilon 0.1 --use_lora --lora_r 4 --lora_alpha 8 --reward_type matching --device 1  
  ```

  

- 带lora（）

  ```python
  # 将 batch_size 从 2 改为 1
  python -m train.train_grpo \
      --model_path ./save/official_humanml_enc_512_50steps/model000750000.pt \
      --save_dir ./save/grpo_finetuned_humanml_enc_512_50steps_750000 \
      --dataset humanml \
      --batch_size 1 \  
      --group_size 4 \
      --num_steps 10000 \
      --learning_rate 1e-5 \
      --reward_type matching \
  		--device 2
  ```

  

#! /bin/bash


export CUDA_VISIBLE_DEVICES=0,1

MODEL_SIZE=3.1B_it  # or 1.1B
MODELNAME=gemma
DSNAME=base
epochs=5
model=${DSNAME}_${MODELNAME}_${MODEL_SIZE}_ep${epochs}

python -m eval.val_eval.run_eval \
     --model_name_or_path /scratch/network/cn1182/hf_cache/hub/models--unsloth--gemma-3-1b-it/snapshots/72055c9ab7b8b8d950101d4ee62add8ab9c66ca6 \
     --tokenizer_name /scratch/network/cn1182/hf_cache/hub/models--unsloth--gemma-3-1b-it/snapshots/72055c9ab7b8b8d950101d4ee62add8ab9c66ca6 \
     --save_dir results/val_eval/${model}/ \
     --eval_batch_size 10 \
     --use_chat_format \
     --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format \
     --use_vllm

# alpaca_eval \
#     --model_outputs results/val_eval/${model}/${model}-greedy-long-output.json \
#     --reference_outputs eval/val_eval/val-gpt3.5-2.json


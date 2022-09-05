#!/bin/bash
set -x

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TORCH_EXTENSIONS_DIR="/scratch/acd14245px/torch_extension"
export TRANSFORMERS_CACHE=/scratch/acd14245px/huggingface_models

port=$(shuf -i25000-30000 -n1)

deepspeed --master_port $port examples/pytorch/summarization/run_summarization.py --deepspeed ds_config.json --model_name_or_path t5-3b --do_train --do_eval --dataset_name cnn_dailymail --dataset_config "3.0.0" --source_prefix "summarize: " --output_dir /scratch/acd14245px/huggingface_outputs --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --overwrite_output_dir --predict_with_generate --cache_dir /scratch/acd14245px/huggingface_models


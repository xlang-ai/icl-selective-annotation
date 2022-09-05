import os

if not os.path.isdir('/scratch/acd14245px/sst5_100'):
    os.makedirs('/scratch/acd14245px/sst5_100',exist_ok=True)

for seed in range(300):
    os.system(f"python run_glue.py --model_name_or_path roberta-large --task_name sst5 --do_train --do_eval --seed {seed} "
              f"--max_seq_length 128 --per_device_train_batch_size 16 --learning_rate 2e-5 --num_train_epochs 3 "
              f"--output_dir /scratch/acd14245px/huggingface_sst5 --cache_dir /scratch/acd14245px/huggingface_models "
              f"--selection_method random --annotation_size 100 --overwrite_output_dir > /scratch/acd14245px/sst5_100/{seed}.txt")

import os

if not os.path.isdir('/scratch/acd14245px/hella_2800'):
    os.makedirs('/scratch/acd14245px/hella_2800',exist_ok=True)

for seed in range(300):
    os.system(f"python run_hellaswag.py --model_name_or_path roberta-large --do_train --do_eval --learning_rate 5e-7 "
              f"--num_train_epochs 100 --output_dir /scratch/acd14245px/HellaSwag_roberta_large1 --seed {seed} "
              f"--per_gpu_eval_batch_size=16 --per_device_train_batch_size=16 --selection_method random "
              f"--annotation_size 2800 --overwrite_output --cache_dir /scratch/acd14245px/huggingface_models "
              f"> /scratch/acd14245px/hella_2800/{seed}.txt")

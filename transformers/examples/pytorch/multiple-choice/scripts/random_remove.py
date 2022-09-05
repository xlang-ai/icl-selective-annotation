import os
import shutil

if not os.path.isdir('outputs'):
    os.makedirs('outputs',exist_ok=True)

for seed in [2,4,12,24,32,36,38,42,56,58]:
    os.system(f"python run_hellaswag.py --model_name_or_path roberta-base --do_train --do_eval --learning_rate 2e-5 "
              f"--num_train_epochs 6 --output_dir /nvme/.cache/swag_base_random --per_gpu_eval_batch_size=16 "
              f"--per_device_train_batch_size=2 --overwrite_output --selection_method random "
              f"--train_emb_dir /nvme/.cache/embeddings --remove_some "
              f"--annotation_size 100 --sentence_transformer_model sentence-transformers/all-mpnet-base-v2 "
              f"--seed {seed} > outputs/random_remove_{seed}.txt")
    try:
        if os.path.isdir(f'/nvme/.cache/swag_base_random'):
            shutil.rmtree(f'/nvme/.cache/swag_base_random')
    except:
        pass
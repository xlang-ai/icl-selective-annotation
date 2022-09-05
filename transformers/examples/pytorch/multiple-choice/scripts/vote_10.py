import os
import shutil

if not os.path.isdir('outputs'):
    os.makedirs('outputs',exist_ok=True)

for seed in [2,4,12,24,32,36,38,42,56,58]:
    os.system(f"python run_hellaswag.py --model_name_or_path roberta-base --do_train --do_eval --learning_rate 2e-5 "
              f"--num_train_epochs 6 --output_dir /nvme/.cache/swag_base_vote_10_select --per_gpu_eval_batch_size=16 "
              f"--per_device_train_batch_size=2 --overwrite_output --selection_method vote_10_select "
              f"--train_emb_dir /nvme/.cache/embeddings "
              f"--annotation_size 100 --sentence_transformer_model sentence-transformers/all-mpnet-base-v2 "
              f"--seed {seed} > outputs/vote_10_select_{seed}.txt")
    try:
        if os.path.isdir(f'/nvme/.cache/swag_base_vote_10_select'):
            shutil.rmtree(f'/nvme/.cache/swag_base_vote_10_select')
    except:
        pass
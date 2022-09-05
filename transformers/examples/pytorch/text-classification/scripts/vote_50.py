import os
import shutil

if not os.path.isdir('outputs'):
    os.makedirs('outputs',exist_ok=True)

for seed in [2,4,12,24,32,36,38,42,56,58]:
    os.system(f"python run_glue.py --model_name_or_path roberta-base --task_name mrpc --do_train --do_eval "
              f"--max_seq_length 128 --per_device_train_batch_size 2 --learning_rate 2e-5 --num_train_epochs 4 "
              f"--output_dir /nvme/.cache/outputs/mrpc_vote_50_select --annotation_size 100 --seed {seed} "
              f"--train_emb_dir /nvme/.cache/embeddings "
              f"--sentence_transformer_model sentence-transformers/all-mpnet-base-v2 "
              f"--selection_method vote_50_select --overwrite_output > outputs/vote_50_select_{seed}.txt")
    try:
        if os.path.isdir(f'/nvme/.cache/outputs/mrpc_vote_50_select'):
            shutil.rmtree(f'/nvme/.cache/outputs/mrpc_vote_50_select')
    except:
        pass
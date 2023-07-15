import argparse
import random
import os
# import copy
import torch
import numpy as np
import json
import nltk
import time
# import time
# from torch import nn
from tqdm import tqdm
from datasets import load_metric
from transformers import AutoTokenizer,GPTJForCausalLM
# from sentence_transformers import SentenceTransformer
# from datasets import load_dataset
# from sklearn.metrics import f1_score
from MetaICL.metaicl.data import MetaICLData
from MetaICL.metaicl.model import MetaICLModel
# from collections import defaultdict
from get_task import get_task
from utils import calculate_sentence_transformer_embedding,codex_execution,expand_to_aliases
from two_steps import selective_annotation,prompt_retrieval

parser = argparse.ArgumentParser()
parser.add_argument('--task_name', required=True,type=str)
parser.add_argument('--selective_annotation_method', required=True,type=str)
parser.add_argument('--model_cache_dir', required=True,type=str)
parser.add_argument('--data_cache_dir', required=True,type=str)
parser.add_argument('--output_dir', required=True,type=str)
parser.add_argument('--model_key', type=str)
parser.add_argument('--prompt_retrieval_method', default='similar',type=str)
parser.add_argument('--model_name', default='EleutherAI/gpt-j-6B',type=str)
parser.add_argument('--embedding_model', default='sentence-transformers/paraphrase-mpnet-base-v2',type=str)
parser.add_argument('--annotation_size', default=100,type=int)
parser.add_argument('--seed', default=0,type=int)
parser.add_argument('--batch_size', default=10,type=int)
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
    return preds, labels

if __name__=='__main__':
    set_seed(args.seed)
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir,exist_ok=True)
    train_examples,eval_examples,train_text_to_encode,eval_text_to_encode,format_example,label_map = get_task(args=args)
    total_train_embeds = calculate_sentence_transformer_embedding(text_to_encode=train_text_to_encode,
                                                                  args=args)
    total_eval_embeds = calculate_sentence_transformer_embedding(text_to_encode=eval_text_to_encode,
                                                                  args=args)

    if args.task_name in ['mnli','rte','sst5','mrpc','dbpedia_14','hellaswag','xsum','nq']:
        if args.task_name=='xsum':
            tokenizer_gpt = AutoTokenizer.from_pretrained(args.model_name,cache_dir=args.model_cache_dir)
            inference_model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B",cache_dir=args.model_cache_dir)
            inference_model.cuda()
            inference_model.eval()
            data_module = None
            return_string = True
            device = torch.device('cuda')
            single_input_len = None
            maximum_input_len = 1900
        elif args.task_name=='nq':
            maximum_input_len = 3800
            return_string = True
            single_input_len = None
            inference_model = None
            data_module = None
            tokenizer_gpt = None
            model_keys = args.model_key.split('##')
        else:
            data_module = MetaICLData(method="direct", max_length=1024, max_length_per_example=256)
            inference_model = MetaICLModel(args=args)
            inference_model.load()
            inference_model.cuda()
            inference_model.eval()
            tokenizer_gpt = None
            return_string = False
            single_input_len = 250
            maximum_input_len = 1000


        if os.path.isfile(os.path.join(args.output_dir,'first_phase_selected_indices.json')):
            with open(os.path.join(args.output_dir,'first_phase_selected_indices.json')) as f:
                first_phase_selected_indices = json.load(f)
        else:
            first_phase_selected_indices = selective_annotation(embeddings=total_train_embeds,
                                                                train_examples=train_examples,
                                                                return_string=return_string,
                                                                format_example=format_example,
                                                                maximum_input_len=maximum_input_len,
                                                                label_map=label_map,
                                                                single_context_example_len=single_input_len,
                                                                inference_model=inference_model,
                                                                inference_data_module=data_module,
                                                                tokenizer_gpt=tokenizer_gpt,
                                                                args=args)
            with open(os.path.join(args.output_dir,'first_phase_selected_indices.json'),'w') as f:
                json.dump(first_phase_selected_indices,f,indent=4)
        processed_train_examples = [train_examples[idx] for idx in first_phase_selected_indices]
        processed_eval_examples = eval_examples

        prompt_retrieval(train_embs=total_train_embeds[first_phase_selected_indices],test_embs=total_eval_embeds,train_examples=processed_train_examples,
                         eval_examples=eval_examples,return_string=return_string,format_example=format_example,
                         maximum_input_len=maximum_input_len,single_context_example_len=single_input_len,label_map=label_map,args=args)

        prompt_cache_dir = os.path.join(args.output_dir, 'prompts')
        candidate_prompt_files = os.listdir(prompt_cache_dir)
        prompt_files = [f for f in candidate_prompt_files if f.endswith('.json')]
        assert len(prompt_files) == len(processed_eval_examples), f"len(prompt_files)={len(prompt_files)}," \
                                                                  f"len(processed_eval_examples)={len(processed_eval_examples)}"
        output_dir = os.path.join(args.output_dir,'results')
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        count = 0
        running_flag = True
        golds = []
        preds = []
        if not args.task_name in ['hellaswag','xsum','nq']:
            all_labels = []
            label_to_digit = {}
            for k, v in label_map.items():
                all_labels.append(v)
                label_to_digit[v] = k
        execution_count = 0
        while running_flag:
            running_flag = False
            count += 1
            bar = tqdm(range(len(prompt_files)), desc=f"  LLM inference")
            for file in prompt_files:
                bar.update(1)
                if not os.path.isfile(os.path.join(output_dir,file)):
                    running_flag = True

                    if args.task_name == 'hellaswag':
                        with open(os.path.join(prompt_cache_dir, file)) as f:
                            one_test_example = json.load(f)
                        cur_train_data = one_test_example[1]
                        cur_input = {'input': format_example(one_test_example[2], label_map=label_map, args=args)[0],
                                     'options': one_test_example[2]['endings']}
                        data_module.k = len(cur_train_data)
                        data_module.tensorize(cur_train_data, [cur_input])
                        prediction = inference_model.do_predict(data_module)[0]
                        assert prediction in one_test_example[2]['endings']
                        with open(f"{output_dir}/{file}", 'w') as f:
                            json.dump([prediction, one_test_example[2]['endings'][one_test_example[2]['label']]], f)
                        preds.append(prediction)
                        golds.append(one_test_example[2]['endings'][one_test_example[2]['label']])
                    elif args.task_name == 'xsum':
                        with open(os.path.join(prompt_cache_dir, file)) as f:
                            one_test_example = json.load(f)
                        context = one_test_example[1]
                        input_ids = tokenizer_gpt(context, return_tensors="pt").input_ids
                        input_ids = input_ids[:, :1900]
                        input_len = input_ids.shape[1]
                        input_ids = input_ids.to(device)
                        gen_tokens = inference_model.generate(input_ids, do_sample=False, temperature=0.7,
                                                              max_length=input_len + 64,
                                                              output_scores=True, return_dict_in_generate=True)
                        generated_text = tokenizer_gpt.batch_decode(gen_tokens.sequences.view(-1, 1))
                        stop = ['--', '\n', ';', '#']
                        stop_index = len(generated_text)
                        for i, c in enumerate(generated_text):
                            if i > input_len and c.strip(' ') in stop:
                                stop_index = i
                                break
                        prediction = ' '.join(generated_text[input_len:stop_index])
                        golds.append(one_test_example[2]['summary'])
                        preds.append(prediction)
                        with open(f"{output_dir}/{file}", 'w') as f:
                            json.dump(
                                [' '.join(generated_text[input_len:]), ' '.join(generated_text[input_len:stop_index]),
                                 one_test_example[2]['summary'], input_len, stop_index], f, indent=4)
                    elif args.task_name == 'nq':
                        cur_key = model_keys[execution_count % len(model_keys)]
                        execution_count += 1
                        try:
                            codex_execution(key=cur_key, output_path=os.path.join(output_dir, file),
                                            prompt_path=os.path.join(prompt_cache_dir, file))
                        except Exception as e:
                            print(e)
                            time.sleep(3)
                    else:
                        with open(os.path.join(prompt_cache_dir, file)) as f:
                            one_test_example = json.load(f)
                        cur_train_data = one_test_example[1]
                        for idx in range(len(cur_train_data)):
                            cur_train_data[idx]['options'] = all_labels
                        for idx in range(len(cur_train_data)):
                            cur_train_data[idx]['options'] = all_labels
                        cur_input = format_example(one_test_example[2], label_map=label_map, args=args)[0]
                        data_module.k = len(cur_train_data)
                        data_module.tensorize(cur_train_data, [cur_input], options=all_labels)
                        prediction = inference_model.do_predict(data_module)[0]
                        with open(os.path.join(output_dir, file), 'w') as f:
                            json.dump([prediction, one_test_example[2]['label']], f)
                        preds.append(label_to_digit[prediction])
                        golds.append(one_test_example[2]['label'])
        if args.task_name=='xsum':
            assert len(golds) == len(preds), f"len(golds)={len(golds)}, len(preds)={len(preds)}"
            preds, golds = postprocess_text(preds, golds)
            metric = load_metric("rouge")
            result = metric.compute(predictions=preds, references=golds, use_stemmer=True)
            result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
            result = {k: round(v, 4) for k, v in result.items()}
            with open(os.path.join(args.output_dir,'result_summary.json'), 'w') as f:
                json.dump(result, f)
            print(result)
        elif args.task_name=='nq':
            correct = 0
            total = 0
            for file in prompt_files:
                with open(os.path.join(prompt_cache_dir, file)) as f:
                    one_test_example = json.load(f)
                answers = expand_to_aliases(one_test_example[2]["long"] + one_test_example[2]["short_targets"],
                                            make_sub_answers=True)
                with open(os.path.join(output_dir, file)) as f:
                    pred_dict = json.load(f)
                prediction = pred_dict['choices'][0]['text'].replace('\n', ' ')
                prediction = ' '.join(prediction.split(' ')[1:])
                predictions = expand_to_aliases([prediction])
                if len(list(answers & predictions)) > 0:
                    correct += 1
                total += 1
            with open(os.path.join(args.output_dir,'result_summary.txt'), 'w') as f:
                f.write(f"{total} examples, accuracy is: {correct / total}\n")
            print(f"{total} examples, accuracy is: {correct / total}\n")
        else:
            assert len(golds) == len(preds), f"len(golds)={len(golds)}, len(preds)={len(preds)}"
            total = len(golds)
            correct = 0
            for p, g in zip(golds, preds):
                if p == g:
                    correct += 1
            with open(os.path.join(args.output_dir,'result_summary.txt'), 'w') as f:
                f.write(f"{len(golds)} examples, accuracy is: {correct / total}\n")
            print(f'The accuracy is {correct / total}\n')

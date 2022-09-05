#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import numpy as np
from datasets import load_dataset, load_metric

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.18.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

logger = logging.getLogger(__name__)

def lexical_overlap(premise,hypothesis):
    prem_words = []
    hyp_words = []
    for word in premise.split():
        if word not in [".", "?", "!"]:
            prem_words.append(word.lower())
    for word in hypothesis.split():
        if word not in [".", "?", "!"]:
            hyp_words.append(word.lower())
    all_in = True
    for word in hyp_words:
        if word not in prem_words:
            all_in = False
            break
    return all_in

def check_common_t_form_in_sentence(s):
    s = s.lower()
    return 'don\'t' in s or 'didn\'t' in s or 'doesn\'t' in s or 'do n\'t' in s or 'did n\'t' in s or \
           'can\'t' in s or 'couldn\'t' in s or 'cudn\'t' in s or \
           'won\'t' in s or 'wouldn\'t' in s or \
           'isn\'t' in s or 'weren\'t' in s or 'wasn\'t' in s or 'aren\'t' in s or 'warn\'t' in s or 'ain\'t' in s or \
           'hasn\'t' in s or 'haven\'t' in s or 'hadn\'t' in s or \
           'shouldn\'t' in s or 'shan\'t' in s or \
           'needn\'t' in s or \
           'mayn\'t' in s or 'mightn\'t' in s or \
           'mustn\'t' in s or \
           'oughtn\'t' in s

def negation_present(s):
    s = s.lower()
    return 'not' in s or check_common_t_form_in_sentence(s)

def spurious_correlation_type_not_present(e):
    if (negation_present(e['premise']) and not negation_present(e['hypothesis'])) or \
        (not negation_present(e['premise']) and negation_present(e['hypothesis'])):
        if e['label']!=0:
            return 'support'
        return 'not support'
    return 'unknown'

def calculate_overlap_ratio(s1,s2):
    from collections import Counter
    import nltk
    c1 = Counter(nltk.word_tokenize(s1.lower()))
    c2 = Counter(nltk.word_tokenize(s2.lower()))
    total = sum(c1.values()) + sum(c2.values())
    cur = 0
    for k, v in c1.items():
        if k in c2:
            cur += v + c2[k]
    return cur/total

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    my_task_name: Optional[str] = field(
        default=None,
        metadata={"help": "custom defined task name"},
    )
    selected_indices_store_path: Optional[str] = field(
        default=None,
        metadata={"help": "selected_indices_store_path"},
    )
    cur_mode: Optional[str] = field(
        default=None,
        metadata={"help": "current mode"},
    )
    cur_mode_1: Optional[str] = field(
        default=None,
        metadata={"help": "current mode 1"},
    )
    confidence_prediction: bool = field(
        default=False, metadata={"help": "confidence_prediction"}
    )
    check_token_coverage: bool = field(
        default=False, metadata={"help": "check_token_coverage"}
    )
    index_map_store_path: Optional[str] = field(
        default=None,
        metadata={"help": "acquisition_function"},
    )
    calculate_dissimilar_embeds: bool = field(
        default=False, metadata={"help": "calculate_dissimilar_embeds"}
    )
    acquisition_function: Optional[str] = field(
        default=None,
        metadata={"help": "acquisition_function"},
    )
    select_num: int = field(
        default=-1,
        metadata={"help": "selection number."
        },
    )
    self_dissimilar_cap: int = field(
        default=-1,
        metadata={"help": "self_dissimilar_cap"
        },
    )
    seed_set_size: int = field(
        default=-1,
        metadata={"help": "seed_set_size."
        },
    )
    acquisition_batch_size: int = field(
        default=-1,
        metadata={"help": "acquisition_batch_size."
        },
    )
    tag: Optional[str] = field(
        default=None,
        metadata={"help": "tag"},
    )

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.dataset_name is not None:
            pass
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task, a training/validation file or a dataset name.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if data_args.selected_indices_store_path and not os.path.isdir(data_args.selected_indices_store_path):
        os.makedirs(data_args.selected_indices_store_path,exist_ok=True)
    elif not data_args.selected_indices_store_path:
        print("data_args.selected_indices_store_path is ",data_args.selected_indices_store_path)

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.
    #
    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.my_task_name in ['hans']:
        raw_datasets = load_dataset(data_args.my_task_name, cache_dir=model_args.cache_dir)
    elif data_args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset("glue", data_args.task_name, cache_dir=model_args.cache_dir)
    elif data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.cache_dir
        )
    else:
        # Loading a dataset from your local files.
        # CSV/JSON training and evaluation files are needed.
        data_files = {"train": data_args.train_file, "validation": data_args.validation_file}

        # Get the test dataset: you can provide your own CSV/JSON test file (see below)
        # when you use `do_predict` without specifying a GLUE benchmark task.
        if training_args.do_predict:
            if data_args.test_file is not None:
                train_extension = data_args.train_file.split(".")[-1]
                test_extension = data_args.test_file.split(".")[-1]
                assert (
                    test_extension == train_extension
                ), "`test_file` should have the same extension (csv or json) as `train_file`."
                data_files["test"] = data_args.test_file
            else:
                raise ValueError("Need either a GLUE task or a test file for `do_predict`.")

        for key in data_files.keys():
            logger.info(f"load a local file for {key}: {data_files[key]}")

        if data_args.train_file.endswith(".csv"):
            # Loading a dataset from local csv files
            raw_datasets = load_dataset("csv", data_files=data_files, cache_dir=model_args.cache_dir)
        else:
            # Loading a dataset from local json files
            raw_datasets = load_dataset("json", data_files=data_files, cache_dir=model_args.cache_dir)
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    if data_args.task_name is not None:
        is_regression = data_args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # Preprocessing the raw_datasets
    if data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and data_args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif data_args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        return result

    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
    if data_args.task_name == "mnli" and data_args.my_task_name == 'hans':
        mnli_dataset = load_dataset("glue", 'mnli', cache_dir=model_args.cache_dir)
        with training_args.main_process_first(desc="dataset map pre-processing"):
            mnli_dataset = mnli_dataset.map(
                preprocess_function,
                batched=True,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        if data_args.task_name == "qqp" and data_args.my_task_name=='paws':
            import json
            full_train_dataset = raw_datasets["train"]
            if data_args.cur_mode=='default':
                if os.path.isfile(os.path.join(data_args.selected_indices_store_path,
                                               f"qqp_paws_train_{data_args.cur_mode}_{training_args.seed}.json")):
                    with open(os.path.join(data_args.selected_indices_store_path,
                                               f"qqp_paws_train_{data_args.cur_mode}_{training_args.seed}.json")) as f:
                        selected_indices = json.load(f)
                else:
                    selected_indices = random.sample(range(len(full_train_dataset)),10000)
                    with open(os.path.join(data_args.selected_indices_store_path,
                                           f"qqp_paws_train_{data_args.cur_mode}_{training_args.seed}.json"),'w') as f:
                        json.dump(selected_indices,f)
            else:
                all_no_paraphrase_examples = []
                all_paraphrase_examples = []
                ori_seed = training_args.seed
                if data_args.acquisition_function in ['upperbound_checking']:
                    training_args.seed = 2
                if os.path.isfile(os.path.join(model_args.cache_dir,f"qqp_paws_train_examples_{data_args.cur_mode}_{training_args.seed}.json")) and \
                        (data_args.cur_mode_1 is None or
                        (data_args.cur_mode_1 is not None and os.path.isfile(os.path.join(model_args.cache_dir,f"qqp_paws_train_examples_{data_args.cur_mode_1}_{training_args.seed}.json")))):
                    with open(os.path.join(model_args.cache_dir,f"qqp_paws_train_examples_{data_args.cur_mode}_{training_args.seed}.json")) as f:
                        pre_full_train_dataset = json.load(f)
                    training_args.seed = ori_seed
                    print(f"train seed is back to {training_args.seed}")
                elif not data_args.acquisition_function in ['upperbound_checking']:
                    ratio = float(data_args.cur_mode.split('_')[1])
                    no_paraphrase = int(10000*1/(ratio+1))
                    paraphrase = 10000 -  no_paraphrase
                    # all_no_paraphrase_examples = []
                    if data_args.cur_mode.startswith('spurious'):
                        with open('paws_train.tsv') as f:
                            lines = f.readlines()
                        for l in lines[1:]:
                            idx, sentence1, sentence2, label = l.split('\t')
                            label = label.strip()
                            sentence1 = sentence1[2:-1]
                            sentence2 = sentence2[2:-1]
                            if calculate_overlap_ratio(sentence1,sentence2)>0.9 and label=='0':
                                all_no_paraphrase_examples.append({
                                    'question1':sentence1,
                                    'question2':sentence2,
                                    'label':int(label),
                                })
                    else:
                        for e in full_train_dataset:
                            if calculate_overlap_ratio(e['question1'], e['question2']) > 0.9 and e['label'] == 0:
                                all_no_paraphrase_examples.append({
                                    'question1': e['question1'],
                                    'question2': e['question2'],
                                    'label': e['label'],
                                })
                    # all_paraphrase_examples = []
                    for e in full_train_dataset:
                        if calculate_overlap_ratio(e['question1'],e['question2'])>0.9 and e['label']==1:
                            all_paraphrase_examples.append({
                                'question1': e['question1'],
                                'question2': e['question2'],
                                'label': e['label'],
                            })
                    pre_full_train_dataset = random.sample(all_no_paraphrase_examples,no_paraphrase)+\
                                         random.sample(all_paraphrase_examples,paraphrase)
                    with open(os.path.join(model_args.cache_dir,f"qqp_paws_train_examples_{data_args.cur_mode}_{training_args.seed}.json"),'w') as f:
                        json.dump(pre_full_train_dataset,f)
                else:
                    raise ValueError("not implemented")

                    # import copy
                    # all_no_paraphrase_examples_1 = copy.deepcopy(all_no_paraphrase_examples)
                    # all_paraphrase_examples_1 = copy.deepcopy(all_paraphrase_examples)

                if data_args.cur_mode_1 is not None:
                    if os.path.isfile(os.path.join(model_args.cache_dir,
                                                   f"qqp_paws_train_examples_{data_args.cur_mode_1}_{training_args.seed}.json")):
                        with open(os.path.join(model_args.cache_dir,
                                               f"qqp_paws_train_examples_{data_args.cur_mode_1}_{training_args.seed}.json")) as f:
                            pre_full_train_dataset_1 = json.load(f)
                    else:
                        import copy
                        ratio_1 = float(data_args.cur_mode_1.split('_')[-1])
                        no_paraphrase_1 = int(10000 * 1 / (ratio_1 + 1))
                        paraphrase_1 = 10000 - no_paraphrase_1
                        all_no_paraphrase_examples_1 = copy.deepcopy(all_no_paraphrase_examples)
                        all_paraphrase_examples_1 = copy.deepcopy(all_paraphrase_examples)
                        pre_full_train_dataset_1 = random.sample(all_no_paraphrase_examples_1, no_paraphrase_1) + \
                                                 random.sample(all_paraphrase_examples_1, paraphrase_1)
                        print(no_paraphrase_1)
                        for i,e in enumerate(pre_full_train_dataset_1):
                            if i < no_paraphrase_1:
                                assert e['label']==0
                            else:
                                assert e['label']==1
                        with open(os.path.join(model_args.cache_dir,
                                               f"qqp_paws_train_examples_{data_args.cur_mode_1}_{training_args.seed}.json"),
                                  'w') as f:
                            json.dump(pre_full_train_dataset_1, f)
                    full_train_dataset_1 = []
                    for i, e in enumerate(pre_full_train_dataset_1):
                        my_label = e['label']
                        e = preprocess_function(e)
                        e['label'] = my_label
                        e['idx'] = i
                        full_train_dataset_1.append(e)

                full_train_dataset = []
                for i,e in enumerate(pre_full_train_dataset):
                    my_label = e['label']
                    e = preprocess_function(e)
                    e['label'] = my_label
                    e['idx'] = i
                    full_train_dataset.append(e)
                selected_indices = list(range(len(full_train_dataset)))

            if data_args.check_token_coverage:
                import nltk
                store = []
                for e in pre_full_train_dataset:
                    s1 = set(nltk.word_tokenize(e['question1'].lower()))
                    s2 = set(nltk.word_tokenize(e['question2'].lower()))
                    s1.union(s2)
                    store.append(s1)
                all_tokens = set()
                for s in store:
                    all_tokens = all_tokens.union(s)
                if data_args.acquisition_function in ['random']:
                    all_indices = list(range(10000))
                    random.shuffle(all_indices)
                elif data_args.acquisition_function in ['least-confidence']:
                    with open(os.path.join(data_args.selected_indices_store_path,f"confidence_{data_args.tag}.json")) as f:
                        all_indices = json.load(f)
                    assert len(all_indices)>=1000
                elif data_args.acquisition_function in ['self_dissimilar']:
                    with open(os.path.join(data_args.selected_indices_store_path,f"self_dissimilar_{data_args.tag}.json")) as f:
                        all_indices = json.load(f)
                    assert len(all_indices)>=1000
                cur = set()
                coverage = []
                for i in range(50,1050,50):
                    for idx in all_indices[i-50:i]:
                        cur = cur.union(store[idx])
                    coverage.append(len(cur)/len(all_tokens))
                with open(os.path.join(data_args.selected_indices_store_path,f"coverage_{data_args.tag}.json"),'w') as f:
                    json.dump(coverage,f)
                exit(0)



            if data_args.acquisition_function in ['random']:
                final_selected_indices = random.sample(selected_indices,data_args.select_num)
            elif data_args.acquisition_function in ['least-confidence']:
                if os.path.isfile(os.path.join(data_args.selected_indices_store_path,f"confidence_{data_args.tag}.json")):
                    with open(os.path.join(data_args.selected_indices_store_path,f"confidence_{data_args.tag}.json")) as f:
                        final_selected_indices = json.load(f)
                else:
                    final_selected_indices = random.sample(selected_indices, data_args.seed_set_size)
                    with open(
                        os.path.join(data_args.selected_indices_store_path, f"confidence_{data_args.tag}.json"),'w') as f:
                        json.dump(final_selected_indices,f)
            elif data_args.acquisition_function in ['self_dissimilar']:
                if os.path.isfile(os.path.join(data_args.selected_indices_store_path,f"self_dissimilar_{data_args.tag}.json")):
                    with open(os.path.join(data_args.selected_indices_store_path,f"self_dissimilar_{data_args.tag}.json")) as f:
                        final_selected_indices = json.load(f)[:data_args.select_num]
                else:
                    final_selected_indices = random.sample(selected_indices, data_args.seed_set_size)
                    with open(
                            os.path.join(data_args.selected_indices_store_path, f"self_dissimilar_{data_args.tag}.json"),
                            'w') as f:
                        json.dump(final_selected_indices, f)
            elif data_args.acquisition_function in ['all_plus_test']:
                final_selected_indices = selected_indices
            elif data_args.acquisition_function in ['upperbound_checking']:
                from collections import defaultdict
                with open('/home/sysuser/domain/Github/my_scripts_0505/indices/0519/upper_bound/all_plus_test_no_paraphrase_all_plus_test_upper_bound_2.json') as f:
                    ori_no_paraphrase = json.load(f)
                with open('/home/sysuser/domain/Github/my_scripts_0505/indices/0519/upper_bound/all_plus_test_paraphrase_all_plus_test_upper_bound_2.json') as f:
                    ori_paraphrase = json.load(f)
                print("len(ori_no_paraphrase): ",len(ori_no_paraphrase))
                print("len(ori_paraphrase): ",len(ori_paraphrase))
                c = defaultdict(int)
                for k, v in ori_no_paraphrase.items():
                    for p in v[:50]:
                        c[p[0]] += 1
                c = sorted(c.items(), key=lambda x: x[1], reverse=True)
                no_paraphrase = [i[0] for i in c]
                cc = defaultdict(int)
                for k, v in ori_paraphrase.items():
                    for p in v[:50]:
                        cc[p[0]] += 1
                cc = sorted(cc.items(), key=lambda x: x[1], reverse=True)
                paraphrase = [i[0] for i in cc]
                for idx in paraphrase:
                    assert full_train_dataset[idx]['label']==1,f"idx={idx}"
                for idx in no_paraphrase:
                    assert full_train_dataset[idx]['label']==0,f"idx={idx}"
                ratio = 0.393
                no_paraphrase_num = int(data_args.select_num * 1 / (ratio + 1))
                paraphrase_num = data_args.select_num - no_paraphrase_num
                print(f"paraphrase num: {paraphrase_num}, non-paraphrase num: {no_paraphrase_num}")
                final_selected_indices = paraphrase[:paraphrase_num]+no_paraphrase[:no_paraphrase_num]



            train_dataset = [full_train_dataset[i] for i in final_selected_indices]
            print(f"There are {len(train_dataset)} examples to train now")
            ratios = []
            label_distribute = {0: 0, 1: 1}
            for e in train_dataset:
                if data_args.cur_mode=='default':
                    ratios.append(calculate_overlap_ratio(e['question1'],e['question2']))
                label_distribute[e['label']] += 1
            if data_args.cur_mode=='default':
                with open(
                        os.path.join(data_args.selected_indices_store_path, f"{data_args.acquisition_function}_ratio_{data_args.tag}.json"),
                        'w') as f:
                    json.dump(ratios, f)
            with open(
                    os.path.join(data_args.selected_indices_store_path, f"{data_args.acquisition_function}_label_distribute_{data_args.tag}.json"),
                    'w') as f:
                json.dump(label_distribute, f)

        elif data_args.task_name == "mnli" and data_args.my_task_name=='hans':
            from collections import defaultdict
            import json
            hans_train_dataset = raw_datasets["train"]
            old_mnli_dataset = mnli_dataset['train']
            mnli_dataset = []
            for e in old_mnli_dataset:
                e.pop('idx')
                mnli_dataset.append(e)
            offset = len(hans_train_dataset)
            if not os.path.isdir(data_args.selected_indices_store_path):
                os.makedirs(data_args.selected_indices_store_path,exist_ok=True)
            
            print("train indices: ")
            print(os.path.join(data_args.selected_indices_store_path,
                                           f"hans_train_{data_args.cur_mode}_{training_args.seed}.json"))
            print(os.path.join(data_args.selected_indices_store_path,
                                   f"mnli_train_{data_args.cur_mode}_{training_args.seed}.json"))
            if os.path.isfile(os.path.join(data_args.selected_indices_store_path,
                                           f"hans_train_{data_args.cur_mode}_{training_args.seed}.json")) and \
                os.path.isfile(os.path.join(data_args.selected_indices_store_path,
                                   f"mnli_train_{data_args.cur_mode}_{training_args.seed}.json")):
                with open(os.path.join(data_args.selected_indices_store_path,
                                           f"hans_train_{data_args.cur_mode}_{training_args.seed}.json")) as f_hans:
                    hans_selected_indices = json.load(f_hans)
                with open(os.path.join(data_args.selected_indices_store_path,
                                   f"mnli_train_{data_args.cur_mode}_{training_args.seed}.json")) as f_mnli:
                    mnli_selected_indices = json.load(f_mnli)
            else:
                hans_selected_indices = []
                hans_selected_indices_entailment = []
                nd1 = 5000
                ratio = float(data_args.cur_mode.split('_')[-1])/100
                e_num = int(nd1*ratio)
                ne_num = nd1-e_num
                label_format = {0: 'entailment', 1: 'not entailment', 2: 'not entailment'}
                group_indices = {'entailment':defaultdict(list),'not entailment':defaultdict(list)}
                for idx,e in enumerate(hans_train_dataset):
                    if lexical_overlap(e['premise'],e['hypothesis']) and spurious_correlation_type_not_present(e)=='unknown':
                        group_indices[label_format[e['label']]][e['subcase']].append(idx)
                for k1,v1 in group_indices.items():
                    for k2,v2 in v1.items():
                        print(f"train subcase {k2} has {len(group_indices[k1][k2])} examples with label {k1}")
                e_remain = e_num%15
                e_sub = e_num//15
                count = 0
                for k,v in group_indices['entailment'].items():
                    hans_selected_indices_entailment += v
                    if count<e_remain:
                        hans_selected_indices += random.sample(v,e_sub+1)
                        print(f"{k}: {e_sub+1}",end=' ')
                    else:
                        hans_selected_indices += random.sample(v, e_sub)
                        print(f"{k}: {e_sub}",end=' ')
                    count += 1
                index_map = {}
                if data_args.index_map_store_path is not None:
                    for i in hans_selected_indices_entailment:
                        index_map[i] = ['lexical overlap entailment']
                ne_remain = ne_num % 15
                ne_sub = ne_num // 15
                count = 0
                hans_selected_indices_not_entailment = []
                for k, v in group_indices['not entailment'].items():
                    hans_selected_indices_not_entailment += v
                    if count < ne_remain:
                        hans_selected_indices += random.sample(v, ne_sub + 1)
                        print(f"{k}: {ne_sub + 1}", end=' ')
                    else:
                        hans_selected_indices += random.sample(v, ne_sub)
                        print(f"{k}: {ne_sub}", end=' ')
                    count += 1
                if data_args.index_map_store_path is not None:
                    for i in hans_selected_indices_not_entailment:
                        index_map[i] = ['lexical overlap not entailment']
                print(f"\nTrain, lexical overlap with spurious contradiction selects {ne_num} not entailment, {e_num} entailment")
    
                mnli_selected_indices = []
                nd2 = 5000
                ne_num = int(nd2 * ratio)
                e_num = nd2 - ne_num
                group_indices = defaultdict(list)
                for idx,e in enumerate(mnli_dataset):
                    if not lexical_overlap(e['premise'],e['hypothesis']) and spurious_correlation_type_not_present(e)!='unknown':
                        group_indices[label_format[e['label']]].append(idx)
                mnli_selected_indices += random.sample(group_indices['not entailment'],ne_num)
                if data_args.index_map_store_path is not None:
                    for i in group_indices['not entailment']:
                        index_map[i+offset] = ['spurious contradiction not entailment']
                mnli_selected_indices += random.sample(group_indices['entailment'], e_num)
                print(f"Train, spurious contradiction selects {ne_num} not entailment, {e_num} entailment")
                if data_args.index_map_store_path is not None:
                    for i in group_indices['entailment']:
                        index_map[i+offset] = ['spurious contradiction entailment']
                    with open(data_args.index_map_store_path,'w') as f:
                        json.dump(index_map,f)
                    exit(0)
                
                with open(os.path.join(data_args.selected_indices_store_path,f"hans_train_{data_args.cur_mode}_{training_args.seed}.json"),'w') as f:
                    json.dump(hans_selected_indices,f)
                with open(os.path.join(data_args.selected_indices_store_path,
                                       f"mnli_train_{data_args.cur_mode}_{training_args.seed}.json"), 'w') as f:
                    json.dump(mnli_selected_indices, f)
            hans_train_selected_indices = hans_selected_indices
            mnli_train_selected_indices = mnli_selected_indices
            # train_dataset = []
            # for i in hans_selected_indices:
            #     train_dataset.append(hans_train_dataset[i])
            #     train_dataset[-1]['idx'] = i
            # for i in mnli_selected_indices:
            #     train_dataset.append(mnli_dataset[i])
            #     train_dataset[-1]['idx'] = i+offset
            # train_dataset = [hans_train_dataset[i] for i in hans_selected_indices]
            # train_dataset += [mnli_dataset[i] for i in mnli_selected_indices]
            print(f"Total training pool: {len(hans_selected_indices)+len(mnli_selected_indices)}")

            agg_dataset = [e for e in hans_train_dataset]
            agg_dataset += [e for e in mnli_dataset]
            agg_indices = hans_selected_indices + [i+offset for i in mnli_selected_indices]

            if data_args.check_token_coverage:
                import nltk
                def get_token_set(e):
                    s1 = set(nltk.word_tokenize(e['premise'].lower()))
                    s2 = set(nltk.word_tokenize(e['hypothesis'].lower()))
                    s1.union(s2)
                    return s1
                store = []
                pre_full_train_dataset = [agg_dataset[i] for i in agg_indices]
                for e in pre_full_train_dataset:
                    s1 = set(nltk.word_tokenize(e['premise'].lower()))
                    s2 = set(nltk.word_tokenize(e['hypothesis'].lower()))
                    s1.union(s2)
                    store.append(s1)
                all_tokens = set()
                for s in store:
                    all_tokens = all_tokens.union(s)
                if data_args.acquisition_function in ['random']:
                    all_indices = random.sample(agg_indices,10000)
                    random.shuffle(all_indices)
                elif data_args.acquisition_function in ['least-confidence']:
                    with open(os.path.join(data_args.selected_indices_store_path,f"confidence_{data_args.tag}.json")) as f:
                        all_indices = json.load(f)
                    assert len(all_indices)>=1000
                elif data_args.acquisition_function in ['self_dissimilar']:
                    with open(os.path.join(data_args.selected_indices_store_path,f"self_dissimilar_{data_args.tag}.json")) as f:
                        all_indices = json.load(f)
                    assert len(all_indices)>=1000
                cur = set()
                coverage = []
                for i in range(50,1050,50):
                    for idx in all_indices[i-50:i]:
                        cur = cur.union(get_token_set(agg_dataset[idx]))
                    coverage.append(len(cur)/len(all_tokens))
                with open(os.path.join(data_args.selected_indices_store_path,f"coverage_{data_args.tag}.json"),'w') as f:
                    json.dump(coverage,f)
                exit(0)

            if data_args.acquisition_function in ['random']:
                my_hans_train_selected_indices = random.sample(agg_indices,data_args.select_num)
            elif data_args.acquisition_function in ['least-confidence']:
                if os.path.isfile(os.path.join(data_args.selected_indices_store_path,f"confidence_{data_args.tag}.json")):
                    with open(os.path.join(data_args.selected_indices_store_path,f"confidence_{data_args.tag}.json")) as f:
                        my_hans_train_selected_indices = json.load(f)
                else:
                    my_hans_train_selected_indices = random.sample(agg_indices, data_args.seed_set_size)
                    with open(
                        os.path.join(data_args.selected_indices_store_path, f"confidence_{data_args.tag}.json"),'w') as f:
                        json.dump(my_hans_train_selected_indices,f)
            elif data_args.acquisition_function in ['self_dissimilar']:
                if os.path.isfile(os.path.join(data_args.selected_indices_store_path,f"self_dissimilar_{data_args.tag}.json")):
                    with open(os.path.join(data_args.selected_indices_store_path,f"self_dissimilar_{data_args.tag}.json")) as f:
                        my_hans_train_selected_indices = json.load(f)[:data_args.select_num]
                else:
                    my_hans_train_selected_indices = random.sample(agg_indices, data_args.seed_set_size)
                    with open(
                            os.path.join(data_args.selected_indices_store_path, f"self_dissimilar_{data_args.tag}.json"),
                            'w') as f:
                        json.dump(my_hans_train_selected_indices, f)

            # train_dataset = []
            my_hans_train_selected_indices = sorted(my_hans_train_selected_indices)
            # for i in my_hans_train_selected_indices:
            #     train_dataset.append(agg_dataset[i])
            #     train_dataset[-1]['idx'] = i
            train_dataset = [agg_dataset[i] for i in my_hans_train_selected_indices]
            # print(train_dataset[0])
            label_format = {0: 0, 1: 1, 2: 1}
            for e in train_dataset:
                e['label'] = label_format[e['label']]
            print(f"final training set size {len(train_dataset)}")

        elif data_args.task_name == "mnli" and data_args.my_task_name == 'aug_major':
            aug_final_selected_indices = random.sample(list(range(len(raw_datasets['train']))),50)
            train_dataset = [raw_datasets["train"][idx] for idx in aug_final_selected_indices]

        else:
            train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            if isinstance(train_dataset,list):
                train_dataset = train_dataset[:data_args.max_train_samples]
                # train_dataset[-1]['label'] = 0
                # for e in train_dataset[24:30]:
                #     print(e)
                # exit(0)
            else:
                train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if training_args.do_eval:
        if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        if data_args.task_name == "qqp" and data_args.my_task_name=='paws':
            with open('paws_dev_and_test.tsv') as f:
                lines = f.readlines()
            eval_dataset_1 = []
            eval_dataset_0 = []
            train_eval_dataset_1 = []
            train_eval_dataset_0 = []
            import copy
            for l in lines[1:]:
                idx,question1,question2,label = l.split('\t')
                e = preprocess_function({
                    'question1':question1[2:-1],
                    'question2':question2[2:-1],
                    'label':int(label.strip()),
                    'idx':idx
                })
                e['label'] = int(label.strip())
                if e['label']==1:
                    eval_dataset_1.append((e))
                    train_eval_dataset_1.append(copy.deepcopy(e))
                    train_eval_dataset_1[-1]['idx'] = idx
                elif e['label']==0:
                    eval_dataset_0.append((e))
                    train_eval_dataset_0.append(copy.deepcopy(e))
                    train_eval_dataset_0[-1]['idx'] = idx
                else:
                    raise ValueError("unrecognized label")
            print(f"In evaluation, label 1 has {len(eval_dataset_1)} examples, label 0 has {len(eval_dataset_0)} examples")
            # for e in train_dataset:
            #     if not 'idx' in e:
            #         print(1)
            #         print(e)
            #         exit(0)
            offset = len(train_dataset)
            for i in range(len(train_eval_dataset_0)):
                train_eval_dataset_0[i]['idx'] = offset+i
            offset = len(train_dataset)+len(train_eval_dataset_0)
            for i in range(len(train_eval_dataset_1)):
                train_eval_dataset_1[i]['idx'] = offset+i
            print(f"first {len(train_eval_dataset_0)} second {len(train_eval_dataset_1)}")
            if data_args.acquisition_function in ['all_plus_test']:
                if data_args.max_train_samples is not None:
                    train_dataset += train_eval_dataset_0[:data_args.max_train_samples]+train_eval_dataset_1[:data_args.max_train_samples]
                    full_train_dataset_1 += train_eval_dataset_0[:data_args.max_train_samples]+train_eval_dataset_1[:data_args.max_train_samples]
                else:
                    train_dataset += train_eval_dataset_0[:] + train_eval_dataset_1[:]
                    full_train_dataset_1 += train_eval_dataset_0[:] + train_eval_dataset_1[:]
            # for e in eval_dataset_0+eval_dataset_1:
            #     if not 'idx' in e:
            #         print(2)
            #         print(e)
            #         exit(0)
            # print("all examples have idx")
                # print('label' in e)
                # print('label' in eval_dataset[0])
                # print(eval_dataset[0])
                # exit(0)


        elif data_args.task_name == "mnli" and data_args.my_task_name == 'hans':
            from collections import defaultdict
            import json
            # hans_train_dataset = raw_datasets["train"]
            # mnli_dataset = load_dataset("glue", 'mnli', cache_dir=model_args.cache_dir)['train']

            if os.path.isfile(os.path.join(data_args.selected_indices_store_path,
                                           f"hans_test_{data_args.cur_mode}_{training_args.seed}.json")) and \
                    os.path.isfile(os.path.join(data_args.selected_indices_store_path,
                                                f"mnli_test_{data_args.cur_mode}_{training_args.seed}.json")):
                with open(os.path.join(data_args.selected_indices_store_path,
                                       f"hans_test_{data_args.cur_mode}_{training_args.seed}.json")) as f_hans:
                    hans_selected_indices = json.load(f_hans)
                with open(os.path.join(data_args.selected_indices_store_path,
                                       f"mnli_test_{data_args.cur_mode}_{training_args.seed}.json")) as f_mnli:
                    mnli_selected_indices = json.load(f_mnli)
            else:
                hans_selected_indices = []
                nd1 = 1500
                ratio = 0.5
                e_num = int(nd1 * ratio)
                ne_num = nd1 - e_num
                label_format = {0: 'entailment', 1: 'not entailment', 2: 'not entailment'}
                group_indices = {'entailment': defaultdict(list), 'not entailment': defaultdict(list)}
                for idx, e in enumerate(hans_train_dataset):
                    if lexical_overlap(e['premise'], e['hypothesis']) and spurious_correlation_type_not_present(
                            e) == 'unknown' and not idx in hans_train_selected_indices:
                        group_indices[label_format[e['label']]][e['subcase']].append(idx)
                for k1, v1 in group_indices.items():
                    for k2, v2 in v1.items():
                        print(f"test subcase {k2} has {len(group_indices[k1][k2])} examples with label {k1}")
                e_remain = e_num % 15
                e_sub = e_num // 15
                count = 0
                for k, v in group_indices['entailment'].items():
                    if count < e_remain:
                        hans_selected_indices += random.sample(v, e_sub + 1)
                        print(f"{k}: {e_sub + 1}", end=' ')
                    else:
                        hans_selected_indices += random.sample(v, e_sub)
                        print(f"{k}: {e_sub}", end=' ')
                    count += 1
                ne_remain = ne_num % 15
                ne_sub = ne_num // 15
                count = 0
                for k, v in group_indices['not entailment'].items():
                    if count < ne_remain:
                        hans_selected_indices += random.sample(v, ne_sub + 1)
                        print(f"{k}: {ne_sub + 1}", end=' ')
                    else:
                        hans_selected_indices += random.sample(v, ne_sub)
                        print(f"{k}: {ne_sub}", end=' ')
                    count += 1
                print(
                    f"\nTest, lexical overlap with spurious contradiction selects {len(hans_selected_indices)} examples")

                mnli_selected_indices = []
                nd2 = 1500
                ne_num = int(nd2 * ratio)
                e_num = nd2 - ne_num
                group_indices = defaultdict(list)
                for idx, e in enumerate(mnli_dataset):
                    if not lexical_overlap(e['premise'], e['hypothesis']) and spurious_correlation_type_not_present(
                            e) != 'unknown' and not idx in mnli_train_selected_indices:
                        group_indices[label_format[e['label']]].append(idx)
                mnli_selected_indices += random.sample(group_indices['not entailment'], ne_num)
                mnli_selected_indices += random.sample(group_indices['entailment'], e_num)
                print(f"Test, spurious contradiction selects {ne_num} not entailment, {e_num} entailment")

                with open(os.path.join(data_args.selected_indices_store_path,
                                       f"hans_test_{data_args.cur_mode}_{training_args.seed}.json"), 'w') as f:
                    json.dump(hans_selected_indices, f)
                with open(os.path.join(data_args.selected_indices_store_path,
                                       f"mnli_test_{data_args.cur_mode}_{training_args.seed}.json"), 'w') as f:
                    json.dump(mnli_selected_indices, f)
            eval_dataset = []
            for i in hans_selected_indices:
                eval_dataset.append(hans_train_dataset[i])
                # eval_dataset[-1]['idx'] = i
            for i in mnli_selected_indices:
                eval_dataset.append(mnli_dataset[i])
                # eval_dataset[-1]['idx'] = i+offset
            label_format = {0: 0, 1: 1, 2: 1}
            for e in eval_dataset:
                e['label'] = label_format[e['label']]
            # eval_dataset = [hans_train_dataset[i] for i in hans_selected_indices]
            # eval_dataset += [mnli_dataset[i] for i in mnli_selected_indices]
            print(f"Total evaluation pool: {len(eval_dataset)}")
        else:
            eval_dataset = raw_datasets["validation_matched" if data_args.task_name == "mnli" and data_args.my_task_name is None else "validation"]
        if data_args.max_eval_samples is not None:
            if data_args.task_name=='qqp' and data_args.my_task_name=='paws':
                eval_dataset_0 = eval_dataset_0[:data_args.max_eval_samples]
                eval_dataset_1 = eval_dataset_1[:data_args.max_eval_samples]
            elif isinstance(eval_dataset,list):
                eval_dataset = eval_dataset[:data_args.max_eval_samples]
            else:
                eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

    # if training_args.do_predict or data_args.task_name is not None or data_args.test_file is not None:
    if training_args.do_predict or data_args.acquisition_function in ['least-confidence','self_dissimilar','all_plus_test']:
        if data_args.task_name == "mnli" and data_args.my_task_name == 'hans':
            selected_indices = agg_indices
            final_selected_indices = my_hans_train_selected_indices
            full_train_dataset = agg_dataset

        if data_args.task_name == "mnli" and data_args.my_task_name == 'aug_major':
            import copy
            available_indices = list(set(list(range(len(raw_datasets['train']))))-set(aug_final_selected_indices))
            predict_indices = random.sample(available_indices,50)
            ori_predict_dataset = [raw_datasets["train"][idx] for idx in predict_indices]
            predict_dataset = []

            def add_comma(s):
                components = s.split(' ')
                random_indices = random.sample(list(range(len(components))),2)
                components[random_indices[0]] = components[random_indices[0]]+','
                components[random_indices[1]] = components[random_indices[1]] + ','
                return ' '.join(components)

            def add_period(s):
                components = s.split(' ')
                random_indices = random.sample(list(range(len(components))), 2)
                components[random_indices[0]] = components[random_indices[0]] + '.'
                # components[random_indices[1]] = components[random_indices[1]] + '.'
                return ' '.join(components)

            def add_word(s):
                components = s.split(' ')
                random_indices = random.sample(list(range(len(components))),2)
                components[random_indices[0]] = components[random_indices[0]]+', huh,'
                # components[random_indices[1]] = components[random_indices[1]] + ', huh,'
                return ' '.join(components)

            for e in ori_predict_dataset:
                e.pop('label')
                predict_dataset.append(e)

                comma = copy.deepcopy(e)
                comma['premise'] = add_comma(comma['premise'])
                comma['hypothesis'] = add_comma(comma['hypothesis'])
                predict_dataset.append(comma)

                period = copy.deepcopy(e)
                period['premise'] = add_period(period['premise'])
                period['hypothesis'] = add_period(period['hypothesis'])
                predict_dataset.append(period)

                prefix1 = copy.deepcopy(e)
                prefix1['premise'] = 'The premise: '+prefix1['premise']
                prefix1['hypothesis'] = 'The hypothesis: ' + prefix1['hypothesis']
                predict_dataset.append(prefix1)

                prefix2 = copy.deepcopy(e)
                prefix2['premise'] = 'Sentence 1: ' + prefix2['premise']
                prefix2['hypothesis'] = 'Sentence 2: ' + prefix2['hypothesis']
                predict_dataset.append(prefix2)

                word_added = copy.deepcopy(e)
                word_added['premise'] = add_word(word_added['premise'])
                word_added['hypothesis'] = add_word(word_added['hypothesis'])
                predict_dataset.append(word_added)




        if data_args.acquisition_function in ['least-confidence','self_dissimilar']:
            if data_args.acquisition_function in ['least-confidence']:
                training_args.do_predict = True
            indices_to_predict = []
            predict_dataset = []
            for idx in final_selected_indices:
                if not idx in selected_indices:
                    print("not in")
            for idx in selected_indices:
                if data_args.acquisition_function in ['least-confidence']:
                    if not idx in final_selected_indices:
                        indices_to_predict.append(idx)
                if data_args.acquisition_function in ['self_dissimilar']:
                    indices_to_predict.append(idx)
            if data_args.acquisition_function in ['all_plus_test']:
                indices_to_predict = range(len(full_train_dataset))
            import copy
            for idx in indices_to_predict:
                predict_dataset.append(copy.deepcopy(full_train_dataset[idx]))
                if 'idx' not in predict_dataset[-1]:
                    predict_dataset[-1]['idx'] = idx
                else:
                    assert idx==predict_dataset[-1]['idx']
                if not data_args.acquisition_function in ['all_plus_test']:
                    predict_dataset[-1].pop('label')
                # if len(predict_dataset[-1])<3:
                #     print(predict_dataset[-1])
                #     exit(0)
            # label_format = {0: 0, 1: 1, 2: 1}
            # for e in predict_dataset:
            #     e['label'] = label_format[e['label']]
            print(f"{data_args.acquisition_function} prediction pool: {len(predict_dataset)}")

        elif data_args.acquisition_function in ['all_plus_test']:
            predict_dataset = full_train_dataset_1
            print(f"all plus test has {len(full_train_dataset_1)} examples")

        elif not (data_args.task_name == "mnli" and data_args.my_task_name == 'aug_major'):
            if "test" not in raw_datasets and "test_matched" not in raw_datasets:
                raise ValueError("--do_predict requires a test dataset")
            predict_dataset = raw_datasets["test_matched" if data_args.task_name == "mnli" else "test"]

        if data_args.max_predict_samples is not None:
            if isinstance(predict_dataset,list):
                predict_dataset = predict_dataset[:data_args.max_predict_samples]
            else:
                predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))

    # Log a few random samples from the training set:
    # if training_args.do_train:
    #     for index in random.sample(range(len(train_dataset)), 3):
    #         logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Get the metric function
    if data_args.task_name is not None:
        metric = load_metric("glue", data_args.task_name)
    else:
        metric = load_metric("accuracy")

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        if data_args.task_name is not None:
            if data_args.task_name == "mnli" and data_args.my_task_name=='hans':
                print("label is formatted")
                label_format = {0:0,1:1,2:1}
                preds = [label_format[i] for i in preds]
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    # we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Initialize our Trainer
    # print('train_dataset[0] ',train_dataset[0])
    # exit(0)
    if data_args.task_name == "qqp" and data_args.my_task_name == 'paws':
        trainer = Trainer(
            model=model,
            args=training_args,
            data_args=data_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset_1 if training_args.do_eval else None,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            data_args=data_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        if data_args.task_name=='qqp' and data_args.my_task_name=='paws':
            eval_datasets = [eval_dataset_1,eval_dataset_0]
            tasks.append(data_args.task_name)
        else:
            eval_datasets = [eval_dataset]
        if data_args.task_name == "mnli" and data_args.my_task_name is None:
            tasks.append("mnli-mm")
            eval_datasets.append(raw_datasets["validation_mismatched"])

        for eval_dataset, task in zip(eval_datasets, tasks):
            if data_args.task_name == 'qqp' and data_args.my_task_name == 'paws':
                print(f'current group has label {eval_dataset[0]["label"]}')
            metrics = trainer.evaluate(eval_dataset=eval_dataset)

            max_eval_samples = (
                data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
            )
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        predict_datasets = [predict_dataset]
        if data_args.task_name == "mnli" and data_args.my_task_name is None:
            tasks.append("mnli-mm")
            predict_datasets.append(raw_datasets["test_mismatched"])
        if data_args.acquisition_function in ['least-confidence']:
            data_args.confidence_prediction = True

        for predict_dataset, task in zip(predict_datasets, tasks):
            # Removing the `label` columns because it contains -1 and Trainer won't like that.
            if not data_args.acquisition_function in ['least-confidence'] and not (data_args.task_name == "mnli" and data_args.my_task_name == 'aug_major'):
                predict_dataset = predict_dataset.remove_columns("label")
            predictions = trainer.predict(predict_dataset, metric_key_prefix="predict").predictions
            # print("is_regression: ",is_regression)
            # print("predictions: ",type(predictions),len(predictions),predictions[0].shape,predictions[1].shape)
            if (data_args.task_name == "mnli" and data_args.my_task_name == 'aug_major') or (data_args.task_name != "mnli" and not data_args.acquisition_function in ['least-confidence']):
                predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions[0], axis=1)

                output_predict_file = os.path.join(training_args.output_dir, f"predict_results_{task}.txt")
                if trainer.is_world_process_zero():
                    with open(output_predict_file, "w") as writer:
                        logger.info(f"***** Predict results {task} *****")
                        writer.write("index\tprediction\n")
                        for index, item in enumerate(predictions):
                            if is_regression:
                                writer.write(f"{index}\t{item:3.3f}\n")
                            else:
                                item = label_list[item]
                                writer.write(f"{index}\t{item}\n")
            if data_args.my_task_name in ['aug_major']:
                set_seed(training_args.seed+10)
                predictions = trainer.predict(predict_dataset, metric_key_prefix="predict").predictions
                if (data_args.task_name == "mnli" and data_args.my_task_name == 'aug_major') or (
                        data_args.task_name != "mnli" and not data_args.acquisition_function in ['least-confidence']):
                    predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions[0], axis=1)
                    output_predict_file = os.path.join(training_args.output_dir, f"predict_results_{task}_1.txt")
                    if trainer.is_world_process_zero():
                        with open(output_predict_file, "w") as writer:
                            logger.info(f"***** Predict results {task} *****")
                            writer.write("index\tprediction\n")
                            for index, item in enumerate(predictions):
                                if is_regression:
                                    writer.write(f"{index}\t{item:3.3f}\n")
                                else:
                                    item = label_list[item]
                                    writer.write(f"{index}\t{item}\n")
        data_args.confidence_prediction = False

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-classification"}
    if data_args.task_name is not None:
        kwargs["language"] = "en"
        kwargs["dataset_tags"] = "glue"
        kwargs["dataset_args"] = data_args.task_name
        kwargs["dataset"] = f"GLUE {data_args.task_name.upper()}"

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

    if data_args.acquisition_function in ['self_dissimilar'] and data_args.calculate_dissimilar_embeds:
        import torch
        from tqdm import tqdm
        import json
        from sklearn.metrics.pairwise import cosine_similarity
        dataloader = trainer.get_test_dataloader(predict_dataset)
        trainer.model.eval()
        local_indices = []
        embeds = []
        for step, inputs in enumerate(dataloader):
            # print(inputs)
            local_indices += inputs.pop('idx').tolist()
            inputs = trainer._prepare_input(inputs)
            # print(inputs)
            # exit(0)
            outputs = model(**inputs,return_dict=True)
            embeds += outputs.pooled_output.cpu().tolist()
        indices_map = {}
        for i,idx in enumerate(local_indices):
            indices_map[idx] = i
        # print(local_indices)
        # print(my_hans_train_selected_indices)
        downstream_representations = torch.tensor(embeds, dtype=torch.float)
        downstream_representations_mean = torch.mean(downstream_representations, 0, True)
        downstream_representations = downstream_representations - downstream_representations_mean
        if data_args.task_name == "mnli" and data_args.my_task_name == 'hans':
            final_selected_indices = my_hans_train_selected_indices
        selected_indices = [indices_map[i] for i in final_selected_indices]
        num_instance = min(len(downstream_representations), data_args.self_dissimilar_cap)
        newly_selected_representations = downstream_representations[selected_indices]
        scores = np.array([0 for i in range(len(downstream_representations))], dtype=np.float64)
        progress_bar = tqdm(range(num_instance), desc="calculate self similarity")
        for i in selected_indices:
            scores[i] = float('inf')
        for count in range(num_instance):
            scores += np.sum(cosine_similarity(downstream_representations, newly_selected_representations), axis=1)
            min_idx = np.argmin(scores)
            newly_selected_representations = downstream_representations[min_idx].reshape(1, -1)
            selected_indices.append(min_idx.item())
            scores[min_idx.item()] = float('inf')
            progress_bar.update(1)
        selected_indices = [local_indices[i] for i in selected_indices]
        with open(os.path.join(data_args.selected_indices_store_path, f"self_dissimilar_{data_args.tag}.json"),'w') as f:
            json.dump(selected_indices,f)
    if data_args.acquisition_function in ['all_plus_test'] and data_args.calculate_dissimilar_embeds:
        ratio_1 = float(data_args.cur_mode_1.split('_')[-1])
        no_paraphrase_1 = int(10000 * 1 / (ratio_1 + 1))
        no_paraphrase_train = predict_dataset[:no_paraphrase_1]
        paraphrase_train = predict_dataset[no_paraphrase_1:10000]
        print("no_paraphrase_1: ",no_paraphrase_1)

        for e in no_paraphrase_train:
            assert e['label']==0
        for e in paraphrase_train:
            assert e['label']==1
        no_paraphrase_test = predict_dataset[10000:10486]
        paraphrase_test = predict_dataset[10486:10677]
        for e in no_paraphrase_test:
            assert e['label']==0
        for e in paraphrase_test:
            assert e['label']==1
        import torch
        from tqdm import tqdm
        import json
        from sklearn.metrics.pairwise import cosine_similarity
        def get_indices(train_split,test_split,offset):
            dataloader = trainer.get_test_dataloader(train_split+test_split)
            trainer.model.eval()
            local_indices = []
            embeds = []
            for step, inputs in enumerate(dataloader):
                # print(inputs)
                local_indices += inputs.pop('idx').tolist()
                inputs = trainer._prepare_input(inputs)
                # print(inputs)
                # exit(0)
                outputs = model(**inputs, return_dict=True)
                embeds += outputs.pooled_output.cpu().tolist()
            downstream_representations = torch.tensor(embeds, dtype=torch.float)
            downstream_representations_mean = torch.mean(downstream_representations, 0, True)
            downstream_representations = downstream_representations - downstream_representations_mean
            train_embs = downstream_representations[:len(train_split)]
            test_embs = downstream_representations[len(train_split):]
            scores = np.array([0 for i in range(len(train_embs))], dtype=np.float64)
            top_scores_indices = {}
            for idx,one_test_emb in enumerate(test_embs):
                scores += np.sum(cosine_similarity(train_embs, one_test_emb.reshape(1,-1)), axis=1)
                processed_scores = sorted([[i+offset,s] for i,s in enumerate(scores)],key=lambda x:x[1])
                top_scores_indices[idx] = processed_scores[-500:]
            return top_scores_indices
        no_paraphrase_indices = get_indices(no_paraphrase_train,no_paraphrase_test,0)
        paraphrase_indices = get_indices(paraphrase_train,paraphrase_test,no_paraphrase_1)
        with open(os.path.join(data_args.selected_indices_store_path, f"all_plus_test_paraphrase_{data_args.tag}.json"),
                  'w') as f:
            json.dump(paraphrase_indices, f)
        with open(os.path.join(data_args.selected_indices_store_path, f"all_plus_test_no_paraphrase_{data_args.tag}.json"),
                  'w') as f:
            json.dump(no_paraphrase_indices, f)








def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()

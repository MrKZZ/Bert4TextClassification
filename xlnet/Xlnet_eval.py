# -*- encoding: utf-8 -*-
'''
@File    :   Xlnet_eval.py
@Time    :   2019/11/09 12:12:36
@Author  :   Kang Zhezhou
@Version :   1.0
@Contact :   kangzhezhou18@mails.ucas.ac.cn, kangzhezhou@iie.ac.cn
@License :   (C)Copyright 2014-3XXX, kang NLP
@Description : 
    测试xlnet训练结果，调用transformers库
'''

import torch
from transformers import *
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
import random
import numpy as np
import os, pickle
import argparse
from tqdm import tqdm
import copy, json
import glob
from apex import amp
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import glue_compute_metrics as compute_metrics

def load_dataset(lines, labels):
    """
        convert examples for the training sets for document classification
    """
    examples = []
    for (i, (line, label)) in enumerate(zip(lines, labels)):
        line = line.strip()
        label = label.strip()
        # label = str(i % 2)
        guid = i
        examples.append(
            InputExample(guid=guid, text_a=line, label=label)
        )
    return examples

def load_and_cache_examples(args, tokenizer):
    with open(args.data_dir+"test_texts", "r") as fr:
        texts = fr.readlines()
    with open(args.data_dir+"test_labels", "r") as fr:
        labels = fr.readlines()
    examples = load_dataset(texts, labels)
    label_list = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    features = convert_examples_to_features(examples, tokenizer,
                                 label_list=label_list,
                                 output_mode="classification", max_length=32)  #这个过程中添加了special token
    
    cached_path = "cached_file"
    torch.save(features, cached_path)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def evaluate(args, model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir
    eval_dataset = load_and_cache_examples(args, tokenizer)

    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                    'attention_mask': batch[1],
                    'labels':         batch[3]}
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs['labels'].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
    eval_loss = eval_loss / nb_eval_steps
    if args.output_mode == "classification":
        preds = np.argmax(preds, axis=-1)
    elif args.output_mode == "regression":
        preds = np.squeeze(preds)
    result = simple_accuracy(preds, out_label_ids)
    with open("result_watch_{}.txt".format(prefix), "w") as fw:
        for e in preds:
            fw.write(str(e)+" ")
        fw.write("\n")
        for e in out_label_ids:
            fw.write(str(e)+" ")
    return result



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/home/54215/center_project/data/",
                        help="data path for train or test")
    parser.add_argument("--n_gpu", type=int, default=1, help="number of gpus to run")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--output_dir", default="./checkpoint_", type=str, 
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--task_name", default="multi", type=str, 
                        help="task name.")
    parser.add_argument("--per_gpu_eval_batch_size", default=32, type=int, 
                        help="task name.")
    parser.add_argument("--output_mode", default="classification", type=str, 
                        help="task name.")
    parser.add_argument("--eval_checkpoint", default="5000", type=str, help="the checkpoint to reload")
    args = parser.parse_args()
    args.device = torch.device("cuda")
    results = {}
    model_class = XlNetForSequenceClassification
    tokenizer_class = XlNetTokenizer
    # tokenizer = tokenizer_class.from_pretrained("-base-chinese")
    checkpoints = [args.output_dir+"/checkpoint-"+args.eval_checkpoint]
    for checkpoint in checkpoints:
        global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
        prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""
        # tokenizer = tokenizer_class.from_pretrained(checkpoint)
        model = model_class.from_pretrained(checkpoint)
        model.to(args.device)
        result = evaluate(args, model, tokenizer, prefix=prefix)
    print("result: ", result)
    
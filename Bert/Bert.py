# -*- encoding: utf-8 -*-
'''
@File    :   Bert.py
@Time    :   2019/11/07 09:10:50
@Author  :   Kang Zhezhou
@Version :   1.0
@Contact :   kangzhezhou18@mails.ucas.ac.cn, kangzhezhou@iie.ac.cn
@License :   (C)Copyright 2014-3XXX, kang NLP
@Description : 
    文本分类的bert方法，使用开源框架实现
'''

import torch
from transformers import *
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
import random
import numpy as np
import os, pickle
import argparse
from tqdm import tqdm, trange
import copy, json
import glob
from apex import amp
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import glue_compute_metrics as compute_metrics
from Bert_eval import evaluate

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def train(args, model, tokenizer):
    with open("../../data/train_texts", "r") as fr:
        texts = fr.readlines()
    with open("../../data/train_labels", "r") as fr:
        labels = fr.readlines()
    examples = load_dataset(texts, labels)
    label_list = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    features = convert_examples_to_features(examples, tokenizer,
                                 label_list=label_list,
                                 output_mode="classification",
                                 max_length=32)  #这个过程中添加了special token
    
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    train_dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size)
    t_total = len(train_dataloader) * args.num_train_epochs
    args.t_total = t_total
    args.warmup_steps = 0.1 * t_total
    
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)  #!
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=args.t_total)  #!!!
    
    # Train!
    print("***** Running training *****")
    print("  Num examples = %d", len(train_dataset))
    print("  Num Epochs = %d", args.num_train_epochs)
    print("  Total optimization steps = %d", args.t_total)
    tr_loss = 0.0
    global_step = 0
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    best_acc = 0.0
    for epoch, _ in enumerate(train_iterator):
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            if step % 100 ==0:
                print("lr: {}".format(scheduler.get_lr()))
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids':      batch[0],
                    'attention_mask': batch[1],
                    'token_type_ids': batch[2],
                    'labels':         batch[3]}

            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
            loss.backward()
            tr_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            global_step += 1
        result = evaluate(args, model, tokenizer)
        print("result: {}, best: {}".format(result, best_acc))
        if result >= best_acc:
            print("model saved")
            best_acc = result
            output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(epoch))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
            model_to_save.save_pretrained(output_dir)
            torch.save(args, os.path.join(output_dir, 'training_args.bin'))


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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/home/54215/center_project/data/",
                        help="data path for train or test")
    parser.add_argument("--train_batch_size", type=int, default=300, help="training batch size")#256
    parser.add_argument("--eval_batch_size", type=int, default=32, help="eval batch size")
    parser.add_argument("--t_total", type=int, default=100, help="training epoch")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=100, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=10, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--output_dir", default="./checkpoint", type=str, 
                        help="The output directory where the model predictions and checkpoints will be written.")
    
    parser.add_argument("--per_gpu_eval_batch_size", default=32, type=int, 
                        help="per gpu batch size.")
    parser.add_argument("--output_mode", default="classification", type=str, 
                        help="task name.")
    parser.add_argument("--eval_checkpoint", default="5000", type=str, help="the checkpoint to reload")
    parser.add_argument("--n_gpu", type=int, default=1, help="number of gpus to run")
    
    # parser.add_argument("--gpu", type=int, default=0, help="choose gpu device")

    args = parser.parse_args()
    args.retrain = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    model_class = BertForSequenceClassification
    tokenizer_class = BertTokenizer
    pretrained_weights = "bert-base-chinese"
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    if args.retrain:
        model = model_class.from_pretrained(args.output_dir+"/checkpoint-final")
    else:
        model = model_class.from_pretrained(pretrained_weights, num_labels=10)
    model.to(args.device)
    train(args, model, tokenizer)
    
    # Create output directory if needed
    output_dir = args.output_dir+"/checkpoint-final"
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(args.output_dir)
    # Good practice: save your training arguments together with the trained model
    torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

    results = {}
    
    tokenizer = tokenizer_class.from_pretrained(args.output_dir)
    checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
        
    for checkpoint in checkpoints:
        global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
        prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""
        
        model = model_class.from_pretrained(checkpoint)
        model.to(args.device)
        result = evaluate(args, model, tokenizer, prefix=prefix)
        result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
        results.update(result)

    print("result: ", result)
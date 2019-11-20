# -*- encoding: utf-8 -*-
'''
@File    :   Bert_1.py
@Time    :   2019/11/08 13:32:13
@Author  :   Kang Zhezhou
@Version :   1.0
@Contact :   kangzhezhou18@mails.ucas.ac.cn, kangzhezhou@iie.ac.cn
@License :   (C)Copyright 2014-3XXX, kang NLP
@Description : 
    按照相关文档的bert实现，完成Bert模型训练，多分类任务。
'''

from transformers import DataProcessor, InputExample

class MultiClassifcationProcessor(DataProcessor):
    def get_examples(self, data_dir):
        with open(data_dir, "r") as fr:
            data = fr.readlines()
        return data

    def get_labels(self, data_dir):
        with open(data_dir, "r") as fr:
            data = fr.readlines()
        return data

    def _create_examples(self, data, labels):
        examples = []
        for (i, (line, label)) in enumerate(zip(data, labels)):
            line = line.strip()
            label = label.strip()
            guid = i
            examples.append(InputExample(guid=guid, text_a=line, label=label))
        return examples


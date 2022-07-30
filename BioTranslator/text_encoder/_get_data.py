import torch
import os
import json
import random
from tqdm import tqdm
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig
import scipy.stats
import numpy as np


def read_ontology_names(folder_path):
    texts = set()
    name_lines = open(f'{folder_path}/name.txt').readlines()
    for i, name_line in enumerate(name_lines):
        texts.add(name_line)
    return texts


def read_graph_to_dict(folder_path):
    return json.load(open(f'{folder_path}/graph.json', 'r', encoding='utf8'))


def read_text_to_dict(folder_path):
    texts = dict()
    name_lines = open(f'{folder_path}/name.txt').readlines()
    def_lines = open(f'{folder_path}/def.txt').readlines()
    for i, name in enumerate(name_lines):
        texts[name.strip()] = f'{name.strip()}. {def_lines[i].strip()}'
    return texts


def get_data(dir_path: str):
    file_list = os.listdir(dir_path)
    file_exclude = ['go', 'cl']
    train_texts, test_texts = [], []
    exclude_n = 0
    exclude_name = set()
    for file in file_exclude:
        text = read_ontology_names(dir_path + file)
        exclude_name.update(text)
    for file in tqdm(file_list):
        graph = read_graph_to_dict(dir_path + file)
        text = read_text_to_dict(dir_path + file)
        if file in file_exclude:
            test_texts.extend([(text[n1.strip()], text[n2.strip()]) for n1 in graph for n2 in graph[n1]])
            continue
        for n1 in graph.keys():
            for n2 in graph[n1]:
                if n1.strip() in exclude_name or n2.strip() in exclude_name:
                    exclude_n += 1
                    continue
                train_texts.append((text[n1.strip()], text[n2.strip()]))
    print('Exclude {} edges related to GO or CL!'.format(exclude_n))
    random.shuffle(train_texts)
    print(len(train_texts))
    return train_texts.copy(), test_texts.copy()
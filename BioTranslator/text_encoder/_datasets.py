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


class TrainOntologyDataset(Dataset):
    def __init__(self, data, tokenizer, max_len, transform=None, target_transform=None):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.transform = transform
        self.target_transform = target_transform

    def text_to_id(self, source):
        sample = self.tokenizer([source[0], source[1]], max_length=self.max_len, truncation=True, padding='max_length',
                                return_tensors='pt')
        return sample

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.text_to_id(self.data[idx])


class TestOntologyDataset:
    def __init__(self, data, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.target_idxs = self.text_to_id([x[0] for x in data])
        self.source_idxs = self.text_to_id([x[1] for x in data])
        self.label_list = [int(x[2]) for x in data]
        assert len(self.target_idxs['input_ids']) == len(self.source_idxs['input_ids'])

    def text_to_id(self, source):
        sample = self.tokenizer(source, max_length=self.max_len, truncation=True, padding='max_length',
                                return_tensors='pt')
        return sample

    def get_data(self):
        return self.target_idxs, self.source_idxs, self.label_list
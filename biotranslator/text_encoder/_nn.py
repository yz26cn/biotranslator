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


class NeuralNetwork(nn.Module):
    def __init__(self, model_path, output_way, config):
        super(NeuralNetwork, self).__init__()
        self.bert = AutoModel.from_pretrained(model_path, config=config)
        self.output_way = output_way
        assert output_way in ['cls', 'pooler']

    def forward(self, input_ids, attention_mask, token_type_ids):
        x1 = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        if self.output_way == 'cls':
            output = x1.last_hidden_state[:, 0]
        elif self.output_way == 'pooler':
            output = x1.pooler_output
        return output
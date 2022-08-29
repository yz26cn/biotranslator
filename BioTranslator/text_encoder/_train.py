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


def compute_loss(y_pred, device, lamda=0.05):
    idxs = torch.arange(y_pred.shape[0], device=device)
    y_true = idxs + 1 - idxs % 2 * 2
    similarities = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=2)
    similarities = similarities - torch.eye(y_pred.shape[0], device=device) * 1e12
    similarities = similarities / lamda
    loss = F.cross_entropy(similarities, y_true)
    return torch.mean(loss)


def test(model, test_dl, batch, device):
    test_size = len(test_dl.dataset)
    for test_d in test_dl:
        with torch.no_grad():
            t_input_ids = test_d['input_ids'].view(len(test_d['input_ids']) * 2, -1).to(device)
            t_attention_mask = test_d['attention_mask'].view(len(test_d['attention_mask']) * 2, -1).to(
                device)
            t_token_type_ids = test_d['token_type_ids'].view(len(test_d['token_type_ids']) * 2, -1).to(
                device)
            t_pred = model(t_input_ids, t_attention_mask, t_token_type_ids)
            t_loss = compute_loss(t_pred, device)
        t_loss, t_current = t_loss.item(), batch * int(len(t_input_ids) / 2)
        print(f"Batch: {batch} -> Test loss: {t_loss:>7f}  [{t_current:>5d}/{test_size:>5d}]")
        break


def train(train_dl, test_dl, model, optimizer, save_path, device):
    model.train()
    train_size = len(train_dl.dataset)
    for batch, data in tqdm(enumerate(train_dl)):
        input_ids = data['input_ids'].view(len(data['input_ids']) * 2, -1).to(device)
        attention_mask = data['attention_mask'].view(len(data['attention_mask']) * 2, -1).to(device)
        token_type_ids = data['token_type_ids'].view(len(data['token_type_ids']) * 2, -1).to(device)
        pred = model(input_ids, attention_mask, token_type_ids)
        loss = compute_loss(pred, device)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 100 == 0:
            loss, current = loss.item(), batch * int(len(input_ids) / 2)
            print(f"Batch: {batch} -> Training loss: {loss:>7f}  [{current:>5d}/{train_size:>5d}]")
            test(model, test_dl, batch, device)
    torch.save(model.state_dict(), save_path)
    print("Saved PyTorch Model State to {}".format(save_path))

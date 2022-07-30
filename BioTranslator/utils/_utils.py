import os
import sys
import torch
import logging
import time
import pickle
import collections
import numpy as np
import pandas as pd
from torch import nn
import networkx as nx
from tqdm import tqdm
from torch import squeeze
from torch.nn import init
from collections import deque
from gensim import corpora, models
from scipy.sparse.linalg import svds
from nltk.tokenize import word_tokenize
from torch.utils.data import ConcatDataset
from gensim.models.word2vec import Word2Vec
import torchvision.transforms as transforms
from transformers import AutoTokenizer, AutoModel
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import nltk
nltk.download('punkt')

def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


def load(filename, with_rels=True):
    ont = dict()
    obj = None
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line == '[Term]':
                if obj is not None:
                    ont[obj['id']] = obj
                obj = dict()
                obj['is_a'] = list()
                obj['part_of'] = list()
                obj['regulates'] = list()
                obj['alt_ids'] = list()
                obj['def'] = list()
                obj['is_obsolete'] = False
                continue
            elif line == '[Typedef]':
                if obj is not None:
                    ont[obj['id']] = obj
                obj = None
            else:
                if obj is None:
                    continue
                l = line.split(": ")
                if l[0] == 'id':
                    obj['id'] = l[1]
                elif l[0] == 'alt_id':
                    obj['alt_ids'].append(l[1])
                elif l[0] == 'namespace':
                    obj['namespace'] = l[1]
                elif l[0] == 'def':
                    obj['def'].append(l[1].split('"')[1])
                elif l[0] == 'is_a':
                    obj['is_a'].append(l[1].split(' ! ')[0])
                elif with_rels and l[0] == 'relationship':
                    it = l[1].split()
                    # add all types of relationships
                    obj['is_a'].append(it[1])
                elif l[0] == 'name':
                    obj['name'] = l[1]
                elif l[0] == 'is_obsolete' and l[1] == 'true':
                    obj['is_obsolete'] = True
        if obj is not None:
            ont[obj['id']] = obj
    for term_id in list(ont.keys()):
        for t_id in ont[term_id]['alt_ids']:
            ont[t_id] = ont[term_id]
        if ont[term_id]['is_obsolete']:
            del ont[term_id]
    for term_id, val in ont.items():
        if 'children' not in val:
            val['children'] = set()
        for p_id in val['is_a']:
            if p_id in ont:
                if 'children' not in ont[p_id]:
                    ont[p_id]['children'] = set()
                ont[p_id]['children'].add(term_id)
    return ont

def term2preds_label(preds, label, terms, terms2id):
    new_preds, new_label = [], []
    for t_id in terms:
        new_preds.append(preds[:, terms2id[t_id]].reshape((-1, 1)))
        new_label.append(label[:, terms2id[t_id]].reshape((-1, 1)))
    new_preds = np.concatenate(new_preds, axis=1)
    new_label = np.concatenate(new_label, axis=1)
    return new_preds, new_label

def get_logger(logfile):
    '''
    This function are copied from https://blog.csdn.net/a232884c/article/details/117453011
    :param logfile:
    :return:
    '''
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

def get_namespace_terms(terms, go, namespace):
    # select terms in the namespace
    name_dict = {'bp': 'biological_process', 'mf': 'molecular_function', 'cc': 'cellular_component'}
    new_terms = []
    for t_id in terms:
        if go[t_id]['namespace'] == name_dict[namespace]:
            new_terms.append(t_id)
    return new_terms


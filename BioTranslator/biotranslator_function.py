import torch
import os
import json
import random
import logging
import collections
from tqdm import tqdm
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig
import scipy.stats
import numpy as np
from .emb_loaders import vector_encoder, sequence_encoder, graph_encoder
from .text_encoder import TestOntologyDataset, TrainOntologyDataset, NeuralNetwork, get_data, train
from .config import config
from .utils import load, save_obj
from .non_text_encoder import GraphTranslator, SeqTranslator, VecTranslator

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def setup_config(data_type='Sequence', **kwargs):
    args_names = {
        'Sequence': [
            'seq_dataset',
            'seq_task',
            'seq_max_length',
        ],
        'Vector': [
            'vec_dataset',
            'vec_eval_dataset',
            'vec_task',
            'vec_ontology_repo'
        ],
        'Graph': [
            'graph_pathway_dataset',
            'graph_dataset',
            'graph_excludes',
            'graph_max_length',
        ],
        'general': [
            'method',
            'data_repo',
            'encoder_path',
            'save_path',
            'emb_path',
            'working_space',
            'hidden_dim',
            'features',
            'lr',
            'epoch',
            'batch_size',
            'gpu_ids',
        ]
    }
    args_need = args_names[data_type].extend(args_names['general'])
    model_args = {k: kwargs[k] for k in args_need}
    return config(data_type, model_args)


def train_text_encoder(data_dir: str, save_dir: str):
    """Fine-tune the PubMedBert on 225 Ontologies, except cl and go
    Parameters
    ----------
    data_dir
        the Ontologies dataset
    save_dir
        where you save the model
    """
    save_path = f'{save_dir}/text_encoder.pth'
    print("Using {} device".format(device))
    bert_name = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
    tokenizer = AutoTokenizer.from_pretrained(bert_name)
    config = AutoConfig.from_pretrained(bert_name)
    config.attention_probs_dropout_prob = 0.3
    config.hidden_dropout_prob = 0.3
    output_way = 'pooler'
    assert output_way in ['pooler', 'cls', 'avg']

    lr = 1e-5
    batch_size = 16
    max_len = 256
    print(f'Batch Size: {batch_size}, Max Length: {max_len}')

    model = NeuralNetwork(bert_name, output_way, config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    train_texts, test_texts = get_data(data_dir)

    train_data = TrainOntologyDataset(train_texts, tokenizer, max_len)
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    test_data = TestOntologyDataset(test_texts, tokenizer, max_len)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    epochs = 1
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, test_dataloader, model, optimizer, save_path, device=device)
    print("Train Done!")


def get_ontology_embeddings(cfg):
    """Get the textual description embeddings of Gene Ontology or Cell Ontology terms

    Parameters
    ----------

    model_path:
        where you save the fine-tuned text encoder
    data_dir:
        the GO or CL data
    save_dir:
        where you save the term embeddings

    Returns
    -------
    ont_embeddings:
        the ontology embeddings
    """

    '''
    This function uses the BioTranslator Text Encoder to embed the Gene Ontology terms
    :param cfg:
    :return:
    '''
    bert_name = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
    go_data = load(cfg.go_file)
    go_embeddings = collections.OrderedDict()
    go_classes = list(go_data.keys())
    texts = []
    for i in tqdm(range(len(go_classes))):
        with torch.no_grad():
            texts.append(go_data[go_classes[i]]['name'] + '. ' + go_data[go_classes[i]]['def'][0])

    tokenizer = AutoTokenizer.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')
    model = NeuralNetwork('None', 'cls', bert_name)
    model.load_state_dict(torch.load(cfg.encoder_path))
    model = model.to('cuda')
    model.eval()

    with torch.no_grad():
        for i in tqdm(range(len(go_classes))):
            text = texts[i]
            inputs = tokenizer(text, return_tensors='pt').to('cuda')
            if len(cfg.gpu_ids) > 0:
                inputs = inputs.to('cuda')
            sents_len = min(inputs['input_ids'].size(1), 512)
            input_ids = inputs['input_ids'][0, 0: sents_len].view(len(inputs['input_ids']), -1).to('cuda')
            attention_mask = inputs['attention_mask'][0, 0: sents_len].view(len(inputs['attention_mask']), -1).to(
                'cuda')
            token_type_ids = inputs['token_type_ids'][0, 0: sents_len].view(len(inputs['token_type_ids']), -1).to(
                'cuda')

            pred = model(input_ids, attention_mask, token_type_ids)
            go_embeddings[go_classes[i]] = np.asarray(pred.cpu()).reshape([-1, 768])
        save_obj(go_embeddings, cfg.emb_path + cfg.emb_name)


class SequenceEncoder(SeqTranslator):
    """
    Embed the sequence features into the low dimensional space
    """
    def __init__(self, cfg):
        super(SequenceEncoder, self).__init__(cfg)


class VectorEncoder(VecTranslator):
    """
    Embed the Vector features (gene expression) into the low dimensional space
    """

    def __init__(self, cfg):
        super(VectorEncoder, self).__init__(cfg)


class GraphEncoder(GraphTranslator):
    """
    Embed the Graph features into the low dimensional space
    """

    def __init__(self, cfg):
        super(GraphEncoder, self).__init__(cfg)


def train_biotranslator(type2cfg: dict):
    """Train the BioTranslator

    Parameters
    ----------
    type2cfg:
        {
        'Sequence': config1
        'Vector':, config2,
        'Graph', config3,
        ...
        }

    data_dir
        Input data path.
    data_types
        ['Sequence', 'Vector', 'Graph', ...]
    cfgs
        [config1, config2, config3, ...]
    text_emb
        Text embedding path.
    model_path
        Where you save the model.
    lr
        Learning rate.
    epoch

    batch_size

    Return
    ------
    EncoderList
        List of non-text data encoders
    """
    EncoderList = []

    for tp in type2cfg.keys():
        if tp == 'Sequence':
            EncoderList.append(SequenceEncoder(type2cfg[tp]))
        if tp == 'Vector':
            EncoderList.append(VectorEncoder(type2cfg[tp]))
        if tp == 'Graph':
            EncoderList.append(GraphEncoder(type2cfg[tp]))


def test_biotranslator(data_dir, data_types, model_path, text_emb):
    """
    Annotate the proteins with textual description embeddings
    :param data_dir:
    :param model_path:
    :return:
    """
    raise NotImplementedError

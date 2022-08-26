import os
import json
import torch
import random
import logging
import collections
import scipy.stats
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from .config import BioConfig
from .loader import BioLoader
from .trainer import build_trainer
from .utils import load, save_obj
from torch.nn import functional as F
from .biotranslator import BioTranslator
from .utils import NeuralNetwork as nn_bert
from torch.utils.data import Dataset, DataLoader
from .text_encoder import NeuralNetwork as nn_config
from transformers import AutoTokenizer, AutoModel, AutoConfig
from .text_encoder import TestOntologyDataset, TrainOntologyDataset, NeuralNetwork, get_data, train

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def setup_config(config, data_type='seq'):
    args_names = dict(
        seq=[
            'task', 'max_length', 'features'
        ],
        vec=[
            'task', 'eval_dataset', 'vec_ontology_repo'
        ],
        graph=[
            'max_length', 'eval_dataset', 'graph_excludes', 'features'
        ],
        general=[
            'data_repo', 'dataset', 'encoder_path', 'rst_dir', 'emb_dir',
            'ws_dir', 'hidden_dim', 'lr', 'epoch', 'batch_size', 'gpu_ids',
        ]
    )
    args_need = args_names[data_type]
    args_need.extend(list(args_names['general']))
    model_args = {k: config[k] for k in args_need}
    print(model_args)
    return BioConfig(data_type, model_args)


def train_text_encoder(data_dir: str, save_path: str):
    """Fine-tune the PubMedBert on 225 Ontologies, except cl and go
    Parameters
    ----------
    data_dir
        the Ontologies dataset
    save_path
        where you save the model
    """
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

    model = nn_config(bert_name, output_way, config).to(device)
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


def get_ontology_embeddings(model_path, data_dir, cfg):
    """Get the textual description embeddings of Gene Ontology or Cell Ontology terms

    Parameters
    ----------
    model_path:
        where you save the fine-tuned text encoder
    data_dir:
        the GO or CL data
    cfg:
        config

    Returns
    -------
    ont_embeddings:
        the ontology embeddings
    """

    bert_name = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
    go_file = data_dir + 'go.obo'
    go_data = load(go_file)
    go_embeddings = collections.OrderedDict()
    go_classes = list(go_data.keys())
    texts = []
    for i in tqdm(range(len(go_classes))):
        with torch.no_grad():
            texts.append(go_data[go_classes[i]]['name'] + '. ' + go_data[go_classes[i]]['def'][0])

    tokenizer = AutoTokenizer.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')
    model = nn_bert('None', 'cls', bert_name)
    model.load_state_dict(torch.load(model_path))
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
        save_obj(go_embeddings, cfg.emb_path)
    return go_embeddings


def train_biotranslator(cfgs):
    """Train the BioTranslator

    Parameters
    ----------
    cfgs: [config1, config2, config3, ...]

    Returns
    ------
    encoder_list
        List of trained translator
    """
    trainer_dict = collections.OrderedDict()
    encoders = collections.OrderedDict()
    # We may only train one data type each time in this method
    for cfg in cfgs:
        if cfg.tp in ['graph', 'seq', 'vec']:
            files = BioLoader(cfg)
        else:
            logging.info('Data type is not supported yet.')
            raise NotImplementedError
        trainer_dict[cfg.tp] = (build_trainer(files=files, cfg=cfg), files, cfg)
    for key, trainer_tup in trainer_dict.items():
        trainer = trainer_tup[0]
        trainer.train(trainer_tup[1], trainer_tup[2])
        encoders[key] = trainer
    return encoders


def test_biotranslator(data_dir, anno_data, cfg, encoder, task):
    """
    Annotate the proteins with textual description embeddings

    Parameters
    ----------
    data_dir: Input data path.
    anno_data: data needs to be annotated
    cfg: config
    encoder: biotranslator encoder
    task: task name

    Returns
    -------
    """
    files = BioLoader(cfg)
    anno = encoder.annotate(files, cfg, data_dir, anno_data, task)
    return anno

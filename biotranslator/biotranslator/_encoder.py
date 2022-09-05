# -*- coding: utf-8 -*-
import itertools
import numpy as np
from torch import nn
from torch.nn import init
from transformers import AutoTokenizer, AutoModel
import torch
import collections
from ..utils import init_weights


class BioDataEncoder(nn.Module):
    def __init__(self,
                 feature=None,
                 hidden_dim=1000,
                 seq_input_nc=4,
                 seq_in_nc=512,
                 seq_max_kernels=129,
                 seq_length=2000,
                 network_dim=800,
                 description_dim=768,
                 expression_dim=1000,
                 drop_out=0.01,
                 text_dim=768):
        super(BioDataEncoder, self).__init__()
        if feature is None:
            feature = ['seqs', 'network', 'description', 'expression']
        self.feature = feature
        self.text_dim = text_dim
        if 'seqs' in self.feature:
            self.para_conv, self.para_pooling = [], []
            kernels = range(8, seq_max_kernels, 8)
            self.kernel_num = len(kernels)
            for i in range(len(kernels)):
                exec(
                    "self.conv1d_{} = nn.Conv1d(in_channels=seq_input_nc, out_channels=seq_in_nc, kernel_size=kernels[i], padding=0, stride=1)".format(
                        i))
                exec("self.pool1d_{} = nn.MaxPool1d(kernel_size=seq_length - kernels[i] + 1, stride=1)".format(i))
            self.fc_seq = [nn.Linear(len(kernels) * seq_in_nc, hidden_dim), nn.LeakyReLU(inplace=True)]
            self.fc_seq = nn.Sequential(*self.fc_seq)
        if 'description' in self.feature:
            self.fc_description = [nn.Linear(description_dim, hidden_dim), nn.LeakyReLU(inplace=True)]
            self.fc_description = nn.Sequential(*self.fc_description)
        if 'network' in self.feature:
            self.fc_network = [nn.Linear(network_dim, hidden_dim), nn.LeakyReLU(inplace=True)]
            self.fc_network = nn.Sequential(*self.fc_network)
        if 'expression' in self.feature:
            self.fc_expr = [nn.Linear(expression_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(p=drop_out)]
            self.fc_expr = nn.Sequential(*self.fc_expr)
        self.cat2emb = nn.Linear(len(self.feature) * hidden_dim, text_dim)

    def forward(self, x=None, x_description=None, x_vector=None, x_expr=None):
        x_list = []
        features = collections.OrderedDict()
        if 'seqs' in self.feature:
            for i in range(self.kernel_num):
                exec("x_i = self.conv1d_{}(x)".format(i))
                exec("x_i = self.pool1d_{}(x_i)".format(i))
                exec("x_list.append(torch.squeeze(x_i).reshape([x.size(0), -1]))")
            features['seqs'] = self.fc_seq(torch.cat(tuple(x_list), dim=1))
        if 'description' in self.feature:
            features['description'] = self.fc_description(x_description)
        if 'network' in self.feature:
            features['network'] = self.fc_network(x_vector)
        if 'expression' in self.feature:
            features['expression'] = self.fc_expr(x_expr)
        for i in range(len(self.feature)):
            if i == 0:
                x_enc = features[self.feature[0]]
            else:
                x_enc = torch.cat((x_enc, features[self.feature[i]]), dim=1)
        # x_enc = torch.nn.functional.normalize(x_cat, p=2, dim=1)
        return self.cat2emb(x_enc)


class BioTranslator(nn.Module):
    def __init__(self, cfg):
        super(BioTranslator, self).__init__()
        self.loss_func = torch.nn.BCELoss()
        if cfg.tp in ['seq', 'graph']:
            kwargs = dict(feature=cfg.features,
                          hidden_dim=cfg.hidden_dim,
                          seq_input_nc=cfg.seq_input_nc,
                          seq_in_nc=cfg.seq_in_nc,
                          seq_max_kernels=cfg.seq_max_kernels,
                          network_dim=cfg.network_dim,
                          seq_length=cfg.max_length,
                          text_dim=cfg.term_enc_dim)
        elif cfg.tp == 'vec':
            kwargs = dict(feature=cfg.features,
                           hidden_dim=cfg.hidden_dim,
                           expression_dim=cfg.expr_dim,
                           drop_out=cfg.drop_out,
                           text_dim=cfg.term_enc_dim)
        else:
            raise NotImplementedError
        self.data_encoder = BioDataEncoder(**kwargs)

        if cfg.tp == "seq" or cfg.tp == "graph":
            self.activation = torch.nn.Sigmoid()
            # self.text_encoder = torch.load(model_config.text_encoder)
            self.temperature = torch.tensor(0.07, requires_grad=True)
        elif cfg.tp == "vec":
            self.activation = torch.nn.Softmax(dim=1)
        else:
            raise NotImplementedError

        self.text_dim = cfg.term_enc_dim
        init_weights(self.data_encoder, init_type='xavier')
        if len(cfg.gpu_ids) > 0:
            self.data_encoder = self.data_encoder.to('cuda')
            self.temperature = self.temperature.to('cuda')
            self.activation = self.activation.to('cuda')

    def forward(self, data_type=None, input_seq=None, input_description=None, input_vector=None, input_expr=None, texts=None):
        # get textual description encodings
        text_encodings = texts.permute(1, 0)
        # get biology instance encodings
        if data_type == "seq" or data_type == "graph":
            data_encodings = self.data_encoder(x=input_seq,
                                               x_description=input_description,
                                               x_vector=input_vector)
        elif data_type == "vec":
            data_encodings = self.data_encoder(x_expr=input_expr)
        else:
            raise NotImplementedError

        # compute logits
        logits = torch.matmul(data_encodings, text_encodings)
        return self.activation(logits)

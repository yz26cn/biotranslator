# -*- coding: utf-8 -*-
import torch
import collections
from torch import nn
from torch.nn import init
from .BioConfig import BioConfig
from .BioUtils import init_weights

class BioDataEncoder(nn.Module):
    """BioTranslator Data Encoder"""
    def __init__(self,
                 feature=['seqs', 'network', 'description', 'expression'],
                 hidden_dim=1000,
                 seq_input_nc=4,
                 seq_in_nc=512,
                 seq_max_kernels=129,
                 seq_length=2000,
                 network_dim=800,
                 description_dim=768,
                 text_dim=768):
        """

        :param seq_input_nc:
        :param seq_in_nc:
        :param seq_max_kernels:
        :param seq_length:
        """
        super(BioDataEncoder, self).__init__()
        self.feature = feature
        self.text_dim = text_dim
        self.seqs_in_feature = 'seqs' in self.feature
        self.desc_in_feature = 'description' in self.feature
        self.nets_in_feature = 'network' in self.feature
        if self.seqs_in_feature:
            self.para_conv, self.para_pooling = [], []
            kernels = range(8, seq_max_kernels, 8)
            self.kernel_num = len(kernels)
            for i in range(len(kernels)):
                exec(
                    f"self.conv1d_{i} = nn.Conv1d(in_channels=seq_input_nc, out_channels=seq_in_nc, kernel_size=kernels[i], padding=0, stride=1)")
                exec(f"self.pool1d_{i} = nn.MaxPool1d(kernel_size=seq_length - kernels[i] + 1, stride=1)")
            fc_seq = [nn.Linear(len(kernels) * seq_in_nc, hidden_dim), nn.LeakyReLU(inplace=True)]
            self.fc_seq = nn.Sequential(*fc_seq)
        if self.desc_in_feature:
            fc_description = [nn.Linear(description_dim, hidden_dim), nn.LeakyReLU(inplace=True)]
            self.fc_description = nn.Sequential(*fc_description)
        if self.nets_in_feature:
            fc_network = [nn.Linear(network_dim, hidden_dim), nn.LeakyReLU(inplace=True)]
            self.fc_network = nn.Sequential(*fc_network)
        self.cat2emb = nn.Linear(len(self.feature) * hidden_dim, text_dim)

    def forward(self, x=None, x_description=None, x_vector=None):
        x_list = []
        features = collections.OrderedDict()
        if self.seqs_in_feature:
            for i in range(self.kernel_num):
                exec(f"x_i = self.conv1d_{i}(x)")
                exec(f"x_i = self.pool1d_{i}(x_i)")
                exec(f"x_list.append(torch.squeeze(x_i).reshape([x.size(0), -1]))")
            features['seqs'] = self.fc_seq(torch.cat(tuple(x_list), dim=1))
        if self.desc_in_feature:
            features['description'] = self.fc_description(x_description)
        if self.nets_in_feature:
            features['network'] = self.fc_network(x_vector)
        for i in range(len(self.feature)):
            if i == 0:
                x_enc = features[self.feature[0]]
            else:
                x_enc = torch.cat((x_enc, features[self.feature[i]]), dim=1)
        return self.cat2emb(x_enc)


class BioTranslator(nn.Module):
    """BioTranslator Model"""
    def __init__(self, cfg: BioConfig):
        super(BioTranslator, self).__init__()
        self.loss_func = torch.nn.BCELoss()
        self.data_encoder = BioDataEncoder(feature=cfg.features,
                                           hidden_dim=cfg.hidden_dim,
                                           seq_input_nc=cfg.seq_input_nc,
                                           seq_in_nc=cfg.seq_in_nc,
                                           seq_max_kernels=cfg.seq_max_kernels,
                                           seq_length=cfg.max_length,
                                           network_dim=cfg.network_dim,
                                           text_dim=cfg.term_enc_dim)
        self.activation = torch.nn.Sigmoid()
        self.temperature = torch.tensor(0.07, requires_grad=True)
        self.text_dim = cfg.term_enc_dim
        self.model = init_weights(self.model, init_type='xavier')
        if len(cfg.gpu_ids) > 0:
            self.data_encoder = self.data_encoder.cuda()
            self.temperature = self.temperature.cuda()
            self.activation = self.activation.cuda()

    def forward(self, input_seq, input_description, input_vector, texts):
        # get textual description encodings
        text_encodings = texts.permute(1, 0)
        # get biology instance encodings
        data_encodings = self.data_encoder(input_seq, input_description, input_vector)
        # compute logits
        logits = torch.matmul(data_encodings, text_encodings)
        return self.activation(logits)


class BaseDataEncoder(BioDataEncoder):
    """Base DataEncoder"""
    def __init__(self,
                 feature=['seqs', 'network', 'description', 'expression'],
                 hidden_dim=1000,
                 seq_input_nc=4,
                 seq_in_nc=512,
                 seq_max_kernels=129,
                 seq_length=2000,
                 network_dim=800,
                 description_dim=768,
                 text_dim=768):
        """
        :param seq_input_nc:
        :param seq_in_nc:
        :param seq_max_kernels:
        :param seq_length:
        """
        super(BaseDataEncoder, self).__init__()
        self.feature = feature
        self.text_dim = text_dim
        self.seqs_in_feature = 'seqs' in self.feature
        self.desc_in_feature = 'description' in self.feature
        self.nets_in_feature = 'network' in self.feature
        if self.seqs_in_feature:
            self.para_conv, self.para_pooling = [], []
            kernels = range(8, seq_max_kernels, 8)
            self.kernel_num = len(kernels)
            for i in range(len(kernels)):
                exec(
                    f"self.conv1d_{i} = nn.Conv1d(in_channels=seq_input_nc, out_channels=seq_in_nc, "
                    f"kernel_size=kernels[i], padding=0, stride=1)")
                exec(f"self.pool1d_{i} = nn.MaxPool1d(kernel_size=seq_length - kernels[i] + 1, stride=1)")
            fc_seq = [nn.Linear(len(kernels) * seq_in_nc, hidden_dim), nn.ReLU(inplace=True)]
            self.fc_seq = nn.Sequential(*fc_seq)
        if self.desc_in_feature:
            fc_description = [nn.Linear(description_dim, hidden_dim), nn.ReLU(inplace=True)]
            self.fc_description = nn.Sequential(*fc_description)
        if self.nets_in_feature:
            fc_network = [nn.Linear(network_dim, hidden_dim), nn.ReLU(inplace=True)]
            self.fc_network = nn.Sequential(*fc_network)
        self.cat2emb = nn.Linear(len(self.feature) * hidden_dim, text_dim)


class BaseModel(BioTranslator):
    def __init__(self, cfg):
        super(BaseModel, self).__init__(cfg)


class DGPDataEncoder(nn.Module):
    """Our implementation of  the DeepGOPlus Model"""

    def __init__(self,
                 seq_input_nc=4,
                 hidden_dim=1000,
                 seq_in_nc=512,
                 seq_max_kernels=129,
                 seq_length=2000):
        """
        :param

        """
        super(DGPDataEncoder, self).__init__()
        self.para_conv, self.para_pooling = [], []
        kernels = range(8, seq_max_kernels, 8)
        self.kernel_num = len(kernels)
        for i in range(len(kernels)):
            exec(
                f"self.conv1d_{i} = nn.Conv1d(in_channels=seq_input_nc, out_channels=seq_in_nc, kernel_size=kernels[i], padding=0, stride=1)")
            exec(f"self.pool1d_{i} = nn.MaxPool1d(kernel_size=seq_length - kernels[i] + 1, stride=1)")
        self.fc_seq = [nn.Linear(len(kernels) * seq_in_nc, hidden_dim)]
        self.fc_seq = nn.Sequential(*self.fc_seq)

    def forward(self, x):
        x_list = []
        for i in range(self.kernel_num):
            exec(f"x_i = self.conv1d_{i}(x)")
            exec(f"x_i = self.pool1d_{i}(x_i)")
            exec(f"x_list.append(torch.squeeze(x_i))")
        x_enc = torch.cat(tuple(x_list), dim=1)
        x_enc = self.fc_seq(x_enc)
        return x_enc


class DeepGOPlus(nn.Module):
    """
    Our Implementation of the DeepGOPlus Model
    """

    def __init__(self, cfg, hidden_dim=1000):
        super(DeepGOPlus, self).__init__()
        self.loss_func = torch.nn.BCELoss()
        self.model = DGPDataEncoder(seq_input_nc=cfg.seq_input_nc,
                                     hidden_dim=hidden_dim,
                                     seq_in_nc=cfg.seq_input_nc,
                                     seq_max_kernels=cfg.seq_max_kernels,
                                     seq_length=cfg.max_length)
        self.activation = torch.nn.Sigmoid()
        self.model = init_weights(self.model, init_type='xavier')
        if len(cfg.gpu_ids) > 0:
            self.model = self.model.cuda()
            self.activation = self.activation.cuda()

    def forward(self, input_seq):
        # get biology instance encodings
        return self.model(input_seq)

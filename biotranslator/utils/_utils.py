import os
import sys
import torch
import logging
import scipy
import time
import pickle
import collections
import numpy as np
import pandas as pd
import warnings
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
from scipy import sparse
from scipy import spatial
from anndata import read_h5ad
from collections import Counter
from scipy.sparse.csr import csr_matrix
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.csgraph._shortest_path import shortest_path
# from sklearn.utils.graph_shortest_path import graph_shortest_path
from sklearn.metrics import auc, average_precision_score, precision_recall_curve

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


def get_ontology_embeddings(cfg):
    """Get the textual description embeddings of Gene Ontology (GO) or Cell Ontology (CO) terms

    Parameters
    ----------
    cfg:
        config

    Returns
    -------
    ont_embeddings:
        the ontology embeddings
    """

    """This function uses the BioTranslator Text Encoder to embed the Gene Ontology terms
    :param cfg:
    :return:
    """
    embeddings = collections.OrderedDict()
    bert_name = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
    tokenizer = AutoTokenizer.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')
    model = NeuralNetwork('None', 'cls', bert_name)
    model.load_state_dict(torch.load(cfg.encoder_path))
    model = model.to('cuda')
    model.eval()

    if cfg.tp in ['seq', 'graph']:
        if cfg.task not in ["single-dataset", "cross-dataset"]:
            # go embedding
            data = load(cfg.go_file)
            classes = list(data.keys())
            texts = []
            for i in tqdm(range(len(classes))):
                with torch.no_grad():
                    texts.append(data[classes[i]]['name'] + '. ' + data[classes[i]]['def'][0])
        else:
            # co embedding
            text = load_co_text(cfg.ontology_repo + 'cl.obo')
            classes = list(text.keys())
            texts = []
            for i in tqdm(range(len(classes))):
                with torch.no_grad():
                    texts.append(text[classes[i]]['text'])
    elif cfg.tp == 'cell':
        # co embedding
        text = load_co_text(cfg.ontology_repo + 'cl.obo')
        classes = list(text.keys())
        texts = []
        for i in tqdm(range(len(classes))):
            with torch.no_grad():
                texts.append(text[classes[i]]['text'])

    with torch.no_grad():
        for i in tqdm(range(len(classes))):
            text = texts[i]
            inputs = tokenizer(text, return_tensors='pt').to('cuda')
            if len(cfg.gpu_ids) > 0:
                inputs = inputs.to('cuda')
            sents_len = min(inputs['input_ids'].size(1), 512)
            input_ids = inputs['input_ids'][0, 0: sents_len].view(len(inputs['input_ids']), -1).to('cuda')
            attention_mask = inputs['attention_mask'][0, 0: sents_len].view(len(inputs['attention_mask']),
                                                                            -1).to(
                'cuda')
            token_type_ids = inputs['token_type_ids'][0, 0: sents_len].view(len(inputs['token_type_ids']),
                                                                            -1).to(
                'cuda')

            pred = model(input_ids, attention_mask, token_type_ids)
            embeddings[classes[i]] = np.asarray(pred.cpu()).reshape([-1, 768])
        save_obj(embeddings, cfg.emb_path + cfg.emb_name)
    return embeddings


def get_anchestors(go, term_id):
    if term_id not in go:
        return []
    term_set = []
    q = deque()
    q.append(term_id)
    while (len(q) > 0):
        t_id = q.popleft()
        if t_id not in term_set:
            term_set.append(t_id)
            for parent_id in go[t_id]['is_a']:
                if parent_id in go:
                    q.append(parent_id)
    return term_set


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


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


def renorm(X):
    Y = X.copy()
    Y = Y.astype(float)
    ngene, nsample = Y.shape
    s = np.sum(Y, axis=0)
    # print s.shape()
    for i in range(nsample):
        if s[i] == 0:
            s[i] = 1
            if i < ngene:
                Y[i, i] = 1
            else:
                for j in range(ngene):
                    Y[j, i] = 1. / ngene
        Y[:, i] = Y[:, i] / s[i]
    return Y


def DCA_vector(Q, dim):
    nnode = Q.shape[0]
    alpha = 1. / (nnode ** 2)
    Q = np.log(Q + alpha) - np.log(alpha)

    # Q = Q * Q';
    [U, S, V] = svds(Q, dim)
    S = np.diag(S)
    X = np.dot(U, np.sqrt(S))
    Y = np.dot(np.sqrt(S), V)
    Y = np.transpose(Y)
    return X, U, S, V, Y


class Dataset(object):

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __add__(self, other):
        return ConcatDataset([self, other])


def one_hot(seq, start=0, max_len=2000):
    '''
    One-Hot encodings of protein sequences,
    this function was copied from DeepGOPlus paper
    :param seq:
    :param start:
    :param max_len:
    :return:
    '''
    AALETTER = [
        'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
        'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    AAINDEX = dict()
    for i in range(len(AALETTER)):
        AAINDEX[AALETTER[i]] = i + 1
    onehot = np.zeros((21, max_len), dtype=np.int32)
    l = min(max_len, len(seq))
    for i in range(start, start + l):
        onehot[AAINDEX.get(seq[i - start], 0), i] = 1
    onehot[0, 0:start] = 1
    onehot[0, start + l:] = 1
    return onehot


class proteinData(Dataset):
    '''
    The dataset of protein
    '''

    def __init__(self, data_df, terms, prot_vector, prot_description, gpu_ids='0'):
        self.p_ids = []
        self.pSeq = []
        self.label = []
        self.vector = []
        self.description = []
        self.gpu_ids = gpu_ids
        sequences = list(data_df['sequences'])
        prot_ids = list(data_df['proteins'])
        annots = list(data_df['annotations'])
        for i in range(len(annots)):
            annots[i] = list(annots[i])
        for i in range(data_df.shape[0]):
            seqT, annT, protT = sequences[i], annots[i], prot_ids[i]
            labelT = np.zeros([len(terms), 1])
            for j in range(len(annT)):
                if annT[j] in terms.keys():
                    labelT[terms[annT[j]]] = 1
            self.p_ids.append(protT)
            self.pSeq.append(one_hot(seqT))
            self.label.append(labelT)
            self.vector.append(prot_vector[protT])
            self.description.append((prot_description[protT]))

    def __getitem__(self, item):
        in_seq, label = transforms.ToTensor()(self.pSeq[item]), transforms.ToTensor()(self.label[item])
        description = torch.from_numpy(self.description[item])
        vector = torch.from_numpy(self.vector[item])
        if len(self.gpu_ids) > 0:
            return {'proteins': self.p_ids[item],
                    'prot_seq': squeeze(in_seq).float().cuda(),
                    'prot_description': squeeze(description).float().cuda(),
                    'prot_network': squeeze(vector).float().cuda(),
                    'label': squeeze(label).float().cuda()}
        else:
            return {'proteins': self.p_ids[item],
                    'prot_seq': squeeze(in_seq).float(),
                    'prot_description': squeeze(description).float(),
                    'prot_network': squeeze(vector).float(),
                    'label': squeeze(label).float()}

    def __len__(self):
        return len(self.pSeq)


class NeuralNetwork(nn.Module):
    def __init__(self, model_path, output_way, bert_name):
        super(NeuralNetwork, self).__init__()
        self.bert = AutoModel.from_pretrained(bert_name)
        self.output_way = output_way

    def forward(self, input_ids, attention_mask, token_type_ids):
        x1 = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        if self.output_way == 'cls':
            output = x1.last_hidden_state[:, 0]
        elif self.output_way == 'pooler':
            output = x1.pooler_output
        return output


def term2preds_label(preds, label, terms, terms2id):
    new_preds, new_label = [], []
    for t_id in terms:
        new_preds.append(preds[:, terms2id[t_id]].reshape((-1, 1)))
        new_label.append(label[:, terms2id[t_id]].reshape((-1, 1)))
    new_preds = np.concatenate(new_preds, axis=1)
    new_label = np.concatenate(new_label, axis=1)
    return new_preds, new_label


def organize_workingspace(workingspace, task=None):
    """
    Make sure that the working space include the zero shot folder, few shot folder,
    model folder, training log folder and results folder
    :param workingspace:
    :return:
    """
    task_path = workingspace + task
    cache_path = task_path + '/cache'
    model_path = task_path + '/model'
    logger_path = task_path + '/log'
    results_path = task_path + '/results'
    if not os.path.exists(workingspace):
        os.mkdir(workingspace)
        print('Warning: We created the working space: {}'.format(workingspace))
    if not os.path.exists(task_path):
        os.mkdir(task_path)
    if not os.path.exists(cache_path):
        os.mkdir(cache_path)
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    if not os.path.exists(logger_path):
        os.mkdir(logger_path)
    if not os.path.exists(results_path):
        os.mkdir(results_path)


def get_namespace_terms(terms, go, namespace):
    # select terms in the namespace
    name_dict = {'bp': 'biological_process', 'mf': 'molecular_function', 'cc': 'cellular_component'}
    new_terms = []
    for t_id in terms:
        if go[t_id]['namespace'] == name_dict[namespace]:
            new_terms.append(t_id)
    return new_terms


def get_few_shot_namespace_terms(term_count, go, namespace, n=10):
    # select terms in namespace with less than n training sample
    name_dict = {'bp': 'biological_process', 'mf': 'molecular_function', 'cc': 'cellular_component'}
    new_terms = []
    for t_id in term_count.keys():
        if go[t_id]['namespace'] == name_dict[namespace]:
            if term_count[t_id] <= n:
                new_terms.append(t_id)
    return new_terms


def term_training_numbers(val_data, train_data):
    '''
    This function calculates how many training samples exists in the
    :param val_data:
    :param train_data:
    :return:
    '''
    term2number = collections.OrderedDict()
    for annt_val in list(val_data['annotations']):
        for a_id in annt_val:
            term2number[a_id] = 0
    training_annt = list(train_data['annotations'])
    for annt in training_annt:
        annt = list(set(annt))
        for a_id in annt:
            if a_id not in term2number.keys():
                continue
            term2number[a_id] += 1
    return term2number


def compute_blast_preds(diamond_scores,
                        test,
                        train_data):
    '''
    This function computes the predictions of blast results.
    The codes here are borrowed from the DeepGOPlus paper
    :param diamond_scores:
    :param test:
    :param train_data:
    :param is_load:
    :param save_path:
    :return:
    '''
    blast_preds = collections.OrderedDict()
    print('Diamond preds')
    annotation_dict = collections.OrderedDict()
    for i in train_data.index:
        prot_id = train_data.loc[i]['proteins']
        annts = set(train_data.loc[i]['annotations'])
        if prot_id not in annotation_dict.keys():
            annotation_dict[prot_id] = collections.OrderedDict()
            for ann_id in annts:
                annotation_dict[prot_id][ann_id] = None
        else:
            for ann_id in annts:
                annotation_dict[prot_id][ann_id] = None

    for prot_id in tqdm(test.proteins):
        annots = {}

        # BlastKNN
        sim = collections.OrderedDict()
        if prot_id in diamond_scores.keys():
            sim_prots = diamond_scores[prot_id]
            allgos = set()
            total_score = 0.0
            for p_id, score in sim_prots.items():
                allgos |= set(annotation_dict[p_id].keys())
                total_score += score
            allgos = set(allgos)
            for go_id in allgos:
                s = 0.0
                for p_id in sim_prots.keys():
                    score = sim_prots[p_id]
                    if go_id in annotation_dict[p_id].keys():
                        s += score
                sim[go_id] = s / total_score
        blast_preds[prot_id] = sim
    return blast_preds


def extract_terms_from_dataset(dataset):
    id = 0
    terms2id = collections.OrderedDict()
    id2terms = collections.OrderedDict()
    for i in dataset.index:
        annts = dataset.iloc[i]['annotations']
        for annt in annts:
            if annt not in terms2id:
                terms2id[annt] = id
                id2terms[id] = annt
                id += 1
    return terms2id, id2terms


def coef_vec(A, B):
    coefs_array = []
    for i in range(np.size(A, 0)):
        a1 = A[i, :]
        b1 = B[i, :]
        coefs = np.corrcoef(a1, b1)[0, 1]
        coefs_array.append(coefs)
    return np.asarray(coefs_array)


def sample_edges(st=0, ed=1000, n=100, exclude_edges=[]):
    samples = []
    while True:
        nodes = np.arange(st, ed)
        np.random.shuffle(nodes)
        edge = list(nodes[0: 2])
        if edge not in exclude_edges and edge not in samples:
            samples.append(edge)
        if len(samples) == n:
            break
    return np.asarray(samples)


def edge_probability(encodings_i, encodings_j, text_encodings_p):
    s1 = np.dot(encodings_i, np.transpose(text_encodings_p))
    s1 = 1 / (1 + np.exp(-s1))
    s2 = np.dot(encodings_j, np.transpose(text_encodings_p))
    s2 = 1 / (1 + np.exp(-s2))
    s3 = coef_vec(encodings_i, encodings_j)
    score = s3 * (0.5 * s1.squeeze() + 0.5 * s2.squeeze())
    return score


def RandomWalkRestart(A, rst_prob, delta=1e-4, reset=None, max_iter=50, use_torch=False, return_torch=False):
    device = torch.device('cpu')
    if use_torch and torch.cuda.is_available():
        device = torch.device('cuda:0')
    nnode = A.shape[0]
    if reset is None:
        reset = np.eye(nnode)
    nsample, nnode = reset.shape
    # print nsample,nnode
    P = renorm(A)
    P = P.T
    norm_reset = renorm(reset.T)
    norm_reset = norm_reset.T
    if use_torch:
        norm_reset = torch.from_numpy(norm_reset).float().to(device)
        P = torch.from_numpy(P).float().to(device)
    Q = norm_reset

    for i in range(1, max_iter):
        if use_torch:
            Q_new = rst_prob * norm_reset + (1 - rst_prob) * torch.mm(Q, P)  # .as_numpy_array()
            delta = torch.norm(Q - Q_new, 2)
        else:
            Q_new = rst_prob * norm_reset + (1 - rst_prob) * np.dot(Q, P)  # .as_numpy_array()
            delta = np.linalg.norm(Q - Q_new, 'fro')
        Q = Q_new
        # print (i,Q)
        sys.stdout.flush()
        if delta < 1e-4:
            break
    if use_torch and not return_torch:
        Q = Q.cpu().numpy()
    return Q


def read_cell_type_nlp_network(nlp_emb_file, cell_type_network_file):
    cell_ontology_ids = set()
    fin = open(cell_type_network_file)
    co2co_graph = {}
    for line in fin:
        w = line.strip().split('\t')
        if w[0] not in co2co_graph:
            co2co_graph[w[0]] = set()
        co2co_graph[w[0]].add(w[1])
        cell_ontology_ids.add(w[0])
        cell_ontology_ids.add(w[1])
    fin.close()
    if nlp_emb_file is not None:
        fin = open(nlp_emb_file)
        co2vec_nlp = {}
        for line in fin:
            w = line.strip().split('\t')
            vec = []
            for i in range(1, len(w)):
                vec.append(float(w[i]))
            co2vec_nlp[w[0]] = np.array(vec)
        fin.close()
        co2co_nlp = {}
        for id1 in co2co_graph:
            co2co_nlp[id1] = {}
            for id2 in co2co_graph[id1]:
                sc = 1 - spatial.distance.cosine(co2vec_nlp[id1], co2vec_nlp[id2])
                co2co_nlp[id1][id2] = sc
    else:
        co2co_nlp = {}
        for id1 in co2co_graph:
            co2co_nlp[id1] = {}
            for id2 in co2co_graph[id1]:
                co2co_nlp[id1][id2] = 1.
        co2vec_nlp = {}
        for c in cell_ontology_ids:
            co2vec_nlp[c] = np.ones((10))
    return co2co_graph, co2co_nlp, co2vec_nlp, cell_ontology_ids


def subset_cell_type_nlp_network(co2co_graph, co2co_nlp, co2vec_nlp, cell_ontology_ids, subsets):
    new_co2co_graph, new_co2co_nlp, new_co2vec_nlp = {}, {}, {}
    new_co_ids = set()
    for co_i in subsets:
        if co_i in co2co_graph.keys():
            co_i_cncts = co2co_graph[co_i]
            new_cncts = set()
            for co_i_cnct in co_i_cncts:
                if co_i_cnct in subsets:
                    new_cncts.add(co_i_cnct)
            if len(new_cncts) > 0:
                new_co2co_graph[co_i] = new_cncts
        if co_i in co2co_nlp.keys():
            co_i_cncts = co2co_nlp[co_i]
            new_cncts = {}
            for co_i_cnct in co_i_cncts.keys():
                if co_i_cnct in subsets:
                    new_co_ids.add(co_i_cnct)
                    new_cncts[co_i_cnct] = co_i_cncts[co_i_cnct]
            if len(new_cncts.keys()) > 0:
                new_co_ids.add(co_i)
                new_co2co_nlp[co_i] = new_cncts
    for co_i in new_co_ids:
        if co_i in co2vec_nlp.keys():
            new_co2vec_nlp[co_i] = co2vec_nlp[co_i]
    return new_co2co_graph, new_co2co_nlp, new_co2vec_nlp, new_co_ids


def map_genes(test_X, test_genes, train_genes, num_batches=10, memory_saving_mode=False):
    """
    Takes in the test set features, test_X, and a list of all the genes in the training and test sets
    and returns new_test_X, which makes the gene indices of test_X correspond to the same ones from the
    training set.
    """
    ntest_cell = np.shape(test_X)[0]  # Number of cells in the test set
    ntrain_gene = len(train_genes)  # Number of genes in the train set
    genes = list(set(test_genes) & set(train_genes))  # Genes that are in both the train and test sets
    train_genes = list(train_genes)
    test_genes = list(test_genes)
    # print("Mapping genes with memory mode =", memory_saving_mode,"...")
    if not memory_saving_mode:
        new_test_x = np.zeros((ntest_cell, ntrain_gene))
        ind1 = []
        ind2 = []
        for i, g in enumerate(genes):
            ind1.append(train_genes.index(g))
            ind2.append(test_genes.index(g))
        ind1 = np.array(ind1)
        ind2 = np.array(ind2)
        new_test_x[:, ind1] = test_X[:, ind2]
    else:
        new_test_x = sparse.csr_matrix((ntest_cell, ntrain_gene))
        batch_size = int(len(genes) / num_batches)
        for i in range(num_batches):
            # print("\tbatch #",i)
            ind1 = []
            ind2 = []
            if i != num_batches - 1:
                batch = genes[i * batch_size: (i + 1) * batch_size]
            else:
                batch = genes[i * batch_size:]
            for g in batch:
                ind1.append(train_genes.index(g))
                ind2.append(test_genes.index(g))
            ind1 = np.array(ind1)
            ind2 = np.array(ind2)
            new_test_x[:, ind1] = test_X[:, ind2]
    return new_test_x


def process_expression(train_X, test_X, train_genes, test_genes):
    # this data process function is adapted from ACTINN, please check ACTINN for more information.
    test_X = map_genes(test_X, test_genes, train_genes)
    c2g = np.vstack([train_X, test_X])
    c2g = np.array(c2g, dtype=np.float64)
    c2g = c2g.T
    index = np.sum(c2g, axis=1) > 0
    c2g = c2g[index, :]
    train_genes = train_genes[index]
    c2g = np.divide(c2g, np.sum(c2g, axis=0, keepdims=True)) * 10000
    c2g = np.log2(c2g + 1)
    expr = np.sum(c2g, axis=1)
    # total_set = total_set[np.logical_and(expr >= np.percentile(expr, 1), expr <= np.percentile(expr, 99)),]
    index = np.logical_and(expr >= np.percentile(expr, 1), expr <= np.percentile(expr, 99))
    c2g = c2g[index,]
    train_genes = train_genes[index]
    # print (np.logical_and(expr >= np.percentile(expr, 1), expr <= np.percentile(expr, 99)))

    cv = np.std(c2g, axis=1) / np.mean(c2g, axis=1)
    index = np.logical_and(cv >= np.percentile(cv, 1), cv <= np.percentile(cv, 99))
    c2g = c2g[index,]
    train_genes = train_genes[index]
    c2g = c2g.T
    c2g_list_new = []
    index = 0
    for c in [train_X, test_X]:
        ncell = np.shape(c)[0]
        c2g_list_new.append(c2g[index:index + ncell, :])
        index = ncell
        assert (len(train_genes) == np.shape(c2g)[1])
    return c2g_list_new[0], c2g_list_new[1], train_genes


def mean_normalization(train_X_mean, test_X):
    test_X = np.log1p(test_X)
    test_X_mean = np.mean(test_X, axis=0)
    test_X = test_X - test_X_mean + train_X_mean
    return test_X


def get_gene_mapping(test_genes, train_genes):
    """
    Low-memory version of map_genes. Takes the list of genes used by the test set and the list of
    genes used in the training set. Provides a numpy array that maps the indices of the test genes
    to the indices of the training genes.
    For instance, if gene 'abcd' is at index 1 in train_genes, and at index 7 in test_genes, then
    the numpy array returned from this function will have a 1 at index 7.
    """
    genes = list(set(test_genes) & set(train_genes))  # Genes that are in both the train and test sets
    train_genes = list(train_genes)
    test_genes = list(test_genes)
    # print("Getting gene mapping...")

    gene_idx = np.arange(len(test_genes))
    for g in genes:
        train_idx = train_genes.index(g)
        test_idx = test_genes.index(g)
        gene_idx[train_idx] = test_idx

    return gene_idx


def graph_embedding_dca(A, i2l, mi=0, dim=20, unseen_l=None):
    nl = np.shape(A)[0]
    seen_ind = []
    unseen_ind = []
    for i in range(nl):
        if i2l[i] in unseen_l:
            unseen_ind.append(i)
        else:
            seen_ind.append(i)
    seen_ind = np.array(seen_ind)
    unseen_ind = np.array(unseen_ind)

    # if len(seen_ind) * 0.8 < dim:
    #	dim = int(len(seen_ind) * 0.8)
    if mi == 0 or mi == 1:
        # sp = graph_shortest_path(A, method='FW', directed=False)
        sp = shortest_path(A, method='FW', directed=False)
    else:
        sp = RandomWalkRestart(A, 0.8)

    sp = sp[seen_ind, :]
    sp = sp[:, seen_ind]
    X = np.zeros((np.shape(sp)[0], dim))
    print("dim: ", np.shape(sp)[0] - 1)
    print("dim0: ", dim)
    svd_dim = min(dim, np.shape(sp)[0] - 1)
    if mi == 0 or mi == 2:
        print('please set mi=3')
        # X[:, :svd_dim] = svd_emb(sp, dim=svd_dim)
    else:
        X[:, :svd_dim] = DCA_vector(sp, dim=svd_dim)[0]
    X_ret = np.zeros((nl, dim))
    X_ret[seen_ind, :] = X
    if mi == 2 or mi == 3:
        sp *= -1
    return sp, X_ret


def emb_ontology(i2l, ontology_mat, co2co_nlp, dim=5, mi=0, unseen_l=None):
    nco = len(i2l)
    network = np.zeros((nco, nco))
    for i in range(nco):
        c1 = i2l[i]
        for j in range(nco):
            if ontology_mat[i, j] == 1:
                network[i, j] = co2co_nlp[c1][i2l[j]]
                network[j, i] = co2co_nlp[c1][i2l[j]]
    idd = 0
    sp, i2emb = graph_embedding_dca(network, i2l, mi=mi, dim=dim, unseen_l=unseen_l)
    return i2emb


def get_ontology_parents(GO_net, g, dfs_depth=100):
    term_valid = set()
    ngh_GO = set()
    ngh_GO.add(g)
    depth = {}
    depth[g] = 0
    while len(ngh_GO) > 0:
        for GO in list(ngh_GO):
            for GO1 in GO_net[GO]:
                ngh_GO.add(GO1)
                depth[GO1] = depth[GO] + 1
            ngh_GO.remove(GO)
            if depth[GO] < dfs_depth:
                term_valid.add(GO)
    return term_valid


def creat_cell_ontology_matrix(train_Y, co2co_graph, cell_ontology_ids, dfs_depth):
    lset = set(cell_ontology_ids)

    seen_l = sorted(np.unique(train_Y))
    unseen_l = sorted(lset - set(train_Y))
    ys = np.concatenate((seen_l, unseen_l))
    i2l = {}
    l2i = {}
    for l in ys:
        nl = len(i2l)
        l2i[l] = nl
        i2l[nl] = l
    nco = len(i2l)

    net_dict = collections.defaultdict(dict)
    net_mat = np.zeros((nco, nco))
    for co1 in co2co_graph:
        l1 = l2i[co1]
        for co2 in co2co_graph[co1]:
            l2 = l2i[co2]
            net_dict[l1][l2] = 1
            net_mat[l1][l2] = 1
    for n in range(nco):
        ngh = get_ontology_parents(net_dict, n, dfs_depth)
        net_dict[n][n] = 1
        for n1 in ngh:
            net_dict[n][n1] = 1
    return unseen_l, l2i, i2l, net_dict, net_mat


class DataProcessing:
    def __init__(self, cell_type_network_file='../../OnClass_data/cell_ontology/cl.ontology',
                 cell_type_nlp_emb_file='../../OnClass_data/cell_ontology/cl.ontology.nlp.emb',
                 terms_with_def=set(),
                 memory_saving_mode=False):
        """
        Initialize OnClass model with a given cell-type network and cell-type embedding file.
        Also, you may set the memory_saving_mode to True to get a model that uses less RAM.
        """
        self.cell_type_nlp_emb_file = cell_type_nlp_emb_file
        self.cell_type_network_file = cell_type_network_file
        self.co2co_graph, self.co2co_nlp, self.co2vec_nlp, self.cell_ontology_ids = read_cell_type_nlp_network(
            self.cell_type_nlp_emb_file, self.cell_type_network_file)
        self.co2co_graph, self.co2co_nlp, self.co2vec_nlp, self.cell_ontology_ids = subset_cell_type_nlp_network(
            self.co2co_graph, self.co2co_nlp, self.co2vec_nlp, self.cell_ontology_ids, terms_with_def)
        self.mode = memory_saving_mode

    def ProcessTrainFeature(self, train_feature, train_label, train_genes, test_feature=None, test_genes=None,
                            batch_correct=False, log_transform=True):
        """
        Process the gene expression matrix used to train the model, and optionally the test data.
        Parameters
        ----------
        train_feature: `numpy.ndarray` or `scipy.sparse.csr_matrix` (depends on mode)
            gene expression matrix of cell types
        train_label: `numpy.ndarray`
            labels for the training features
        train_genes: `list`
            list of genes used during training
        test_feature: `numpy.ndarray` or `scipy.sparse.csr_matrix` (depends on mode), optional (None)
            gene expression matrix of cell types for the test set
        test_genes: `list`, optional (None)
            list of genes used in test set
        batch_correct: `bool`, optional (False)
            whether to correct for batch effect in data
        log_transform:`bool`, optional (True)
            whether to apply log transform to data
        Returns
        -------
        train_feature, test_feature, self.genes, self.genes
            returns the training feature gene expression matrix and the list of genese associated
            with it. If test_feature was not none, also returns the test features and their genes.
        """

        if log_transform is False and np.max(train_feature) > 1000:
            warnings.warn(
                "Max expression is" + str(np.max(train_feature)) + '. Consider setting log transform = True\n')
        self.genes = train_genes
        # batch correction is currently not supported for memory_saving_mode
        if batch_correct and test_feature is not None and test_genes is not None and self.mode:
            train_feature, test_feature, selected_train_genes = process_expression(train_feature, test_feature,
                                                                                   train_genes, test_genes)
            self.genes = selected_train_genes
        elif log_transform:
            if self.mode:
                train_feature = csr_matrix.log1p(train_feature)
            else:
                train_feature = np.log1p(train_feature)

            if test_feature is not None:
                if self.mode:
                    test_feature = csr_matrix.log1p(test_feature)
                else:
                    test_feature = np.log1p(test_feature)
        self.train_feature = train_feature
        self.train_label = train_label
        if test_feature is not None:
            return train_feature, test_feature, self.genes, self.genes
        else:
            return train_feature, self.genes

    def ProcessTestFeature(self, test_feature, test_genes, use_pretrain=None, batch_correct=False, log_transform=True):
        """
        Process the gene expression matrix used to test the model.
        Parameters
        ----------
        test_feature: `numpy.ndarray` or `scipy.sparse.csr_matrix` (depends on mode)
            gene expression matrix of cell types for the test set
        test_genes: `list`
            list of genes used in test set
        use_pretrain: `string`, optional (None)
            name of the pretrained model
        batch_correct: `bool`, optional (False)
            whether to correct for batch effect in data
        log_transform:`bool`, optional (True)
            whether to apply log transform to data
        Returns
        -------
        gene_mapping or test_feature
            processes the test features and returns a data structure that encodes the gene
            expression matrix that should be used for testing. If the model is in memory saving
            mode, then the function will return a tuple of gene expression matrix and index array,
            otherwise, it will just return the matrix.
        """
        if log_transform is False and np.max(test_feature) > 1000:
            warnings.warn("Max expression is" + str(np.max(test_feature)) + '. Consider setting log transform = True\n')

        if log_transform:
            test_feature = np.log1p(test_feature)
        if batch_correct and not self.mode:
            test_feature = mean_normalization(self.train_feature_mean, test_feature)

        if self.mode:
            gene_mapping = get_gene_mapping(test_genes, self.genes)
            return test_feature, gene_mapping
        else:
            test_feature = map_genes(test_feature, test_genes, self.genes, memory_saving_mode=self.mode)
            return test_feature

    def EmbedCellTypes(self, train_Y_str, dim=5, emb_method=3, use_pretrain=None, write2file=None):
        """
        Embed the cell ontology
        Parameters
        ----------
        cell_type_network_file : each line should be cell_type_1\tcell_type_2\tscore for weighted network or cell_type_1\tcell_type_2 for unweighted network
        dim: `int`, optional (500)
            Dimension of the cell type embeddings
        emb_method: `int`, optional (3)
            dimensionality reduction method
        use_pretrain: `string`, optional (None)
            use pretrain file. This should be the numpy file of cell type embeddings. It can read the one set in write2file parameter.
        write2file: `string`, optional (None)
            write the cell type embeddings to this file path
        Returns
        -------
        co2emb, co2i, i2co
            returns three dicts, cell type name to embeddings, cell type name to cell type id and cell type id to embeddings.
        """

        self.unseen_co, self.co2i, self.i2co, self.ontology_dict, self.ontology_mat = creat_cell_ontology_matrix(
            train_Y_str, self.co2co_graph, self.cell_ontology_ids, dfs_depth=3)
        self.nco = len(self.i2co)
        Y_emb = emb_ontology(self.i2co, self.ontology_mat, dim=dim, mi=emb_method, co2co_nlp=self.co2co_nlp,
                             unseen_l=self.unseen_co)
        self.co2emb = np.column_stack((np.eye(self.nco), Y_emb))
        self.nunseen = len(self.unseen_co)
        self.nseen = self.nco - self.nunseen
        self.co2vec_nlp_mat = np.zeros((self.nco, len(self.co2vec_nlp[self.i2co[0]])))
        for i in range(self.nco):
            self.co2vec_nlp_mat[i, :] = self.co2vec_nlp[self.i2co[i]]
        return self.co2emb, self.co2i, self.i2co, self.ontology_mat


class cellData(Dataset):

    def __init__(self, expression_matrix, labels, nlabel=0, gpu_ids='0'):
        self.expression_matrix = expression_matrix
        self.labels = labels
        self.nlabel = nlabel
        self.gpu_ids = gpu_ids

    def __getitem__(self, item):
        x = np.asarray(self.expression_matrix[item, :].todense()).squeeze()
        y = np.zeros(self.nlabel)
        y[self.labels[item]] = float(1)
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        if len(self.gpu_ids) > 0:
            x = x.cuda()
            y = y.cuda()
        return {'features': x, 'label': y}

    def __len__(self):
        return len(self.labels)


def emb2tensor(def_embeddings, terms, add_bias=False):
    '''
    This function re-sort the textual description embeddings
    according to the mapping between terms and index
    :param text_embeddings:
    :param terms:
    :return:
    '''
    ann_id = list(terms.keys())
    embedding_array = np.zeros((len(ann_id), np.size(def_embeddings[ann_id[0]], 1)))

    for t_id in ann_id:
        t_def = def_embeddings[t_id].reshape([1, -1])
        t_def = t_def / np.sqrt(np.sum(np.power(t_def, 2), axis=1))
        embedding_array[terms[t_id], :] = t_def
    rank_e = np.linalg.matrix_rank(embedding_array)
    if add_bias:
        embedding_array = np.column_stack((np.eye(len(ann_id)), embedding_array))
    print('Rank of your embeddings is {}'.format(rank_e))
    embedding_array = torch.from_numpy(embedding_array)
    return embedding_array.float()


def read_data_file(dname, data_dir):
    if 'microcebus' in dname and 'tabula_microcebus' not in dname:
        tech = '10x'
        feature_file = data_dir + 'Lemur/' + dname + '.h5ad'
        filter_key = {'method': tech}
        label_file = None
        gene_file = ''
        label_key = 'cell_ontology_class'
    elif 'tabula_microcebus' in dname:
        tech = dname.split('_')[1]
        feature_file = data_dir + 'Tabula_Microcebus/' + 'LCA_complete_wRaw_toPublish.h5ad'
        filter_key = {}
        label_file = None
        gene_file = ''
        batch_key = ''
        label_key = 'cell_ontology_class_v1'
    elif 'sapiens' in dname:
        feature_file = data_dir + 'Tabula_Sapiens/' + 'TabulaSapiens.h5ad'
        filter_key = {}
        label_file = None
        gene_file = ''
        batch_key = ''
        label_key = 'cell_ontology_class'
    elif 'muris' in dname:
        tech = dname.split('_')[1]
        feature_file = data_dir + 'Tabula_Muris_Senis/' + 'tabula-muris-senis-' + tech + '-official-raw-obj.h5ad'
        filter_key = {}
        label_file = None
        gene_file = ''
        batch_key = ''
        label_key = 'cell_ontology_class'
    elif 'sapiens' in dname:
        feature_file = data_dir + 'sapiens/' + 'Pilot1_Pilot2_decontX_Oct2020.h5ad'
        filter_key = {}
        label_file = None
        gene_file = ''
        batch_key = ''
        label_key = 'cell_ontology_type'
    elif 'allen' in dname:
        feature_file = data_dir + '/Allen_Brain/features.pkl'
        label_file = data_dir + '/Allen_Brain/labels.pkl'
        gene_file = data_dir + '/Allen_Brain/genes.pkl'
        label_key = ''
        filter_key = {}
    elif 'krasnow' in dname:
        tech = dname.split('_')[1]
        feature_file = data_dir + '/HLCA/' + tech + '_features.pkl'
        label_file = data_dir + '/HLCA/' + tech + '_labels.pkl'
        gene_file = data_dir + '/HLCA/' + tech + '_genes.pkl'
        label_key = ''
        filter_key = {}
    else:
        sys.exit('wrong dname ' + dname)
    if feature_file.endswith('.pkl'):
        return feature_file, filter_key, label_key, label_file, gene_file
    elif feature_file.endswith('.h5ad'):
        return feature_file, filter_key, label_key, label_file, gene_file
    sys.exit('wrong file suffix')


def read_ontology_file(dname, data_folder):
    if 'allen' in dname:
        cell_type_network_file = data_folder + 'allen.ontology'
        cell_type_nlp_emb_file = None
        cl_obo_file = None
        if not os.path.isfile(cell_type_network_file):
            sys.error(cell_type_network_file + ' not found!')
    else:
        cell_type_network_file = data_folder + 'cl.ontology'
        cell_type_nlp_emb_file = data_folder + 'cl.ontology.nlp.emb'
        cl_obo_file = data_folder + 'cl.obo'
        if not os.path.isfile(cell_type_nlp_emb_file):
            sys.exit(cell_type_nlp_emb_file + ' not found!')
        if not os.path.isfile(cell_type_network_file):
            sys.exit(cell_type_network_file + ' not found!')
        if not os.path.isfile(cl_obo_file):
            sys.exit(cl_obo_file + ' not found!')
    return cell_type_nlp_emb_file, cell_type_network_file, cl_obo_file


def select_cells_based_on_keys(x, features, tissues=None, labels=None, filter_key=None):
    ncell = np.shape(x.X)[0]
    select_cells = set(range(ncell))
    for key in filter_key:
        value = filter_key[key]
        select_cells = select_cells & set(np.where(np.array(x.obs[key]) == value)[0])
    select_cells = sorted(select_cells)
    features = features[select_cells, :]
    if labels is not None:
        labels = labels[select_cells]
    if tissues is not None:
        tissues = tissues[select_cells]
    x = x[select_cells, :]
    return features, labels, tissues, x


def get_ontology_name(obo_file, lower=True):
    fin = open(obo_file)
    co2name = {}
    name2co = {}
    tag_is_syn = {}
    for line in fin:
        if line.startswith('id: '):
            co = line.strip().split('id: ')[1]
        if line.startswith('name: '):
            if lower:
                name = line.strip().lower().split('name: ')[1]
            else:
                name = line.strip().split('name: ')[1]
            co2name[co] = name
            name2co[name] = co
        if line.startswith('synonym: '):
            if lower:
                syn = line.strip().lower().split('synonym: "')[1].split('" ')[0]
            else:
                syn = line.strip().split('synonym: "')[1].split('" ')[0]
            if syn in name2co:
                continue
            name2co[syn] = co
    fin.close()
    return co2name, name2co


def fine_nearest_co_using_nlp(sentences, co2emb, obo_file, nlp_mapping_cutoff=0.8):
    co2name, name2co = get_ontology_name(obo_file=obo_file)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    sentences = np.array([sentence.lower() for sentence in sentences])
    sentence_embeddings = model.encode(sentences)
    co_embeddings = []
    cos = []
    for co in co2emb:
        co_embeddings.append(co2emb[co])
        cos.append(co)
    co_embeddings = np.array(co_embeddings)
    sent2co = {}
    for sentence, embedding, ind in zip(sentences, sentence_embeddings, range(len(sentences))):
        scs = cosine_similarity(co_embeddings, embedding.reshape(1, -1))

        co_id = np.argmax(scs)
        sc = scs[co_id]
        if sc > nlp_mapping_cutoff:
            sent2co[sentence.lower()] = cos[co_id]
            names = set()
            for name in name2co:
                if name2co[name].upper() == cos[co_id]:
                    names.add(name)
        # print (sentence, cos[co_id], sc, co2name[cos[co_id]],names)
    return sent2co


def exact_match_co_name_2_co_id(labels, lab2co, cl_obo_file=None):
    if cl_obo_file is None:
        return lab2co
    co2name, name2co = get_ontology_name(obo_file=cl_obo_file)
    for label in labels:
        if label.lower() in name2co:
            lab2co[label.lower()] = name2co[label.lower()]
    for name in name2co:
        lab2co[name.lower()] = name2co[name]
    return lab2co


def map_and_select_labels(labels, cell_ontology_ids, obo_file, ct_mapping_key={}, nlp_mapping=True,
                          nlp_mapping_cutoff=0.8, co2emb=None, cl_obo_file=None):
    lab2co = {}
    if nlp_mapping:
        if co2emb is None:
            sys.exit('Please provide cell type embedding to do NLP-based mapping.')
        lab2co = fine_nearest_co_using_nlp(np.unique(labels), co2emb, obo_file, nlp_mapping_cutoff=nlp_mapping_cutoff)
    lab2co = exact_match_co_name_2_co_id(np.unique(labels), lab2co, cl_obo_file=cl_obo_file)
    for ct in ct_mapping_key:
        lab2co[ct_mapping_key[ct]] = lab2co[ct]
    ind = []
    lab_id = []
    unfound_labs = set()
    for i, l in enumerate(labels):
        if l in cell_ontology_ids:
            ind.append(i)
            lab_id.append(l)
        elif l.lower() in lab2co:
            ind.append(i)
            lab_id.append(lab2co[l.lower()])
        else:
            unfound_labs.add(l)
    frac = len(ind) * 1. / len(labels)
    ind = np.array(ind)
    labels = np.array(lab_id)
    unfound_labs = set(unfound_labs)
    warn_message = 'Warning: Only: %f precentage of labels are in the Cell Ontology. The remaining cells are excluded! Consider using NLP mapping and choose a small mapping cutoff (nlp_mapping_cutoff)' % (
            frac * 100)
    if frac < 0.5:
        print(warn_message)
        print('Here are unfound labels:', unfound_labs)
    return ind, labels, unfound_labs


def exclude_parent_child_nodes(cell_ontology_file, labels):
    uniq_labels = np.unique(labels)
    excludes = set()
    net = collections.defaultdict(dict)
    fin = open(cell_ontology_file)
    for line in fin:
        s, p = line.strip().split('\t')
        net[s][p] = 1  # p is parent
    fin.close()
    for n in list(net.keys()):
        ngh = get_ontology_parents(net, n)
        for n1 in ngh:
            net[n][n1] = 1
    for l1 in uniq_labels:
        for l2 in uniq_labels:
            if l1 in net[l2] and l1 != l2:  # l1 is l2 parent
                excludes.add(l1)
    # print (excludes)
    new_ids = []
    for i in range(len(labels)):
        if labels[i] not in excludes:
            new_ids.append(i)
    new_ids = np.array(new_ids)
    return new_ids, excludes


def read_data(feature_file, cell_ontology_ids, exclude_non_leaf_ontology=False, ct_mapping_key={}, tissue_key=None,
              seed=1, filter_key=None, AnnData_label_key=None, nlp_mapping=True, nlp_mapping_cutoff=0.8, co2emb=None,
              label_file=None, cl_obo_file=None, cell_ontology_file=None, memory_saving_mode=False,
              backup_file='sparse_featurefile_backup'):
    """
	Read data from the given feature file, and processes it so that it fits with the other
	given paramters as needed.
	Parameters
	----------
	feature_file: `string`
		name of file to extract data from. The data in the file must be stored in h5ad file format.
	cell_ontology_ids: `set`
		set of ids from the cell ontology.
	AnnData_label_key: `numpy.ndarray`, optional (None)
		mapping of the cell type classes to reindex the labels in the AnnData object
	co2emb: `map`, optional (None)
		maps cell-type from the cell ontology to its embedding
	label_file: `string`, optional (None)
		file from which to get the labels of the feature file
	memory_saving_mode: `bool`, optional (False)
		whether the method should be run under tight RAM constraints.
	backup_file: `string`, optional ('sparse_featurefile_backup')
		the name of the file to copy the sparse feature dataset to.

	Returns
	-------
	dataset: `numpy.ndarray` or `scipy.sparse.csr_matrix` (depends on mode)
		gene expression matrix of cell types for the test set
	genes: `list`
		list of genes in the dataset
	labels: `numpy.ndarray`
		labels from the feature file
	x: AnnData object stored in the given feature file
	"""

    np.random.seed(seed)

    if memory_saving_mode:
        x = read_h5ad(feature_file, backed='r+')
        if 'Tabula_Microcebus' in feature_file or 'TabulaSapiens' in feature_file:
            x.raw = None
        dataset = x.X.to_memory()  # Gets a sparse array in csr matrix form
    else:
        x = read_h5ad(feature_file)
        dataset = x.X.toarray()

    # if memory_saving_mode:
    #    print_memory_usage("while reading data")

    ncell = np.shape(x.X)[0]
    genes = np.array([x.upper() for x in x.var.index])

    if tissue_key is not None and 'TabulaSapiens' not in feature_file:
        tissues = np.array(x.obs[tissue_key].tolist())
    else:
        tissues = None
    if AnnData_label_key is None and label_file is None:
        print('no label file is provided')
        labels = None
        dataset, labels, tissues, x = select_cells_based_on_keys(x, dataset, labels=labels, tissues=tissues,
                                                                 filter_key=filter_key)
        return dataset, genes, labels, tissues, x
    if AnnData_label_key is not None:
        labels = x.obs[AnnData_label_key].tolist()
    else:
        fin = open(label_file)
        labels = []
        for line in fin:
            labels.append(line.strip())
        fin.close()
    labels = np.array(labels)
    dataset, labels, tissues, x = select_cells_based_on_keys(x, dataset, labels=labels, tissues=tissues,
                                                             filter_key=filter_key)

    if memory_saving_mode:
        x = x.copy(filename=backup_file)

    ind, labels, unfound_labs = map_and_select_labels(labels, cell_ontology_ids, cl_obo_file,
                                                      ct_mapping_key=ct_mapping_key, nlp_mapping=nlp_mapping,
                                                      co2emb=co2emb, nlp_mapping_cutoff=nlp_mapping_cutoff,
                                                      cl_obo_file=cl_obo_file)
    if tissue_key is not None and 'TabulaSapiens' not in feature_file:
        tissues = tissues[ind]
    dataset = dataset[ind, :]

    if memory_saving_mode:
        # Need to copy to disk for rewriting to the sparse dataset
        x = x[ind, :].copy(filename=backup_file)
    else:
        x = x[ind, :]

    if exclude_non_leaf_ontology:
        new_ids, exclude_terms = exclude_parent_child_nodes(cell_ontology_file, labels)
        if tissues is not None:
            tissues = tissues[new_ids]
        dataset = dataset[new_ids, :]
        labels = labels[new_ids]
        x = x[new_ids, :]

    ncell = np.shape(dataset)[0]
    index = np.random.choice(ncell, ncell, replace=False)
    dataset = dataset[index, :]  # cell by gene matrix
    labels = labels[index]
    if tissue_key is not None and 'TabulaSapiens' not in feature_file:
        tissues = tissues[index]
    return dataset, genes, labels, tissues, x


def load_co_text(co_data_path):
    ont = dict()
    obj = None
    with open(co_data_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line == '[Term]':
                if obj is not None and len(obj['text']) > 0:
                    ont[obj['id']] = obj
                obj = dict()
                obj['text'] = ""
                continue
            elif line == '[Typedef]' and len(obj['text']) > 0:
                if obj is not None:
                    ont[obj['id']] = obj
                obj = None
            else:
                if obj is None:
                    continue
                l = line.split(": ")
                if l[0] == 'id':
                    obj['id'] = l[1]
                elif l[0] == 'name':
                    obj['text'] += line.split('name: ')[1].strip() + '. '
                elif l[0] == 'def':
                    obj['text'] += line.split('def: ')[1].split('"')[1].strip()
        if obj is not None and len(obj['text']) > 0:
            ont[obj['id']] = obj
    return ont

def extract_data_based_on_class(feats, labels, sel_labels):
    ind = []
    for l in sel_labels:
        id = np.where(labels == l)[0]
        ind.extend(id)
    np.random.shuffle(ind)
    X = feats[ind, :]
    Y = labels[ind]
    return X, Y, ind


def SplitTrainTest(all_X, all_Y, all_tissues=None, random_state=10, nfold_cls=0.3, nfold_sample=0.2, nmin_size=10,
                   memory_saving_mode=False):
    """
    Utility function for splitting the dataset into a train and test set.
    Parameters
    ----------
    all_X: all the feature data
    all_Y: the corresponding labels

    Returns
    -------
    The labeled training and test sets
    """
    np.random.seed(random_state)

    cls = np.unique(all_Y)
    cls2ct = Counter(all_Y)
    ncls = len(cls)
    test_cls = list(np.random.choice(cls, int(ncls * nfold_cls), replace=False))
    for c in cls2ct:
        if cls2ct[c] < nmin_size:
            test_cls.append(c)
    test_cls = np.unique(test_cls)
    # add rare class to test, since they cannot be split into train and test by using train_test_split(stratify=True)
    train_cls = [x for x in cls if x not in test_cls]
    train_cls = np.array(train_cls)
    train_X, train_Y, train_ind = extract_data_based_on_class(all_X, all_Y, train_cls)
    test_X, test_Y, test_ind = extract_data_based_on_class(all_X, all_Y, test_cls)
    if all_tissues is not None:
        train_tissues = all_tissues[train_ind]
        test_tissues = all_tissues[test_ind]
        train_X_train, train_X_test, train_Y_train, train_Y_test, train_tissues_train, train_tissues_test = train_test_split(
            train_X, train_Y, train_tissues, test_size=nfold_sample, stratify=train_Y, random_state=random_state)
        test_tissues = np.concatenate((test_tissues, train_tissues_test))
        train_tissues = train_tissues_train
    else:
        train_X_train, train_X_test, train_Y_train, train_Y_test = train_test_split(
            train_X, train_Y, test_size=nfold_sample, stratify=train_Y, random_state=random_state)

    # TODO: Added this memory saving mode toggle
    if memory_saving_mode:
        test_X = scipy.sparse.vstack((test_X, train_X_test)).tocsr()
    else:
        test_X = np.vstack((test_X, train_X_test))

    test_Y = np.concatenate((test_Y, train_Y_test))
    train_X = train_X_train
    train_Y = train_Y_train
    if all_tissues is not None:
        return train_X, train_Y, train_tissues, test_X, test_Y, test_tissues
    else:
        return train_X, train_Y, test_X, test_Y


def compute_roc(labels, preds):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)
    return roc_auc


def compute_prc(labels, preds):
    # Compute PRC curve and PRC area for each class
    pr, rc, _ = precision_recall_curve(labels.flatten(), preds.flatten())
    prc_auc = auc(rc, pr)
    return prc_auc


def evaluate_unseen_auroc(inference_preds, inference_label, unseen2i):
    roc_macro = np.zeros(len(unseen2i.values()))
    unseen_id = list(unseen2i.values())
    cl_i = 0
    for i in unseen_id:
        unseen_preds, unseen_labels = inference_preds[:, i], inference_label[:, i]
        auroc = compute_roc(unseen_labels, unseen_preds)
        roc_macro[cl_i] = auroc
        cl_i += 1
    return np.mean(roc_macro)


def evaluate_auroc(inference_preds, inference_label):
    roc_macro = np.zeros(np.size(inference_label, 1))
    for cl_i in range(np.size(inference_label, 1)):
        roc_auc = compute_roc(inference_label[:, cl_i], inference_preds[:, cl_i])
        roc_macro[cl_i] = roc_auc
    roc_auc = np.mean(roc_macro)
    return roc_auc


def sampled_auprc(truths, preds):
    pos = np.where(truths == 1)[0]
    neg = np.where(truths == 0)[0]
    assert (len(pos) + len(neg) == len(truths))
    nneg = len(neg)
    npos = len(pos)
    select_neg = np.random.choice(nneg, npos * 3, replace=True)
    select_ind = np.concatenate((pos, select_neg))
    return average_precision_score(truths[select_ind], preds[select_ind])


def evaluate_unseen_auprc(inference_preds, inference_label, unseen2i):
    roc_macro = np.zeros(len(unseen2i.values()))
    unseen_id = list(unseen2i.values())
    cl_i = 0
    for i in unseen_id:
        unseen_preds, unseen_labels = inference_preds[:, i], inference_label[:, i]
        auroc = sampled_auprc(unseen_labels, unseen_preds)
        roc_macro[cl_i] = auroc
        cl_i += 1
    return np.mean(roc_macro)


def evaluate_auprc(inference_preds, inference_label):
    roc_macro = np.zeros(np.size(inference_label, 1))
    for cl_i in range(np.size(inference_label, 1)):
        roc_auc = sampled_auprc(inference_label[:, cl_i], inference_preds[:, cl_i])
        roc_macro[cl_i] = roc_auc
    roc_auc = np.mean(roc_macro)
    return roc_auc


def find_gene_ind(genes, common_genes):
    gid = []
    for g in common_genes:
        gid.append(np.where(genes == g)[0][0])
    gid = np.array(gid)
    return gid

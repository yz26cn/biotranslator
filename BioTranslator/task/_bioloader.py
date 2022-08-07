"""
Non-text Encoder
"""

import os
import numpy as np
import pandas as pd
import collections
from tqdm import tqdm
from ..utils import load, load_obj, proteinData, emb2tensor, gen_go_emb, organize_workingspace, \
    term_training_numbers, extract_terms_from_dataset, gen_co_emb, cellData, SplitTrainTest, find_gene_ind, \
    DataProcessing, read_data, read_data_file, read_ontology_file


class BioLoader:
    """
    This class loads and stores the data we used in BioTranslator
    """

    def __init__(self, cfg):
        # Organize the working space
        organize_workingspace(cfg.working_space, cfg.task)
        if cfg.data_type in ["graph", "seq"]:
            # load gene ontology data
            self.go_data = load(cfg.go_file)
            # load mappings between terms and index, also find the intersection with go terms
            self.load_terms(cfg)
            if cfg.data_type == 'graph':
                # load proteins which need to be excluded
                self.load_eval_uniprots(cfg)
            # load dataset
            self.load_dataset(cfg)
            if cfg.data_type == 'seq':
                # register dataset for zero shot task or few shot task
                self.register_task(cfg)
            # generate proteinData, which is defined in BioUtils
            self.gen_protein_data(cfg)
        # load textual description embeddings of go terms
        self.load_text_emb(cfg)
        if cfg.data_type == 'vec':
            # load files
            self.load_files(cfg)
        print('Data Loading Finished!')

    def load_files(self, cfg):
        self.cfg = cfg
        self.terms_with_def = self.text_embeddings.keys()
        self.cell_type_nlp_emb_file, self.cell_type_network_file, self.cl_obo_file = read_ontology_file(
            'cell ontology', cfg.ontology_repo)
        self.DataProcess_obj = DataProcessing(self.cell_type_network_file, self.cell_type_nlp_emb_file,
                                              memory_saving_mode=cfg.memory_saving_mode,
                                              terms_with_def=self.terms_with_def)
        feature_file, filter_key, label_key, label_file, gene_file = read_data_file(cfg.dataset, cfg.data_repo)
        self.feature, self.genes, self.label, _, _ = read_data(feature_file,
                                                               cell_ontology_ids=self.DataProcess_obj.cell_ontology_ids,
                                                               exclude_non_leaf_ontology=True, tissue_key='tissue',
                                                               filter_key=filter_key, AnnData_label_key=label_key,
                                                               nlp_mapping=False, cl_obo_file=self.cl_obo_file,
                                                               cell_ontology_file=self.cell_type_network_file,
                                                               co2emb=self.DataProcess_obj.co2vec_nlp,
                                                               memory_saving_mode=cfg.memory_saving_mode,
                                                               backup_file=cfg.backup_file)

    def load_terms(self, cfg):
        if cfg.data_type == "graph":
            terms = pd.read_pickle(cfg.train_terms_file)
        elif cfg.data_type == "seq":
            terms = pd.read_pickle(cfg.terms_file)
        else:
            raise NotImplementedError
        self.i2terms = list(terms['terms'])
        go_terms = list(self.go_data.keys())
        self.i2terms = list(set(self.i2terms).intersection(set(go_terms)))
        self.terms2i = collections.OrderedDict()
        self.n_classes = len(self.i2terms)
        print('terms number:{}'.format(len(self.i2terms)))
        for i in range(len(self.i2terms)): self.terms2i[self.i2terms[i]] = i

    def load_eval_uniprots(self, cfg):
        eval_dataset = cfg.excludes
        eval_uniprots = []
        for d in eval_dataset:
            data_i = load_obj(cfg.data_repo + d + '/pathway_dataset.pkl')
            eval_uniprots += list(data_i['proteins'])
        self.eval_uniprots = set(eval_uniprots)

    def load_dataset(self, cfg):
        if cfg.data_type == 'graph':
            # load train dataset
            self.train_data = pd.read_pickle(cfg.train_file)

            # exclude proteins in pathway dataset from the train data
            drop_index = []
            for i in self.train_data.index:
                if self.train_data.loc[i]['proteins'] in self.eval_uniprots:
                    drop_index.append(i)
            self.train_data = self.train_data.drop(index=drop_index)

            # load protein network fatures and description features
            self.train_prot_network = load_obj(cfg.train_prot_network_file)
            self.network_dim = np.size(list(self.train_prot_network.values())[0])
            self.train_prot_description = load_obj(cfg.train_prot_description_file)

            # load eval dataset (pathway dataset)
            self.eval_data = pd.read_pickle(cfg.eval_file)
            self.eval_prot_network = load_obj(cfg.eval_prot_network_file)
            self.eval_prot_description = load_obj(cfg.eval_prot_description_file)

            # load eval terms
            self.eval_terms2i, self.eval_i2terms = extract_terms_from_dataset(self.eval_data)
            print('eval pathway number:{}'.format(len(self.eval_i2terms)))
        elif cfg.data_type == 'seq':
            self.k_fold = cfg.k_fold
            self.fold_train, self.fold_val = self.load_fold_data(cfg.k_fold,
                                                                 cfg.train_fold_file,
                                                                 cfg.validation_fold_file)

            # load protein network fatures and description features
            self.prot_network = load_obj(cfg.prot_network_file)
            self.network_dim = np.size(list(self.prot_network.values())[0])
            self.prot_description = load_obj(cfg.prot_description_file)

    def register_task(self, cfg):
        if cfg.task == 'zero_shot':
            self.fold_zero_shot_terms_list = self.zero_shot_terms(cfg)
            self.zero_shot_fold_data(self.fold_zero_shot_terms_list)
            for fold_i in range(cfg.k_fold):
                print('Fold {} contains {} zero shot terms'.format(fold_i,
                                                                   len(self.fold_zero_shot_terms_list[fold_i])))
        if cfg.task == 'few_shot':
            self.fold_few_shot_terms_list = self.few_shot_terms(cfg)
            self.diamond_list = self.load_diamond_score(cfg)

    def gen_protein_data(self, cfg):
        # generate protein data which can be loaded by torch
        # the raw train and test data is for blast preds function in BioTrainer
        if cfg.data_type == 'graph':
            self.raw_train, self.raw_val = self.train_data.copy(deep=True), self.eval_data.copy(deep=True)

            self.train_data = proteinData(self.train_data, self.terms2i, self.train_prot_network,
                                          self.train_prot_description, gpu_ids=cfg.gpu_ids)
            self.eval_data = proteinData(self.eval_data, self.eval_terms2i, self.eval_prot_network,
                                         self.eval_prot_description, gpu_ids=cfg.gpu_ids)
        elif cfg.data_type == 'seq':
            self.raw_train, self.raw_val = [], []
            for fold_i in range(cfg.k_fold):
                self.raw_train.append(self.fold_train[fold_i].copy(deep=True))
            self.raw_val.append(self.fold_val[fold_i].copy(deep=True))
            self.fold_train[fold_i] = proteinData(self.fold_train[fold_i], self.terms2i, self.prot_network,
                                        self.prot_description, gpu_ids = cfg.gpu_ids)
            self.fold_val[fold_i] = proteinData(self.fold_val[fold_i], self.terms2i, self.prot_network,
                                      self.prot_description, gpu_ids = cfg.gpu_ids)


    def zero_shot_fold_data(self, fold_zero_shot_terms_list):
        for i in range(self.k_fold):
            zero_terms_k = fold_zero_shot_terms_list[i]
            training, valid = self.fold_train[i], self.fold_val[i]
            drop_index = []
            for j in training.index:
                annts = training.loc[j]['annotations']
                insct = list(set(annts).intersection(zero_terms_k))
                if len(insct) > 0: drop_index.append(j)
            self.fold_train[i] = training.drop(index=drop_index)

    def zero_shot_terms(self, cfg):
        fold_zero_shot_terms = []
        for i in tqdm(range(cfg.k_fold)):
            fold_zero_shot_terms.append(load_obj(cfg.zero_shot_term_path.format(i)))
        return fold_zero_shot_terms

    def few_shot_terms(self, cfg):
        fold_few_shot_terms = []
        for i in tqdm(range(cfg.k_fold)):
            few_shot_terms = collections.OrderedDict()
            few_shot_count = term_training_numbers(self.fold_val[i], self.fold_train[i])
            for j in few_shot_count.keys():
                if 0 < few_shot_count[j] <= 20:
                    few_shot_terms[j] = few_shot_count[j]
            fold_few_shot_terms.append(few_shot_terms)
        return fold_few_shot_terms

    def load_fold_data(self, k, train_fold_file, validation_fold_file):
        train_fold, val_fold = [], []
        for i in range(k):
            train_fold.append(pd.read_pickle(train_fold_file.format(i)))
            val_fold.append(pd.read_pickle(validation_fold_file.format(i)))
        return train_fold, val_fold

    def load_diamond_score(self, cfg):
        diamond_list = []
        for i in range(cfg.k_fold):
            diamond_scores = {}
            with open(cfg.diamond_score_path.format(i)) as f:
                for line in f:
                    it = line.strip().split()
                    if it[0] not in diamond_scores:
                        diamond_scores[it[0]] = {}
                    diamond_scores[it[0]][it[1]] = float(it[2])
            diamond_list.append(diamond_scores)
        return diamond_list

    def load_text_emb(self, cfg):
        if not os.path.exists(cfg.emb_path):
            os.mkdir(cfg.emb_path)
            print('Warning: We created the embedding folder: {}'.format(cfg.emb_path))
        if cfg.data_type in ['graph', 'seq']:
            if cfg.emb_name not in os.listdir(cfg.emb_path):
                gen_go_emb(cfg)
            cfg.text_embedding_file = cfg.emb_path + cfg.emb_name
            self.text_embeddings = pd.read_pickle(cfg.text_embedding_file)

            self.text_embeddings = emb2tensor(self.text_embeddings, self.terms2i)
            if cfg.data_type == 'graph':
                self.pathway_embeddings = pd.read_pickle(cfg.pathway_emb_file)
                self.pathway_embeddings = emb2tensor(self.pathway_embeddings, self.eval_terms2i)
            if len(cfg.gpu_ids) > 0:
                self.text_embeddings = self.text_embeddings.float().cuda()
                if cfg.data_type == 'graph':
                    self.pathway_embeddings = self.pathway_embeddings.float().cuda()
        elif cfg.data_type == 'vec':
            if cfg.emb_name not in os.listdir(cfg.emb_path):
                gen_co_emb(cfg)
            cfg.text_embedding_file = cfg.emb_path + cfg.emb_name
            self.text_embeddings = pd.read_pickle(cfg.text_embedding_file)
        else:
            raise NotImplementedError
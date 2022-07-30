import os
import collections
import numpy as np
import pandas as pd
from .BioConfig import BioConfig
from tqdm import tqdm
from .BioUtils import load, load_obj, proteinData, \
    emb2tensor, gen_go_emb, term_training_numbers

def organize_workingspace(workingspace, task=None):
    """
    Make sure that the working space include the zero shot folder, few shot folder,
    model folder, training log folder and results folder
    :param workingspace:
    :return:
    """
    if task:
        task_path = workingspace + task
    else:
        task_path = workingspace
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

def bioloader(cfg: BioConfig):
    """
    This class loads and stores the data we used in BioTranslator
    """

    def load_terms(terms_file: str):
        terms = pd.read_pickle(terms_file)
        i2terms = list(terms['terms'])
        go_terms = list(go_data.keys())
        # intersection between given term file and go terms
        i2terms = list(set(i2terms).intersection(set(go_terms)))
        terms2i = collections.OrderedDict()
        n_classes = len(i2terms)
        print(f'terms under: {n_classes}')
        for i, term in enumerate(i2terms):
            terms2i[term] = i
        return terms2i

    def load_eval_uniprots(cfg):
        eval_dataset = cfg.excludes
        eval_uniprots = []
        for d in eval_dataset:
            data_i = load_obj(cfg.data_repo + d + '/pathway_dataset.pkl')
            eval_uniprots += list(data_i['proteins'])
        self.eval_uniprots = set(eval_uniprots)

    def load_dataset(cfg: BioConfig):
        if self.data_type == "protein":
            self.k_fold = cfg.k_fold
            self.fold_train, self.fold_val = self.load_fold_data(cfg.k_fold,
                                                                 cfg.train_fold_file,
                                                                 cfg.validation_fold_file)
            self.prot_network = load_obj(cfg.prot_network_file)
            self.network_dim = np.size(list(self.prot_network.values())[0])
            self.prot_description = load_obj(cfg.prot_description_file)
        elif self.data_type == "pathway":
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
        elif self.data_type == "cell":

        else:
            raise NotImplementedError

    def load_fold_data(k_fold: int, train_fold_file: str, validation_fold_file: str):
        train_fold, val_fold = [], []
        for i in range(k_fold):
            train_fold.append(pd.read_pickle(train_fold_file).format(i))
            val_fold.append(pd.read_pickle(validation_fold_file).format(i))
        return train_fold, val_fold

    def register_task(cfg: BioConfig):
        if cfg.task == 'zero_shot':
            self.fold_zero_shot_terms_list = self.zero_shot_terms(cfg)
            self.zero_shot_fold_data(self.fold_zero_shot_terms_list)
            for fold_i in range(cfg.k_fold):
                print(f"Fold {fold_i} contains"
                      f"{len(self.fold_zero_shot_terms_list[fold_i])} zero shot terms")
        if cfg.task == 'few_shot':
            self.fold_few_shot_terms_list = self.few_shot_terms(cfg)
            self.diamond_list = self.load_diamond_score(cfg)

    def zero_shot_terms(cfg: BioConfig):
        fold_zero_shot_terms = []
        for i in tqdm(range(cfg.k_fold)):
            fold_zero_shot_terms.append(load_obj(cfg.zero_shot_term_path.format(i)))
        return fold_zero_shot_terms

    def zero_shot_fold_data(fold_zero_shot_terms_list: list):
        for i in range(self.k_fold):
            zero_terms_k = fold_zero_shot_terms_list[i]
            training, _ = self.fold_train[i], self.fold_val[i]
            drop_index = []
            for j in training.index:
                annts = training.loc[j]['annotations']
                insct = list(set(annts).intersection(zero_terms_k))
                if len(insct) > 0:
                    drop_index.append(j)
            self.fold_train[i] = training.drop(index=drop_index)

    def few_shot_terms(cfg: BioConfig):
        fold_few_shot_terms = []
        for i in tqdm(range(cfg.k_fold)):
            few_shot_terms = collections.OrderedDict()
            few_shot_count = term_training_numbers(self.fold_val[i], self.fold_train[i])
            for j in few_shot_count.keys():
                if 0 < few_shot_count[j] <= 20:
                    few_shot_terms[j] = few_shot_count[j]
            fold_few_shot_terms.append(few_shot_terms)
        return fold_few_shot_terms

    def load_diamond_score(cfg: BioConfig):
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

    def gen_protein_data(cfg: BioConfig):

        # generate protein data which can be loaded by torch
        # the raw train and test data is for blast function in BioTrainer
        self.raw_train, self.raw_val = [], []
        for fold_i in range(cfg.k_fold):
            self.raw_train.append(self.fold_train[fold_i].copy(deep=True))
            self.raw_val.append(self.fold_val[fold_i].copy(deep=True))
            self.fold_train[fold_i] = proteinData(self.fold_train[fold_i],
                                                  self.terms2i, self.prot_network,
                                                  self.prot_description, gpu_ids=cfg.gpu_ids)
            self.fold_val[fold_i] = proteinData(self.fold_val[fold_i],
                                                self.terms2i, self.prot_network,
                                                self.prot_description, gpu_ids=cfg.gpu_ids)

    def load_text_emb(cfg: BioConfig):
        if not os.path.exists(cfg.emb_path):
            os.mkdir(cfg.emb_path)
            print(f'Warning: We created the embedding folder: {cfg.emb_path}')
        if cfg.method == 'DeepGoPlus':
            self.text_embeddings = None
        else:
            # BioTranslator, clusDCA, TF-IDF, Word2Vec, Doc2Vec
            if cfg.emb_name not in os.listdir(cfg.emb_path):
                gen_go_emb(cfg)
            cfg.text_embedding_file = cfg.emb_path + cfg.emb_name
            self.text_embeddings = pd.read_pickle(cfg.text_embedding_file)
            self.text_embeddings = emb2tensor(self.text_embeddings, self.terms2i)
            if len(cfg.gpu_ids) > 0:
                self.text_embeddings = self.text_embeddings.float().cuda()

    if cfg.output_type == "protein":
        # Organize the working space
        organize_workingspace(cfg.working_space, cfg.task)
        # load gene ontology data
        go_data = load(cfg.go_file)
        # load mappings between terms and index, also find the intersection with go terms
        load_terms(cfg.terms_file)
        # load dataset
        load_dataset(cfg)
        # register dataset for zero shot task or few shot task
        register_task(cfg)
        # generate proteinData. which is defined in BioUtils
        gen_protein_data(cfg)
        # load textual description embeddings of go terms
        load_text_emb(cfg)
        print('Data Loading Finished!')
    elif cfg.output_type == "pathway":




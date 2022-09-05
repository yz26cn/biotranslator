import os
import torch
import collections
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from ..metrics import auroc_metrics
from ..biotranslator import BioTranslator
from ..utils import term2preds_label, term_training_numbers, get_logger, get_namespace_terms, \
    get_few_shot_namespace_terms, save_obj, compute_blast_preds, load_obj


class SeqTrainer:

    def __init__(self, files, cfg):
        """
        This class mainly include the training of BioTranslator Model
        :param files:
        """
        # pass the dataset details to the config class
        cfg.network_dim = files.network_dim

    def setup_model(self, cfg):
        self.model = BioTranslator(cfg)

    def setup_training(self, files, cfg):
        # train epochs
        self.epoch = cfg.epoch
        # loss function
        self.loss_func = torch.nn.BCELoss()
        # use Adam as optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        # set up the training dataset and validation dataset
        self.train_dataset = files.fold_train
        self.val_dataset = files.fold_val

    def backward_model(self, input_seq, input_description, input_vector, emb_tensor, label):
        logits = self.model(data_type='seq',
                            input_seq=input_seq,
                            input_description=input_description,
                            input_vector=input_vector,
                            texts=emb_tensor)
        self.loss = self.loss_func(logits, label)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        return self.loss.item()

    def train(self, files, cfg):
        # store the results of each fold in one list
        results_list = []
        if cfg.task == 'few_shot':
            with_blast_res_list = []
        # record the performance of our model
        self.logger = get_logger(cfg.logger_name)
        for fold_i in range(cfg.k_fold):
            print('Start Cross-Validation Fold :{} ...'.format(fold_i))

            # setup the model architecture according the methods
            self.setup_model(cfg)
            # setup dataset, the training loss and optimizer
            self.setup_training(files, cfg)
            train_loader = DataLoader(files.fold_train[fold_i], batch_size=cfg.batch_size, shuffle=True)
            eval_loader = DataLoader(files.fold_val[fold_i], batch_size=cfg.batch_size, shuffle=True)
            # train the model
            pbar = tqdm(range(self.epoch), desc='Training')
            for epoch_i in pbar:
                train_loss = 0
                train_count = 0
                for batch in train_loader:
                    train_loss += self.backward_model(batch['prot_seq'], batch['prot_description'],
                                                      batch['prot_network'], files.text_embeddings, batch['label'])
                    train_count += len(batch)
                    pbar.set_postfix({"Fold": fold_i,
                                      "epoch": epoch_i,
                                      "train loss": train_loss / train_count})
            # evaluate the model
            print('Evaluate Our Model ...')
            eval_loss, eval_count = 0, 0
            logits, prots, label = [], [], []
            with torch.no_grad():
                for batch in eval_loader:
                    pred_batch = self.model(batch['prot_seq'], batch['prot_description'],
                                            batch['prot_network'], files.text_embeddings)
                    prots += batch['proteins']
                    eval_loss += self.loss_func(pred_batch, batch['label']).item()
                    eval_count += len(batch)
                    self.loss_func.zero_grad()
                    logits.append(pred_batch.cpu().numpy())
                    label.append(batch['label'].cpu().float().numpy())
            logits, label = np.concatenate(logits, axis=0), np.concatenate(label, axis=0)

            # evaluate the model on the zero shot tasks or the few shot tasks
            if cfg.task == 'zero_shot':
                results = collections.OrderedDict()
                for ont in ['bp', 'mf', 'cc']:
                    # extract the predictions and labels of zero shot terms
                    ont_terms = get_namespace_terms(list(files.fold_zero_shot_terms_list[fold_i].keys()), files.go_data, ont)
                    logits_ont, label_ont = term2preds_label(logits, label, ont_terms, files.terms2i)
                    auroc_mean, auroc_pct = auroc_metrics(label_ont, logits_ont)
                    results[ont] = collections.OrderedDict()
                    results[ont]['auroc'], results[ont]['percentage'] = auroc_mean, auroc_pct
                    self.logger.info('fold:{} ont: {} roc auc:{}'.format(fold_i, ont, auroc_mean))
                    for T in auroc_pct.keys():
                        self.logger.info('fold:{} ont: {} the AUROC of {}% terms is greater than {}'.format(fold_i, ont, auroc_pct[T], T))
                results_list.append(results)
                torch.save(self.model, cfg.save_model_path.format(cfg.method, cfg.dataset, fold_i))

            if cfg.task == 'few_shot':
                # generate blast preds for the few shot task
                if not os.path.exists(cfg.blast_preds_path.format(fold_i)):
                    blast_preds = compute_blast_preds(files.diamond_list[fold_i],
                                                      files.raw_val[fold_i],
                                                      files.raw_train[fold_i])
                    save_obj(blast_preds, cfg.blast_preds_path.format(fold_i))
                else:
                    blast_preds = load_obj(cfg.blast_preds_path.format(fold_i))

                # compute logits with blast
                logits_with_blast = logits.copy()
                for go_id in files.terms2i.keys():
                    ont = cfg.ont_term_syn[files.go_data[go_id]['namespace']]
                    for p_i in range(np.size(logits_with_blast, 0)):
                        if go_id not in blast_preds[prots[p_i]].keys():
                            logits_with_blast[p_i, files.terms2i[go_id]] = (1 - cfg.alphas[ont]) * logits[p_i, files.terms2i[go_id]] + cfg.alphas[ont] * 0
                        else:
                            logits_with_blast[p_i, files.terms2i[go_id]] = (1 - cfg.alphas[ont]) * logits[p_i, files.terms2i[go_id]] + cfg.alphas[ont] * blast_preds[prots[p_i]][go_id]

                results = collections.OrderedDict()
                with_blast_res = collections.OrderedDict()
                for ont in ['bp', 'mf', 'cc']:
                    # extract the predictions and labels of zero shot terms
                    results[ont] = collections.OrderedDict()
                    with_blast_res[ont] = collections.OrderedDict()
                    for n in range(1, 20):
                        ont_terms = get_few_shot_namespace_terms(files.fold_few_shot_terms_list[fold_i], files.go_data, ont, n)
                        # evaluate without blast
                        logits_ont, label_ont = term2preds_label(logits, label, ont_terms, files.terms2i)
                        auroc_mean, auroc_pct = auroc_metrics(label_ont, logits_ont)
                        results[ont][n] = auroc_mean
                        # evaluate with blast
                        logits_blast_ont, label_ont = term2preds_label(logits_with_blast, label, ont_terms, files.terms2i)
                        auroc_blast_mean, auroc_blast_pct = auroc_metrics(label_ont, logits_blast_ont)
                        with_blast_res[ont][n] = auroc_blast_mean
                        self.logger.info('fold:{} ont: {} sample number: {} roc auc:{} add blast: {}'.format(fold_i, ont, n, auroc_mean, auroc_blast_mean))
                # append the results of fold i without blast predictions and with blast predictions
                results_list.append(results)
                with_blast_res_list.append(with_blast_res)

        # save the results
        save_obj(results_list, cfg.results_name)
        if cfg.task == 'few_shot':
            save_obj(with_blast_res_list, cfg.blast_res_name)

    def annotate(self, files, cfg, data_dir, anno_data, task='prot_func_pred'):
        def zero_shot_terms():
            zero_shot_term_path = data_dir + 'zero_shot_terms.pkl'
            zero_shot_terms = load_obj(zero_shot_term_path)
            return zero_shot_terms

        def term2preds_label(preds, terms, terms2id):
            new_preds = []
            for t_id in terms:
                new_preds.append(preds[:, terms2id[t_id]].reshape((-1, 1)))
            new_preds = np.concatenate(new_preds, axis=1)
            return new_preds

        def zero_shot_data(zero_shot_terms_list, train):
            drop_index = []
            for j in train.index:
                annts = train.loc[j]['annotations']
                insct = list(set(annts).intersection(zero_shot_terms_list))
                if len(insct) > 0:
                    drop_index.append(j)
            train = train.drop(index=drop_index)
            return train

        # data_path = data_dir + anno_file
        # print('Annotate {} dataset in zero shot with our model'.format(data_path))
        # anno_data = pd.read_pickle(data_path)
        anno_loader = DataLoader(anno_data, batch_size=cfg.batch_size, shuffle=True)

        print('Annotate Data with Our Model ...')
        logits, prots = [], []
        with torch.no_grad():
            for batch in anno_loader:
                pred_batch = self.model(batch['prot_seq'], batch['prot_description'],
                                        batch['prot_network'], files.text_embeddings)
                prots += batch['proteins']
                logits.append(pred_batch.cpu().numpy())
        logits = np.concatenate(logits, axis=0)

        # if cfg.task == 'zero_shot':
        zero_shot_terms = zero_shot_terms()
        zero_shot_data(zero_shot_terms, files.raw_train)
        logits_ont = collections.OrderedDict()
        for ont in ['bp', 'mf', 'cc']:
            # extract the predictions and labels of zero shot terms
            ont_terms = get_namespace_terms(list(files.zero_shot_terms.keys()), files.go_data, ont)
            logits_ont[ont] = term2preds_label(logits, ont_terms, files.terms2i)
        return logits_ont

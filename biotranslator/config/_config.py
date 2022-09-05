class BioConfig:
    def __init__(self, data_type, args: dict):
        # load args
        self.tp = data_type
        self.load_args(args)
        # load dir of the dataaset
        data_dir = self.data_repo + self.dataset + '/'
        print("data_dir: ", data_dir)
        # load dataset file
        if self.tp == "graph":
            self.go_file, self.train_file, _, _, \
            self.train_terms_file, _, self.train_prot_network_file, \
            self.train_prot_description_file = self.read_dataset_file(data_dir)
        elif self.tp == "seq":
            self.go_file, _, self.train_fold_file, self.validation_fold_file, \
            self.terms_file, self.zero_shot_term_path, self.prot_network_file, \
            self.prot_description_file = self.read_dataset_file(data_dir)

        if self.tp == "graph":
            # load pathway dataset file
            eval_data_dir = self.data_repo + self.eval_dataset + '/'
            _, _, _, _, \
            self.eval_terms_file, _, self.eval_prot_network_file, \
            self.eval_prot_description_file = self.read_dataset_file(eval_data_dir)
            self.eval_file = eval_data_dir + 'pathway_dataset.pkl'
        # generate other parameters
        self.gen_other_parameters()

    def load_args(self, args):
        # load the settings in args
        # Note: eval_dataset, task, ontology_repo, max_length, excludes
        self.dataset = args['dataset'].strip()
        self.eval_dataset = args['eval_dataset'].strip() if 'eval_dataset' in args else ""
        self.excludes = args['graph_excludes'] if 'graph_excludes' in args else ""
        self.method = 'BioTranslator'
        self.data_repo = args['data_repo'].strip()
        self.encoder_path = args['encoder_path'].strip()
        self.ontology_repo = args['vec_ontology_repo'].strip() if 'vec_ontology_repo' in args else ""
        self.emb_dir = args['emb_dir'].strip()
        self.task = args['task'].strip() if 'task' in args else ""
        self.ws_dir = args['ws_dir'].strip()
        self.rst_dir = args['rst_dir'].strip()
        self.max_length = args['max_length'] if 'max_length' in args else -1
        self.hidden_dim = args['hidden_dim']
        self.features = args['features'].split(', ') if 'features' in args else ""
        self.lr = args['lr']
        self.epoch = args['epoch']
        self.batch_size = args['batch_size']
        self.gpu_ids = args['gpu_ids'].strip()

    def read_dataset_file(self, data_dir: str):
        go_file = data_dir + 'go.obo'
        train_file = data_dir + 'dataset.pkl'
        train_fold_file = data_dir + 'train_data_fold_{}.pkl'
        validation_fold_file = data_dir + 'validation_data_fold_{}.pkl'
        terms_file = data_dir + 'terms.pkl'
        zero_shot_term_path = data_dir + 'zero_shot_terms_fold_{}.pkl'
        prot_network_file = data_dir + 'prot_network.pkl'
        prot_description_file = data_dir + 'prot_description.pkl'
        return go_file, train_file, train_fold_file, validation_fold_file, terms_file, zero_shot_term_path, prot_network_file, prot_description_file

    def gen_other_parameters(self):
        # Other parameters
        # number of amino acids of protein sequences
        self.seq_input_nc = 21
        # number of channels in the CNN architecture
        self.seq_in_nc = 512
        # the max size of CNN kernels
        self.seq_max_kernels = 129
        # the dimension of term text/graph embeddings
        self.term_enc_dim = 768
        # where you store the deep learning model
        self.save_model_path = self.ws_dir + f'{self.task}/' + 'model/{}_{}_{}_{}.pth'
        # the name of logger file, that contains the information of training process
        self.logger_name = self.ws_dir + f'{self.task}/log/{self.method}_{self.dataset}.log'
        # where you save the results of BioTranslator
        self.results_name = self.ws_dir + f'{self.task}/results/{self.method}_{self.dataset}.pkl'
        if self.tp in ['seq', 'graph']:
            if self.tp == 'seq':
                self.k_fold = 3
                # load the Diamond score related results
                if self.task == 'few_shot':
                    self.diamond_score_path = self.data_repo + self.dataset + '/validation_data_fold_{}.res'
                    self.blast_preds_path = self.data_repo + self.dataset + '/blast_preds_fold_{}.pkl'
                    # the alhpa paramter we used in DeepGOPlus
                    self.ont_term_syn = {'biological_process': 'bp', 'molecular_function': 'mf',
                                         'cellular_component': 'cc'}
                    self.alphas = {"mf": 0.68, "bp": 0.63, "cc": 0.46}
                    self.blast_res_name = self.ws_dir + f'{self.task}/results/{self.method}_{self.dataset}_blast.pkl'
            elif self.tp == 'graph':
                # select the nearest k GO term embeddings when annotate the pathway
                self.nearest_k = 5
                # get the path of pathway textual description embeddings
                self.pathway_emb_file = self.data_repo + self.eval_dataset + '/pathway_embeddings.pkl'
            # get the name of train data textual description embeddings
            self.emb_name = f'{self.method}_go_embeddings.pkl'
        elif self.tp == 'vec':
            # use expression as the features
            self.features = ['expression']
            # k-fold cross-validation
            self.n_iter = 5
            self.unseen_ratio = [0.9, 0.7, 0.5, 0.3, 0.1]
            self.nfold_sample = 0.2
            # the dropout
            self.drop_out = 0.05
            # set the memory saving mode to True
            self.memory_saving_mode = True
            # where you store the backup files
            self.backup_file = self.ws_dir + f'{self.task}/cache/sparse_backup_raw.h5ad'
            # get the name of textual description embeddings
            self.emb_name = f'{self.method}_co_embeddings.pkl'
            # when the task is cross_dataset
            if self.task == 'cross_dataset':
                self.n_iter = 1
                self.unseen_ratio = ['cross_dataset']
                self.save_model_path = self.ws_dir + f'{self.task}/' + 'model/{}_{}_{}.pth'
                self.logger_name = self.ws_dir + f'{self.task}/log/{self.method}_{self.dataset}_{self.eval_dataset}.log'
                self.eval_backup_file = self.ws_dir + f'{self.task}/cache/sparse_backup_eval.h5ad'
                self.results_name = self.ws_dir + f'{self.task}/results/{self.method}_{self.dataset}_{self.eval_dataset}.pkl'

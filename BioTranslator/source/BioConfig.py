class BioConfig:
    def __init__(self, args: dict):
        # load args
        self.load_args(args)
        # load dir of the dataset
        data_dir = self.data_repo + self.dataset + '/'
        # load dataset file
        self.go_file, self.train_fold_file, self.validation_fold_file, \
        self.terms_file, self.zero_shot_term_path, self.prot_network_file, \
        self.prot_description_file = self.read_dataset_file(data_dir)
        # generate other parameters
        self.gen_other_parameters()

    def load_args(self, args: dict):
        """load the settings"""
        self.dataset = args['dataset'].strip()
        self.method = args['method'].strip()
        self.task = args['task'].strip()
        self.data_repo = args['data_repo'].strip()
        self.encoder_path = args['encoder_path'].strip()
        self.emb_path = args['emb_path'].strip()
        self.working_space = args['working_space'].strip()
        self.save_path = args['save_path'].strip()
        self.max_length = args['max_length']
        self.hiddle_dim = args['hidden_dim']
        self.features = args['features'].split(', ')
        self.lr = args['lr']
        self.epoch = args['epoch']
        self.batch_size = args['batch_size']
        self.gpu_ids = args['gpu_ids'].strip()
        self.use_multi_gpus = args['use_multi_gpus']

    def read_dataset_file(self, data_dir: str):
        go_file = data_dir + 'go.obo'
        # Changeable
        train_fold_file = data_dir + 'train_data_fold_{}.pkl'
        # Changeable
        validation_fold_file = data_dir + 'validation_data_fold_{}.pkl'
        terms_file = data_dir + 'terms.pkl'
        # Changeable
        zero_shot_term_path = data_dir + 'zero_shot_term_fold_{}.pkl'
        prot_network_file = data_dir + 'prot_network.pkl'
        prot_description_file = data_dir + 'prot_description.pkl'
        return go_file, train_fold_file, validation_fold_file, terms_file, \
            zero_shot_term_path, prot_network_file, prot_description_file

    def gen_other_parameters(self):
        """Generate other parameters."""
        # k-fold cross-validation
        self.k_fold = 3
        # number of amino acids of protein sequences
        self.seq_input_nc = 21
        # number of channels in the CNN architecture
        self.seq_in_nc = 512
        # the max size of CNN kernels
        self.seq_max_kernels = 129
        # the dimension of term text/graph embeddings
        if self.method in ['BioTranslator', 'ProTranslator']:
            self.term_enc_dim = 768
        elif self.method == 'clusDCA':
            self.term_enc_dim = 500
        else:
            # TF-IDF, Word2Vec, Doc2Vec
            self.term_enc_dim = 1000
        # load the Diamond score related results
        if self.task == 'few_shot':
            # Changeable
            self.diamond_score_path = self.data_repo + self.dataset + '/validation_data_fold_{}.res'
            # Changeable
            self.blast_preds_path = self.data_repo + self.dataset + '/blast_preds_fold_{}.pkl'
            self.ont_terms_syn = {'biological_process': 'bp',
                                  'molecular_function': 'mf',
                                  'cellular_component': 'cc'}
            # the alpha parameter used in DeepGoPlus
            # Yunwei 0714: in the paper of DeepGoPlus, \
            # they mentioned the parameters to be 0.55, 0.59, 0.46
            self.alphas = {'mf': 0.68, 'bp': 0.63, "cc": 0.46}
            self.blast_res_name = f'{self.working_space}{self.task}/results/' \
                                  f'{self.method}_{self.dataset}_blast.pkl'
        # the path where you store the deep learning model, Changeable
        self.save_model_path = f'{self.working_space}{self.task}/' + 'model/{}_{}_{}.pth'
        # the name of logger file, that contains the information of training process
        self.logger_name = f'{self.working_space}{self.task}/log/{self.method}_{self.dataset}.log'
        # the path where you save the results of BioTranslator
        self.results_name = f'{self.working_space}{self.task}/results/' \
                            f'{self.method}_{self.dataset}.pkl'
        # the name of textual description embeddings
        self.emb_name = f'{self.method}_go_embeddings.pkl'

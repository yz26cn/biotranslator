import torch
import logging
import collections
from .config import BioConfig
from .loader import BioLoader
from .trainer import build_trainer
from torch.utils.data import DataLoader
from .text_encoder import NeuralNetwork as nn_config
from transformers import AutoTokenizer, AutoConfig
from .text_encoder import TrainOntologyDataset, get_data, train

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
    print("Using {} device".format('cuda'))
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

    model = nn_config(bert_name, output_way, config).to('cuda')
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    train_texts, test_texts = get_data(data_dir)

    train_data = TrainOntologyDataset(train_texts, tokenizer, max_len)
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    test_data = TrainOntologyDataset(test_texts, tokenizer, max_len)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    epochs = 1
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, test_dataloader, model, optimizer, save_path, device='cuda')
    print("Train Done!")


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
    translators = collections.OrderedDict()
    # We may only train one data type each time in this method
    for cfg in cfgs:
        if cfg.tp in ['graph', 'seq', 'vec']:
            files = BioLoader(cfg)
        else:
            logging.info('Data type is not supported yet.')
            raise NotImplementedError
        trainer_dict[cfg.tp] = (build_trainer(files=files, cfg=cfg), files, cfg)
    for key, trainer_tup in trainer_dict.items():
        torch.cuda.set_device(eval(trainer_tup[2].gpu_ids))
        trainer = trainer_tup[0]
        print(f'Train encoder for {trainer_tup[2].tp} data:')
        trainer.train(trainer_tup[1], trainer_tup[2])
        translators[key] = trainer
    return translators


def test_biotranslator(data_dir, anno_data, cfg, translator, task):
    """
    Annotate the proteins with textual description embeddings

    Parameters
    ----------
    data_dir: Input data path.
    anno_data: data needs to be annotated
    cfg: config
    translator: biotranslator encoder
    task: task name

    Returns
    -------
    """
    files = BioLoader(cfg)
    anno = translator.annotate(files, cfg, data_dir, anno_data, task)
    return anno

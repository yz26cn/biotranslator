# Tutorial

#### Import BioTranslator as
```
import biotranslator as bt
```

#### Import methods from API
```
from biotranslator.biotranslator_function import setup_config, train_text_encoder, train_biotranslator, \
    test_biotranslator
```


#### Train Text Encoder
```
model_path = './TextEncoder/model/encoder.pth'
graphine_repo = './TextEncoder/data/Graphine/dataset/'
train_text_encoder(graphine_repo, model_path)
```


#### Build Configs
Build config for protein sequence dataset:
```
seq_repo = f'./Protein'
seq_config = {
    'task': 'few_shot',
    'max_length': 2000,
    'data_repo': f'{seq_repo}/data/',
    'dataset': 'GOA_Human',
    'encoder_path': './Codebase/TextEncoder/Encoder/text_encoder.pth',
    'rst_dir': f'{seq_repo}/results/',
    'emb_dir': f'{seq_repo}/embeddings/',
    'ws_dir': f'{seq_repo}/working_space/',
    'hidden_dim': 1500,
    'features': 'seqs, description, network',
    'lr': 0.0003,
    'epoch': 30,
    'batch_size': 32,
    'gpu_ids': '0',
}

seq_config = setup_config(seq_config, 'seq')
```

Build config for cell description vector dataset:
```
vec_repo = f'./SingleCell'
vec_config = {
    'task': 'cross_dataset',
    'eval_dataset': 'muris_facs',
    'vec_ontology_repo': f'{vec_repo}/data/Ontology_data/',
    'data_repo': f'{vec_repo}/data/sc_data/',
    'dataset': 'muris_droplet',
    'encoder_path': './Codebase/TextEncoder/Encoder/text_encoder.pth',
    'rst_dir': f'{vec_repo}/results/',
    'emb_dir': f'{vec_repo}/embeddings/',
    'ws_dir': f'{vec_repo}/working_space/',
    'hidden_dim': 30,
    'lr': 0.0001,
    'epoch': 15,
    'batch_size': 128,
    'gpu_ids': '0',
}

vec_config = setup_config(vec_config, 'vec')
```

Build config for pathway graph dataset:
```
graph_repo = f'./Pathway'
graph_config = {
    'max_length': 2000,
    'eval_dataset': 'KEGG',
    'graph_excludes': ['Reactome', 'KEGG', 'PharmGKB'],
    'data_repo': f'{graph_repo}/data/',
    'dataset': 'GOA_Human',
    'encoder_path': './Codebase/TextEncoder/Encoder/text_encoder.pth',
    'rst_dir': f'{graph_repo}/results/',
    'emb_dir': f'{graph_repo}/embeddings/',
    'ws_dir': f'{graph_repo}/working_space/',
    'hidden_dim': 1500,
    'features': 'seqs, description, network',
    'lr': 0.0003,
    'epoch': 30,
    'batch_size': 32,
    'gpu_ids': '0',
}

graph_config = setup_config(graph_config, 'graph')
```

#### Train BioTranlators
```
cfgs = [seq_config, vec_config, graph_config]
translators = train_biotranslator(cfgs)
```

#### Test BioTranslators
```
tasks = dict(
        seq=['prot_func_pred'],
        vec=['cell_type_cls'],
        graph=['node_cls', 'edge_pred'])
    vec_files = BioLoader(vec_config)
    anno_data = dict(
        seq=[pd.read_pickle(f'{seq_config.data_repo}{seq_config.dataset}/validation_data_fold_0.pkl')],
        vec=[vec_files.test_data],
        graph=[pd.read_pickle(f'{graph_config.data_repo}{graph_config.eval_dataset}/pathway_dataset.pkl'),
               pd.read_pickle(f'{graph_config.data_repo}{graph_config.eval_dataset}/pathway_dataset.pkl')],
    )
    for tp_idx, tp in enumerate(list(tasks.keys())):
        for task_idx in range(len(tasks[tp])):
            cfg = cfgs[tp_idx]
            encoder = translators[tp_idx]
            annos = test_biotranslator(cfg.data_repo, anno_data[tp][task_idx], cfg, encoder, tasks[tp][task_idx])
    print(annos)
```
from ._graph_trainer import GraphTrainer
from ._sequence_trainer import SeqTrainer
from ._vector_trainer import VecTrainer


def build_trainer(files, cfg):
    if cfg.tp == 'graph':
        return GraphTrainer(files, cfg)
    elif cfg.tp == 'seq':
        return SeqTrainer(files, cfg)
    elif cfg.tp == 'vec':
        return VecTrainer(files, cfg)
    else:
        raise NotImplementedError

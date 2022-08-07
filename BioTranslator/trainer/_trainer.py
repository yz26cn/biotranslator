from ._graph_trainer import GraphTrainer
from ._sequence_trainer import SeqTrainer
from ._vector_trainer import VecTrainer


def build_trainer(data_type, files, cfg):
    if data_type == 'graph':
        return GraphTrainer(files, cfg)
    elif data_type == 'seq':
        return SeqTrainer(files, cfg)
    elif data_type == 'vec':
        return VecTrainer(files, cfg)
    else:
        raise NotImplementedError

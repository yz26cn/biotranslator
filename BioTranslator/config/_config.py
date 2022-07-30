from ._sequence_config import SeqConfig
from ._vector_config import VectorConfig
from ._graph_config import GraphConfig


def config(data_type, model_args):
    if data_type == 'Sequence':
        return SeqConfig(model_args)
    elif data_type == 'Graph':
        return GraphConfig(model_args)
    elif data_type == 'Vector':
        return VectorConfig(model_args)
    else:
        raise NotImplementedError

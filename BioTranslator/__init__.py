"""BioTranslator in Python."""
__version__ = '0.1'

from .biotranslator_function import setup_config, train_text_encoder, get_ontology_embeddings, train_biotranslator, \
    test_biotranslator
from .biotranslator import BioTranslator
from .config import BioConfig
from .loader import BioLoader
from .trainer import build_trainer
"""BioTranslator in Python."""
__version__ = '0.1.1'

from . import biotranslator as bt
from . import config, loader, text_encoder, trainer, utils

from .biotranslator_function import setup_config, train_text_encoder, train_biotranslator, \
    test_biotranslator
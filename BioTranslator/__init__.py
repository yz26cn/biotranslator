"""BioTranslator in Python."""

__version__ = '0.1'


if not within_flit():
    from . import biotranslator as bt
    from . import config, loader, metrics, text_encoder, trainer, utils

    from .biotranslator_function import setup_config, train_text_encoder, get_ontology_embeddings, train_biotranslator, \
        test_biotranslator
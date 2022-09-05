from datetime import datetime

project = 'BioTranslator'
copyright = f'{datetime.now():%Y}, BioTranslator team.'
author = 'BioTranslator team'

# -- General configuration ---------------------------------------------------
extensions = [
    'myst_parser'
]

templates_path = ['_templates']
master_doc = 'index'
exclude_patterns = []
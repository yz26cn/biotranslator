import io
from setuptools import setup, find_packages

with io.open('./README.md', encoding='utf-8') as f:
    readme = f.read()

setup(
    name = "BioTranslator",
    version = "0.1.2",
    keywords = ("pip", "BioTranslator"),
    description = "BioTranslator: Cross-modal Translation for Zero-shot Biomedical Classification",
    long_description = readme,
    license = "MIT Licence",

    url = "https://github.com/ywzhao2002/biotranslator",
    author = "Sheng Wang",
    author_email = "swang91@uw.edu",

    packages = find_packages(),
    include_package_data = True,
    platforms = "any",
    install_requires=[
        "sentence-transformers",
        "anndata>=0.6.4",
        "fbpca>=1.0",
        "umap-learn>=0.3.10",
        "matplotlib>=2.0.2",
        "numpy>=1.16.4",
        "scipy>=1.3.1",
        "scikit-learn>=0.21.3",
        "scanorama>=1.7.1",
        "torch==1.7.1",
        "pandas",
        "tqdm",
        "networkx",
        "gensim",
        "nltk",
        "torchvision==0.8.2",
    ]
)

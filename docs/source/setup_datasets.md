# Setup Dataset

Processed datasets including *CAFA3*, *GOA_Human*, *GOA_Mouse*, *GOA_Yeast*, *KEGG*, *PharmGKB*, *Reactome*, and *Swissprot* are available at
<https://figshare.com/articles/dataset/Protein_Pathway_data_tar/20120447>

Processed datasets including *Tabula_Microcebus* and *Tabula_Sapiens* can be found at: <https://figshare.com/ndownloader/files/31777475> and <https://figshare.com/ndownloader/files/28846647>
. Remaining datasets can be found from [OnClass](https://onclass.readthedocs.io/en/latest/introduction.html) package.

Graphine dataset used for training text encoder can be downloaded from <https://zenodo.org/record/5320310/files/Graphine.zip?download=1>.


#### Example dataset structure for protein sequence prediction and pathway analysis tasks

```
├── data
│   ├── CAFA3
│   ├── GOA_Human
│   ├── GOA_Mouse
│   ├── GOA_Yeast
│   ├── KEGG
│   ├── PharmGKB
│   ├── Reactome
│   └── SwissProt
```

#### Example dataset structure for single cell classification task
```
├── data
│   ├── ont_data
│   │   ├── allen.ontology
│   │   ├── cl.obo
│   │   ├── cl.ontology
│   │   └── cl.ontology.nlp.emb
│   ├── sc_data
│   │   ├── 26-datasets
│   │   │   ├── 293t_jurkat
│   │   │   ├── brain
│   │   │   ├── hsc
│   │   │   ├── macrophage
│   │   │   ├── pancreas
│   │   │   └── pbmc
│   │   ├── Allen_Brain
│   │   │   ├── features.pkl
│   │   │   ├── genes.pkl
│   │   │   └── labels.pkl
│   │   ├── gene_marker_expert_curated.txt
│   │   ├── HLCA
│   │   │   ├── 10x_features.pkl
│   │   │   ├── 10x_genes.pkl
│   │   │   └── 10x_labels.pkl
│   │   ├── Lemur
│   │   │   ├── microcebusAntoine.h5ad
│   │   │   ├── microcebusBernard.h5ad
│   │   │   ├── microcebusMartine.h5ad
│   │   │   └── microcebusStumpy.h5ad
│   │   ├── Tabula_Microcebus
│   │   │   └── LCA_complete_wRaw_toPublish.h5ad
│   │   ├── Tabula_Muris_Senis
│   │   │   ├── tabula-muris-senis-droplet-official-raw-obj.h5ad
│   │   │   └── tabula-muris-senis-facs-official-raw-obj.h5ad
│   │   └── Tabula_Sapiens
│   │       └── TabulaSapiens.h5ad
```

#### Example dataset structure for text encoder
```
├── data
│   ├── Graphine
│   │   └── dataset
```
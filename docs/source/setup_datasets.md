# Download datasets

The datasets used for protein function prediction and pathway analysis are available at:
https://figshare.com/articles/dataset/Protein_Pathway_data_tar/20120447


The processed
datasets for cell type classification are available at:
https://figshare.com/ndownloader/files/28846647 and
https://figshare.com/ndownloader/files/31777475. Other datasets used for single cell analysis
are obtained from OnClass.


## Example dataset structure for protein sequence prediction and pathway analysis:

```
├── data
│   ├── CAFA3
│   │   ├── dataset.pkl
│   │   ├── go.obo
│   │   ├── prot_description.pkl
│   │   ├── prot_network.pkl
│   │   ├── terms.pkl
│   │   ├── train_data_fold_0.fa
│   │   ├── train_data_fold_0.pkl
│   │   ├── train_data_fold_1.fa
│   │   ├── train_data_fold_1.pkl
│   │   ├── train_data_fold_2.fa
│   │   ├── train_data_fold_2.pkl
│   │   ├── validation_data_fold_0.fa
│   │   ├── validation_data_fold_0.pkl
│   │   ├── validation_data_fold_1.fa
│   │   ├── validation_data_fold_1.pkl
│   │   ├── validation_data_fold_2.fa
│   │   ├── validation_data_fold_2.pkl
│   │   ├── zero_shot_terms_fold_0.pkl
│   │   ├── zero_shot_terms_fold_1.pkl
│   │   └── zero_shot_terms_fold_2.pkl
│   ├── GOA_Human
│   │   ├── blast_preds_fold_0.pkl
│   │   ├── blast_preds_fold_1.pkl
│   │   ├── blast_preds_fold_2.pkl
│   │   ├── dataset.pkl
│   │   ├── go.obo
│   │   ├── prot_description.pkl
│   │   ├── prot_network.pkl
│   │   ├── terms.pkl
│   │   ├── train_data_fold_0.dmnd
│   │   ├── train_data_fold_0.fa
│   │   ├── train_data_fold_0.pkl
│   │   ├── train_data_fold_1.dmnd
│   │   ├── train_data_fold_1.fa
│   │   ├── train_data_fold_1.pkl
│   │   ├── train_data_fold_2.dmnd
│   │   ├── train_data_fold_2.fa
│   │   ├── train_data_fold_2.pkl
│   │   ├── validation_data_fold_0.fa
│   │   ├── validation_data_fold_0.pkl
│   │   ├── validation_data_fold_0.res
│   │   ├── validation_data_fold_1.fa
│   │   ├── validation_data_fold_1.pkl
│   │   ├── validation_data_fold_1.res
│   │   ├── validation_data_fold_2.fa
│   │   ├── validation_data_fold_2.pkl
│   │   ├── validation_data_fold_2.res
│   │   ├── zero_shot_terms_fold_0.pkl
│   │   ├── zero_shot_terms_fold_1.pkl
│   │   └── zero_shot_terms_fold_2.pkl
│   ├── GOA_Mouse
│   │   ├── dataset.pkl
│   │   ├── go.obo
│   │   ├── prot_description.pkl
│   │   ├── prot_network.pkl
│   │   ├── terms.pkl
│   │   ├── train_data_fold_0.fa
│   │   ├── train_data_fold_0.pkl
│   │   ├── train_data_fold_1.fa
│   │   ├── train_data_fold_1.pkl
│   │   ├── train_data_fold_2.fa
│   │   ├── train_data_fold_2.pkl
│   │   ├── validation_data_fold_0.fa
│   │   ├── validation_data_fold_0.pkl
│   │   ├── validation_data_fold_1.fa
│   │   ├── validation_data_fold_1.pkl
│   │   ├── validation_data_fold_2.fa
│   │   ├── validation_data_fold_2.pkl
│   │   ├── zero_shot_terms_fold_0.pkl
│   │   ├── zero_shot_terms_fold_1.pkl
│   │   └── zero_shot_terms_fold_2.pkl
│   ├── GOA_Yeast
│   │   ├── dataset.pkl
│   │   ├── go.obo
│   │   ├── prot_description.pkl
│   │   ├── prot_network.pkl
│   │   ├── terms.pkl
│   │   ├── train_data_fold_0.fa
│   │   ├── train_data_fold_0.pkl
│   │   ├── train_data_fold_1.fa
│   │   ├── train_data_fold_1.pkl
│   │   ├── train_data_fold_2.fa
│   │   ├── train_data_fold_2.pkl
│   │   ├── validation_data_fold_0.fa
│   │   ├── validation_data_fold_0.pkl
│   │   ├── validation_data_fold_1.fa
│   │   ├── validation_data_fold_1.pkl
│   │   ├── validation_data_fold_2.fa
│   │   ├── validation_data_fold_2.pkl
│   │   ├── zero_shot_terms_fold_0.pkl
│   │   ├── zero_shot_terms_fold_1.pkl
│   │   └── zero_shot_terms_fold_2.pkl
│   ├── KEGG
│   │   ├── pathway_dataset.pkl
│   │   ├── pathway_embeddings.pkl
│   │   ├── pathway_graph.pkl
│   │   ├── prot_description.pkl
│   │   ├── prot_network.pkl
│   │   ├── protranslator_pathway_embeddings.pkl
│   │   ├── terms.pkl
│   │   ├── train_data_fold_0.pkl
│   │   ├── train_data_fold_1.pkl
│   │   ├── train_data_fold_2.pkl
│   │   ├── validation_data_fold_0.pkl
│   │   ├── validation_data_fold_1.pkl
│   │   ├── validation_data_fold_2.pkl
│   │   ├── zero_shot_terms_fold_0.pkl
│   │   ├── zero_shot_terms_fold_1.pkl
│   │   └── zero_shot_terms_fold_2.pkl
│   ├── PharmGKB
│   │   ├── pathway_dataset.pkl
│   │   ├── pathway_embeddings.pkl
│   │   ├── prot_description.pkl
│   │   ├── prot_network.pkl
│   │   ├── protranslator_pathway_embeddings.pkl
│   │   ├── terms.pkl
│   │   ├── train_data_fold_0.pkl
│   │   ├── train_data_fold_1.pkl
│   │   ├── train_data_fold_2.pkl
│   │   ├── validation_data_fold_0.pkl
│   │   ├── validation_data_fold_1.pkl
│   │   ├── validation_data_fold_2.pkl
│   │   ├── zero_shot_terms_fold_0.pkl
│   │   ├── zero_shot_terms_fold_1.pkl
│   │   └── zero_shot_terms_fold_2.pkl
│   ├── Reactome
│   │   ├── pathway_dataset.pkl
│   │   ├── pathway_embeddings.pkl
│   │   ├── prot_description.pkl
│   │   ├── prot_network.pkl
│   │   ├── protranslator_pathway_embeddings.pkl
│   │   └── terms.pkl
│   └── SwissProt
│       ├── dataset.pkl
│       ├── go.obo
│       ├── prot_description.pkl
│       ├── prot_network.pkl
│       ├── terms.pkl
│       ├── train_data_fold_0.fa
│       ├── train_data_fold_0.pkl
│       ├── train_data_fold_1.fa
│       ├── train_data_fold_1.pkl
│       ├── train_data_fold_2.fa
│       ├── train_data_fold_2.pkl
│       ├── validation_data_fold_0.fa
│       ├── validation_data_fold_0.pkl
│       ├── validation_data_fold_1.fa
│       ├── validation_data_fold_1.pkl
│       ├── validation_data_fold_2.fa
│       ├── validation_data_fold_2.pkl
│       ├── zero_shot_terms_fold_0.pkl
│       ├── zero_shot_terms_fold_1.pkl
│       └── zero_shot_terms_fold_2.pkl
```

## Example dataset structure for single cell classification task:
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
│   │   │   │   ├── 293t
│   │   │   │   │   ├── barcodes.tsv
│   │   │   │   │   ├── genes.tsv
│   │   │   │   │   ├── matrix.mtx
│   │   │   │   │   ├── tab.genes.txt
│   │   │   │   │   └── tab.npz
│   │   │   │   ├── jurkat
│   │   │   │   │   ├── barcodes.tsv
│   │   │   │   │   ├── genes.tsv
│   │   │   │   │   ├── matrix.mtx
│   │   │   │   │   ├── tab.genes.txt
│   │   │   │   │   └── tab.npz
│   │   │   │   ├── jurkat_293t_50_50
│   │   │   │   │   ├── barcodes.tsv
│   │   │   │   │   ├── genes.tsv
│   │   │   │   │   ├── matrix.mtx
│   │   │   │   │   ├── tab.genes.txt
│   │   │   │   │   └── tab.npz
│   │   │   │   └── jurkat_293t_99_1
│   │   │   │       ├── barcodes.tsv
│   │   │   │       ├── genes.tsv
│   │   │   │       ├── matrix.mtx
│   │   │   │       ├── tab.genes.txt
│   │   │   │       └── tab.npz
│   │   │   ├── brain
│   │   │   │   └── neuron_9k
│   │   │   │       ├── barcodes.tsv
│   │   │   │       ├── genes.tsv
│   │   │   │       ├── matrix.mtx
│   │   │   │       ├── tab.genes.txt
│   │   │   │       └── tab.npz
│   │   │   ├── hsc
│   │   │   │   ├── hsc_mars.npz
│   │   │   │   ├── hsc_mars.txt
│   │   │   │   ├── hsc_ss2.npz
│   │   │   │   └── hsc_ss2.txt
│   │   │   ├── macrophage
│   │   │   │   ├── infected.npz
│   │   │   │   ├── infected.txt
│   │   │   │   ├── mcsf_day3_1.txt
│   │   │   │   ├── mcsf_day3_2.txt
│   │   │   │   ├── mcsf_day6_1.txt
│   │   │   │   ├── mcsf_day6_2.txt
│   │   │   │   ├── mixed_infected.npz
│   │   │   │   ├── mixed_infected.txt
│   │   │   │   ├── monocytes.txt
│   │   │   │   ├── mono_macro_corrected_table.txt
│   │   │   │   ├── mono_macro_diffexpr_mnn.txt
│   │   │   │   ├── mono_macro_diffexpr_scanorama.txt
│   │   │   │   ├── mono_macro_diffexpr_uncorrected.txt
│   │   │   │   ├── mono_macro_genes.txt
│   │   │   │   ├── mono_macro_hours.txt
│   │   │   │   ├── mono_macro_meta.txt
│   │   │   │   ├── mono_macro_mnn_corrected_table.txt
│   │   │   │   ├── mono_macro_table.txt
│   │   │   │   ├── uninfected_donor2.npz
│   │   │   │   ├── uninfected_donor2.txt
│   │   │   │   ├── uninfected.npz
│   │   │   │   └── uninfected.txt
│   │   │   ├── pancreas
│   │   │   │   ├── pancreas_inDrop.npz
│   │   │   │   ├── pancreas_inDrop.txt
│   │   │   │   ├── pancreas_multi_celseq2_expression_matrix.npz
│   │   │   │   ├── pancreas_multi_celseq2_expression_matrix.txt
│   │   │   │   ├── pancreas_multi_celseq_expression_matrix.npz
│   │   │   │   ├── pancreas_multi_celseq_expression_matrix.txt
│   │   │   │   ├── pancreas_multi_fluidigmc1_expression_matrix.npz
│   │   │   │   ├── pancreas_multi_fluidigmc1_expression_matrix.txt
│   │   │   │   ├── pancreas_multi_smartseq2_expression_matrix.npz
│   │   │   │   └── pancreas_multi_smartseq2_expression_matrix.txt
│   │   │   └── pbmc
│   │   │       ├── 10x
│   │   │       │   ├── 68k_pbmc
│   │   │       │   │   ├── barcodes.tsv
│   │   │       │   │   ├── genes.tsv
│   │   │       │   │   ├── matrix.mtx
│   │   │       │   │   ├── tab.genes.txt
│   │   │       │   │   └── tab.npz
│   │   │       │   ├── b_cells
│   │   │       │   │   ├── barcodes.tsv
│   │   │       │   │   ├── genes.tsv
│   │   │       │   │   ├── matrix.mtx
│   │   │       │   │   ├── tab.genes.txt
│   │   │       │   │   └── tab.npz
│   │   │       │   ├── cd14_monocytes
│   │   │       │   │   ├── barcodes.tsv
│   │   │       │   │   ├── genes.tsv
│   │   │       │   │   ├── matrix.mtx
│   │   │       │   │   ├── tab.genes.txt
│   │   │       │   │   └── tab.npz
│   │   │       │   ├── cd4_t_helper
│   │   │       │   │   ├── barcodes.tsv
│   │   │       │   │   ├── genes.tsv
│   │   │       │   │   ├── matrix.mtx
│   │   │       │   │   ├── tab.genes.txt
│   │   │       │   │   └── tab.npz
│   │   │       │   ├── cd56_nk
│   │   │       │   │   ├── barcodes.tsv
│   │   │       │   │   ├── genes.tsv
│   │   │       │   │   ├── matrix.mtx
│   │   │       │   │   ├── tab.genes.txt
│   │   │       │   │   └── tab.npz
│   │   │       │   ├── cytotoxic_t
│   │   │       │   │   ├── barcodes.tsv
│   │   │       │   │   ├── genes.tsv
│   │   │       │   │   ├── matrix.mtx
│   │   │       │   │   ├── tab.genes.txt
│   │   │       │   │   └── tab.npz
│   │   │       │   ├── memory_t
│   │   │       │   │   ├── barcodes.tsv
│   │   │       │   │   ├── genes.tsv
│   │   │       │   │   ├── matrix.mtx
│   │   │       │   │   ├── tab.genes.txt
│   │   │       │   │   └── tab.npz
│   │   │       │   └── regulatory_t
│   │   │       │       ├── barcodes.tsv
│   │   │       │       ├── genes.tsv
│   │   │       │       ├── matrix.mtx
│   │   │       │       ├── tab.genes.txt
│   │   │       │       └── tab.npz
│   │   │       ├── pbmc_10X.npz
│   │   │       ├── pbmc_10X.txt
│   │   │       ├── pbmc_kang.npz
│   │   │       └── pbmc_kang.txt
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
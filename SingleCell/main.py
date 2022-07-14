import torch
import argparse
from BioTranslator.BioConfig import BioConfig
from BioTranslator.BioLoader import BioLoader
from BioTranslator.BioTrainer import BioTrainer


def main(cfg: BioConfig):
    torch.cuda.set_device(eval(cfg.gpu_ids))
    files = BioLoader(cfg)
    trainer = BioTrainer()
    trainer.train(files, cfg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # choose method and dataset
    parser.add_argument('--method', type=str, default='BioTranslator', help='Please set this value to BioTranslator when running this script')
    parser.add_argument('--dataset', type=str, default='muris_droplet', help='Specify the dataset for cross-validation, choose between sapiens, tabula_microcebus, muris_droplet, microcebusAntoine, microcebusBernard, microcebusMartine, microcebusStumpy, muris_facs')
    # The eval dataset option works only in the cross_dataset task
    parser.add_argument('--eval_dataset', type=str, default='muris_facs', help='Specify the dataset for cross-validation, choose between sapiens, tabula_microcebus, muris_droplet, microcebusAntoine, microcebusBernard, microcebusMartine, microcebusStumpy, muris_facs')
    # choose the task you want evaluate:  same_dataset task or cross_dataset
    parser.add_argument('--task', type=str, default='cross_dataset', help='Choose between same_dataset task and cross_dataset task')
    # specify the dataset root dir
    parser.add_argument('--data_repo', type=str, default='./data/sc_data/', help='Where you store the single cell dataset.')
    # specify the ontology data path
    parser.add_argument('--ontology_repo', type=str, default='./data/Ontology_data/', help='Where you store the Cell Ontology data.')
    # Specify the encoder model path, this model will be used only when you do not have embeddings in emb_path
    parser.add_argument('--encoder_path', type=str, default='../TextEncoder/Encoder/text_encoder.pth', help='The path of text encoder model')
    # please specify where you cache the cell ontology term embeddings
    parser.add_argument('--emb_path', type=str, default='embeddings/', help='Where you cache the embeddings.')
    # please specify the working space, that means, the files generated by our codes will be cached here
    parser.add_argument('--working_space', type=str, default='working_space/')
    # please specify where you choose to save the results, the results will be saved in the format of a dictionary
    parser.add_argument('--save_path', type=str, default='results/')
    # The following are parameters we used in our model
    parser.add_argument('--hidden_dim', type=int, default=30, help='The dimension of the second to the last layer.')
    # The follwoing are paramters used in the training process
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate. ')
    parser.add_argument('--epoch', type=int, default=15, help='Training epochs.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    # The GPU ids
    parser.add_argument('--gpu_ids', type=str, default='1', help='Specify which GPU you want to use')

    args = parser.parse_args()
    args = args.__dict__

    cfg = BioConfig(args)
    main(cfg)



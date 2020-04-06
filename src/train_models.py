import argparse

from helper.utils import *
from qa_lstm import *

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_pickle", default=None, type=str, required=True,
    help="Path to training data in .pickle format")
    parser.add_argument("--valid_pickle", default=None, type=str, required=True,
    help="Path to validation data in .pickle format")
    parser.add_argument("--model_type", default=None, type=str, required=True,
    help="Specify model type as 'qa-lstm' or 'bert'")
    parser.add_argument("--bert_model_name", default=None, type=str, required=True,
    help="Specify BERT model name to use from 'bert-base', 'finbert-domain', \
    'finbert-task', 'finbert-qa'")

    parser.add_argument("--device", default='gpu', type=str, required=False,
    help="Specify 'gpu' or 'cpu'")
    parser.add_argument("--max_seq_len", default=512, type=int, required=False,
    help="Maximum sequence length for a given input.")
    parser.add_argument("--batch_size", default=16, type=int, required=False,
    help="Batch size.")
    parser.add_argument("--n_epochs", default=3, type=int, required=False,
    help="Number of epochs.")
    parser.add_argument("--lr", default=3e-6, type=float, required=False,
    help="Number of epochs.")

    parser.add_argument("--emb_dim", default=100, type=int, required=False,
    help="Embedding dimension. Specify only if model type is 'qa_lstm'")
    parser.add_argument("--hidden_size", default=256, type=int, required=False,
    help="Hidden size. Specify only if model type is 'qa_lstm'")
    parser.add_argument("--dropout", default=0.2, type=float, required=False,
    help="Dropout rate. Specify only if model type is 'qa_lstm'")
    parser.add_argument("--margin", default=0.2, type=float, required=False,
    help="Margin for pairwise loss. Specify only if model type is 'qa_lstm' \
    or if 'learning_approach' is pairwise")


    args = parser.parse_args()

    device = torch.device('cuda' if args.device == 'gpu' else 'cpu')

    config = {
        'model_type': args.model_type,
        'device': args.device,
        'max_seq_len': args.max_seq_len,
        'batch_size': args.batch_size,
        'n_epochs': args.n_epochs,
        'lr': args.lr,
        'emb_dim': args.emb_dim,
        'hidden_size': args.hidden_size,
        'dropout': args.dropout,
        'margin:' args.margin,
        'train_set': args.train_pickle,
        'valid_set': args.valid_pickle
    }

    if config['model_type'] == 'qa-lstm':
        train_qa_lstm_model(config)
    else:
        pass

if __name__ == "__main__":
    main()

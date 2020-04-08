import argparse
import sys

from helper.utils import *
from qa_lstm import *
from finbert_qa import *

def main():
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument("--test_pickle", default=None, type=str, required=True,\
                        help="Path to test data in .pickle format")
    parser.add_argument("--label_pickle", default=None, type=str, required=True,\
                        help="Path to label data in .pickle format")
    parser.add_argument("--model_type", default=None, type=str, required=True,
    help="Specify model type as 'qa-lstm' or 'bert'")
    parser.add_argument("--max_seq_len", default=None, type=int, required=True,
    help="Maximum sequence length for a sequence.")

    # Optional arguments
    parser.add_argument("--use_trained_model", default=False, \
                        action="store_true", \
                        help="Use already fine-tuned model.")
    parser.add_argument("--bert_model_name", default="bert-qa", type=str, required=False, \
    help="Specify BERT model name from bert-base', 'finbert-domain', 'finbert-task', 'bert-qa'")
    parser.add_argument("--bert_finetuned_model", default=None, type=str, required=False,
    help="Specify the name of the fine-tuned model from bert-pointwise', 'bert-pairwise', 'finbert-domain', 'finbert-task', 'finbert-qa'")
    parser.add_argument("--model_path", default=None, type=str, required=False,
    help="Specify model path if use_trained_model is not used")
    parser.add_argument("--use_rank_pickle", default=False, \
                        action="store_true", help="Use pre-computed rank.")
    parser.add_argument("--device", default='gpu', type=str, required=False,
    help="Specify 'gpu' or 'cpu'")

    # Optional arguments when model_type is 'qa-lstm'
    parser.add_argument("--emb_dim", default=100, type=int, required=False,
    help="Embedding dimension. Specify only if model_type is 'qa-lstm'")
    parser.add_argument("--hidden_size", default=256, type=int, required=False,
    help="Hidden size. Specify only if model_type is 'qa-lstm'")
    parser.add_argument("--dropout", default=0.2, type=float, required=False,
    help="Dropout rate. Specify only if model_type is 'qa-lstm'")

    args = parser.parse_args()

    config = {'test_set': args.test_pickle,
              'labels': args.label_pickle,
              'model_type': args.model_type,
              'max_seq_len': args.max_seq_len,
              'use_trained_model': args.use_trained_model,
              'bert_model_name': args.bert_model_name,
              'bert_finetuned_model': args.bert_finetuned_model,
              'model_path': args.model_path,
              'use_rank_pickle': args.use_rank_pickle,
              'device': args.device,
              'emb_dim': args.emb_dim,
              'hidden_size': args.hidden_size,
              'dropout': args.dropout}

    # TO-DO: Catch error for invalid datasets

    if config['model_type'] == 'qa-lstm':
        evaluate_qa_lstm_model(config)
    elif config['model_type'] == 'bert':
        evaluate_bert_model(config)
    else:
        print("Please specify 'qa-lstm' or 'bert' for model_type")
        sys.exit()

if __name__ == "__main__":
    main()

import argparse
import sys

from helper.utils import *
from qa_lstm import *
from finbert_qa import *

def main():
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument("--model_type", default=None, type=str, required=True,
    help="Specify model type as 'qa-lstm' or 'bert'")
    parser.add_argument("--bert_model_name", default="bert-qa", type=str, required=True, \
    help="Specify BERT model name from bert-base', 'finbert-domain', 'finbert-task', 'bert-qa'")

    # Optional arguments
    parser.add_argument("--use_trained_model", default=False, \
                        action="store_true", \
                        help="Use already fine-tuned model.")
    parser.add_argument("--model_name", default=None, type=str, required=False,
    help="Specify the name of the trained model from 'qa-lstm', 'bert-pointwise', 'bert-pairwise', 'finbert-domain', 'finbert-task', 'finbert-qa'")
    parser.add_argument("--model_path", default=None, type=str, required=False,
    help="Specify model path if use_trained_model is not used")
    parser.add_argument("--use_rank_pickle", default=False, \
                        action="store_true", help="Use pre-computed rank.")

    args = parser.parse_args()

    config = {'model_type': args.model_type,
              'use_default_config': args.use_default_config,
              'train_set': args.train_pickle,
              'valid_set': args.valid_pickle}

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

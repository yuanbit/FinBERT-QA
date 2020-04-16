import argparse
import sys

from utils import *
from finbert_qa import *

def main():
    parser = argparse.ArgumentParser()

    # Optional arguments
    parser.add_argument("--user_input", default=False, \
                        action="store_true", \
                        help="Type query in the command line if query not specified.")
    parser.add_argument("--query", default=None, type=str, required=False,
    help="Specify query if user_input is not used.")
    parser.add_argument("--top_k", default=5, type=int, required=False, \
    help="Top-k answers to output.")
    parser.add_argument("--device", default='gpu', type=str, required=False,
    help="Specify 'gpu' or 'cpu'")


    args = parser.parse_args()

    config = {'user_input': args.user_input,
              'query': args.query,
              'top_k': args.top_k,
              'bert_model_name': 'bert-qa',
              'device': args.device,
              'max_seq_len': 512}

    FinBERT_QA(config).search()

if __name__ == "__main__":
    main()

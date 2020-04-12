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
    parser.add_argument("--k", default=5, type=int, required=False, \
    help="Top-k answers to output.")


    args = parser.parse_args()

    config = {'user_input': args.user_input,
              'query': args.query,
              'k': args.k,
              'bert_model_name': 'bert-qa',
              'device': 'gpu',
              'max_seq_len': 512}

    FinBERT_QA(config).search()

if __name__ == "__main__":
    main()

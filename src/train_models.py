import argparse
import os

from utils import *

def main():

    os.chdir("../fiqa/data/qa_lstm_tokenizer/")
    vocab = load_pickle("word2index.pickle")
    os.chdir("../../../fiqa/data/processed_data/")
    train_set = load_pickle('train_set_50.pickle')

    # parser = argparse.ArgumentParser()
    #
    # parser.add_argument("--train_pickle", default=None, type=str, required=True,
    # help="Path to training data in .pickle format")
    # parser.add_argument("--valid_pickle", default=None, type=str, required=True,
    # help="Path to validation data in .pickle format")
    # parser.add_argument("--model_type", default=None, type=str, required=True,
    # help="Specify model type as 'qa_lstm' or 'bert'")
    # parser.add_argument("--device", default='gpu', type=str, required=False,
    # help="Use GPU or CPU")
    #
    # args = parser.parse_args()
    #
    # train_set = load_pickle(args.train_pickle)
    # valid_set = load_pickle(args.valid_pickle)
    # device = torch.device('cuda' if args.device == 'gpu' else 'cpu')
    #
    # if args.model_type == 'qa_lstm'
    #     os.chdir("../fiqa/data/qa_lstm_tokenizer/")
    #     vocab = load_pickle("word2index.pickle")
    #     qid_to_tokenized_text = load_pickle('qid_to_tokenized_text.pickle')
    #     docid_to_tokenized_text = load_pickle('docid_to_tokenized_text.pickle')


#
# config = {
#     'model_type': "qa_lstm",
#     'emb_dim': 100,
#     'hidden_size': 256,
#     'dropout': 0.2,
#     'max_seq_len': 512,
#     'batch_size': 64,
#     'n_epochs': 6,
#     'learning_rate': 0.001,
#     'device': device,
#     'vocab': vocab,
#     'qid_to_tokenized_text': qid_to_tokenized_text,
#     'docid_to_tokenized_text': docid_to_tokenized_text,
#     'train_set': train_set,
#     'valid_set': valid_set
# }

if __name__ == "__main__":
    main()

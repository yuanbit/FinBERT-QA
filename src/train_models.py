import argparse

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", default='gpu', type=str, required=False,
    help="Use GPU or CPU")

    args = parser.parse_args()

    device = torch.device('cuda' if args.device == 'gpu' else 'cpu')

train_set = load_pickle('fiqa/data/data_50/train_set_50.pickle')
valid_set = load_pickle('fiqa/data/data_50/valid_set_50.pickle')

vocab = load_pickle('fiqa/data/qa_lstm_tokenizer/word2index.pickle')
qid_to_tokenized_text = load_pickle('fiqa/data/qa_lstm_tokenizer/qid_to_tokenized_text.pickle')
docid_to_tokenized_text = load_pickle('fiqa/data/qa_lstm_tokenizer/docid_to_tokenized_text.pickle')

config = {
    'model_type': "qa_lstm",
    'emb_dim': 100,
    'hidden_size': 256,
    'dropout': 0.2,
    'max_seq_len': 512,
    'batch_size': 64,
    'n_epochs': 6,
    'learning_rate': 0.001,
    'device': device,
    'vocab': vocab,
    'qid_to_tokenized_text': qid_to_tokenized_text,
    'docid_to_tokenized_text': docid_to_tokenized_text,
    'train_set': train_set,
    'valid_set': valid_set
}

if __name__ == "__main__":
    main()

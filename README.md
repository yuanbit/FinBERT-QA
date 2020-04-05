work in progress....

## Train
```
python3 src/train_models.py  [--train_pickle TRAIN_PICKLE] [--valid_pickle VALID_PICKLE] \
[--model_type MODEL_TYPE] [--device DEVICE] [--max_seq_len MAX_SEQ_LEN] [--batch_size BATCH_SIZE] \
[--n_epochs N_EPOCHS] [--lr LR] [--emb_dim EMB_DIM] [--hidden_size HIDDEN_SIZE] [--dropout DROPOUT]

Arguments:
  TRAIN_PICKLE - Path to training data in .pickle format
  VALID_PICKLE - Path to validation data in .pickle format
  MODEL_TYPE - Specify model type as 'qa_lstm' or 'bert'
  DEVICE - Specify 'gpu' or 'cpu'
  MAX_SEQ_LEN - Maximum sequence length for a given input
  BATCH_SIZE - Batch size
  N_EPOCHS - Number of epochs
  LR - Learning rate
  EMB_DIM - Embedding dimension. Specify only if model type is 'qa_lstm'
  HIDDEN_SIZE - Hidden size. Specify only if model type is 'qa_lstm'
  DROPOUT - Dropout rate. Specify only if model type is 'qa_lstm'
```

### Example - Training QA-LSTM
```
python3 src/train_models.py --train_pickle data/processed_data/train_set_50.pickle \
--valid_pickle data/processed_data/valid_set_50.pickle --model_type 'qa_lstm' \
--max_seq_len 128 --batch_size 64 --n_epochs 3 --lr 0.001
```

## Generate Training, Validation, Test Samples

```
python3 src/generate_dataset.py  [--train_cands TRAIN_CANDS] \
[--valid_cands VALID_CANDS] [--test_cands TEST_CANDS] \
[--train_label TRAIN_LABEL] [--valid_label VALID_LABEL] \
[--test_label TEST_LABEL] [--output_dir OUTPUT_DIR] [--cands_size CANDS_SIZE]

Arguments:
  TRAIN_CANDS - Path to the training candidates data in .tsv format. Each line should have three items: (questiod id, answer id, rank) separated by tab.
  VALID_CANDS - Path to the validation candidates data in .tsv format. Each line should have three items: (questiod id, answer id, rank) separated by tab.
  TEST_CANDS - Path to the testing candidates data in .tsv format. Each line should have three items: (questiod id, answer id, rank) separated by tab.
  TRAIN_LABEL - Path to the training label data in .pickle format.
  VALID_LABEL - Path to the validation label data in .pickle format.
  TEST_LABEL - Path to the testing label data in .pickle format.
  OUTPUT_DIR - The output directory where the generated data will be stored.
  CANDS_SIZE - Number of candidates per question.
```
#### Example
```
python3 src/generate_dataset.py --train_cands data/retrieval/train/cands_train_50.tsv \
--valid_cands data/retrieval/valid/cands_valid_50.tsv --test_cands data/retrieval/test/cands_test_50.tsv \
--train_label data/labels/qid_rel_train.pickle --valid_label data/labels/qid_rel_valid.pickle \
--test_label data/labels/qid_rel_test.pickle --output_dir data/processed_data \
--cands_size 50
```

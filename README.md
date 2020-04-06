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

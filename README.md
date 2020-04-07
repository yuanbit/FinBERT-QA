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

### Example - Train QA-LSTM with default data and configurations
```
python3 src/train_models.py --model_type 'qa-lstm' --use_default_config
```
```
DEFAULT_CONFIG = {'model_type': 'qa-lstm',
                  'use_default_config': True,
                  'device': 'gpu',
                  'max_seq_len': 128,
                  'batch_size': 64,
                  'n_epochs': 3,
                  'lr': 1e-3,
                  'emb_dim': 100,
                  'hidden_size': 256,
                  'dropout': 0.2,
                  'margin:' 0.2}
```


### Example - Train custom QA-LSTM
```
python3 src/train_models.py --model_type 'qa-lstm' \
--train_pickle data/sample/train_toy.pickle \
--valid_pickle data/sample/valid_toy.pickle \
--max_seq_len 256 --batch_size 128 --n_epochs 6 --lr 0.001
```

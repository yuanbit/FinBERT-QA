work in progress....

[Train](#train)

[Evaluate](#evaluate)

## Train
```
python3 src/train_models.py  [--model_type MODEL_TYPE] [--use_default_config] \
[--train_pickle TRAIN_PICKLE] [--valid_pickle VALID_PICKLE] [--device DEVICE] \
[--max_seq_len MAX_SEQ_LEN] [--batch_size BATCH_SIZE] [--n_epochs N_EPOCHS] \
[--lr LR] [--emb_dim EMB_DIM] [--hidden_size HIDDEN_SIZE] [--dropout DROPOUT] \
[--bert_model_name BERT_MODEL_NAME] [--learning approach LEARNING_APPROACH] \
[--margin MARGIN] [--weight_decay WEIGHT_DECAY] [--num_warmup_steps NUM_WARMUP_STEPS]

Arguments:
  MODEL_TYPE - Specify model type as 'qa-lstm' or 'bert'
  TRAIN_PICKLE - Path to training data in .pickle format
  VALID_PICKLE - Path to validation data in .pickle format
  DEVICE - Specify 'gpu' or 'cpu'
  MAX_SEQ_LEN - Maximum sequence length for a given input
  BATCH_SIZE - Batch size
  N_EPOCHS - Number of epochs
  LR - Learning rate
  EMB_DIM - Embedding dimension. Specify only if model_type is 'qa-lstm'
  HIDDEN_SIZE - Hidden size. Specify only if model_type is 'qa-lstm'
  DROPOUT - Dropout rate. Specify only if model_type is 'qa-lstm'
  BERT_MODEL_NAME - Specify the pre-trained BERT model to use from 'bert-base', 'finbert-domain', 'finbert-task', 'bert-qa'
  LEARNING_APPROACH - Learning approach. Specify 'pointwise' or 'pairwise' only if model_type is 'bert'
  WEIGHT_DECAY - Weight decay. Specify only if model_type is 'bert'
  NUM_WARMUP_STEPS - Number of warmup steps. Specify only if model type is 'bert'
```

### Example - Train fine-tuned BERT models with default data and configurations
```
python3 src/train_models.py --model_type 'bert' \
--use_default_config \
--bert_model_name 'bert-qa' \
--learning_approach 'pointwise'
```
```
DEFAULT_CONFIG = {'model_type': 'bert',
                  'use_default_config': True,
                  'device': 'gpu',
                  'max_seq_len': 512,
                  'batch_size': 8,
                  'n_epochs': 3,
                  'lr': 3e-6,
                  'weight_decay': 0.01,
                  'num_warmup_steps': 10000}
```
### Example - Train custom fine-tuned BERT models
```
python3 src/train_models.py --model_type 'bert' \
--train_pickle data/sample/train_sample.pickle \
--valid_pickle data/sample/valid_sample.pickle \
--bert_model_name 'finbert-domain' \
--learning_approach 'pointwise' \
--max_seq_len 64 \
--batch_size 128 \
--n_epochs 1 \
--lr 2e-5
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
--train_pickle data/sample/train_sample.pickle \
--valid_pickle data/sample/valid_sample.pickle \
--max_seq_len 256 \
--batch_size 128 \
--n_epochs 1 \
--lr 0.001
```
## Evaluate
### Example - Evaluate with fine-tuned model (FinBERT-QA)
```
python3 src/evaluate_models.py --test_pickle data/processed_data/test_set_50.pickle \
--label_pickle data/labels/qid_rel_test.pickle \
--model_type 'bert' \
--max_seq_len 512 \
--use_trained_model \
--bert_finetuned_model 'finbert-qa' 
```
### Example - Evaluate with pre-computed ranking
```
python3 src/evaluate_models.py --test_pickle data/processed_data/test_set_50.pickle 
--label_pickle data/labels/qid_rel_test.pickle 
--model_type 'bert' 
--bert_finetuned_model 'finbert-qa' 
--use_rank_pickle
```


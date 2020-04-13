# FinBERT-QA: Financial Question Answering using BERT

## Installation
If no GPU available, an alternative and low-effort way to train a QA-LSTM model and fine-tune a pre-trained BERT model for the Opionated Financial Question and Answering [FiQA](https://sites.google.com/view/fiqa) dataset is through the following [online notebooks](https://github.com/yuanbit/FinBERT-QA-notebooks) using Colab.

### With Docker
This repo can be used as a container with [Docker](https://www.docker.com/). Run the commands as root if Docker not configured.
```
docker pull yuanbit/finbert_qa
docker run --runtime=nvidia -it yuanbit/finbert_qa
```
## Usage
* [Train](#train)
* [Evaluate](#evaluate)
* [Predict](#predict)

### Train
```
python3 src/train_models.py  [--model_type MODEL_TYPE] [--train_pickle TRAIN_PICKLE] \
                             [--valid_pickle VALID_PICKLE] [--device DEVICE] \
                             [--max_seq_len MAX_SEQ_LEN] [--batch_size BATCH_SIZE] \
                             [--n_epochs N_EPOCHS] [--lr LR] [--emb_dim EMB_DIM] \
                             [--hidden_size HIDDEN_SIZE] [--dropout DROPOUT] \
                             [--bert_model_name BERT_MODEL_NAME] \
                             [--learning approach LEARNING_APPROACH] \
                             [--margin MARGIN] [--weight_decay WEIGHT_DECAY] \
                             [--num_warmup_steps NUM_WARMUP_STEPS]

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
  MARGIN - margin for pariwise loss
  WEIGHT_DECAY - Weight decay. Specify only if model_type is 'bert'
  NUM_WARMUP_STEPS - Number of warmup steps. Specify only if model type is 'bert'
```
#### Example - Train and fine-tune FinBERT-QA with pointwise learning approach
```
python3 src/train_models.py --model_type 'bert' 
                            --train_pickle data/data_pickle/train_set_50.pickle \
                            --valid_pickle data/data_pickle/valid_set_50.pickle \
                            --bert_model_name 'bert-qa' \
                            --learning_approach 'pointwise' \
                            --max_seq_len 512 \
                            --batch_size 16 \
                            --n_epochs 3 \
                            --lr 3e-6
```

#### Example - Train and fine-tune BERT further pre-trained on the FiQA task dataset with pairwise learning approach
```
python3 src/train_models.py --model_type 'bert' 
                            --train_pickle data/data_pickle/train_set_50.pickle \
                            --valid_pickle data/data_pickle/valid_set_50.pickle \
                            --bert_model_name 'finbert-task' \
                            --learning_approach 'pairwise' \
                            --max_seq_len 128 \
                            --batch_size 32 \
                            --n_epochs 3 \
                            --lr 3e-6
```

#### Example - Train QA-LSTM
```
python3 src/train_models.py --model_type 'qa-lstm' \
                            --train_pickle data/data_pickle/train_set_50.pickle \
                            --valid_pickle data/data_pickle/valid_set_50.pickle \
                            --max_seq_len 128 \
                            --batch_size 64 \
                            --n_epochs 3 \
                            --lr 1e-3
```
### Evaluate
```
python3 src/evaluate_models.py  [--model_type MODEL_TYPE] [--test_pickle TEST_PICKLE] \
                                [--bert_model_name BERT_MODEL_NAME] \
                                [--bert_finetuned_model BERT_FINETUNED_MODEL] \
                                [--model_path MODEL_PATH] [--device DEVICE] \
                                [--max_seq_len MAX_SEQ_LEN] [--emb_dim EMB_DIM] \
                                [--hidden_size HIDDEN_SIZE] [--dropout DROPOUT]
                          

Arguments:
  MODEL_TYPE - Specify model type as 'qa-lstm' or 'bert'
  TEST_PICKLE - Path to training data in .pickle format
  BERT_MODEL_NAME - Specify the pre-trained BERT model to use from 'bert-base', 'finbert-domain', 'finbert-task', 'bert-qa'
  BERT_FINETUNED_MODEL - Specify the name of the fine-tuned model from bert-pointwise', 'bert-pairwise', 'finbert-domain', 'finbert-task', 'finbert-qa'
  MODEL_PATH - Specify model path if use_trained_model is not used
  DEVICE - Specify 'gpu' or 'cpu'
  MAX_SEQ_LEN - Maximum sequence length for a given input
  EMB_DIM - Embedding dimension. Specify only if model_type is 'qa-lstm'
  HIDDEN_SIZE - Hidden size. Specify only if model_type is 'qa-lstm'
  DROPOUT - Dropout rate. Specify only if model_type is 'qa-lstm'
```
#### Example - Evaluate FinBERT-QA - fine-tuned on [Nogueira and Cho's](https://arxiv.org/pdf/1901.04085.pdf) MS MACRO model
```
python3 src/evaluate_models.py --test_pickle data/data_pickle/test_set_50.pickle \
                                --model_type 'bert' \
                                --max_seq_len 512 \
                                --bert_finetuned_model 'finbert-qa' \
                                --use_trained_model 
```
### Predict
```
python3 src/predict.py  [--user_input] [--query QUERY] [--k K]

Arguments:
  QUERY - Specify query if user_input is not used
  K - Top-k answers to output
```
#### Example - Predict with FinBERT-QA search
```
python3 src/predict.py --user_input --k 5
```

# FinBERT-QA: Financial Question Answering using BERT
FinBERT-QA is a Question Answering search engine for retrieving opinionated financial text from task 2 of the [FiQA](https://sites.google.com/view/fiqa) dataset. 

The system uses techniques from both information retrieval and natural language processing by first retrieving the top-50 answer candidates of each query using the Lucene toolkit, [Anserini](https://github.com/castorini/anserini), then re-ranking the answer candidates using variants of pre-trained [BERT](https://arxiv.org/pdf/1810.04805.pdf) models. 

The [state-of-the-art](https://dl.acm.org/doi/10.1145/3184558.3191830) results were improved by an average of 20% on three ranking evaluation metrics.

Built using Huggingface's [transformer](https://github.com/huggingface/transformers) library and the transfer and adapt [[TANDA](https://arxiv.org/pdf/1911.04118.pdf)] method, FinBERT-QA first transfers and fine-tunes a pre-trained BERT model to a general QA task, then adapts this model to the financial domain using the FiQA dataset. The transfer step uses the fine-tuned BERT model on the [MS MACRO Passage Retrieval](https://microsoft.github.io/msmarco/) dataset from [Nogueira et al.](https://arxiv.org/pdf/1901.04085.pdf), where it was converted from a TensorFlow to PyTorch model.

## Installation
If no GPU is available, an alternative and low-effort way to train and evaluate a model as well as predicting the results is through the following [online notebooks](https://github.com/yuanbit/FinBERT-QA-notebooks) using Colab.

### With Docker
This repo can be used as a container with [Docker](https://www.docker.com/). Run the commands as root if Docker not configured.

#### Docker pull command
```
docker pull yuanbit/finbert_qa
```
#### Run
```
docker run --runtime=nvidia -it yuanbit/finbert_qa
```
## Quickstart
Run to query the top-k opinionated answers from the financial domain
```
python3 src/predict.py --user_input --k 5
```
Sample questions:
```
• Getting financial advice: Accountant vs. Investment Adviser vs. Internet/self-taught?
• Are individual allowed to use accrual based accounting for federal income tax?
• What are 'business fundamentals'?
• How would IRS treat reimbursement in a later year of moving expenses?
• Can I claim mileage for traveling to a contract position?
• Tax planning for Indian TDS on international payments
• Should a retail trader bother about reading SEC filings
• Why are American Express cards are not as popular as Visa or MasterCard?
• Why do companies have a fiscal year different from the calendar year?
• Are credit histories/scores international?
```

## Data
The [raw dataset](https://sites.google.com/view/fiqa) has been cleaned and split into training, validation, and test sets in the form of lists where each sample is a list of```[question id, [label answer ids], [answer candidate ids]]```. The datasets are stored in the pickle files in ```data/data_pickle```. The generation of the datasets can be replicated by running the ```src/generate_data.py``` script, more details please see usage.

Sample QA:
```
Question: Why are big companies like Apple or Google not included in the Dow Jones Industrial Average (DJIA) index?

Answer: That is a pretty exclusive club and for the most part they are not interested in highly volatile companies like Apple and Google. Sure, IBM is part of the DJIA, but that is about as stalwart as you can get these days. The typical profile for a DJIA stock would be one that pays fairly predictable dividends, has been around since money was invented, and are not going anywhere unless the apocalypse really happens this year. In summary, DJIA is the boring reliable company index.
```

## Usage
* [Train](#train)
* [Evaluate](#evaluate)
* [Predict](#predict)
* [Generate data](#generate)

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

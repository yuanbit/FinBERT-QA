# FinBERT-QA: Financial Question Answering using BERT
FinBERT-QA is a Question Answering system for retrieving opinionated financial passages from task 2 of the [FiQA](https://sites.google.com/view/fiqa) dataset. 

The system uses techniques from both information retrieval and natural language processing by first retrieving the top-50 answer candidates of each query using the Lucene toolkit, [Anserini](https://github.com/castorini/anserini), then re-ranking the answer candidates using variants of pre-trained [BERT](https://arxiv.org/pdf/1810.04805.pdf) models. 

The [state-of-the-art](https://dl.acm.org/doi/10.1145/3184558.3191830) results were improved by an average of ~20% on three ranking evaluation metrics.

Built using Huggingface's [transformers](https://github.com/huggingface/transformers) library and the transfer and adapt [[TANDA](https://arxiv.org/pdf/1911.04118.pdf)] method, FinBERT-QA first transfers and fine-tunes a pre-trained BERT model to a general QA task, then adapts this model to the financial domain using the FiQA dataset. The transfer step uses the fine-tuned BERT model on the [MS MACRO Passage Retrieval](https://microsoft.github.io/msmarco/) dataset from [Nogueira et al.](https://arxiv.org/pdf/1901.04085.pdf), where it was converted from a TensorFlow to PyTorch model.

## Sections
* [Installation](#installation)
* [Quickstart](#quickstart)
* [Retriever](#retriever)
* [Dataset](#data)
* [Models](#models)
* [Basic Usage](#basic-usage)
  * [Train](#train)
  * [Evaluate](#evaluate)
  * [Predict](#predict)
  * [Generate data](#generate-data)

## Installation
If no GPU is available, an alternative and low-effort way to train and evaluate a model as well as predicting the results is through the following [online notebooks](https://github.com/yuanbit/FinBERT-QA-notebooks) using Colab.

### With Docker
This repo can be used as a container with [Docker](https://www.docker.com/). Run the commands as root if Docker is not configured.

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

## Retriever
The retriever uses the BM25 implementation from [Anserini](https://github.com/castorini/anserini). To replicate the creation of Lucene index for the FiQA dataset run the following inside the docker image:
```
cd retriever
git clone https://github.com/castorini/anserini.git
sh indexer.sh
```
## Dataset
The [raw dataset](https://sites.google.com/view/fiqa) has been cleaned and split into training, validation, and test sets in the form of lists where each sample is a list of```[question id, [label answer ids], [answer candidate ids]]```. There are  The datasets are stored in the pickle files in ```data/data_pickle```. The generation of the datasets can be replicated by running the ```src/generate_data.py``` script, more details please see usage.```data/data_pickle``` is a pickle file consisting of a python dictionary where the keys are the question ids and the values are lists of relevant answer ids.

Since creating inputs for to fine-tune a pre-trained BERT model can take some time, sample datasets are provided in ```data/sample/``` for testing.

Example QA:
```
Question: Why are big companies like Apple or Google not included in the Dow Jones Industrial Average (DJIA) index?

Answer: That is a pretty exclusive club and for the most part they are not interested in highly volatile companies like Apple and Google. Sure, IBM is part of the DJIA, but that is about as stalwart as you can get these days. The typical profile for a DJIA stock would be one that pays fairly predictable dividends, has been around since money was invented, and are not going anywhere unless the apocalypse really happens this year. In summary, DJIA is the boring reliable company index.
```

## Models
The download of the pre-trained and fine-tuned models are automated by the scripts. As an alternative they can also be downloaded manually. Make sure you are inside the ```FinBERT-QA``` directory.
### Pre-trained BERT models
For training usage.
* ```'bert-qa'```: pre-trained BERT model fine-tuned on the MS Macro passage dataset of [Nogueira et al.](https://arxiv.org/pdf/1901.04085.pdf)
* ```'finbert-domain'```: further pre-trained BERT model of [Araci](https://arxiv.org/pdf/1908.10063.pdf) on a large financial corpus
* ```'finbert-task'```: further pre-trained BERT model on the FiQA dataset
* ```'bert-base'```: ```'bert-base-uncased'``` model from [transformers](https://github.com/huggingface/transformers)
```python
from src.utils import *

get_model('bert-qa')
``` 
The model will be downloaded in ```model/bert-qa/```

### Trained baseline QA-LSTM and fine-tuned BERT models
For evaluation and prediction usage.
* ```'finbert-qa'```: ```'bert-qa'``` fine-tuned on FiQA
* ```'finbert-domain'```: ```'finbert-domain'``` fine-tuned on FiQA
* ```'finbert-task'```: ```'finberr-task'``` fine-tuned on FiQA
* ```'bert-pointwise'```: ```'bert-base-uncase'``` fine-tuned on FiQA using the cross-entropy loss
* ```'bert-pairwise'```: ```'bert-base-uncase'``` fine-tuned on FiQA using a pairwise loss
* ```'qa-lstm'```: QA-LSTM model
```python
from src.utils import *

get_trained_model('finbert-qa')
``` 
The model will be downloaded in ```model/trained/finbert-qa/```

## Basic Usage
* [Train](#train)
* [Evaluate](#evaluate)
* [Predict](#predict)
* [Generate data](#generate-data)

### Train
#### `src/train_models.py`: trains and fine-tunes model
#### Train and fine-tune FinBERT-QA with pointwise learning approach
This example code further fine-tunes [Nogueira et al.](https://arxiv.org/pdf/1901.04085.pdf)'s BERT model on the FiQA dataset using the pointwise learning approach (cross-entropy loss).
```
python3 src/train_models.py --model_type 'bert' 
                            --train_pickle data/data_pickle/train_set_50.pickle \
                            --valid_pickle data/data_pickle/valid_set_50.pickle \
                            --bert_model_name 'bert-qa' \
                            --learning_approach 'pointwise' \
                            --max_seq_len 512 \
                            --batch_size 16 \
                            --n_epochs 3 \
                            --lr 3e-6 \
                            --weight_decay 0.01 \
                            --num_warmup_steps 10000
```
Training with these hyperparameters produced the SOTA results:
```
MRR@10: 0.436
nDCG@10: 0.482
P@1: 0.366
```
#### Train a baseline QA-LSTM model
```
python3 src/train_models.py --model_type 'qa-lstm' \
                            --train_pickle data/data_pickle/train_set_50.pickle \
                            --valid_pickle data/data_pickle/valid_set_50.pickle \
                            --emb_dim 100 \
                            --hidden_size 256 \
                            --max_seq_len 128 \
                            --batch_size 64 \
                            --n_epochs 3 \
                            --lr 1e-3 \
                            --dropout 0.2
```
Detailed Usage
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
### Evaluate
#### `src/evaluate_models.py`: evaluates the models
#### Evaluate FinBERT-QA
```
python3 src/evaluate_models.py --test_pickle data/data_pickle/test_set_50.pickle \
                                --model_type 'bert' \
                                --max_seq_len 512 \
                                --bert_finetuned_model 'finbert-qa' \
                                --use_trained_model 
```
Detailed Usage
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
### Predict
#### Answer Re-ranking with FinBERT-QA
#### `src/predict.py`: given a query, retrieves the top-50 candidate answers and re-ranks them with the FinBERT-QA model
Retrieve the top-5 answers for a user given query
```
python3 src/predict.py --user_input --k 5
```
Detailed usage
```
python3 src/predict.py  [--user_input] [--query QUERY] [--k K]

Arguments:
  QUERY - Specify query if user_input is not used
  K - Top-k answers to output
```

### Generate data
#### `src/generate_data.py`: creates pickle files of the training, validation, and test set
```
python3 src/generate_data.py --query_path data/raw/FiQA_train_question_final.tsv \
                             --label_path data/raw/FiQA_train_question_doc_final.tsv
```
The data wil be stored in ```data/data_pickle```

Detailed usage:
```
python3 src/generate_data.py [--query_path QUERY_PATH] [--label_path LABEL_PATH] \
                             [--cands_size CANDS_SIZE] [--output_dir OUTPUT_DIR]

Arguments:
  QUERY_PATH - Path to the question id to text data in .tsv format. Each line should have at least two columns named (qid, question) separated by tab
  LABEL_PATH - Path to the question id and answer id data in .tsv format. Each line should have at two columns named (qid, docid) separated by tab
  CANDS_SIZE - Number of candidates to retrieve per question.
  OUTPUT_DIR - The output directory where the generated data will be stored.
                             
```

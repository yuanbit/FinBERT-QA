from tqdm import tqdm
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.nn.functional import softmax
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup, BertConfig

from helper.utils import *

# Dictionary mapping docid and qid to raw text
docid_to_text = load_pickle('../fiqa/data/id_to_text/docid_to_text.pickle')
qid_to_text = load_pickle('../fiqa/data/id_to_text/qid_to_text.pickle')

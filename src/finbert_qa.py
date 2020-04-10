from tqdm import tqdm
import numpy as np
import random
import torch
import json
import os
from torch.nn import CrossEntropyLoss
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.nn.functional import softmax
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup, BertConfig
from pyserini.search import pysearch

from helper.utils import *
from helper.download import *
from helper.evaluate import *

# Set Java path for PySerini
# os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"

# Set the random seed manually for reproducibility.
torch.backends.cudnn.deterministic = True
torch.manual_seed(1234)

# Lucene index
FIQA_INDEX = "../fiqa/retriever/lucene-index-fiqa"
# Dictionary mapping of docid and qid to raw text
docid_to_text = load_pickle('../fiqa/data/id_to_text/docid_to_text.pickle')
qid_to_text = load_pickle('../fiqa/data/id_to_text/qid_to_text.pickle')
# Labels
labels = load_pickle('../fiqa/data/data_pickle/labels.pickle')

DEFAULT_CONFIG = {'model_type': 'bert',
                  'use_default_config': True,
                  'device': 'gpu',
                  'max_seq_len': 512,
                  'batch_size': 8,
                  'n_epochs': 3,
                  'lr': 3e-6,
                  'margin': 0.5,
                  'weight_decay': 0.01,
                  'num_warmup_steps': 10000}

class BERT_QA():
    """Fine-tuned BERT model for non-factoid question answering.
    """
    def __init__(self, bert_model_name):
        """Initialize which pre-trained BERT model to use.
        """
        self.bert_model_name = bert_model_name

    def initialize_model(self):
        """Initialize which pre-trained BERT model to use.
        BertForSequenceClassification is a model from Huggingface's transformer
        library that contains the pretrained BERT model with a single linear
        classification layer.

        Returns:
            model: Torch model
        """
        if self.bert_model_name == "bert-base":
            model_path = "bert-base-uncased"
        elif self.bert_model_name == "finbert-domain":
            get_model("finbert-domain")
            model_path = '../fiqa/model/finbert-domain'
        elif self.bert_model_name == "finbert-task":
            get_model("finbert-task")
            model_path = '../fiqa/model/finbert-task'
        else:
            get_model("bert-qa")
            model_path = '../fiqa/model/bert-qa'

        print("\nLoading pre-trained BERT model...")
        model = BertForSequenceClassification.from_pretrained(model_path, \
                                                              cache_dir=None, \
                                                              num_labels=2)
        return model

class PointwiseBERT():
    def __init__(self, config, tokenizer, model, optimizer):
        self.config = config
        # Overwrite config to default
        if self.config['use_default_config'] == False:
            self.train_set = load_pickle(self.config['train_set'])
            # Load validation set
            self.valid_set = load_pickle(self.config['valid_set'])
        # Use GPU or CPU
        self.device = torch.device('cuda' if self.config['device'] == 'gpu' else 'cpu')
        # Maximum sequence length
        self.max_seq_len = config['max_seq_len']
        # Batch size
        self.batch_size = config['batch_size']
        # Number of epochs
        self.n_epochs = config['n_epochs']
        # Load the BERT tokenizer.
        self.tokenizer = tokenizer
        # Generate training and validation data
        print("\nGenerating training and validation data...\n")
        self.train_dataloader, self.validation_dataloader = self.get_dataloader()
        # Initialize model
        self.model = model
        self.optimizer = optimizer
        # Total number of training steps is number of batches * number of epochs.
        total_steps = len(self.train_dataloader) * self.n_epochs
        # Create a schedule with a learning rate that decreases linearly
        # after linearly increasing during a warmup period
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, \
                    num_warmup_steps = config['num_warmup_steps'], \
                    num_training_steps = total_steps)

    def get_input_data(self, dataset):
        """Creates input parameters for training and validation.

        Returns:
            input_ids: List of lists
                    Each element contains a list of padded/truncated numericalized
                    tokens of the sequences including [CLS] and [SEP] tokens
                    e.g. [[101, 2054, 2003, 102, 2449, 1029, 102], ...]
            token_type_ids: List of lists
                    Each element contains a list of segment token indices to
                    indicate first (question) and second (answer) parts of the inputs.
                    0 corresponds to a question token, 1 corresponds an answer token
                    e.g. [[0, 0, 0, 0, 1, 1, 1], ...]
            att_masks: List of lists
                    Each element contains a list of mask values to avoid
                    performing attention on padding token indices.
                    1 for tokens that are NOT MASKED, 0 for MASKED tokens.
                    e.g. [[1, 1, 1, 1, 1, 1, 1], ...]
            labels: List of 1's and 0's incidating relevacy of answer
        -----------------
        Arguements:
            dataset: List of lists in the form of [qid, [pos ans], [ans cands]]
        """
        input_ids = []
        token_type_ids = []
        att_masks = []
        labels = []

        for i, seq in enumerate(tqdm(dataset)):
            qid, ans_labels, cands = seq[0], seq[1], seq[2]
            # Map question id to text
            q_text = qid_to_text[qid]
            # For each answer in the candidates
            for docid in cands:
                # Map the docid to text
                ans_text = docid_to_text[docid]
                # Encode the sequence using BERT tokenizer
                encoded_seq = self.tokenizer.encode_plus(q_text, ans_text,
                                                    max_length=self.max_seq_len,
                                                    pad_to_max_length=True,
                                                    return_token_type_ids=True,
                                                    return_attention_mask = True)
                # Get parameters
                input_id = encoded_seq['input_ids']
                token_type_id = encoded_seq['token_type_ids']
                att_mask = encoded_seq['attention_mask']

                # If an answer is in the list of relevant answers assign
                # positive label
                if docid in ans_labels:
                    label = 1
                else:
                    label = 0

                # Each parameter list has the length of the max_seq_len
                assert len(input_id) == self.max_seq_len, "Input id dimension incorrect!"
                assert len(token_type_id) == self.max_seq_len, "Token type id dimension incorrect!"
                assert len(att_mask) == self.max_seq_len, "Attention mask dimension incorrect!"

                input_ids.append(input_id)
                token_type_ids.append(token_type_id)
                att_masks.append(att_mask)
                labels.append(label)

        return input_ids, token_type_ids, att_masks, labels

    def get_dataloader(self):
        """Creates train and validation DataLoaders with input_ids,
        token_type_ids, att_masks, and labels

        Returns:
            train_dataloader: DataLoader object
            validation_dataloader: DataLoader object
        """
        # Use default data
        if self.config['use_default_config'] == True:
            train_input, train_type_id, train_att_mask, \
            train_label, valid_input, valid_type_id, \
            valid_att_mask, valid_label = load_input_data("pointwise-bert")
        else:
            # Create training input parameters
            train_input, train_type_id, \
            train_att_mask, train_label = self.get_input_data(self.train_set)
            # Create validation input parameters
            valid_input, valid_type_id, \
            valid_att_mask, valid_label = self.get_input_data(self.valid_set)

        # Convert all train inputs and labels into torch tensors
        train_inputs = torch.tensor(train_input)
        train_type_ids = torch.tensor(train_type_id)
        train_masks = torch.tensor(train_att_mask)
        train_labels = torch.tensor(train_label)

        # Create the DataLoader for our training set.
        train_data = TensorDataset(train_inputs, train_type_ids, train_masks, \
                                   train_labels)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, \
                                      batch_size=self.batch_size)

        # Convert all validation inputs and labels into torch tensors
        validation_inputs = torch.tensor(valid_input)
        validation_type_ids = torch.tensor(valid_type_id)
        validation_masks = torch.tensor(valid_att_mask)
        validation_labels = torch.tensor(valid_label)

        # Create the DataLoader for our validation set.
        validation_data = TensorDataset(validation_inputs, validation_type_ids,\
                                        validation_masks, validation_labels)
        validation_sampler = SequentialSampler(validation_data)
        validation_dataloader = DataLoader(validation_data, \
                                           sampler=validation_sampler, \
                                           batch_size=self.batch_size)

        return train_dataloader, validation_dataloader

    def get_accuracy(self, preds, labels):
        """Compute the accuracy of binary predictions.

        Returns:
            accuracy: float
        -----------------
        Arguments:
            preds: Numpy list with two columns of probabilities for each label
            labels: List of labels
        """
        # Get the label (column) with the higher probability
        predictions = np.argmax(preds, axis=1).flatten()
        labels = labels.flatten()
        # Compute accuracy
        accuracy = np.sum(predictions == labels) / len(labels)

        return accuracy

    def train(self, model, train_dataloader, optimizer, scheduler):
        """Trains the model and returns the average loss and accuracy.

        Returns:
            avg_loss: Float
            avg_acc: Float
        ----------
        Arguements:
            model: Torch model
            train_dataloader: DataLoader object
            optimizer: Optimizer object
            scheduler: Scheduler object
        """
        # Cumulated Training loss and accuracy
        total_loss = 0
        train_accuracy = 0
        # Track the number of steps
        num_steps = 0
        # Set model in train mode
        model.train()
        # For each batch of training data
        for step, batch in enumerate(tqdm(train_dataloader)):
            # Get tensors and move to gpu
            # batch contains four PyTorch tensors:
            #   [0]: input ids
            #   [1]: token_type_ids
            #   [2]: attention masks
            #   [3]: labels
            b_input_ids = batch[0].to(self.device)
            b_token_type_ids = batch[1].to(self.device)
            b_input_mask = batch[2].to(self.device)
            b_labels = batch[3].to(self.device)

            # Zero the gradients
            model.zero_grad()
            # Forward pass: the model will return the loss and the logits
            outputs = model(b_input_ids,
                            token_type_ids = b_token_type_ids,
                            attention_mask = b_input_mask,
                            labels = b_labels)

            # Get loss and predictions
            loss = outputs[0]
            logits = outputs[1]
            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Calculate the accuracy for a batch
            tmp_accuracy = self.get_accuracy(logits, label_ids)

            # Accumulate the total accuracy.
            train_accuracy += tmp_accuracy

            # Track the number of batches
            num_steps += 1

            # Accumulate the training loss over all of the batches
            total_loss += loss.item()

            # Perform a backward pass to calculate the gradients
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient
            optimizer.step()

            # Update scheduler
            scheduler.step()

        # Calculate the average loss over the training data.
        avg_loss = total_loss / len(train_dataloader)
        avg_acc = train_accuracy/num_steps

        return avg_loss, avg_acc

    def validate(self, model, validation_dataloader):
        """Validates the model and returns the average loss and accuracy.

        Returns:
            avg_loss: Float
            avg_acc: Float
        ----------
        Arguements:
            model: Torch model
            validation_dataloader: DataLoader object
        """
        # Set model to evaluation mode
        model.eval()
        # Cumulated Training loss and accuracy
        total_loss = 0
        eval_accuracy = 0
        # Track the number of steps
        num_steps = 0

        # For each batch of the validation data
        for batch in tqdm(validation_dataloader):
            # Move tensors from batch to GPU
            batch = tuple(t.to(self.device) for t in batch)
            # Unpack the inputs from the dataloader
            b_input_ids, b_token_type_ids, b_input_masks, b_labels = batch
            # Don't to compute or store gradients
            with torch.no_grad():
                outputs = model(b_input_ids,
                                token_type_ids = b_token_type_ids,
                                attention_mask = b_input_masks,
                                labels= b_labels)
            # Get loss and logits
            loss = outputs[0]
            logits = outputs[1]
            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences.
            tmp_eval_accuracy = self.get_accuracy(logits, label_ids)

            # Accumulate the total accuracy.
            eval_accuracy += tmp_eval_accuracy

            # Track the number of steps
            num_steps += 1

            total_loss += loss.item()

        # Calculate loss and accuracy
        avg_loss = total_loss / len(validation_dataloader)
        avg_acc = eval_accuracy/num_steps

        return avg_loss, avg_acc

    def train_pointwise(self):
        """Train and validate the model and print the average loss and accuracy.
        """
        # Lowest validation lost
        best_valid_loss = float('inf')

        print("\nTraining model...\n")
        for epoch in range(self.n_epochs):
            # Evaluate training loss
            train_loss, train_acc = self.train(self.model, \
                                               self.train_dataloader, \
                                               self.optimizer, \
                                               self.scheduler)
            # Evaluate validation loss
            valid_loss, valid_acc = self.validate(self.model, \
                                                  self.validation_dataloader)
            # At each epoch, if the validation loss is the best
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(self.model.state_dict(), '../fiqa/model/' + \
                str(epoch+1)+ '_pointwise_' + self.config['bert_model_name'] + '.pt')

            print("\n\n Epoch {}:".format(epoch+1))
            print("\t Train Loss: {} | Train Accuracy: {}%".format(round(train_loss, 3), round(train_acc*100, 2)))
            print("\t Validation Loss: {} | Validation Accuracy: {}%\n".format(round(valid_loss, 3), round(valid_acc*100, 2)))

class PairwiseBERT():
    def __init__(self, config, tokenizer, model, optimizer):
        self.config = config
        # Overwrite config to default
        if self.config['use_default_config'] == False:
            self.train_set = load_pickle(self.config['train_set'])
            # Load validation set
            self.valid_set = load_pickle(self.config['valid_set'])
        # Use GPU or CPU
        self.device = torch.device('cuda' if self.config['device'] == 'gpu' else 'cpu')
        # Maximum sequence length
        self.max_seq_len = config['max_seq_len']
        # Batch size
        self.batch_size = config['batch_size']
        # Number of epochs
        self.n_epochs = config['n_epochs']
        # Margin for loss function
        self.margin = config['margin']
        # Load the BERT tokenizer.
        self.tokenizer = tokenizer
        # Generate training and validation data
        print("\nGenerating training and validation data...\n")
        self.train_dataloader, self.validation_dataloader = self.get_dataloader()
        # Initialize model
        self.model = model
        self.optimizer = optimizer
        # Total number of training steps is number of batches * number of epochs.
        total_steps = len(self.train_dataloader) * self.n_epochs
        # Create a schedule with a learning rate that decreases linearly
        # after linearly increasing during a warmup period
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, \
                    num_warmup_steps = config['num_warmup_steps'], \
                    num_training_steps = total_steps)

    def get_input_data(self, dataset):
        """Creates input parameters for training and validation.

        Returns:
            pos_input_ids: List of lists
                    Each element contains a list of padded/truncated numericalized
                    tokens of the positive QA sequences
            pos_type_ids: List of lists
                    Each element contains a list of segment token indices to
                    indicate first (question) and second (answer) parts of the inputs.
            pos_masks: List of lists
                    Each element contains a list of mask values to avoid
                    performing attention on padding token indices.
            pos_labels: List of 1's
            neg_input_ids: List of lists
                    Each element contains a list of padded/truncated numericalized
                    tokens of the negative QA sequences
            neg_type_ids: List of lists
                    Each element contains a list of segment token indices to
                    indicate first (question) and second (answer) parts of the inputs.
            neg_masks: List of lists
                    Each element contains a list of mask values to avoid
                    performing attention on padding token indices.
            neg_labels: List of 0's
        -----------------
        Arguements:
            dataset: List of lists in the form of [qid, [pos ans], [ans cands]]
        """
        pos_input_ids = []
        neg_input_ids = []

        pos_type_ids = []
        neg_type_ids = []

        pos_masks = []
        neg_masks = []

        pos_labels = []
        neg_labels = []

        for i, seq in enumerate(tqdm(dataset)):
            qid, ans_labels, cands = seq[0], seq[1], seq[2]
            # Get a list of negative candidate answers
            filtered_cands = list(set(cands)-set(ans_labels))
            # Select a positive answer from the labels
            pos_docid = random.choice(ans_labels)
            # Map question id to text
            q_text = qid_to_text[qid]
            # For each negative answer
            for neg_docid in filtered_cands:
                # Map the docid to text
                pos_ans_text = docid_to_text[pos_docid]
                neg_ans_text = docid_to_text[neg_docid]
                # Encode positive QA pair
                pos_encoded_seq = self.tokenizer.encode_plus(q_text, pos_ans_text,
                                                    max_length=self.max_seq_len,
                                                    pad_to_max_length=True,
                                                    return_token_type_ids=True,
                                                    return_attention_mask = True)
                # Encode negative QA pair
                neg_encoded_seq = self.tokenizer.encode_plus(q_text, neg_ans_text,
                                                    max_length=self.max_seq_len,
                                                    pad_to_max_length=True,
                                                    return_token_type_ids=True,
                                                    return_attention_mask = True)
                # Get parameters
                pos_input_id = pos_encoded_seq['input_ids']
                pos_type_id = pos_encoded_seq['token_type_ids']
                pos_mask = pos_encoded_seq['attention_mask']

                neg_input_id = neg_encoded_seq['input_ids']
                neg_type_id = neg_encoded_seq['token_type_ids']
                neg_mask = neg_encoded_seq['attention_mask']

                pos_input_ids.append(pos_input_id)
                pos_type_ids.append(pos_type_id)
                pos_masks.append(pos_mask)
                pos_labels.append(1)

                neg_input_ids.append(neg_input_id)
                neg_type_ids.append(neg_type_id)
                neg_masks.append(neg_mask)
                neg_labels.append(0)

        return pos_input_ids, pos_type_ids, pos_masks, pos_labels, \
               neg_input_ids, neg_type_ids, neg_masks, neg_labels

    def get_dataloader(self):
        """Creates train and validation DataLoaders with input_ids,
        token_type_ids, att_masks, and labels

        Returns:
            train_dataloader: DataLoader object
            validation_dataloader: DataLoader object
        """
        # Use default data
        if self.config['use_default_config'] == True:
            train_pos_input, train_pos_type_id, train_pos_mask, \
            train_pos_label, train_neg_input, train_neg_type_id, \
            train_neg_mask, train_neg_label, valid_pos_input, \
            valid_pos_type_id, valid_pos_mask, valid_pos_label, \
            valid_neg_input, valid_neg_type_id, valid_neg_mask, \
            valid_neg_label = load_input_data("pairwise-bert")
        else:
            # Create training input parameters
            train_pos_input, train_pos_type_id, train_pos_mask, \
            train_pos_label, train_neg_input, train_neg_type_id, \
            train_neg_mask, train_neg_label = self.get_input_data(self.train_set)
            # Create validation input parameters
            valid_pos_input, valid_pos_type_id, valid_pos_mask, \
            valid_pos_label, valid_neg_input, valid_neg_type_id, \
            valid_neg_mask, valid_neg_label = self.get_input_data(self.valid_set)

        # Convert all train inputs and labels into torch tensors
        train_pos_inputs = torch.tensor(train_pos_input)
        train_pos_type_ids = torch.tensor(train_pos_type_id)
        train_pos_masks = torch.tensor(train_pos_mask)
        train_pos_labels = torch.tensor(train_pos_label)
        train_neg_inputs = torch.tensor(train_neg_input)
        train_neg_type_ids = torch.tensor(train_neg_type_id)
        train_neg_masks = torch.tensor(train_neg_mask)
        train_neg_labels = torch.tensor(train_neg_label)

        # Create the DataLoader for our training set.
        train_data = TensorDataset(train_pos_inputs, train_pos_type_ids, \
                                   train_pos_masks, train_pos_labels, \
                                   train_neg_inputs, train_neg_type_ids, \
                                   train_neg_masks, train_neg_labels)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, \
                                      batch_size=self.batch_size)

        # Convert all validation inputs and labels into torch tensors
        valid_pos_inputs = torch.tensor(valid_pos_input)
        valid_pos_type_ids = torch.tensor(valid_pos_type_id)
        valid_pos_masks = torch.tensor(valid_pos_mask)
        valid_pos_labels = torch.tensor(valid_pos_label)
        valid_neg_inputs = torch.tensor(valid_neg_input)
        valid_neg_type_ids = torch.tensor(valid_neg_type_id)
        valid_neg_masks = torch.tensor(valid_neg_mask)
        valid_neg_labels = torch.tensor(valid_neg_label)

        # Create the DataLoader for our validation set.
        validation_data = TensorDataset(valid_pos_inputs, valid_pos_type_ids, \
                                        valid_pos_masks, valid_pos_labels, \
                                        valid_neg_inputs, valid_neg_type_ids, \
                                        valid_neg_masks, valid_neg_labels)
        validation_sampler = SequentialSampler(validation_data)
        validation_dataloader = DataLoader(validation_data, \
                                           sampler=validation_sampler, \
                                           batch_size=self.batch_size)

        return train_dataloader, validation_dataloader

    def get_accuracy(self, preds, labels):
        """Compute the accuracy of binary predictions.

        Returns:
            accuracy: float
        -----------------
        Arguments:
            preds: Numpy list with two columns of probabilities for each label
            labels: List of labels
        """
        # Get the label (column) with the higher probability
        predictions = np.argmax(preds, axis=1).flatten()
        labels = labels.flatten()
        # Compute accuracy
        accuracy = np.sum(predictions == labels) / len(labels)

        return accuracy

    def pairwise_loss(self, pos_scores, neg_scores):
        """Pairwise loss introduced in https://arxiv.org/pdf/1905.07588.pdf

        Returns:
            loss: Torch tensor of floats
        -----------------
        Arguements:
            pos_scores: Torch tensor of positive QA pair probabilies
            neg_scores: Torch tensor of negative QA pair probabilies
        """
        cross_entropy_loss = -torch.log(pos_scores) - torch.log(1 - neg_scores)

        hinge_loss = torch.max(torch.tensor(0, dtype=torch.float).to(self.device), \
                               self.margin - pos_scores + neg_scores)

        loss = (0.5 * cross_entropy_loss + 0.5 * hinge_loss)

        return loss

    def train(self, model, train_dataloader, optimizer, scheduler):
        """Trains the model and returns the average loss and accuracy.

        Returns:
            avg_loss: Float
            avg_acc: Float
        ----------
        Arguements:
            model: Torch model
            train_dataloader: DataLoader object
            optimizer: Optimizer object
            scheduler: Scheduler object
        """
        # Reset the loss and accuracy for each epoch
        total_loss = 0
        num_steps = 0
        train_accuracy = 0
        # Set model in training mode
        model.train()
        # For each batch of training data
        for step, batch in enumerate(tqdm(train_dataloader)):
            # Get input tensors and move to gpu:
            pos_input = batch[0].to(self.device)
            pos_type_id = batch[1].to(self.device)
            pos_mask = batch[2].to(self.device)
            pos_labels = batch[3].to(self.device)
            neg_input = batch[4].to(self.device)
            neg_type_id = batch[5].to(self.device)
            neg_mask = batch[6].to(self.device)
            neg_labels = batch[7].to(self.device)

            # Zero gradients
            model.zero_grad()
            # Compute predictinos for postive and negative QA pairs
            pos_outputs = model(pos_input,
                                token_type_ids=pos_type_id,
                                attention_mask=pos_mask,
                                labels=pos_labels)
            neg_outputs = model(neg_input,
                                token_type_ids=neg_type_id,
                                attention_mask=neg_mask,
                                labels=neg_labels)

            # Get the logits from the model for positive and negative QA pairs
            pos_logits = pos_outputs[1]
            neg_logits = neg_outputs[1]

            # Get the column of the relevant scores and apply activation function
            pos_scores = softmax(pos_logits, dim=1)[:,1]
            neg_scores = softmax(neg_logits, dim=1)[:,1]

            # Compute pairwise loss and get the mean of each batch
            loss = self.pairwise_loss(pos_scores, neg_scores).mean()

            # Move logits and labels to CPU
            p_logits = pos_logits.detach().cpu().numpy()
            p_labels = pos_labels.to('cpu').numpy()
            n_logits = neg_logits.detach().cpu().numpy()
            n_labels = neg_labels.to('cpu').numpy()

            # Calculate the accuracy for each batch
            tmp_pos_accuracy = self.get_accuracy(p_logits, p_labels)
            tmp_neg_accuracy = self.get_accuracy(n_logits, n_labels)

            # Accumulate the total accuracy.
            train_accuracy += tmp_pos_accuracy
            train_accuracy += tmp_neg_accuracy

            # Track the number of batches (2 for pos and neg accuracies)
            num_steps += 2

            # Accumulate the training loss over all of the batches
            total_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            optimizer.step()

            # Update scheduler
            scheduler.step()

        # Calculate the average loss over the training data.
        avg_loss = total_loss / len(train_dataloader)
        # Compute accuracy for each epoch
        avg_acc = train_accuracy/num_steps

        return avg_loss, avg_acc

    def validate(self, model, validation_dataloader):
        """Validates the model and returns the average loss and accuracy.

        Returns:
            avg_loss: Float
            avg_acc: Float
        ----------
        Arguements:
            model: Torch model
            validation_dataloader: DataLoader object
        """
        # Set model in evaluation mode
        model.eval()
        # Tracking variables
        total_loss = 0
        num_steps = 0
        eval_accuracy = 0

        # Evaluate data for one epoch
        for batch in tqdm(validation_dataloader):
            # Add batch to GPU
            batch = tuple(t.to(self.device) for t in batch)
            # Unpack the inputs from our dataloader
            pos_input, pos_type_id, pos_mask, pos_labels, \
            neg_input, neg_type_id, neg_mask, neg_labels = batch

            # Don't compute and store gradients
            with torch.no_grad():
                # Compute predictinos for postive and negative QA pairs
                pos_outputs = model(pos_input,
                                    token_type_ids=pos_type_id,
                                    attention_mask=pos_mask,
                                    labels=pos_labels)
                neg_outputs = model(neg_input,
                                    token_type_ids=neg_type_id,
                                    attention_mask=neg_mask,
                                    labels=neg_labels)

                # Get logits
                pos_logits = pos_outputs[1]
                neg_logits = neg_outputs[1]

                # Apply activation function
                pos_scores = softmax(pos_logits, dim=1)[:,1]
                neg_scores = softmax(neg_logits, dim=1)[:,1]

            loss = self.pairwise_loss(pos_scores, neg_scores).mean()

            # Move logits and labels to CPU
            p_logits = pos_logits.detach().cpu().numpy()
            p_labels = pos_labels.to('cpu').numpy()
            n_logits = neg_logits.detach().cpu().numpy()
            n_labels = neg_labels.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences.
            tmp_pos_accuracy = self.get_accuracy(p_logits, p_labels)
            tmp_neg_accuracy = self.get_accuracy(n_logits, n_labels)

            # Accumulate the total accuracy.
            eval_accuracy += tmp_pos_accuracy
            eval_accuracy += tmp_neg_accuracy

            # Track the number of batches
            num_steps += 2
            # Cumulate loss
            total_loss += loss.item()
        # Compute average loss and accuracy
        avg_loss = total_loss / len(validation_dataloader)
        avg_acc = eval_accuracy/num_steps

        return avg_loss, avg_acc

    def train_pairwise(self):
        """Train and validate the model and print the average loss and accuracy.
        """
        # Lowest validation lost
        best_valid_loss = float('inf')

        print("\nTraining model...\n")
        for epoch in range(self.n_epochs):
            # Evaluate training loss
            train_loss, train_acc = self.train(self.model, \
                                               self.train_dataloader, \
                                               self.optimizer, \
                                               self.scheduler)
            # Evaluate validation loss
            valid_loss, valid_acc = self.validate(self.model, \
                                                  self.validation_dataloader)
            # At each epoch, if the validation loss is the best
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(self.model.state_dict(), '../fiqa/model/' + \
                str(epoch+1)+ '_pairwise_' + self.config['bert_model_name'] + '.pt')

            print("\n\n Epoch {}:".format(epoch+1))
            print("\t Train Loss: {} | Train Accuracy: {}%".format(round(train_loss, 3), round(train_acc*100, 2)))
            print("\t Validation Loss: {} | Validation Accuracy: {}%\n".format(round(valid_loss, 3), round(valid_acc*100, 2)))

class train_bert_model():
    """
    Train the fine-tuned BERT model.
    """
    def __init__(self, config):
        # Pre-trained BERT model name
        bert_model_name = config['bert_model_name']
        learning_approach = config['learning_approach']
        # Overwrite config to default
        if config['use_default_config'] == True:
            config = DEFAULT_CONFIG
        else:
            config = config
        # Use GPU or CPU
        device = torch.device('cuda' if config['device'] == 'gpu' else 'cpu')
        # Load the BERT tokenizer.
        print('\nLoading BERT tokenizer...')
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        # Initialize model
        model = BERT_QA(bert_model_name).initialize_model().to(device)
        optimizer = AdamW(model.parameters(), lr = config['lr'], \
                          weight_decay=config['weight_decay'])

        # Train and validate model based on learning approach
        if learning_approach == 'pointwise':
            trainer = PointwiseBERT(config, tokenizer, model, optimizer)
            trainer.train_pointwise()
        else:
            trainer = PairwiseBERT(config, tokenizer, model, optimizer)
            trainer.train_pairwise()

class evaluate_bert_model():
    def __init__(self, config):
        self.config = config
        # Pre-trained BERT model name
        self.bert_model_name = config['bert_model_name']
        # Load test set
        self.test_set = load_pickle(self.config['test_set'])
        # Use GPU or CPU
        self.device = torch.device('cuda' if config['device'] == 'gpu' else 'cpu')
        # Maximum sequence length
        self.max_seq_len = self.config['max_seq_len']
        # Fine-tuned BERT model name
        self.bert_finetuned_model = self.config['bert_finetuned_model']

        print('\nLoading BERT tokenizer...')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        # Initialize model
        self.model = BERT_QA(self.bert_model_name).initialize_model().to(self.device)

        # Evaluate model
        self.evaluate_model()

    def get_rank(self, model):
        """Re-ranks the candidates answers for each question.

        Returns:
            qid_pred_rank: Dictionary
                key - qid
                value - list of re-ranked candidates
        -------------------
        Arguments:
            model - PyTorch model
        """
        # Initiate empty dictionary
        qid_pred_rank = {}
        # Set model to evaluation mode
        model.eval()
        # For each element in the test set
        for i, seq in enumerate(tqdm(self.test_set)):
            # question id, list of rel answers, list of candidates
            qid, label, cands = seq[0], seq[1], seq[2]
            # Map question id to text
            q_text = qid_to_text[qid]
            # Convert list to numpy array
            cands_id = np.array(cands)
            # Empty list for the probability scores of relevancy
            scores = []

            # For each answer in the candidates
            for docid in cands:
                # Map the docid to text
                ans_text = docid_to_text[docid]
                # Create inputs for the model
                encoded_seq = self.tokenizer.encode_plus(q_text, ans_text,
                                                max_length=self.max_seq_len,
                                                pad_to_max_length=True,
                                                return_token_type_ids=True,
                                                return_attention_mask = True)

                # Numericalized, padded, clipped seq with special tokens
                input_ids = torch.tensor([encoded_seq['input_ids']]).to(self.device)
                # Specify question seq and answer seq
                token_type_ids = torch.tensor([encoded_seq['token_type_ids']]).to(self.device)
                # Sepecify which position is part of the seq which is padded
                att_mask = torch.tensor([encoded_seq['attention_mask']]).to(self.device)

                # Don't calculate gradients
                with torch.no_grad():
                # Forward pass, calculate logit predictions for each QA pair
                    outputs = model(input_ids,
                                    token_type_ids=token_type_ids,
                                    attention_mask=att_mask)

                # Get the predictions
                logits = outputs[0]
                # Apply activation function
                pred = softmax(logits, dim=1)
                # Move logits and labels to CPU
                pred = pred.detach().cpu().numpy()
                # Append relevant scores to list (where label = 1)
                scores.append(pred[:,1][0])
            # Get the indices of the sorted similarity scores
            sorted_index = np.argsort(scores)[::-1]
            # Get the list of docid from the sorted indices
            ranked_ans = cands_id[sorted_index]
            # Dict - key: qid, value: ranked list of docids
            qid_pred_rank[qid] = ranked_ans

        return qid_pred_rank

    def evaluate_model(self):
        """Prints the nDCG@10, MRR@10, Precision@1
        """
        k = 10
        # Number of questions
        num_q = len(self.test_set)

        # If not use pre-computed rank
        if self.config['use_rank_pickle'] == False:
            # If use trained model
            if self.config['use_trained_model'] == True:
                # Download model
                model_name = get_trained_model(self.bert_finetuned_model)
                model_path = "../fiqa/model/trained/" + \
                             self.bert_finetuned_model + "/" + model_name
            else:
                model_path = self.config['model_path']
            # Load model
            self.model.load_state_dict(torch.load(model_path), strict=False)
            print("\nEvaluating...\n")
            # Get rank
            qid_pred_rank = self.get_rank(self.model)
        else:
            print("\nEvaluating...\n")
            # Get pre-computed rank
            rank_path = "../fiqa/data/rank/" + self.bert_finetuned_model + "_rank.pickle"
            qid_pred_rank = load_pickle(rank_path)

        # Evaluate
        MRR, average_ndcg, precision, rank_pos = evaluate(qid_pred_rank, labels, k)

        print("Average nDCG@{0} for {1} queries: {2:.3f}".format(k, num_q, average_ndcg))
        print("MRR@{0} for {1} queries: {2:.3f}".format(k, num_q, MRR))
        print("Average Precision@1 for {0} queries: {1:.3f}".format(num_q, precision))

class FinBERT_QA():
    """Financial answer retriever based on fine-tuned BERT model.
    """
    def __init__(self, config):
        self.config = config
        self.searcher = pysearch.SimpleSearcher(FIQA_INDEX)
        self.max_seq_len = 512
        # Use GPU or CPU
        self.device = torch.device('cuda')
        print('\nLoading BERT tokenizer...')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        # Initialize model
        self.model = BERT_QA('bert-qa').initialize_model().to(self.device)

        self.user_input = self.config['user_input']
        self.k = self.config['k']

        self.search()

    def predict(self, model, q_text, cands):
        """Re-ranks the candidates answers for each question.

        Returns:
            ranked_ans: list of re-ranked candidate docids
            sorted_scores: list of relevancy scores of the answers
        -------------------
        Arguments:
            model - PyTorch model
            q_text - str - query
            cands -List of retrieved candidate docids
        """
        self.model.eval()
        # Convert list to numpy array
        cands_id = np.array(cands)
        # Empty list for the probability scores of relevancy
        scores = []
        # For each answer in the candidates
        for docid in cands:
            # Map the docid to text
            ans_text = docid_to_text[docid]
            # Create inputs for the model
            encoded_seq = self.tokenizer.encode_plus(q_text, ans_text,
                                                max_length=self.max_seq_len,
                                                pad_to_max_length=True,
                                                return_token_type_ids=True,
                                                return_attention_mask = True)

            # Numericalized, padded, clipped seq with special tokens
            input_ids = torch.tensor([encoded_seq['input_ids']]).to(self.device)
            # Specify question seq and answer seq
            token_type_ids = torch.tensor([encoded_seq['token_type_ids']]).to(self.device)
            # Sepecify which position is part of the seq which is padded
            att_mask = torch.tensor([encoded_seq['attention_mask']]).to(self.device)
            # Don't calculate gradients
            with torch.no_grad():
                # Forward pass, calculate logit predictions for each QA pair
                outputs = model(input_ids, token_type_ids=token_type_ids, attention_mask=att_mask)
            # Get the predictions
            logits = outputs[0]
            # Apply activation function
            pred = softmax(logits, dim=1)
            # Move logits and labels to CPU
            pred = pred.detach().cpu().numpy()
            # Append relevant scores to list (where label = 1)
            scores.append(pred[:,1][0])
            # Get the indices of the sorted similarity scores
            sorted_index = np.argsort(scores)[::-1]
            # Get the list of docid from the sorted indices
            ranked_ans = list(cands_id[sorted_index])
            sorted_scores = list(np.around(sorted(scores, reverse=True),decimals=3))

        return ranked_ans, sorted_scores

    def search(self):
        """Search engine based on FinBERT_QA.
        Retrieves and re-ranks the answer candidates given a query.
        Renders the top-k answers for a query.
        """
        if self.user_input == True:
            # Ask the user for a keyword query.
            self.query = input("\nPlease enter your question: ")
            print("\n")
        else:
            self.query = self.config['query']

        hits = self.searcher.search(self.query, k=50)
        self.cands = []
        # Print the first 10 hits:
        for i in range(0, len(hits)):
            self.cands.append(int(hits[i].docid))

        print("\nRanking...\n")
        # Download model
        model_name = get_trained_model("finbert-qa")
        model_path = "../fiqa/model/trained/finbert-qa/" + model_name
        # Load model
        self.model.load_state_dict(torch.load(model_path), strict=False)

        self.rank, self.scores = self.predict(self.model, self.query, self.cands)

        print("Question: \n\t{}\n".format(self.query))
        print("Top-{} Answers: \n".format(self.k))
        for i in range(0, self.k):
            print("{}.\t{}\n".format(i+1, docid_to_text[self.rank[i]]))

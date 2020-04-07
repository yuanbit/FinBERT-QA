from tqdm import tqdm
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.nn.functional import softmax
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup, BertConfig

from helper.utils import *
from helper.download import *

# Dictionary mapping of docid and qid to raw text
docid_to_text = load_pickle('../fiqa/data/id_to_text/docid_to_text.pickle')
qid_to_text = load_pickle('../fiqa/data/id_to_text/qid_to_text.pickle')

DEFAULT_CONFIG = {'model_type': 'bert',
                  'use_default_config': True,
                  'device': 'gpu',
                  'max_seq_len': 512,
                  'batch_size': 16,
                  'n_epochs': 3,
                  'lr': 3e-6,
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
            model_path = "bert-base-uncase"
        elif self.bert_model_name == "finbert-domain":
            model_path = '../fiqa/model/finbert-domain'
        elif self.bert_model_name == "finbert-task":
            model_path = '../fiqa/model/finbert-task'
        else:
            model_path = '../fiqa/model/bert-qa'

        model = BertForSequenceClassification.from_pretrained(model_path, \
                                                              cache_dir=None, \
                                                              num_labels=2)
        return model

class PointwiseBERT():
    def __init__(self, config, tokenizer, model, optimizer):
        # Overwrite config to default
        if config['use_default_config'] == False:
            self.train_set = load_pickle(self.config['train_set'])
            # Load validation set
            self.valid_set = load_pickle(self.config['valid_set'])
        # Use GPU or CPU
        self.device = config['device']
        # Maximum sequence length
        self.max_seq_len = config['max_seq_len']
        # Batch size
        self.batch_size = config['batch_size']
        # Number of epochs
        self.n_epochs = config['n_epochs']
        # Load the BERT tokenizer.
        self.tokenizer = tokenizer
        # Generate training and validation data
        print("\nGenerating training and validation data...")
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
            print("\t Train Loss: {} | Train Accuracy: \
                  {}%".format(round(train_loss, 3), round(train_acc*100, 2)))
            print("\t Validation Loss: {} | Validation Accuracy: \
                  {}%\n".format(round(valid_loss, 3), round(valid_acc*100, 2)))


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
        print('Loading BERT tokenizer...')
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        # Initialize model
        model = BERT_QA(bert_model_name).initialize_model.to(device)
        optimizer = AdamW(model.parameters(), lr = config['lr'], \
                          weight_decay=config['weight_decay'])

        # Train and validate model based on learning approach
        if learning_approach == 'pointwise':
            trainer = PointwiseBERT(config, tokenizer, model, optimizer)
            trainer.train_pointwise()
        else:
            pass

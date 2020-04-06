from tqdm import tqdm
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.nn.functional import softmax
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup, BertConfig

from helper.utils import *
from download_model import *

# Dictionary mapping docid and qid to raw text
docid_to_text = load_pickle('../fiqa/data/id_to_text/docid_to_text.pickle')
qid_to_text = load_pickle('../fiqa/data/id_to_text/qid_to_text.pickle')

class BERT_QA():
    def __init__(self, config):
        self.config = config
        self.bert_model_name = self.config['bert_model_name']

    def initialize_model(self):

        # Load BertForSequenceClassification:
        # the pretrained BERT model with a single linear classification layer
        if self.bert_model_name == "bert-base":
            model_path = "bert-base-uncase"
        elif self.bert_model_name == "finbert-domain":
            model_path = '../fiqa/model/finbert-domain'
        elif self.bert_model_name == "finbert-task":
            model_path = '../fiqa/model/finbert-task'
        else:
            model_path = '../fiqa/model/bert-qa'

        model = BertForSequenceClassification.from_pretrained(model_path, cache_dir=None, num_labels=2)

        return model

class train_pointwise_bert_model():
    def __init__(self, config):
        self.config = config
        # Use GPU or CPU
        self.device = torch.device('cuda' if config['device'] == 'gpu' else 'cpu')
        # Maximum sequence length
        self.max_seq_len = config['max_seq_len']
        # Batch size
        self.batch_size = config['batch_size']
        # Number of epochs
        self.n_epochs = config['n_epochs']
        # Load the BERT tokenizer.
        print('Loading BERT tokenizer...')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        # Load training set
        self.train_set = load_pickle(config['train_set'])
        # Load validation set
        self.valid_set = load_pickle(config['valid_set'])
        # Initialize model
        model = BERT_QA(config).initialize_model.to(self.device)
        optimizer = AdamW(model.parameters(), lr = self.config['lr'], \
                          weight_decay=self.config['weight_decay'])
        # Create the learning rate scheduler.
        scheduler = get_linear_schedule_with_warmup(optimizer, \
                    num_warmup_steps = self.config['num_warmup_steps'], \
                    num_training_steps = self.total_steps)

        # Lowest validation lost
        best_valid_loss = float('inf')
        print("\nGenerating training and validation data...\n")
        train_dataloader, validation_dataloader = self.get_dataloader()

        for epoch in range(self.n_epochs):
            # Evaluate training loss
            train_loss, train_acc = self.train(model, train_dataloader, optimizer, scheduler)
            # Evaluate validation loss
            valid_loss, valid_acc = self.validate(model, validation_dataloader)
            # At each epoch, if the validation loss is the best
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), '../fiqa/model/' + \
                str(epoch+1)+ '_pointwise_' + self.config['bert_model_name'] + '.pt')

            print("\n\n Epoch {}:".format(epoch+1))
            print("\t Train Loss: {} | Train Accuracy: {}%".format(round(train_loss, 3), round(train_acc*100, 2)))
            print("\t Validation Loss: {} | Validation Accuracy: {}%\n".format(round(valid_loss, 3), round(valid_acc*100, 2)))

def get_input_data(self, dataset):
    input_ids = []
    token_type_ids = []
    att_masks = []
    labels = []

    for i, seq in enumerate(tqdm(dataset)):
        qid, ans_labels, cands = seq[0], seq[1], seq[2]

        # Map question id to text
        q_text = qid_to_text[qid]

        for docid in cands:

            # Map the docid to text
            ans_text = docid_to_text[docid]

            encoded_seq = tokenizer.encode_plus(q_text, ans_text,
                                                max_length=self.max_seq_len,
                                                pad_to_max_length=True,
                                                return_token_type_ids=True,
                                                return_attention_mask = True)

            input_id = encoded_seq['input_ids']
            token_type_id = encoded_seq['token_type_ids']
            att_mask = encoded_seq['attention_mask']

            if docid in ans_labels:
                label = 1
            else:
                label = 0

            assert len(input_id) == self.max_seq_len, "Input id dimension incorrect!"
            assert len(token_type_id) == self.max_seq_len, "Token type id dimension incorrect!"
            assert len(att_mask) == self.max_seq_len, "Attention mask dimension incorrect!"

            input_ids.append(input_id)
            token_type_ids.append(token_type_id)
            att_masks.append(att_mask)
            labels.append(label)

    return input_ids, token_type_ids, att_masks, labels

    def get_dataloader(self):
        train_input, train_type_id, \
        train_att_mask, train_label = self.get_input_data(self.train_set)
        valid_input, valid_type_id, \
        valid_att_mask, valid_label = self.get_input_data(self.valid_set)

        # Convert all train inputs and labels into torch tensors
        train_inputs = torch.tensor(train_input)
        train_type_ids = torch.tensor(train_type_id)
        train_masks = torch.tensor(train_att_mask)
        train_labels = torch.tensor(train_label)

        # Create the DataLoader for our training set.
        train_data = TensorDataset(train_inputs, train_type_ids, train_masks, train_labels)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

        # Total number of training steps is number of batches * number of epochs.
        self.total_steps = len(train_dataloader) * self.n_epochs

        # Convert all validation inputs and labels into torch tensors
        validation_inputs = torch.tensor(valid_input)
        validation_type_ids = torch.tensor(valid_type_id)
        validation_masks = torch.tensor(valid_att_mask)
        validation_labels = torch.tensor(valid_label)

        # Create the DataLoader for our validation set.
        validation_data = TensorDataset(validation_inputs, validation_type_ids, validation_masks, validation_labels)
        validation_sampler = SequentialSampler(validation_data)
        validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

        return train_dataloader, validation_dataloader

    # Function to calculate the accuracy of our predictions vs labels
    def flat_accuracy(self, preds, labels):
        # Get the column with the higher probability
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    def train(self, model, train_dataloader, optimizer, scheduler):

        # Reset the total loss each epoch
        total_loss = 0
        train_accuracy = 0
        # Track the number of batches
        num_steps = 0

        # Set model in train mode
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(tqdm(train_dataloader)):

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

            # Forward pass
            # The model will return the loss and the logits
            outputs = model(b_input_ids,
                        token_type_ids = b_token_type_ids,
                        attention_mask = b_input_mask,
                        labels = b_labels)

            loss = outputs[0]
            logits = outputs[1]

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Calculate the accuracy for this batch
            tmp_accuracy = self.flat_accuracy(logits, label_ids)

            # Accumulate the total accuracy.
            train_accuracy += tmp_accuracy

            # Track the number of batches
            num_steps += 1

            # Accumulate the training loss over all of the batches
            total_loss += loss.item()

            # Perform a backward pass to calculate the gradients
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            optimizer.step()

            # Update scheduler
            scheduler.step()

        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)

        acc = train_accuracy/num_steps

        return avg_train_loss, acc

    def validate(self, model, validation_dataloader):

        # Evaluation mode
        model.eval()

        total_loss = 0
        eval_accuracy = 0
        nb_eval_steps = 0

        # For each batch of the validation data
        for batch in tqdm(validation_dataloader):

            # Add batch to GPU
            batch = tuple(t.to(self.device) for t in batch)

            # Unpack the inputs from the dataloader
            b_input_ids, b_token_type_ids, b_input_masks, b_labels = batch

            # Don't to compute or store gradients
            with torch.no_grad():
                outputs = model(b_input_ids,
                                token_type_ids = b_token_type_ids,
                                attention_mask = b_input_masks,
                                labels= b_labels)

            loss = outputs[0]
            logits = outputs[1]

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences.
            tmp_eval_accuracy = self.flat_accuracy(logits, label_ids)

            # Accumulate the total accuracy.
            eval_accuracy += tmp_eval_accuracy

            # Track the number of batches
            nb_eval_steps += 1

            total_loss += loss.item()

        acc = eval_accuracy/nb_eval_steps
        avg_loss = total_loss / len(validation_dataloader)

        return avg_loss, acc

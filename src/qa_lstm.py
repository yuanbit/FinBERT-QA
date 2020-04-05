import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchtext import data
import torchtext
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm

from utils import *

vocab = load_pickle("../fiqa/data/qa_lstm_tokenizer/word2index.pickle")
qid_to_tokenized_text = load_pickle('../fiqa/data/qa_lstm_tokenizer/qid_to_tokenized_text.pickle')
docid_to_tokenized_text = load_pickle('../fiqa/data/qa_lstm_tokenizer/docid_to_tokenized_text.pickle')

train_set = load_pickle('../fiqa/data/processed_data/train_set_50.pickle')
valid_set = load_pickle('../fiqa/data/processed_data/valid_set_50.pickle')

class QA_LSTM():
    def __init__(self, config):
        super(QA_LSTM, self).__init__()
        self.emb_dim = config['emb_dim']
        self.hidden_size = config['hidden_size']
        self.dropout = config['dropout']
        self.vocab_size = len(vocab)

        # Shape - (max_seq_len, emb_dim)
        self.embedding = self.create_emb_layer()

        self.shared_lstm = nn.LSTM(self.emb_dim, \
                                   self.hidden_size, \
                                   num_layers=1, \
                                   batch_first=True, \
                                   bidirectional=True)
        self.cos = nn.CosineSimilarity(dim=1)
        self.dropout = nn.Dropout(self.dropout)

    def create_emb_layer(self):
        print("Downloading pre-trained GloVe embeddings...\n")
        emb = torchtext.vocab.GloVe("6B", dim=self.emb_dim)
        # dictionary mapping of word idx to glove vectors
        emb_weights = np.zeros((self.vocab_size, self.emb_dim))
        words_found = 0

        for token, idx in self.vocab.items():
            # emb.stoi is a dict of token to idx mapping
            if token in emb.stoi:
                emb_weights[idx] = emb[token]
                words_found += 1

        print(words_found, "words are found in GloVe\n")
        # Convert numpy matrix to tensor
        emb_weights = torch.from_numpy(emb_weights).float()

        vocab_size, emb_dim = emb_weights.shape
        emb_layer = nn.Embedding(vocab_size, emb_dim)
        emb_layer.load_state_dict({'weight': emb_weights})

        return emb_layer

    def forward(self, q, a):
        # embedding
        q = self.embedding(q) # (bs, L, E)
        a = self.embedding(a) # (bs, L, E)

        # LSTM
        q, (hidden, cell) = self.shared_lstm(q) # (bs, L, 2H)
        a, (hidden, cell) = self.shared_lstm(a) # (bs, L, 2H)

        # Output shape (batch size, seq_len, num_direction * hidden_size)
        # There are n of word level biLSTM representations for the seq where n is the number of seq len
        # Use max pooling to generate the best representation
        q = torch.max(q, 1)[0]
        a = torch.max(a, 1)[0] # (bs, 2H)

        q = self.dropout(q)
        a = self.dropout(a)

        return self.cos(q, a) # (bs,)

class train_qa_lstm_model():
    def __init__(self, config):
        self.config = config
        self.device = config['device']
        self.max_seq_len = config['max_seq_len']
        self.batch_size = config['batch_size']
        self.n_epochs = config['n_epochs']
        self.model = QA_LSTM(self.config)
        self.model = self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        # Lowest validation lost
        best_valid_loss = float('inf')

        for epoch in range(self.n_epochs):
            # Evaluate training loss
            print("Training model...\n")
            train_loss = QA_LSTM.train(self.model, self.optimizer)
            # Evaluate validation loss
            print("Validating...\n")
            valid_loss = QA_LSTM.validate(self.model)

            # At each epoch, if the validation loss is the best
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                # Save the parameters of the model
                torch.save(model.state_dict(), '../model/'+str(epoch+1)+'_qa_lstm.pt')

            print("\n\n Epoch {}:".format(epoch+1))
            print("\t Train Loss: {}".format(round(train_loss, 3)))
            print("\t Validation Loss: {}\n".format(round(valid_loss, 3)))

    def hinge_loss(self, pos_sim, neg_sim):
        margin = 0.2

        loss = torch.max(torch.tensor(0, dtype=torch.float).to(self.device), margin - pos_sim + neg_sim)

        return loss

    def get_lstm_input_data(self, dataset):
        q_input_ids = []
        pos_input_ids = []
        neg_input_ids = []

        for i, seq in enumerate(tqdm(dataset)):
            qid, ans_labels, cands = seq[0], seq[1], seq[2]

            filtered_cands = list(set(cands)-set(ans_labels))

            pos_docid = random.choice(ans_labels)

            # Map question id to text
            q_text = qid_to_tokenized_text[qid]
            q_input_id = self.vectorize(q_text)

            for neg_docid in filtered_cands:

                # Map the docid to text
                pos_ans_text = docid_to_tokenized_text[pos_docid]
                neg_ans_text = docid_to_tokenized_text[neg_docid]

                pos_input_id = self.vectorize(pos_ans_text)
                neg_input_id = self.vectorize(neg_ans_text)

                q_input_ids.append(q_input_id)
                pos_input_ids.append(pos_input_id)
                neg_input_ids.append(neg_input_id)

        return q_input_ids, pos_input_ids, neg_input_ids

    def get_dataloader(self):
        print("Generating training data...\n")
        train_q_input, train_pos_input, train_neg_input = self.get_lstm_input_data(train_set)

        train_q_inputs = torch.tensor(train_q_input)
        train_pos_inputs = torch.tensor(train_pos_input)
        train_neg_inputs = torch.tensor(train_neg_input)

        # Create the DataLoader for our training set.
        train_data = TensorDataset(train_q_inputs, train_pos_inputs, train_neg_inputs)
        train_sampler = RandomSampler(train_data)
        self.train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.batch_size)

        print("Generating validation data...\n")
        valid_q_input, valid_pos_input, valid_neg_input = self.get_lstm_input_data(valid_set)

        valid_q_inputs = torch.tensor(valid_q_input)
        valid_pos_inputs = torch.tensor(valid_pos_input)
        valid_neg_inputs = torch.tensor(valid_neg_input)

        # Create the DataLoader for our validation set.
        validation_data = TensorDataset(valid_q_inputs, valid_pos_inputs, valid_neg_inputs)
        validation_sampler = SequentialSampler(validation_data)
        self.validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=self.batch_size)

        return self.train_dataloader, self.validation_dataloader

    def train(self, model, optimizer):
        # Cumulated Training loss
        train_loss = 0.0
        # Set model to training mode
        model.train()
        # For each batch of training data...
        for step, batch in enumerate(tqdm(self.train_dataloader)):
            # batch contains eight PyTorch tensors:
            question = batch[0].to(device)
            pos_ans = batch[1].to(device)
            neg_ans = batch[2].to(device)
            # 1. Zero gradients
            model.zero_grad()
            # 2. Compute predictions
            pos_sim = model(question, pos_ans)
            neg_sim = model(question, neg_ans)
            # 3. Compute loss
            loss = self.hinge_loss(pos_sim, neg_sim).mean()
            # 4. Use loss to compute gradients
            loss.backward()
            # 5. Use optimizer to take gradient step
            optimizer.step()
            train_loss += loss.item()
        avg_loss = train_loss/len(self.train_dataloader)
        return avg_loss

    def validate(model):
        # Cumulated Training loss
        valid_loss = 0.0
        # Set model to evaluation mode
        model.eval()
        # Evaluate data for one epoch
        for batch in tqdm(self.validation_dataloader):
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            question, pos_ans, neg_ans = batch
            # Don't calculate the gradients
            with torch.no_grad():
                pos_sim = model(question, pos_ans)
                neg_sim = model(question, neg_ans)
                loss = self.hinge_loss(pos_sim, neg_sim).mean()
                valid_loss += loss.item()
        avg_loss = valid_loss/len(self.validation_dataloader)

        return avg_loss

import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torchtext
from tqdm import tqdm

from helper.utils import *

# Dictonary with token to id mapping
vocab = load_pickle("../fiqa/data/qa_lstm_tokenizer/word2index.pickle")
# Dictonary with qid to tokenized text mapping
qid_to_tokenized_text = load_pickle('../fiqa/data/qa_lstm_tokenizer/qid_to_tokenized_text.pickle')
# Dictionary with docid to tokenized text mapping
docid_to_tokenized_text = load_pickle('../fiqa/data/qa_lstm_tokenizer/docid_to_tokenized_text.pickle')

class QA_LSTM(nn.Module):
    """
    QA-LSTM model
    """
    def __init__(self, config):
        super(QA_LSTM, self).__init__()
        # Embedding dimension
        self.emb_dim = config['emb_dim']
        # Hidden size
        self.hidden_size = config['hidden_size']
        # Dropout rate
        self.dropout = config['dropout']
        # Vocabulary size
        self.vocab_size = len(vocab)
        # Create embedding layer
        self.embedding = self.create_emb_layer()
        # The question and answer representations share the same biLSTM network
        self.lstm = nn.LSTM(self.emb_dim, \
                            self.hidden_size, \
                            num_layers=1, \
                            batch_first=True, \
                            bidirectional=True)
        # Cosine similiarty metric
        self.sim = nn.CosineSimilarity(dim=1)
        # Apply dropout
        self.dropout = nn.Dropout(self.dropout)

    def create_emb_layer(self):
        """Creates embedding layerself using pre-trained
        GloVe embeddings (6B tokens)

        Returns:
            emb_layer: torch embedding layer
        """
        print("\nDownloading pre-trained GloVe embeddings...\n")
        # Use GloVe embeddings from torchtext
        emb = torchtext.vocab.GloVe("6B", dim=self.emb_dim)
        # Dictionary mapping of word idx to GloVe vectors
        emb_weights = np.zeros((self.vocab_size, self.emb_dim))
        # Count
        words_found = 0

        for token, idx in vocab.items():
            # emb.stoi is a dict of token to idx mapping
            # If token from the vocabulary exist in GloVe
            if token in emb.stoi:
                # Add the embedding to the index
                emb_weights[idx] = emb[token]
                words_found += 1

        print("\n")
        print(words_found, "words are found in GloVe\n")
        # Convert matrix to tensor
        emb_weights = torch.from_numpy(emb_weights).float()

        vocab_size, emb_dim = emb_weights.shape
        # Create embedding layer
        emb_layer = nn.Embedding(vocab_size, emb_dim)
        # Load the embeddings
        emb_layer.load_state_dict({'weight': emb_weights})

        return emb_layer

    def forward(self, question, answer):
        """Forward pass to generate biLSTM representations for the question and
        answer independently, and then utilize cosine similarity to measure
        their distance.

        Returns:
            similarity: torch tensor with cosine similarity score.
        ----------
        Arguements:
            question: tensor of vectorized question
            answer: tensor of vectorized answer
        """
        # Embedding layers - (batch_size, max_seq_len, emb_dim)
        question_embedding = self.embedding(q)
        answer_embedding = self.embedding(a)

        # biLSTM - (batch_size, max_seq_len, 2*hidden_size)
        question_lstm, (hidden, cell) = self.lstm(question_embedding)
        answer_lstm, (hidden, cell) = self.lstm(answer_embedding)

        # Max-pooling - (batch_size, 2*hidden_size)
        # There are n word level biLSTM representations where n is the max_seq_len
        # Use max pooling to generate the best representation
        question_max_pool = torch.max(question_lstm, 1)[0]
        answer_maxpool = torch.max(answer_lstm, 1)[0]

        # Apply dropout
        question_output = self.dropout(question_maxpool)
        answer_output = self.dropout(answer_maxpool)

        # Similarity -(batch_size,)
        similarity = self.sim(question_output, answer_output)

        return similarity

class train_qa_lstm_model():
    """Train the QA-LSTM model
    """
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
        # Margin for hinge loss
        self.margin = config['margin']
        # Load training set
        self.train_set = load_pickle(config['train_set'])
        # Load validation set
        self.valid_set = load_pickle(config['valid_set'])
        # Initialize model
        model = QA_LSTM(self.config).to(self.device)
        # Use Adam optimizer
        optimizer = optim.Adam(model.parameters(), lr=config['lr'])
        # Lowest validation lost
        best_valid_loss = float('inf')

        print("\nGenerating training and validation data...\n")
        train_dataloader, validation_dataloader = self.get_dataloader()

        print("\nTraining model...\n")
        for epoch in range(self.n_epochs):
            # Evaluate training loss
            train_loss = self.train(model, train_dataloader, optimizer)
            # Evaluate validation loss
            valid_loss = self.validate(model, validation_dataloader)

            # At each epoch, if the validation loss is the best
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                # Save the parameters of the model
                torch.save(model.state_dict(), '../fiqa/model/'+str(epoch+1)+'_qa_lstm.pt')

            print("\n\n Epoch {}:".format(epoch+1))
            print("\t Train Loss: {}".format(round(train_loss, 3)))
            print("\t Validation Loss: {}\n".format(round(valid_loss, 3)))

    def hinge_loss(self, pos_sim, neg_sim):
        """
        Returns:
            loss: tensor with hinge loss value
        ----------
        Arguements:
            pos_sim: tensor with similarity of a question and a positive answer
            neg_sim: tensor with similarity of a question and a negative answer
        """
        loss = torch.max(torch.tensor(0, dtype=torch.float).to(self.device), \
                         self.margin - pos_sim + neg_sim)
        return loss

    def pad_seq(self, seq_idx):
        """Creates padded sequence.

        Returns:
            seq: list of padded vectorized sequence
        ----------
        Arguements:
            seq_idx: tensor with similarity of a question and a positive answer
        """
        # Pad each sequence to be the same length to process in batches
        # pad_token = 0
        if len(seq_idx) >= self.max_seq_len:
            seq = seq_idx[:self.max_seq_len]
        else:
            seq += [0]*(self.max_seq_len - len(seq_idx))
        return seq

    def vectorize(self, seq):
        """Creates vectorized sequence.

        Returns:
            vectorized_seq: list of padded vectorized sequence
        ----------
        Arguements:
            seq: list of tokens in a sequence
        """
        # Map tokens in seq to idx
        seq_idx = [vocab[token] for token in seq]
        # Pad seq idx
        vectorized_seq = self.pad_seq(seq_idx)

        return vectorized_seq

    def get_lstm_input_data(self, dataset):
        """Creates input data for model.

        Returns:
            q_input_ids: list of lists of vectorized question sequence
            pos_input_ids: list of lists of vectorized positve ans sequence
            neg_input_ids: list of lists of vectorized negative ans sequence
        ----------
        Arguements:
            dataset: list of lists in the form of [qid, [pos ans], [ans cands]]
        """
        q_input_ids = []
        pos_input_ids = []
        neg_input_ids = []

        for i, seq in enumerate(tqdm(dataset)):
            qid, ans_labels, cands = seq[0], seq[1], seq[2]

            # Remove the positive answers for the candidates
            filtered_cands = list(set(cands)-set(ans_labels))
            # Select a positive answer from the list of positive answers
            pos_docid = random.choice(ans_labels)
            # Map question id to text
            q_text = qid_to_tokenized_text[qid]
            # Pad and vectorize text
            q_input_id = self.vectorize(q_text)

            # For all the negative answers
            for neg_docid in filtered_cands:
                # Map the docid to text
                pos_ans_text = docid_to_tokenized_text[pos_docid]
                neg_ans_text = docid_to_tokenized_text[neg_docid]
                # Pad and vectorize sequences
                pos_input_id = self.vectorize(pos_ans_text)
                neg_input_id = self.vectorize(neg_ans_text)

                q_input_ids.append(q_input_id)
                pos_input_ids.append(pos_input_id)
                neg_input_ids.append(neg_input_id)

        return q_input_ids, pos_input_ids, neg_input_ids

    def get_dataloader(self):
        """Creates train and validation DataLoaders with question, positive
        answer, and negative answer vectorized inputs.

        Returns:
            train_dataloader: DataLoader object
            validation_dataloader: DataLoader object
        """
        train_q_input, train_pos_input, train_neg_input = self.get_lstm_input_data(self.train_set)

        train_q_inputs = torch.tensor(train_q_input)
        train_pos_inputs = torch.tensor(train_pos_input)
        train_neg_inputs = torch.tensor(train_neg_input)

        # Create the DataLoader for our training set.
        train_data = TensorDataset(train_q_inputs, train_pos_inputs, train_neg_inputs)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.batch_size)

        valid_q_input, valid_pos_input, valid_neg_input = self.get_lstm_input_data(self.valid_set)

        valid_q_inputs = torch.tensor(valid_q_input)
        valid_pos_inputs = torch.tensor(valid_pos_input)
        valid_neg_inputs = torch.tensor(valid_neg_input)

        # Create the DataLoader for our validation set.
        validation_data = TensorDataset(valid_q_inputs, valid_pos_inputs, valid_neg_inputs)
        validation_sampler = SequentialSampler(validation_data)
        validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=self.batch_size)

        return train_dataloader, validation_dataloader

    def train(self, model, train_dataloader, optimizer):
        """Trains the model and returns the average loss

        Returns:
            avg_loss: float
        ----------
        Arguements:
            model: torch model
            train_dataloader: DataLoader object
            optimizer: Optimizer object
        """
        # Cumulated Training loss
        train_loss = 0.0
        # Set model to training mode
        model.train()
        # For each batch of training data
        for step, batch in enumerate(tqdm(train_dataloader)):
            # batch contains 3 PyTorch tensors
            # Move tensors to gpu
            question = batch[0].to(self.device)
            pos_ans = batch[1].to(self.device)
            neg_ans = batch[2].to(self.device)

            # 1. Zero gradients
            model.zero_grad()
            # 2. Compute similarity scores of pos and neg QA pairs
            pos_sim = model(question, pos_ans)
            neg_sim = model(question, neg_ans)
            # 3. Compute loss
            loss = self.hinge_loss(pos_sim, neg_sim).mean()
            # 4. Use loss to compute gradients
            loss.backward()
            # 5. Use optimizer to take gradient step
            optimizer.step()
            # Cumulate loss
            train_loss += loss.item()
        # Compute average loss
        avg_loss = train_loss/len(train_dataloader)

        return avg_loss

    def validate(self, model, validation_dataloader):
        """Validates the model and returns the average loss

        Returns:
            avg_loss: float
        ----------
        Arguements:
            model: torch model
            validation_dataloader: DataLoader object
        """
        # Cumulated validation loss
        valid_loss = 0.0
        # Set model to evaluation mode
        model.eval()
        # Evaluate data
        for batch in tqdm(validation_dataloader):
            # Add batch to GPU
            batch = tuple(t.to(self.device) for t in batch)
            # Unpack the inputs from Dataloader
            question, pos_ans, neg_ans = batch
            # Don't calculate the gradients
            with torch.no_grad():
                # Compute similarity score of pos and neg QA pairs
                pos_sim = model(question, pos_ans)
                neg_sim = model(question, neg_ans)
                # Compute loss
                loss = self.hinge_loss(pos_sim, neg_sim).mean()
                # Coumulate loss
                valid_loss += loss.item()
        # Compute average loss
        avg_loss = valid_loss/len(validation_dataloader)

        return avg_loss

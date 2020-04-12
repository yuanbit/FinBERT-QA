import os
from pathlib import Path
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torchtext
from tqdm import tqdm

from utils import *
from evaluate import *

# Set the random seed manually for reproducibility.
torch.backends.cudnn.deterministic = True
torch.manual_seed(1234)

path = str(Path.cwd())

# Dictonary with token to id mapping
vocab = load_pickle(path + "/data/qa_lstm_tokenizer/word2index.pickle")
# Dictonary with qid to tokenized text mapping
qid_to_tokenized_text = load_pickle(path + '/data/qa_lstm_tokenizer/qid_to_tokenized_text.pickle')
# Dictionary with docid to tokenized text mapping
docid_to_tokenized_text = load_pickle(path + '/data/qa_lstm_tokenizer/docid_to_tokenized_text.pickle')
# Labels
labels = load_pickle(path + '/data/data_pickle/labels.pickle')

class LSTM_MODEL(nn.Module):
    """
    QA-LSTM model
    """
    def __init__(self, config):
        """Initialize model parameters.

        Arguments:
            config: Dictionary
        """
        super(LSTM_MODEL, self).__init__()
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
            emb_layer: Torch embedding layer
        """
        print("\nInitializing model...")
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

        #print("\n")
        #print(words_found, "words are found in GloVe")
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
            similarity: Torch tensor with cosine similarity score.
        ----------
        Arguements:
            question: Torch tensor of vectorized question
            answer: Torch tensor of vectorized answer
        """
        # Embedding layers - (batch_size, max_seq_len, emb_dim)
        question_embedding = self.embedding(question)
        answer_embedding = self.embedding(answer)

        # biLSTM - (batch_size, max_seq_len, 2*hidden_size)
        question_lstm, (hidden, cell) = self.lstm(question_embedding)
        answer_lstm, (hidden, cell) = self.lstm(answer_embedding)

        # Max-pooling - (batch_size, 2*hidden_size)
        # There are n word level biLSTM representations where n is the max_seq_len
        # Use max pooling to generate the best representation
        question_maxpool = torch.max(question_lstm, 1)[0]
        answer_maxpool = torch.max(answer_lstm, 1)[0]

        # Apply dropout
        question_output = self.dropout(question_maxpool)
        answer_output = self.dropout(answer_maxpool)

        # Similarity -(batch_size,)
        similarity = self.sim(question_output, answer_output)

        return similarity

class QA_LSTM():
    """QA-LSTM model
    """
    def __init__(self, config):
        """Initialize parameters.

        Arguments:
            config: Dictionary
        """
        self.config = config
        # Load data from custom path
        self.train_set = load_pickle(self.config['train_set'])
        self.valid_set = load_pickle(self.config['valid_set'])
        self.test_set = load_pickle(self.config['test_set'])
        # Use GPU or CPU
        self.device = torch.device('cuda' if config['device'] == 'gpu' else 'cpu')
        # Maximum sequence length
        self.max_seq_len = self.config['max_seq_len']
        # Initialize model
        self.model = LSTM_MODEL(self.config).to(self.device)

    def hinge_loss(self, pos_sim, neg_sim):
        """
        Returns:
            loss: Tensor with hinge loss value
        ----------
        Arguements:
            pos_sim: Tensor with similarity of a question and a positive answer
            neg_sim: Tensor with similarity of a question and a negative answer
        """
        # Margin for hinge loss
        margin = self.config['margin']

        loss = torch.max(torch.tensor(0, dtype=torch.float).to(self.device), \
                         margin - pos_sim + neg_sim)
        return loss

    def pad_seq(self, seq_idx):
        """Creates padded or truncated sequence.

        Returns:
            seq: list of padded vectorized sequence
        ----------
        Arguements:
            seq_idx: tensor with similarity of a question and a positive answer
        """
        # Pad each sequence to be the same length to process in batches
        # pad_token = 0
        if len(seq_idx) >= self.max_seq_len:
            seq_idx = seq_idx[:self.max_seq_len]
        else:
            seq_idx += [0]*(self.max_seq_len - len(seq_idx))
        seq = seq_idx

        return seq

    def vectorize(self, seq):
        """Creates vectorized sequence.

        Returns:
            vectorized_seq: List of padded vectorized sequence
        ----------
        Arguements:
            seq: List of tokens in a sequence
        """
        # Map tokens in seq to idx
        seq_idx = [vocab[token] for token in seq]
        # Pad seq idx
        vectorized_seq = self.pad_seq(seq_idx)

        return vectorized_seq

    def get_input_data(self, dataset):
        """Creates input data for model.

        Returns:
            q_input_ids: List of lists of vectorized question sequence
            pos_input_ids: List of lists of vectorized positve ans sequence
            neg_input_ids: List of lists of vectorized negative ans sequence
        ----------
        Arguements:
            dataset: List of lists in the form of [qid, [pos ans], [ans cands]]
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

    def get_dataloader(self, dataset, type):
        """Creates train and validation DataLoaders with question, positive
        answer, and negative answer vectorized inputs.

        Returns:
            dataloader: DataLoader object
        ----------
        Arguements:
            dataset: List of lists in the form of [qid, [pos ans], [ans cands]]
            type: str - 'train' or 'validation'
        """
        question_input, pos_ans_input, neg_ans_input = self.get_input_data(dataset)

        question_inputs = torch.tensor(question_input)
        pos_ans_inputs = torch.tensor(pos_ans_input)
        neg_ans_inputs = torch.tensor(neg_ans_input)

        # Create the DataLoader
        data = TensorDataset(question_inputs, pos_ans_inputs, neg_ans_inputs)
        if type == "train":
            sampler = RandomSampler(data)
        else:
            sampler = SequentialSampler(data)
        dataloader = DataLoader(data, sampler=sampler, batch_size=self.batch_size)

        return dataloader

    def train(self, model, train_dataloader, optimizer):
        """Trains the model and returns the average loss

        Returns:
            avg_loss: Float
        ----------
        Arguements:
            model: Torch model
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
            avg_loss: Float
        ----------
        Arguements:
            model: Torch model
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

    def run_train(self):
        """Train and validate the model and print the average loss and accuracy.
        """
        # Batch size
        self.batch_size = self.config['batch_size']
        # Number of epochs
        self.n_epochs = self.config['n_epochs']

        print("\nGenerating training and validation data...\n")
        train_dataloader = self.get_dataloader(self.train_set, "train")
        validation_dataloader = self.get_dataloader(self.valid_set, "validation")

        # Use Adam optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=self.config['lr'])

        # Lowest validation lost
        best_valid_loss = float('inf')

        print("\nTraining model...\n")
        for epoch in range(self.n_epochs):
            # Evaluate training loss
            train_loss = self.train(self.model, train_dataloader, optimizer)
            # Evaluate validation loss
            valid_loss = self.validate(self.model, validation_dataloader)
            # At each epoch, if the validation loss is the best
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                # Save the parameters of the model
                torch.save(self.model.state_dict(), path + '/model/'+str(epoch+1)+'_qa_lstm.pt')

            print("\n\n Epoch {}:".format(epoch+1))
            print("\t Train Loss: {0:.3f}".format(train_loss))
            print("\t Validation Loss: {0:.3f}\n".format(valid_loss))

    def get_rank(self, model):
        """Re-ranks the answer candidates per question using trained model.

        Returns:
            qid_pred_rank: Dictionary
                    key - qid
                    value - List of re-ranked candidate answers
        -------------------
        Arguments:
            model - Trained PyTorch model
        """
        # Dictionary - key: qid, value: ranked list of docids
        qid_pred_rank = {}
        # Set model to evaluation mode
        model.eval()
        # For each sample in the test set
        for i, seq in enumerate(tqdm(self.test_set)):
            # Extract input data
            ques, pos_ans, cands = seq[0], seq[1], seq[2]
            # Tokenize and vectorize question
            q_text = qid_to_tokenized_text[ques]
            q_vec = torch.tensor([self.vectorize(q_text)]).to(self.device)
            # Tokenize candidate answers
            cands_text = [docid_to_tokenized_text[c] for c in cands]
            cands_id = np.array(cands)
            # List to store similarity score of QA pair
            scores = []
            # For each candidate answer
            for cand in cands_text:
                # Vectorize the answers
                a_vec = torch.tensor([self.vectorize(cand)]).to(self.device)
                # Compute similarity score of QA pair and add to scores
                scores.append(model(q_vec, a_vec).item())

            # Get the indices of the sorted (descending) similarity scores
            sorted_index = np.argsort(scores)[::-1]
            # Get the cand ans docid from the sorted indices
            ranked_ans = cands_id[sorted_index]
            # Set the qid keys to the list of re-ranked docids
            qid_pred_rank[ques] = ranked_ans

        return qid_pred_rank

    def evaluate_model(self):
        """Prints the nDCG@10, MRR@10, Precision@1
        """
        k = 10
        # Number of questions
        num_q = len(self.test_set)

        # If use trained model
        if self.config['use_trained_model'] == True:
            # Download model
            model_name = get_trained_model("qa-lstm")
            model_path = path + "/model/trained/qa-lstm/" + model_name
        else:
            model_path = self.config['model_path']
        # Load model
        self.model.load_state_dict(torch.load(model_path), strict=False)
        print("\nEvaluating...\n")
        # Get rank
        qid_pred_rank = self.get_rank(self.model)

        # Evaluate
        MRR, average_ndcg, precision, rank_pos = evaluate(qid_pred_rank, labels, k)

        print("\nAverage nDCG@{0} for {1} queries: {2:.3f}".format(k, num_q, average_ndcg))
        print("MRR@{0} for {1} queries: {2:.3f}".format(k, num_q, MRR))
        print("Average Precision@1 for {0} queries: {1:.3f}".format(num_q, precision))

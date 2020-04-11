import regex as re
import nltk
nltk.download('punkt')
from nltk.tokenize import wordpunct_tokenize
from collections import Counter

from .utils import *

def pre_process(text):
    """Returns a lower-cased string with punctuations and special characters removed.

    Returns:
        processed_text: str
    ----------
    Arguments:
        text: str of answer or question text
    """
    text = str(text)
    # Substitute punctuations and special characters for a space
    x = re.sub('[…“”%!&"@#()\-\*\+,/:;<=>?@[\]\^_`{\}~]', ' ', text)
    # Remove periods
    y = re.sub('[\.\']', "", x)
    # Apply lower-case
    processed_text = y.lower()

    return processed_text

def process_questions(queries):
    """Returns a dataframe with tokenized questions.

    Returns:
        queries: Dataframe
    ----------
    Arguments:
        queries: Dataframe containing qids and questions
    """
    # Pre-process questions
    queries['q_processed'] = queries['question'].apply(pre_process)
    # Apply tokenizer
    queries['tokenized_q'] = queries.apply(lambda row: wordpunct_tokenize(row['q_processed']), axis=1)
    # Count the length of each question
    queries['q_len'] = queries.apply(lambda row: len(row['tokenized_q']), axis=1)

    return queries

def process_answers(collection):
    """Returns a dataframe with tokenized answers.

    Returns:
        collection: Dataframe
    ----------
    Arguments:
        collection: Dataframe containing docids and answers
    """
    # Pre-process answers
    collection['doc_processed'] = collection['doc'].apply(pre_process)
    # Apply tokenizer
    collection['tokenized_ans'] = collection.apply(lambda row: wordpunct_tokenize(row['doc_processed']), axis=1)
    # Count the length of each answer
    collection['ans_len'] = collection.apply(lambda row: len(row['tokenized_ans']), axis=1)

    return collection

def create_vocab(processed_answers, processed_questions):
    """Returns two dictionaries containing the token to id mapping and token
    to count mapping.

    Returns:
        word2index: dictionary
            key - token
            value - token id
        word2count: dictionary
            key - token
            value - frequency count
    ----------
    Arguments:
        processed_answers: Dataframe containing docid and tokenized answers
        processed_questions: Dataframe containing qid and tokenized questions
    """
    # Initialize dictionary with special token
    word2index = {"PAD": 0}
    word2count = {}

    idx = 1

    # Add tokens from answers to vocab
    for index, row in processed_answers.iterrows():
        for word in row['tokenized_ans']:
            # If the token is not in the dictionary
            if word not in word2index:
                # Map each unique token to an index
                word2index[word] = idx
                idx += 1
                # Set count to 1
                word2count[word] = 1
            else:
                # Increment count
                word2count[word] += 1

    # Add tokens from questions to vocab
    idx = len(word2index)

    for index, row in processed_questions.iterrows():
        for word in row['tokenized_q']:
            if word not in word2index:
                word2index[word] = idx
                idx += 1
                word2count[word] = 1
            else:
                word2count[word] += 1

    return word2index, word2count

def id_to_text(collection, queries):
    """Returns two dictionaries mapping id to text.

    Returns:
        qid_to_text: dictionary
            key - qid
            value - question text
        docid_to_text: dictionary
            key - docid
            value - answer text
    ----------
    Arguments:
        collection: dataframe
            Dataframe containing docids and answers
        queries: dataframe
            Dataframe containing qids and questions
    """
    qid_to_text = {}
    docid_to_text = {}

    for index, row in queries.iterrows():
        qid_to_text[row['qid']] = row['question']

    for index, row in collection.iterrows():
        docid_to_text[row['docid']] = row['doc']

    return qid_to_text, docid_to_text

def id_to_tokenized_text(processed_answers, processed_questions):
    """Returns two dictionaries mapping id to tokenized text.

    Retuns:
        qid_to_text: dictionary
            key - qid
            value - list of question tokens
        docid_to_text: dictionary
            key - docid
            value - list of answer tokens
    ----------
    Arguments:
        collection: dataframe
            Dataframe containing docids and tokenized answers
        queries: dataframe
            Dataframe containing qids and tokenized questions
    """
    qid_to_tokenized_text = {}
    docid_to_tokenized_text = {}

    for index, row in processed_questions.iterrows():
        qid_to_tokenized_text[row['qid']] = row['tokenized_q']

    for index, row in processed_answers.iterrows():
        docid_to_tokenized_text[row['docid']] = row['tokenized_ans']

    return qid_to_tokenized_text, docid_to_tokenized_text

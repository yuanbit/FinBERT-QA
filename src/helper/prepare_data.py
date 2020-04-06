import pandas as pd
from pathlib import Path
import csv
from itertools import islice
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import json

from utils import *

def get_empty_docs(collection):
    """Returns a list of docids with empty answers and a corresponding list
    of ids for the documents dataframe.
    ----------
    Arguments:
        collection: Dataframe with a column of docid and a column of answer text.
    """
    empty_docs = []
    empty_id = []

    for index, row in collection.iterrows():
        # If the answer text is empty
        if pd.isna(row['doc']):
            empty_docs.append(row['docid'])
            empty_id.append(index)

    return empty_docs, empty_id

def label_to_dict(df):
    """
    Returns a dictionary converted from the labels dataframe which contains the
    question id and the relevant docids.

    Returns:
        qid_rel: Dictonary
            key - question id
            value - list of relevant docids
    ----------
    Arguments:
        df: Dataframe to convert to dictionary
    """
    qid_rel = {}
    for index, row in df.iterrows():
        if row['qid'] not in qid_rel:
            # Create a list for each unique question
            qid_rel[row['qid']] = []
        qid_rel[row['qid']].append(row['docid'])

    return qid_rel

def load_answers_to_df(path):
    """
    Returns a dataframe of docids and answer text.

    Returns:
        collection: Dataframe
    ----------
    Arguments:
        path: str
    """
    # Doc ID to Doc text
    collection = pd.read_csv(path, sep="\t")
    collection = collection[['docid', 'doc']]
    collection = collection.sort_values(by=['docid'])

    return collection

def load_questions_to_df(path):
    """
    Returns a dataframe of question ids and question text.

    Returns:
        queries: Dataframe
    ----------
    Arguments:
        path: str
    """
    # Question ID and Question text
    query_df = pd.read_csv(path, sep="\t")
    queries = query_df[['qid', 'question']]

    return queries

def load_qid_docid_to_df(path):
    """Returns a dataframe of question id and relevant docid answers.

    Returns:
        qid_docid: Dataframe
    ----------
    Arguments:
        path: str
    """
    qid_docid = pd.read_csv(path, sep="\t")
    qid_docid = qid_docid [['qid', 'docid']]

    return qid_docid

def save_tsv(path, df):
    """Saves a dataframe to tsv file.
    ----------
    Arguments:
        path: str
        df: Dataframe
    """
    with open(path,'w') as write_tsv:
        write_tsv.write(df.to_csv(sep='\t', index=False, header=False))

def collection_to_json(json_path, collection_path):
    """Converts a df to JSON file for Anserini's document indexer.
    ----------
    Arguments:
        json_path: str output file path
        collection_path: str input file path
    """
    output_jsonl_file = open(json_path, 'w', encoding='utf-8', newline='\n')

    with open(collection_path, encoding='utf-8') as f:
        for i, line in enumerate(f):
            # Extract data
            id_text = line.rstrip().split('\t')
            doc_id, doc_text = id_text[0], id_text[1]
            # Create dictionary with docid and doc text
            output_dict = {'id': doc_id, 'contents': doc_text}
            # Dump each docid to json file
            output_jsonl_file.write(json.dumps(output_dict) + '\n')

def split_label(qid_docid):
    """
    Split question answer pairs into train, test, validation sets.

    Returns:
        train_label: Dictonary
            key - question id
            value - list of relevant docids
        test_label: Dictonary
            key - question id
            value - list of relevant docids
        valid_label: Dictonary
            key - question id
            value - list of relevant docids
    ----------
    Arguments:
        qid_docid: Dataframe containing the question id and relevant docids
    """
    # Group the answers for each question into a list
    qid_docid = qid_docid.groupby(['qid']).agg(lambda x: tuple(x)).applymap(list).reset_index()
    # Split data
    train, test_set = train_test_split(qid_docid, test_size=0.05)
    train_set, valid_set = train_test_split(train, test_size=0.1)
    # Expand the list of docids into individual rows to represent a single sample
    train_data = train_set.explode('docid')
    test_data = test_set.explode('docid')
    valid_data = valid_set.explode('docid')

    # Convert data into dictionary - key: qid, value: list of relevant docid
    train_label = label_to_dict(train_data)
    test_label = label_to_dict(test_data)
    valid_label = label_to_dict(valid_data)

    return train_label, test_label, valid_label

def split_question(train_label, test_label, valid_label, queries):
    """
    Split questions into train, test, validation sets.

    Returns:
        train_questions: Dataframe with qids
        test_questions: Dataframe with qids
        valid_questions: Dataframe with qids
    ----------
    Arguments:
        train_label: Dictionary contraining qid and list of relevant docid
        test_label: Dictionary contraining qid and list of relevant docid
        valid_label: Dictionary contraining qid and list of relevant docid
        queries: Dataframe containing the question id and question text
    """
    # Get a list of question ids
    train_q = list(train_label.keys())
    test_q = list(test_label.keys())
    valid_q = list(valid_label.keys())

    # Split question dataframe into train, test, valid set
    train_questions = queries[queries['qid'].isin(train_q)]
    test_questions = queries[queries['qid'].isin(test_q)]
    valid_questions = queries[queries['qid'].isin(valid_q)]

    return train_questions, test_questions, valid_questions

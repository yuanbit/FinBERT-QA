import pandas as pd
from pathlib import Path
import csv
from itertools import islice
import nltk
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import json

from utils import *

def get_empty_docs(collection):
    empty_docs = []
    empty_id = []

    for index, row in collection.iterrows():
        if pd.isna(row['doc']):
            empty_docs.append(row['docid'])
            empty_id.append(index)

    return empty_docs, empty_id

def label_to_dict(df):
    """
    Convert dataframe to dictionary
    """
    qid_rel = {}
    for index, row in df.iterrows():
        if row['qid'] not in qid_rel:
            qid_rel[row['qid']] = []
        qid_rel[row['qid']].append(row['docid'])

    return qid_rel

def load_answers_to_dict(path):
    """
    dict - docid map to doc text
    """
    docs = {}

    with open(path,'r') as f:
        for line in f:
            # [qid, doc_id, rank]
            line = line.strip().split('\t')

            if len(line) == 4:
                docid = int(line[1])
                doc = line[2]

                docs[docid] = []
                docs[docid].append(doc)

    return docs

def load_answers_to_df(path):
    """
    Returns a dataframe of cols: docid, doc
    """
    # Doc ID to Doc text
    collection = pd.read_csv(path, sep="\t")
    collection = collection[['docid', 'doc']]
    collection = collection.sort_values(by=['docid'])

    return collection

def load_questions_to_df(path):
    """
    Returns a dataframe of cols: qid, question
    """
    # Question ID and Question text
    query_df = pd.read_csv(path, sep="\t")
    queries = query_df[['qid', 'question']]

    return queries

def load_qid_docid_to_df(path):
    qid_docid = pd.read_csv(path, sep="\t")
    qid_docid = qid_docid [['qid', 'docid']]

    return qid_docid

def save_tsv(path, df):
    with open(path,'w') as write_tsv:
        write_tsv.write(df.to_csv(sep='\t', index=False, header=False))

def collection_to_json(json_path, collection_path):
    # Convert collection df to JSON file for Anserini's document indexer
    output_jsonl_file = open(json_path, 'w', encoding='utf-8', newline='\n')

    with open(collection_path, encoding='utf-8') as f:
        for i, line in enumerate(f):
            id_text = line.rstrip().split('\t')
            doc_id, doc_text = id_text[0], id_text[1]
            output_dict = {'id': doc_id, 'contents': doc_text}
            output_jsonl_file.write(json.dumps(output_dict) + '\n')

def split_label(qid_docid):
    """
    Split question answer pairs into train, test, validation sets
    """
    # Group the answers for each question into a list
    qid_docid = qid_docid.groupby(['qid']).agg(lambda x: tuple(x)).applymap(list).reset_index()

    # Split data
    train, test_set = train_test_split(qid_docid, test_size=0.05)
    train_set, valid_set = train_test_split(train, test_size=0.1)

    train_data = train_set.explode('docid')
    test_data = test_set.explode('docid')
    valid_data = valid_set.explode('docid')

    # Convert data into dictionary - key: qd, value: list of relevant docid
    train_label = label_to_dict(train_data)
    test_label = label_to_dict(test_data)
    valid_label = label_to_dict(valid_data)

    return train_label, test_label, valid_label

def split_question(train_label, test_label, valid_label, queries):
    # Get a list of question ids
    train_q = list(train_label.keys())
    test_q = list(test_label.keys())
    valid_q = list(valid_label.keys())

    # Split question dataframe into train, test, valid set
    train_questions = queries[queries['qid'].isin(train_q)]
    test_questions = queries[queries['qid'].isin(test_q)]
    valid_questions = queries[queries['qid'].isin(valid_q)]

    return train_questions, test_questions, valid_questions

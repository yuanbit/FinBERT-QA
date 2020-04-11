import pandas as pd
from pathlib import Path
from itertools import islice
import pickle
import json

from .download import *

def take(n, iterable):
    """Return first n items of a dictiionary as a list.
    ----------
    Arguments:
        n: int
            First n itmes to return
        iterable: dictionary
            Dictionary to return
    """
    return list(islice(iterable, n))

def load_pickle(path):
    """Load pickle file.
    ----------
    Arguments:
        path: str file path
    """
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_pickle(path, data):
    """Save Python object to pickle.
    ----------
    Arguments:
        path: str file path
        data: Python object
    """
    with open(path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

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

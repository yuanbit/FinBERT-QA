import pandas as pd
import regex as re
import csv
from itertools import islice
import pickle
import numpy as np
import json
import os
import sys
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
from pathlib import Path
# os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"
from pyserini.search import pysearch

from utils import *

path = str(Path.cwd())

# Lucene indexer
fiqa_index = path + "/retriever/lucene-index-fiqa/"

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

def create_dataset(question_df, labels, cands_size):
    """Retrieves the top-k candidate answers for a question and
    creates a list of lists of the dataset containing the question id,
    list of relevant answer ids, and the list of answer candidates

    Returns:
        dataset: list of list in the form [qid, [pos ans], [ans candidates]]
    ----------
    Arguments:
        question_df: Dataframe containing the qid and question text
        labels: Dictonary containing the qid to text map
        cands_size: int - number of candidates to retrieve
    """
    dataset = []
    # Calls retriever
    searcher = pysearch.SimpleSearcher(fiqa_index)
    # For each question
    for i, row in question_df.iterrows():
        qid = row['qid']
        tmp = []
        # Append qid
        tmp.append(qid)
        # Append list of relevant docs
        tmp.append(labels[qid])
        # Retrieves answer candidates
        cands = []
        query = row['question']
        query = re.sub('[£€§]', '', query)
        hits = searcher.search(query, k=cands_size)

        for docid in range(0, len(hits)):
            cands.append(int(hits[docid].docid))
        # Append candidate answers
        tmp.append(cands)
        dataset.append(tmp)

    return dataset

def get_dataset(query_path, labels_path, cands_size):
    """Splits the dataset into train, validation, and test set and creates
    the dataset form for training, validation, and testing.

    Returns:
        train_set: list of list in the form [qid, [pos ans], [ans candidates]]
        valid_set: list of list in the form [qid, [pos ans], [ans candidates]]
        test_set: list of list in the form [qid, [pos ans], [ans candidates]]
    ----------
    Arguments:
        query_path: str - path containing a list of qid and questions
        labels_path: str - path containing a list of qid and relevant docid
        cands_size: int - number of candidates to retrieve
    """
    # Question id and Question text
    queries = load_questions_to_df(query_path)
    # Question id and Answer id pair
    qid_docid = load_qid_docid_to_df(labels_path)
    # qid to docid label map
    labels = label_to_dict(qid_docid)
    train_label, test_label, valid_label = split_label(qid_docid)
    # Split Questions
    train_questions, test_questions, \
    valid_questions = split_question(train_label, test_label, valid_label, queries)

    print("\nGenerating training set...\n")
    train_set = create_dataset(train_questions, labels, cands_size)
    print("Generating validation set...\n")
    valid_set = create_dataset(valid_questions, labels, cands_size)
    print("Generating test set...\n")
    test_set = create_dataset(test_questions, labels, cands_size)

    return train_set, valid_set, test_set

def main():

    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--query_path", default=None, type=str, required=True,
    help="Path to the question id to text data in .tsv format. Each line should have at least two columns named (qid, question) separated by tab")
    parser.add_argument("--label_path", default=None, type=str, required=True,
    help="Path to the question id and answer id data in .tsv format. Each line should have at two columns named (qid, docid) separated by tab")

    # Optional parameters
    parser.add_argument("--cands_size", default=50, type=int, required=False,
    help="Number of candidates to retrieve per question.")
    parser.add_argument("--output_dir", default=Path.cwd()/'data/data_pickle/',
    type=str, required=False, help="The output directory where the generated data will be stored.")

    args = parser.parse_args()

    if len(sys.argv) < 4:
        print("Usage: python3 src/generate_data.py <query_path> <label_path>")
        sys.exit()

    train_set, valid_set, test_set = get_dataset(args.query_path, \
                                                 args.label_path, \
                                                 args.cands_size)

    save_pickle(args.output_dir + "train_set.pickle", train_set)
    save_pickle(args.output_dir + "valid_set.pickle", valid_set)
    save_pickle(args.output_dir + "test_set.pickle", test_set)

    print("Done. The pickle files are saved in {}".format(args.output_dir))

if __name__ == "__main__":
    main()

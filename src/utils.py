import pandas as pd
from itertools import islice
import pickle
import json
import os
import zipfile
import requests
from pathlib import Path
from tqdm import tqdm

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

def download_file(url, path, filename, zip=False):
    """Downloads and extracts zip file.
    ----------
    Arguments:
        url: str - zip url
        path: str - the path to download the file
        filename: str - name of the file
        zip - bool - if file is zip or not
    """
    # https://stackoverflow.com/questions/37573483/progress-bar-while-download-file-over-http-with-requests/37573701
    # Streaming
    r = requests.get(url, stream=True)
    # Total size in bytes.
    total_size = int(r.headers.get('content-length', 0))
    block_size = 1024 #1 Kibibyte

    t=tqdm(total=total_size, unit='iB', unit_scale=True)
    # Download file
    with open(path/filename, 'wb') as f:
        for data in r.iter_content(block_size):
            t.update(len(data))
            f.write(data)
    t.close()
    if total_size != 0 and t.n != total_size:
        print("ERROR, download failed.")

    if zip == True:
        # Extract and delete zip file
        with open(path/filename, 'rb') as fileobj:
            z = zipfile.ZipFile(fileobj)
            z.extractall(path)
            z.close()
        os.remove(path/filename)
    else:
        pass

def get_model(model_name):
    """Creates model directory and downloads models.
    ----------
    Arguments:
        model_name: str
    """
    model_path = Path.cwd()/'model'/model_name

    # If dir does not exist, make new dir
    if not os.path.isdir(model_path):
        os.makedirs(model_path)

    zip_name = model_name + ".zip"

    # If model does not exist
    if not os.path.exists(model_path/"pytorch_model.bin"):

        if model_name == "finbert-domain":
            url = "https://www.dropbox.com/s/3vp2fje2x0hwd84/finbert-domain.zip?dl=1"
        elif model_name == "finbert-task":
            url = "https://www.dropbox.com/s/0vgwzcjt9tx8b1b/finbert-task.zip?dl=1"
        else:
            url = "https://www.dropbox.com/s/sh2h9o5yd7v4ku6/bert-qa.zip?dl=1"

        print("\nDownloading {} model...\n".format(model_name))
        download_file(url, model_path, zip_name, zip=True)

def get_trained_model(model_name):
    """Creates trained/fine-tuned model directory and downloads models.
    ----------
    Arguments:
        model_name: str
    """
    model_path = Path.cwd()/'model'/'trained'/model_name

    # If dir does not exist, make new dir
    if not os.path.isdir(model_path):
        os.makedirs(model_path)

    if model_name == "qa-lstm":
        filename = "3_lstm50_128_64_1e3.pt"
        url = "https://www.dropbox.com/s/6ohy8r1risxom3e/3_lstm50_128_64_1e3.pt?dl=1"
    elif model_name == "bert-pointwise":
        filename = "2_pointwise50_512_16_3e6.pt"
        url = "https://www.dropbox.com/s/wow4d8n9jn3lgof/2_pointwise50_512_16_3e6.pt?dl=1"
    elif model_name == "bert-pairwise":
        filename = "1_pairwisewise50_128_32_3e6_05.pt"
        url = "https://www.dropbox.com/s/k6ey5ez55uslosk/1_pairwisewise50_128_32_3e6_05.pt?dl=1"
    elif model_name == "finbert-domain":
        filename = "2_finbert-domain-50_512_16_3e6.pt"
        url = "https://www.dropbox.com/s/a3h5oszxn6d7azj/2_finbert-domain-50_512_16_3e6.pt?dl=1"
    elif model_name == "finbert-task":
        filename = "2_finbert-task-50_512_16_3e6.pt"
        url = "https://www.dropbox.com/s/h29fk9xi2cennp7/2_finbert-task-50_512_16_3e6.pt?dl=1"
    else:
        filename = "2_finbert-qa-50_512_16_3e6.pt"
        url = "https://www.dropbox.com/s/12uiuumz4vbqvhk/2_finbert-qa-50_512_16_3e6.pt?dl=1"

    # If model does not exist
    if not os.path.exists(model_path/filename):
        print("\nDownloading trained/fine-tuned {} model...\n".format(model_name))
        download_file(url, model_path, filename)
    else:
        pass

    return filename

import os
import zipfile
import requests
from pathlib import Path
from tqdm import tqdm

def download_zip(url, path, zip_name):
    """Downloads and extracts zip file.
    ----------
    Arguments:
        url: str - zip url
        path: str - the path to download the file
        zip_name: str - name of the zip file
    """
    # https://stackoverflow.com/questions/37573483/progress-bar-while-download-file-over-http-with-requests/37573701
    # Streaming
    r = requests.get(url, stream=True)
    # Total size in bytes.
    total_size = int(r.headers.get('content-length', 0))
    block_size = 1024 #1 Kibibyte

    t=tqdm(total=total_size, unit='iB', unit_scale=True)

    with open(path/zip_name, 'wb') as f:
        for data in r.iter_content(block_size):
            t.update(len(data))
            f.write(data)
    t.close()
    if total_size != 0 and t.n != total_size:
        print("ERROR, download failed.")

    # Extract and delete zip file
    with open(path/zip_name, 'rb') as fileobj:
        z = zipfile.ZipFile(fileobj)
        z.extractall(path)
        z.close()
    os.remove(path/zip_name)

def get_data(data_name):
    """Creates input data directory and downloads input data.
    ----------
    Arguments:
        data_name: str
    """
    data_path = Path.cwd()/'data'/data_name

    # If dir does not exist, make new dir
    if not os.path.isdir(data_path):
        os.makedirs(data_path)

    zip_name = data_name + ".zip"

    # If model does not exist
    if not os.path.exists(data_path/"train_q_input.pickle"):
        print("\nDownloading {} input data...\n".format(data_name))
        
        if data_name == "qa-lstm":
            url = "https://www.dropbox.com/s/1u9a2mu1oqujw03/qa-lstm.zip?dl=1"
            download_zip(url, data_path, zip_name)
        elif data_name == "pointwise-bert":
            url = "https://www.dropbox.com/s/w6rctzpbvun1c0x/pointwise-bert.zip?dl=1"
            download_zip(url, data_path, zip_name)
        else:
            url = "https://www.dropbox.com/s/m9a21c4ubwrzkwx/pairwise-bert.zip?dl=1"
            download_zip(url, data_path, zip_name)

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
            url = "https://www.dropbox.com/s/6yiye1qdzrnv96e/finbert-domain.zip?dl=1"
        elif model_name == "finbert-task":
            url = "https://www.dropbox.com/s/0vgwzcjt9tx8b1b/finbert-task.zip?dl=1"
        else:
            url = "https://www.dropbox.com/s/sh2h9o5yd7v4ku6/bert-qa.zip?dl=1"

        print("\nDownloading {} model...\n".format(model_name))
        download_zip(url, model_path, zip_name)

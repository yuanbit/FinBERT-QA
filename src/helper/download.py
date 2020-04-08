import os
import zipfile
import requests
from pathlib import Path
from tqdm import tqdm

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

    if data_name == "qa-lstm":
        # If model does not exist
        if not os.path.exists(data_path/"train_q_input.pickle"):
            print("\nDownloading {} input data...\n".format(data_name))
            url = "https://www.dropbox.com/s/1u9a2mu1oqujw03/qa-lstm.zip?dl=1"
            download_file(url, data_path, zip_name, zip=True)
        else:
            pass
    elif data_name == "pointwise-bert":
        if not os.path.exists(data_path/"train_input_512.pickle"):
            print("\nDownloading {} input data...\n".format(data_name))
            url = "https://www.dropbox.com/s/w6rctzpbvun1c0x/pointwise-bert.zip?dl=1"
            download_file(url, data_path, zip_name, zip=True)
        else:
            pass
    else:
        if not os.path.exists(data_path/"train_pos_input_128_50.pickle"):
            print("\nDownloading {} input data...\n".format(data_name))
            url = "https://www.dropbox.com/s/m9a21c4ubwrzkwx/pairwise-bert.zip?dl=1"
            download_file(url, data_path, zip_name, zip=True)
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

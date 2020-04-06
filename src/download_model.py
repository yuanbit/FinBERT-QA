import os
import zipfile
import requests
from pathlib import Path
from tqdm import tqdm

def get_model(model_name):
    """Creates model directory and downloads models.

    Arguments:
        model_name: str
    """
    model_path = Path.cwd()/'model'/model_name

    # If dir does not exist, make new dir
    if not os.path.isdir(model_path):
        os.makedirs(model_path)

    zip_name = model_name + ".zip"

    # If zip file does not exist
    if not os.path.exists(model_path/zip_name):

        if model_name == "finbert-domain":
            url = "https://www.dropbox.com/s/6yiye1qdzrnv96e/finbert-domain.zip?dl=1"
        elif model_name == "finbert-task":
            url = "https://www.dropbox.com/s/0vgwzcjt9tx8b1b/finbert-task.zip?dl=1"
        else:
            url = "https://www.dropbox.com/s/sh2h9o5yd7v4ku6/bert-qa.zip?dl=1"

        # https://stackoverflow.com/questions/37573483/progress-bar-while-download-file-over-http-with-requests/37573701
        # Streaming
        r = requests.get(url, stream=True)
        # Total size in bytes.
        total_size = int(r.headers.get('content-length', 0))
        block_size = 1024 #1 Kibibyte

        print("\nDownloading {} model...\n".format(model_name))
        t=tqdm(total=total_size, unit='iB', unit_scale=True)

        with open(model_path/zip_name, 'wb') as f:
            for data in r.iter_content(block_size):
                t.update(len(data))
                f.write(data)
        t.close()
        if total_size != 0 and t.n != total_size:
            print("ERROR, something went wrong")

        # Extract and delete zip file
        with open(model_path/zip_name, 'rb') as fileobj:
            z = zipfile.ZipFile(fileobj)
            z.extractall(model_path)
            z.close()
        os.remove(model_path/zip_name)

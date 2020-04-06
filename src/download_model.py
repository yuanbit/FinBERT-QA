import os
import zipfile
import requests
from pathlib import Path
from tqdm import tqdm

model_path = Path.cwd()/'model'/'finbert-domain'

if not os.path.isdir(model_path):
    os.makedirs(model_path)

if not os.path.exists(model_path/"finbert-domain.zip"):
    url = "https://www.dropbox.com/s/6yiye1qdzrnv96e/finbert-domain.zip?dl=1"

    # Streaming, so we can iterate over the response.
    r = requests.get(url, stream=True)
    # Total size in bytes.
    total_size = int(r.headers.get('content-length', 0))
    block_size = 1024 #1 Kibibyte
    t=tqdm(total=total_size, unit='iB', unit_scale=True)

    print("\nDownloading finbert-domain model...\n")
    with open(model_path/"finbert-domain.zip", 'wb') as f:
        for data in r.iter_content(block_size):
            t.update(len(data))
            f.write(data)
    t.close()
    if total_size != 0 and t.n != total_size:
        print("ERROR, something went wrong")

    with open(model_path/"finbert-domain.zip", 'rb') as fileobj:
        z = zipfile.ZipFile(fileobj)
        z.extractall(model_path)
        z.close()
    os.remove(model_path/"finbert-domain.zip")

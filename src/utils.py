from pathlib import Path
from itertools import islice
import pickle
import json

def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_pickle(path, data):
    with open(path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

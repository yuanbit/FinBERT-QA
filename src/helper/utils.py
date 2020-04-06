from pathlib import Path
from itertools import islice
import pickle
import json

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

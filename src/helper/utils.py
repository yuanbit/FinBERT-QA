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

def load_input_data(data_name):
    """Download and load default input data.
    ----------
    Arguments:
        data_name: str - type of input data based on the model
    """
    print("\nLoading input data...\n")
    # Download and extract input data
    get_data(data_name)
    # Load data
    if data_name == "qa-lstm":
        path = "../fiqa/data/qa-lstm/"

        train_q_input = load_pickle(path+'train_q_input.pickle')
        train_pos_input = load_pickle(path+'train_pos_input.pickle')
        train_neg_input = load_pickle(path+'train_neg_input.pickle')

        valid_q_input = load_pickle(path+'valid_q_input.pickle')
        valid_pos_input = load_pickle(path+'valid_pos_input.pickle')
        valid_neg_input = load_pickle(path+'valid_neg_input.pickle')

        return train_q_input, train_pos_input, train_neg_input, \
               valid_q_input, valid_pos_input, valid_neg_input
    elif data_name == "pointwise-bert":
        path = "../fiqa/data/pointwise-bert/"

        train_input = load_pickle(path+'train_input_512.pickle')
        train_type_id = load_pickle(path+'train_type_id_512.pickle')
        train_att_mask = load_pickle(path+'train_mask_512.pickle')
        train_label = load_pickle(path+'train_labels_512.pickle')

        valid_input = load_pickle(path+'valid_input_512.pickle')
        valid_type_id = load_pickle(path+'valid_type_id_512.pickle')
        valid_att_mask = load_pickle(path+'valid_mask_512.pickle')
        valid_label = load_pickle(path+'valid_labels_512.pickle')

        return train_input, train_type_id, train_att_mask, train_label, \
               valid_input, valid_type_id, valid_att_mask, valid_label
    else:
        path = "../fiqa/data/pairwise-bert/"

        train_pos_input = load_pickle(path+'train_pos_input_128_50.pickle')
        train_pos_type_id = load_pickle(path+'train_pos_type_id_128_50.pickle')
        train_pos_mask = load_pickle(path+'train_pos_mask_128_50.pickle')
        train_pos_label = load_pickle(path+'train_pos_labels_50.pickle')

        train_neg_input = load_pickle(path+'train_neg_input_128_50.pickle')
        train_neg_type_id = load_pickle(path+'train_neg_type_id_128_50.pickle')
        train_neg_mask = load_pickle(path+'train_neg_mask_128_50.pickle')
        train_neg_label = load_pickle(path+'train_neg_labels_50.pickle')

        valid_pos_input = load_pickle(path+'valid_pos_input_128_50.pickle')
        valid_pos_type_id = load_pickle(path+'valid_pos_type_id_128_50.pickle')
        valid_pos_mask = load_pickle(path+'valid_pos_mask_128_50.pickle')
        valid_pos_label = load_pickle(path+'valid_pos_labels_50.pickle')

        valid_neg_input = load_pickle(path+'valid_neg_input_128_50.pickle')
        valid_neg_type_id = load_pickle(path+'valid_neg_type_id_128_50.pickle')
        valid_neg_mask = load_pickle(path+'valid_neg_mask_128_50.pickle')
        valid_neg_label = load_pickle(path+'valid_neg_labels_50.pickle')

        return train_pos_input, train_pos_type_id, train_pos_mask, \
               train_pos_label, train_neg_input, train_neg_type_id, \
               train_neg_mask, train_neg_label, valid_pos_input, \
               valid_pos_type_id, valid_pos_mask, valid_pos_label, \
               valid_neg_input, valid_neg_type_id, valid_neg_mask, valid_neg_label

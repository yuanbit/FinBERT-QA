import argparse
from tqdm import tqdm
import random
import math

from helper.utils import *

class get_data:
    def __init__(self, cands_path, qid_rel_path, cands_size):
        """
        qid_rel: dictionary
            Dictionary with question ids and list of relevant answer ids
        cands_path: string
            Path of candidate answers
        cand_size: int
            Candidate answers size
        """
        self.cands_path = cands_path
        self.qid_rel_path = qid_rel_path
        self.cands_size = cands_size

    def load_cands(self):
        """Returns a dictionary of candidate answers for each question.

        qid_ranked_docs: dictionary
            key - qid
            value - list of k ranked candidates
        """
        qid_ranked_docs = {}

        with open(self.cands_path,'r') as f:
            for line in f:
                # Extract data in the form [qid, doc_id, rank]
                line = line.strip().split('\t')
                qid = int(line[0])
                doc_id = int(line[1])
                rank = int(line[2])

                if qid not in qid_ranked_docs:
                    # Create a list for each query to store the candidates
                    candidates = [0]*self.cands_size
                    qid_ranked_docs[qid] = candidates
                qid_ranked_docs[qid][rank-1] = doc_id

        return qid_ranked_docs

    def create_dataset(self):
        """Returns a list of lists of the dataset containing the question id,
        list of relevant answer ids, and the list of answer candidates
        ----------
        dataset: list of list in the form [qid, [pos ans], [ans candidates]]
        """
        # Dictionary of question id and list of positive answers
        qid_rel = load_pickle(self.qid_rel_path)
        # Dictionary of question id and list of candidate answers
        cands = self.load_cands()

        dataset = []

        for qid, docid in qid_rel.items():
            for ques, cand in cands.items():
                if 0 not in cand:
                    cand_ans = cand
                    if ques == qid:
                        tmp = []
                        tmp.append(qid)
                        tmp.append(docid)
                        tmp.append(cand_ans)
                        dataset.append(tmp)

        for row in dataset:
            assert len(row[2]) == self.cands_size, "Dataset size is incorrect!"

        return dataset


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--train_cands", default=None, type=str, required=True,
                        help="Path to the training candidates data in .tsv format. Each line should have three items "
                             "(questiod id, answer id, rank) separated by tab")
    parser.add_argument("--valid_cands", default=None, type=str, required=True,
                        help="Path to the validation candidates data in .tsv format. Each line should have three items "
                             "(questiod id, answer id, rank) separated by tab")
    parser.add_argument("--test_cands", default=None, type=str, required=True,
                        help="Path to the testing candidates data in .tsv format. Each line should have three items "
                             "(questiod id, answer id, rank) separated by tab")
    parser.add_argument("--train_label", default=None, type=str, required=True,
                        help="Path to the training label data in .pickle format.")
    parser.add_argument("--valid_label", default=None, type=str, required=True,
                        help="Path to the validation label data in .pickle format.")
    parser.add_argument("--test_label", default=None, type=str, required=True,
                        help="Path to the testing label data in .pickle format.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the generated data will be stored.")

    # Other parameters
    parser.add_argument("--cands_size", default=50, type=int, required=False,
                        help="Number of candidates per question.")

    args = parser.parse_args()

    print("\nCreating training set...")
    train = get_data(args.train_cands, args.train_label, args.cands_size)
    train_set = train.create_dataset()

    valid = get_data(args.valid_cands, args.valid_label, args.cands_size)
    print("\nCreating validation set...")
    valid_set = valid.create_dataset()

    print("\nCreating test set...")
    test = get_data(args.test_cands, args.test_label, args.cands_size)
    test_set = test.create_dataset()

    save_pickle(args.output_dir + "/train_set.pickle", train_set)
    save_pickle(args.output_dir + "/valid_set.pickle", valid_set)
    save_pickle(args.output_dir + "/test_set.pickle", test_set)

    print("\nDone.")

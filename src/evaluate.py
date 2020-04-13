import pandas as pd
from statistics import mean
import math
import numpy as np
from itertools import islice

def get_rel(labels, cands):
    """Get relevant positions of the hits.

    Returns: List of 0's and 1's incidating a relevant answer
    -------------------
    Arguments:
        labels: List of relevant docids
        cands: List of candidate docids
    """
    rel = []
    for cand in cands:
        if cand in labels:
            rel.append(1)
        else:
            rel.append(0)

    return rel

def dcg(rels, k):
    """
    Discounted Cumulative Gain. Computes the cumulated DCG of the top-k
    relevant docs across all queries.

    Returns:
        cumulated_sum: float - cumulated DCG
    ----------
    Arguments:
        rels: list
            List of relevant scores of 0 or 1 e.g. [0, 1, 0, 1]
        k: int
            Top-k relevant docs
    """
    cumulated_sum = rels[0]
    for i in range(1, k):
        cumulated_sum += rels[i]/math.log(i+1,2)
    return cumulated_sum

def avg_ndcg(rel_score, k):
    """
    Average Normalized Discounted Cumulative Gain. Computes the DCG, iDCG, and
    nDCG for each query and returns the averyage nDCG across all queries.

    Returns:
        avg: float - average nDCG
    ----------
    Arguments:
        rel_score: dictionary
            key - question id
            value - list of relevancy scores with 1 (relevant) and 0 (irrelevant)
            e.g. {0: [0, 1, 0], 1: [1, 1, 0]}
        k: int
            Top-k relevant docs
    """
    ndcg_list = []
    for qid, rels in rel_score.items():
        # Compute DCG for each question
        dcg_val = dcg(rels, k)
        sorted_rel = sorted(rels, reverse=True)
        # Compute iDCG for each question
        idcg_val = dcg(sorted_rel, k)

        try:
            ndcg_val = dcg_val/idcg_val
            ndcg_list.append(ndcg_val)
        except ZeroDivisionError:
            ndcg_list.append(0)

    assert len(ndcg_list) == len(rel_score), "Relevant score doesn't match"

    # Get the average nDCG across all queries
    avg = mean(ndcg_list)

    return avg

def compute_RR(cand_docs, rel_docs, cumulated_reciprocal_rank, rank_pos, k):
    """
    Computes the reciprocal rank - probability of correctness of rank. Returns
    the cumulated reciprocal rank across all queries and the positions of the
    relevant docs in the candidates.

    Returns:
        cumulated_reciprocal_rank: float - cumulated Reciprocal Rank across all queries
        rank_pos: list - index of the relevant docs in the candidates
    ----------
    Arguments:
        cand_docs: list
            List of ranked docids for a question
        rel_docs: list
            List of the relevancy of docids for a question
        cumulated_reciprocal_rank: int
            Initial value = 0
        rank_pos: list
            Initial list = []
        k: int
            Top-k relevant docs
    """

    for i in range(0, k):
        # If the doc_id of the top k ranked candidate passages is in
        # the list of relevant passages
        if cand_docs[i] in rel_docs:
            # Compute the reciprocal rank (i is the ranking)
            rank_pos.append(i+1)
            cumulated_reciprocal_rank += 1/(i+1)
            break

    return cumulated_reciprocal_rank, rank_pos

def create_qid_pred_rank(test_set):
    """Creates dictionary of qid and list of candidates from test set.

    Returns:
        qid_pred_rank: dictionary
            key - qid
            value - list of candidates
    ----------
    Arguments:
        test_set: list
            [[qid, [positive docids], [list of candidates]]]
    """
    qid_pred_rank = {}

    for row in test_set:
        qid_pred_rank[row[0]] = row[2]

    return qid_pred_rank

def evaluate(qid_ranked_docs, qid_rel, k):
    """
    Evaluate. Computes the MRR@k, average nDCG@k, and average precision@k1

    Returns:
        MRR: float
        average_ndcg: float
        avg_precision: float
        r_pos: int
    ----------
    Arguments:
        qid_ranked_docs: dictionary
            key - qid
            value - list of cand ans
        qid_rel:  dinctionary
            key- qid
            value - list of relevant ans
    """
    cumulated_reciprocal_rank = 0
    num_rel_docs = 0
    # Dictionary of the top-k relevancy scores of docs in the candidate answers
    rel_scores = {}
    precision_list = {}
    rank_pos = []

    # For each query
    for qid in qid_ranked_docs:
        # If the query has a relevant passage
        if qid in qid_rel:
            # Get the list of relevant docs for a query
            rel_docs = qid_rel[qid]
            # Get the list of ranked docs for a query
            cand_docs = qid_ranked_docs[qid]
            # Compute relevant scores of the candidates
            if qid not in rel_scores:
                rel_scores[qid] = []
                for i in range(0, k):
                    if cand_docs[i] in rel_docs:
                        rel_scores[qid].append(1)
                    else:
                        rel_scores[qid].append(0)
            # Compute th reciprocal rank and rank positions
            cumulated_reciprocal_rank, r_pos = compute_RR(cand_docs, rel_docs, cumulated_reciprocal_rank, rank_pos, k)

    # Compute the average MRR@k across all queries
    MRR = cumulated_reciprocal_rank/len(qid_ranked_docs)
    # Compute the nDCG@k across all queries
    average_ndcg = avg_ndcg(rel_scores, k)

    # Compute precision@1
    precision_at_k = []
    for qid, score in rel_scores.items():
        num_rel = 0
        for i in range(0, 1):
            if score[i] == 1:
                num_rel += 1
        precision_at_k.append(num_rel/1)

    avg_precision = mean(precision_at_k)

    return MRR, average_ndcg, avg_precision, r_pos

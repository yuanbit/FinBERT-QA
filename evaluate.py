import pandas as pd
from statistics import mean
import math
import numpy as np
from itertools import islice

# Helper functions
def dcg(rels, k):
    """
    Discounted Cumulative Gain

    Returns the cumulated DCG of the top-k relevant docs across all queries
    """
    cumulated_sum = rels[0]
    for i in range(1, k):
        cumulated_sum += rels[i]/math.log(i+1,2)
    return cumulated_sum

def avg_ndcg(rel_score, k):
    """
    Average Normalized Discounted Cumulative Gain

    Computes the DCG, iDCG, and nDCG for each query

    Returns the averyage nDCG across all queries
    """
    ndcg_list = []
    for qid, rels in rel_score.items():
        dcg_val = dcg(rels, k)
        sorted_rel = sorted(rels, reverse=True)
        idcg_val = dcg(sorted_rel, k)

        try:
            ndcg_val = dcg_val/idcg_val
            ndcg_list.append(ndcg_val)
        except ZeroDivisionError:
            ndcg_list.append(0)

    assert len(ndcg_list) == len(rel_score), "Relevant score doesn't match"

    avg = mean(ndcg_list)

    return avg

def compute_RR(cand_docs, rel_docs, cumulated_reciprocal_rank, rank_pos, k):
    """
    Computes the reciprocal rank - probability of correctness of rank

    Returns the cumulated reciprocal rank across all queries and the
    positions of the relevant docs in the candidates
    """

    for i in range(0, k):
        # If the doc_id of the top k ranked candidate passages is in the list of relevant passages
        if cand_docs[i] in rel_docs:
            # Compute the reciprocal rank (i is the ranking)
            rank_pos.append(i+1)
            cumulated_reciprocal_rank += 1/(i+1)
            break

    return cumulated_reciprocal_rank, rank_pos


# Evaluate top-k candidates and precision@1
def evaluate(qid_ranked_docs, qid_rel, k):
    """
    qid_ranked_docs: dict - key - qid, value - list of cand ans
    qid_rel:  key- qid, value - list of relevant ans
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

            cumulated_reciprocal_rank, r_pos = compute_RR(cand_docs, rel_docs, cumulated_reciprocal_rank, rank_pos, k)

    MRR = cumulated_reciprocal_rank/len(qid_rel)
    average_ndcg = avg_ndcg(rel_scores, k)

    precision_at_k = []
    for qid, score in rel_scores.items():
        num_rel = 0
        for i in range(0, 1):
            if score[i] == 1:
                num_rel += 1
        precision_at_k.append(num_rel/1)

    return MRR, average_ndcg, mean(precision_at_k)

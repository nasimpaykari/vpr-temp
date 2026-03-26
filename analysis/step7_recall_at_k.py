# -*- coding: utf-8 -*-
"""
STEP 7: Recall@K evaluation for VPR
Python 2.7 compatible
"""

from __future__ import print_function
import os
import numpy as np

# ================= CONFIG =================
DATASET_NAME = "MY_DATASET"
METHOD_NAME = "NetVLAD"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PRECOMPUTED_BASE = os.path.abspath(
    os.path.join(BASE_DIR, "../precomputed_matches", DATASET_NAME + "_ALL")
)

GT_PATH = os.path.abspath(
    os.path.join(BASE_DIR, "../datasets", DATASET_NAME, "ground_truth_new.npy")
)

# Choose K values to evaluate
K_VALUES = [1, 5, 10, 20]


# ================= LOAD GROUND TRUTH =================
def load_ground_truth(gt_path):
    data = np.load(gt_path, allow_pickle=True)

    gt_dict = {}
    for item in data:
        q_idx = int(item[0])
        refs = [int(r) for r in item[1]]
        gt_dict[q_idx] = refs

    return gt_dict


# ================= LOAD METHOD =================
def load_method(method_name):
    npy_path = os.path.join(
        PRECOMPUTED_BASE,
        method_name,
        "precomputed_data_corrected.npy"
    )

    data = np.load(npy_path, allow_pickle=True)

    queries = data[0]
    similarity_matrix = data[3]  # (220, 328)

    return queries, similarity_matrix


# ================= RECALL@K =================
def compute_recall_at_k(gt_dict, queries, similarity_matrix, K):

    correct = 0
    total = 0

    for i in range(len(queries)):

        q_idx = int(queries[i])

        if q_idx not in gt_dict:
            continue

        total += 1

        sim_row = similarity_matrix[i]

        # Get top-K indices (sorted descending)
        top_k_indices = np.argsort(sim_row)[::-1][:K]

        gt_refs = gt_dict[q_idx]

        # Check if any of top-K is in ground truth
        found = False
        for idx in top_k_indices:
            if idx in gt_refs:
                found = True
                break

        if found:
            correct += 1

    recall = float(correct) / total if total > 0 else 0

    return recall, correct, total


# ================= MAIN =================
if __name__ == "__main__":

    gt_dict = load_ground_truth(GT_PATH)
    queries, sim_matrix = load_method(METHOD_NAME)

    print("\n===== Recall@K Evaluation =====")

    for K in K_VALUES:
        recall, correct, total = compute_recall_at_k(
            gt_dict, queries, sim_matrix, K
        )

        print("\nRecall@{}: {:.4f}".format(K, recall))
        print("Correct:", correct, " / Total:", total)

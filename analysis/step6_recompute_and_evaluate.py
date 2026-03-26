# -*- coding: utf-8 -*-
"""
STEP 6: Recompute predictions from similarity matrix
and compute correct / incorrect matches (VPR-Bench style)

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

# ================= LOAD GROUND TRUTH =================
def load_ground_truth(gt_path):
    data = np.load(gt_path, allow_pickle=True)

    gt_dict = {}
    for item in data:
        q_idx = int(item[0])
        refs = [int(r) for r in item[1]]
        gt_dict[q_idx] = refs

    return gt_dict


# ================= LOAD METHOD DATA =================
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


# ================= EVALUATION =================
def evaluate_vpr(gt_dict, queries, similarity_matrix):

    correct_matches = []
    incorrect_matches = []

    tp = 0
    total = 0

    for i in range(len(queries)):

        q_idx = int(queries[i])

        if q_idx not in gt_dict:
            continue

        total += 1

        # ----- Prediction from similarity -----
        sim_row = similarity_matrix[i]

        # higher similarity = better match
        pred_idx = int(np.argmax(sim_row))

        gt_refs = gt_dict[q_idx]

        if pred_idx in gt_refs:
            correct_matches.append(q_idx)
            tp += 1
        else:
            incorrect_matches.append(q_idx)

    recall_at_1 = float(tp) / total if total > 0 else 0

    print("\n===== VPR-Bench Style Evaluation =====")
    print("Total queries evaluated:", total)
    print("Correct matches:", len(correct_matches))
    print("Incorrect matches:", len(incorrect_matches))
    print("Recall@1:", recall_at_1)

    return correct_matches, incorrect_matches


# ================= MAIN =================
if __name__ == "__main__":

    gt_dict = load_ground_truth(GT_PATH)
    queries, sim_matrix = load_method(METHOD_NAME)

    correct_matches, incorrect_matches = evaluate_vpr(
        gt_dict, queries, sim_matrix
    )

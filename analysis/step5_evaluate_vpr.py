# -*- coding: utf-8 -*-
"""
STEP 5: Correct VPR evaluation (TP / FP / FN)
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


# ================= LOAD METHOD =================
def load_method(method_name):
    npy_path = os.path.join(
        PRECOMPUTED_BASE,
        method_name,
        "precomputed_data_corrected.npy"
    )

    data = np.load(npy_path, allow_pickle=True)

    return {
        'query_indices': data[0],
        'predictions': data[1]
    }


# ================= EVALUATION =================
def evaluate(gt_dict, method_data):
    query_indices = method_data['query_indices']
    predictions = method_data['predictions']

    tp = 0
    fp = 0

    total = 0

    for i, q_idx in enumerate(query_indices):

        if q_idx not in gt_dict:
            continue

        total += 1

        pred = int(predictions[i])
        gt_refs = gt_dict[q_idx]

        if pred in gt_refs:
            tp += 1
        else:
            fp += 1

    fn = fp  # in top-1 retrieval

    accuracy = float(tp) / total if total > 0 else 0
    precision = float(tp) / (tp + fp) if (tp + fp) > 0 else 0
    recall = float(tp) / (tp + fn) if (tp + fn) > 0 else 0

    print("\n===== RESULTS =====")
    print("Total queries:", total)
    print("TP:", tp)
    print("FP:", fp)
    print("FN:", fn)

    print("\nAccuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)


# ================= MAIN =================
if __name__ == "__main__":
    gt_dict = load_ground_truth(GT_PATH)
    method_data = load_method(METHOD_NAME)

    evaluate(gt_dict, method_data)

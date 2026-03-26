# -*- coding: utf-8 -*-
"""
Multi-method, multi-dataset overlay:
- One figure per method
- Each figure compares:
    corridor vs MY_DATASET

Outputs:
- Rank CDF overlay
- Recall@K overlay

Python 2.7 compatible
"""

from __future__ import print_function
import os
import numpy as np
import matplotlib.pyplot as plt

# ================= CONFIG =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "results_plots")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

DATASETS = {
    "corridor": {
        "path": os.path.abspath(os.path.join(BASE_DIR, "../precomputed_matches/corridor")),
        "gt": os.path.abspath(os.path.join(BASE_DIR, "../datasets/corridor/ground_truth_new.npy"))
    },
    "MY_DATASET": {
        "path": os.path.abspath(os.path.join(BASE_DIR, "../precomputed_matches/MY_DATASET_ALL")),
        "gt": os.path.abspath(os.path.join(BASE_DIR, "../datasets/MY_DATASET/ground_truth_new.npy"))
    }
}

METHODS = [
    "AlexNet_VPR",
    "AMOSNet",
    "CALC",
    "CoHOG",
    "HOG",
    "HybridNet",
    "NetVLAD",
    "RegionVLAD"
]

K_VALUES = [1, 5, 10, 20]


# ================= LOAD GT =================
def load_gt(gt_path):
    data = np.load(gt_path, allow_pickle=True)
    gt_dict = {}
    for item in data:
        q_idx = int(item[0])
        refs = [int(r) for r in item[1]]
        gt_dict[q_idx] = refs
    return gt_dict


# ================= LOAD METHOD =================
def load_method(dataset_path, method):
    path = os.path.join(dataset_path, method, "precomputed_data_corrected.npy")
    data = np.load(path, allow_pickle=True)

    return data[0], data[3]


# ================= RANKS =================
def compute_ranks(gt_dict, queries, sim_matrix):
    ranks = []

    for i in range(len(queries)):
        q_idx = int(queries[i])

        if q_idx not in gt_dict:
            continue

        sim_row = sim_matrix[i]
        sorted_indices = np.argsort(sim_row)[::-1]

        gt_refs = gt_dict[q_idx]

        rank = None
        for r, idx in enumerate(sorted_indices):
            if idx in gt_refs:
                rank = r + 1
                break

        if rank is not None:
            ranks.append(rank)

    return np.array(ranks)


# ================= RECALL@K =================
def compute_recall_at_k(gt_dict, queries, sim_matrix, K):
    correct = 0
    total = 0

    for i in range(len(queries)):
        q_idx = int(queries[i])

        if q_idx not in gt_dict:
            continue

        total += 1

        sim_row = sim_matrix[i]
        top_k = np.argsort(sim_row)[::-1][:K]

        gt_refs = gt_dict[q_idx]

        found = False
        for idx in top_k:
            if idx in gt_refs:
                found = True
                break

        if found:
            correct += 1

    return float(correct) / total if total > 0 else 0


# ================= MAIN LOOP =================
for method in METHODS:

    print("\n==============================")
    print("Processing method:", method)
    print("==============================\n")

    dataset_results = {}

    # ---- Load both datasets ----
    for dataset_name in DATASETS:

        gt_dict = load_gt(DATASETS[dataset_name]["gt"])
        queries, sim_matrix = load_method(DATASETS[dataset_name]["path"], method)

        ranks = compute_ranks(gt_dict, queries, sim_matrix)

        recall_curve = []
        for K in K_VALUES:
            r = compute_recall_at_k(gt_dict, queries, sim_matrix, K)
            recall_curve.append(r)

        dataset_results[dataset_name] = {
            "ranks": ranks,
            "recall": recall_curve
        }

    # ================= RANK CDF PLOT =================
    plt.figure()

    for dataset_name in dataset_results:
        ranks = dataset_results[dataset_name]["ranks"]
        sorted_ranks = np.sort(ranks)
        cdf = np.arange(1, len(sorted_ranks) + 1) / float(len(sorted_ranks))

        plt.plot(sorted_ranks, cdf, label=dataset_name)

    plt.xlabel("Rank")
    plt.ylabel("Cumulative Probability")
    plt.title("Rank CDF - " + method)
    plt.legend()

    out_png = os.path.join(OUTPUT_DIR, "rank_cdf_" + method + ".png")
    out_pdf = os.path.join(OUTPUT_DIR, "rank_cdf_" + method + ".pdf")

    plt.savefig(out_png)
    plt.savefig(out_pdf)
    plt.close()

    # ================= RECALL@K PLOT =================
    plt.figure()

    for dataset_name in dataset_results:
        plt.plot(K_VALUES, dataset_results[dataset_name]["recall"], marker='o', label=dataset_name)

    plt.xlabel("K")
    plt.ylabel("Recall@K")
    plt.title("Recall@K - " + method)
    plt.legend()

    out_png = os.path.join(OUTPUT_DIR, "recall_" + method + ".png")
    out_pdf = os.path.join(OUTPUT_DIR, "recall_" + method + ".pdf")

    plt.savefig(out_png)
    plt.savefig(out_pdf)
    plt.close()

print("\nAll multi-method overlay plots saved in:", OUTPUT_DIR)

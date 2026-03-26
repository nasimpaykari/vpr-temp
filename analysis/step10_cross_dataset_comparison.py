# -*- coding: utf-8 -*-
"""
Cross-dataset VPR comparison:
- corridor (base case)
- MY_DATASET (your dataset)

Outputs:
- Rank CDF comparison per dataset
- Recall@K comparison per dataset

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
    "corridor": os.path.abspath(os.path.join(BASE_DIR, "../precomputed_matches/corridor")),
    "MY_DATASET": os.path.abspath(os.path.join(BASE_DIR, "../precomputed_matches/MY_DATASET_ALL"))
}

GT_PATHS = {
    "corridor": os.path.abspath(os.path.join(BASE_DIR, "../datasets/corridor/ground_truth_new.npy")),
    "MY_DATASET": os.path.abspath(os.path.join(BASE_DIR, "../datasets/MY_DATASET/ground_truth_new.npy"))
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

    return data[0], data[3]  # queries, similarity


# ================= RANK COMPUTATION =================
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


# ================= PROCESS EACH DATASET =================
for dataset_name in DATASETS.keys():

    print("\n==============================")
    print("Processing dataset:", dataset_name)
    print("==============================\n")

    dataset_path = DATASETS[dataset_name]
    gt_dict = load_gt(GT_PATHS[dataset_name])

    ranks_dict = {}
    recall_dict = {}

    # ---- Load all methods ----
    for method in METHODS:
        try:
            queries, sim_matrix = load_method(dataset_path, method)

            ranks = compute_ranks(gt_dict, queries, sim_matrix)
            ranks_dict[method] = ranks

            recall_dict[method] = []
            for K in K_VALUES:
                r = compute_recall_at_k(gt_dict, queries, sim_matrix, K)
                recall_dict[method].append(r)

            print("Loaded:", method)

        except Exception as e:
            print("Skipping method:", method, "Error:", str(e))

    # ================= PLOT RANK CDF =================
    plt.figure()

    for method in ranks_dict:
        ranks = ranks_dict[method]
        sorted_ranks = np.sort(ranks)
        cdf = np.arange(1, len(sorted_ranks) + 1) / float(len(sorted_ranks))
        plt.plot(sorted_ranks, cdf, label=method)

    plt.xlabel("Rank")
    plt.ylabel("Cumulative Probability")
    plt.title("Rank CDF - " + dataset_name)
    plt.legend()

    out_png = os.path.join(OUTPUT_DIR, "rank_cdf_" + dataset_name + ".png")
    out_pdf = os.path.join(OUTPUT_DIR, "rank_cdf_" + dataset_name + ".pdf")

    plt.savefig(out_png)
    plt.savefig(out_pdf)
    plt.close()

    # ================= PLOT RECALL@K =================
    plt.figure()

    for method in recall_dict:
        plt.plot(K_VALUES, recall_dict[method], marker='o', label=method)

    plt.xlabel("K")
    plt.ylabel("Recall@K")
    plt.title("Recall@K Curve - " + dataset_name)
    plt.legend()

    out_png = os.path.join(OUTPUT_DIR, "recall_curve_" + dataset_name + ".png")
    out_pdf = os.path.join(OUTPUT_DIR, "recall_curve_" + dataset_name + ".pdf")

    plt.savefig(out_png)
    plt.savefig(out_pdf)
    plt.close()

print("\nAll plots saved in:", OUTPUT_DIR)

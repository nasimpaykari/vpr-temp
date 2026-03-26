# -*- coding: utf-8 -*-
"""
Leaderboard table for VPR methods across datasets.

Computes:
- Recall@K (K = 1, 5, 10, 20)

Outputs:
- CSV table
- Printed table
"""

from __future__ import print_function
import os
import numpy as np
import csv

# ================= CONFIG =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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

OUTPUT_CSV = os.path.join(BASE_DIR, "results_plots", "leaderboard.csv")

if not os.path.exists(os.path.dirname(OUTPUT_CSV)):
    os.makedirs(os.path.dirname(OUTPUT_CSV))


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


# ================= BUILD TABLE =================
rows = []

for dataset_name in DATASETS:

    print("\nProcessing dataset:", dataset_name)

    gt_dict = load_gt(DATASETS[dataset_name]["gt"])

    for method in METHODS:
        try:
            queries, sim_matrix = load_method(DATASETS[dataset_name]["path"], method)

            recalls = []
            for K in K_VALUES:
                r = compute_recall_at_k(gt_dict, queries, sim_matrix, K)
                recalls.append(r)

            row = [
                method,
                dataset_name
            ] + recalls

            rows.append(row)

            print("Done:", method)

        except Exception as e:
            print("Skipping:", method, "Error:", str(e))


# ================= PRINT TABLE =================
header = ["Method", "Dataset"] + ["Recall@{}".format(k) for k in K_VALUES]

print("\n\n===== LEADERBOARD =====\n")
print("\t".join(header))

for row in rows:
    print("\t".join([str(x) for x in row]))


# ================= SAVE CSV =================
with open(OUTPUT_CSV, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for row in rows:
        writer.writerow(row)

print("\nLeaderboard saved to:", OUTPUT_CSV)

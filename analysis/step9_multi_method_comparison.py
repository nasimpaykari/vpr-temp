# -*- coding: utf-8 -*-
"""
Multi-method comparison for VPR
- Rank CDF comparison
- Recall@K comparison
Python 2.7 compatible
"""

from __future__ import print_function
import os
import numpy as np
import matplotlib.pyplot as plt

# ================= CONFIG =================
DATASET_NAME = "corridor"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PRECOMPUTED_BASE = os.path.abspath(
    os.path.join(BASE_DIR, "../precomputed_matches", DATASET_NAME)
)

GT_PATH = os.path.abspath(
    os.path.join(BASE_DIR, "../datasets", "corridor", "ground_truth_new.npy")
)

OUTPUT_DIR = os.path.join(BASE_DIR, "results_plots")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

METHODS = [
    "AlexNet_VPR",
    "AMOSNet",
    "ap-gem-r101",
    "CALC",
    "CoHOG",
    "denseVLAD",
    "HOG",
    "HybridNet",
    "NetVLAD",
    "RegionVLAD"
]

K_VALUES = [1, 5, 10, 20]


# ================= LOAD GT =================
gt_data = np.load(GT_PATH, allow_pickle=True)

gt_dict = {}
for item in gt_data:
    q_idx = int(item[0])
    refs = [int(r) for r in item[1]]
    gt_dict[q_idx] = refs


# ================= FUNCTIONS =================
def load_method(method_name):
    path = os.path.join(
        PRECOMPUTED_BASE,
        method_name,
        "precomputed_data_corrected.npy"
    )
    data = np.load(path, allow_pickle=True)

    return data[0], data[3]  # queries, similarity


def compute_ranks(queries, sim_matrix):
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


def compute_recall_at_k(ranks_dict, queries_dict, sim_dict, K):
    results = {}

    for method in METHODS:
        queries = queries_dict[method]
        sim_matrix = sim_dict[method]

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

        recall = float(correct) / total if total > 0 else 0
        results[method] = recall

    return results


# ================= LOAD ALL METHODS =================
queries_dict = {}
sim_dict = {}
ranks_dict = {}

for method in METHODS:
    print("Loading:", method)

    queries, sim_matrix = load_method(method)

    queries_dict[method] = queries
    sim_dict[method] = sim_matrix

    ranks_dict[method] = compute_ranks(queries, sim_matrix)

# ================= PLOT RANK CDF =================
plt.figure()

for method in METHODS:
    ranks = ranks_dict[method]
    sorted_ranks = np.sort(ranks)
    cdf = np.arange(1, len(sorted_ranks) + 1) / float(len(sorted_ranks))

    plt.plot(sorted_ranks, cdf, label=method)

plt.xlabel("Rank")
plt.ylabel("Cumulative Probability")
plt.title("Rank CDF Comparison")
plt.legend()

cdf_png = os.path.join(OUTPUT_DIR, "rank_cdf_comparison.png")
cdf_pdf = os.path.join(OUTPUT_DIR, "rank_cdf_comparison.pdf")

plt.savefig(cdf_png)
plt.savefig(cdf_pdf)
plt.close()

# ================= RECALL@K =================
recall_results = {}

for K in K_VALUES:
    print("Computing Recall@{}".format(K))

    plt.figure()

    for method in METHODS:
        queries = queries_dict[method]
        sim_matrix = sim_dict[method]

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

        recall = float(correct) / total if total > 0 else 0

        if method not in recall_results:
            recall_results[method] = []

        recall_results[method].append(recall)

    # Plot per K (bar or line)
    methods = METHODS
    values = [recall_results[m][-1] for m in methods]

    plt.bar(range(len(methods)), values)
    plt.xticks(range(len(methods)), methods, rotation=45)
    plt.ylabel("Recall@{}".format(K))
    plt.title("Recall@{} Comparison".format(K))

    out_png = os.path.join(OUTPUT_DIR, "recall_at_{}.png".format(K))
    out_pdf = os.path.join(OUTPUT_DIR, "recall_at_{}.pdf".format(K))

    plt.tight_layout()
    plt.savefig(out_png)
    plt.savefig(out_pdf)
    plt.close()

# ================= PRINT SUMMARY =================
print("\n===== Recall Summary =====")
for method in METHODS:
    print(method, ":", recall_results[method])

print("\nPlots saved in:", OUTPUT_DIR)

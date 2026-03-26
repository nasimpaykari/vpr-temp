# -*- coding: utf-8 -*-
"""
Rank analysis + histogram + CDF for VPR
Python 2.7 compatible
"""

from __future__ import print_function
import os
import numpy as np
import matplotlib.pyplot as plt

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

OUTPUT_DIR = os.path.join(BASE_DIR, "results_plots")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ================= LOAD GROUND TRUTH =================
gt_data = np.load(GT_PATH, allow_pickle=True)

gt_dict = {}
for item in gt_data:
    q_idx = int(item[0])
    refs = [int(r) for r in item[1]]
    gt_dict[q_idx] = refs

# ================= LOAD METHOD =================
method_path = os.path.join(
    PRECOMPUTED_BASE,
    METHOD_NAME,
    "precomputed_data_corrected.npy"
)

data = np.load(method_path, allow_pickle=True)

queries = data[0]
similarity_matrix = data[3]

# ================= COMPUTE RANKS =================
ranks = []

for i in range(len(queries)):
    q_idx = int(queries[i])

    if q_idx not in gt_dict:
        continue

    sim_row = similarity_matrix[i]

    # Sort by descending similarity
    sorted_indices = np.argsort(sim_row)[::-1]

    gt_refs = gt_dict[q_idx]

    rank = None
    for r, idx in enumerate(sorted_indices):
        if idx in gt_refs:
            rank = r + 1
            break

    if rank is not None:
        ranks.append(rank)

ranks = np.array(ranks)

# ================= STATISTICS =================
print("Total queries evaluated:", len(ranks))
print("Mean Rank:", np.mean(ranks))
print("Median Rank:", np.median(ranks))

# ================= HISTOGRAM =================
plt.figure()
plt.hist(ranks, bins=30)
plt.xlabel("Rank")
plt.ylabel("Frequency")
plt.title("Rank Histogram - " + METHOD_NAME)

hist_png = os.path.join(OUTPUT_DIR, "rank_histogram.png")
hist_pdf = os.path.join(OUTPUT_DIR, "rank_histogram.pdf")

plt.savefig(hist_png)
plt.savefig(hist_pdf)
plt.close()

# ================= CDF =================
sorted_ranks = np.sort(ranks)
cdf = np.arange(1, len(sorted_ranks) + 1) / float(len(sorted_ranks))

plt.figure()
plt.plot(sorted_ranks, cdf)
plt.xlabel("Rank")
plt.ylabel("Cumulative Probability")
plt.title("Rank CDF - " + METHOD_NAME)

cdf_png = os.path.join(OUTPUT_DIR, "rank_cdf.png")
cdf_pdf = os.path.join(OUTPUT_DIR, "rank_cdf.pdf")

plt.savefig(cdf_png)
plt.savefig(cdf_pdf)
plt.close()

print("\nSaved plots in:", OUTPUT_DIR)

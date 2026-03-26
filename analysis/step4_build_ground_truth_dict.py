# -*- coding: utf-8 -*-
"""
STEP 4: Build ground truth dictionary
Python 2.7 compatible
"""

from __future__ import print_function
import os
import numpy as np

# ================= CONFIG =================
DATASET_NAME = "MY_DATASET"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

GT_PATH = os.path.abspath(
    os.path.join(BASE_DIR, "../datasets", DATASET_NAME, "ground_truth_new.npy")
)

# ================= LOAD =================
def load_ground_truth(gt_path):
    data = np.load(gt_path, allow_pickle=True)

    ground_truth_dict = {}

    print("\nBuilding ground truth dictionary...\n")

    for item in data:
        query_idx = int(item[0])
        ref_indices = item[1]

        # Convert to Python list of ints
        refs = [int(r) for r in ref_indices]

        ground_truth_dict[query_idx] = refs

    print("Total queries:", len(ground_truth_dict))

    # Show samples
    print("\nSample entries:")
    for i in range(5):
        if i in ground_truth_dict:
            print("Query {} -> Refs {}".format(i, ground_truth_dict[i]))

    return ground_truth_dict


# ================= MAIN =================
if __name__ == "__main__":
    gt_dict = load_ground_truth(GT_PATH)

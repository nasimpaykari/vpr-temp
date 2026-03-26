# -*- coding: utf-8 -*-
"""
STEP 3: Inspect ground truth structure
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
def load_and_inspect_gt(gt_path):
    print("\n==========================================")
    print("Loading Ground Truth from:")
    print(gt_path)
    print("==========================================\n")

    if not os.path.exists(gt_path):
        print("ERROR: Ground truth file not found!")
        return

    data = np.load(gt_path, allow_pickle=True)

    # ===== BASIC INFO =====
    print("TYPE:", type(data))

    try:
        print("SHAPE:", data.shape)
    except:
        print("NO SHAPE")

    try:
        print("LEN:", len(data))
    except:
        print("NO LEN")

    # ===== FIRST ELEMENT =====
    print("\n--- FIRST ELEMENT ---")
    try:
        print("TYPE:", type(data[0]))
        print("CONTENT:", data[0])
    except:
        print("Cannot access data[0]")

    # ===== SAMPLE ENTRIES =====
    print("\n--- SAMPLE ENTRIES ---")
    try:
        for i in range(min(10, len(data))):
            print("Entry {}: {}".format(i, data[i]))
    except:
        print("Cannot iterate")

    # ===== STRUCTURE CHECK =====
    print("\n--- STRUCTURE ANALYSIS ---")
    try:
        sample = data[0]
        print("Entry format type:", type(sample))

        if isinstance(sample, (list, tuple, np.ndarray)):
            print("Length of entry:", len(sample))

            if len(sample) > 1:
                print("Query index:", sample[0])
                print("Reference indices:", sample[1])
    except:
        print("Structure unclear")

    print("\n==========================================\n")

    return data


# ================= MAIN =================
if __name__ == "__main__":
    gt = load_and_inspect_gt(GT_PATH)

    if gt is not None:
        print("Ground truth loaded successfully!")

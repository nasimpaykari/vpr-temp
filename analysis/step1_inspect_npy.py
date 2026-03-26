# -*- coding: utf-8 -*-
"""
STEP 1: Inspect .npy structure for ONE VPR method
Python 2.7 compatible
"""

from __future__ import print_function
import os
import numpy as np

# ================= CONFIG =================
DATASET_NAME = "MY_DATASET"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PRECOMPUTED_BASE = os.path.abspath(
    os.path.join(BASE_DIR, "../precomputed_matches", DATASET_NAME + "_ALL")
)

METHOD_NAME = "NetVLAD"   # <-- change if needed

# ================= LOAD FUNCTION =================
def load_and_inspect(method_name):
    npy_path = os.path.join(
        PRECOMPUTED_BASE,
        method_name,
        "precomputed_data_corrected.npy"
    )

    print("\n==========================================")
    print("Loading method:", method_name)
    print("Path:", npy_path)
    print("==========================================\n")

    if not os.path.exists(npy_path):
        print("ERROR: File does not exist!")
        return

    data = np.load(npy_path, allow_pickle=True)

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
        print("CONTENT (truncated):", str(data[0])[:500])
    except:
        print("Cannot access data[0]")

    # ===== STRUCTURE OVERVIEW =====
    print("\n--- STRUCTURE OVERVIEW ---")
    try:
        for i in range(min(5, len(data))):
            try:
                print("Index {} type: {}".format(i, type(data[i])))
            except:
                print("Index {}: ERROR".format(i))
    except:
        print("Cannot iterate over data")

    # ===== SHAPES (VERY IMPORTANT) =====
    print("\n--- SHAPES ---")
    try:
        for i in range(min(5, len(data))):
            try:
                print("data[{}] shape: {}".format(i, np.shape(data[i])))
            except:
                print("data[{}]: no shape".format(i))
    except:
        print("Cannot print shapes")

    # ===== CHECK IF DICT =====
    print("\n--- CHECK DICT ---")
    try:
        if isinstance(data, dict):
            print("Top-level is DICT, keys:", data.keys())
        elif isinstance(data[0], dict):
            print("data[0] is DICT, keys:", data[0].keys())
    except:
        print("Not a dict structure")

    print("\n==========================================\n")


# ================= MAIN =================
if __name__ == "__main__":
    load_and_inspect(METHOD_NAME)

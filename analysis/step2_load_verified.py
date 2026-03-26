# -*- coding: utf-8 -*-
"""
STEP 2: Proper loader for VPR .npy data
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

# ================= LOAD FUNCTION =================
def load_method_data(method_name):
    npy_path = os.path.join(
        PRECOMPUTED_BASE,
        method_name,
        "precomputed_data_corrected.npy"
    )

    print("\nLoading:", npy_path)

    if not os.path.exists(npy_path):
        print("ERROR: file not found")
        return None

    data = np.load(npy_path, allow_pickle=True)

    # ===== BASIC VALIDATION =====
    if len(data) < 4:
        print("ERROR: Unexpected data format")
        return None

    query_indices = data[0]
    predictions = data[1]
    scores = data[2]
    similarity_matrix = data[3]

    # ===== PRINT SUMMARY =====
    print("\n--- DATA SUMMARY ---")
    print("Queries:", np.shape(query_indices))
    print("Predictions:", np.shape(predictions))
    print("Scores:", np.shape(scores))
    print("Similarity matrix:", np.shape(similarity_matrix))

    # ===== SANITY CHECK =====
    print("\n--- SANITY CHECK ---")
    print("First 5 queries:", query_indices[:5])
    print("First 5 predictions:", predictions[:5])
    print("First 5 scores:", scores[:5])

    print("\nSimilarity row 0 (first 10 values):")
    print(similarity_matrix[0][:10])

    return {
        'query_indices': query_indices,
        'predictions': predictions,
        'scores': scores,
        'similarity_matrix': similarity_matrix
    }

# ================= MAIN =================
if __name__ == "__main__":
    data = load_method_data(METHOD_NAME)

    if data is not None:
        print("\nLoaded successfully!")

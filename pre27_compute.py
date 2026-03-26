# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import sys
import shutil
import random
import json
import numpy as np

# ================= CONFIG =================
dataset_main   = "CLR_dataset"
output_root    = "MY_DATASET"

dataset_ref    = os.path.join(output_root, "ref")
dataset_query  = os.path.join(output_root, "query")

gt_grouped_file  = os.path.join(output_root, "ground_truth_new.npy")
gt_pairwise_file = os.path.join(output_root, "ground_truth_pairwise.npy")
metadata_file    = os.path.join(output_root, "processed_images.json")

# =============== SETUP ====================
for d in [dataset_ref, dataset_query]:
    if not os.path.exists(d):
        os.makedirs(d)

# ============ LOAD METADATA ===============
if os.path.exists(metadata_file):
    with open(metadata_file, "r") as f:
        metadata = json.load(f)
else:
    metadata = {
        "processed_images": [],
        "next_query_idx": 0,
        "next_ref_idx": 0
    }

processed_images = set(metadata["processed_images"])
query_index = int(metadata["next_query_idx"])
ref_index   = int(metadata["next_ref_idx"])

# ============ LOAD IMAGES ==================
images = [
    f for f in os.listdir(dataset_main)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
]

new_images = [img for img in images if img not in processed_images]

if not new_images:
    print("No new images to process. Exiting.")
    sys.exit(0)

# ============ GROUP BY PLACE ===============
places = {}
for img in new_images:
    name = os.path.splitext(img)[0]
    if "_" in name:
        place = name.rsplit("_", 1)[0]
    else:
        place = name
    places.setdefault(place, []).append(img)

print("Found %d new places/groups to process" % len(places))

# ============ LOAD EXISTING GTs ============
if os.path.exists(gt_pairwise_file):
    gt_pairwise = np.load(gt_pairwise_file, allow_pickle=True).tolist()
else:
    gt_pairwise = []

if os.path.exists(gt_grouped_file):
    gt_grouped = np.load(gt_grouped_file, allow_pickle=True).tolist()
else:
    gt_grouped = []

# ============ PROCESS DATA =================
for place, imgs in places.items():
    if len(imgs) < 2:
        continue

    q_img = random.choice(imgs)
    imgs.remove(q_img)

    q_name = "%04d.jpg" % query_index
    shutil.copy(
        os.path.join(dataset_main, q_img),
        os.path.join(dataset_query, q_name)
    )

    ref_list = []

    for img in imgs:
        r_name = "%04d.jpg" % ref_index
        shutil.copy(
            os.path.join(dataset_main, img),
            os.path.join(dataset_ref, r_name)
        )

        gt_pairwise.append([query_index, ref_index])
        ref_list.append(ref_index)
        ref_index += 1

    # ⚠️ THIS IS THE FORMAT VPR-BENCH EXPECTS
    gt_grouped.append([query_index, ref_list])

    query_index += 1
    processed_images.add(q_img)
    for i in imgs:
        processed_images.add(i)

# ============ SAVE GTs =====================

# Pairwise → safe numeric array
np.save(
    gt_pairwise_file,
    np.asarray(gt_pairwise, dtype=np.int32)
)

# Grouped → OBJECT ARRAY (REQUIRED by VPR-Bench)
np.save(
    gt_grouped_file,
    np.array(gt_grouped, dtype=object)
)

# ============ SAVE METADATA ================
metadata = {
    "processed_images": list(processed_images),
    "next_query_idx": query_index,
    "next_ref_idx": ref_index
}

with open(metadata_file, "w") as f:
    json.dump(metadata, f, indent=2)

# ============ SUMMARY ======================
print("Incremental dataset update complete")
print("Total queries:", query_index)
print("Total references:", ref_index)
print("Grouped GT saved to:", gt_grouped_file)
print("Pairwise GT saved to:", gt_pairwise_file)
print("Metadata saved to:", metadata_file)

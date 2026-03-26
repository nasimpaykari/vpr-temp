# -*- coding: utf-8 -*-

import os
import shutil
import random
import numpy as np
import json

# === CONFIG ===
dataset_main   = "CLR_dataset"
output_root    = "MY_DATASET"
dataset_ref    = os.path.join(output_root, "ref")
dataset_query  = os.path.join(output_root, "query")
gt_grouped_file = os.path.join(output_root, "ground_truth_new.npy")
gt_pairwise_file = os.path.join(output_root, "ground_truth_pairwise.npy")
metadata_file = os.path.join(output_root, "processed_images.json")  # keep track of processed files

# Reset folders only if they don't exist
for folder in [dataset_ref, dataset_query]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Load already processed metadata
if os.path.exists(metadata_file):
    with open(metadata_file, "r") as f:
        metadata = json.load(f)
else:
    metadata = {
        "processed_images": [],   # list of original filenames processed
        "next_query_idx": 0,
        "next_ref_idx": 0
    }

processed_images = set(metadata["processed_images"])
query_index = metadata["next_query_idx"]
ref_index = metadata["next_ref_idx"]

# Group images by place
images = [f for f in os.listdir(dataset_main) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
new_images = [img for img in images if img not in processed_images]

if not new_images:
    print("No new images to process. Exiting.")
    exit()

places = {}
for img in new_images:
    name, _ = os.path.splitext(img)
    if '_' in name:
        place_name = name.rsplit('_', 1)[0]
    else:
        place_name = name
    places.setdefault(place_name, []).append(img)

print("Found %d new places/groups to process" % len(places))

# Load existing GTs or initialize
if os.path.exists(gt_grouped_file):
    gt_grouped = list(np.load(gt_grouped_file, allow_pickle=True))
else:
    gt_grouped = []

if os.path.exists(gt_pairwise_file):
    gt_pairwise = list(np.load(gt_pairwise_file, allow_pickle=True))
else:
    gt_pairwise = []

# Process new images
for place, imgs in places.items():
    if len(imgs) < 2:
        continue  # skip single-image places

    # Pick one random query
    q_img = random.choice(imgs)
    imgs.remove(q_img)

    # Copy query
    q_filename = "%04d.jpg" % query_index
    shutil.copy(os.path.join(dataset_main, q_img), os.path.join(dataset_query, q_filename))

    ref_indices_for_this_query = []
    for img in imgs:
        ref_filename = "%04d.jpg" % ref_index
        shutil.copy(os.path.join(dataset_main, img), os.path.join(dataset_ref, ref_filename))
        ref_indices_for_this_query.append(ref_index)
        gt_pairwise.append([query_index, ref_index])
        ref_index += 1

    gt_grouped.append([query_index, ref_indices_for_this_query])
    query_index += 1
    processed_images.add(q_img)
    processed_images.update(imgs)

# Save GTs
np.save(gt_grouped_file, np.array(gt_grouped, dtype=object))
np.save(gt_pairwise_file, np.array(gt_pairwise, dtype=object))

# Update metadata
metadata = {
    "processed_images": list(processed_images),
    "next_query_idx": query_index,
    "next_ref_idx": ref_index
}
with open(metadata_file, "w") as f:
    json.dump(metadata, f, indent=2)

# Summary
print("Incremental dataset update complete")
print("Total queries: %d" % query_index)
print("Total references: %d" % ref_index)
print("Grouped GT saved to:", gt_grouped_file)
print("Pairwise GT saved to:", gt_pairwise_file)
print("Processed metadata saved to:", metadata_file)


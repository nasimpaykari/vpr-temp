#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Comprehensive VPR Analysis with Proper TP/FP/FN/TN Calculations
Python 2.7 Compatible - Fixed Unicode Error
"""

from __future__ import print_function, division
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ================= CONFIGURATION =================
VPR_BENCH_ROOT = "/home/labaccount/CLR/VPR-Bench"
DATASET_NAME = "MY_DATASET"
RESULTS_DIR = os.path.join(VPR_BENCH_ROOT, "analysis_results", DATASET_NAME)

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

METHODS = [
    'AlexNet_VPR',
    'AMOSNet',
    'CALC',
    'CoHOG',
    'HOG',
    'HybridNet',
    'NetVLAD',
    'RegionVLAD'
]

GROUND_TRUTH_PATH = os.path.join(VPR_BENCH_ROOT, "datasets", DATASET_NAME, "ground_truth_new.npy")
PRECOMPUTED_BASE = os.path.join(VPR_BENCH_ROOT, "precomputed_matches", DATASET_NAME+"_ALL")

# ============= LOAD GROUND TRUTH =============
def load_ground_truth(gt_path):
    """Load ground truth in VPR-Bench format [query_idx, [ref_indices]]"""
    gt_data = np.load(gt_path, allow_pickle=True)
    
    ground_truth_dict = {}
    place_to_queries = {}  # For TN calculation
    ref_to_place = {}  # Map each reference to its place
    
    for item in gt_data:
        query_idx = int(item[0])
        ref_indices = [int(x) for x in item[1]]
        ground_truth_dict[query_idx] = ref_indices
        
        # Use first reference as place identifier
        place_id = ref_indices[0]
        if place_id not in place_to_queries:
            place_to_queries[place_id] = []
        place_to_queries[place_id].append(query_idx)
        
        # Map each reference to this place
        for ref in ref_indices:
            ref_to_place[ref] = place_id
    
    return ground_truth_dict, place_to_queries, ref_to_place

# ============= LOAD METHOD DATA =============
def load_method_data(method_name):
    """Load precomputed data for a specific method"""
    npy_path = os.path.join(PRECOMPUTED_BASE, method_name, 'precomputed_data_corrected.npy')
    
    if not os.path.exists(npy_path):
        print("Warning: Data not found for {}".format(method_name))
        return None
    
    try:
        data = np.load(npy_path, allow_pickle=True)
        
        if len(data) >= 4:
            return {
                'query_indices': data[0],
                'predictions': data[1],
                'scores': data[2],
                'similarity_matrix': data[3],
                'name': method_name
            }
        else:
            print("Warning: Unexpected data structure for {}".format(method_name))
            return None
    except Exception as e:
        print("Error loading {}: {}".format(method_name, e))
        return None

# ============= PROPER TP/FP/FN/TN CALCULATION =============

def calculate_confusion_matrix(predictions, ground_truth_dict, place_to_queries, ref_to_place, query_indices):
    """
    Calculate TP, FP, FN, TN properly for VPR context.
    
    Definitions:
    - TP: Query correctly matched to a reference from its place
    - FP: Query matched to a reference from a DIFFERENT place
    - FN: Query that SHOULD have matched to its place but didn't (same as FP in top-1)
    - TN: For each place, count queries from OTHER places that correctly did NOT match to it
    """
    
    tp = 0
    fp = 0
    incorrect_matches = 0  # Same as FP for clarity
    
    correct_match_details = []
    incorrect_match_details = []
    
    # First pass: Calculate TP, FP, and collect details
    for i, query_idx in enumerate(query_indices):
        if query_idx not in ground_truth_dict:
            continue
            
        true_refs = ground_truth_dict[query_idx]  # References from the correct place
        true_place = true_refs[0]  # Use first ref as place identifier
        predicted_ref = predictions[i]
        
        # Find which place the predicted reference belongs to
        predicted_place = ref_to_place.get(predicted_ref, None)
        
        # Determine TP or FP
        if true_place == predicted_place:
            tp += 1
            correct_match_details.append({
                'query': int(query_idx),
                'true_place': int(true_place),
                'predicted_ref': int(predicted_ref),
                'predicted_place': int(predicted_place) if predicted_place is not None else -1,
                'score': float(predictions[i])  # Note: this is the prediction, not score
            })
        else:
            fp += 1
            incorrect_matches += 1
            incorrect_match_details.append({
                'query': int(query_idx),
                'true_place': int(true_place),
                'true_refs': [int(r) for r in true_refs],
                'predicted_ref': int(predicted_ref),
                'predicted_place': int(predicted_place) if predicted_place is not None else -1,
                'score': float(predictions[i])
            })
    
    # Calculate FN (same as FP in top-1 evaluation)
    fn = fp
    
    # Calculate TN
    tn = 0
    all_places = list(place_to_queries.keys())
    
    # For each place, count correct non-matches
    for place in all_places:
        # Get all queries NOT from this place
        queries_from_other_places = []
        for q_idx in query_indices:
            if q_idx in ground_truth_dict:
                true_place = ground_truth_dict[q_idx][0]
                if true_place != place:
                    queries_from_other_places.append(q_idx)
        
        # For each such query, check if its prediction belongs to this place
        for q_idx in queries_from_other_places:
            q_index_in_list = list(query_indices).index(q_idx) if q_idx in query_indices else -1
            if q_index_in_list >= 0:
                pred_ref = predictions[q_index_in_list]
                pred_place = ref_to_place.get(pred_ref, None)
                
                if pred_place != place:
                    tn += 1  # Correctly did NOT match to this place
    
    # Normalize TN (average per place)
    if len(all_places) > 0:
        tn = tn // len(all_places)
    
    total = len(query_indices)
    
    # Calculate metrics
    accuracy = float(tp) / total if total > 0 else 0
    precision = float(tp) / (tp + fp) if (tp + fp) > 0 else 0
    recall = float(tp) / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # For specificity, we need total negatives = (total_database_images - 1) * total_queries? 
    # This is complex, so let's simplify
    specificity = float(tn) / (tn + fp) if (tn + fp) > 0 else 0
    
    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn,
        'total_queries': total,
        'correct_matches': tp,
        'incorrect_matches': fp,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'specificity': specificity,
        'error_rate': float(fp) / total if total > 0 else 0,
        'correct_details': correct_match_details[:10],  # First 10 only
        'incorrect_details': incorrect_match_details[:10]  # First 10 only
    }

def calculate_recall_at_k(similarity_matrix, ground_truth_dict, query_indices, max_k=25):
    """Calculate Recall@K correctly"""
    recall_at_k = []
    
    for k in range(1, max_k + 1):
        correct_at_k = 0
        
        for i, query_idx in enumerate(query_indices):
            if query_idx not in ground_truth_dict:
                continue
                
            true_refs = ground_truth_dict[query_idx]
            top_k_indices = np.argsort(similarity_matrix[i])[:k]
            
            if any(ref in top_k_indices for ref in true_refs):
                correct_at_k += 1
        
        recall = float(correct_at_k) / len(query_indices) if len(query_indices) > 0 else 0
        recall_at_k.append(recall)
    
    return recall_at_k

def calculate_precision_recall_curve(similarity_matrix, ground_truth_dict, query_indices):
    """Calculate PR curve for VPR"""
    all_scores = []
    all_labels = []
    
    for i, query_idx in enumerate(query_indices):
        if query_idx not in ground_truth_dict:
            continue
            
        true_refs = ground_truth_dict[query_idx]
        
        for db_idx in range(similarity_matrix.shape[1]):
            all_scores.append(similarity_matrix[i, db_idx])
            all_labels.append(1 if db_idx in true_refs else 0)
    
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    sorted_indices = np.argsort(all_scores)
    sorted_labels = all_labels[sorted_indices]
    
    precisions = []
    recalls = []
    
    total_positives = float(np.sum(all_labels))
    
    for i in range(1, len(sorted_labels) + 1):
        tp = np.sum(sorted_labels[:i])
        fp = i - tp
        
        precision = float(tp) / i if i > 0 else 0
        recall = float(tp) / total_positives if total_positives > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
    
    # Calculate AP
    ap = 0.0
    for i in range(1, len(precisions)):
        ap += precisions[i] * (recalls[i] - recalls[i-1])
    
    return {
        'precisions': precisions,
        'recalls': recalls,
        'average_precision': ap
    }

# ============= PLOTTING FUNCTIONS =============
def plot_confusion_matrix_comparison(all_metrics, save_path):
    """Plot TP/FP/FN/TN comparison"""
    methods = all_metrics.keys()
    methods = sorted(methods, key=lambda x: all_metrics[x]['accuracy'], reverse=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    metrics_to_plot = [
        ('tp', 'True Positives (TP)', '#2E86AB'),
        ('fp', 'False Positives (FP)', '#A23B72'),
        ('fn', 'False Negatives (FN)', '#F18F01'),
        ('tn', 'True Negatives (TN)', '#73AB84')
    ]
    
    for idx, (metric, title, color) in enumerate(metrics_to_plot):
        ax = axes[idx // 2, idx % 2]
        values = [all_metrics[m][metric] for m in methods]
        
        bars = ax.bar(range(len(methods)), values, color=color, alpha=0.7)
        ax.set_xlabel('Method', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                   str(val), ha='center', va='bottom', fontsize=8)
    
    plt.suptitle('Confusion Matrix Components by Method', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'confusion_matrix_components.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_path, 'confusion_matrix_components.pdf'), bbox_inches='tight')
    plt.close()

def plot_metrics_comparison(all_metrics, save_path):
    """Plot accuracy, precision, recall, F1, specificity, and correct/incorrect matches"""
    methods = all_metrics.keys()
    methods = sorted(methods, key=lambda x: all_metrics[x]['f1_score'], reverse=True)
    
    # Plot 1: Performance metrics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # Subplot 1: Performance metrics
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'specificity']
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#73AB84', '#E15554']
    
    x = np.arange(len(methods))
    width = 0.15
    
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        values = [all_metrics[m][metric] for m in methods]
        offset = (i - 2) * width
        bars = ax1.bar(x + offset, values, width, color=color, alpha=0.8, label=metric.capitalize())
    
    ax1.set_xlabel('Method', fontsize=12)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title('Performance Metrics Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=45, ha='right', fontsize=10)
    ax1.legend(loc='upper right', fontsize=9, ncol=3)
    ax1.set_ylim([0, 1])
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Subplot 2: Correct vs Incorrect matches
    correct_vals = [all_metrics[m]['correct_matches'] for m in methods]
    incorrect_vals = [all_metrics[m]['incorrect_matches'] for m in methods]
    
    x2 = np.arange(len(methods))
    width2 = 0.35
    
    bars_correct = ax2.bar(x2 - width2/2, correct_vals, width2, label='Correct Matches', color='#2E86AB', alpha=0.8)
    bars_incorrect = ax2.bar(x2 + width2/2, incorrect_vals, width2, label='Incorrect Matches', color='#A23B72', alpha=0.8)
    
    # Add value labels
    for bars in [bars_correct, bars_incorrect]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 5,
                   str(int(height)), ha='center', va='bottom', fontsize=9)
    
    ax2.set_xlabel('Method', fontsize=12)
    ax2.set_ylabel('Number of Matches', fontsize=12)
    ax2.set_title('Correct vs Incorrect Matches', fontsize=14, fontweight='bold')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(methods, rotation=45, ha='right', fontsize=10)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'complete_metrics_comparison.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_path, 'complete_metrics_comparison.pdf'), bbox_inches='tight')
    plt.close()

def plot_recall_at_k(all_recall_data, save_path):
    """Plot Recall@K curves"""
    plt.figure(figsize=(12, 8))
    
    for method_name, recall_data in all_recall_data.items():
        k_values = range(1, len(recall_data) + 1)
        plt.plot(k_values, recall_data, 'o-', linewidth=2, markersize=4, label=method_name)
    
    plt.xlabel('K', fontsize=12)
    plt.ylabel('Recall', fontsize=12)
    plt.title('Recall@K Curves - All Methods', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right', fontsize=10, ncol=2)
    plt.xlim([0, 25])
    plt.ylim([0, 1])
    plt.xticks(range(0, 26, 5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'recall_at_k_curves.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_path, 'recall_at_k_curves.pdf'), bbox_inches='tight')
    plt.close()

def plot_precision_recall_curves(all_pr_data, save_path):
    """Plot Precision-Recall curves"""
    plt.figure(figsize=(12, 8))
    
    for method_name, pr_data in all_pr_data.items():
        plt.plot(pr_data['recalls'], pr_data['precisions'], 
                linewidth=2, label="{} (AP={:.4f})".format(method_name, pr_data['average_precision']))
    
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves - All Methods', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right', fontsize=10, ncol=2)
    plt.xlim([0, 1])
    plt.ylim([0, 0.02])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'precision_recall_curves.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_path, 'precision_recall_curves.pdf'), bbox_inches='tight')
    plt.close()

# ============= MAIN =============
def main():
    print("=" * 70)
    print("VPR-BENCH COMPREHENSIVE ANALYSIS WITH PROPER CONFUSION MATRIX")
    print("Dataset: {}".format(DATASET_NAME))
    print("Results will be saved to: {}".format(RESULTS_DIR))
    print("=" * 70)
    
    # Load ground truth
    print("\n[1] Loading ground truth...")
    ground_truth_dict, place_to_queries, ref_to_place = load_ground_truth(GROUND_TRUTH_PATH)
    print("    Found {} queries with ground truth".format(len(ground_truth_dict)))
    print("    Found {} distinct places".format(len(place_to_queries)))
    
    # Get total database size from first method
    total_db_images = 0
    first_method = load_method_data(METHODS[0])
    if first_method is not None:
        total_db_images = first_method['similarity_matrix'].shape[1]
        print("    Database size: {} images".format(total_db_images))
    
    # Storage
    all_metrics = {}
    all_recall_data = {}
    all_pr_data = {}
    all_detailed = {}
    
    # Analyze each method
    print("\n[2] Analyzing methods...")
    for method in METHODS:
        print("\n    Processing {}...".format(method))
        
        method_data = load_method_data(method)
        if method_data is None:
            continue
        
        # Get valid queries
        valid_queries = []
        valid_indices = []
        for i, q in enumerate(method_data['query_indices']):
            if q in ground_truth_dict:
                valid_queries.append(q)
                valid_indices.append(i)
        
        if len(valid_queries) == 0:
            print("    Warning: No valid queries for {}".format(method))
            continue
        
        valid_indices = np.array(valid_indices)
        
        # Calculate confusion matrix with proper TP/FP/FN/TN
        metrics = calculate_confusion_matrix(
            method_data['predictions'][valid_indices],
            ground_truth_dict,
            place_to_queries,
            ref_to_place,
            valid_queries
        )
        
        # Calculate Recall@K
        recall_at_k = calculate_recall_at_k(
            method_data['similarity_matrix'][valid_indices],
            ground_truth_dict,
            valid_queries,
            max_k=25
        )
        
        # Calculate PR curve
        pr_data = calculate_precision_recall_curve(
            method_data['similarity_matrix'][valid_indices],
            ground_truth_dict,
            valid_queries
        )
        
        # Store results
        all_metrics[method] = metrics
        all_recall_data[method] = recall_at_k
        all_pr_data[method] = pr_data
        
        all_detailed[method] = {
            'confusion_matrix': {
                'tp': metrics['tp'],
                'fp': metrics['fp'],
                'fn': metrics['fn'],
                'tn': metrics['tn']
            },
            'matches': {
                'correct': metrics['correct_matches'],
                'incorrect': metrics['incorrect_matches'],
                'total': metrics['total_queries']
            },
            'performance': {
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score'],
                'specificity': metrics['specificity']
            },
            'recall_at_k': recall_at_k,
            'average_precision': pr_data['average_precision'],
            'sample_correct_matches': metrics['correct_details'],
            'sample_incorrect_matches': metrics['incorrect_details']
        }
        
        print("    ✓ Correct: {}, Incorrect: {}, Total: {}".format(
            metrics['correct_matches'], metrics['incorrect_matches'], metrics['total_queries']))
        print("      TP: {}, FP: {}, FN: {}, TN: {}".format(
            metrics['tp'], metrics['fp'], metrics['fn'], metrics['tn']))
        print("      Acc: {:.4f}, Prec: {:.4f}, Rec: {:.4f}, F1: {:.4f}".format(
            metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1_score']))
    
    # Generate plots
    print("\n[3] Generating plots...")
    plot_confusion_matrix_comparison(all_metrics, RESULTS_DIR)
    plot_metrics_comparison(all_metrics, RESULTS_DIR)
    plot_recall_at_k(all_recall_data, RESULTS_DIR)
    plot_precision_recall_curves(all_pr_data, RESULTS_DIR)
    
    # Create summary (without Unicode characters)
    print("\n[4] Creating summary...")
    
    # Sort by F1 score
    sorted_methods = sorted(all_metrics.items(), key=lambda x: x[1]['f1_score'], reverse=True)
    
    summary = """
===============================================================================
VPR-BENCH COMPREHENSIVE ANALYSIS WITH PROPER CONFUSION MATRIX
Dataset: {}
Date: {}
Total Queries: {}
Total Places: {}
===============================================================================

METHOD RANKING (by F1 Score):
""".format(DATASET_NAME, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 
           len(ground_truth_dict), len(place_to_queries))
    
    for i, (method, m) in enumerate(sorted_methods, 1):
        summary += """
{}. {} 
   =========================================================================
   MATCH SUMMARY:
   --> Correct Matches: {} / {} ({:.2f}%)
   --> Incorrect Matches: {} / {} ({:.2f}%)
   
   CONFUSION MATRIX:
   +----------------+------------------+------------------+
   |                | Predicted Pos    | Predicted Neg    |
   +----------------+------------------+------------------+
   | Actual Pos     | TP = {:<4}       | FN = {:<4}       |
   | Actual Neg     | FP = {:<4}       | TN = {:<4}       |
   +----------------+------------------+------------------+
   
   PERFORMANCE METRICS:
   * Accuracy:  {:.4f}   | Precision: {:.4f}   | Recall: {:.4f}
   * F1-Score:  {:.4f}   | Specificity: {:.4f}
   * Error Rate: {:.4f}
   -------------------------------------------------------------------------
""".format(i, method, 
           m['correct_matches'], m['total_queries'], m['correct_matches']/m['total_queries']*100,
           m['incorrect_matches'], m['total_queries'], m['incorrect_matches']/m['total_queries']*100,
           m['tp'], m['fn'], m['fp'], m['tn'],
           m['accuracy'], m['precision'], m['recall'],
           m['f1_score'], m['specificity'],
           m['error_rate'])
    
    # Add Recall@K comparison
    summary += "\nRECALL@K COMPARISON:\n"
    summary += "{:<14} {:>8} {:>8} {:>8} {:>8} {:>8}\n".format(
        "Method", "R@1", "R@5", "R@10", "R@15", "R@20")
    summary += "-" * 70 + "\n"
    
    for method in sorted(all_recall_data.keys(), 
                        key=lambda x: all_recall_data[x][0] if len(all_recall_data[x]) > 0 else 0, 
                        reverse=True):
        recall = all_recall_data[method]
        r1 = recall[0] if len(recall) > 0 else 0
        r5 = recall[4] if len(recall) > 4 else 0
        r10 = recall[9] if len(recall) > 9 else 0
        r15 = recall[14] if len(recall) > 14 else 0
        r20 = recall[19] if len(recall) > 19 else 0
        summary += "{:<14} {:8.4f} {:8.4f} {:8.4f} {:8.4f} {:8.4f}\n".format(
            method, r1, r5, r10, r15, r20)
    
    # Add sample incorrect matches (first few)
    summary += "\n\nSAMPLE INCORRECT MATCHES (First 3 per method):\n"
    summary += "=" * 80 + "\n"
    
    for method, m in sorted_methods[:3]:  # Top 3 methods
        if 'incorrect_details' in m and m['incorrect_details']:
            summary += "\n{}:\n".format(method)
            for i, detail in enumerate(m['incorrect_details'][:3]):
                summary += "  {}. Query {} -> Predicted Ref {} (should be in place {})\n".format(
                    i+1, detail['query'], detail['predicted_ref'], detail['true_place'])
    
    print(summary)
    
    # Save text summary (ASCII only, no Unicode)
    with open(os.path.join(RESULTS_DIR, 'complete_analysis.txt'), 'w') as f:
        f.write(summary)
    
    # Save CSV with all metrics
    csv = ["Method,Correct,Incorrect,Total,Accuracy,TP,FP,FN,TN,Precision,Recall,F1,Specificity,Error_Rate,R@1,R@5,R@10,R@15,R@20"]
    for method in sorted(all_metrics.keys()):
        m = all_metrics[method]
        recall = all_recall_data[method]
        csv.append("{},{},{},{},{:.4f},{},{},{},{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}".format(
            method, m['correct_matches'], m['incorrect_matches'], m['total_queries'],
            m['accuracy'], m['tp'], m['fp'], m['fn'], m['tn'],
            m['precision'], m['recall'], m['f1_score'], m['specificity'], m['error_rate'],
            recall[0] if len(recall) > 0 else 0,
            recall[4] if len(recall) > 4 else 0,
            recall[9] if len(recall) > 9 else 0,
            recall[14] if len(recall) > 14 else 0,
            recall[19] if len(recall) > 19 else 0
        ))
    
    with open(os.path.join(RESULTS_DIR, 'complete_results.csv'), 'w') as f:
        f.write('\n'.join(csv))
    
    # Save detailed JSON
    with open(os.path.join(RESULTS_DIR, 'complete_results.json'), 'w') as f:
        json.dump(all_detailed, f, indent=2)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("Results saved to: {}".format(RESULTS_DIR))
    print("=" * 70)
    
    # List generated files
    print("\nGenerated files:")
    for f in sorted(os.listdir(RESULTS_DIR)):
        print("  - {}".format(f))

if __name__ == "__main__":
    main()

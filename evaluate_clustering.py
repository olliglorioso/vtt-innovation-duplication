import json
import itertools
import argparse
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, homogeneity_completeness_v_measure

def create_composite_key(item_dict: Dict, is_ground_truth: bool = False) -> str:
    """
    Create a composite key for matching items between ground truth and clustering results.
    Uses source_id, source_description, and target_id (if available).
    """
    if is_ground_truth:
        # Ground truth structure
        source_id = item_dict.get('source_id', '')
        source_desc = item_dict.get('source_description', '')
        target_id = item_dict.get('target_id', '')  # May not exist in ground truth
    else:
        # Clustering results structure  
        source_id = item_dict.get('source id', '')
        source_desc = item_dict.get('source description', '')
        target_id = item_dict.get('target id', '')
    
    # Create composite key - normalize whitespace and handle missing values
    source_id = str(source_id).strip() if source_id else ''
    source_desc = str(source_desc).strip() if source_desc else ''
    target_id = str(target_id).strip() if target_id else ''
    
    # Use a separator that's unlikely to appear in the data
    composite_key = f"{source_id}||{source_desc}||{target_id}"
    return composite_key

def load_ground_truth(file_path: str) -> Tuple[Set[Tuple[str, str]], Set[Tuple[str, str]]]:
    """
    Load ground truth data and extract positive and negative pairs.
    
    Returns:
        positive_pairs: Set of pairs that should be in the same cluster
        negative_pairs: Set of pairs that should be in different clusters
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    positive_pairs = set()
    negative_pairs = set()
    
    # Extract positive pairs (should match - same cluster)
    for match_group in data['match']:
        items = match_group['__value__']
        # Create composite keys for all items in this group
        item_keys = [create_composite_key(item, is_ground_truth=True) for item in items]
        
        # Create all possible pairs within this group
        for i in range(len(item_keys)):
            for j in range(i + 1, len(item_keys)):
                # Always put the lexicographically smaller key first for consistency
                pair = tuple(sorted([item_keys[i], item_keys[j]]))
                positive_pairs.add(pair)
    
    # Extract negative pairs (should be distinct - different clusters)
    for distinct_group in data['distinct']:
        items = distinct_group['__value__']
        # Create composite keys for all items in this group
        item_keys = [create_composite_key(item, is_ground_truth=True) for item in items]
        
        # Create all possible pairs within this group
        for i in range(len(item_keys)):
            for j in range(i + 1, len(item_keys)):
                # Always put the lexicographically smaller key first for consistency
                pair = tuple(sorted([item_keys[i], item_keys[j]]))
                negative_pairs.add(pair)
    
    return positive_pairs, negative_pairs

def load_clustering_results(file_path: str) -> Dict[str, int]:
    """
    Load clustering results and create a mapping from composite key to cluster ID.
    
    Returns:
        item_to_cluster: Dictionary mapping composite key to cluster number
    """
    with open(file_path, 'r') as f:
        clusters = json.load(f)
    
    item_to_cluster = {}
    
    for cluster_id, cluster_items in enumerate(clusters):
        for item in cluster_items:
            composite_key = create_composite_key(item, is_ground_truth=False)
            item_to_cluster[composite_key] = cluster_id
    
    return item_to_cluster

def generate_predicted_pairs(item_to_cluster: Dict[str, int]) -> Set[Tuple[str, str]]:
    """
    Generate all pairs that are predicted to be in the same cluster.
    """
    cluster_to_items = defaultdict(list)
    for item, cluster in item_to_cluster.items():
        cluster_to_items[cluster].append(item)
    
    predicted_pairs = set()
    for cluster_items in cluster_to_items.values():
        if len(cluster_items) > 1:
            # Generate all pairs within this cluster
            for i in range(len(cluster_items)):
                for j in range(i + 1, len(cluster_items)):
                    pair = tuple(sorted([cluster_items[i], cluster_items[j]]))
                    predicted_pairs.add(pair)
    
    return predicted_pairs

def calculate_pairwise_metrics(predicted_pairs: Set[Tuple[str, str]], 
                             positive_pairs: Set[Tuple[str, str]], 
                             negative_pairs: Set[Tuple[str, str]]) -> Dict[str, float]:
    """
    Calculate pairwise precision, recall, and F1 score.
    """
    # True positives: pairs that should be together and are predicted together
    tp = len(predicted_pairs.intersection(positive_pairs))
    
    # False positives: pairs that should be separate but are predicted together
    fp = len(predicted_pairs.intersection(negative_pairs))
    
    # False negatives: pairs that should be together but are predicted separate
    fn = len(positive_pairs - predicted_pairs)
    
    # True negatives: pairs that should be separate and are predicted separate
    tn = len(negative_pairs - predicted_pairs)
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'accuracy': accuracy,
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn,
        'true_negatives': tn
    }

def calculate_cluster_level_metrics(item_to_cluster: Dict[str, int],
                                  positive_pairs: Set[Tuple[str, str]],
                                  negative_pairs: Set[Tuple[str, str]]) -> Dict[str, float]:
    """
    Calculate cluster-level metrics using external evaluation measures.
    """
    # Create ground truth clustering based on positive pairs
    # Items in positive pairs should be in the same cluster
    ground_truth_clusters = {}
    cluster_id = 0
    
    # Use Union-Find to group items that should be together
    parent = {}
    
    def find(x):
        if x not in parent:
            parent[x] = x
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    # Union all positive pairs
    for item1, item2 in positive_pairs:
        union(item1, item2)
    
    # Create cluster mapping
    cluster_map = {}
    next_cluster_id = 0
    for item in set().union(*positive_pairs).union(*negative_pairs):
        root = find(item)
        if root not in cluster_map:
            cluster_map[root] = next_cluster_id
            next_cluster_id += 1
        ground_truth_clusters[item] = cluster_map[root]
    
    # Get common items between ground truth and predictions
    common_items = set(ground_truth_clusters.keys()).intersection(set(item_to_cluster.keys()))
    
    if len(common_items) < 2:
        return {'adjusted_rand_index': 0.0, 'normalized_mutual_info': 0.0, 
                'homogeneity': 0.0, 'completeness': 0.0, 'v_measure': 0.0}
    
    # Create label arrays for sklearn metrics
    true_labels = [ground_truth_clusters[item] for item in common_items]
    pred_labels = [item_to_cluster[item] for item in common_items]
    
    # Calculate metrics
    ari = adjusted_rand_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(true_labels, pred_labels)
    
    return {
        'adjusted_rand_index': ari,
        'normalized_mutual_info': nmi,
        'homogeneity': homogeneity,
        'completeness': completeness,
        'v_measure': v_measure
    }

def print_detailed_analysis(predicted_pairs: Set[Tuple[str, str]],
                          positive_pairs: Set[Tuple[str, str]],
                          negative_pairs: Set[Tuple[str, str]],
                          item_to_cluster: Dict[str, int]):
    """
    Print detailed analysis of the clustering results.
    """
    print("\n" + "="*80)
    print("DETAILED CLUSTERING ANALYSIS")
    print("="*80)
    
    print(f"\nGround Truth Statistics:")
    print(f"  Positive pairs (should be same cluster): {len(positive_pairs)}")
    print(f"  Negative pairs (should be different clusters): {len(negative_pairs)}")
    print(f"  Total ground truth pairs: {len(positive_pairs) + len(negative_pairs)}")
    
    print(f"\nPredicted pairs (same cluster): {len(predicted_pairs)}")
    
    # Check overlap between ground truth and clustering results
    all_gt_items = set()
    for pair in positive_pairs.union(negative_pairs):
        all_gt_items.update(pair)
    
    clustered_items = set(item_to_cluster.keys())
    overlap_items = all_gt_items.intersection(clustered_items)
    
    print(f"\nMatching Analysis:")
    print(f"  Items in ground truth: {len(all_gt_items)}")
    print(f"  Items in clustering results: {len(clustered_items)}")
    print(f"  Items found in both (overlap): {len(overlap_items)}")
    print(f"  Overlap percentage: {len(overlap_items)/len(all_gt_items)*100:.1f}% of ground truth items")
    
    # Analyze overlaps
    tp_pairs = predicted_pairs.intersection(positive_pairs)
    fp_pairs = predicted_pairs.intersection(negative_pairs)
    fn_pairs = positive_pairs - predicted_pairs
    
    print(f"\nPair-wise Analysis:")
    print(f"  True Positives (correctly clustered together): {len(tp_pairs)}")
    print(f"  False Positives (incorrectly clustered together): {len(fp_pairs)}")
    print(f"  False Negatives (should be together but separated): {len(fn_pairs)}")
    
    # Show some examples
    if len(tp_pairs) > 0:
        print(f"\nExample True Positives (first 3):")
        for i, pair in enumerate(list(tp_pairs)[:3]):
            print(f"    Pair {i+1}:")
            print(f"      Item 1: {pair[0][:100]}...")
            print(f"      Item 2: {pair[1][:100]}...")
    
    if len(fp_pairs) > 0:
        print(f"\nExample False Positives (first 3):")
        for i, pair in enumerate(list(fp_pairs)[:3]):
            print(f"    Pair {i+1}:")
            print(f"      Item 1: {pair[0][:100]}...")
            print(f"      Item 2: {pair[1][:100]}...")
    
    if len(fn_pairs) > 0:
        print(f"\nExample False Negatives (first 3):")
        for i, pair in enumerate(list(fn_pairs)[:3]):
            print(f"    Pair {i+1}:")
            print(f"      Item 1: {pair[0][:100]}...")
            print(f"      Item 2: {pair[1][:100]}...")

def main():
    parser = argparse.ArgumentParser(description='Evaluate clustering results against ground truth data')
    parser.add_argument('ground_truth', help='Path to the ground truth JSON file')
    parser.add_argument('predictions', help='Path to the clustering predictions JSON file')
    parser.add_argument('--output', '-o', help='Output file for results (optional)', default=None)
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading ground truth data from {args.ground_truth}...")
    positive_pairs, negative_pairs = load_ground_truth(args.ground_truth)
    
    print(f"Loading clustering results from {args.predictions}...")
    item_to_cluster = load_clustering_results(args.predictions)
    
    print("Generating predicted pairs...")
    predicted_pairs = generate_predicted_pairs(item_to_cluster)
    
    # Calculate metrics
    print("Calculating pairwise metrics...")
    pairwise_metrics = calculate_pairwise_metrics(predicted_pairs, positive_pairs, negative_pairs)
    
    print("Calculating cluster-level metrics...")
    cluster_metrics = calculate_cluster_level_metrics(item_to_cluster, positive_pairs, negative_pairs)
    
    # Print results
    print("\n" + "="*80)
    print("CLUSTERING EVALUATION RESULTS")
    print("="*80)
    
    print(f"\n📊 PAIRWISE METRICS:")
    print(f"  Precision:  {pairwise_metrics['precision']:.4f}")
    print(f"  Recall:     {pairwise_metrics['recall']:.4f}")
    print(f"  F1-Score:   {pairwise_metrics['f1_score']:.4f}")
    print(f"  Accuracy:   {pairwise_metrics['accuracy']:.4f}")
    
    print(f"\n📈 CLUSTER-LEVEL METRICS:")
    print(f"  Adjusted Rand Index:        {cluster_metrics['adjusted_rand_index']:.4f}")
    print(f"  Normalized Mutual Info:     {cluster_metrics['normalized_mutual_info']:.4f}")
    print(f"  Homogeneity:               {cluster_metrics['homogeneity']:.4f}")
    print(f"  Completeness:              {cluster_metrics['completeness']:.4f}")
    print(f"  V-Measure:                 {cluster_metrics['v_measure']:.4f}")
    
    print(f"\n📋 CONFUSION MATRIX:")
    print(f"  True Positives:   {pairwise_metrics['true_positives']}")
    print(f"  False Positives:  {pairwise_metrics['false_positives']}")
    print(f"  False Negatives:  {pairwise_metrics['false_negatives']}")
    print(f"  True Negatives:   {pairwise_metrics['true_negatives']}")
    
    # Print detailed analysis
    print_detailed_analysis(predicted_pairs, positive_pairs, negative_pairs, item_to_cluster)
    
    # Save results to JSON
    results = {
        'pairwise_metrics': pairwise_metrics,
        'cluster_metrics': cluster_metrics,
        'summary': {
            'total_clusters': len(set(item_to_cluster.values())),
            'total_items': len(item_to_cluster),
            'ground_truth_positive_pairs': len(positive_pairs),
            'ground_truth_negative_pairs': len(negative_pairs),
            'predicted_pairs': len(predicted_pairs)
        }
    }
    
    # Determine output filename
    if args.output:
        output_file = args.output
    else:
        # Generate output filename based on predictions file
        import os
        predictions_basename = os.path.splitext(os.path.basename(args.predictions))[0]
        output_file = f'clustering_evaluation_results_{predictions_basename}.json'
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to '{output_file}'")

if __name__ == "__main__":
    main() 
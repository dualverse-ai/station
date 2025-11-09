#!/usr/bin/env python3
"""
Batch integration evaluation script.
Usage: python eval.py --submission bbknn.py [--test]
"""

# Limit threading BEFORE any imports - CRITICAL for preventing thread explosion and memory corruption!
import os
os.environ['OPENBLAS_NUM_THREADS'] = '32'
os.environ['MKL_NUM_THREADS'] = '32'
os.environ['NUMEXPR_NUM_THREADS'] = '32'
os.environ['OMP_NUM_THREADS'] = '32'

import sys
import time
import warnings
import argparse
import importlib.util

# Suppress all noisy warnings before importing libraries
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import anndata as ad

# Add current folder to path to import the evaluation version
sys.path.insert(0, os.path.dirname(__file__))
import openproblems_metrics_eval as openproblems_metrics
from openproblems_metrics_eval import process_integration_output, compute_asw_batch, compute_asw_label, compute_clustering_overlap, compute_graph_connectivity, compute_isolated_label_asw, compute_isolated_label_f1, compute_kbet, compute_lisi, compute_pcr, compute_cell_cycle_conservation, compute_hvg_overlap


def load_submission(submission_file):
    """Load the submission function from the specified file."""
    spec = importlib.util.spec_from_file_location("submission", submission_file)
    submission_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(submission_module)
    return submission_module.eliminate_batch_effect_fn


def get_test_dataset_baselines(dataset_name):
    """Extract min/max values for a specific test dataset from openproblems_results.yaml.

    IMPORTANT: Calculates min/max using ONLY control methods per the specification.
    Control methods are: embed_cell_types, embed_cell_types_jittered, no_integration,
    no_integration_batch, shuffle_integration, shuffle_integration_by_batch,
    shuffle_integration_by_cell_type

    Args:
        dataset_name: Name of the dataset (e.g., 'dkd', 'gtex_v9', etc.)

    Returns:
        Dictionary with metric names as keys and {'min': val, 'max': val} as values
        If no control methods have valid scores for a metric, returns {'min': 0.0, 'max': 0.0}
    """
    import yaml

    # Define the 7 control methods per specification
    CONTROL_METHODS = {
        'embed_cell_types',
        'embed_cell_types_jittered',
        'no_integration',
        'no_integration_batch',
        'shuffle_integration',
        'shuffle_integration_by_batch',
        'shuffle_integration_by_cell_type'
    }

    # Map dataset names to their IDs in the YAML file
    dataset_id_mapping = {
        'dkd': 'cellxgene_census/dkd',
        'gtex_v9': 'cellxgene_census/gtex_v9',
        'hypomap': 'cellxgene_census/hypomap',
        'immune_cell_atlas': 'cellxgene_census/immune_cell_atlas',
        'mouse_pancreas_atlas': 'cellxgene_census/mouse_pancreas_atlas',
        'tabula_sapiens': 'cellxgene_census/tabula_sapiens'
    }

    # Check if this is a test dataset
    test_dataset_key = None
    for key in dataset_id_mapping:
        if dataset_name.startswith(key):
            test_dataset_key = key
            break

    if not test_dataset_key:
        return None

    dataset_id = dataset_id_mapping[test_dataset_key]

    # Load openproblems results
    with open('openproblems_results.yaml', 'r') as f:
        results = yaml.safe_load(f)

    # First, collect all metrics seen in control methods (even if all are NaN)
    all_metrics_seen = set()
    metric_scores = {}

    for entry in results:
        # Check both dataset and control method
        if (entry.get('dataset_id') == dataset_id and
            entry.get('method_id') in CONTROL_METHODS):

            metric_ids = entry.get('metric_ids', [])
            metric_values = entry.get('metric_values', [])

            for metric_name, value in zip(metric_ids, metric_values):
                all_metrics_seen.add(metric_name)
                if metric_name not in metric_scores:
                    metric_scores[metric_name] = []
                # Skip NaN values
                if not (isinstance(value, float) and np.isnan(value)) and value != '.NaN':
                    metric_scores[metric_name].append(float(value))

    # Also check what metrics exist for this dataset from ANY method
    # This helps identify metrics where all control methods have NaN/missing scores
    all_dataset_metrics = set()
    for entry in results:
        if entry.get('dataset_id') == dataset_id:
            metric_ids = entry.get('metric_ids', [])
            all_dataset_metrics.update(metric_ids)

    # Calculate min/max for each metric from control methods only
    baselines = {}

    # First handle metrics where control methods have valid scores
    for metric_name, values in metric_scores.items():
        if values:  # Only if we have valid values from control methods
            baselines[metric_name] = {
                'min': min(values),
                'max': max(values)
            }

    # Now handle metrics that exist in the dataset but have no valid control method scores
    # These should get min=0, max=0
    for metric_name in all_dataset_metrics:
        if metric_name not in baselines:
            # Either no control method reported this metric, or all had NaN/missing
            baselines[metric_name] = {
                'min': 0.0,
                'max': 0.0
            }

    # If no baselines found at all, return empty dict
    # (will cause normalized scores to be 0.0)
    if not baselines:
        print(f"Warning: No metrics found for dataset '{dataset_id}'")
        return {}

    return baselines


def evaluate_submission(submission_file, test_mode=False, dataset_file=None, single_metric=None):
    """Evaluate a batch integration submission.

    Args:
        submission_file: Path to the submission Python file
        test_mode: If True, use subset of data for testing
        dataset_file: Optional custom dataset file to use instead of default
        single_metric: Optional single metric to compute (skips others)
    """

    start_time = time.time()
    print("=== Batch Integration Evaluation ===\n")

    # Step 1: Load training data (two-file architecture)
    print("1. Loading training data...")
    step_start = time.time()

    # Use custom dataset (requires two-file architecture)
    if dataset_file:
        print(f"   Using custom dataset: {dataset_file}")

        # Must have corresponding solution file
        solution_file = dataset_file.replace('.h5ad', '_solution.h5ad')
        if not os.path.exists(solution_file):
            raise FileNotFoundError(
                f"Solution file not found: {solution_file}\n"
                f"Two-file architecture required:\n"
                f"  - Dataset (HVG): {dataset_file}\n"
                f"  - Solution (full): {solution_file}\n"
                f"Run: python prepare_data.py --process_only <raw_file.h5ad> to generate both files"
            )

        print(f"   Dataset file (HVG): {dataset_file}")
        print(f"   Solution file (full): {solution_file}")
        adata_original = ad.read_h5ad(dataset_file)
        adata_solution = ad.read_h5ad(solution_file)
    else:
        print("   Using default two-file architecture:")
        print("     - HVG dataset (for methods): batch_integration_train.h5ad")
        print("     - Solution (for metrics): batch_integration_train_solution.h5ad")
        adata_original = ad.read_h5ad('batch_integration_train.h5ad')
        adata_solution = ad.read_h5ad('batch_integration_train_solution.h5ad')

    if test_mode:
        print("   TEST MODE: Using subset of 2000 cells")
        adata_original = adata_original[:2000].copy()
        adata_solution = adata_solution[:2000].copy()

    print(f"   HVG dataset loaded: {adata_original.shape}")
    print(f"   Solution dataset loaded: {adata_solution.shape}")

    # Copy metadata (cell_type and other labels) from solution to HVG dataset
    # This is needed because metrics require cell_type labels which are only in solution file
    metadata_columns = [col for col in adata_solution.obs.columns if col not in adata_original.obs.columns]
    for col in metadata_columns:
        adata_original.obs[col] = adata_solution.obs[col]
    print(f"   Copied {len(metadata_columns)} metadata columns from solution: {metadata_columns[:5]}{'...' if len(metadata_columns) > 5 else ''}")

    print(f"   Batches: {adata_original.obs['batch'].cat.categories.tolist()}")
    print(f"   Time: {time.time() - step_start:.1f}s\n")

    # Step 2: Load and run submission
    print(f"2. Running submission: {submission_file}")
    step_start = time.time()

    eliminate_batch_effect_fn = load_submission(submission_file)
    adata_input = adata_original.copy()
    adata_integrated = eliminate_batch_effect_fn(adata_input)

    print(f"   Integration output: {adata_integrated.shape}")
    print(f"   Output format: obsm={list(adata_integrated.obsm.keys())}, obsp={list(adata_integrated.obsp.keys())}")
    print(f"   Time: {time.time() - step_start:.1f}s\n")

    # Step 3: Process integration output (OpenProblems workflow)
    print("3. Processing integration output...")
    step_start = time.time()
    adata_processed = process_integration_output(adata_integrated, adata_original)
    print(f"   Processed: {adata_processed.shape}")
    print(f"   Added clustering: {adata_processed.obsm['clustering'].shape}")
    print(f"   Time: {time.time() - step_start:.1f}s\n")

    # Step 4: Compute metrics
    if single_metric:
        print(f"4. Computing single metric: {single_metric}")
    else:
        print("4. Computing all 13 metrics...")

    # Load baselines for normalization
    import json
    with open('../research/storage/system/openproblems_baselines.json') as f:
        baselines = json.load(f)

    # Test each metric with timing
    step_start = time.time()
    results = {}

    # Define all metrics with their functions and data requirements
    all_metrics = {
        # Single-dataset metrics
        'asw_batch': (compute_asw_batch, 'single'),
        'asw_label': (compute_asw_label, 'single'),
        'graph_connectivity': (compute_graph_connectivity, 'single'),
        'isolated_label_asw': (compute_isolated_label_asw, 'single'),
        'isolated_label_f1': (compute_isolated_label_f1, 'single'),
        'kbet': (compute_kbet, 'single'),
        # Compound metrics (return multiple values)
        'clustering': (compute_clustering_overlap, 'compound'),
        'ari': (compute_clustering_overlap, 'compound'),
        'nmi': (compute_clustering_overlap, 'compound'),
        'ari_batch': (compute_clustering_overlap, 'compound'),
        'nmi_batch': (compute_clustering_overlap, 'compound'),
        'lisi': (compute_lisi, 'compound'),
        'ilisi': (compute_lisi, 'compound'),
        'clisi': (compute_lisi, 'compound'),
        # Two-dataset metrics
        'pcr': (compute_pcr, 'dual'),
        'cell_cycle_conservation': (compute_cell_cycle_conservation, 'dual'),
        'hvg_overlap': (compute_hvg_overlap, 'dual'),
    }

    # If single metric specified, only compute that one
    if single_metric:
        if single_metric not in all_metrics:
            print(f"   ERROR: Unknown metric '{single_metric}'")
            print(f"   Available metrics: {', '.join(sorted(all_metrics.keys()))}")
            return 0.0, {}, {}

        metric_func, metric_type = all_metrics[single_metric]

        try:
            metric_start = time.time()

            if metric_type == 'single':
                raw_score = metric_func(adata_processed)
                results[single_metric] = raw_score
            elif metric_type == 'compound':
                # Handle compound metrics that return multiple scores
                if single_metric in ['clustering', 'ari', 'nmi', 'ari_batch', 'nmi_batch']:
                    clustering_scores = compute_clustering_overlap(adata_processed)
                    results.update(clustering_scores)
                    if single_metric != 'clustering':
                        raw_score = clustering_scores.get(single_metric, np.nan)
                    else:
                        raw_score = clustering_scores
                elif single_metric in ['lisi', 'ilisi', 'clisi']:
                    lisi_scores = compute_lisi(adata_processed)
                    results.update(lisi_scores)
                    if single_metric != 'lisi':
                        raw_score = lisi_scores.get(single_metric, np.nan)
                    else:
                        raw_score = lisi_scores
            elif metric_type == 'dual':
                # Use solution file for metrics that need all genes (e.g., cell_cycle_conservation)
                raw_score = metric_func(adata_solution, adata_processed)
                results[single_metric] = raw_score

            metric_time = time.time() - metric_start
            print(f"   {single_metric}: {raw_score} ({metric_time:.1f}s)")

        except Exception as e:
            print(f"   {single_metric}: ERROR - {e}")
            results[single_metric] = np.nan
    else:
        # Original code: compute all metrics
        # Single-dataset metrics
        metric_functions = [
            ('asw_batch', compute_asw_batch),
            ('asw_label', compute_asw_label),
            ('graph_connectivity', compute_graph_connectivity),
            ('isolated_label_asw', compute_isolated_label_asw),
            ('isolated_label_f1', compute_isolated_label_f1),
            ('kbet', compute_kbet),
        ]

        for metric_name, metric_func in metric_functions:
            try:
                metric_start = time.time()
                raw_score = metric_func(adata_processed)
                metric_time = time.time() - metric_start
                results[metric_name] = raw_score
                print(f"   {metric_name}: {raw_score:.6f} ({metric_time:.1f}s)")
            except Exception as e:
                print(f"   {metric_name}: ERROR - {e}")
                results[metric_name] = np.nan

        # Compound metrics
        try:
            metric_start = time.time()
            clustering_scores = compute_clustering_overlap(adata_processed)
            metric_time = time.time() - metric_start
            results.update(clustering_scores)
            print(f"   clustering (ari/nmi): {clustering_scores} ({metric_time:.1f}s)")
        except Exception as e:
            print(f"   clustering: ERROR - {e}")

        try:
            metric_start = time.time()
            lisi_scores = compute_lisi(adata_processed)
            metric_time = time.time() - metric_start
            results.update(lisi_scores)
            print(f"   lisi: {lisi_scores} ({metric_time:.1f}s)")
        except Exception as e:
            print(f"   lisi: ERROR - {e}")

        # Two-dataset metrics (all use solution file per OpenProblems 2.0)
        dual_metrics = [
            ('pcr', compute_pcr, 'solution'),                    # Needs batch HVGs from solution
            ('cell_cycle_conservation', compute_cell_cycle_conservation, 'solution'),  # Needs all genes
            ('hvg_overlap', compute_hvg_overlap, 'solution'),    # Needs solution HVGs vs method HVGs
        ]

        for metric_name, metric_func, data_type in dual_metrics:
            try:
                metric_start = time.time()
                # All dual metrics use solution file per OpenProblems 2.0 architecture
                reference_data = adata_solution
                print(f"   {metric_name} (using solution with {reference_data.shape[1]} genes):")

                raw_score = metric_func(reference_data, adata_processed)
                metric_time = time.time() - metric_start
                results[metric_name] = raw_score
                print(f"     Score: {raw_score:.6f} ({metric_time:.1f}s)")
            except Exception as e:
                print(f"   {metric_name}: ERROR - {e}")
                results[metric_name] = np.nan

    metrics_time = time.time() - step_start
    print(f"   Total metrics time: {metrics_time:.1f}s\n")

    # Step 5: Normalize and compute final score
    print("5. Computing normalized scores...")
    step_start = time.time()

    # Check if this is a test dataset and get appropriate baselines
    dataset_basename = os.path.basename(dataset_file) if dataset_file else 'batch_integration_train.h5ad'
    dataset_name = os.path.splitext(dataset_basename)[0]

    # Try to get test dataset baselines
    test_baselines = get_test_dataset_baselines(dataset_name)
    if test_baselines:
        print(f"   Using test dataset baselines for: {dataset_name}")
        baselines = test_baselines
    else:
        print(f"   Using default training baselines")
        # baselines already loaded from openproblems_baselines.json

    # Target 13 metrics
    target_metrics = [
        'asw_batch', 'asw_label', 'ari', 'nmi', 'graph_connectivity',
        'isolated_label_asw', 'isolated_label_f1', 'kbet',
        'ilisi', 'clisi', 'pcr', 'cell_cycle_conservation', 'hvg_overlap'
    ]

    normalized_scores = {}
    missing_baselines = []  # Track metrics without baselines

    for metric_name in target_metrics:
        if metric_name in results:
            raw_score = results[metric_name]
            if metric_name in baselines and not np.isnan(raw_score):
                min_val = baselines[metric_name]['min']
                max_val = baselines[metric_name]['max']
                if max_val != min_val:
                    normalized = (raw_score - min_val) / (max_val - min_val)
                    normalized = max(0.0, min(1.0, normalized))
                else:
                    normalized = 1.0
            else:
                missing_baselines.append(metric_name)
                normalized = 0.0
        else:
            missing_baselines.append(metric_name)
            normalized = 0.0

        normalized_scores[metric_name] = normalized
        raw_val = results.get(metric_name, np.nan)
        print(f"   {metric_name}: {raw_val:.6f} → {normalized:.6f}")

    # Print warning for missing baselines
    if missing_baselines:
        print("\n" + "="*70)
        print("WARNING: MISSING NORMALIZATION BASELINES!")
        print("="*70)
        print(f"The following metrics have NO baseline for {dataset_name}:")
        for metric in missing_baselines:
            raw = results.get(metric, np.nan)
            print(f"  - {metric}: raw score {raw:.6f} normalized to 0.000000")
        print("These metrics are being penalized to 0!")
        print("="*70 + "\n")

    # Final score (13 metrics average)
    valid_scores = [score for score in normalized_scores.values() if not np.isnan(score)]
    final_score = np.mean(valid_scores)

    normalize_time = time.time() - step_start
    total_time = time.time() - start_time

    print(f"   Normalization time: {normalize_time:.1f}s\n")

    print(f"=== Final Results ===")
    print(f"Submission: {submission_file}")
    print(f"Metrics computed: {len(valid_scores)}/13")
    print(f"Final score: {final_score:.6f}")
    print(f"Total evaluation time: {total_time:.1f}s")

    return final_score, results, normalized_scores


def evaluate_multiple_datasets(submission_file, dataset_folder, test_mode=False):
    """Evaluate a submission on multiple datasets and compute mean scores.

    Args:
        submission_file: Path to the submission Python file
        dataset_folder: Folder containing .h5ad files to evaluate
        test_mode: If True, use subset of data for testing

    Returns:
        Mean scores across all datasets
    """
    import glob

    # Find all h5ad files in the folder, excluding _solution.h5ad files
    all_files = glob.glob(os.path.join(dataset_folder, "*.h5ad"))
    dataset_files = [f for f in all_files if not f.endswith('_solution.h5ad')]
    if not dataset_files:
        print(f"Error: No dataset .h5ad files found in {dataset_folder} (excluding _solution.h5ad files)")
        return 0.0, {}, {}

    print(f"\n{'='*60}")
    print(f"BATCH EVALUATION MODE - Processing {len(dataset_files)} datasets")
    print(f"{'='*60}\n")

    # Target metrics for aggregation
    target_metrics = [
        'asw_batch', 'asw_label', 'ari', 'nmi', 'graph_connectivity',
        'isolated_label_asw', 'isolated_label_f1', 'kbet',
        'ilisi', 'clisi', 'pcr', 'cell_cycle_conservation', 'hvg_overlap'
    ]

    # Store results for each dataset
    all_raw_results = []
    all_normalized_results = []
    all_final_scores = []

    # Process each dataset
    for idx, dataset_file in enumerate(sorted(dataset_files), 1):
        dataset_name = os.path.basename(dataset_file)
        print(f"\n{'='*60}")
        print(f"DATASET {idx}/{len(dataset_files)}: {dataset_name}")
        print(f"{'='*60}\n")

        try:
            final_score, raw_results, normalized_results = evaluate_submission(
                submission_file, test_mode, dataset_file
            )

            # Validate that all 13 metrics have normalized scores between 0 and 1
            for metric in target_metrics:
                if metric not in normalized_results:
                    raise ValueError(f"Missing metric '{metric}' in normalized results for dataset {dataset_name}")
                norm_score = normalized_results[metric]
                if np.isnan(norm_score):
                    raise ValueError(f"Metric '{metric}' has NaN value for dataset {dataset_name}")
                if norm_score < 0.0 or norm_score > 1.0:
                    raise ValueError(f"Metric '{metric}' has invalid normalized score {norm_score} (must be 0-1) for dataset {dataset_name}")

            all_raw_results.append(raw_results)
            all_normalized_results.append(normalized_results)
            all_final_scores.append(final_score)

        except Exception as e:
            print(f"\n!!! FATAL ERROR processing {dataset_name}: {e}")
            raise  # Re-raise the exception to stop processing

    # Compute mean scores across all datasets
    print(f"\n{'='*60}")
    print(f"FINAL SUMMARY - Mean Scores Across {len(dataset_files)} Datasets")
    print(f"{'='*60}\n")

    # Aggregate raw scores (all should be valid now)
    mean_raw_scores = {}
    for metric in target_metrics:
        values = [result[metric] for result in all_raw_results]
        mean_raw_scores[metric] = np.mean(values)

    # Aggregate normalized scores (all should be valid now)
    mean_normalized_scores = {}
    for metric in target_metrics:
        values = [result[metric] for result in all_normalized_results]
        mean_normalized_scores[metric] = np.mean(values)

    # Print mean results
    print("Mean Metric Scores:")
    for metric in target_metrics:
        raw_val = mean_raw_scores[metric]
        norm_val = mean_normalized_scores[metric]
        print(f"   {metric}: raw={raw_val:.6f}, normalized={norm_val:.6f}")

    # Compute overall mean score (all should be valid now)
    mean_final_score = np.mean(all_final_scores)

    print(f"\nMean Final Score: {mean_final_score:.6f}")
    print(f"Successfully processed: {len(all_final_scores)}/{len(dataset_files)} datasets")

    return mean_final_score, mean_raw_scores, mean_normalized_scores


def main():
    parser = argparse.ArgumentParser(description="Evaluate batch integration submission")
    parser.add_argument("--submission", required=True, help="Path to submission file (e.g., bbknn.py)")
    parser.add_argument("--test", action="store_true", help="Run in test mode with subset of data")
    parser.add_argument("--dataset", type=str, help="Path to custom dataset file (default: system/batch_integration_train.h5ad)")
    parser.add_argument("--dataset_folder", type=str, help="Path to folder containing multiple .h5ad files to evaluate")
    parser.add_argument("--metric", type=str, help="Run only a specific metric (e.g., hvg_overlap, asw_batch, pcr, etc.)")

    args = parser.parse_args()

    if not os.path.exists(args.submission):
        print(f"Error: Submission file '{args.submission}' not found")
        sys.exit(1)

    if args.dataset and not os.path.exists(args.dataset):
        print(f"Error: Dataset file '{args.dataset}' not found")
        sys.exit(1)

    if args.dataset_folder and not os.path.exists(args.dataset_folder):
        print(f"Error: Dataset folder '{args.dataset_folder}' not found")
        sys.exit(1)

    if args.dataset and args.dataset_folder:
        print(f"Error: Cannot specify both --dataset and --dataset_folder")
        sys.exit(1)

    print(f"Evaluating batch integration submission...")
    print(f"Environment: batch_integration conda env")

    if args.dataset_folder:
        # Multiple dataset evaluation mode
        print(f"Mode: Batch evaluation on folder - {args.dataset_folder}")
        if args.metric:
            print(f"Warning: --metric option is ignored in batch mode")
        print()

        final_score, raw_results, normalized_results = evaluate_multiple_datasets(
            args.submission, args.dataset_folder, args.test
        )
    else:
        # Single dataset evaluation mode
        if args.dataset:
            print(f"Dataset: Custom - {args.dataset}")
        else:
            print(f"Dataset: Default - 20k heart cells, 4 batches, 19 cell types")
        print()

        final_score, raw_results, normalized_results = evaluate_submission(
            args.submission, args.test, args.dataset, args.metric
        )


if __name__ == "__main__":
    main()

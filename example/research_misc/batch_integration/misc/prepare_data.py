#!/usr/bin/env python3
"""
Data preparation script for batch integration research task.
This script downloads and processes human training data and 6 test datasets.

Usage:
    # Full pipeline - generate all datasets
    python prepare_data.py                                    # Save to current directory
    python prepare_data.py --dir /path/to/data                # Save to specified directory

    # Process only mode - process existing h5ad file
    python prepare_data.py --process_only file.h5ad           # Save to current directory
    python prepare_data.py --process_only file.h5ad --dir /path/to/output

Output structure:
    dir/
    ├── cellxgene_364bd0c7.h5ad          # Raw human heart dataset
    ├── human_train.h5ad                  # Human training (HVG)
    ├── human_train_solution.h5ad         # Human training (full genes)
    ├── human_val.h5ad                    # Human validation (HVG)
    ├── human_val_solution.h5ad           # Human validation (full genes)
    ├── human_metadata.json
    ├── test_data_raw/                    # Raw test datasets
    │   ├── dkd.h5ad
    │   ├── gtex_v9.h5ad
    │   ├── hypomap.h5ad
    │   ├── immune_cell_atlas.h5ad
    │   ├── mouse_pancreas_atlas.h5ad
    │   └── tabula_sapiens.h5ad
    └── test_data/                        # Processed test datasets
        ├── dkd_processed.h5ad
        ├── dkd_processed_solution.h5ad
        └── ... (12 files total)
"""

import os
import sys
import gc
import numpy as np
import anndata as ad
import scanpy as sc
import pandas as pd
from pathlib import Path
import json
import argparse
import subprocess

# Add station to path for imports
current_dir = Path(__file__).parent
station_root = current_dir.parent.parent.parent
sys.path.insert(0, str(station_root))

# Test dataset S3 URLs
TEST_DATASETS = {
    'dkd': 's3://openproblems-data/resources/datasets/cellxgene_census/dkd/log_cp10k/dataset.h5ad',
    'gtex_v9': 's3://openproblems-data/resources/datasets/cellxgene_census/gtex_v9/log_cp10k/dataset.h5ad',
    'hypomap': 's3://openproblems-data/resources/datasets/cellxgene_census/hypomap/log_cp10k/dataset.h5ad',
    'immune_cell_atlas': 's3://openproblems-data/resources/datasets/cellxgene_census/immune_cell_atlas/log_cp10k/dataset.h5ad',
    'mouse_pancreas_atlas': 's3://openproblems-data/resources/datasets/cellxgene_census/mouse_pancreas_atlas/log_cp10k/dataset.h5ad',
    'tabula_sapiens': 's3://openproblems-data/resources/datasets/cellxgene_census/tabula_sapiens/log_cp10k/dataset.h5ad'
}

def check_human_dataset_status(data_dir):
    """Check if human dataset files exist.

    Returns:
        dict with 'raw_exists', 'processed_exist', 'need_download', 'need_process'
    """
    data_path = Path(data_dir)

    # Check raw file
    raw_file = data_path / "cellxgene_364bd0c7.h5ad"
    raw_exists = raw_file.exists()

    # Check processed files
    human_files = [
        "human_train.h5ad",
        "human_train_solution.h5ad",
        "human_val.h5ad",
        "human_val_solution.h5ad"
    ]

    processed_exist = all((data_path / f).exists() for f in human_files)

    return {
        'raw_exists': raw_exists,
        'processed_exist': processed_exist,
        'need_download': not raw_exists and not processed_exist,
        'need_process': raw_exists and not processed_exist,
        'all_complete': processed_exist
    }

def check_test_datasets_status(data_dir):
    """Check status of test datasets.

    Returns:
        dict with status for each test dataset
    """
    data_path = Path(data_dir)
    raw_path = data_path / "test_data_raw"
    processed_path = data_path / "test_data"

    status = {}
    for dataset_name in TEST_DATASETS.keys():
        raw_file = raw_path / f"{dataset_name}.h5ad"
        processed_file = processed_path / f"{dataset_name}_processed.h5ad"
        solution_file = processed_path / f"{dataset_name}_processed_solution.h5ad"

        status[dataset_name] = {
            'raw_exists': raw_file.exists(),
            'processed_exist': processed_file.exists() and solution_file.exists(),
            'need_download': not raw_file.exists(),
            'need_process': raw_file.exists() and not (processed_file.exists() and solution_file.exists())
        }

    return status

def download_with_aws_cli(s3_url, output_path):
    """Download file from S3 using AWS CLI with --no-sign-request."""
    print(f"  Downloading from: {s3_url}")
    print(f"  Saving to: {output_path}")

    # Use AWS CLI with no-sign-request for public bucket
    cmd = [
        'aws', 's3', 'cp',
        s3_url,
        str(output_path),
        '--no-sign-request'
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  ✓ Download complete: {output_path.name}")
            return True
        else:
            print(f"  ✗ Download failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"  ✗ AWS CLI error: {e}")
        print("  Make sure AWS CLI is installed: pip install awscli")
        return False

def check_and_download_heart_dataset(data_dir):
    """Check if heart dataset exists, download if not."""
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    dataset_path = data_path / "cellxgene_364bd0c7.h5ad"

    if dataset_path.exists():
        print(f"✓ Heart dataset already exists: {dataset_path}")
        return str(dataset_path)

    print("Heart dataset not found. Downloading...")

    try:
        import cellxgene_census

        dataset_id = "364bd0c7-f7fd-48ed-99c1-ae26872b1042"

        print(f"Downloading heart dataset {dataset_id}...")
        cellxgene_census.download_source_h5ad(
            dataset_id,
            to_path=str(dataset_path),
            census_version='stable',
            progress_bar=True
        )

        print(f"✓ Download complete: {dataset_path}")
        return str(dataset_path)

    except Exception as e:
        print(f"✗ Download failed: {e}")
        return None

def download_test_datasets(data_dir):
    """Download all test datasets that are missing."""
    data_path = Path(data_dir)
    raw_path = data_path / "test_data_raw"
    raw_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("DOWNLOADING TEST DATASETS")
    print("="*80)

    status = check_test_datasets_status(data_dir)

    to_download = [name for name, info in status.items() if info['need_download']]

    if not to_download:
        print("✓ All test datasets already downloaded!")
        return True

    print(f"\nNeed to download {len(to_download)} datasets: {', '.join(to_download)}")

    success_count = 0
    for dataset_name in to_download:
        print(f"\n[{success_count + 1}/{len(to_download)}] Downloading {dataset_name}...")

        s3_url = TEST_DATASETS[dataset_name]
        output_file = raw_path / f"{dataset_name}.h5ad"

        if download_with_aws_cli(s3_url, output_file):
            success_count += 1
        else:
            print(f"  ✗ Failed to download {dataset_name}")

    print(f"\n✓ Downloaded {success_count}/{len(to_download)} datasets successfully")
    return success_count == len(to_download)

def fix_format_issues(adata, dataset_name):
    """Fix format compatibility issues for batch integration (same for all datasets)."""

    print(f"Fixing format issues for {dataset_name}...")

    # 1. Rename Dataset → batch (if needed)
    if 'Dataset' in adata.obs.columns and 'batch' not in adata.obs.columns:
        adata.obs['batch'] = adata.obs['Dataset'].astype(str)
        print("✓ Added 'batch' column from 'Dataset'")
    elif 'batch' not in adata.obs.columns:
        # Some datasets may use other batch identifiers
        if 'donor_id' in adata.obs.columns:
            adata.obs['batch'] = adata.obs['donor_id'].astype(str)
            print("✓ Added 'batch' column from 'donor_id'")
        else:
            raise ValueError("No batch column found (tried 'batch', 'Dataset', 'donor_id')")

    # 2. Add dataset metadata
    adata.uns['dataset_id'] = dataset_name
    adata.uns['dataset_name'] = dataset_name
    adata.uns['normalization_id'] = 'raw_counts'

    # CRITICAL FIX: Add dataset_organism for cell cycle conservation metric
    # Convert from obs['organism'] format to OpenProblems format
    if 'organism' in adata.obs.columns:
        organism_value = adata.obs['organism'].iloc[0]
        if organism_value == 'Homo sapiens':
            adata.uns['dataset_organism'] = 'human'
        elif organism_value == 'Mus musculus':
            adata.uns['dataset_organism'] = 'mouse'
        else:
            # Auto-detect organism from gene symbols if annotation is unclear
            sample_genes = list(adata.var_names[:1000]) if len(adata.var_names) >= 1000 else list(adata.var_names)
            mouse_patterns = sum(1 for g in sample_genes if g.startswith('Gm') or g.startswith('Xkr') or any(p in str(g) for p in ['1700', '2900', '4930']))
            human_patterns = sum(1 for g in sample_genes if any(p in str(g) for p in ['LINC', 'ENSG', 'LOC', '-AS']) or (g.isupper() and len(g) > 2))

            if mouse_patterns > 2:
                adata.uns['dataset_organism'] = 'mouse'
                print(f"  Auto-detected organism: mouse (based on {mouse_patterns} mouse gene patterns)")
            elif human_patterns > 10:
                adata.uns['dataset_organism'] = 'human'
                print(f"  Auto-detected organism: human (based on {human_patterns} human gene patterns)")
            else:
                raise ValueError(f"Cannot determine organism from genes or annotation. Unknown organism: '{organism_value}'")
    else:
        # Auto-detect organism from gene symbols if no organism annotation
        sample_genes = list(adata.var_names[:1000]) if len(adata.var_names) >= 1000 else list(adata.var_names)
        mouse_patterns = sum(1 for g in sample_genes if g.startswith('Gm') or g.startswith('Xkr') or any(p in str(g) for p in ['1700', '2900', '4930']))
        human_patterns = sum(1 for g in sample_genes if any(p in str(g) for p in ['LINC', 'ENSG', 'LOC', '-AS']) or (g.isupper() and len(g) > 2))

        if mouse_patterns > 2:
            adata.uns['dataset_organism'] = 'mouse'
            print(f"  Auto-detected organism: mouse (based on {mouse_patterns} mouse gene patterns)")
        elif human_patterns > 10:
            adata.uns['dataset_organism'] = 'human'
            print(f"  Auto-detected organism: human (based on {human_patterns} human gene patterns)")
        else:
            raise ValueError("Cannot determine organism: no organism annotation and unclear gene patterns")

    print("✓ Added dataset metadata")
    print(f"  dataset_organism: {adata.uns['dataset_organism']}")

    # 3. Ensure categorical types
    adata.obs['batch'] = adata.obs['batch'].astype('category')
    adata.obs['cell_type'] = adata.obs['cell_type'].astype('category')

    # 4. Add gene metadata and fix var_names for cell cycle scoring
    if 'feature_name' not in adata.var.columns:
        adata.var['feature_name'] = adata.var_names
        adata.var['feature_id'] = adata.var_names
    else:
        # CRITICAL FIX: Replace ENSEMBL IDs with gene symbols for cell cycle scoring
        if 'feature_id' not in adata.var.columns:
            adata.var['feature_id'] = adata.var_names

        # Set var_names to gene symbols (required for cell cycle gene matching)
        print("  Converting var_names from ENSEMBL IDs to gene symbols...")
        adata.var_names = adata.var['feature_name'].astype(str)
        adata.var_names.name = None
        print(f"  Sample var_names after conversion: {adata.var_names[:5].tolist()}")

    # 5. Exact OpenProblems preprocessing pipeline
    print("Computing batch-aware HVGs (exact OpenProblems method)...")

    # Import scib for batch-aware HVG computation
    import scib

    # Ensure we have normalized layer for scib.pp.hvg_batch
    if 'normalized' not in adata.layers:
        adata.layers['normalized'] = adata.X.copy()

    # Batch-aware HVGs using scib (exact OpenProblems method)
    original_X = adata.X
    adata.X = adata.layers["normalized"]

    n_hvgs = 2000
    hvg_list = scib.pp.hvg_batch(
        adata, batch_key="batch", target_genes=n_hvgs, adataOut=False
    )

    # Restore original X
    adata.X = original_X
    adata.var["batch_hvg"] = adata.var_names.isin(hvg_list)

    # Standard HVG scores (exact OpenProblems method)
    print("Recomputing standard HVG scores...")
    out = sc.pp.highly_variable_genes(
        adata, layer="normalized", n_top_genes=n_hvgs, flavor="cell_ranger", inplace=False
    )
    adata.var["hvg"] = out["highly_variable"].values
    adata.var["hvg_score"] = out["dispersions_norm"].values

    print(f"✓ Format fixes applied")
    print(f"  Batches: {adata.obs['batch'].cat.categories.tolist()[:5]}{'...' if len(adata.obs['batch'].cat.categories) > 5 else ''}")
    print(f"  Cell types: {adata.obs['cell_type'].nunique()}")
    print(f"  HVGs: {adata.var['hvg'].sum()}")
    print(f"  Batch HVGs: {adata.var['batch_hvg'].sum()}")

    # Extract raw counts from adata.raw (cellxgene standard) - RESTORED FUNCTION
    if adata.raw is not None:
        print("\nExtracting raw counts from adata.raw...")
        gc.collect()
        adata.layers['counts'] = adata.raw.X.copy()
    elif 'counts' not in adata.layers:
        print("\nWarning: No raw data found, using normalized as placeholder for counts...")
        adata.layers['counts'] = adata.layers['normalized'].copy()
    else:
        print("\nUsing existing counts layer...")

    # Final step: Set X to raw counts
    print("Setting X to raw counts for method compatibility...")
    gc.collect()
    adata.X = adata.layers['counts'].copy()

    # Data validation checks - RESTORED FUNCTION
    print("\n=== Data Validation Checks ===")

    # Check X (should be raw counts)
    X_sample = adata.X[:100, :100]
    if hasattr(X_sample, 'toarray'):
        X_sample = X_sample.toarray()
    X_is_integer = np.all(X_sample == X_sample.astype(int))
    X_min, X_max, X_mean = X_sample.min(), X_sample.max(), X_sample.mean()
    print(f"  X matrix (should be raw counts):")
    print(f"    - Contains integers only: {X_is_integer} {'✓' if X_is_integer else '✗'}")
    print(f"    - Range: [{X_min:.1f}, {X_max:.1f}]")
    print(f"    - Mean: {X_mean:.3f}")

    # Verify cell cycle compatibility
    print("\n  Cell cycle scoring compatibility:")
    print(f"    - dataset_organism: {adata.uns.get('dataset_organism', 'NOT SET')}")
    is_gene_symbol = not adata.var_names[0].startswith('ENSG')
    print(f"    - var_names are gene symbols: {'✓' if is_gene_symbol else '✗'}")

    # Check for organism-specific cell cycle genes
    if adata.uns['dataset_organism'] == 'human':
        test_genes = ['MCM5', 'PCNA', 'TYMS']
    else:  # mouse
        test_genes = ['Mcm5', 'Pcna', 'Tyms']

    genes_found = sum(1 for g in test_genes if g in adata.var_names)
    print(f"    - Cell cycle genes found: {genes_found}/{len(test_genes)} {'✓' if genes_found > 0 else '✗'}")

    return adata

def sample_and_split_data(adata, n_train=20000, n_val=20000):
    """Sample cells and split into train/validation sets with no overlap - RESTORED ORIGINAL VERSION."""

    print(f"\nSampling data: {n_train:,} train + {n_val:,} validation cells...")

    total_needed = n_train + n_val
    available_cells = adata.n_obs

    print(f"Available cells: {available_cells:,}")
    print(f"Requested cells: {total_needed:,}")

    if available_cells < total_needed:
        print(f"⚠️  Not enough cells available. Using proportional split.")
        train_ratio = n_train / total_needed
        n_train = int(available_cells * train_ratio)
        n_val = available_cells - n_train
        print(f"Adjusted: {n_train:,} train + {n_val:,} validation")

    # Set random seed for reproducibility
    np.random.seed(42)

    # Stratified sampling to maintain batch and cell type proportions
    print("Performing stratified sampling...")

    # Get unique combinations of batch and cell type
    adata.obs['batch_celltype'] = (
        adata.obs['batch'].astype(str) + '_' +
        adata.obs['cell_type'].astype(str)
    )

    # Sample from each stratum
    sampled_indices = []

    for stratum in adata.obs['batch_celltype'].unique():
        stratum_mask = adata.obs['batch_celltype'] == stratum
        stratum_indices = np.where(stratum_mask)[0]
        stratum_size = len(stratum_indices)

        if stratum_size == 0:
            continue

        # Calculate proportional sampling
        stratum_proportion = stratum_size / available_cells
        stratum_target = int(total_needed * stratum_proportion)
        stratum_target = min(stratum_target, stratum_size)

        if stratum_target > 0:
            sampled_from_stratum = np.random.choice(
                stratum_indices,
                size=stratum_target,
                replace=False
            )
            sampled_indices.extend(sampled_from_stratum)

    # Convert to array and shuffle
    sampled_indices = np.array(sampled_indices)
    np.random.shuffle(sampled_indices)

    # Trim to exact size
    if len(sampled_indices) > total_needed:
        sampled_indices = sampled_indices[:total_needed]

    print(f"✓ Sampled {len(sampled_indices):,} cells total")

    # Ensure exact split sizes and NO OVERLAP
    actual_n_train = min(n_train, len(sampled_indices))
    actual_n_val = min(n_val, len(sampled_indices) - actual_n_train)

    # Split with NO OVERLAP
    train_indices = sampled_indices[:actual_n_train]
    val_indices = sampled_indices[actual_n_train:actual_n_train + actual_n_val]

    # Verify no overlap
    assert len(set(train_indices) & set(val_indices)) == 0, "ERROR: Train/val sets overlap!"

    adata_train = adata[train_indices].copy()
    adata_val = adata[val_indices].copy()

    # Clean up temporary columns
    for data in [adata_train, adata_val]:
        if 'batch_celltype' in data.obs:
            del data.obs['batch_celltype']

    print(f"✓ No overlap verified: {len(train_indices)} train + {len(val_indices)} val")
    print(f"✓ Split complete:")
    print(f"  Train: {adata_train.shape}")
    print(f"  Validation: {adata_val.shape}")

    return adata_train, adata_val

def save_two_file_architecture(adata, output_prefix):
    """Save dataset in two-file architecture: HVG-filtered + solution."""

    output_path = Path(output_prefix)
    hvg_path = output_path.with_suffix('.h5ad')
    solution_path = output_path.parent / (output_path.stem + '_solution.h5ad')

    print(f"\nSaving with two-file architecture...")
    print(f"HVG dataset (for methods): {hvg_path}")
    print(f"Solution dataset (for metrics): {solution_path}")

    # Save HVG-filtered dataset
    adata_hvg = adata[:, adata.var['hvg']].copy()
    print(f"  HVG genes: {adata_hvg.shape[1]}/{adata.shape[1]}")

    # Clean obs columns to prevent data leakage
    essential_obs_columns = ['batch', 'observation_joinid'] if 'observation_joinid' in adata_hvg.obs.columns else ['batch']
    columns_to_remove = [col for col in adata_hvg.obs.columns if col not in essential_obs_columns]

    if columns_to_remove:
        print(f"  Removing {len(columns_to_remove)} obs columns to prevent data leakage:")
        print(f"    Keeping: {essential_obs_columns}")
        print(f"    Removing: {columns_to_remove[:5]}{'...' if len(columns_to_remove) > 5 else ''}")
        adata_hvg.obs = adata_hvg.obs[essential_obs_columns].copy()

    adata_hvg.write_h5ad(hvg_path, compression='gzip')

    # Save solution dataset
    print(f"  Solution genes: {adata.shape[1]}")
    adata.write_h5ad(solution_path, compression='gzip')

    return str(hvg_path), str(solution_path)

def process_human_dataset(adata_original, data_dir):
    """Process human heart dataset: train + val splits."""

    print("\n" + "="*80)
    print("PROCESSING HUMAN DATASET (Heart)")
    print("="*80)

    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    # Fix format issues
    adata_processed = fix_format_issues(adata_original, 'heart_cellxgene_364bd0c7')

    # Sample and split data
    adata_train, adata_val = sample_and_split_data(
        adata_processed,
        n_train=20000,
        n_val=20000
    )

    # Save training data
    train_prefix = data_path / "human_train"
    train_dataset_path, train_solution_path = save_two_file_architecture(adata_train, train_prefix)

    # Save validation data
    val_prefix = data_path / "human_val"
    val_dataset_path, val_solution_path = save_two_file_architecture(adata_val, val_prefix)

    # Save metadata
    metadata = {
        'dataset_id': 'heart_cellxgene_364bd0c7',
        'original_dataset_id': '364bd0c7-f7fd-48ed-99c1-ae26872b1042',
        'dataset_name': 'Heart scRNA-seq Batch Integration Dataset',
        'organism': 'human',
        'tissue': 'heart',
        'architecture': 'two_file_openproblems_v2',
        'n_train': adata_train.shape[0],
        'n_val': adata_val.shape[0],
        'n_genes_total': adata_train.shape[1],
        'n_genes_hvg': sum(adata_train.var['hvg']),
        'n_batches': adata_train.obs['batch'].nunique(),
        'n_cell_types': adata_train.obs['cell_type'].nunique(),
        'batches': adata_train.obs['batch'].cat.categories.tolist(),
        'cell_types': adata_train.obs['cell_type'].cat.categories.tolist(),
        'train_dataset_path': train_dataset_path,
        'train_solution_path': train_solution_path,
        'val_dataset_path': val_dataset_path,
        'val_solution_path': val_solution_path,
        'data_type': 'raw_counts',
        'format': 'h5ad',
        'preparation_date': pd.Timestamp.now().isoformat()
    }

    metadata_path = data_path / "human_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ Human dataset processing complete!")
    print(f"  Train dataset (HVG): {Path(train_dataset_path).stat().st_size / (1024**2):.1f} MB")
    print(f"  Train solution (full): {Path(train_solution_path).stat().st_size / (1024**2):.1f} MB")
    print(f"  Val dataset (HVG): {Path(val_dataset_path).stat().st_size / (1024**2):.1f} MB")
    print(f"  Val solution (full): {Path(val_solution_path).stat().st_size / (1024**2):.1f} MB")
    print(f"  Metadata: {metadata_path}")

    return metadata

def process_test_datasets(data_dir):
    """Process all test datasets using the EXACT same logic as process_only."""
    data_path = Path(data_dir)
    raw_path = data_path / "test_data_raw"
    processed_path = data_path / "test_data"
    processed_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("PROCESSING TEST DATASETS")
    print("="*80)

    status = check_test_datasets_status(data_dir)

    to_process = [name for name, info in status.items()
                  if info['need_process'] or (info['raw_exists'] and not info['processed_exist'])]

    if not to_process:
        print("✓ All test datasets already processed!")
        return True

    print(f"\nNeed to process {len(to_process)} datasets: {', '.join(to_process)}")

    success_count = 0
    for idx, dataset_name in enumerate(to_process):
        print(f"\n[{idx + 1}/{len(to_process)}] Processing {dataset_name}...")

        raw_file = raw_path / f"{dataset_name}.h5ad"
        if not raw_file.exists():
            print(f"  ✗ Raw file not found: {raw_file}")
            continue

        # Use the process_only function to ensure identical processing
        output_prefix = processed_path / f"{dataset_name}_processed"
        success = process_only(str(raw_file), str(processed_path))

        if success:
            print(f"  ✓ Processed successfully")
            success_count += 1
        else:
            print(f"  ✗ Processing failed")

    print(f"\n✓ Processed {success_count}/{len(to_process)} datasets successfully")
    return success_count == len(to_process)

def process_only(input_file, data_dir):
    """Process an existing dataset file without downloading or sampling.

    THIS IS THE EXACT ORIGINAL FUNCTION - UNCHANGED

    Args:
        input_file: Path to the h5ad file to process
        data_dir: Directory to save processed files

    Returns:
        True if successful, False otherwise
    """
    input_path = Path(input_file)

    if not input_path.exists():
        print(f"✗ File not found: {input_file}")
        return False

    print(f"=== Processing Only Mode ===")
    print(f"Input file: {input_path}")
    print(f"Output directory: {data_dir}")

    # Ensure output directory exists
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print(f"\nLoading dataset...")
    adata = ad.read_h5ad(input_path)
    print(f"✓ Loaded: {adata.shape}")
    print(f"  Cells: {adata.n_obs:,}")
    print(f"  Genes: {adata.n_vars:,}")

    # Check required columns
    if 'batch' not in adata.obs.columns and 'Dataset' not in adata.obs.columns:
        print("✗ Error: Dataset must have 'batch' or 'Dataset' column in obs")
        return False

    if 'cell_type' not in adata.obs.columns:
        print("✗ Error: Dataset must have 'cell_type' column in obs")
        return False

    # Apply format fixes and preprocessing
    print("\nApplying OpenProblems preprocessing...")
    dataset_name = input_path.stem
    adata_processed = fix_format_issues(adata, dataset_name)

    # Generate output prefix for two-file architecture
    output_prefix = data_path / f"{dataset_name}_processed"

    # Save with two-file architecture
    hvg_path, solution_path = save_two_file_architecture(adata_processed, output_prefix)

    # Display summary statistics
    print(f"\n=== Processing Complete ===")
    print(f"✓ Two-file architecture created:")
    print(f"  HVG dataset: {hvg_path}")
    print(f"  Solution dataset: {solution_path}")
    print(f"  Shape: {adata_processed.shape}")
    print(f"  Batches: {adata_processed.obs['batch'].nunique()}")
    print(f"  Cell types: {adata_processed.obs['cell_type'].nunique()}")
    print(f"  HVGs: {adata_processed.var['hvg'].sum()}")
    print(f"  Batch HVGs: {adata_processed.var['batch_hvg'].sum()}")
    print(f"  HVG file size: {Path(hvg_path).stat().st_size / (1024**2):.1f} MB")
    print(f"  Solution file size: {Path(solution_path).stat().st_size / (1024**2):.1f} MB")

    return True

def main():
    """Main execution function."""

    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Prepare batch integration datasets (human training + 6 test datasets)"
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=str(Path(__file__).parent),
        help="Directory to download and save dataset files (default: current directory)"
    )
    parser.add_argument(
        "--process_only",
        type=str,
        help="Process an existing h5ad file without downloading or sampling"
    )

    args = parser.parse_args()
    data_dir = Path(args.dir)

    # Process only mode
    if args.process_only:
        return process_only(args.process_only, data_dir)

    print("=== Batch Integration Dataset Preparation ===")
    print(f"Data directory: {data_dir}\n")

    # Ensure directory exists
    data_dir.mkdir(parents=True, exist_ok=True)

    # ========== STEP 1: HUMAN DATASET ==========
    human_status = check_human_dataset_status(data_dir)

    if human_status['all_complete']:
        print("✓ All 4 human dataset files already exist, skipping human dataset preparation")
        for f in ["human_train.h5ad", "human_train_solution.h5ad", "human_val.h5ad", "human_val_solution.h5ad"]:
            fpath = data_dir / f
            if fpath.exists():
                print(f"  {f}: {fpath.stat().st_size / (1024**2):.1f} MB")
    else:
        print("="*80)
        print("PREPARING HUMAN DATASET")
        print("="*80)

        # Download if needed
        if human_status['need_download']:
            heart_path = check_and_download_heart_dataset(data_dir)
            if not heart_path:
                print("✗ Failed to obtain human heart dataset")
                return False
        else:
            heart_path = data_dir / "cellxgene_364bd0c7.h5ad"
            print(f"✓ Using existing raw file: {heart_path}")

        # Process the dataset
        print(f"\nLoading heart dataset...")
        adata_heart = ad.read_h5ad(heart_path)
        print(f"✓ Loaded: {adata_heart.shape}")

        # Process human dataset (train + val)
        human_metadata = process_human_dataset(adata_heart, data_dir)

        # Clean up memory
        del adata_heart
        gc.collect()

        print("\n✓ Human dataset preparation complete!")

    # ========== STEP 2: TEST DATASETS ==========

    # Download test datasets
    download_test_datasets(data_dir)

    # Process test datasets
    process_test_datasets(data_dir)

    # ========== FINAL SUMMARY ==========

    print("\n" + "="*80)
    print("=== FINAL SUMMARY ===")
    print("="*80)

    # Check final status
    human_status = check_human_dataset_status(data_dir)
    test_status = check_test_datasets_status(data_dir)

    print(f"\nData directory: {data_dir}")

    print(f"\nHUMAN DATASET:")
    if human_status['all_complete']:
        print("  ✓ All 4 human files ready")
    else:
        print("  ✗ Some human files missing")

    print(f"\nTEST DATASETS:")
    all_test_ready = all(info['processed_exist'] for info in test_status.values())
    if all_test_ready:
        print("  ✓ All 6 test datasets processed and ready")
    else:
        missing = [name for name, info in test_status.items() if not info['processed_exist']]
        print(f"  ✗ Missing processed files for: {', '.join(missing)}")

    if human_status['all_complete'] and all_test_ready:
        print("\n✅ ALL DATASETS READY FOR BATCH INTEGRATION RESEARCH!")

        # Show directory structure
        print("\nFinal directory structure:")
        print(f"{data_dir}/")
        print(f"├── cellxgene_364bd0c7.h5ad")
        print(f"├── human_train.h5ad")
        print(f"├── human_train_solution.h5ad")
        print(f"├── human_val.h5ad")
        print(f"├── human_val_solution.h5ad")
        print(f"├── human_metadata.json")
        print(f"├── test_data_raw/")
        for name in TEST_DATASETS.keys():
            print(f"│   ├── {name}.h5ad")
        print(f"└── test_data/")
        for name in TEST_DATASETS.keys():
            print(f"    ├── {name}_processed.h5ad")
            print(f"    ├── {name}_processed_solution.h5ad")

        return True
    else:
        print("\n⚠️  Some datasets are not ready. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
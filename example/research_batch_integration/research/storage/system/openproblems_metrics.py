#!/usr/bin/env python3
"""
OpenProblems 2.0 metric wrapper functions - exact implementations
Clean version with only wrapper functions
"""

import numpy as np
import anndata as ad
import pandas as pd
import scanpy as sc
import warnings
import gc
from scipy.sparse import issparse, csr_matrix

# Suppress noisy warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# GPU clustering control
USE_GPU = False

# Import clustering function based on GPU setting
if USE_GPU:
    from rapids_singlecell.tl import leiden
else:
    from scanpy.tl import leiden

# Import all required scib functions
from scib.metrics import (
    silhouette_batch, silhouette, ari, nmi,
    graph_connectivity, isolated_labels_asw, isolated_labels_f1,
    kBET, pcr_comparison, cell_cycle
)
from scib.metrics.clustering import cluster_optimal_resolution
from scib.metrics.lisi import lisi_graph_py

# Apply thread-safe patch for kBET to prevent segmentation faults
# This fixes crashes when parallel evaluations compute kBET simultaneously
from thread_safe_kbet import patch_scib_kbet
patch_scib_kbet()

# =============================================================================
# PROCESS INTEGRATION - Called before metrics to standardize format
# =============================================================================

def process_integration_output(adata_integrated, adata_original):
    """
    Process integration output to format expected by metrics.
    This function is called automatically before metric computation.

    Steps (following OpenProblems 2.0 workflow):
    1. Transform output formats and validate structure
    2. Precompute clustering at multiple resolutions
    3. Merge clustering results into the dataset

    Args:
        adata_integrated: Agent's integration output
        adata_original: Original training data

    Returns:
        adata_processed: Processed integration output ready for metrics
    """

    # Step 1: Transform output formats
    adata_processed = adata_integrated.copy()

    # Copy original metadata and batch information
    adata_processed.obs = adata_original.obs.copy()

    # Only copy var metadata for genes that exist in integrated output
    if adata_integrated.shape[1] != adata_original.shape[1]:
        # Method filtered genes - use subset of original var
        common_genes = adata_original.var_names.isin(adata_integrated.var_names)
        adata_processed.var = adata_original.var[common_genes].copy()
    else:
        # Same genes - copy all var metadata
        adata_processed.var = adata_original.var.copy()

    adata_processed.uns.update(adata_original.uns)

    # Transform output formats (exact OpenProblems 2.0 logic from transform/script.py)
    # If corrected_counts exists but no embedding, compute PCA from corrected counts
    if "corrected_counts" in adata_processed.layers and "X_emb" not in adata_processed.obsm:
        print("  Run PCA from corrected counts...")
        adata_processed.obsm["X_emb"] = sc.pp.pca(
            adata_processed.layers["corrected_counts"],
            n_comps=50,
            use_highly_variable=False,
            svd_solver="arpack",
            return_info=False
        )

    # If embedding exists but no neighbors, compute kNN from embedding
    if "X_emb" in adata_processed.obsm and "neighbors" not in adata_processed.uns:
        print("  Run kNN from embedding...")
        sc.pp.neighbors(adata_processed, use_rep="X_emb")


    # Process graph outputs if they exist
    if 'connectivities' in adata_processed.obsp and 'distances' in adata_processed.obsp:
        # Ensure matrices are sorted (required for LISI C++ implementation)
        if not adata_processed.obsp['connectivities'].has_sorted_indices:
            print("  Sorting connectivity matrix indices...")
            adata_processed.obsp['connectivities'].sort_indices()
        if not adata_processed.obsp['distances'].has_sorted_indices:
            print("  Sorting distances matrix indices...")
            adata_processed.obsp['distances'].sort_indices()

        # Add neighbors key if missing (required for LISI)
        if 'neighbors' not in adata_processed.uns:
            print("  Adding neighbors key for graph metrics compatibility...")
            adata_processed.uns['neighbors'] = {
                'connectivities_key': 'connectivities',
                'distances_key': 'distances'
            }

    # Step 2-3: Precompute clustering at multiple resolutions (OpenProblems optimized approach)
    resolutions = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    clustering_results = pd.DataFrame(index=adata_processed.obs_names)

    # Use OpenProblems optimized clustering parameters
    clustering_kwargs = {
        'flavor': 'igraph',
        'n_iterations': 2,
        'directed': False
    }

    # Set clustering parameters (exact OpenProblems logic)
    if USE_GPU:
        clustering_kwargs = {}  # Rapids uses default parameters
    else:
        clustering_kwargs = {
            'flavor': 'igraph',
            'n_iterations': 2,
        }

    for resolution in resolutions:
        resolution_key = f"leiden_{resolution}"
        leiden(
            adata_processed,
            resolution=resolution,
            key_added=resolution_key,
            **clustering_kwargs
        )
        clustering_results[resolution_key] = adata_processed.obs[resolution_key]

    # Store clustering results in obsm (OpenProblems format)
    adata_processed.obsm['clustering'] = clustering_results

    return adata_processed

# Metric 1: ASW Batch
def compute_asw_batch(adata):
    """Exact OpenProblems ASW Batch implementation."""
    score = silhouette_batch(
        adata,
        batch_key='batch',
        label_key="cell_type",
        embed='X_emb',
    )
    return score

# Metric 2: ASW Label
def compute_asw_label(adata):
    """Exact OpenProblems ASW Label implementation."""
    score = silhouette(
        adata,
        label_key="cell_type",
        embed='X_emb'
    )
    return score

# Metric 3-4: Clustering Overlap (ARI + NMI)
def compute_clustering_overlap(adata):
    """Exact OpenProblems Clustering Overlap implementation."""

    cluster_key = "leiden"

    # get existing clusters (exactly as OpenProblems)
    cluster_df = adata.obsm.get('clustering', pd.DataFrame(index=adata.obs_names))

    # Only add clustering columns that don't already exist (avoid duplicates)
    existing_leiden_cols = [col for col in adata.obs.columns if 'leiden' in col]
    new_clustering_cols = [col for col in cluster_df.columns if col not in existing_leiden_cols]

    if new_clustering_cols:
        adata.obs = pd.concat([adata.obs, cluster_df[new_clustering_cols]], axis=1)

    # Use optimal resolution (already computed in process_integration_output)
    cluster_optimal_resolution(
        adata=adata,
        label_key="cell_type",
        cluster_key=cluster_key,
        cluster_function=sc.tl.leiden,
        resolutions=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    )

    # Compute scores
    ari_score = ari(adata, cluster_key=cluster_key, label_key="cell_type")
    nmi_score = nmi(adata, cluster_key=cluster_key, label_key="cell_type")
    ari_batch_score = 1 - ari(adata, cluster_key=cluster_key, label_key="batch")
    nmi_batch_score = 1 - nmi(adata, cluster_key=cluster_key, label_key="batch")

    return {
        'ari': ari_score,
        'nmi': nmi_score,
        'ari_batch': ari_batch_score,
        'nmi_batch': nmi_batch_score
    }

# Metric 5: Graph Connectivity
def compute_graph_connectivity(adata):
    """Exact OpenProblems Graph Connectivity implementation."""
    score = graph_connectivity(
        adata,
        label_key="cell_type"
    )
    return score

# Metric 6: Isolated Label ASW
def compute_isolated_label_asw(adata):
    """Exact OpenProblems Isolated Label ASW implementation."""
    score = isolated_labels_asw(
        adata,
        label_key="cell_type",
        batch_key='batch',
        embed='X_emb',
        iso_threshold=None,
        verbose=True,
    )
    return score

# Metric 7: Isolated Label F1
def compute_isolated_label_f1(adata):
    """Exact OpenProblems Isolated Label F1 implementation."""
    # Get existing clusters (avoid duplicates)
    cluster_df = adata.obsm.get('clustering', pd.DataFrame(index=adata.obs_names))
    existing_leiden_cols = [col for col in adata.obs.columns if 'leiden' in col]
    new_clustering_cols = [col for col in cluster_df.columns if col not in existing_leiden_cols]

    if new_clustering_cols:
        adata.obs = pd.concat([adata.obs, cluster_df[new_clustering_cols]], axis=1)

    score = isolated_labels_f1(
        adata,
        label_key="cell_type",
        batch_key="batch",
        cluster_key="leiden",
        resolutions=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        embed=None,
        iso_threshold=None,
        verbose=True,
    )
    return score

# Metric 8: kBET
def compute_kbet(adata):
    """Exact OpenProblems kBET implementation."""
    score = kBET(
        adata,
        batch_key="batch",
        label_key="cell_type",
        type_="embed",
        embed="X_emb",
        scaled=True,
        verbose=False,
    )
    return score

# Metric 9-10: LISI (iLISI + cLISI)
def compute_lisi(adata):
    """Exact OpenProblems LISI implementation."""

    # iLISI
    ilisi_scores = lisi_graph_py(
        adata=adata,
        obs_key='batch',
        n_neighbors=90,
        perplexity=None,
        subsample=None,
        n_cores=1,
        verbose=False,
    )
    ilisi = np.nanmedian(ilisi_scores)
    ilisi = (ilisi - 1) / (adata.obs['batch'].nunique() - 1)

    # cLISI
    clisi_scores = lisi_graph_py(
        adata=adata,
        obs_key="cell_type",
        n_neighbors=90,
        perplexity=None,
        subsample=None,
        n_cores=1,
        verbose=False,
    )
    clisi = np.nanmedian(clisi_scores)
    nlabs = adata.obs["cell_type"].nunique()
    clisi = (nlabs - clisi) / (nlabs - 1)

    return {'ilisi': ilisi, 'clisi': clisi}

# Metric 11: PCR (Principal Component Regression)
def compute_pcr(adata_original, adata_integrated):
    """Exact OpenProblems PCR implementation."""
    # Copy batch information to integrated data
    adata_integrated.obs['batch'] = adata_original.obs['batch']

    score = pcr_comparison(
        adata_original[:, adata_original.var["batch_hvg"]],  # Original with batch_hvg
        adata_integrated,
        embed='X_emb',
        covariate='batch',
        verbose=True
    )
    return score

# Metric 12: Cell Cycle Conservation
def compute_cell_cycle_conservation(adata_original, adata_integrated):
    """Exact OpenProblems Cell Cycle Conservation implementation."""
    try:
        # CRITICAL: Rebuild AnnData objects cleanly to fix scanpy gene matching issue
        # The original AnnData structure has internal inconsistencies that prevent
        # scanpy from properly recognizing genes even when they exist in var_names

        # Rebuild original data with clean structure
        adata_original_clean = ad.AnnData(
            X=adata_original.X,
            obs=adata_original.obs.copy(),
            var=adata_original.var.copy(),
            uns=adata_original.uns.copy() if adata_original.uns else {}
        )
        # Set clean var_names from feature_name
        adata_original_clean.var_names = pd.Index([str(x) for x in adata_original.var['feature_name']], dtype=object)
        adata_original_clean.var_names.name = None

        # Rebuild integrated data with clean structure
        adata_integrated_clean = ad.AnnData(
            X=adata_integrated.X,
            obs=adata_integrated.obs.copy(),
            var=adata_integrated.var.copy(),
            obsm=adata_integrated.obsm.copy() if adata_integrated.obsm else {},
            uns=adata_integrated.uns.copy() if adata_integrated.uns else {}
        )

        # For integrated data, ensure var_names match original (needed for cell cycle)
        if 'feature_name' in adata_integrated.var.columns:
            adata_integrated_clean.var_names = pd.Index([str(x) for x in adata_integrated.var['feature_name']], dtype=object)
        else:
            # If no feature_name, use current var_names but clean them
            adata_integrated_clean.var_names = pd.Index([str(x) for x in adata_integrated.var_names], dtype=object)
        adata_integrated_clean.var_names.name = None

        # Copy batch information to integrated data
        adata_integrated_clean.obs['batch'] = adata_original_clean.obs['batch']

        # Get organism from dataset annotation (required for datasets)
        organism_raw = adata_original_clean.uns.get('dataset_organism', 'human')

        # Convert different organism formats to scib-compatible format
        if organism_raw == 'mus_musculus':
            organism = 'mouse'
        elif organism_raw == 'homo_sapiens':
            organism = 'human'
        else:
            organism = organism_raw  # Assume 'human' or 'mouse' already correct

        score = cell_cycle(
            adata_original_clean,    # Clean original data
            adata_integrated_clean,  # Clean integrated data
            batch_key='batch',
            embed='X_emb',
            organism=organism,
        )
        return score
    except Exception as e:
        print(f"  Cell cycle failed (method filtered genes): {e}")
        return 0.0  # Score 0 if genes are missing

# Metric 13: HVG Overlap
def compute_hvg_overlap(adata_original, adata_integrated):
    """Memory-efficient HVG Overlap implementation."""
    try:
        # Force garbage collection before starting
        gc.collect()

        # Copy batch information to integrated data
        adata_integrated.obs["batch"] = adata_original.obs["batch"]

        # Ensure integrated data has corrected_counts for HVG overlap function
        if 'corrected_counts' not in adata_integrated.layers:
            print(f"  HVG overlap skipped: No corrected_counts layer in integrated data")
            return 0.0

        # Set X matrix from corrected_counts
        adata_integrated.X = adata_integrated.layers['corrected_counts']

        # Convert to sparse if dense and eliminate zeros
        if adata_integrated.X is not None:
            if not issparse(adata_integrated.X):
                adata_integrated.X = csr_matrix(adata_integrated.X)
            adata_integrated.X.eliminate_zeros()

        # Use memory-efficient implementation that processes batches sequentially
        score = compute_hvg_overlap_memory_efficient(
            adata_original,
            adata_integrated,
            batch_key="batch",
            n_hvg=500,
            verbose=False
        )

        # Force garbage collection after completion
        gc.collect()
        return score
    except Exception as e:
        print(f"  HVG overlap failed: {e}")
        gc.collect()
        return 0.0

def compute_hvg_overlap_memory_efficient(adata_pre, adata_post, batch_key="batch", n_hvg=500, verbose=False):
    """
    Memory-efficient version of HVG overlap metric.

    Processes batches sequentially instead of using split_batches which creates
    all copies at once and causes memory issues with dense matrices.
    """

    # Get list of genes from integrated data
    hvg_post = adata_post.var_names

    # Get unique batches
    batch_categories = adata_post.obs[batch_key].cat.categories

    # Process each batch sequentially to compute HVGs
    overlap_scores = []

    # First pass: compute HVGs for each batch in pre-integration data
    hvg_pre_dict = {}

    for batch_name in batch_categories:
        # Get indices for this batch
        batch_mask_pre = adata_pre.obs[batch_key] == batch_name

        # Create a view (not copy) of the batch data, filtered to hvg_post genes
        adata_batch_pre = adata_pre[batch_mask_pre, adata_pre.var_names.isin(hvg_post)]

        # Filter genes (at least 1 cell)
        sc.pp.filter_genes(adata_batch_pre, min_cells=1)

        # Determine number of HVGs to compute
        n_hvg_tmp = min(n_hvg, int(0.5 * adata_batch_pre.n_vars))

        if n_hvg_tmp < n_hvg and verbose:
            print(f"  Batch {batch_name} has fewer genes ({adata_batch_pre.n_vars}), using {n_hvg_tmp} HVGs")

        # Compute HVGs for this batch
        if adata_batch_pre.n_vars > 0 and n_hvg_tmp > 0:
            hvg_result = sc.pp.highly_variable_genes(
                adata_batch_pre,
                flavor="cell_ranger",
                n_top_genes=n_hvg_tmp,
                inplace=False
            )
            hvg_pre_dict[batch_name] = adata_batch_pre.var.index[hvg_result["highly_variable"]].tolist()
        else:
            hvg_pre_dict[batch_name] = []

        # Clean up
        del adata_batch_pre
        gc.collect()

    # Second pass: compute HVGs for each batch in post-integration data and compare
    for batch_name in batch_categories:
        if batch_name not in hvg_pre_dict or len(hvg_pre_dict[batch_name]) == 0:
            continue

        # Get indices for this batch
        batch_mask_post = adata_post.obs[batch_key] == batch_name

        # Create a view of the batch data
        adata_batch_post = adata_post[batch_mask_post, :]

        # Filter genes (at least 1 cell)
        sc.pp.filter_genes(adata_batch_post, min_cells=1)

        # Use same number of HVGs as pre-integration
        n_hvg_batch = len(hvg_pre_dict[batch_name])

        if verbose:
            print(f"  Batch {batch_name}: computing {n_hvg_batch} HVGs")

        # Compute HVGs for post-integration
        if adata_batch_post.n_vars > 0 and n_hvg_batch > 0:
            hvg_result_post = sc.pp.highly_variable_genes(
                adata_batch_post,
                flavor="cell_ranger",
                n_top_genes=n_hvg_batch,
                inplace=False
            )
            hvg_post_batch = adata_batch_post.var.index[hvg_result_post["highly_variable"]].tolist()

            # Compute overlap
            hvg_pre_set = set(hvg_pre_dict[batch_name])
            hvg_post_set = set(hvg_post_batch)

            n_overlap = len(hvg_pre_set.intersection(hvg_post_set))
            n_hvg_real = min(len(hvg_pre_set), len(hvg_post_set))

            if n_hvg_real > 0:
                overlap = n_overlap / n_hvg_real
                overlap_scores.append(overlap)

                if verbose:
                    print(f"    Overlap: {n_overlap}/{n_hvg_real} = {overlap:.3f}")

        # Clean up
        del adata_batch_post
        gc.collect()

    # Return mean overlap across batches
    if len(overlap_scores) > 0:
        return np.mean(overlap_scores)
    else:
        return 0.0
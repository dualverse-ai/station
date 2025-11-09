#!/usr/bin/env python3
"""
Batch integration evaluation main script.

This script loads the training data, imports the agent's submission,
executes the batch integration function, and saves the result.
"""

# Limit threading BEFORE any imports - CRITICAL for preventing thread explosion!
# These must be set before numpy/scipy/anndata are imported, or they have no effect.
import os
os.environ['OPENBLAS_NUM_THREADS'] = '32'
os.environ['MKL_NUM_THREADS'] = '32'
os.environ['NUMEXPR_NUM_THREADS'] = '32'
os.environ['OMP_NUM_THREADS'] = '32'

# Disable GPU access
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import sys
import uuid
import warnings
import traceback
import anndata as ad
import numpy as np

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def main():
    """Main execution function."""

    # Load training dataset
    print("Loading training dataset...")
    try:
        adata_human = ad.read_h5ad('storage/system/human_train.h5ad')
        print(f"  Data shape: {adata_human.shape}")
        print(f"  Batches: {adata_human.obs['batch'].nunique()} batches")
    except Exception as e:
        print(f"ERROR: Could not load training data: {e}")
        traceback.print_exc()
        return

    # Check for test function first
    try:
        import submission
        if hasattr(submission, 'test'):
            print("=== Test Mode Detected ===")
            print("Running test() function from submission...")
            test_result = submission.test(adata_human.copy())
            print(f"Test result: {test_result}")
            return
    except Exception as e:
        print(f"ERROR: Could not import submission for test check: {e}")
        traceback.print_exc()
        return

    # Import and execute main function
    try:
        eliminate_batch_effect_fn = getattr(submission, 'eliminate_batch_effect_fn')
        print("Found eliminate_batch_effect_fn, executing...")
    except AttributeError:
        print("ERROR: submission.py must implement eliminate_batch_effect_fn(adata)")
        return
    except Exception as e:
        print(f"ERROR: Could not import eliminate_batch_effect_fn: {e}")
        traceback.print_exc()
        return

    # Execute batch integration
    print("\n=== Running batch integration ===")
    adata_input = adata_human.copy()
    adata_result = eliminate_batch_effect_fn(adata_input)

    if not isinstance(adata_result, ad.AnnData):
        print("ERROR: Function must return AnnData object")
        return

    print(f"  Result shape: {adata_result.shape}")

    # Check for required outputs
    has_embedding = 'X_emb' in adata_result.obsm
    has_graph = 'connectivities' in adata_result.obsp and 'distances' in adata_result.obsp

    print(f"  Has embedding: {has_embedding}")
    print(f"  Has graph: {has_graph}")

    if not (has_embedding or has_graph):
        print("ERROR: Result must have at least one of: X_emb or connectivities/distances")
        return

    # Save result
    try:
        result_uuid = str(uuid.uuid4())
        result_dir = f'storage/shared/tmp/{result_uuid}'
        os.makedirs(result_dir, exist_ok=True)

        # Save result
        result_file_human = os.path.join(result_dir, 'result_human.h5ad')
        adata_result.write_h5ad(result_file_human)
        print(f"\nResult saved to: {result_file_human}")

        print(f"\nRESULT_UUID: {result_uuid}")

    except Exception as e:
        print(f"ERROR: Could not save results: {e}")
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()
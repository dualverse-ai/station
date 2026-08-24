#!/usr/bin/env python3
"""Process ZAPBench data with proper condition boundary handling.

This script downloads the official ZAPBench data and creates train/val/test splits
with condition labels to prevent cross-condition contamination.

Requirements:
    pip install git+https://github.com/google-research/zapbench.git
    pip install tensorstore numpy
"""

import os
import numpy as np
import tensorstore as ts
from zapbench import constants
from zapbench import data_utils


def download_raw_data(output_dir='data'):
    """Download raw ZAPBench data if not already cached."""
    os.makedirs(output_dir, exist_ok=True)
    raw_file = os.path.join(output_dir, 'raw_data.npy')

    if os.path.exists(raw_file):
        print(f"Raw data already exists at {raw_file}, skipping download.")
        return np.load(raw_file)

    print("Downloading full ZAPBench traces (all 71,721 neurons)...")

    # Get the full traces dataset using ZAPBench's official method
    spec = data_utils.get_spec(constants.TIMESERIES_NAME)
    volume = ts.open(spec).result()

    print(f"Downloading data shape: {volume.shape}")

    # Read all data
    data = volume.read().result()
    print(f"Downloaded data shape: {data.shape}")

    # Save raw data
    np.save(raw_file, data)
    print(f"Raw data saved to {raw_file}")

    return data


def create_official_splits_with_conditions(raw_data, output_dir='data', context_length=4):
    """Create train/val/test splits with proper condition tracking.

    This creates separate files for each condition's data AND condition labels,
    preventing cross-condition contamination when creating sequences.
    """

    print("\nCreating official train/val/test splits WITH condition tracking...")

    # Initialize data storage
    all_train_data = []
    all_val_data = []
    all_test_data = []

    # Track condition labels for ALL splits (not just test)
    train_condition_labels = []
    val_condition_labels = []
    test_condition_labels = []

    # Process training conditions (all except holdout condition 3)
    for condition in constants.CONDITIONS_TRAIN:
        print(f"\nProcessing condition {condition} ({constants.CONDITION_NAMES[condition]}):")

        # Get condition bounds
        inclusive_min, exclusive_max = data_utils.get_condition_bounds(condition)
        print(f"  Condition bounds: {inclusive_min} to {exclusive_max}")

        # Adjust bounds for train/val/test splits using official method
        train_min, train_max = data_utils.adjust_condition_bounds_for_split(
            'train', inclusive_min, exclusive_max, num_timesteps_context=context_length
        )
        val_min, val_max = data_utils.adjust_condition_bounds_for_split(
            'val', inclusive_min, exclusive_max, num_timesteps_context=context_length
        )
        test_min, test_max = data_utils.adjust_condition_bounds_for_split(
            'test', inclusive_min, exclusive_max, num_timesteps_context=context_length
        )

        print(f"  Train: {train_min} to {train_max} ({train_max - train_min} timesteps)")
        print(f"  Val: {val_min} to {val_max} ({val_max - val_min} timesteps)")
        print(f"  Test: {test_min} to {test_max} ({test_max - test_min} timesteps)")

        # Extract data for this condition
        if train_max > train_min:
            train_data = raw_data[train_min:train_max, :]
            all_train_data.append(train_data)
            # Create condition labels for train data
            condition_labels = np.full(train_max - train_min, condition, dtype=np.int32)
            train_condition_labels.append(condition_labels)

        if val_max > val_min:
            val_data = raw_data[val_min:val_max, :]
            all_val_data.append(val_data)
            # Create condition labels for val data
            condition_labels = np.full(val_max - val_min, condition, dtype=np.int32)
            val_condition_labels.append(condition_labels)

        if test_max > test_min:
            test_data = raw_data[test_min:test_max, :]
            all_test_data.append(test_data)
            # Create condition labels for test data
            condition_labels = np.full(test_max - test_min, condition, dtype=np.int32)
            test_condition_labels.append(condition_labels)

    # Process holdout condition (condition 3 - taxis) for additional test data
    holdout_condition = constants.CONDITIONS_HOLDOUT[0]  # condition 3
    print(f"\nProcessing holdout condition {holdout_condition} ({constants.CONDITION_NAMES[holdout_condition]}):")

    inclusive_min, exclusive_max = data_utils.get_condition_bounds(holdout_condition)
    print(f"  Condition bounds: {inclusive_min} to {exclusive_max}")

    # Use 'test_holdout' split for the holdout condition
    holdout_min, holdout_max = data_utils.adjust_condition_bounds_for_split(
        'test_holdout', inclusive_min, exclusive_max, num_timesteps_context=context_length
    )
    print(f"  Holdout test: {holdout_min} to {holdout_max} ({holdout_max - holdout_min} timesteps)")

    if holdout_max > holdout_min:
        holdout_data = raw_data[holdout_min:holdout_max, :]
        all_test_data.append(holdout_data)
        # Create condition labels for holdout test data
        holdout_condition_labels = np.full(holdout_max - holdout_min, holdout_condition, dtype=np.int32)
        test_condition_labels.append(holdout_condition_labels)

    # Concatenate all splits
    train_data = np.concatenate(all_train_data, axis=0) if all_train_data else np.array([])
    val_data = np.concatenate(all_val_data, axis=0) if all_val_data else np.array([])
    test_data = np.concatenate(all_test_data, axis=0) if all_test_data else np.array([])

    train_labels = np.concatenate(train_condition_labels, axis=0) if train_condition_labels else np.array([])
    val_labels = np.concatenate(val_condition_labels, axis=0) if val_condition_labels else np.array([])
    test_labels = np.concatenate(test_condition_labels, axis=0) if test_condition_labels else np.array([])

    print(f"\nFinal split shapes:")
    print(f"  Train: {train_data.shape} (timesteps × neurons)")
    print(f"  Val: {val_data.shape}")
    print(f"  Test: {test_data.shape}")
    print(f"  Train condition labels: {train_labels.shape}")
    print(f"  Val condition labels: {val_labels.shape}")
    print(f"  Test condition labels: {test_labels.shape}")

    # Save concatenated splits (for backward compatibility)
    train_path = os.path.join(output_dir, 'train_data.npy')
    val_path = os.path.join(output_dir, 'val_data.npy')
    test_data_path = os.path.join(output_dir, 'test_data.npy')

    # Save condition labels for ALL splits
    train_labels_path = os.path.join(output_dir, 'train_condition_labels.npy')
    val_labels_path = os.path.join(output_dir, 'val_condition_labels.npy')
    test_labels_path = os.path.join(output_dir, 'test_condition_labels.npy')

    np.save(train_path, train_data)
    np.save(val_path, val_data)
    np.save(test_data_path, test_data)
    np.save(train_labels_path, train_labels)
    np.save(val_labels_path, val_labels)
    np.save(test_labels_path, test_labels)

    print(f"\nConditioning-aware data saved:")
    print(f"  Train: {train_path}")
    print(f"  Val: {val_path}")
    print(f"  Test: {test_data_path}")
    print(f"  Train labels: {train_labels_path}")
    print(f"  Val labels: {val_labels_path}")
    print(f"  Test labels: {test_labels_path}")

    # Print condition breakdown for all splits
    for split_name, labels in [('Train', train_labels), ('Val', val_labels), ('Test', test_labels)]:
        if len(labels) > 0:
            unique_conditions, counts = np.unique(labels, return_counts=True)
            print(f"\n{split_name} condition breakdown:")
            for cond, count in zip(unique_conditions, counts):
                print(f"  Condition {cond} ({constants.CONDITION_NAMES[cond]}): {count} timesteps")

    return {
        'train': train_data,
        'val': val_data,
        'test': test_data,
        'train_labels': train_labels,
        'val_labels': val_labels,
        'test_labels': test_labels
    }


def main():
    """Main processing function."""
    import argparse
    parser = argparse.ArgumentParser(description='Process ZAPBench data with proper condition handling')
    parser.add_argument('--output-dir', type=str, default='./data',
                        help='Output directory for processed data (default: ./data)')
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Step 1: Download raw data (or load if cached)
    raw_data = download_raw_data(output_dir)

    # Step 2: Create official train/val/test splits WITH condition tracking
    splits = create_official_splits_with_conditions(raw_data, output_dir)

    print(f"\n{'='*60}")
    print(f"DATA PROCESSING COMPLETE!")
    print(f"{'='*60}")
    print(f"Raw data: {raw_data.shape}")
    print(f"Train: {splits['train'].shape} with labels: {splits['train_labels'].shape}")
    print(f"Val: {splits['val'].shape} with labels: {splits['val_labels'].shape}")
    print(f"Test: {splits['test'].shape} with labels: {splits['test_labels'].shape}")
    print(f"\n{'='*60}")
    print(f"CRITICAL FILES CREATED:")
    print(f"{'='*60}")
    print(f"Main data files:")
    print(f"  - {output_dir}/train_data.npy")
    print(f"  - {output_dir}/val_data.npy")
    print(f"  - {output_dir}/test_data.npy")
    print(f"\nCondition label files (ESSENTIAL for preventing cross-condition contamination):")
    print(f"  - {output_dir}/train_condition_labels.npy")
    print(f"  - {output_dir}/val_condition_labels.npy")
    print(f"  - {output_dir}/test_condition_labels.npy")
    print(f"\n{'='*60}")
    print(f"USAGE INSTRUCTIONS:")
    print(f"{'='*60}")
    print(f"When creating sequences, use the condition labels to ensure:")
    print(f"1. No sequence spans across condition boundaries")
    print(f"2. Each sequence comes from a single experimental condition")
    print(f"3. This prevents invalid cross-condition predictions")


if __name__ == "__main__":
    main()
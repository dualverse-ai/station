# station_data/rooms/research/evaluators/task_1_evaluator.py
"""
Evaluator for Research Task 1: Single-cell RNA-seq Batch Integration
Implements exact OpenProblems 2.0 scoring methodology
"""

import os
import sys
import re
import shutil
import numpy as np
import anndata as ad
import tempfile
import traceback
import json
import scanpy as sc
from pathlib import Path
from typing import Tuple, Optional
import contextlib
import io

from station.eval_research.base_evaluator import ResearchTaskEvaluator
from station import constants

# Import wrapper functions
import sys
sys.path.append(str(Path(__file__).parent.parent / "storage" / "system"))
from openproblems_metrics import *

class Task1Evaluator(ResearchTaskEvaluator):
    def __init__(self):
        super().__init__("1")
        self.timeout = 30 * 60  # 30 minutes
        self.datasets = None
        self._cleanup_orphaned_tmp_dirs()

    def _cleanup_orphaned_tmp_dirs(self):
        """Remove any orphaned tmp directories from previous failed evaluations."""
        try:
            research_room_path = os.path.join(
                constants.BASE_STATION_DATA_PATH,
                constants.ROOMS_DIR_NAME,
                constants.SHORT_ROOM_NAME_RESEARCH
            )
            tmp_dir = os.path.join(
                research_room_path,
                constants.RESEARCH_STORAGE_DIR,
                constants.RESEARCH_STORAGE_SHARED_DIR,
                'tmp'
            )

            if os.path.exists(tmp_dir):
                # Remove all subdirectories in tmp/
                removed_count = 0
                removed_size = 0
                for item in os.listdir(tmp_dir):
                    item_path = os.path.join(tmp_dir, item)
                    if os.path.isdir(item_path):
                        # Calculate size before removing
                        try:
                            for dirpath, dirnames, filenames in os.walk(item_path):
                                for filename in filenames:
                                    filepath = os.path.join(dirpath, filename)
                                    removed_size += os.path.getsize(filepath)
                        except:
                            pass

                        shutil.rmtree(item_path)
                        removed_count += 1

                if removed_count > 0:
                    size_mb = removed_size / (1024 * 1024)
                    print(f"Task1Evaluator: Cleaned up {removed_count} orphaned tmp directories ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"Task1Evaluator: Warning - could not clean orphaned tmp dirs: {e}")

    def get_expected_function_name(self) -> str:
        return "eliminate_batch_effect_fn"

    def get_task_description(self) -> str:
        return "Single-cell RNA-seq Batch Integration"

    def get_secondary_metrics_format(self):
        """All 10 OpenProblems metrics as secondary metrics."""
        return {
            "ASW_batch": ".3f", "ASW_label": ".3f", "ARI": ".3f", "NMI": ".3f",
            "Graph_conn": ".3f", "kBET": ".3f", "iLISI": ".3f", "cLISI": ".3f",
            "PCR": ".3f", "Cell_cycle": ".3f"
        }

    def _load_datasets(self):
        """Load human dataset (train + solution files)."""
        if self.datasets is not None:
            return self.datasets

        system_path = Path(__file__).parent.parent / "storage" / "system"

        # Load human dataset files
        human_train_path = system_path / "human_train.h5ad"
        human_solution_path = system_path / "human_train_solution.h5ad"

        if not human_train_path.exists():
            raise FileNotFoundError(f"Human training data not found: {human_train_path}")
        if not human_solution_path.exists():
            raise FileNotFoundError(f"Human solution data not found: {human_solution_path}")

        self.datasets = {
            'human': {
                'train': ad.read_h5ad(human_train_path),
                'solution': ad.read_h5ad(human_solution_path)
            }
        }

        return self.datasets

    def _load_baselines(self):
        """Load OpenProblems baselines for normalization."""
        baseline_path = Path(__file__).parent.parent / "storage" / "system" / "openproblems_baselines.json"
        with open(baseline_path) as f:
            return json.load(f)


    def _normalize_metric(self, raw_score, metric_name, baselines):
        """Normalize metric: (raw_score - min) / (max - min), clamped to [0,1]."""
        if metric_name not in baselines or np.isnan(raw_score):
            return 0.0  # NaN values replaced by 0 as per OpenProblems

        min_val = baselines[metric_name]["min"]
        max_val = baselines[metric_name]["max"]

        if max_val == min_val:
            return 1.0

        # Linear scaling and clamping to [0,1]
        normalized = (raw_score - min_val) / (max_val - min_val)
        return max(0.0, min(1.0, normalized))

    def get_execution_mode(self) -> str:
        """Use command mode for this task."""
        return "command"

    def get_execution_command(self) -> str:
        """
        Return the command to execute for this task.
        The submission code will be saved as submission.py in the working directory.
        """
        # Use storage-relative path that works in both Docker and Python sandbox modes
        return "python storage/system/main.py"

    @contextlib.contextmanager
    def _suppress_stdout(self):
        """Context manager to suppress stdout output while keeping stderr."""
        old_stdout = sys.stdout
        try:
            sys.stdout = io.StringIO()
            yield
        finally:
            sys.stdout = old_stdout

    def _compute_metrics(self, adata_result, adata_original, adata_solution, dataset_name):
        """Compute all 10 OpenProblems metrics with normalization for a single dataset."""
        try:
            baselines = self._load_baselines()

            # Step 1: Process integration output (following OpenProblems workflow)
            print(f"Processing integration output for {dataset_name}...")
            with self._suppress_stdout():
                adata_processed = process_integration_output(adata_result, adata_original)

            # Inject cell_type from solution file (removed from training to prevent leakage)
            adata_processed.obs['cell_type'] = adata_solution.obs['cell_type']

            # Compute all metrics (following eval.py approach)
            results = {}

            # Single-dataset metrics
            metric_functions = [
                ('asw_batch', compute_asw_batch),
                ('asw_label', compute_asw_label),
                ('graph_connectivity', compute_graph_connectivity),
                ('kbet', compute_kbet),
            ]

            for metric_name, metric_func in metric_functions:
                try:
                    with self._suppress_stdout():
                        raw_score = metric_func(adata_processed)
                    results[metric_name] = raw_score
                except Exception as e:
                    print(f"Error computing {metric_name}: {e}")
                    results[metric_name] = np.nan

            # Compound metrics
            try:
                with self._suppress_stdout():
                    clustering_scores = compute_clustering_overlap(adata_processed)
                results.update(clustering_scores)
            except Exception as e:
                print(f"Error computing clustering metrics: {e}")
                results.update({'ari': np.nan, 'nmi': np.nan, 'ari_batch': np.nan, 'nmi_batch': np.nan})

            try:
                with self._suppress_stdout():
                    lisi_scores = compute_lisi(adata_processed)
                results.update(lisi_scores)
            except Exception as e:
                print(f"Error computing LISI metrics: {e}")
                results.update({'ilisi': np.nan, 'clisi': np.nan})

            # Two-dataset metrics (all use solution file per OpenProblems 2.0)
            dual_metrics = [
                ('pcr', compute_pcr, 'solution'),                    # Needs batch HVGs from solution
                ('cell_cycle_conservation', compute_cell_cycle_conservation, 'solution'),  # Needs all genes
            ]

            for metric_name, metric_func, data_type in dual_metrics:
                try:
                    with self._suppress_stdout():
                        # All dual metrics use solution file per OpenProblems 2.0 architecture
                        raw_score = metric_func(adata_solution, adata_processed)
                    results[metric_name] = raw_score
                    print(f"   {metric_name} (using solution with {adata_solution.shape[1]} genes): {raw_score:.6f}")
                except Exception as e:
                    print(f"Error computing {metric_name}: {e}")
                    results[metric_name] = np.nan

            # Target 10 metrics for normalization
            target_metrics = [
                'asw_batch', 'asw_label', 'ari', 'nmi', 'graph_connectivity',
                'kbet', 'ilisi', 'clisi', 'pcr', 'cell_cycle_conservation'
            ]

            # Normalize scores
            normalized_scores = {}
            raw_scores = {}
            for metric_name in target_metrics:
                if metric_name in results:
                    raw_score = results[metric_name]
                    raw_scores[metric_name] = raw_score
                    normalized_scores[metric_name] = self._normalize_metric(raw_score, metric_name, baselines)
                else:
                    raw_scores[metric_name] = np.nan
                    normalized_scores[metric_name] = 0.0

            # Display format (matching eval.py)
            display_metrics = {
                "ASW_batch": normalized_scores['asw_batch'],
                "ASW_label": normalized_scores['asw_label'],
                "ARI": normalized_scores['ari'],
                "NMI": normalized_scores['nmi'],
                "Graph_conn": normalized_scores['graph_connectivity'],
                "kBET": normalized_scores['kbet'],
                "iLISI": normalized_scores['ilisi'],
                "cLISI": normalized_scores['clisi'],
                "PCR": normalized_scores['pcr'],
                "Cell_cycle": normalized_scores['cell_cycle_conservation']
            }

            # Store raw scores for display in message
            display_metrics['_raw_scores'] = raw_scores

            return display_metrics

        except Exception as e:
            print(f"Error computing metrics for {dataset_name}: {e}")
            return {"error": str(e)}

    def _format_metrics_line(self, metrics_dict, include_raw=False):
        """Format metrics dict into readable lines with proper precision."""
        lines = []

        # Normalized scores (always show)
        lines.append("Normalized Scores [0-1]:")
        # First line: batch correction metrics
        line1 = f"  ASW_batch: {metrics_dict['ASW_batch']:.3f}  ASW_label: {metrics_dict['ASW_label']:.3f}  ARI: {metrics_dict['ARI']:.3f}  NMI: {metrics_dict['NMI']:.3f}"
        lines.append(line1)
        # Second line: mixing/connectivity metrics
        line2 = f"  Graph_conn: {metrics_dict['Graph_conn']:.3f}  kBET: {metrics_dict['kBET']:.3f}  iLISI: {metrics_dict['iLISI']:.3f}  cLISI: {metrics_dict['cLISI']:.3f}"
        lines.append(line2)
        # Third line: bio conservation metrics
        line3 = f"  PCR: {metrics_dict['PCR']:.3f}  Cell_cycle: {metrics_dict['Cell_cycle']:.3f}"
        lines.append(line3)

        # Raw scores (if available)
        if include_raw and '_raw_scores' in metrics_dict:
            raw = metrics_dict['_raw_scores']
            lines.append("\nRaw Scores (before normalization):")
            lines.append(f"  ASW_batch: {raw['asw_batch']:.6f}  ASW_label: {raw['asw_label']:.6f}  ARI: {raw['ari']:.6f}  NMI: {raw['nmi']:.6f}")
            lines.append(f"  Graph_conn: {raw['graph_connectivity']:.6f}  kBET: {raw['kbet']:.6f}  iLISI: {raw['ilisi']:.6f}  cLISI: {raw['clisi']:.6f}")
            lines.append(f"  PCR: {raw['pcr']:.6f}  Cell_cycle: {raw['cell_cycle_conservation']:.6f}")

        return "\n".join(lines)

    def evaluate_submission(self, result=None, eval_id=None, author=None):
        """Evaluate batch integration submission on human dataset."""

        # Disable GPU access
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

        if result is None:
            return False, 'n.a.', "No output received from main script", (float('-inf'),)

        result_dir = None  # Track for cleanup

        try:
            output_str = str(result)

            # Check if this is test mode - if so, don't attempt result loading
            if "=== Test Mode Detected ===" in output_str:
                return False, 'n.a.', "Test mode - no scoring", (float('-inf'),)

            # Look for the RESULT_UUID line
            uuid_match = re.search(r'RESULT_UUID:\s*([a-f0-9-]+)', output_str)
            if not uuid_match:
                # Check for error messages
                if "ERROR:" in output_str:
                    error_lines = [line.strip() for line in output_str.split('\n') if line.strip().startswith('ERROR:')]
                    error_msg = error_lines[0] if error_lines else "Unknown error during execution"
                    return False, 'n.a.', error_msg, (float('-inf'),)
                else:
                    return False, 'n.a.', "Could not parse RESULT_UUID from output", (float('-inf'),)

            result_uuid = uuid_match.group(1)

            # Load result from temp directory using absolute path
            research_room_path = os.path.join(
                constants.BASE_STATION_DATA_PATH,
                constants.ROOMS_DIR_NAME,
                constants.SHORT_ROOM_NAME_RESEARCH
            )
            result_dir = os.path.join(
                research_room_path,
                constants.RESEARCH_STORAGE_DIR,
                constants.RESEARCH_STORAGE_SHARED_DIR,
                'tmp',
                result_uuid
            )

            # Load result file
            result_file_human = os.path.join(result_dir, 'result_human.h5ad')

            if not os.path.exists(result_file_human):
                return False, 'n.a.', f"Result file not found: {result_file_human}", (float('-inf'),)

            adata_result_human = ad.read_h5ad(result_file_human)

            # Load dataset
            datasets = self._load_datasets()

            # Validate output
            if not isinstance(adata_result_human, ad.AnnData):
                return False, 0, "Output must be AnnData object", (float('-inf'),)

            has_embedding = 'X_emb' in adata_result_human.obsm
            has_graph = ('connectivities' in adata_result_human.obsp and 'distances' in adata_result_human.obsp)

            if not (has_embedding or has_graph):
                return False, 0, "Result must provide X_emb or graph (connectivities/distances)", (float('-inf'),)

            # Evaluate on human dataset
            print("\n=== Evaluating on Human Dataset ===")
            human_metrics = self._compute_metrics(
                adata_result_human,
                datasets['human']['train'],
                datasets['human']['solution'],
                'human'
            )

            # Check for errors
            if 'error' in human_metrics:
                error_msg = human_metrics.get('error')
                return False, 'n.a.', f"Metric computation failed: {error_msg}", (float('-inf'),)

            # Extract metrics (exclude _raw_scores metadata)
            metric_keys = [k for k in human_metrics.keys() if k != '_raw_scores']
            final_metrics = {}
            for key in metric_keys:
                final_metrics[key] = human_metrics[key]

            # Primary score: average of all 10 metrics
            overall_score = np.mean(list(final_metrics.values()))

            # Format evaluation details
            human_n_cells = datasets['human']['train'].shape[0]

            eval_details = f"Human Dataset ({human_n_cells:,} cells):\n"
            eval_details += self._format_metrics_line(human_metrics, include_raw=True)

            # Combine into final metrics dict
            final_metrics['Message'] = eval_details

            return True, overall_score, final_metrics, (overall_score,)

        except Exception as e:
            traceback.print_exc()
            return False, 0, f"Evaluation failed: {str(e)}", (float('-inf'),)

        finally:
            # Clean up temp directory (ALWAYS runs, even on failure)
            if result_dir and os.path.exists(result_dir):
                try:
                    shutil.rmtree(result_dir)
                    print(f"Cleaned up temp directory: {result_dir}")
                except Exception as cleanup_error:
                    print(f"Warning: Could not clean up temp directory {result_dir}: {cleanup_error}")
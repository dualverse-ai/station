# station_data/rooms/research/evaluators/task_1_evaluator.py
"""
Evaluator for Research Task 1: Circle Packing in Unit Square
"""

import os
import sys
import numpy as np
from typing import Tuple, Optional

from station.eval_research.base_evaluator import ResearchTaskEvaluator
from station import constants
from station import file_io_utils


class Task1Evaluator(ResearchTaskEvaluator):
    """
    Evaluator for Research Task 1: Circle Packing in Unit Square
    """
    
    def __init__(self):
        super().__init__("1")
    
    def evaluate_submission(self, result: tuple, eval_id: str = None, author: str = None) -> Tuple[bool, float, str]:
        """
        Evaluate circle packing submission using the verification function from task spec.
        """
        try:
            # Call the verification function
            score = self._evaluate(result)
            
            # Save successful configuration to internal storage
            save_result = self._save_successful_config(result, score, eval_id, author)
            
            return True, score, f"Valid circle packing configuration; sum of all circle radii: {score:.6f}"
            
        except AssertionError as e:
            return False, 0.0, f"Verification failed: {str(e)}"
        except Exception as e:
            return False, 0.0, f"Evaluation error: {str(e)}"
    
    def get_expected_function_name(self) -> str:
        return "construct_packing"
    
    def get_task_description(self) -> str:
        return "Circle Packing in Unit Square"
    
    def _evaluate(self, result: tuple) -> float:
        """Vectorized verification function for Circle Packing in Unit Square."""
        # Unpack the result tuple
        centers, radii = result
        
        # Ensure they are numpy arrays
        if not isinstance(centers, np.ndarray):
            centers = np.array(centers)
        if not isinstance(radii, np.ndarray):
            radii = np.array(radii)
        
        n = centers.shape[0]
        
        # Check that we have exactly 26 circles
        if n != 26:
            raise AssertionError(f"Must pack exactly 26 circles, got {n}")
        
        # Validate shape
        if centers.shape != (26, 2):
            raise AssertionError(f"Centers must have shape (26, 2), got {centers.shape}")
        if radii.shape != (26,):
            raise AssertionError(f"Radii must have shape (26,), got {radii.shape}")
        
        # Check for NaN or infinite values
        if np.any(np.isnan(centers)) or np.any(np.isinf(centers)):
            raise AssertionError("Centers contain NaN or infinite values")
        if np.any(np.isnan(radii)) or np.any(np.isinf(radii)):
            raise AssertionError("Radii contain NaN or infinite values")
        
        # Check that all radii are positive
        if np.any(radii <= 0):
            raise AssertionError(f"All radii must be positive, found: {radii[radii <= 0]}")
        
        # Check if circles are inside the unit square
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1 :
                raise AssertionError(f"Circle {i} at ({x}, {y}) with radius {r} is outside the unit square")
        
        # Check for overlaps between circles
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                if dist < radii[i] + radii[j]:
                    raise AssertionError(f"Circles {i} and {j} overlap: dist={dist}, r1+r2={radii[i]+radii[j]}")
        
        # If all validations pass, return the sum of radii
        return float(np.sum(radii))
    
    def validate_submission_code(self, content: str, author: str, agent_module) -> Tuple[bool, Optional[str]]:
        """
        Validate submitted code before execution.
        
        Args:
            content: The submitted code content
            author: The author of the submission
            agent_module: Module for loading agent data
            
        Returns:
            Tuple of (is_valid, error_message)
            - is_valid: True if validation passes, False if violations detected
            - error_message: Description of violation if found, None otherwise
        """
    
        return True, None
    
    def _save_successful_config(self, result: tuple, score: float, eval_id: str = None, author: str = None) -> str:
        """Save successful configuration to internal/packings directory."""
        try:
            centers, radii = result
            
            # Create internal/packings directory path
            research_room_path = os.path.join(
                constants.BASE_STATION_DATA_PATH,
                constants.ROOMS_DIR_NAME,
                constants.SHORT_ROOM_NAME_RESEARCH
            )
            packings_dir = os.path.join(
                research_room_path,
                constants.RESEARCH_INTERNAL_DIR,
                "packings"
            )
            
            # Ensure directory exists
            file_io_utils.ensure_dir_exists(packings_dir)
            
            # Create filename with format: {author}_{score:.6f}_{eval_id}.npz
            if author and eval_id:
                author_clean = author.replace(" ", "_")
                filename = f"{author_clean}_{score:.6f}_{eval_id}.npz"
            elif eval_id:
                filename = f"unknown_{score:.6f}_{eval_id}.npz"
            else:
                filename = f"unknown_{score:.6f}_legacy.npz"
            
            filepath = os.path.join(packings_dir, filename)
            
            # Save the configuration using numpy's savez
            np.savez(filepath, centers=centers, radii=radii, score=score)
            print(f"Task1Evaluator: Saved packing configuration (score={score:.6f}) to {filepath}")
            return "saved"
            
        except Exception as e:
            print(f"Task1Evaluator: Failed to save configuration: {e}")
            return "error"
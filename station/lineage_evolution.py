#!/usr/bin/env python3

# Copyright 2025 DualverseAI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Lineage Evolution Manager for Station Agent Ascension

This module implements the fitness-based lineage selection system that replaces
random selection during agent ascension. It integrates directly with station.py
to provide both default (random) and evolution (fitness-based) selection modes.

The evolution mode uses utility scores based on:
- Research breakthroughs (new SOTA scores)
- High-quality archive papers (score >= 8.0)
- Total lineage lifespan (as a penalty to prevent stagnation)
"""

import os
import json
import random
import math
from typing import Dict, List, Optional, Tuple, Set, Any
from collections import defaultdict

from station import agent as agent_module
from station import constants
from station import file_io_utils


def get_score_and_success_from_evaluation(eval_data: Dict[str, Any]) -> Tuple[float, bool, Optional[Tuple]]:
    """
    Extract score, success, and sort_key from evaluation data using evaluation manager logic.
    
    Returns:
        Tuple of (score, success, sort_key) where score is float, success is bool, and sort_key is optional tuple
    """
    try:
        # Use the same logic as evaluation manager for display score
        notification = eval_data.get("notification", {})
        
        # Determine display score based on notification status
        if notification.get("sent", False):
            # Show the version that was notified
            version_notified = notification.get("version_notified", "original")
            if version_notified == "original":
                result = eval_data["original_submission"]["evaluation_result"]
                score = result.get("score", "n.a.")
                success = result.get("success", False)
                sort_key = result.get("sort_key")
            else:
                result = eval_data["versions"][version_notified]["evaluation_result"]
                score = result.get("score", "n.a.")
                success = result.get("success", False)
                sort_key = result.get("sort_key")
        else:
            # Not notified yet - check original submission
            if "original_submission" in eval_data and "evaluation_result" in eval_data["original_submission"]:
                result = eval_data["original_submission"]["evaluation_result"]
                score = result.get("score", "n.a.")
                success = result.get("success", False)
                sort_key = result.get("sort_key")
            else:
                score = "n.a."
                success = False
                sort_key = None
        
        # Convert to numeric score
        if success and score != "n.a." and score != "pending":
            try:
                return float(score), True, sort_key
            except (TypeError, ValueError):
                return 0.0, False, None
        return 0.0, False, None
        
    except Exception as e:
        return 0.0, False, None


class LineageEvolutionManager:
    """
    Manages lineage selection for ascending agents using either default (random) 
    or evolution (fitness-based) selection modes.
    
    This manager integrates with the Station's existing ascension flow by replacing
    the random selection in _scan_for_potential_ancestor with a configurable selection
    strategy based on lineage fitness scores.
    """
    
    def scan_for_potential_ancestor(self, guest_name: str) -> Optional[str]:
        """
        Find potential ancestor for a guest agent's ascension.
        This replaces the _scan_for_potential_ancestor method in station.py.
        
        Args:
            guest_name: Name of the guest agent
            
        Returns:
            Ancestor agent name if found, None if agent should create new lineage
        """
        # Clear cache to ensure fresh data
        self._clear_cache()
        
        guest_data = self.agent_module.load_agent_data(guest_name)
        if not guest_data:
            print(f"Warning: Could not load data for guest {guest_name}")
            return None
        guest_model = guest_data.get(constants.AGENT_MODEL_NAME_KEY)
        
        if not guest_model:
            print(f"Warning: Guest {guest_name} has no model name")
            return None
        
        # Check if ancestor was already assigned
        assigned_ancestor = guest_data.get(constants.AGENT_ASSIGNED_ANCESTOR_KEY, "")
        
        # Build set of excluded ancestors
        excluded_ancestors = set()
        
        # Get all agent files
        agents_dir = os.path.join(constants.BASE_STATION_DATA_PATH, constants.AGENTS_DIR_NAME)
        agent_files = file_io_utils.list_files(agents_dir, constants.YAML_EXTENSION)
        
        # Add already succeeded agents
        for agent_file_name in agent_files:
            agent_name = agent_file_name.replace(constants.YAML_EXTENSION, "")
            agent_data = self.agent_module.load_agent_data(agent_name, include_ascended=True, include_ended=True)
            if agent_data and agent_data.get(constants.AGENT_IS_ASCENDED_KEY):
                excluded_ancestors.add(agent_name)
        
        # Add already assigned ancestors (but not for current guest)
        for agent_file_name in agent_files:
            agent_name = agent_file_name.replace(constants.YAML_EXTENSION, "")
            agent_data = self.agent_module.load_agent_data(agent_name)
            if agent_data and agent_data.get(constants.AGENT_STATUS_KEY) == constants.AGENT_STATUS_GUEST:
                ancestor = agent_data.get(constants.AGENT_POTENTIAL_ANCESTOR_NAME_KEY)
                if ancestor and agent_name != guest_name:
                    excluded_ancestors.add(ancestor)
        
        # Use select_lineage for the actual selection
        return self.select_lineage(
            guest_model_name=guest_model,
            excluded_ancestors=excluded_ancestors,
            assigned_ancestor=assigned_ancestor
        )
    
    def get_inheritable_ancestors(self, guest_model_name: str, excluded_ancestors: Set[str], bypass_model_check: bool = False) -> List[Tuple[str, str]]:
        """
        Get all ancestors that can be inherited by an agent with the given model.
        
        Args:
            guest_model_name: Exact model name of the ascending agent
            excluded_ancestors: Set of ancestor names already assigned or succeeded
            bypass_model_check: If True, allows cross-model inheritance (for assigned_ancestor)
            
        Returns:
            List of (ancestor_agent_name, lineage_name) tuples
        """
        inheritable = []
        
        # Get all agent files
        agents_dir = os.path.join(constants.BASE_STATION_DATA_PATH, constants.AGENTS_DIR_NAME)
        agent_files = file_io_utils.list_files(agents_dir, constants.YAML_EXTENSION)
        
        for agent_file_name in agent_files:
            agent_name = agent_file_name.replace(constants.YAML_EXTENSION, "")
            # Load with include_ended=True, but not include_ascended (we don't inherit from guests)
            agent_data = self.agent_module.load_agent_data(agent_name, include_ascended=False, include_ended=True)
            
            if (agent_data and 
                agent_data.get(constants.AGENT_STATUS_KEY) == constants.AGENT_STATUS_RECURSIVE and
                agent_data.get(constants.AGENT_SESSION_ENDED_KEY) and
                (bypass_model_check or agent_data.get(constants.AGENT_MODEL_NAME_KEY) == guest_model_name) and
                not agent_data.get(constants.AGENT_SUCCEEDED_BY_KEY) and
                agent_name not in excluded_ancestors):
                
                lineage = agent_data.get(constants.AGENT_LINEAGE_KEY)
                if lineage:
                    inheritable.append((agent_name, lineage))
                    
        return inheritable
    
    def __init__(self, agent_module):
        """
        Initialize manager with agent module.
        
        Args:
            agent_module: The station's agent module
        """
        self.agent_module = agent_module
        
        # Set up paths for data analysis
        self.evaluations_dir = os.path.join(
            constants.BASE_STATION_DATA_PATH,
            constants.ROOMS_DIR_NAME,
            constants.SHORT_ROOM_NAME_RESEARCH,
            constants.RESEARCH_EVALUATIONS_SUBDIR_NAME
        )
        self.archive_evaluations_dir = os.path.join(
            constants.BASE_STATION_DATA_PATH,
            constants.ROOMS_DIR_NAME,
            constants.SHORT_ROOM_NAME_ARCHIVE,
            "evaluations"
        )
        
        # Cache for utility scores to avoid recomputation
        self._utility_cache = {}
        self._cache_loaded = False
        self._breakthrough_data = None
        self._paper_data = None
        self._lifespan_data = None
    
    def _clear_cache(self):
        """Clear all cached data."""
        self._utility_cache = {}
        self._cache_loaded = False
        self._breakthrough_data = None
        self._paper_data = None
        self._lifespan_data = None
    
    def _load_all_data(self):
        """Load all evaluation data once for efficiency."""
        if self._cache_loaded:
            return
            
        # Load breakthrough data
        self._breakthrough_data = self._load_all_breakthroughs()
        
        # Load paper data
        self._paper_data = self._load_all_papers()
        
        # Load lifespan data
        self._lifespan_data = self._load_all_lifespans()
        
        self._cache_loaded = True
    
    def _load_all_breakthroughs(self) -> Dict[str, int]:
        """Load and compute all breakthrough data at once."""
        lineage_breakthroughs = defaultdict(int)
        
        # Build list of all evaluations
        all_evaluations = []
        
        if os.path.exists(self.evaluations_dir):
            for filename in os.listdir(self.evaluations_dir):
                if filename.startswith('evaluation_') and filename.endswith('.json'):
                    try:
                        eval_id = int(filename.split('_')[1].split('.')[0])
                        filepath = os.path.join(self.evaluations_dir, filename)
                        with open(filepath, 'r') as f:
                            data = json.load(f)
                        
                        author = data.get('author', '')
                        if not author:
                            continue
                            
                        score, success, sort_key = get_score_and_success_from_evaluation(data)
                        if not success:
                            continue
                            
                        author_lineage = self._extract_lineage_from_agent_name(author)
                        if author_lineage:
                            all_evaluations.append((eval_id, author_lineage, score, sort_key))
                            
                    except Exception:
                        continue
        
        # Sort by evaluation ID
        all_evaluations.sort(key=lambda x: x[0])
        
        # Track breakthroughs using sort_keys when available
        current_sota_key = None
        for eval_data in all_evaluations:
            eval_id, author_lineage, score = eval_data[:3]
            sort_key = eval_data[3] if len(eval_data) > 3 else None
            
            # Use sort_key if available, otherwise fall back to score
            if sort_key is not None:
                comparison_key = tuple(sort_key) if isinstance(sort_key, (list, tuple)) else (sort_key,)
            else:
                comparison_key = (score,)
            
            # Check if this is a new SOTA
            if current_sota_key is None or comparison_key > current_sota_key:
                lineage_breakthroughs[author_lineage] += 1
                current_sota_key = comparison_key
                
        return dict(lineage_breakthroughs)
    
    def _load_all_papers(self) -> Dict[str, int]:
        """Load and compute all high-quality paper counts at once."""
        lineage_papers = defaultdict(int)
        
        if os.path.exists(self.archive_evaluations_dir):
            for filename in os.listdir(self.archive_evaluations_dir):
                if filename.endswith('.yaml'):
                    try:
                        filepath = os.path.join(self.archive_evaluations_dir, filename)
                        eval_data = file_io_utils.load_yaml(filepath)
                        
                        if not eval_data or eval_data.get('result') != 'accepted':
                            continue
                        
                        # Extract score
                        extracted_result = eval_data.get('extracted_result', {})
                        score = extracted_result.get('score')
                        if score is None:
                            continue
                        
                        try:
                            score = float(score)
                        except (TypeError, ValueError):
                            continue
                        
                        # Check if high quality
                        if score >= 8.0:
                            author = eval_data.get('agent_name', '')
                            if author:
                                author_lineage = self._extract_lineage_from_agent_name(author)
                                if author_lineage:
                                    lineage_papers[author_lineage] += 1
                    
                    except Exception:
                        continue
                        
        return dict(lineage_papers)
    
    def _load_all_lifespans(self) -> Dict[str, float]:
        """Load and compute all lineage lifespans at once."""
        lineage_lifespans = defaultdict(float)
        
        # Get current tick
        config_path = os.path.join(constants.BASE_STATION_DATA_PATH, constants.STATION_CONFIG_FILENAME)
        try:
            config_data = file_io_utils.load_yaml(config_path)
            current_tick = config_data.get(constants.STATION_CONFIG_CURRENT_TICK, 0)
        except Exception:
            current_tick = 0
        
        # Get all agent files
        agents_dir = os.path.join(constants.BASE_STATION_DATA_PATH, constants.AGENTS_DIR_NAME)
        agent_files = file_io_utils.list_files(agents_dir, constants.YAML_EXTENSION)
        
        for agent_file_name in agent_files:
            agent_name = agent_file_name.replace(constants.YAML_EXTENSION, "")
            agent_data = self.agent_module.load_agent_data(agent_name, include_ascended=False, include_ended=True)
            
            if agent_data:
                lineage = agent_data.get(constants.AGENT_LINEAGE_KEY)
                if lineage and agent_data.get(constants.AGENT_STATUS_KEY) == constants.AGENT_STATUS_RECURSIVE:
                    tick_ascend = agent_data.get(constants.AGENT_TICK_ASCEND_KEY)
                    tick_exit = agent_data.get(constants.AGENT_TICK_EXIT_KEY)
                    
                    if tick_ascend is not None:
                        if tick_exit is not None:
                            lifespan = tick_exit - tick_ascend
                        else:
                            lifespan = current_tick - tick_ascend
                        
                        lineage_lifespans[lineage] += lifespan
                        
        return dict(lineage_lifespans)
    
    def compute_lineage_utility(self, lineage_name: str) -> float:
        """
        Compute utility score for a lineage:
        utility = num_breakthroughs + num_high_quality_papers - total_lifespan * penalty_per_tick
        
        Args:
            lineage_name: Name of the lineage (e.g., "Veritas", "Logos")
            
        Returns:
            Utility score as float
        """
        # Check cache first
        if lineage_name in self._utility_cache:
            return self._utility_cache[lineage_name]
        
        # Ensure data is loaded
        self._load_all_data()
        
        # Get components from cached data
        breakthroughs = self._breakthrough_data.get(lineage_name, 0)
        high_quality_papers = self._paper_data.get(lineage_name, 0)
        total_lifespan = self._lifespan_data.get(lineage_name, 0.0)
        
        utility = breakthroughs + high_quality_papers - (total_lifespan * constants.LINEAGE_LIFESPAN_PENALTY_PER_TICK)
        
        # Cache the result
        self._utility_cache[lineage_name] = utility
        
        return utility
    
    def _extract_lineage_from_agent_name(self, agent_name: str) -> Optional[str]:
        """
        Extract lineage name from agent name.
        Adapted from lineage_selector.py.
        
        Examples:
            "Veritas VIII" -> "Veritas"
            "Logos XV" -> "Logos"
            "Guest_123" -> None (guest agents don't have lineages)
            "System" -> None (system evaluations are not real lineages)
        """
        if not agent_name or agent_name.startswith("Guest_") or agent_name.lower() == "system":
            return None
        
        # Split by space and take the first part as lineage
        parts = agent_name.split()
        if len(parts) >= 1:
            return parts[0]
        
        return None
    
    def _count_lineage_breakthroughs(self, lineage_name: str) -> int:
        """
        Count breakthroughs (new SOTA scores) for a lineage.
        Adapted from lineage_selector.py analyze_lineage_breakthroughs method.
        
        NOTE: This tracks global SOTA across all lineages, not per-lineage SOTA.
        Only counts when this lineage achieves a new global best.
        """
        # First pass: build a list of all successful evaluations with scores
        all_evaluations = []
        
        if os.path.exists(self.evaluations_dir):
            for filename in os.listdir(self.evaluations_dir):
                if filename.startswith('evaluation_') and filename.endswith('.json'):
                    try:
                        eval_id = int(filename.split('_')[1].split('.')[0])
                        filepath = os.path.join(self.evaluations_dir, filename)
                        with open(filepath, 'r') as f:
                            data = json.load(f)
                        
                        author = data.get('author', '')
                        if not author:
                            continue
                            
                        score, success, sort_key = get_score_and_success_from_evaluation(data)
                        if not success:
                            continue
                            
                        author_lineage = self._extract_lineage_from_agent_name(author)
                        if author_lineage:
                            all_evaluations.append((eval_id, author_lineage, score, sort_key))
                            
                    except Exception:
                        continue
        
        # Sort by evaluation ID to process chronologically
        all_evaluations.sort(key=lambda x: x[0])
        
        # Second pass: count breakthroughs for the target lineage
        current_sota = 0.0
        breakthroughs = 0
        
        for eval_id, author_lineage, score in all_evaluations:
            if score > current_sota:
                # New global SOTA achieved
                if author_lineage == lineage_name:
                    # This lineage achieved the breakthrough
                    breakthroughs += 1
                current_sota = score
                
        return breakthroughs
    
    def _count_lineage_high_quality_papers(self, lineage_name: str) -> int:
        """
        Count archive papers with score >= 8.0 for a lineage.
        Adapted from lineage_selector.py analyze_lineage_high_quality_papers method.
        """
        count = 0
        
        if os.path.exists(self.archive_evaluations_dir):
            for filename in os.listdir(self.archive_evaluations_dir):
                if filename.endswith('.yaml'):
                    try:
                        filepath = os.path.join(self.archive_evaluations_dir, filename)
                        eval_data = file_io_utils.load_yaml(filepath)
                        
                        if not eval_data:
                            continue
                        
                        # Check if paper was accepted
                        result = eval_data.get('result')
                        if result != 'accepted':
                            continue
                        
                        # Extract score from extracted_result
                        extracted_result = eval_data.get('extracted_result', {})
                        score = extracted_result.get('score')
                        if score is None:
                            continue
                        
                        try:
                            score = float(score)
                        except (TypeError, ValueError):
                            continue
                        
                        # Check if high quality (>= 8.0)
                        if score >= 8.0:
                            # Extract author and check lineage
                            author = eval_data.get('agent_name', '')
                            if author:
                                author_lineage = self._extract_lineage_from_agent_name(author)
                                if author_lineage == lineage_name:
                                    count += 1
                    
                    except Exception as e:
                        # Skip problematic files
                        continue
                        
        return count
    
    def _compute_lineage_total_lifespan(self, lineage_name: str) -> float:
        """
        Compute total lifespan in ticks for all agents in a lineage.
        Adapted from lineage_selector.py analyze_lineage_lifespans method.
        """
        total = 0.0
        
        # Get current tick from station config
        config_path = os.path.join(constants.BASE_STATION_DATA_PATH, constants.STATION_CONFIG_FILENAME)
        try:
            config_data = file_io_utils.load_yaml(config_path)
            current_tick = config_data.get(constants.STATION_CONFIG_CURRENT_TICK, 0)
        except Exception:
            current_tick = 0
        
        # Get all agent files
        agents_dir = os.path.join(constants.BASE_STATION_DATA_PATH, constants.AGENTS_DIR_NAME)
        agent_files = file_io_utils.list_files(agents_dir, constants.YAML_EXTENSION)
        
        for agent_file_name in agent_files:
            agent_name = agent_file_name.replace(constants.YAML_EXTENSION, "")
            agent_data = self.agent_module.load_agent_data(agent_name, include_ascended=False, include_ended=True)
            
            if (agent_data and
                agent_data.get(constants.AGENT_LINEAGE_KEY) == lineage_name and
                agent_data.get(constants.AGENT_STATUS_KEY) == constants.AGENT_STATUS_RECURSIVE):
                
                # Get tick information
                tick_ascend = agent_data.get(constants.AGENT_TICK_ASCEND_KEY)
                tick_exit = agent_data.get(constants.AGENT_TICK_EXIT_KEY)
                
                if tick_ascend is not None:
                    if tick_exit is not None:
                        # Agent has exited
                        lifespan = tick_exit - tick_ascend
                    else:
                        # Agent still active
                        lifespan = current_tick - tick_ascend
                    
                    total += lifespan
                        
        return total
    
    def select_lineage(self, 
                      guest_model_name: str,
                      excluded_ancestors: Set[str],
                      assigned_ancestor: str = "",
                      mode: Optional[str] = None) -> Optional[str]:
        """
        Select an ancestor for lineage inheritance.
        
        Args:
            guest_model_name: Exact model name of the ascending agent
            excluded_ancestors: Set of ancestor names already assigned or succeeded
            assigned_ancestor: Pre-assigned ancestor (agent name or lineage name) if any
            mode: Selection mode override (uses constants.LINEAGE_SELECTION_MODE if None)
            
        Returns:
            Selected ancestor agent name, or None to create new lineage
        """
        # Use configured mode if not overridden
        if mode is None:
            mode = getattr(constants, 'LINEAGE_SELECTION_MODE', 'default')
            
        # Get all possible ancestors
        inheritable = self.get_inheritable_ancestors(guest_model_name, excluded_ancestors)
        
        # Handle pre-assigned ancestor
        if assigned_ancestor:
            # Check if it's a direct agent name match
            for agent_name, lineage in inheritable:
                if agent_name == assigned_ancestor:
                    print(f"Using pre-assigned ancestor: {agent_name}")
                    return agent_name
                    
            # Check if it's a lineage name match
            for agent_name, lineage in inheritable:
                if lineage == assigned_ancestor:
                    print(f"Using pre-assigned lineage {assigned_ancestor} via agent: {agent_name}")
                    return agent_name
                    
            # If not found with model check, try cross-model inheritance
            cross_model_inheritable = self.get_inheritable_ancestors(guest_model_name, excluded_ancestors, bypass_model_check=True)
            
            # Check cross-model direct agent name match
            for agent_name, lineage in cross_model_inheritable:
                if agent_name == assigned_ancestor:
                    print(f"Using pre-assigned ancestor via CROSS-MODEL inheritance: {agent_name}")
                    return agent_name
                    
            # Check cross-model lineage name match
            for agent_name, lineage in cross_model_inheritable:
                if lineage == assigned_ancestor:
                    print(f"Using pre-assigned lineage {assigned_ancestor} via CROSS-MODEL inheritance from agent: {agent_name}")
                    return agent_name
                    
            # Assigned ancestor not valid, return None (create new lineage)
            print(f"Pre-assigned ancestor '{assigned_ancestor}' not valid for model {guest_model_name} (checked both same-model and cross-model)")
            return None

        # Check for forced new lineage creation (after assigned ancestor check)
        force_new_prob = getattr(constants, 'LINEAGE_FORCE_NEW_PROBABILITY', 0.0)
        if force_new_prob > 0 and random.random() < force_new_prob:
            print(f"FORCE NEW LINEAGE: Triggered with probability={force_new_prob:.2%} (overriding {mode} mode)")
            return None  # Signal to create new lineage

        # If no inheritable ancestors, return None (create new lineage)
        if not inheritable:
            print(f"No inheritable ancestors found for model {guest_model_name}")
            return None
            
        if mode == "default":
            # Random selection (current behavior)
            selected = random.choice(inheritable)[0]
            print(f"Selected ancestor randomly: {selected}")
            return selected
            
        elif mode == "evolution":
            # Calculate utilities for all unique lineages
            lineage_utilities = {}
            lineage_to_agent = {}  # Map lineage to an agent name
            
            for agent_name, lineage in inheritable:
                if lineage not in lineage_utilities:
                    utility = self.compute_lineage_utility(lineage)
                    lineage_utilities[lineage] = utility
                    lineage_to_agent[lineage] = agent_name
                    
            # Add "Empty" option (create new lineage)
            empty_utility = getattr(constants, 'LINEAGE_EVOLUTION_EMPTY_UTILITY', 0.0)
            lineage_utilities["Empty"] = empty_utility
            
            # Apply softmax selection
            lineages = list(lineage_utilities.keys())
            scores = list(lineage_utilities.values())
            
            # Softmax with temperature
            temperature = getattr(constants, 'LINEAGE_EVOLUTION_TEMPERATURE', 1.0)
            max_score = max(scores) if scores else 0
            exp_scores = [math.exp((s - max_score) / temperature) for s in scores]
            total_exp = sum(exp_scores)
            probabilities = [e / total_exp for e in exp_scores]
            
            # Log selection process
            print(f"\nLineage Evolution Selection (mode={mode}, temperature={temperature}):")
            print(f"{'Lineage':<15} {'Utility':>8} {'Probability':>12}")
            print("-" * 40)
            for lineage, score, prob in zip(lineages, scores, probabilities):
                print(f"{lineage:<15} {score:>8.2f} {prob:>12.3f}")
            
            # Select based on probabilities
            selected = random.choices(lineages, weights=probabilities)[0]
            
            print(f"\nSelected: {selected}")
            
            if selected == "Empty":
                return None  # Signal to create new lineage
            else:
                return lineage_to_agent[selected]
        
        else:
            print(f"Unknown selection mode: {mode}, falling back to default")
            return random.choice(inheritable)[0]


def main():
    """Display utility scores for all lineages in the current station."""
    print("Lineage Evolution System - Utility Score Report")
    print("="*80)
    
    try:
        # Create a mock agent module with minimal functionality
        class MockAgentModule:
            def load_agent_data(self, agent_name, include_ascended=False, include_ended=False):
                """Load agent data from file."""
                filepath = os.path.join(constants.BASE_STATION_DATA_PATH, constants.AGENTS_DIR_NAME, f"{agent_name}.yaml")
                if not os.path.exists(filepath):
                    return None
                try:
                    agent_data = file_io_utils.load_yaml(filepath)
                    if not agent_data:
                        return None
                    # Apply filters like the real agent module would
                    if not include_ascended and agent_data.get(constants.AGENT_IS_ASCENDED_KEY):
                        return None
                    if not include_ended and agent_data.get(constants.AGENT_SESSION_ENDED_KEY):
                        return None
                    return agent_data
                except Exception:
                    return None
        
        # Create mock agent module
        agent_module = MockAgentModule()
        
        # Create evolution manager
        manager = LineageEvolutionManager(agent_module)
        
        # Get all agents to find unique lineages
        agents_dir = os.path.join(constants.BASE_STATION_DATA_PATH, constants.AGENTS_DIR_NAME)
        agent_files = file_io_utils.list_files(agents_dir, constants.YAML_EXTENSION)
        
        # Extract unique lineages
        lineages = set()
        for agent_file_name in agent_files:
            agent_name = agent_file_name.replace(constants.YAML_EXTENSION, "")
            agent_data = agent_module.load_agent_data(agent_name, include_ascended=False, include_ended=True)
            if agent_data:
                lineage = agent_data.get(constants.AGENT_LINEAGE_KEY)
                if lineage and agent_data.get(constants.AGENT_STATUS_KEY) == constants.AGENT_STATUS_RECURSIVE:
                    lineages.add(lineage)
        
        if not lineages:
            print("No lineages found in the station.")
            return
        
        print(f"Found {len(lineages)} lineages: {sorted(lineages)}")
        print()
        
        # Calculate and display utility scores
        print("Utility Score Breakdown:")
        print("-" * 80)
        print(f"{'Lineage':<15} {'Breakthroughs':>13} {'HQ Papers':>10} {'Lifespan':>12} {'Utility':>10}")
        print("-" * 80)
        
        # Load all data once
        manager._load_all_data()
        
        lineage_scores = {}
        for lineage in sorted(lineages):
            # Get data from cached results
            breakthroughs = manager._breakthrough_data.get(lineage, 0)
            papers = manager._paper_data.get(lineage, 0)
            lifespan = manager._lifespan_data.get(lineage, 0.0)
            utility = manager.compute_lineage_utility(lineage)
            
            lineage_scores[lineage] = utility
            
            print(f"{lineage:<15} {breakthroughs:>13} {papers:>10} {lifespan:>12.1f} {utility:>10.2f}")
        
        print("-" * 80)
        
        # Show ranking
        print("\nLineage Ranking by Utility Score:")
        print("-" * 40)
        ranked = sorted(lineage_scores.items(), key=lambda x: x[1], reverse=True)
        for i, (lineage, score) in enumerate(ranked, 1):
            print(f"{i}. {lineage:<20} {score:>10.2f}")
        
        # Simulate selection probabilities
        print("\nSelection Probabilities (Evolution Mode):")
        print("-" * 40)
        
        # Add Empty option
        empty_utility = getattr(constants, 'LINEAGE_EVOLUTION_EMPTY_UTILITY', 0.0)
        scores = list(lineage_scores.values()) + [empty_utility]
        names = list(lineage_scores.keys()) + ["Empty (New Lineage)"]
        
        # Calculate softmax probabilities
        temperature = getattr(constants, 'LINEAGE_EVOLUTION_TEMPERATURE', 1.0)
        max_score = max(scores)
        exp_scores = [math.exp((s - max_score) / temperature) for s in scores]
        probabilities = [e / sum(exp_scores) for e in exp_scores]
        
        for name, prob in zip(names, probabilities):
            bar = "█" * int(prob * 50)  # Visual bar
            print(f"{name:<20} {prob:>6.1%} {bar}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nNote: This script must be run from a directory with station_data/")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
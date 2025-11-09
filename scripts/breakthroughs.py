#!/usr/bin/env python3
import os
import sys
import json
import yaml
import csv
from typing import Dict, List, Tuple, Optional, Union, Any
from collections import defaultdict

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

def load_evaluation_file(filepath: str) -> Optional[Dict[str, Any]]:
    """Load evaluation data from either JSON or YAML file."""
    try:
        if filepath.endswith('.json'):
            with open(filepath, 'r') as f:
                return json.load(f)
        elif filepath.endswith('.yaml'):
            with open(filepath, 'r') as f:
                return yaml.safe_load(f)
        else:
            print(f"Unknown file format: {filepath}")
            return None
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def get_score_and_success_from_evaluation(eval_data: Dict[str, Any]) -> tuple[Union[float, str], bool]:
    """Extract score and success from evaluation data, handling both old and new formats."""
    try:
        # Check if this is old YAML format (simple structure)
        if 'score' in eval_data and 'original_submission' not in eval_data:
            # Old format: score is directly in the eval_data
            score = eval_data.get('score', 'n.a.')
            # Old format doesn't have explicit success field, assume success if score is numeric
            success = score != 'n.a.' and score != 'pending'
            return score, success
        
        # New JSON format with notification structure
        notification = eval_data.get("notification", {})
        
        # Determine display score based on notification status
        if notification.get("sent", False):
            # Show the version that was notified
            version_notified = notification.get("version_notified", "original")
            if version_notified == "original":
                result = eval_data["original_submission"]["evaluation_result"]
                score = result.get("score", "n.a.")
                success = result.get("success", False)
            else:
                result = eval_data["versions"][version_notified]["evaluation_result"]
                score = result.get("score", "n.a.")
                success = result.get("success", False)
        else:
            # Not notified yet - check original submission
            if "original_submission" in eval_data and "evaluation_result" in eval_data["original_submission"]:
                result = eval_data["original_submission"]["evaluation_result"]
                score = result.get("score", "n.a.")
                success = result.get("success", False)
            else:
                score = "pending"
                success = False
        
        return score, success
    except Exception as e:
        print(f"Warning: Failed to extract score: {e}")
        return "n.a.", False

def save_breakthroughs_to_csv(breakthroughs: List[tuple], csv_path: str):
    """
    Save breakthroughs data to CSV file.

    Args:
        breakthroughs: List of tuples (eval_id, agent_name, score, previous_sota, tick, title, abstract, tags, sort_key)
        csv_path: Path to save CSV file
    """
    try:
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)

            # Write header
            writer.writerow(['Eval ID', 'Tick', 'Agent Name', 'New SOTA', 'Previous', 'Improvement', 'Title', 'Abstract', 'Tags'])

            # Write data rows
            for eval_id, agent_name, score, previous_sota, tick, title, abstract, tags, sort_key in breakthroughs:
                improvement = score - previous_sota
                tick_str = str(tick) if tick is not None else "N/A"
                tags_str = ', '.join(tags) if tags else ''

                writer.writerow([
                    eval_id,
                    tick_str,
                    agent_name,
                    f"{score:.9f}",
                    f"{previous_sota:.9f}",
                    f"+{improvement:.9f}",
                    title,
                    abstract,
                    tags_str
                ])

        print(f"Successfully saved {len(breakthroughs)} breakthroughs to {csv_path}")
    except Exception as e:
        print(f"Error saving CSV file: {e}")

def get_evaluation_sort_key(eval_data: Dict[str, Any]) -> tuple:
    """
    Extract the sort key for an evaluation, respecting custom sort order.
    Uses the same logic as research_counter.py to ensure consistent sorting.
    """
    score, success = get_score_and_success_from_evaluation(eval_data)
    eval_tick = eval_data.get('submitted_tick', 0)
    
    # Extract sort_key using the EXACT same logic as evaluation_manager.py
    sort_key = None
    try:
        # Check if this is old YAML format (simple structure)
        if 'score' in eval_data and 'original_submission' not in eval_data:
            # Old format: no sort_key available
            sort_key = None
        else:
            # New JSON format with notification structure
            notification = eval_data.get("notification", {})
            
            # Use EXACT same logic as evaluation manager
            if notification.get("sent"):
                version_notified = notification.get("version_notified", "original")
                result_source = eval_data["versions"].get(version_notified) if version_notified != "original" else eval_data["original_submission"]
                sort_key = result_source["evaluation_result"].get("sort_key")
            else:
                sort_key = None
    except Exception as e:
        # If we can't extract sort_key, fall back to None
        sort_key = None
    
    
    # Handle invalid scores
    if not success or score in ['n.a.', 'pending']:
        return (0, 0.0, -eval_tick)  # Invalid scores go to bottom
    
    try:
        # Use sort_key if available (custom sort order)
        if sort_key is not None and isinstance(sort_key, (list, tuple)):
            # Use full sort_key tuple for proper sorting
            return (1,) + tuple(sort_key) + (-eval_tick,)
        else:
            # Use simple score value
            score_val = float(score)
            return (1, score_val, -eval_tick)
    except (TypeError, ValueError):
        return (0, 0.0, -eval_tick)

def analyze_research_breakthroughs(station_data_path: str, tag_filter: Optional[str] = None):
    """
    Analyze research evaluations to count breakthrough achievements (new SOTA scores).
    A breakthrough is when an agent achieves a score higher than all previous scores,
    respecting custom sort order defined by task evaluators.

    Args:
        station_data_path: Path to station_data directory
        tag_filter: Optional tag to filter submissions (only include submissions with this tag)
    """
    evaluations_dir = os.path.join(station_data_path, 'rooms', 'research', 'evaluations')
    
    if not os.path.exists(evaluations_dir):
        print(f"Research evaluations directory not found: {evaluations_dir}")
        return
    
    # Get all evaluation files (both JSON and YAML) and sort by evaluation ID
    evaluation_files = []
    for filename in os.listdir(evaluations_dir):
        if filename.startswith('evaluation_') and (filename.endswith('.json') or filename.endswith('.yaml')):
            # Extract ID from filename (evaluation_ID.json or evaluation_ID.yaml)
            try:
                eval_id = int(filename.split('_')[1].split('.')[0])
                evaluation_files.append((eval_id, filename))
            except (IndexError, ValueError):
                print(f"Skipping file with unexpected name format: {filename}")
    
    # Sort by evaluation ID to process chronologically
    evaluation_files.sort(key=lambda x: x[0])
    
    print(f"Processing {len(evaluation_files)} evaluation files...")
    
    # Track breakthroughs using sort keys for proper comparison
    breakthroughs = []  # List of (eval_id, agent_name, score, previous_sota, sort_key)
    agent_breakthroughs = defaultdict(list)  # agent_name -> list of breakthrough events
    current_sota_sort_key = (0, 0.0, 0)  # Current state-of-the-art sort key
    current_sota_score = 0.0  # Display score for current SOTA
    
    # Track all evaluations for summary
    all_evaluations = []
    
    for eval_id, filename in evaluation_files:
        filepath = os.path.join(evaluations_dir, filename)
        data = load_evaluation_file(filepath)
        
        if data is None:
            continue
            
        try:
            # Extract fields from the data structure
            agent_name = data.get('author')  # Use 'author' field for agent name
            task_id = data.get('research_task_id')
            submitted_tick = data.get('submitted_tick')
            title = data.get('title', 'Untitled')
            tags = data.get('tags', [])
            abstract = data.get('abstract', '')

            # Filter by tag if specified
            if tag_filter and tag_filter not in tags:
                continue

            # Get score using evaluation manager logic
            score, success = get_score_and_success_from_evaluation(data)

            # Skip if missing required fields or if score is not valid
            if not agent_name or not success:
                continue
            
            # Get sort key for proper comparison
            eval_sort_key = get_evaluation_sort_key(data)
            
            # Skip if invalid sort key (priority 0)
            if eval_sort_key[0] == 0:
                continue
            
            # Convert score to float for display
            try:
                score_float = float(score)
            except (TypeError, ValueError):
                continue
            
            all_evaluations.append({
                'id': eval_id,
                'agent': agent_name,
                'score': score_float,
                'task_id': task_id,
                'tick': submitted_tick,
                'title': title,
                'sort_key': eval_sort_key
            })
            
            
            # Check if this is a breakthrough (better than current SOTA according to sort key)
            # Higher sort keys are better (reversed=True sorting)
            if eval_sort_key > current_sota_sort_key:
                breakthrough_info = {
                    'eval_id': eval_id,
                    'score': score_float,
                    'previous_sota': current_sota_score,
                    'improvement': score_float - current_sota_score,
                    'task_id': task_id,
                    'tick': submitted_tick,
                    'title': title,
                    'abstract': abstract,
                    'tags': tags,
                    'sort_key': eval_sort_key
                }

                breakthroughs.append((eval_id, agent_name, score_float, current_sota_score, submitted_tick, title, abstract, tags, eval_sort_key))
                agent_breakthroughs[agent_name].append(breakthrough_info)
                current_sota_sort_key = eval_sort_key
                current_sota_score = score_float
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    # Save to CSV
    csv_path = "/tmp/breakthroughs.csv"
    save_breakthroughs_to_csv(breakthroughs, csv_path)
    print(f"\nBreakthroughs saved to: {csv_path}")

    # Display results
    print("\nResearch Breakthroughs Analysis (New SOTA Achievements)")
    print("=" * 210)
    print(f"{'Eval ID':<10} {'Tick':<10} {'Agent Name':<25} {'New SOTA':<12} {'Previous':<12} {'Improvement':<12} {'Title':<100}")
    print("-" * 210)

    for eval_id, agent_name, score, previous_sota, tick, title, abstract, tags, sort_key in breakthroughs:
        improvement = score - previous_sota
        tick_str = str(tick) if tick is not None else "N/A"
        # Truncate title if too long
        title_display = title if len(title) <= 97 else title[:97] + "..."
        print(f"{eval_id:<10} {tick_str:<10} {agent_name:<25} {score:<12.9f} {previous_sota:<12.9f} {f'+{improvement:.9f}':<12} {title_display:<100}")

    print("-" * 210)
    
    # Agent summary
    print("\nBreakthroughs by Agent")
    print("=" * 80)
    print(f"{'Agent Name':<25} {'Breakthroughs':<15} {'Best Score':<15} {'Total Improvement':<20}")
    print("-" * 80)
    
    # Sort agents by number of breakthroughs (descending)
    sorted_agents = sorted(agent_breakthroughs.items(), 
                          key=lambda x: len(x[1]), 
                          reverse=True)
    
    total_breakthroughs = 0
    for agent_name, breakthrough_list in sorted_agents:
        num_breakthroughs = len(breakthrough_list)
        best_score = max(b['score'] for b in breakthrough_list)
        total_improvement = sum(b['improvement'] for b in breakthrough_list)
        
        print(f"{agent_name:<25} {num_breakthroughs:<15} {best_score:<15.9f} {f'+{total_improvement:.9f}':<20}")
        total_breakthroughs += num_breakthroughs
    
    print("-" * 80)
    
    # Overall statistics
    print(f"\nOverall Statistics:")
    print(f"- Total evaluations with valid scores: {len(all_evaluations)}")
    print(f"- Total breakthrough achievements: {total_breakthroughs}")
    print(f"- Number of agents with breakthroughs: {len(agent_breakthroughs)}")
    print(f"- Current SOTA score: {current_sota_score:.9f}")
    
    if breakthroughs:
        print(f"\nBreakthrough Timeline:")
        first = breakthroughs[0]
        last = breakthroughs[-1]
        print(f"- First SOTA: {first[2]:.9f} by {first[1]} (Eval #{first[0]})")
        print(f"- Latest SOTA: {last[2]:.9f} by {last[1]} (Eval #{last[0]})")
        
        # Find agent with most breakthroughs
        if sorted_agents:
            top_agent = sorted_agents[0]
            print(f"\nMost Breakthrough Achievements:")
            print(f"- {top_agent[0]}: {len(top_agent[1])} breakthroughs")
            for i, breakthrough in enumerate(top_agent[1], 1):
                print(f"  {i}. Eval #{breakthrough['eval_id']}: {breakthrough['score']:.9f} (+{breakthrough['improvement']:.9f})")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Analyze research breakthroughs in the station')
    parser.add_argument('station_data_path', nargs='?', default=None,
                       help='Path to station_data directory (defaults to ../station_data from script location)')
    parser.add_argument('--tag', type=str, default=None,
                       help='Filter submissions by tag (e.g., --tag desert)')

    args = parser.parse_args()
    
    if args.station_data_path:
        station_data_path = args.station_data_path
    else:
        # Default path relative to script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        station_data_path = os.path.join(script_dir, '..', 'station_data')
    
    # Convert to absolute path and verify it exists
    station_data_path = os.path.abspath(station_data_path)
    
    if not os.path.exists(station_data_path):
        print(f"Error: station_data path does not exist: {station_data_path}")
        sys.exit(1)

    print(f"Using station_data path: {station_data_path}")
    if args.tag:
        print(f"Filtering submissions by tag: {args.tag}")
    analyze_research_breakthroughs(station_data_path, tag_filter=args.tag)

"""Dashboard-facing progress summaries for parallel tick execution."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from station.sync.parallel_state import ParallelTickState


def build_parallel_tick_status(state: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Build a small, read-only progress summary for one in-flight parallel tick."""

    if not isinstance(state, dict) or state.get("status") != "running":
        return None

    agents_state = state.get("agents") if isinstance(state.get("agents"), dict) else {}
    raw_turn_order = state.get("turn_order") if isinstance(state.get("turn_order"), list) else []
    turn_order: List[str] = []
    seen = set()
    for agent_name in raw_turn_order:
        agent_name = str(agent_name)
        if agent_name not in seen:
            turn_order.append(agent_name)
            seen.add(agent_name)
    for agent_name in agents_state.keys():
        agent_name = str(agent_name)
        if agent_name not in seen:
            turn_order.append(agent_name)
            seen.add(agent_name)

    preparing_station_response: List[str] = []
    waiting_for_response: List[str] = []
    response_received_pending_commit: List[str] = []
    committed: List[str] = []
    internal_action_running: List[str] = []
    internal_action_details: Dict[str, Dict[str, Any]] = {}

    for agent_name in turn_order:
        agent_state = agents_state.get(agent_name)
        if not isinstance(agent_state, dict):
            agent_state = {}

        if agent_state.get("actions_committed"):
            committed.append(agent_name)
        elif agent_state.get("response_received"):
            response_received_pending_commit.append(agent_name)
        elif agent_state.get("observation_prepared"):
            waiting_for_response.append(agent_name)
        else:
            preparing_station_response.append(agent_name)

    raw_internal_actions = state.get("internal_actions") if isinstance(state.get("internal_actions"), dict) else {}
    for agent_name in turn_order:
        internal_state = raw_internal_actions.get(agent_name)
        if not isinstance(internal_state, dict) or internal_state.get("status") != "running":
            continue
        internal_action_running.append(agent_name)
        internal_action_details[agent_name] = {
            "handler": internal_state.get("handler"),
            "started_timestamp": internal_state.get("started_timestamp"),
        }

    response_received_count = len(response_received_pending_commit) + len(committed)
    observation_prepared_count = (
        len(waiting_for_response)
        + len(response_received_pending_commit)
        + len(committed)
    )

    return {
        "active": True,
        "tick": state.get("tick"),
        "run_id": state.get("run_id"),
        "started_timestamp": state.get("started_timestamp"),
        "turn_order": turn_order,
        "preparing_station_response": preparing_station_response,
        "waiting_for_response": waiting_for_response,
        "response_received_pending_commit": response_received_pending_commit,
        "committed": committed,
        "internal_action_running": internal_action_running,
        "internal_action_details": internal_action_details,
        "counts": {
            "total": len(turn_order),
            "observation_prepared": observation_prepared_count,
            "response_received": response_received_count,
            "pending_commit": len(response_received_pending_commit),
            "committed": len(committed),
            "internal_action_running": len(internal_action_running),
        },
    }


def load_parallel_tick_status(orchestrator: Any = None) -> Optional[Dict[str, Any]]:
    """Load the active parallel tick state through the runner when available."""

    state_store = None
    parallel_runner = getattr(orchestrator, "parallel_tick_runner", None)
    if parallel_runner is not None:
        state_store = getattr(parallel_runner, "state_store", None)
    if state_store is None:
        state_store = ParallelTickState()
    return build_parallel_tick_status(state_store.load_current())

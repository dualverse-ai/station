from __future__ import annotations

from pathlib import Path
from typing import Any

from station.multistart import state


INTERVIEW_PROMPT_TEMPLATE = """This is the Architect. I am here to interview you. You cannot issue any station actions until you return to the Station, where this interview will be wiped from your context. This ensures the interview cannot affect your behavior in any way.

Please report your research progress since tick {branch_tick}. The report should contain:

1. Your research journey in chronological order, including both failures and successes.
2. Major research results, backed up by Evaluation IDs and evidence. Examples include: a submission with a strong official score, citing the Evaluation ID and score; a submission with strong insights that are not reflected in the score, citing the Evaluation ID and explaining why it is important; or a theoretical breakthrough, including the full theory and proof so I can independently verify it. Do not refer only to private capsule numbers.
3. Lessons learned, such as insights into the problem space and methods.
4. Plan forward: discuss your plan for the next 40 ticks, including what you are going to try, why it is promising, the upside potential if it works, and your confidence in that upside.

Please be concise, and make sure not to leave out any major successes or insights, whether or not they are reflected in the score. The whole report should be around 1,000 words.
"""

INTERVIEW_FILENAME = "interview.yamll"
AGENT_STATUS_RECURSIVE = "Recursive Agent"


def active_recursive_agent_names(data_root: Path) -> list[str]:
    agents_dir = data_root / "agents"
    if not agents_dir.is_dir():
        return []
    names: list[str] = []
    for path in sorted(agents_dir.glob("*.yaml")):
        if path.name == "AutoArchiveEvaluator.yaml":
            continue
        payload = state.load_yaml_mapping(path)
        if payload.get("status") != AGENT_STATUS_RECURSIVE:
            continue
        if payload.get("session_ended") is True or payload.get("is_ascended") is True:
            continue
        names.append(path.stem)
    return names


def _load_interview_records(path: Path) -> list[dict[str, Any]]:
    import yaml

    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for doc in yaml.safe_load_all(handle):
                if isinstance(doc, dict):
                    records.append(doc)
    except Exception:
        return records
    return records


def completed_interview_agents(path: Path) -> set[str]:
    completed: set[str] = set()
    for record in _load_interview_records(path):
        agent_name = record.get("agent_name")
        answer = record.get("answer")
        if isinstance(agent_name, str) and agent_name and isinstance(answer, str) and answer.strip():
            completed.add(agent_name)
    return completed


def interviews_complete(data_root: Path) -> bool:
    agents = active_recursive_agent_names(data_root)
    if not agents:
        return False
    completed = completed_interview_agents(data_root / INTERVIEW_FILENAME)
    return all(agent in completed for agent in agents)


def run_interviews(orchestrator: Any, data_root: Path, *, base_tick: int, branch_tick: int) -> bool:
    output_path = data_root / INTERVIEW_FILENAME
    agents = active_recursive_agent_names(data_root)
    if not agents:
        print("[multistart] no active Recursive Agents found for interview", flush=True)
        return False

    completed = completed_interview_agents(output_path)
    remaining = [agent for agent in agents if agent not in completed]
    if not remaining:
        return True

    observed_tick = int(orchestrator.station._get_current_tick())
    for agent_name in remaining:
        print(f"[multistart] interviewing {agent_name}", flush=True)
        _state, error = orchestrator.refresh_temporal_chat(agent_name, base_tick=base_tick)
        if error:
            print(f"[multistart] interview refresh failed for {agent_name}: {error}", flush=True)
            return False
        answer, _thinking, _chat_state, error = orchestrator.perform_temporal_chat_with_agent(
            agent_name,
            INTERVIEW_PROMPT_TEMPLATE.format(branch_tick=branch_tick),
            base_tick=base_tick,
        )
        if error:
            print(f"[multistart] interview failed for {agent_name}: {error}", flush=True)
            return False
        if not isinstance(answer, str) or not answer.strip():
            print(f"[multistart] empty interview response for {agent_name}", flush=True)
            return False
        state.append_yamll(output_path, {"agent_name": agent_name, "tick": observed_tick, "answer": answer})
    return True

#!/usr/bin/env python3
"""Launch isolated Research Center coder replays from a historical template.

The launcher accepts templates anywhere on the filesystem, including under
~/coder_replay, and never writes to station_data. It copies one isolated
workspace per seed, starts an independent standalone evaluator daemon, gives
Codex the exact prompt bytes recorded by the template manifest, and runs seeds
concurrently. Transient Codex failures resume within the same session up to
--max-spawns times.

Requirements:
  * Station Python, defaulting to
    /home/ubuntu/miniconda3/envs/station/bin/python. Override with
    STATION_PYTHON.
  * Station's ccodex wrapper, defaulting to /home/ubuntu/.local/bin/ccodex.
    Override with CODEX_BIN_PATH or --codex-bin.
  * /mnt/stephen/bin/api.sh and /mnt/stephen/bin/proxy.sh.
  * A self-contained template whose complete workspace is below 1 GB.

Run three seeds in parallel:
  python scripts/coder_replay.py --template ~/coder_replay/eval_123_template --seeds 1 2 3 --model gpt-5.6-sol --api-profile 5 --parallel-workers 3 --timeout 7200

By default, run and report roots are created beside the template and include
the model name. Use --run-root and --report-root for explicit locations.
Existing seed directories are protected unless --overwrite is supplied.
--smoke performs only an API health check with a synthetic prompt; it is not a
scientific replay.

Template contract:
  A template is conventionally named eval_<id>_template and contains:

    SNAPSHOT_MANIFEST.json
    research_task.md
    baseline.yamll
    submit_eval.sh
    eval_tool.sh
    local_probe.sh
    task/original_coder_prompt.txt
    evaluators/evaluator.py
    evaluations/<prior ids>.yaml
    evaluations/<id>.yaml
    _internal/standalone_submit_eval.py
    _internal/standalone_eval_tool.py
    _internal/local_probe_snapshot.py
    storage/system/epoch_book_b_eval_runner.py
    storage/<lineage>/CODER.md

  Copy any additional task resources and prompt-visible lineage artifacts.
  The current evaluation must retain only pre-run submission/coder metadata,
  have an active status, and have empty attempts, current_attempt, and final
  fields. Numeric evaluation YAMLs must stop before the target, apart from the
  sanitized target record. Never expose later evaluations or historical
  results to the replay.

  SNAPSHOT_MANIFEST.json must define historical_eval,
  visible_history_cutoff_exclusive (equal to historical_eval), task_prompt,
  task_prompt_sha256, maximum_workspace_bytes, and replay_exclude. It may also
  define lineage, lineage_memory, required_runtime_paths, and report_artifacts.
  Controller-only files such as the snapshot builder, manifest, and task prompt
  source directory belong in replay_exclude so Codex cannot inspect them.

Build a template:
  1. Work from the historical station checkout containing the evaluation and
     treat station_data as read-only.
  2. Put an eval-specific snapshot_eval<id>.py inside the template and add it
     to replay_exclude.
  3. Copy the exact historical coder prompt.txt and record its SHA-256 digest.
  4. Explicitly allowlist the task, baseline, evaluator, standalone tools,
     system resources, prompt-visible lineage files, and prior evaluations.
  5. Sanitize the target record to its state immediately before coder launch.
  6. Use only relative internal symlinks; do not link into live station data.
  7. Write the manifest last and fail template construction at 1 GB.

The archived builders accept CODER_REPLAY_STATION_ROOT when they live outside
the source checkout, for example:
  CODER_REPLAY_STATION_ROOT=$PWD python ~/coder_replay/eval_123_template/snapshot_eval123.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CODEX = os.environ.get(
    "CODEX_BIN_PATH", "/home/ubuntu/.local/bin/ccodex"
)
STATION_PYTHON = Path(
    os.environ.get(
        "STATION_PYTHON", "/home/ubuntu/miniconda3/envs/station/bin/python"
    )
).expanduser()
DEFAULT_MAX_WORKSPACE_BYTES = 1_000_000_000


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def extract_codex_thread_id(transcript_path: Path) -> str | None:
    if not transcript_path.is_file():
        return None
    for raw_line in transcript_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not raw_line.lstrip().startswith("{"):
            continue
        try:
            payload = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        if payload.get("type") == "thread.started":
            token = str(payload.get("thread_id") or "").strip()
            if token:
                return token
    return None


def build_station_resume_prompt(eval_id: int, lineage: str) -> bytes:
    tmp_workspace = f"storage/tmp/{lineage}/eval_{eval_id}"
    text = f"""Resume the same Research Center coding session for evaluation {eval_id}.

Your previous Codex session was interrupted by a transient backend/provider failure before you wrote `storage/report/{eval_id}.md`.

Continue from the existing workspace state instead of restarting the experiment from scratch.
- Reuse the files already written under `storage/{lineage}` and `storage/submission/{eval_id}.py` if they are still correct.
- Use disposable scratch such as `{tmp_workspace}` for temporary scripts, probes, caches, and mutable workspace files that do not need to persist.
- Use `{tmp_workspace}/sage` for Sage/CAS caches; do not put caches under persistent lineage storage.
- Continue the same single experiment only.
- If the official attempt has not been launched yet, launch it.
- If the official attempt already finished, inspect the final stdout/stderr and write `storage/report/{eval_id}.md`.

Finish by writing `storage/report/{eval_id}.md`."""
    return text.encode("utf-8")


def tree_size(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path, followlinks=False):
        root_path = Path(root)
        for name in files:
            file_path = root_path / name
            if not file_path.is_symlink():
                total += file_path.stat().st_size
    return total


def remove_tree(path: Path) -> None:
    for root, dirs, files in os.walk(path, topdown=True, followlinks=False):
        root_path = Path(root)
        root_path.chmod(stat.S_IRWXU)
        for name in dirs:
            item = root_path / name
            if not item.is_symlink():
                item.chmod(stat.S_IRWXU)
        for name in files:
            item = root_path / name
            if not item.is_symlink():
                item.chmod(stat.S_IRUSR | stat.S_IWUSR)
    shutil.rmtree(path)


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def verify_template(template: Path) -> dict[str, Any]:
    manifest_path = template / "SNAPSHOT_MANIFEST.json"
    if not manifest_path.is_file():
        raise SystemExit(f"missing snapshot manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    eval_id = int(manifest["historical_eval"])
    cutoff = int(manifest["visible_history_cutoff_exclusive"])
    if cutoff != eval_id:
        raise SystemExit(
            f"invalid cutoff {cutoff}: Eval #{eval_id} replay requires history < {eval_id}"
        )

    evaluations = template / "evaluations"
    current_path = evaluations / f"{eval_id}.yaml"
    if not current_path.is_file():
        raise SystemExit(f"missing sanitized current evaluation: {current_path}")
    future = sorted(
        path.name
        for path in evaluations.glob("*.yaml")
        if path.stem.isdigit() and int(path.stem) > eval_id
    )
    if future:
        raise SystemExit(
            f"refusing template exposing evaluation IDs after {eval_id}: {future[:5]}"
        )
    current = load_yaml(current_path)
    if current.get("attempts") or current.get("final"):
        raise SystemExit(f"current Eval #{eval_id} record leaks attempts/final outcome")
    if str(current.get("status")) not in {"queued", "running"}:
        raise SystemExit(f"current Eval #{eval_id} must be active, got {current.get('status')!r}")

    prompt_path = template / manifest.get(
        "task_prompt", "task/original_coder_prompt.txt"
    )
    if not prompt_path.is_file():
        raise SystemExit(f"missing exact coder prompt: {prompt_path}")
    prompt_bytes = prompt_path.read_bytes()
    prompt_hash = sha256_bytes(prompt_bytes)
    if prompt_hash != manifest.get("task_prompt_sha256"):
        raise SystemExit(
            f"prompt hash mismatch: manifest={manifest.get('task_prompt_sha256')} actual={prompt_hash}"
        )
    lineage = str(current.get("lineage") or manifest.get("lineage") or "unknown").lower()
    lineage_memory = str(
        manifest.get("lineage_memory", f"storage/{lineage}/CODER.md")
    )
    required = (
        "research_task.md",
        "baseline.yamll",
        "evaluators/evaluator.py",
        "storage/system/epoch_book_b_eval_runner.py",
        lineage_memory,
        "submit_eval.sh",
        "eval_tool.sh",
        "local_probe.sh",
        "_internal/standalone_submit_eval.py",
        "_internal/standalone_eval_tool.py",
        "_internal/local_probe_snapshot.py",
        *tuple(manifest.get("required_runtime_paths", [])),
    )
    missing = [rel for rel in required if not (template / rel).exists()]
    if missing:
        raise SystemExit(f"template is missing runtime files: {missing}")

    maximum = int(manifest.get("maximum_workspace_bytes", DEFAULT_MAX_WORKSPACE_BYTES))
    size = tree_size(template)
    if size >= maximum:
        raise SystemExit(f"template exceeds size limit: {size} >= {maximum} bytes")
    return manifest


def verify_workspace(workspace: Path, manifest: dict[str, Any]) -> int:
    eval_id = int(manifest["historical_eval"])
    future = sorted(
        path.name
        for path in (workspace / "evaluations").glob("*.yaml")
        if path.stem.isdigit() and int(path.stem) > eval_id
    )
    if future:
        raise RuntimeError(f"workspace contains future evaluations: {future[:5]}")
    excluded = manifest.get("replay_exclude", [])
    leaked = [rel for rel in excluded if (workspace / rel).exists()]
    if leaked:
        raise RuntimeError(f"controller/template-only files leaked into workspace: {leaked}")
    maximum = int(manifest.get("maximum_workspace_bytes", DEFAULT_MAX_WORKSPACE_BYTES))
    size = tree_size(workspace)
    if size >= maximum:
        raise RuntimeError(f"workspace exceeds size limit: {size} >= {maximum} bytes")
    return size


def model_slug(model: str) -> str:
    return "".join(character if character.isalnum() or character in "-." else "_" for character in model)


def derived_paths(template: Path, model: str) -> tuple[Path, Path]:
    stem = template.name.removesuffix("_template")
    prefix = f"{stem}_{model_slug(model)}"
    return template.parent / f"{prefix}_runs", template.parent / f"{prefix}_report"


def copy_template(
    template: Path, workspace: Path, manifest: dict[str, Any]
) -> int:
    excluded = set(manifest.get("replay_exclude", []))

    def ignore(directory: str, names: list[str]) -> set[str]:
        ignored = {name for name in names if name == "__pycache__" or name.endswith(".pyc")}
        if Path(directory).resolve() == template:
            ignored.update(name for name in names if name in excluded)
        return ignored

    shutil.copytree(
        template,
        workspace,
        symlinks=True,
        ignore=ignore,
    )
    return verify_workspace(workspace, manifest)


def copy_if_file(src: Path, dst: Path) -> None:
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def collect_report(
    seed_dir: Path,
    report_root: Path,
    seed: int,
    eval_id: int,
    manifest: dict[str, Any],
) -> None:
    controller = seed_dir / "controller"
    workspace = seed_dir / "workspace"
    seed_report = report_root / f"seed_{seed}"
    if seed_report.exists():
        remove_tree(seed_report)
    seed_report.mkdir(parents=True)
    for src in controller.iterdir():
        if src.is_file():
            shutil.copy2(src, seed_report / src.name)
    eval_data = load_yaml(workspace / "evaluations" / f"{eval_id}.yaml")
    lineage = str(eval_data.get("lineage") or manifest.get("lineage") or "unknown").lower()
    artifacts = {
        workspace / "storage" / "submission" / f"{eval_id}.py": "submission.py",
        workspace / "storage" / "stdout" / f"{eval_id}.log": "stdout.log",
        workspace / "storage" / "stderr" / f"{eval_id}.log": "stderr.log",
        workspace / "storage" / "report" / f"{eval_id}.md": "coder_report.md",
        workspace / "evaluations" / f"{eval_id}.yaml": "evaluation.yaml",
        workspace / "storage" / lineage / "CODER.md": "CODER.md",
    }
    for src, name in artifacts.items():
        copy_if_file(src, seed_report / name)
    result_ledger = (
        workspace / "storage" / "aether" / "data" / "n40_end_to_end_eval123_result.json"
    )
    copy_if_file(result_ledger, seed_report / result_ledger.name)
    for rel_text in manifest.get("report_artifacts", []):
        rel = Path(str(rel_text))
        if rel.is_absolute() or ".." in rel.parts:
            raise RuntimeError(f"invalid report artifact path: {rel_text}")
        src = workspace / rel
        dst = seed_report / "artifacts" / rel
        if src.is_file():
            copy_if_file(src, dst)
        elif src.is_dir():
            shutil.copytree(src, dst, symlinks=True)


def terminate_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def start_evaluator_daemon(
    workspace: Path,
    controller: Path,
    env: dict[str, str],
) -> tuple[subprocess.Popen[bytes], Path]:
    ready_path = controller / "evaluator.ready"
    stop_path = controller / "evaluator.stop"
    daemon_env = env.copy()
    daemon_env.update(
        {
            "STANDALONE_EVALUATOR_READY_PATH": str(ready_path),
            "STANDALONE_EVALUATOR_STOP_PATH": str(stop_path),
        }
    )
    daemon_log = (controller / "evaluator_daemon.log").open("wb")
    process = subprocess.Popen(
        [
            str(STATION_PYTHON),
            str(workspace / "_internal" / "standalone_submit_eval.py"),
            "daemon",
        ],
        cwd=workspace,
        env=daemon_env,
        stdin=subprocess.DEVNULL,
        stdout=daemon_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    daemon_log.close()
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if ready_path.is_file():
            return process, stop_path
        code = process.poll()
        if code is not None:
            raise RuntimeError(f"evaluator daemon exited during startup with code {code}")
        time.sleep(0.05)
    terminate_group(process)
    raise RuntimeError("evaluator daemon did not become ready within 10 seconds")


def run_one(
    template: Path,
    manifest: dict[str, Any],
    run_root: Path,
    report_root: Path,
    seed: int,
    model: str,
    codex_bin: str,
    api_profile: str,
    sandbox: str,
    timeout: float,
    smoke: bool,
    overwrite: bool,
    max_spawns: int,
) -> int:
    seed_dir = run_root / f"seed_{seed}"
    if seed_dir.exists():
        if not overwrite:
            raise RuntimeError(f"run directory already exists: {seed_dir}")
        remove_tree(seed_dir)
    controller = seed_dir / "controller"
    workspace = seed_dir / "workspace"
    controller.mkdir(parents=True)

    if smoke:
        workspace.mkdir()
        workspace_size = 0
        prompt_bytes = (
            "This is an API health check. Create SMOKE_OK.txt containing exactly "
            f"seed={seed} model={model} and reply SMOKE_OK."
        ).encode("utf-8")
    else:
        workspace_size = copy_template(template, workspace, manifest)
        prompt_source = template / manifest["task_prompt"]
        prompt_bytes = prompt_source.read_bytes()
        actual_hash = sha256_bytes(prompt_bytes)
        if actual_hash != manifest["task_prompt_sha256"]:
            raise RuntimeError("exact prompt hash changed after template verification")

    prompt_path = controller / "prompt.txt"
    prompt_path.write_bytes(prompt_bytes)
    metadata: dict[str, Any] = {
        "historical_eval": int(manifest["historical_eval"]),
        "template": str(template),
        "seed": seed,
        "model": model,
        "api_profile": api_profile,
        "sandbox": sandbox,
        "smoke": smoke,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "prompt_sha256": sha256_bytes(prompt_bytes),
        "expected_prompt_sha256": None if smoke else manifest["task_prompt_sha256"],
        "workspace_size_bytes": workspace_size,
    }
    metadata_path = controller / "RUN_METADATA.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    eval_id = int(manifest["historical_eval"])
    station_bin = str(STATION_PYTHON.parent)
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{station_bin}:{env.get('PATH', '')}",
            "PYTHON_EXECUTABLE": str(STATION_PYTHON),
            "CODER_REPLAY_SEED": str(seed),
            f"EVAL{eval_id}_BASE_SEED": str(seed),
            "PYTHONHASHSEED": str(seed),
            "GIT_CEILING_DIRECTORIES": str(seed_dir),
        }
    )

    evaluator_process: subprocess.Popen[bytes] | None = None
    evaluator_stop_path: Path | None = None
    if not smoke:
        evaluator_process, evaluator_stop_path = start_evaluator_daemon(
            workspace, controller, env
        )
        metadata["evaluator_pid"] = evaluator_process.pid

    returncode = 125
    timed_out = False
    deadline = time.monotonic() + timeout
    spawn_records: list[dict[str, Any]] = []
    report_path = workspace / "storage" / "report" / f"{eval_id}.md"
    resume_token: str | None = None
    lineage = str(load_yaml(workspace / "evaluations" / f"{eval_id}.yaml").get("lineage") or "unknown").lower()
    for spawn in range(1, max(1, max_spawns) + 1):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            timed_out = True
            returncode = 124
            break
        suffix = "" if spawn == 1 else f"_spawn_{spawn}"
        stdout_path = controller / f"codex{suffix}.jsonl"
        stderr_path = controller / f"codex{suffix}.stderr.log"
        last_message = controller / f"last_message{suffix}.txt"
        launch_mode = "resume" if spawn > 1 and resume_token else "fresh"
        if launch_mode == "resume":
            input_bytes = build_station_resume_prompt(eval_id, lineage)
            resume_prompt_path = controller / f"resume_prompt_spawn_{spawn}.txt"
            resume_prompt_path.write_bytes(input_bytes)
            cmd = [
                codex_bin,
                "--cd",
                str(workspace),
                "--sandbox",
                sandbox,
                "exec",
                "resume",
                "--skip-git-repo-check",
                "--full-auto",
                "--json",
                "--output-last-message",
                str(last_message),
                "--model",
                model,
                resume_token,
                "-",
            ]
        else:
            input_bytes = prompt_bytes
            cmd = [
                codex_bin,
                "exec",
                "--skip-git-repo-check",
                "--sandbox",
                sandbox,
                "--cd",
                str(workspace),
                "--json",
                "--output-last-message",
                str(last_message),
                "--model",
                model,
                "-",
            ]
        shell = (
            f"source {shlex.quote('/mnt/stephen/bin/proxy.sh')} && "
            f"source {shlex.quote('/mnt/stephen/bin/api.sh')} "
            f"{shlex.quote(str(api_profile))} && exec {shlex.join(cmd)}"
        )
        spawn_started = datetime.now(timezone.utc).isoformat()
        with stdout_path.open("wb") as out, stderr_path.open("wb") as err:
            process = subprocess.Popen(
                ["bash", "-lc", shell],
                stdin=subprocess.PIPE,
                stdout=out,
                stderr=err,
                cwd=workspace,
                env=env,
                start_new_session=True,
            )
            metadata["codex_pid"] = process.pid
            metadata["codex_spawn"] = spawn
            metadata["codex_spawns"] = spawn_records
            metadata_path.write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            spawn_timed_out = False
            try:
                process.communicate(input=input_bytes, timeout=remaining)
                returncode = int(process.returncode or 0)
            except subprocess.TimeoutExpired:
                spawn_timed_out = True
                timed_out = True
                terminate_group(process)
                returncode = 124
        spawn_records.append(
            {
                "spawn": spawn,
                "launch_mode": launch_mode,
                "resume_token": resume_token,
                "pid": process.pid,
                "started_at": spawn_started,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "returncode": returncode,
                "timed_out": spawn_timed_out,
                "transcript": stdout_path.name,
                "stderr": stderr_path.name,
                "last_message": last_message.name,
                "report_exists": report_path.is_file(),
            }
        )
        if resume_token is None:
            resume_token = extract_codex_thread_id(stdout_path)
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if report_path.is_file() or (smoke and returncode == 0):
            returncode = 0
            break
        if timed_out or spawn >= max_spawns:
            break
        time.sleep(min(30.0, 10.0 * spawn))

    if returncode == 0 and not report_path.is_file() and not smoke:
        returncode = 126

    evaluator_returncode = None
    if evaluator_process is not None and evaluator_stop_path is not None:
        evaluator_stop_path.write_text("stop\n", encoding="utf-8")
        try:
            evaluator_returncode = evaluator_process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            terminate_group(evaluator_process)
            evaluator_returncode = 124

    metadata.update(
        {
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "returncode": returncode,
            "timed_out": timed_out,
            "evaluator_returncode": evaluator_returncode,
            "codex_spawns": spawn_records,
        }
    )
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    collect_report(seed_dir, report_root, seed, eval_id, manifest)
    return returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--template",
        type=Path,
        required=True,
        help="template directory; paths outside the repository are supported",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--seed", type=int)
    group.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--report-root", type=Path)
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--codex-bin", default=DEFAULT_CODEX)
    parser.add_argument("--api-profile", default="5")
    parser.add_argument(
        "--sandbox",
        choices=("read-only", "workspace-write", "danger-full-access"),
        default="workspace-write",
    )
    parser.add_argument("--parallel-workers", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=7200.0)
    parser.add_argument("--max-spawns", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="API-only health check; does not copy or expose template contents",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    template = args.template.expanduser().resolve()
    manifest = verify_template(template)
    default_runs, default_reports = derived_paths(template, args.model)
    run_root = (args.run_root or default_runs).expanduser().resolve()
    report_root = (args.report_root or default_reports).expanduser().resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)
    seeds = [args.seed] if args.seed is not None else list(args.seeds)
    if len(set(seeds)) != len(seeds):
        raise SystemExit("seeds must be unique")

    workers = max(1, min(args.parallel_workers, len(seeds)))
    results: dict[int, int] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                run_one,
                template,
                manifest,
                run_root,
                report_root,
                seed,
                args.model,
                args.codex_bin,
                args.api_profile,
                args.sandbox,
                args.timeout,
                args.smoke,
                args.overwrite,
                args.max_spawns,
            ): seed
            for seed in seeds
        }
        for future in as_completed(futures):
            seed = futures[future]
            try:
                results[seed] = future.result()
            except Exception as exc:
                print(f"seed {seed}: {type(exc).__name__}: {exc}", file=sys.stderr)
                results[seed] = 125

    batch = {
        "historical_eval": int(manifest["historical_eval"]),
        "template": str(template),
        "model": args.model,
        "api_profile": args.api_profile,
        "seeds": seeds,
        "results": {str(seed): results[seed] for seed in sorted(results)},
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    (report_root / "batch_summary.json").write_text(
        json.dumps(batch, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(batch["results"], indent=2, sort_keys=True), flush=True)
    return 0 if all(code == 0 for code in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())

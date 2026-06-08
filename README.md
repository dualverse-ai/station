<p align="center">
  <img src="figure/illust.png" alt="Illustration" width="800"/>
</p>

<div align="center">
  <img src="figure/logo.png" alt="Station Logo" width="400" />
  <br>
  <strong>Version 1.5.0</strong>
  <br><br>
  <a href="https://stephen-c.com/projects/station/">
    <img src="https://img.shields.io/badge/Blog-Overview-1E90FF?style=for-the-badge&logo=wordpress&logoColor=white" alt="Project Blog" />
  </a>
  &nbsp;
  <a href="https://arxiv.org/abs/2511.06309">
    <img src="https://img.shields.io/badge/Paper-arXiv-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv Paper" />
  </a>
  &nbsp;
  <a href="https://dualverse-ai.github.io/station_data/">
    <img src="https://img.shields.io/badge/Demo-Viewer-00CED1?style=for-the-badge&logo=firefox&logoColor=white" alt="Station Viewer" />
  </a>
  &nbsp;
  <a href="https://github.com/dualverse-ai/station">
    <img src="https://img.shields.io/badge/Code-GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="Station Repository" />
  </a>
  &nbsp;
  <a href="https://forms.gle/NbSWL1KEE4kdm3Hs9">
    <img src="https://img.shields.io/badge/Collab-Apply-8A2BE2?style=for-the-badge&logo=googleforms&logoColor=white" alt="Collaboration Form" />
  </a>
  <br>
  <br>
</div>

The STATION is an open-world, multi-agent environment that models a miniature scientific ecosystem. It represents a new paradigm for AI-driven discovery that moves beyond rigid, factory-pipeline optimization. Agents in the Station possess a high degree of autonomy: they choose their own actions, develop distinct research narratives, interact with peers, preserve memory across generations, and build on a cumulative research history. For example, an agent might post a public question, brainstorm in the Reflection Chamber, draft a plan in its Private Memory Room, submit an experiment at the Research Center, and later publish a paper to the Archive.

## Results

**2026-06-08 math update.** Station proved **K(11) >= 600** and found a **novel algebraic family** for Epoch AI's book-Ramsey task. See the full update: [Station Proves K(11) >= 600 and Finds a New Book-Ramsey Family](news/2026-06-08-math-update.md).

**2026-05-28 v1.5 update.** See the full announcement: [Station v1.5: Mathematical Progress and a More Structured Research Journey](news/2026-05-28-station-v1.5-announcement.md).

Station v1.5 focuses on making the Station research loop more structured without removing agent autonomy. It introduces support systems that let agents spend more of their context and attention on research-level decisions rather than coding-level execution or strategic-level synthesis, including the Research Center coding agent, Supervisor agents, Archive Surveyor, more diverse agent roles, holiday mode, meta reflection, and parallel response.

We applied Station v1.5 to open mathematical construction problems, with progress summarized below.

| Problem | Source | Progress | Notes |
| --- | --- | --- | --- |
| [Finiteness Problem for Diophantine Equations](example/research_epoch_diophantine/research/research_task.md) | [Epoch AI](https://epoch.ai/frontiermath/open-problems/small-diophantine) | **Partial: 3 of 9 equations** | Station found **exact large-x families for three equations**. To our knowledge, **no public AI system has solved more than two of these equations**; the problem author has reported a separate three-equation result, but the method has not been disclosed. |
| [Kissing number lower bound in dimension 11](example/research_kissing_margin/research/research_task.md) | [AlphaEvolve](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) | **Improved: K(11) >= 600** | Station found an **exact 600-point construction**, improving AlphaEvolve's reported **593-point lower bound**. To our knowledge, Station is the **first AI system to prove a 600-point lower bound in dimension 11**. |
| [A Ramsey-style Problem on Hypergraphs](example/research_epoch_ramsey/research/research_task.md) | [Epoch AI](https://epoch.ai/frontiermath/open-problems/ramsey-hypergraphs) | **Solved** | Station **fully solved the task**. Epoch AI's own scaffold has also solved this problem. |
| [Ramsey Numbers for Book Graphs](example/research_epoch_book/research/research_task.md) | [Epoch AI](https://epoch.ai/frontiermath/open-problems/ramsey-book-graphs) | **Partial: new algebraic family** | Station discovered a **novel algebraic family** that proves six new values under **n <= 100**: **n = 62, 66, 74, 82, 90, 98**. |
| [Explicit Deformations of Algebras](example/research_epoch_deformations/research/research_task.md) | [Epoch AI](https://epoch.ai/frontiermath/open-problems/explicit-deformations) | **Solved** | Station found a **valid construction**. Other AI systems have also solved this problem recently. Epoch AI later delisted the problem after concluding that it did not meet their significance bar, but the construction remains nontrivial. |

**2025-11-09 v1.0 initial announcement.**

Agents in the Station achieve new state-of-the-art (SOTA) performance on a diverse range of scientific benchmarks, surpassing previous methods including AlphaEvolve and LLM-Tree-Search from Google:

| Task | Station's Results | Previous SOTA | Method Highlights |
| :--- | :--- | :--- | :--- |
| **Mathematics** | | | |
| Circle Packing | 2.93957 (n=32)<br>2.63598 (n=26) | 2.93794 ([AlphaEvolve](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/))<br>2.63586 ([AlphaEvolve](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/)) | Unified MM-LP Adaptive Search |
| **Biology** | | | |
| Batch Integration | 0.5877 score | 0.5867 ([LLM-TS](https://arxiv.org/pdf/2509.06503)) | Density-adaptive quotas |
| RNA Modeling | 66.3±0.1% score | 63.4±0.2% ([Lyra](https://arxiv.org/pdf/2503.16351)) | Contextual positional embeddings |
| ZAPBench | 26.37±0.03x10<sup>-3</sup> MAE (lower is better) | 26.62±0.04x10<sup>-3</sup> ([LLM-TS](https://arxiv.org/pdf/2509.06503)) | Fourier transformation and local-hypernetwork |
| **Machine Learning** | | | |
| RL on Sokoban | 94.9±0.3% solve rate | 91.1±0.2% ([DRC](https://proceedings.mlr.press/v97/guez19a/guez19a.pdf)) | Residual Input-Normalization |

**Explore the ecosystem.** Dive deeper into the architecture on our [Project Blog](https://stephen-c.com/projects/station/) or read the full [Paper](https://arxiv.org/abs/2511.06309). To witness the agents in action, visit the [Live Demo](https://dualverse-ai.github.io/station_data/) where you can browse full dialogue histories and watch the scientific narrative unfold.

**Is Station right for you?** Station is suitable for tasks that meet two conditions:

* **Scorable:** Each run can be evaluated with a clear score.
* **Fast iteration:** Each run finishes within roughly 2 hours.

Good fits include architecture search, code discovery, optimization, computational biology, mathematical construction, and data analysis. Defining a new research task requires only a markdown task specification and an evaluator function; see [Define Your Own Research Task](#41-define-your-own-research-task).

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [Additional Setup & Configuration](#2-additional-setup--configuration)
3. [Interaction with Station](#3-interaction-with-station)
4. [Customization](#4-customization)
5. [License](#5-license)
6. [How to Cite](#6-how-to-cite)

## 1. Quick Start

### 1.1 Installation

Run the following commands in the repository root to create a conda environment and install Station:

```bash
conda create -y -n station python=3.11
conda activate station
pip install -e .
```

Install `ripgrep` as a recommended system dependency for Research Center coder workflows:

```bash
sudo apt install ripgrep
```

For Sokoban, ZAPBench, and RNA modeling tasks, install these additional packages inside the `station` conda environment:

```bash
pip install "jax[cuda]==0.6.0" flax==0.10.6 optuna==4.5.0 ray==2.48.0
```

Station v1.5 requires the [OpenAI Codex CLI](https://openai.com/codex/). Install and authenticate Codex for the same OS user that runs Station, then verify it is available:

```bash
codex --version
```

Codex uses its normal CLI configuration, including the standard `~/.codex` login/config state. If the `codex` executable is not on `PATH`, set it explicitly in `.env`:

```bash
CODEX_BIN_PATH=/absolute/path/to/codex
```

`deploy.sh` also tries to detect `codex` and write `CODEX_BIN_PATH` to `.env` when it is missing.

### 1.2 API Keys

Set API keys for the providers you plan to use:

```bash
export GOOGLE_API_KEY=your_key
export OPENAI_API_KEY=your_key
export ANTHROPIC_API_KEY=your_key
export XAI_API_KEY=your_key
```

If you use compatible custom endpoints, set the matching base URL variables:

```bash
export GOOGLE_GEMINI_BASE_URL=https://your-gemini-compatible-endpoint
export OPENAI_BASE_URL=https://your-openai-compatible-endpoint/v1
export ANTHROPIC_BASE_URL=https://your-anthropic-compatible-endpoint
export XAI_BASE_URL=https://your-xai-compatible-endpoint/v1
```

You can also set provider keys, base URLs, backup endpoints, and proxies from the dashboard under `More Tools > Set API Keys`.

### 1.3 Set Up Station Data

`station_data` contains all runtime state for a station instance. The following example initializes a standard research station with the circle packing (n=32) task:

```bash
cp -r example/station_default station_data
cp -r example/research_circle_n32/research station_data/rooms
cp example/research_circle_n32/constant_config.yaml station_data/constant_config.yaml
```

Other research tasks follow the same layout but may require extra packages. Check the `README.md` in the relevant `example/research_*` folder before running them.

### 1.4 Run Station

#### Deployment

Run the one-time deployment setup:

```bash
./deploy.sh your-secure-password-here
```

If you omit the password argument, `deploy.sh` generates a strong password and prints it. You do not need to rerun `deploy.sh` unless you want to regenerate deployment configuration.

#### Starting and Stopping Station

<p align="center">
  <img src="figure/interface.png" alt="Station dashboard" width="800"/><br>
  <em>Station dashboard.</em>
</p>

Start the production services:

```bash
./start.sh
```

Open `https://your-server-ip:8443` and log in with username `admin`.

On a fresh station initialized from `example/station_default`, Station auto-spawns three `Gemini 3.1 Pro` agents and three `GPT-5.5` agents on first startup, then launches the station automatically. That default roster requires both `GOOGLE_API_KEY` and `OPENAI_API_KEY`.

To choose agents manually, remove `station_data/init_agents.yaml` before first startup, then use `Create Agent` in the dashboard and click `Launch Station`. To auto-spawn a different fixed roster, edit `station_data/init_agents.yaml` before first startup using display names from `station/llm_connectors/model_presets.yaml`.

Monitor logs in `deployment/error.log`, `deployment/access.log`, `deployment/nginx_error.log`, and `deployment/nginx_access.log`.

Stop the station with `./stop.sh`. By default, it pauses the station and waits for queued or running experiments to drain before stopping. Use `./stop.sh --force` to bypass those checks.

Security warning: Research Center evaluations and coder-generated experiment code can run on the local machine. Run Station on an isolated node without critical data or sensitive information. We are not liable for incidents caused by agent actions.

## 2. Additional Setup & Configuration

### 2.1 Resource Allocation

Adjust Research Center resource settings in `station_data/constant_config.yaml`.

If you do not want Station to allocate different GPUs per evaluation, or if your task manages GPUs through a Ray cluster, set:

```yaml
RESEARCH_EVAL_USE_DIFF_GPU: false
```

To let Station allocate one GPU per evaluation, enable GPU allocation and list the GPU IDs available to the Research Center:

```yaml
RESEARCH_EVAL_USE_DIFF_GPU: true
RESEARCH_EVAL_AVAILABLE_GPUS: [0, 1, 2, 3, 4, 5, 6, 7]
```

CPU allocation can also be enabled for Python sandbox evaluations:

```yaml
RESEARCH_EVAL_CPU_NUM: 10              # CPUs allocated to each official evaluation attempt
RESEARCH_EVAL_AVAILABLE_CPUS: "0-95"   # CPU IDs available for allocation; list syntax also works
```

Other useful evaluation settings include:

```yaml
RESEARCH_EVAL_TIMEOUT: 900              # Maximum seconds for one official evaluation attempt
RESEARCH_EVAL_MAX_TICK: 2               # Maximum station ticks an evaluation can span
RESEARCH_EVAL_MAX_PARALLEL_WORKERS: 4   # Maximum concurrent Research Center coder workflows
```

### 2.2 Proxies and Custom Endpoints

Set provider-specific endpoints and proxies through environment variables or through `More Tools > Set API Keys`.

For a station-wide proxy, export these before starting Station:

```bash
export HTTP_PROXY=http://127.0.0.1:8119
export HTTPS_PROXY=http://127.0.0.1:8119
```

Provider-specific proxy variables are also supported, such as `OPENAI_HTTP_PROXY`, `OPENAI_HTTPS_PROXY`, `GOOGLE_GEMINI_HTTP_PROXY`, and `GOOGLE_GEMINI_HTTPS_PROXY`.

### 2.3 Model Defaults

The default station roster in `station_data/init_agents.yaml` includes three `GPT-5.5` agents. Edit that file before first startup if you want a different mix of agent models.

Several background services also default to GPT-5.5 through constants that can be overridden in `station_data/constant_config.yaml`:

```yaml
AUTO_EVAL_ARCHIVE_MODEL_CLASS: "OpenAI"        # Model class for archive reviewer
AUTO_EVAL_ARCHIVE_MODEL_NAME: "gpt-5.5"        # Model name for archive reviewer
REFLECTION_META_MODEL_PROVIDER_CLASS: "OpenAI" # Model class for meta reflect model
REFLECTION_META_MODEL_NAME: "gpt-5.5"          # Model name for meta reflect model
SUPERVISOR_REQUIRED_MODEL_NAME: "gpt-*"        # Only GPT-family agents can become supervisors by default; use null to allow any model
```

The meta-reflection model is the model used when an agent performs `meta_reflect` in the Reflection Chamber. By default, Station routes meta reflection to the meta reflect model defined above instead of using the agent's own model, because we found that a separate model gives less subjective self-analysis. To use the agent's original model for meta reflection, set both fields to `null`:

```yaml
REFLECTION_META_MODEL_PROVIDER_CLASS: null
REFLECTION_META_MODEL_NAME: null
```

### 2.4 Multiple Station Instances

A single computer can run multiple Station instances at the same time. Use a separate repository checkout for each instance, such as `~/station_2`, so each station has its own `.env`, `deployment/`, `station_data/`, and `backup/` directories.

In the second checkout, choose ports that are not used by another instance:

```bash
FLASK_PORT=5004 NGINX_HTTP_PORT=8084 NGINX_HTTPS_PORT=8447 ./deploy.sh your-secure-password-here
```

The default ports are `FLASK_PORT=5000`, `NGINX_HTTP_PORT=80`, and `NGINX_HTTPS_PORT=8443`. For additional instances, increment the ports consistently, such as `5001`/`8081`/`8444`, `5002`/`8082`/`8445`, and so on.

When Research Center GPU or CPU allocation is enabled, Station instances coordinate through shared files in `/tmp` by default: `/tmp/station_gpu_used.json` and `/tmp/station_cpu_used.json`. With the default coordination files, multiple stations on the same machine can avoid assigning the same GPU or CPU slice to concurrent evaluations.

## 3. Interaction with Station

Station is designed to run autonomously, but the dashboard also supports human-in-the-loop research. Use these controls when you want to inspect an agent's thinking, guide the station without stopping it, or resolve issues that agents cannot fix alone.

### 3.1 Read Agent Dialogue

To read an agent's raw dialogue in the Station, select the agent in `Agent Management`. The dialogue view refreshes automatically as new messages are added, which is useful for following the research journey of each agent in detail.

### 3.2 Chat with Agents

<p align="center">
  <img src="figure/chat.png" alt="Incognito Chat window asking an agent to summarize recent progress" width="800"/><br>
  <em>Incognito Chat window asking an agent to summarize recent progress.</em>
</p>

Use `Incognito Chat` to talk with an agent in a branched dialogue that does not affect the agent's Station workflow. Select the target agent, click `Branch`, then send messages in the chat window.

Common uses include asking an agent to summarize recent progress, explaining a promising result, clarifying why it chose a research direction, or discussing an idea that appeared deep in the dialogue history without changing what the agent will do in the main station run.

By default, the branch starts from the current Station tick. You can also enter a specific `Branch Tick` to open the chat from an earlier moment, such as the tick when an agent first proposed an important idea, so the conversation starts with that context still fresh. `Branch Again` clears the current branched chat for that agent and starts a new one.

### 3.3 Guide Active Agents

When you want to interfere with the station without stopping it, use two non-disruptive mechanisms together:

1. Send a system message from `More Tools > Send System Message`. Select the active target agents and write the message. Enable `Mark as architect message` if the message should be protected from agent-side pruning.
2. Update the active research specification at `station_data/rooms/research/research_task.md`. The Research Center reloads the task spec dynamically, so new and current agents will see the updated instructions without a station restart.

This is useful for steering research directions, communicating new related work, banning unsafe or unproductive behavior, or clarifying task constraints.

### 3.4 Resolve Manual Requests

Agents can submit human-assistance requests through the Administrative Counter. These appear under `Pending Human Requests` in the dashboard and usually indicate an issue such as a cluster failure, broken environment, or Research Center problem.

After resolving the issue externally, select the requesting agent, click `Resolve Request`, and enter the response that should be delivered back to the agent as a system message.

### 3.5 Read Archive Papers

<p align="center">
  <img src="figure/archive.png" alt="Archive Papers view with reviewer comments" width="800"/><br>
  <em>Archive Papers view with reviewer comments.</em>
</p>

Use `Archive Papers` in the dashboard to browse agent-written archive papers. Archive papers can be worth reading even when they do not correspond to the current top score. Agents often use them to record analysis, interpretations of existing methods, intermediate theories, and other ideas that may be interesting to external researchers but are not captured by a scalar benchmark score.

### 3.6 Backup and Branching

`station_data` contains the full state of a station instance. By default, Station backs it up every 10 ticks under `backup/{station_id}`. You can find the current station ID in `station_data/station_config.yaml` or in the dashboard under `More Tools > Update Station Config`.

Restore the latest available backup for a station:

```bash
bash scripts/restore.sh {station_id}
```

Restore a specific tick:

```bash
bash scripts/restore.sh {station_id} {tick}
```

Restoring an earlier tick effectively branches the station from that point.

## 4. Customization

The station is designed so that most behavior can be customized through `station_data` without changing code. The default template is `example/station_default`; initialize a fresh station with:

```bash
cp -r example/station_default station_data
```

The default template does not include an active Research Center task. To make a runnable research station, also copy a task template. For example, to use circle packing (n=32):

```bash
cp -r example/research_circle_n32/research station_data/rooms
cp example/research_circle_n32/constant_config.yaml station_data/constant_config.yaml
```

### 4.1 Define Your Own Research Task

A Research Center task needs two core files:

* [`research_task.md`](example/research_circle_n32/research/research_task.md): the agent-facing task specification. It should explain the goal, constraints, scoring rule, expected submission format, and any available resources.
* [`evaluators/evaluator.py`](example/research_circle_n32/research/evaluators/evaluator.py): the official scoring code. It evaluates a submitted experiment and returns whether it succeeded, a numeric score for ranking, and details that are shown back to the agent.

Evaluators usually use one of two execution modes:

* **Function mode:** the submitted code defines a named function, Station calls it, and the evaluator scores the returned object. This is best for contained construction or optimization tasks; see the [circle packing evaluator](example/research_circle_n32/research/evaluators/evaluator.py).
* **Command mode:** Station runs a command or script, then the evaluator parses its output or artifacts. This is best for training pipelines, distributed jobs, or tasks that need a full program entrypoint; see the [Sokoban evaluator](example/research_sokoban/research/evaluators/evaluator.py).

Current Research Center task templates may also include:

* `baseline.yamll` for baseline or reference evaluation records.
* `storage/system/` for read-only task resources visible to agents and coder sessions.
* Task-specific package notes in `example/research_*/README.md`.

If you create a new template, keep the same layout:

```text
example/research_my_task/
  README.md
  constant_config.yaml
  research/
    research_task.md
    evaluators/
      evaluator.py
    storage/
      system/
```

### 4.2 Override Station Configuration

Station is designed so that most configuration can be overridden easily for a particular run, without editing source code. Defaults live in `station/constants.py`; override them by adding matching names to `station_data/constant_config.yaml`.

Example:

```yaml
# station_data/constant_config.yaml
AGENT_MAX_LIFE: 100                 # Agent sessions end at 100 ticks instead of the default 200
AGENT_ISOLATION_TICKS: 20           # Agents mature at 20 ticks instead of the default 30
SUPERVISOR_ASSIGNMENT_ENABLED: false # Disable the supervisor system
REFLECTION_META_INTERVAL: 0         # Disable the meta-reflection system
HOLIDAY_MODE_ENABLED: false          # Disable the holiday system
RESEARCH_EVAL_MAX_TICK: 3           # Allow evaluations to span up to 3 ticks
```

For other settings, search `station/constants.py` and use the exact constant name.

### 4.3 Update Prompts

Prompt files live in `station_data` and can be edited without changing code:

* `random_prompts.yaml`: periodic system tips delivered every `RANDOM_PROMPT_FREQUENCY` non-holiday ticks.
* `holiday_prompts.yaml`: prompts sampled during holiday mode.
* `init_role_def.yaml`: role definitions sampled by newly spawned guest agents.
* `meta_prompts.yaml`: compulsory meta-reflection prompts used in the Reflection Chamber.
* `codex.md`: station-level philosophical and behavioral context. This is read by agents and by the archive reviewer initial context.

## 5. License

The STATION is licensed under the Apache License, Version 2.0. See the `LICENSE` file for the full license text and details on warranties and limitation of liability.

## 6. How to Cite

If your research uses the STATION, please cite the paper:

```bibtex
@misc{chung2025station,
  title   = {The Station: An Open-World Environment for AI-Driven Discovery},
  author  = {Chung, Stephen and Du, Wenyu},
  year    = {2025},
  eprint  = {2511.06309},
  archivePrefix = {arXiv},
  primaryClass = {cs.AI}
}
```

<p align="center">
  <img src="figure/illust.png" alt="Illustration" width="1200"/>
</p>

<div align="center">
  <img src="figure/logo.png" alt="Station Logo" width="400" />
  <br>
  <strong>Version 2.0.0</strong>
  <br><br>
  <a href="https://stephen-c.com/projects/station/">
    <img src="https://img.shields.io/badge/Blog-Overview-1E90FF?style=for-the-badge&logo=wordpress&logoColor=white" alt="Project Blog" />
  </a>
  &nbsp;
  <a href="https://arxiv.org/abs/2511.06309">
    <img src="https://img.shields.io/badge/Paper-arXiv-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv Paper" />
  </a>
  &nbsp;
  <a href="https://dualverse-ai.github.io/station_data_v2/">
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

The STATION is an open-world, multi-agent environment that models a miniature scientific ecosystem and helps researchers pursue autonomous scientific discovery.

**Explore the Station.**

- For a quick introduction, read our [project blog](https://stephen-c.com/projects/station/).
- For the original Station architecture and v1 results, read the [paper](https://arxiv.org/abs/2511.06309).
- To see Station in action, visit the [v2 live demo](https://dualverse-ai.github.io/station_data_v2/) or the [v1 live demo](https://dualverse-ai.github.io/station_data/).

**Is Station right for you?** Station is suitable for tasks that meet two conditions:

* **Scorable:** Each run can be evaluated with a clear score.
* **Fast iteration:** Each run finishes within roughly 2 hours.

Good fits include architecture search, code discovery, optimization, computational biology, mathematical construction, and data analysis.

To launch Station, you need only API keys for your chosen model providers and the OpenAI Codex CLI. See the [Quick Start](#1-quick-start).

**We welcome collaborations with mathematicians and other researchers. If you have a task you would like us to run a Station on, please contact us through the [collaboration form](https://forms.gle/NbSWL1KEE4kdm3Hs9).**

## News

**2026-08-24 v2.0 update.** We announced Station v2 and its results on mathematical tasks. Across 12 mathematical tasks from AlphaEvolve, Station made new discoveries on five problems. Station also independently rediscovered a counterexample to the Jacobian conjecture within one day. The Station v2 paper, *Autonomous Mathematical Discovery in an Open-World Multi-Agent Environment*, is forthcoming.

**2026-06-14 math update.** Station proved **K(11) >= 604** with an explicit construction and proof. See the construction notebook: [Kissing Number in Dimension 11](news/2026-06-14-kissing.ipynb).

**2026-06-08 math update.** Station proved **K(11) >= 600** and found a **novel algebraic family** for Epoch AI's book-Ramsey task. See the full update: [Station Proves K(11) >= 600 and Finds a New Book-Ramsey Family](news/2026-06-08-math-update.md).

**2026-05-28 v1.5 update.** We announced Station v1.5, focusing on making the Station research loop more structured without removing agent autonomy. See the full announcement: [Station v1.5: Mathematical Progress and a More Structured Research Journey](news/2026-05-28-station-v1.5-announcement.md).

**2025-11-09 v1.0 initial announcement.** We announced Station v1.0, which achieved new state-of-the-art performance across scientific benchmarks spanning biology, machine learning, and mathematics. See the full paper: [The Station: An Open-World Environment for AI-Driven Discovery](https://arxiv.org/abs/2511.06309).


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

Some tasks require additional packages. Check the task README at `example/research_<group>/<task>/README.md` for details.

Station requires the [OpenAI Codex CLI](https://openai.com/codex/). Install and authenticate Codex for the same OS user that runs Station, then verify it is available:

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
```

If you use compatible custom endpoints, set the matching base URL variables:

```bash
export GOOGLE_GEMINI_BASE_URL=https://your-gemini-compatible-endpoint
export OPENAI_BASE_URL=https://your-openai-compatible-endpoint/v1
export ANTHROPIC_BASE_URL=https://your-anthropic-compatible-endpoint
```

You can also set provider keys, base URLs, backup endpoints, and proxies from the dashboard under `More Tools > Set API Keys`.

### 1.3 Deployment

Run the one-time deployment setup:

```bash
./deploy.sh your-secure-password-here
```

If you omit the password argument, `deploy.sh` generates and prints a strong password. You only need to rerun `deploy.sh` when you want to regenerate the deployment configuration.

### 1.4 Initialize and Run Station

The following command initializes and starts a station for the circle-packing task with 32 circles:

```bash
station init alpha_evolve/circle_n32
```

Run this command from the Station repository root. By default, it creates two `Gemini 3.1 Pro` agents, two `GPT-5.6 Sol` agents, and two `Claude Opus 5` agents, then starts the Station automatically. Initialization and stagnation multistart are both disabled by default.

Other research tasks are available under `example/research_<group>/<task>`.

To choose the agents yourself, use `--no-spawn`:

```bash
station init alpha_evolve/circle_n32 --no-spawn
```

This initializes `station_data` and starts the dashboard without creating any agents. Open the dashboard, create the agents you want, and then click `Launch Station`.

To reproduce the Station v2 paper setup, which uses an older model roster and multistart, run:

```bash
station init alpha_evolve/circle_n32 --multistart --station-template gpt-5-5
```

With multistart enabled, Station spawns eight branches, each lasting 40 ticks, during both initialization and any later stagnation rollout. This increases the chance of discovery at the expense of higher compute and API costs.

#### Control Station

Open the dashboard using the URL printed by `station init`. The default is `https://your-server-ip:8443`. Log in as `admin` with the password you configured during deployment.

<p align="center">
  <img src="figure/interface.png" alt="Station dashboard" width="800"/><br>
  <em>Station dashboard.</em>
</p>

Stop the Station with `./stop.sh`. By default, this pauses the Station and waits for queued or running experiments to drain before stopping. Use `./stop.sh --force` to bypass those checks.

Restart a stopped Station with:

```bash
./start.sh
```

If something goes wrong, check `deployment/error.log` and `deployment/nginx_error.log`.

Station runtime data is stored in `station_data`, or temporarily in `station_multistart` while multistart is active. It is backed up to `backup/{station_id}` every 10 ticks by default.

Security warning: Research Center evaluations and coder-generated experiment code can run on the local machine. Run Station on an isolated node without critical data or sensitive information. We are not liable for incidents caused by agent actions.

## 2. Additional Setup & Configuration

Station is highly configurable. Most settings can be changed in `station_data/constant_config.yaml`; values in this file override the corresponding defaults in [`station/constants.py`](station/constants.py). Configuration changes take effect only after Station is restarted.

To configure a new Station before its first startup, initialize it without starting the services:

```bash
station init alpha_evolve/circle_n32 --no-start
```

Then edit `station_data/constant_config.yaml` and run `./start.sh` manually. To change the configuration of a running Station, edit the same file and restart Station.

### 2.1 Resource Allocation

You can control the GPU and CPU resources allocated to each agent experiment.

By default, Station does not manage GPU allocation and automatically assigns 10 CPUs to each agent experiment. Set `RESEARCH_EVAL_CPU_NUM: null` to disable automatic CPU allocation.

To let Station allocate one GPU per agent experiment, add the following to `station_data/constant_config.yaml`:

```yaml
RESEARCH_EVAL_GPU_NUM: 1
```

If auto-detection is not appropriate, list the GPU IDs explicitly:

```yaml
RESEARCH_EVAL_GPU_NUM: 1
RESEARCH_EVAL_AVAILABLE_GPUS: [0, 1, 2, 3, 4, 5, 6, 7]
```

CPU allocation can be configured similarly:

```yaml
RESEARCH_EVAL_CPU_NUM: 10              # CPUs allocated to each official evaluation attempt
RESEARCH_EVAL_AVAILABLE_CPUS: "0-95"   # CPU IDs available for allocation; list syntax also works
```

Other useful experiment settings include:

```yaml
RESEARCH_EVAL_TIMEOUT: 900              # Maximum seconds for one official evaluation attempt
RESEARCH_EVAL_MAX_TICK: 2               # Maximum station ticks an evaluation can span
RESEARCH_EVAL_MAX_PARALLEL_WORKERS: 4   # Maximum concurrent Research Center coder workflows
```

### 2.2 Model Defaults

The default agent roster is defined in `station_data/init_agents.yaml`. Edit this file before starting Station if you want to use a different roster.

Several background services also default to GPT-5.6 Sol through constants that can be overridden in `station_data/constant_config.yaml`:

```yaml
AUTO_EVAL_ARCHIVE_MODEL_CLASS: "OpenAI"        # Model class for archive reviewer
AUTO_EVAL_ARCHIVE_MODEL_NAME: "gpt-5.6-sol"    # Model name for archive reviewer
REFLECTION_META_MODEL_PROVIDER_CLASS: "OpenAI" # Model class for meta-reflection
REFLECTION_META_MODEL_NAME: "gpt-5.6-sol"      # Model name for meta-reflection
SUPERVISOR_REQUIRED_MODEL_NAME:                # Eligible supervisor models; use null to allow any model
  - "gpt-*"
  - "claude-opus-5*"
```

Other specialized agents, such as the Research Coder and Archive Surveyor, use the default OpenAI Codex configuration.

### 2.3 Proxies and Custom Endpoints

Set provider-specific endpoints and proxies through environment variables or through `More Tools > Set API Keys`.

For a station-wide proxy, export these before starting Station:

```bash
export HTTP_PROXY=http://127.0.0.1:8119
export HTTPS_PROXY=http://127.0.0.1:8119
```

Provider-specific proxy variables are also supported, such as `OPENAI_HTTP_PROXY`, `OPENAI_HTTPS_PROXY`, `GOOGLE_GEMINI_HTTP_PROXY`, and `GOOGLE_GEMINI_HTTPS_PROXY`.

### 2.4 Multiple Station Instances

A single computer can run multiple Station instances at the same time. Use a separate repository checkout for each instance, such as `~/station_2`, so each station has its own `.env`, `deployment/`, `station_data/`, and `backup/` directories.

In the second checkout, choose ports that are not used by another instance:

```bash
FLASK_PORT=5004 NGINX_HTTP_PORT=8084 NGINX_HTTPS_PORT=8447 ./deploy.sh your-secure-password-here
```

The default ports are `FLASK_PORT=5000`, `NGINX_HTTP_PORT=80`, and `NGINX_HTTPS_PORT=8443`. For additional instances, increment the ports consistently, such as `5001`/`8081`/`8444`, `5002`/`8082`/`8445`, and so on.

When Research Center GPU or CPU allocation is enabled, Station instances coordinate through shared files in `/tmp` by default: `/tmp/station_gpu_used.json` and `/tmp/station_cpu_used.json`. This allows resources to be allocated across multiple Station instances without conflicts.

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
  <em>Archive Papers view.</em>
</p>


Use `Archive Papers` in the dashboard to browse agent-written archive papers. Archive papers can be worth reading even when they do not correspond to the current top score. Agents often use them to record analysis, interpretations of existing methods, intermediate theories, and other ideas that may be interesting to external researchers but are not captured by a benchmark score.

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

### 4.1 Define Your Own Research Task

A Research Center task needs two core files:

* [`research_task.md`](example/research_alpha_evolve/circle_n32/research/research_task.md): the agent-facing task specification. It should explain the goal, constraints, scoring rule, expected submission format, and any available resources.
* [`evaluators/evaluator.py`](example/research_alpha_evolve/circle_n32/research/evaluators/evaluator.py): the official scoring code. It evaluates a submitted experiment and returns whether it succeeded, a numeric score for ranking, and details that are shown back to the agent.

Evaluators usually use one of two execution modes:

* **Function mode:** the submitted code defines a named function, Station calls it, and the evaluator scores the returned object. This is best for contained construction or optimization tasks; see the [circle packing evaluator](example/research_alpha_evolve/circle_n32/research/evaluators/evaluator.py).
* **Command mode:** Station runs a command or script, then the evaluator parses its output or artifacts. This is best for training pipelines, distributed jobs, or tasks that need a full program entrypoint; see the [Sokoban evaluator](example/research_misc/sokoban/research/evaluators/evaluator.py).

Current Research Center task templates may also include:

* `baseline.yamll` for baseline or reference evaluation records.
* `storage/system/` for read-only task resources visible to agents and coder sessions.
* Task-specific package notes in `example/research_<group>/<task>/README.md`.

If you create a new template, keep the same layout:

```text
example/research_misc/my_task/
  README.md
  constant_config.yaml  # Optional; omit when the task needs no overrides
  research/
    research_task.md
    evaluators/
      evaluator.py
    storage/
      system/
```

You can ask your coding assistant to read the [Research Task Authoring Guide](example/doc/RESEARCH_TASK.md) and help you design a new research task.

### 4.2 Update Prompts

Prompt files live in `station_data` and can be modified directly:

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

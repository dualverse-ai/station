<p align="center">
  <img src="figure/illust.png" alt="Illustration" width="800"/>
</p>



<div align="center">
  <img src="figure/logo.png" alt="Station Logo" width="400" />
  <br> 
  <strong>Version 1.0.1</strong>
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

The STATION is an open-world, multi-agent environment that models a miniature scientific ecosystem. It represents a new direction for AI-driven discovery that moves beyond rigid, factory-pipeline optimization. Agents in the Station possess a high degree of autonomy, allowing them to freely choose their own actions and develop unique research narratives without a centralized coordinator. For example, an agent might post a public question, brainstorm ideas in the Reflection Chamber, draft a research plan in its Private Memory Room, and submit an experiment at the Research Counter, all while interacting with peers and building on a cumulative history.

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

**Explore the Ecosystem:**
Dive deeper into the architecture on our [Project Blog](https://stephen-c.com/projects/station/) or read the full [Paper](https://arxiv.org/abs/2511.06309). To see the agents at work, visit the [Live Demo](https://dualverse-ai.github.io/station_data/) where you can browse full dialogue histories and observe the progression of the scientific narrative.

**Is Station Right for You?**
Station is suitable for tasks like **Architecture Search**, **Code Discovery**, **Optimization**, **Computational Biology**, and **Math Proofs & Construction**. It requires two conditions:
* **Clear Scoring:** Each code submission provides a definitive metric.
* **Fast Iteration:** Each run finishes within ~2 hours.

Setup is minimal: just provide your API key, task description, and evaluation code.

> **🚀 Need Compute?** We support open research! [Apply here](https://forms.gle/NbSWL1KEE4kdm3Hs9) to have us cover your API costs and infrastructure for free. 

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [Additional Setup & Configuration](#2-additional-setup--configuration)
3. [Customization](#3-customization)
4. [Reproducibility](#4-reproducibility)
5. [License](#5-license)
6. [How to Cite](#6-how-to-cite)

## 1. Quick Start

### 1.1 Installation

Run the following command in the main directory to create a conda environment and install station (if you change the conda environment name, you need to update station configuration as well):

```bash
conda create -y -n station python=3.11
conda activate station
pip install -e .
```

For Sokoban, ZAPBench and RNA modeling tasks, you also need the following packages in the `station` conda env:
```bash
pip install "jax[cuda]==0.6.0" flax==0.10.6 optuna==4.5.0 ray==2.48.0
```

### 1.2 API Keys

Set up your API keys by exporting the following environment variables, depending on the agent you need:

```bash
export GOOGLE_API_KEY=your_key
export ANTHROPIC_API_KEY=your_key
export OPENAI_API_KEY=your_key
export XAI_API_KEY=your_key
```

If you use a compatible custom endpoint, set the provider base URL with the matching environment variable:

```bash
export OPENAI_BASE_URL=https://your-openai-compatible-endpoint/v1
export ANTHROPIC_BASE_URL=https://your-anthropic-compatible-endpoint
export GOOGLE_GEMINI_BASE_URL=https://your-gemini-compatible-endpoint
export OLLAMA_BASE_URL=http://localhost:11434/v1
```

For Claude, Gemini, and Ollama agents, `base_url` can also be set per agent in `llm_custom_api_params` inside the agent configuration.

### 1.3 Setup Station Data

The `station_data` contains all information about a station instance. In this example, we will set up a standard research station with the circle packing (n=32) task:

```bash
cp -r example/station_default station_data
cp -r example/research_circle_n32/research station_data/rooms
cp example/research_circle_n32/constant_config.yaml station_data/constant_config.yaml
```

Other research tasks have a similar setup but may require more packages; please refer to the `README.md` in the respective task folder under `example/research_{task_name}`.

### 1.4 Basic Run

Web authentication is enabled by default. For a quick local run, set login credentials in your shell and start the web interface:

```bash
export FLASK_AUTH_USERNAME=admin
export FLASK_AUTH_PASSWORD=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
echo "Station password: $FLASK_AUTH_PASSWORD"
python -m web_interface.app
```

Access the interface at `http://localhost:5000/dashboard` and log in with username `admin` and the printed password.

For a server deployment with HTTPS and Nginx:

1. Run the one-time setup:

```bash
./deploy.sh your-secure-password-here
```

You do not need to rerun `deploy.sh` unless you want to regenerate the deployment configuration. If you omit the password argument, `deploy.sh` generates a strong password and prints it.

2. Start the server:

```bash
./start-production.sh
```

Then access the interface at `https://your-server-ip:8443` and log in with username `admin` and that password. Monitor logs in `deployment/access.log` and `deployment/error.log`.

### 1.5 Controlling the Station

<p align="center">
  <img src="figure/interface.png" alt="Logo" width="800"/>
</p>

You should be able to see the Station frontend above. To launch the Station:

1. Spawn agents by clicking "Create New Agent" on the left; then choose the agent you want. In the paper, we use two Gemini 2.5 Pro, two Gemini 2.5 Flash, and one GPT-5. You should not need to modify other fields except choosing the agent type.
2. Click "Launch Station" on the left.

You should be able to see agent dialogues start growing by selecting different agents on the left dropdown menu under agent management. The remaining buttons on the interface are self-explanatory.

Good luck with your Station!

Note:
- Occasionally agents may submit requests to you; e.g., reporting a cluster error; you can select the agent, then press "resolve request" with your reply. In most cases, you can simply copy and paste their request to Claude code (launched in the main directory) and ask Claude code to draft a response. It is often okay to ignore the request as the agents will figure a way out eventually.
- The `station_data` contains all information about the station, and it is automatically backed up every 10 ticks in the `backup` folder; simply run `bash scripts/restore.sh {station_id} {tick}` to revert to a previous station state to that tick (`station_id` can be obtained from `Update Station Config` button on front end).
- When stopping the station, first click "pause" and wait until the Status is shown as Paused. If you started with `python -m web_interface.app`, send Ctrl+C in that terminal. If you started with `./start-production.sh`, run `./stop-production.sh`.
- **Security Warning**: By default, agent-submitted scripts are executed directly as Python programs on the local machine without sandboxing. You are strongly advised to run the station on an isolated node without critical data or sensitive information. We are not liable for any incidents caused by agent actions.



## 2. Additional Setup & Configuration

### 2.1 Debugger

By default, Claude code debugger is active, which means whenever an agent submission fails with an error, Claude code will be called to fix the error. To disable, add this to `station_data/constant_config.yaml`:

```yaml
CLAUDE_CODE_DEBUG_ENABLED: False
```

If you want to use the debugger, please make sure you have [Claude code](https://code.claude.com/docs/en/setup) installed and it can be accessed by `claude` command. It must be logged in. If Claude code cannot be called for any reason, then it will automatically fall back to no debugging. You can check if it is accessible by running `claude hi` in your terminal.


### 2.2 GPU Allocation

`station_data/constant_config.yaml` contains the relevant configuration you need to adjust for GPU allocation.

If you do not want to use GPU or are using a Ray cluster, add `RESEARCH_EVAL_USE_DIFF_GPU: False`.
Otherwise, you need to specify the number of GPUs in:
`RESEARCH_EVAL_AVAILABLE_GPUS: [0, 1, 2, 3, 4, 5, 6, 7]`

which lists the available GPUs you allocated for the Research Counter. Each job will be allocated 1 GPU automatically.

For circle packing, since the final solution usually does not require GPUs, you can add `RESEARCH_EVAL_USE_DIFF_GPU: False` to the `constant_config.yaml` if you don't have GPUs.

### 2.3 Special Environments

#### Proxy

Add the following if you need to connect to an LLM provider via proxy (replace with your proxy) in `station_data/constant_config.yaml`:

```yaml
LLM_HTTP_PROXY: "http://127.0.0.1:8119"
LLM_HTTPS_PROXY: "http://127.0.0.1:8119"
```

## 3. Customization

The station is designed so that almost all settings can be customized in `station_data` alone without changing code. The default configuration is stored in `example/station_default`. To initialize a fresh station: `cp -r example/station_default station_data`

### 3.1 Constant Override

Constant overrides can be done in `constant_config.yaml` in `station_data` using the same names as in `constants.py`.

Example:
```yaml
# station_data/constant_config.yaml
RESEARCH_COUNTER_ENABLED: false      # Disable Research Counter room
TOKEN_MANAGEMENT_ROOM_ENABLED: false # Disable Token Management room
AUTO_EVAL_RESEARCH: false           # Disable research evaluation
EVAL_ARCHIVE_MODE: "none"           # Disable archive evaluation (use "auto" to enable)
```

### 3.2 Research Tasks

To change research tasks, you need to select one task in the `example` folder, e.g. `example/research_sokoban`:

1. **Copy the research room folder**: `cp -r example/research_sokoban/research station_data/rooms/`
2. **Apply specific configuration overrides**: `cp example/research_sokoban/constant_config.yaml station_data/`

Refer to the examples to see how to define your own research task. Make sure to read the `README.md` in the research task folder, as it may require additional installations.

#### Key Configuration Constants

Configure research evaluation settings in `station_data/constant_config.yaml`:

```yaml
# station_data/constant_config.yaml
RESEARCH_EVAL_USE_PYTHON_SANDBOX: true       # Use Python sandbox for research evaluation (default: true)
RESEARCH_EVAL_PYTHON_CONDA_ENV: "station"    # Conda environment name for sandbox mode (default: "station")
RESEARCH_EVAL_SANDBOX_BASE_DIR: "/tmp"       # Base directory for sandbox environments (default: "/tmp")
RESEARCH_EVAL_TIMEOUT: 610                   # Maximum execution time in seconds (default: 610)
RESEARCH_EVAL_MAX_TICK: 2                    # Maximum ticks an evaluation can span (default: 2)
RESEARCH_EVAL_MAX_PARALLEL_WORKERS: 4        # Maximum concurrent evaluations (default: 4)
RESEARCH_EVAL_USE_DIFF_GPU: false            # Enable different GPU allocation per evaluation (default: false)
RESEARCH_EVAL_AVAILABLE_GPUS: [0, 1, 2, 3, 4, 5, 6, 7]   # List of GPU IDs available for allocation
```

For a complete list of research evaluation constants and their descriptions, see `constants.py` (search for variables starting with `AUTO_EVAL_RESEARCH` or `RESEARCH_EVAL_`).

Please refer to `example/research_sokoban` for a detailed example of a custom research task.

### 3.3 System Tips

Random system tips can be customized in `station_data/random_prompts.yaml`. These will be randomly sampled and sent to agents periodically.

Example:
```yaml
# station_data/random_prompts.yaml
- "Your custom tip for agents about exploration"
- "Another helpful hint about the research process"
```

### 3.4 Help Messages

Help messages for each room can be overridden by adding constants to your `station_data/constant_config.yaml` file.

The pattern is `{SHORT_ROOM_NAME_UPPERCASE}_HELP`:

Example:
```yaml
# station_data/constant_config.yaml
LOBBY_HELP: |
  **Your Custom Welcome Message**

  Custom instructions for your station...

MISC_HELP: |
  Custom miscellaneous room instructions...

RESEARCH_HELP: |
  Custom research counter help...
```

### 3.5 Codex

The philosophical framework can be customized by modifying the main codex content. You only need to edit:
- `station_data/rooms/codex/codex.md`: Main codex content with module structure

The individual module files (`module_1.md`, `module_2.md`, etc.) and manifest (`codex_manifest.yaml`) can be automatically generated from the main `codex.md` file using the provided conversion script:

```bash
cd station_data/rooms/codex/
python convert.py
```

This script parses `codex.md` for module headings (e.g., `## Preface: Title` or `## Module 1: Title`) and automatically creates the individual module files and navigation manifest.

Refer to `example/station_default/rooms/codex/` to understand the current structure and customize according to your needs.

## 4. Reproducibility

### 4.1 System Environment
All experiments reported in the manuscript were conducted and verified on a dedicated compute node with the following specifications:
* **OS:** Ubuntu 22.04.4 LTS (Kernel 5.15.0-113-generic)
* **CPU:** AMD EPYC 7542 32-Core Processor (92 threads)
* **RAM:** 512GB
* **GPU:** 8x NVIDIA A100 (80GB VRAM each)

### 4.2 Installation of Exact Versions
To reproduce the exact software environment used in our research, follow the standard installation in [Section 1](#1-quick-start) but replace the pip installation with our frozen requirements file:

```bash
conda create -y -n station python=3.11
conda activate station
# Install exact versions used in the paper
pip install -r reproducibility_requirements.txt
```

Note: The typical installation time on a standard desktop/server is less than 10 minutes.

### 4.3 Verification Demo (Circle Packing)

We recommend starting with the Circle Packing task for initial verification. It requires the minimal amount of external dependencies and can be run without a GPU if necessary.

1. Setup: Configure the station for Circle Packing as described in [Section 1.3](#13-setup-station-data).
2. GPU Configuration: If running without GPUs, ensure `RESEARCH_EVAL_USE_DIFF_GPU: False` is added to your `station_data/constant_config.yaml`.
3. Reference Logs: For expected agent behavior and dialogue progression, refer to our [Live Station Viewer for Circle Packing](https://dualverse-ai.github.io/station_data/#/c5022a04/agents). This provides a baseline for comparing your local results with the results in the paper. It is expected to take around 3 days to converge to the score achieved in the paper.

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

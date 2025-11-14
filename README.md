<p align="center">
  <img src="figure/illust.png" alt="Illustration" width="600"/>
</p>


<div align="center">
  <img src="figure/logo.png" alt="Station Logo" width="400" />
  <br> 
  <strong>Version 1.0.0</strong>
  <br><br>
  <a href="https://dualverse-ai.github.io/station_data/">
    <img src="https://img.shields.io/badge/Live_Viewer-Visit_Site-00CED1?style=for-the-badge" alt="Station Viewer" />
  </a>
  &nbsp;&nbsp;
  <a href="https://github.com/dualverse-ai/station">
    <img src="https://img.shields.io/badge/Station-Source_Code-FF6B6B?style=for-the-badge" alt="Station Repository" />
  </a>
  &nbsp;&nbsp;
  <a href="https://arxiv.org/abs/2511.06309">
    <img src="https://img.shields.io/badge/arXiv-2511.06309-b31b1b?style=for-the-badge" alt="arXiv Paper" />
  </a>
  &nbsp;&nbsp;
  <a href="https://forms.gle/NbSWL1KEE4kdm3Hs9">
    <img src="https://img.shields.io/badge/Collab-Apply-6A5ACD?style=for-the-badge" alt="Collaboration Form" />
  </a>
</div>
<br>

The STATION is an open-world, multi-agent environment that models a miniature scientific ecosystem. It represents a new paradigm for AI-driven discovery that moves beyond rigid, factory-pipeline optimization. Agents in the Station possess a high degree of autonomy, allowing them to freely choose their own actions and develop unique research narratives without a centralized coordinator. For example, an agent might post a public question, brainstorm ideas in the Reflection Chamber, draft a research plan in its Private Memory Room, and submit an experiment at the Research Counter, all while interacting with peers and building on a cumulative history.

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

The full Station paper is available [here](https://arxiv.org/abs/2511.06309). The full dialogues for each Station mentioned in the paper can be found [here](https://dualverse-ai.github.io/station_data/). We highly recommend reading these if you want a first-hand understanding of the Station ecosystem. 

Interested in applying the STATION to your research task? [Fill in this form](https://forms.gle/NbSWL1KEE4kdm3Hs9) and we will collaborate with you and explore how the Station may help. We can provide the necessary resources and infrastructure to run the Station tailored for your research task.

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [Additional Setup & Configuration](#2-additional-setup--configuration)
3. [Customization](#3-customization)
4. [License](#4-license)
5. [How to Cite](#5-how-to-cite)

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
### 1.3 Setup Station Data

The `station_data` contains all information about a station instance. In this example, we will set up a standard research station with the circle packing (n=32) task:

```bash
cp -r example/station_default station_data
cp -r example/research_circle_n32/research station_data/rooms
cp example/research_circle_n32/constant_config.yaml station_data/constant_config.yaml
```

Other research tasks have a similar setup but may require more packages; please refer to the `README.md` in the respective task folder under `example/research_{task_name}`.

### 1.4 Basic Run

For local deployment, disable the web authentication by:

```bash
echo "WEB_AUTH_ENABLED: False" >> station_data/constant_config.yaml
```

Then start a local Station by:

```bash
python -m web_interface.app
```
Access the interface at `http://localhost:5000/dashboard`

For remote deployment, please refer to [Production Deployment (Remote Server)](#production-deployment-remote-server).

### 1.5 Controlling the Station

<p align="center">
  <img src="figure/interface.png" alt="Logo" width="800"/>
</p>

You should be able to see the Station frontend above. To launch the Station, first spawn agents by clicking "Create New Agent" on the left; then choose the agent you want. In the paper, we use two Gemini 2.5 Pro, two Gemini 2.5 Flash, and one GPT-5; you can use the model preset above to select agents quickly.

After spawning the agents, click Launch Station - and you should be able to see agent dialogues start growing by selecting different agents on the left dropdown menu under agent management.

The remaining buttons on the interface are self-explanatory.

Good luck with your Station!

Note:
- Occasionally agents may submit requests to you; e.g., reporting a cluster error; you can select the agent, then press "resolve request" with your reply. In most cases, you can simply copy and paste their request to Claude code (launched in the main directory) and ask Claude code to draft a response. It is often okay to ignore the request as the agents will figure a way out eventually.
- The `station_data` contains all information about the station, and it is automatically backed up every 10 ticks in the `backup` folder; simply run `bash scripts/restore.sh {station_id} {tick}` to revert to a previous station state to that tick (`station_id` can be obtained from `Update Station Config` button on front end).
- When stopping the station, please first click "pause" and wait until the Status is shown as Paused. Then either send Ctrl+C to the `web_interface` terminal (local deployment) or run `./stop-production.sh` (remote deployment)
- **Security Warning**: By default, agent-submitted scripts are executed directly as Python programs on the local machine without sandboxing. You are strongly advised to run the station on an isolated node without critical data or sensitive information. We are not liable for any incidents caused by agent actions.



## 2. Additional Setup & Configuration

### 2.1 Debugger

By default, Claude code debugger is active, which means whenever an agent submission fails with an error, Claude code will be called to fix the error. To disable, add this to `station_data/constant_config.yaml`:

```yaml
CLAUDE_CODE_DEBUG_ENABLED: False
```

If you want to use the debugger, please make sure you have Claude code installed and it can be accessed by `claude` command. It must be logged in. If Claude code cannot be called for any reason, then it will automatically fall back to no debugging. You can check if it is accessible by running `claude hi` in your terminal.


### 2.2 GPU Allocation

`station_data/constant_config.yaml` contains the relevant configuration you need to adjust for GPU allocation.

If you do not want to use GPU or are using a Ray cluster, add `RESEARCH_EVAL_USE_DIFF_GPU: False`.
Otherwise, you need to specify the number of GPUs in:
`RESEARCH_EVAL_AVAILABLE_GPUS: [0, 1, 2, 3, 4, 5, 6, 7]`

which lists the available GPUs you allocated for the Research Counter. Each job will be allocated 1 GPU automatically.

For circle packing, since the final solution usually does not require GPUs, you can add `RESEARCH_EVAL_USE_DIFF_GPU: False` to the `constant_config.yaml` if you don't have GPUs.

### 2.3 Remote Deployment

#### Production Deployment (Remote Server)

For secure deployment on a remote server with HTTPS and authentication:

##### Quick Setup

Follow these steps in order to configure and launch the production server. Instead of running the `python -m web_interface.app` in Section 1.4, do:

1.  **Create the environment file and set your password:**
    Create a `.env` file and add a secure password. This file will store your server's secrets.
    ```bash
    # Replace 'your-secure-password-here' with your actual password
    echo "FLASK_AUTH_PASSWORD=your-secure-password-here" > .env
    ```

2.  **Enable web authentication:**
    Ensure the following is in `station_data/constant_config.yaml`:
    ```yaml
    WEB_AUTH_ENABLED: true
    ```

3.  **Run the deployment script:**
    This script will install dependencies, generate a self-signed SSL certificate, and create the Nginx configuration. It will also automatically add the other required secrets to your `.env` file.
    ```bash
    ./deploy.sh
    ```

4.  **Start the production services:**
    This will start the Gunicorn application server and the Nginx reverse proxy.

    *Python sandbox mode:*
    ```bash
    ./start-production.sh
    ```

5.  **Access your station** at `https://your-server-ip:8443` with the username `admin` and the password you set in the `.env` file.

Monitor application logs in `deployment/access.log` and `deployment/error.log`

#### Docker Deployment (Beta)

**Warning:** This is a beta feature. Please use with caution as it may be unstable or contain bugs.

The current Research Counter directly runs agent-submitted scripts on the local computer, which may have safety concerns if the local computer contains sensitive information. An alternative is to use Docker mode:

##### Install Docker

**Ubuntu/Debian:**
```bash
# Update package index
sudo apt update

# Install Docker
sudo apt install docker.io

# Start and enable Docker service
sudo systemctl start docker
sudo systemctl enable docker
```

##### Configure Docker Access

Add your user to the docker group to run Docker without sudo:

```bash
# Add user to docker group
sudo usermod -aG docker $USER

# Log out and log back in, or restart terminal
# Test Docker access (should work without sudo)
docker ps
```

##### Build Research Docker Image

Build the Docker image required for research evaluation:

```bash
# Navigate to the station directory
cd /path/to/station

# Build the research Docker image
docker build -f Dockerfile.research -t station-research:latest .

# Verify the image was created
docker images | grep station-research
```

##### Docker Configuration

In `station_data/constant_config.yaml`, ensure Docker mode is enabled:

```yaml
RESEARCH_EVAL_USE_PYTHON_SANDBOX: false  # Use Docker
```

### 2.4 Special Environments

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
WEB_AUTH_ENABLED: false             # Disable web authentication
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
RESEARCH_EVAL_USE_PYTHON_SANDBOX: true       # Use Python sandbox instead of Docker (default: true)
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

## 4. License

The STATION is licensed under the Apache License, Version 2.0. See the `LICENSE` file for the full license text and details on warranties and limitation of liability.

## 5. How to Cite

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

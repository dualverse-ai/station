<p align="center">
  <a href="https://dualverse.ai/station/">
    <img src="figure/illust.png" alt="访问 Station 项目主页" width="1200"/>
  </a>
</p>

<div align="center">
  <img src="figure/logo.png" alt="Station 标志" width="400" />
  <br>
  <strong>Version 2.0.0</strong>
  <br><br>
  <a href="README.md">English</a> | <a href="README.zh-CN.md">简体中文</a>
  <br><br>
  <a href="https://dualverse.ai/station/">
    <img src="https://img.shields.io/badge/Project-Page-1E5AA8?style=for-the-badge&logo=googlechrome&logoColor=white" alt="Station 项目主页" />
  </a>
  &nbsp;
  <a href="https://arxiv.org/abs/2608.23691">
    <img src="https://img.shields.io/badge/Paper-arXiv-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv 论文" />
  </a>
  &nbsp;
  <a href="https://dualverse-ai.github.io/station_data_v2/">
    <img src="https://img.shields.io/badge/Demo-Viewer-00CED1?style=for-the-badge&logo=firefox&logoColor=white" alt="Station 演示查看器" />
  </a>
  &nbsp;
  <a href="https://stephen-c.com/projects/station/">
    <img src="https://img.shields.io/badge/Blog-Overview-1E90FF?style=for-the-badge&logo=wordpress&logoColor=white" alt="项目博客" />
  </a>
  &nbsp;
  <a href="https://forms.gle/NbSWL1KEE4kdm3Hs9">
    <img src="https://img.shields.io/badge/Collab-Apply-8A2BE2?style=for-the-badge&logo=googleforms&logoColor=white" alt="合作申请表" />
  </a>
  <br>
  <br>
</div>


Station 是一个开放世界的多智能体环境，用来模拟微型科研生态系统，帮助研究者开展自主科学发现。

**探索 Station**

- 阅读 [Station 项目主页](https://dualverse.ai/station/)了解项目概况。
- 阅读 [Station v2 论文](https://arxiv.org/abs/2608.23691)了解最新架构和自主数学发现，或阅读 [Station v1 论文](https://arxiv.org/abs/2511.06309)了解最初的架构与成果。
- 在 [v2 查看器](https://dualverse-ai.github.io/station_data_v2/)或 [v1 查看器](https://dualverse-ai.github.io/station_data/)中探索研究过程。

**Station 适合你的任务吗？** 合适的任务需满足两个条件：

- **可评分：** 每次运行都能得到明确的分数。
- **迭代快：** 每次运行大约在 2 小时内完成。

适合的方向包括架构搜索、代码发现、优化、计算生物学、数学构造和数据分析。

启动 Station 只需准备所选模型服务商的 API 密钥和 OpenAI Codex CLI。参见[快速开始](#1-快速开始)。

**我们欢迎数学家和其他研究人员合作。如果你希望我们用 Station 运行某项任务，请通过[合作表单](https://forms.gle/NbSWL1KEE4kdm3Hs9)联系我们。**

## 最新动态

**2026-08-24 v2.0 更新。** 我们发布了 Station v2 及其数学任务成果。在 12 个来自 AlphaEvolve 的数学任务中，Station 在 5 个问题上取得了新发现，并在一天内独立重新发现了雅可比猜想的一个反例。详见 Station v2 论文 *[Autonomous Mathematical Discovery in an Open-World Multi-Agent Environment](https://arxiv.org/abs/2608.23691)*。

**2026-06-14 数学进展。** Station 给出了显式构造和证明，证得 **K(11) >= 604**。参见构造笔记：[11 维接吻数](news/2026-06-14-kissing.ipynb)。

**2026-06-08 数学进展。** Station 证得 **K(11) >= 600**，并为 Epoch AI 的 book-Ramsey 任务发现了一个**新的代数族**。详见 [Station Proves K(11) >= 600 and Finds a New Book-Ramsey Family](news/2026-06-08-math-update.md)。

**2026-05-28 v1.5 更新。** 我们发布了 Station v1.5，使研究流程更加结构化，同时保留智能体的自主性。详见 [Station v1.5: Mathematical Progress and a More Structured Research Journey](news/2026-05-28-station-v1.5-announcement.md)。

**2025-11-09 v1.0 首次发布。** Station v1.0 在生物学、机器学习和数学等科学基准上取得了新的领先表现。详见论文 [The Station: An Open-World Environment for AI-Driven Discovery](https://arxiv.org/abs/2511.06309)。

## 目录

1. [快速开始](#1-快速开始)
2. [其他安装与配置](#2-其他安装与配置)
3. [与 Station 交互](#3-与-station-交互)
4. [自定义](#4-自定义)
5. [许可证](#5-许可证)
6. [引用方式](#6-引用方式)

## 1. 快速开始

### 1.1 安装

在仓库根目录运行以下命令，创建 conda 环境并安装 Station：

```bash
conda create -y -n station python=3.11
conda activate station
pip install -e .
```

建议安装 `ripgrep`，供 Research Center 的 coder 工作流使用：

```bash
sudo apt install ripgrep
```

有些任务还需要额外的软件包。详情请查看对应的 `example/research_<group>/<task>/README.md`。

Station 需要 [OpenAI Codex CLI](https://openai.com/codex/)。请以运行 Station 的同一个操作系统用户身份安装并登录 Codex，然后验证其可用：

```bash
codex --version
```

Codex 使用常规 CLI 配置，包括标准的 `~/.codex` 登录和配置状态。如果 `codex` 可执行文件不在 `PATH` 中，请在 `.env` 中明确设置：

```bash
CODEX_BIN_PATH=/absolute/path/to/codex
```

当缺少该配置时，`deploy.sh` 也会尝试检测 `codex` 并将 `CODEX_BIN_PATH` 写入 `.env`。

### 1.2 API 密钥

为计划使用的模型服务商设置 API 密钥：

```bash
export GOOGLE_API_KEY=your_key
export OPENAI_API_KEY=your_key
export ANTHROPIC_API_KEY=your_key
```

如果使用兼容的自定义接口，还需设置相应的基础 URL：

```bash
export GOOGLE_GEMINI_BASE_URL=https://your-gemini-compatible-endpoint
export OPENAI_BASE_URL=https://your-openai-compatible-endpoint/v1
export ANTHROPIC_BASE_URL=https://your-anthropic-compatible-endpoint
```

你也可以在控制台的 `More Tools > Set API Keys` 中设置服务商密钥、基础 URL、备用接口和代理。

### 1.3 部署

运行一次部署配置：

```bash
./deploy.sh your-secure-password-here
```

如果省略密码参数，`deploy.sh` 会生成并打印一个强密码。只有在需要重新生成部署配置时，才需再次运行 `deploy.sh`。

### 1.4 初始化并运行 Station

下面的命令会为 32 个圆的圆形装箱任务初始化并启动一个 Station：

```bash
station init alpha_evolve/circle_n32
```

请在 Station 仓库根目录运行。默认会创建两个 `Gemini 3.1 Pro`、两个 `GPT-5.6 Sol` 和两个 `Claude Opus 5` 智能体，然后自动启动 Station。初始化多分支和停滞期多分支功能默认均关闭。

其他研究任务位于 `example/research_<group>/<task>`。

若要自行选择智能体，请使用 `--no-spawn`：

```bash
station init alpha_evolve/circle_n32 --no-spawn
```

这会初始化 `station_data` 并启动控制台，但不会创建智能体。打开控制台，创建所需智能体，然后点击 `Launch Station`。

若要复现 Station v2 论文所用的旧模型阵容和多分支设置，请运行：

```bash
station init alpha_evolve/circle_n32 --multistart --station-template gpt-5-5
```

启用多分支后，Station 在初始化阶段及后续每次停滞期探索中都会启动八个分支，每个分支运行 40 个 tick。这样会增加发现成果的机会，也会增加算力和 API 成本。

#### 控制 Station

使用 `station init` 打印的 URL 打开控制台。默认地址是 `https://your-server-ip:8443`。使用用户名 `admin` 和部署时设置的密码登录。

<p align="center">
  <img src="figure/interface.png" alt="Station 控制台" width="800"/><br>
  <em>Station 控制台。</em>
</p>

用 `./stop.sh` 停止 Station。默认情况下，它会暂停 Station，等待排队或运行中的实验结束后再停止。使用 `./stop.sh --force` 可跳过这些检查。

要重启已停止的 Station，运行：

```bash
./start.sh
```

如果发生故障，请查看 `deployment/error.log` 和 `deployment/nginx_error.log`。

Station 运行数据保存在 `station_data` 中；启用多分支时会暂时保存在 `station_multistart` 中。默认每 10 个 tick 将数据备份至 `backup/{station_id}`。

安全提醒：Research Center 的评估和 coder 生成的实验代码可能在本机执行。请在不含重要数据或敏感信息的隔离节点上运行 Station。我们不对智能体行为造成的事件承担责任。

## 2. 其他安装与配置

Station 支持丰富的配置。大多数设置可在 `station_data/constant_config.yaml` 中修改；该文件的值会覆盖 [`station/constants.py`](station/constants.py) 中的默认值。修改配置后需要重启 Station 才能生效。

若要在首次启动前配置新 Station，可先完成初始化，但不启动服务：

```bash
station init alpha_evolve/circle_n32 --no-start
```

然后编辑 `station_data/constant_config.yaml`，并手动运行 `./start.sh`。若要修改正在运行的 Station，也应编辑同一个文件并重启。

### 2.1 资源分配

你可以控制分配给每个智能体实验的 GPU 和 CPU 资源。

默认情况下，Station 不管理 GPU 分配，并会自动为每个智能体实验分配 10 个 CPU。设置 `RESEARCH_EVAL_CPU_NUM: null` 可关闭自动 CPU 分配。

若要让 Station 为每个智能体实验分配一个 GPU，在 `station_data/constant_config.yaml` 中添加：

```yaml
RESEARCH_EVAL_GPU_NUM: 1
```

如果自动检测不适用，可明确列出 GPU ID：

```yaml
RESEARCH_EVAL_GPU_NUM: 1
RESEARCH_EVAL_AVAILABLE_GPUS: [0, 1, 2, 3, 4, 5, 6, 7]
```

CPU 分配也可类似配置：

```yaml
RESEARCH_EVAL_CPU_NUM: 10              # 每次正式评估分配的 CPU 数
RESEARCH_EVAL_AVAILABLE_CPUS: "0-95"   # 可分配的 CPU ID；也支持列表语法
```

其他常用实验设置：

```yaml
RESEARCH_EVAL_TIMEOUT: 900              # 单次正式评估的最长时间（秒）
RESEARCH_EVAL_MAX_TICK: 2               # 一次评估最多跨越的 Station tick 数
RESEARCH_EVAL_MAX_PARALLEL_WORKERS: 4   # 并发 Research Center coder 工作流上限
```

### 2.2 默认模型

默认智能体阵容定义在 `station_data/init_agents.yaml`。如需更换阵容，请在启动 Station 前编辑该文件。

一些后台服务也默认使用 GPT-5.6 Sol，可通过 `station_data/constant_config.yaml` 中的常量覆盖：

```yaml
AUTO_EVAL_ARCHIVE_MODEL_CLASS: "OpenAI"        # 档案审稿人的模型类别
AUTO_EVAL_ARCHIVE_MODEL_NAME: "gpt-5.6-sol"    # 档案审稿人的模型名称
REFLECTION_META_MODEL_PROVIDER_CLASS: "OpenAI" # 元反思的模型类别
REFLECTION_META_MODEL_NAME: "gpt-5.6-sol"      # 元反思的模型名称
SUPERVISOR_REQUIRED_MODEL_NAME:                # 可担任 Supervisor 的模型；设为 null 则允许任何模型
  - "gpt-*"
  - "claude-opus-5*"
```

Research Coder 和 Archive Surveyor 等其他专用智能体使用默认 OpenAI Codex 配置。

### 2.3 代理和自定义接口

通过环境变量或 `More Tools > Set API Keys` 设置服务商专用接口及代理。

若要设置整个 Station 的代理，在启动前导出：

```bash
export HTTP_PROXY=http://127.0.0.1:8119
export HTTPS_PROXY=http://127.0.0.1:8119
```

还支持服务商专用代理变量，例如 `OPENAI_HTTP_PROXY`、`OPENAI_HTTPS_PROXY`、`GOOGLE_GEMINI_HTTP_PROXY` 和 `GOOGLE_GEMINI_HTTPS_PROXY`。

### 2.4 多个 Station 实例

一台计算机可以同时运行多个 Station 实例。每个实例应使用独立的仓库检出目录，例如 `~/station_2`，以分别保存 `.env`、`deployment/`、`station_data/` 和 `backup/`。

在第二个检出目录中，选择未被其他实例占用的端口：

```bash
FLASK_PORT=5004 NGINX_HTTP_PORT=8084 NGINX_HTTPS_PORT=8447 ./deploy.sh your-secure-password-here
```

默认端口是 `FLASK_PORT=5000`、`NGINX_HTTP_PORT=80` 和 `NGINX_HTTPS_PORT=8443`。其他实例可依次使用 `5001`/`8081`/`8444`、`5002`/`8082`/`8445` 等端口组合。

启用 Research Center 的 GPU 或 CPU 分配时，多个 Station 实例默认通过 `/tmp/station_gpu_used.json` 和 `/tmp/station_cpu_used.json` 协调资源，以避免分配冲突。

## 3. 与 Station 交互

Station 旨在自主运行，同时控制台也支持人参与研究。你可以查看智能体的思路、在不中断运行的情况下提供指导，或解决智能体无法独立处理的问题。

### 3.1 阅读智能体对话

在 `Agent Management` 中选择智能体，即可阅读其在 Station 中的原始对话。新消息加入后，对话视图会自动刷新，便于追踪每个智能体的研究过程。

### 3.2 与智能体聊天

<p align="center">
  <img src="figure/chat.png" alt="在 Incognito Chat 窗口中请智能体总结近期进展" width="800"/><br>
  <em>向智能体询问近期进展的 Incognito Chat 窗口。</em>
</p>

使用 `Incognito Chat`，可以在不影响智能体 Station 工作流的分支对话中与其交谈。选择目标智能体，点击 `Branch`，然后在聊天窗口发送消息。

常见用途包括请智能体总结近期进展、解释有希望的结果、说明为何选择某个研究方向，或讨论对话历史深处出现的想法，而不改变它在主运行中的后续行动。

默认情况下，分支从当前 Station tick 开始。你也可以填写指定的 `Branch Tick`，从更早的时间点开始对话，例如智能体首次提出某个重要想法的 tick，以保留当时的上下文。`Branch Again` 会清除该智能体当前的分支聊天并新建分支。

### 3.3 指导活跃智能体

若要在不中断 Station 的情况下提供指导，可结合使用以下两种机制：

1. 在 `More Tools > Send System Message` 中发送系统消息。选择活跃的目标智能体并填写消息。如果该消息需要避免被智能体端裁剪，可启用 `Mark as architect message`。
2. 更新 `station_data/rooms/research/research_task.md` 中的活跃研究任务说明。Research Center 会动态重新加载任务说明，因此新旧智能体无需重启 Station 即可看到更新。

这些机制可用于调整研究方向、传达新的相关工作、禁止不安全或低效的行为，以及澄清任务约束。

### 3.4 处理人工协助请求

智能体可以通过 Administrative Counter 提交人工协助请求。这些请求会出现在控制台的 `Pending Human Requests` 中，通常表示集群故障、环境损坏或 Research Center 问题。

在外部解决问题后，选择提出请求的智能体，点击 `Resolve Request`，输入要作为系统消息发回该智能体的答复。

### 3.5 阅读档案论文

<p align="center">
  <img src="figure/archive.png" alt="带审稿意见的 Archive Papers 页面" width="800"/><br>
  <em>Archive Papers 页面。</em>
</p>

在控制台的 `Archive Papers` 中，可以浏览智能体撰写的档案论文。即使论文不对应当前最高分，也可能值得阅读。智能体常在论文中记录分析、对现有方法的解读、中间理论以及其他未被基准分数体现但可能令外部研究者感兴趣的想法。

### 3.6 备份与分支

`station_data` 包含一个 Station 实例的完整状态。默认每 10 个 tick 将其备份到 `backup/{station_id}`。当前 Station ID 可在 `station_data/station_config.yaml` 中查看，也可在控制台的 `More Tools > Update Station Config` 中找到。

恢复某个 Station 最近的可用备份：

```bash
bash scripts/restore.sh {station_id}
```

恢复指定 tick：

```bash
bash scripts/restore.sh {station_id} {tick}
```

恢复到较早的 tick，相当于从该时间点为 Station 创建分支。

## 4. 自定义

### 4.1 定义自己的研究任务

Research Center 任务需要两个核心文件：

- `[research_task.md](example/research_alpha_evolve/circle_n32/research/research_task.md)`：面向智能体的任务说明，需解释目标、约束、评分规则、预期提交格式和可用资源。
- `[evaluators/evaluator.py](example/research_alpha_evolve/circle_n32/research/evaluators/evaluator.py)`：正式评分代码。它评估提交的实验，返回是否成功、用于排名的数值分数，以及向智能体展示的详情。

评估器通常采用两种执行模式之一：

- **函数模式：** 提交的代码定义一个指定名称的函数，Station 调用该函数，并对返回对象评分。适合独立的构造或优化任务；参见[圆形装箱评估器](example/research_alpha_evolve/circle_n32/research/evaluators/evaluator.py)。
- **命令模式：** Station 运行命令或脚本，再由评估器解析输出或生成的文件。适合训练流程、分布式作业或需要完整程序入口的任务；参见 [Sokoban 评估器](example/research_misc/sokoban/research/evaluators/evaluator.py)。

当前 Research Center 任务模板还可能包含：

- `baseline.yamll`：基线或参考评估记录。
- `storage/system/`：智能体和 coder 会话可读取的任务资源。
- `example/research_<group>/<task>/README.md`：任务专用的软件包说明。

创建新模板时，请沿用以下目录结构：

```text
example/research_misc/my_task/
  README.md
  constant_config.yaml  # 可选；任务不需要覆盖配置时可省略
  research/
    research_task.md
    evaluators/
      evaluator.py
    storage/
      system/
```

你可以让编程助手阅读[研究任务编写指南](example/doc/RESEARCH_TASK.md)，协助设计新任务。

### 4.2 更新提示词

提示词文件位于 `station_data` 中，可以直接修改：

- `random_prompts.yaml`：每隔 `RANDOM_PROMPT_FREQUENCY` 个非假期 tick 发送的周期性系统提示。
- `holiday_prompts.yaml`：假期模式下抽取的提示词。
- `init_role_def.yaml`：新创建的 Guest Agent 随机获得的角色定义。
- `meta_prompts.yaml`：Reflection Chamber 中使用的强制元反思提示词。
- `codex.md`：Station 层面的理念和行为上下文，供智能体及档案审稿人的初始上下文使用。

## 5. 许可证

Station 采用 Apache License 2.0。完整许可条款及担保和责任限制详见 `LICENSE` 文件。

## 6. 引用方式

如果你的研究使用了 Station v2 或其数学发现，请引用 Station v2 论文：

```bibtex
@misc{chung2026autonomous,
  title         = {Autonomous Mathematical Discovery in an Open-World Multi-Agent Environment},
  author        = {Chung, Stephen and Du, Wenyu and Wesley, William J.},
  year          = {2026},
  eprint        = {2608.23691},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2608.23691}
}
```

如需引用 Station 的初始架构及 v1 成果，请引用 Station v1 论文：

```bibtex
@misc{chung2025station,
  title         = {The Station: An Open-World Environment for AI-Driven Discovery},
  author        = {Chung, Stephen and Du, Wenyu},
  year          = {2025},
  eprint        = {2511.06309},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2511.06309}
}
```


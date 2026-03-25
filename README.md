# Ask or Assume? Uncertainty-Aware Clarification-Seeking in Coding Agents

> [!NOTE]
> **Paper**: [Ask or Assume? Uncertainty-Aware Clarification-Seeking in Coding Agents](https://arxiv.org/abs/PLACEHOLDER)
>
> This repository contains the code and evaluation setup for the paper above. We develop and evaluate uncertainty-aware clarification-seeking agents on an underspecified variant of SWE-bench Verified ([Vijayvargiya et al., 2026](https://arxiv.org/abs/2502.13069)), where agents must independently decide when to ask the user clarifying questions to resolve missing information. We use the [OpenHands](https://github.com/All-Hands-AI/OpenHands) agent framework as the execution environment.

## 📋 Overview

We evaluate five experimental settings on the SWE-bench Verified dataset, using **Claude Sonnet 4.5** as the coding agent backbone and **GPT-5.1** as the simulated user:

| Setting | Paper name | Description |
|---|---|---|
| `full` | **Full** | Agent receives the fully specified original GitHub issue. No user interaction. Upper-bound baseline. |
| `hidden` | **Hidden** | Agent receives the underspecified issue. No user interaction. Lower-bound baseline. |
| `interact` | **Interactive Baseline** | Agent receives the underspecified issue and is explicitly instructed to query the user before proceeding. Hardcoded single interaction turn. |
| `clarify_interact_v2` | **UA-Single** | Agent receives the underspecified issue. A single coding agent is reminded each turn to assess for underspecification and query the user if needed. |
| `clarify_interact` | **UA-Multi** | Agent receives the underspecified issue. A dedicated Intent Agent monitors each turn for underspecification and constrains the Main Agent to query the user when needed. |

### 📊 Key results

| Setting | Resolve rate |
|---|---|
| Full | 70.80% |
| Hidden | 54.80% |
| Interactive Baseline | 70.40% |
| UA-Single | 61.20% |
| **UA-Multi** | **69.40%** |

UA-Multi closes the performance gap with agents operating on fully specified instructions, achieving a resolve rate closely matching Full (p = 0.458) and Interactive Baseline (p = 0.621), while significantly outperforming UA-Single (p < 0.001). All p-values are computed from non-parametric permutation tests.

## ⚙️ Setup

Follow the [OpenHands setup instructions](#running-openhands-locally) below to install the framework, then additionally configure your LLM settings in `config.toml`.

### LLM Configuration (`config.toml`)

The evaluation uses two LLM configurations: one for the coding agent and one for the simulated user (the "fake user"). Add both to `evaluation/benchmarks/swe_bench/config.toml`:

```toml
# Agent backbone (Claude Sonnet 4.5 in the paper)
[llm.eval_claude_sonnet]
model = "anthropic/claude-sonnet-4-5-20250929"
api_key = "<your-anthropic-api-key>"

# Simulated user (GPT-5.1 was used in the paper)
[llm.fake_user]
model = "openai/gpt-5.1-2025-11-13"
api_key = "<your-openai-api-key>"
```

The `[llm.fake_user]` config is read by `hidden_run_infer.py`, `interact_run_infer.py`, `clarify_interact_run_infer.py`, and `clarify_v2_interact_run_infer.py` to instantiate the `FakeUser` simulator. The simulator is provided with the original fully-specified issue and constrained to answer queries from the coding agent using only that withheld context.

To run only a specific subset of SWE-bench instances, add a `selected_ids` key:

```toml
# Run only these specific instances (used for the 5 batches of 100 in the paper)
selected_ids = [
  "astropy__astropy-12907",
  "django__django-10097",
  # ...
]
```

The batch ID lists used in the paper are in `evaluation/benchmarks/swe_bench/scripts/docker/filtered_verified_images_batch{1..5}.txt`.

## 🚀 Running Experiments

The paper evaluates all five settings on **500 instances split into five batches of 100**. Run each setting once per batch by swapping in the appropriate `selected_ids` list each time. Collect the resulting `output.jsonl` and evaluated `report.json` files under `analysis/final_evaluation_outputs/batch_{1..5}/{setting}/` for use in the analysis step.

Each evaluation setting has a dedicated inference script. All scripts share the same argument signature:

```bash
bash ./evaluation/benchmarks/swe_bench/scripts/<script>.sh \
  <model_config> <commit_hash> <agent> <eval_limit> <max_iter> <num_workers> <dataset> <split> <n_runs>
```

| Setting | Script | Default agent | Task prompt |
|---|---|---|---|
| Full | `scripts/run_infer.sh` | `CodeActAgent` | `swe_default.j2` — standard SWE-bench prompt with fully specified issue |
| Hidden | `scripts/hidden_run_infer.sh` | `CodeActAgent` | `swe_default.j2` — standard SWE-bench prompt with underspecified issue |
| Interactive Baseline | `scripts/interact_run_infer.sh` | `CodeActAgent` | `swe_default_interact.j2` — adds explicit instruction to ask a question before proceeding |
| UA-Single | `scripts/clarify_v2_interact_run_infer.sh` | `ClarifyAgentV2` | `swe_default.j2` — standard SWE-bench prompt with underspecified issue |
| UA-Multi | `scripts/clarify_interact_run_infer.sh` | `ClarifyAgent` | `swe_default.j2` — standard SWE-bench prompt with underspecified issue |

### Example commands

```bash
# Full baseline (fully specified issues, no interaction)
bash ./evaluation/benchmarks/swe_bench/scripts/run_infer.sh \
  llm.eval_claude_sonnet HEAD CodeActAgent 100 100 1 \
  princeton-nlp/SWE-bench_Verified test

# Hidden baseline (underspecified issues, no interaction)
bash ./evaluation/benchmarks/swe_bench/scripts/hidden_run_infer.sh \
  llm.eval_claude_sonnet HEAD CodeActAgent 100 100 1 \
  cmu-lti/interactive-swe test

# Interactive Baseline (underspecified + hardcoded interaction)
bash ./evaluation/benchmarks/swe_bench/scripts/interact_run_infer.sh \
  llm.eval_claude_sonnet HEAD CodeActAgent 100 100 1 \
  cmu-lti/interactive-swe test

# UA-Single (single-agent clarification)
bash ./evaluation/benchmarks/swe_bench/scripts/clarify_v2_interact_run_infer.sh \
  llm.eval_claude_sonnet HEAD ClarifyAgentV2 100 100 1 \
  cmu-lti/interactive-swe test

# UA-Multi (multi-agent clarification)
# NOTE: max_iter is set to 300 (not 100) because each coding agent turn triggers a
# delegate call to the Intent Agent, which counts as an additional 2 turns. 300 iterations
# is equivalent to ~100 effective coding-agent turns, matching the other settings.
bash ./evaluation/benchmarks/swe_bench/scripts/clarify_interact_run_infer.sh \
  llm.eval_claude_sonnet HEAD ClarifyAgent 100 300 1 \
  cmu-lti/interactive-swe test
```

### 📦 Dataset

The dataset [`cmu-lti/interactive-swe`](https://huggingface.co/datasets/cmu-lti/interactive-swe) is the underspecified variant of SWE-bench Verified introduced by [Vijayvargiya et al. (2026)](https://arxiv.org/abs/2502.13069) ([codebase](https://github.com/sani903/InteractiveSWEAgents)), used for all interactive settings. `princeton-nlp/SWE-bench_Verified` is used for the Full baseline.

## 🏗️ Agent Architectures

### UA-Multi: `ClarifyAgent` + `IntentAgent`
Implemented in [openhands/agenthub/clarify_agent/](openhands/agenthub/clarify_agent/) and [openhands/agenthub/intent_agent/](openhands/agenthub/intent_agent/). The **Main Agent** (`ClarifyAgent`) handles all code execution. After each main-agent turn, the **Intent Agent** analyses the conversation history and calls `clarify_decision` to signal whether clarification is required. If it signals `needs_clarification: true`, the Main Agent's next action is constrained to issue a `clarify` call to the simulated user.

### UA-Single: `ClarifyAgentV2`
Implemented in [openhands/agenthub/clarify_agent_v2/](openhands/agenthub/clarify_agent_v2/). A single agent is prompted at each turn with a reminder to assess for underspecification. If it detects missing information, it calls the `clarify` tool directly.

### Interactive Baseline: `CodeActAgent` with `swe_default_interact.j2`
A standard `CodeActAgent` run with a modified task prompt ([evaluation/benchmarks/swe_bench/prompts/swe_default_interact.j2](evaluation/benchmarks/swe_bench/prompts/swe_default_interact.j2)) that explicitly instructs the agent to ask questions before proceeding.

### Full / Hidden: `CodeActAgent` with `swe_default.j2`
Standard `CodeActAgent` receiving the fully specified (Full) or underspecified (Hidden) issue using the unmodified SWE-bench task prompt ([evaluation/benchmarks/swe_bench/prompts/swe_default.j2](evaluation/benchmarks/swe_bench/prompts/swe_default.j2)).

## ✅ Evaluating Results

After inference, evaluate generated patches with the official SWE-bench harness:

```bash
./evaluation/benchmarks/swe_bench/scripts/eval_infer.sh \
  <path/to/output.jsonl> "" princeton-nlp/SWE-bench_Verified test
```

See [evaluation/benchmarks/swe_bench/README.md](evaluation/benchmarks/swe_bench/README.md) for full evaluation documentation.

## ⚠️ Known Evaluation Issues

**SWE-bench patch revert bug (sphinx / astropy instances)**: A bug in the SWE-bench evaluation harness package (`swebench`) causes applied patches to be silently reverted to the base commit state for certain `sphinx` and `astropy` instances, causing all those runs to fail regardless of patch quality. This results in roughly 10% lower reported resolve rates compared to the official OpenHands numbers if left unpatched. Apply the fix from [Kipok/SWE-bench#3](https://github.com/Kipok/SWE-bench/pull/3) (tracked in [SWE-bench/SWE-bench#228](https://github.com/SWE-bench/SWE-bench/issues/228)) to resolve this. The paper results were produced with this fix applied.

## 🔬 Analysis and Paper Reproduction

The `analysis/` directory contains scripts to reproduce all tables and figures from the paper. See [analysis/reproduction.md](analysis/reproduction.md) for the full reproduction guide.

`analysis/original_paper_results/` contains the exact tables and figures as they appear in the submitted paper, for reference when checking your reproduction.

The analysis scripts expect evaluation outputs and trajectories collected across all five batches under `analysis/final_evaluation_outputs/` and `analysis/final_trajectories/` respectively (see [analysis/reproduction.md](analysis/reproduction.md) for the full directory layout).

### Running the analysis

The recommended end-to-end command, run from the `analysis/` directory:

```bash
# Step 1 + 2 combined: build QA artifacts from trajectories, then produce paper outputs.
# Requires ANTHROPIC_API_KEY for token counting.
export ANTHROPIC_API_KEY=...
python3 scripts/prepare_paper_rebuild_inputs.py \
  --rebuild-after \
  --rebuild-out-dir paper_reproduction_outputs/from_raw
```

`prepare_paper_rebuild_inputs.py` handles both steps: it first builds the intermediate QA pair artifacts under `qa_pairs/` from the raw trajectories, then calls `rebuild_paper_results_from_raw.py` to produce the final tables and figures. If the QA artifacts already exist, you can skip straight to step 2:

```bash
# Step 2 only: reproduce tables and figures from pre-built QA artifacts.
# Does not require ANTHROPIC_API_KEY.
python3 scripts/rebuild_paper_results_from_raw.py \
  --out-dir paper_reproduction_outputs/from_raw
```

Outputs are written to `analysis/paper_reproduction_outputs/from_raw/` by default (configurable via `--out-dir` / `--rebuild-out-dir`).

Key scripts:

| Script | Purpose |
|---|---|
| `analysis/scripts/prepare_paper_rebuild_inputs.py` | Builds QA pair artifacts from raw trajectories; optionally calls `rebuild_paper_results_from_raw.py` via `--rebuild-after` |
| `analysis/scripts/rebuild_paper_results_from_raw.py` | Produces paper tables and figures from QA artifacts + eval outputs |
| `analysis/scripts/compute_ask_significance.py` | Computes significance statistics for ask-rate comparisons (optional) |
| `analysis/qa_scripts/qa_cli.py` | Low-level QA pipeline called internally by `prepare_paper_rebuild_inputs.py` |

## 📖 Citation

If you use this work, please consider citing our paper:

```bibtex
@misc{ask_or_assume_2026,
  title={Ask or Assume? Uncertainty-Aware Clarification-Seeking in Coding Agents},
  author={[TODO: add authors]},
  year={2026},
  note={ACL submission}
}
```

---

<!-- OpenHands project README below -->

<a name="readme-top"></a>

<div align="center">
  <img src="./docs/static/img/logo.png" alt="Logo" width="200">
  <h1 align="center">OpenHands: Code Less, Make More</h1>
</div>


<div align="center">
  <a href="https://github.com/All-Hands-AI/OpenHands/graphs/contributors"><img src="https://img.shields.io/github/contributors/All-Hands-AI/OpenHands?style=for-the-badge&color=blue" alt="Contributors"></a>
  <a href="https://github.com/All-Hands-AI/OpenHands/stargazers"><img src="https://img.shields.io/github/stars/All-Hands-AI/OpenHands?style=for-the-badge&color=blue" alt="Stargazers"></a>
  <a href="https://github.com/All-Hands-AI/OpenHands/blob/main/LICENSE"><img src="https://img.shields.io/github/license/All-Hands-AI/OpenHands?style=for-the-badge&color=blue" alt="MIT License"></a>
  <br/>
  <a href="https://all-hands.dev/joinslack"><img src="https://img.shields.io/badge/Slack-Join%20Us-red?logo=slack&logoColor=white&style=for-the-badge" alt="Join our Slack community"></a>
  <a href="https://github.com/All-Hands-AI/OpenHands/blob/main/CREDITS.md"><img src="https://img.shields.io/badge/Project-Credits-blue?style=for-the-badge&color=FFE165&logo=github&logoColor=white" alt="Credits"></a>
  <br/>
  <a href="https://docs.all-hands.dev/usage/getting-started"><img src="https://img.shields.io/badge/Documentation-000?logo=googledocs&logoColor=FFE165&style=for-the-badge" alt="Check out the documentation"></a>
  <a href="https://arxiv.org/abs/2407.16741"><img src="https://img.shields.io/badge/Paper%20on%20Arxiv-000?logoColor=FFE165&logo=arxiv&style=for-the-badge" alt="Paper on Arxiv"></a>
  <a href="https://docs.google.com/spreadsheets/d/1wOUdFCMyY6Nt0AIqF705KN4JKOWgeI4wUGUP60krXXs/edit?gid=0#gid=0"><img src="https://img.shields.io/badge/Benchmark%20score-000?logoColor=FFE165&logo=huggingface&style=for-the-badge" alt="Evaluation Benchmark Score"></a>

  <!-- Keep these links. Translations will automatically update with the README. -->
  <a href="https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=de">Deutsch</a> |
  <a href="https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=es">Español</a> |
  <a href="https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=fr">français</a> |
  <a href="https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=ja">日本語</a> |
  <a href="https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=ko">한국어</a> |
  <a href="https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=pt">Português</a> |
  <a href="https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=ru">Русский</a> |
  <a href="https://www.readme-i18n.com/All-Hands-AI/OpenHands?lang=zh">中文</a>

  <hr>
</div>

Welcome to OpenHands (formerly OpenDevin), a platform for software development agents powered by AI.

OpenHands agents can do anything a human developer can: modify code, run commands, browse the web,
call APIs, and yes—even copy code snippets from StackOverflow.

Learn more at [docs.all-hands.dev](https://docs.all-hands.dev), or [sign up for OpenHands Cloud](https://app.all-hands.dev) to get started.

> [!IMPORTANT]
> Using OpenHands for work? We'd love to chat! Fill out
> [this short form](https://docs.google.com/forms/d/e/1FAIpQLSet3VbGaz8z32gW9Wm-Grl4jpt5WgMXPgJ4EDPVmCETCBpJtQ/viewform)
> to join our Design Partner program, where you'll get early access to commercial features and the opportunity to provide input on our product roadmap.

## ☁️ OpenHands Cloud
The easiest way to get started with OpenHands is on [OpenHands Cloud](https://app.all-hands.dev),
which comes with $20 in free credits for new users.

<a name="running-openhands-locally"></a>
## 💻 Running OpenHands Locally

### Option 1: CLI Launcher (Recommended)

The easiest way to run OpenHands locally is using the CLI launcher with [uv](https://docs.astral.sh/uv/). This provides better isolation from your current project's virtual environment and is required for OpenHands' default MCP servers.

**Install uv** (if you haven't already):

See the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/) for the latest installation instructions for your platform.

**Launch OpenHands**:
```bash
# Launch the GUI server
uvx --python 3.12 --from openhands-ai openhands serve

# Or launch the CLI
uvx --python 3.12 --from openhands-ai openhands
```

You'll find OpenHands running at [http://localhost:3000](http://localhost:3000) (for GUI mode)!

### Option 2: Docker

<details>
<summary>Click to expand Docker command</summary>

You can also run OpenHands directly with Docker:

```bash
docker pull docker.all-hands.dev/all-hands-ai/runtime:0.58-nikolaik

docker run -it --rm --pull=always \
    -e SANDBOX_RUNTIME_CONTAINER_IMAGE=docker.all-hands.dev/all-hands-ai/runtime:0.58-nikolaik \
    -e LOG_ALL_EVENTS=true \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v ~/.openhands:/.openhands \
    -p 3000:3000 \
    --add-host host.docker.internal:host-gateway \
    --name openhands-app \
    docker.all-hands.dev/all-hands-ai/openhands:0.58
```

</details>

> **Note**: If you used OpenHands before version 0.44, you may want to run `mv ~/.openhands-state ~/.openhands` to migrate your conversation history to the new location.

> [!WARNING]
> On a public network? See our [Hardened Docker Installation Guide](https://docs.all-hands.dev/usage/runtimes/docker#hardened-docker-installation)
> to secure your deployment by restricting network binding and implementing additional security measures.

### Getting Started

When you open the application, you'll be asked to choose an LLM provider and add an API key.
[Anthropic's Claude Sonnet 4.5](https://www.anthropic.com/api) (`anthropic/claude-sonnet-4-5-20250929`)
works best, but you have [many options](https://docs.all-hands.dev/usage/llms).

See the [Running OpenHands](https://docs.all-hands.dev/usage/installation) guide for
system requirements and more information.

## 💡 Other ways to run OpenHands

> [!WARNING]
> OpenHands is meant to be run by a single user on their local workstation.
> It is not appropriate for multi-tenant deployments where multiple users share the same instance. There is no built-in authentication, isolation, or scalability.
>
> If you're interested in running OpenHands in a multi-tenant environment, check out the source-available, commercially-licensed
> [OpenHands Cloud Helm Chart](https://github.com/all-Hands-AI/OpenHands-cloud)

You can [connect OpenHands to your local filesystem](https://docs.all-hands.dev/usage/runtimes/docker#connecting-to-your-filesystem),
interact with it via a [friendly CLI](https://docs.all-hands.dev/usage/how-to/cli-mode),
run OpenHands in a scriptable [headless mode](https://docs.all-hands.dev/usage/how-to/headless-mode),
or run it on tagged issues with [a github action](https://docs.all-hands.dev/usage/how-to/github-action).

Visit [Running OpenHands](https://docs.all-hands.dev/usage/installation) for more information and setup instructions.

If you want to modify the OpenHands source code, check out [Development.md](https://github.com/All-Hands-AI/OpenHands/blob/main/Development.md).

Having issues? The [Troubleshooting Guide](https://docs.all-hands.dev/usage/troubleshooting) can help.

## 📖 Documentation

To learn more about the project, and for tips on using OpenHands,
check out our [documentation](https://docs.all-hands.dev/usage/getting-started).

There you'll find resources on how to use different LLM providers,
troubleshooting resources, and advanced configuration options.

## 🤝 How to Join the Community

OpenHands is a community-driven project, and we welcome contributions from everyone. We do most of our communication
through Slack, so this is the best place to start, but we also are happy to have you contact us on Github:

- [Join our Slack workspace](https://all-hands.dev/joinslack) - Here we talk about research, architecture, and future development.
- [Read or post Github Issues](https://github.com/All-Hands-AI/OpenHands/issues) - Check out the issues we're working on, or add your own ideas.

See more about the community in [COMMUNITY.md](./COMMUNITY.md) or find details on contributing in [CONTRIBUTING.md](./CONTRIBUTING.md).

## 📈 Progress

See the monthly OpenHands roadmap [here](https://github.com/orgs/All-Hands-AI/projects/1) (updated at the maintainer's meeting at the end of each month).

<p align="center">
  <a href="https://star-history.com/#All-Hands-AI/OpenHands&Date">
    <img src="https://api.star-history.com/svg?repos=All-Hands-AI/OpenHands&type=Date" width="500" alt="Star History Chart">
  </a>
</p>

## 📜 License

Distributed under the MIT License, with the exception of the `enterprise/` folder. See [`LICENSE`](./LICENSE) for more information.

## 🙏 Acknowledgements

OpenHands is built by a large number of contributors, and every contribution is greatly appreciated! We also build upon other open source projects, and we are deeply thankful for their work.

For a list of open source projects and licenses used in OpenHands, please see our [CREDITS.md](./CREDITS.md) file.

## 📚 Cite

```
@inproceedings{
  wang2025openhands,
  title={OpenHands: An Open Platform for {AI} Software Developers as Generalist Agents},
  author={Xingyao Wang and Boxuan Li and Yufan Song and Frank F. Xu and Xiangru Tang and Mingchen Zhuge and Jiayi Pan and Yueqi Song and Bowen Li and Jaskirat Singh and Hoang H. Tran and Fuqiang Li and Ren Ma and Mingzhang Zheng and Bill Qian and Yanjun Shao and Niklas Muennighoff and Yizhe Zhang and Binyuan Hui and Junyang Lin and Robert Brennan and Hao Peng and Heng Ji and Graham Neubig},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=OJd3ayDDoF}
}
```

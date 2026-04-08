---
title: Natural Language Reward Definition Environment
emoji: robot
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
license: bsd-3-clause
---

# Natural Language Reward Definition Environment

An OpenEnv-compliant reinforcement learning environment where the reward function is defined in plain English and interpreted by an LLM judge instead of hardcoded task logic.

## What this repo includes

- A complete OpenEnv environment with typed `Action`, `Observation`, `Reward`, and `State` models.
- A `RewardInterpreter` that scores agent outputs in `[0, 1]` using the OpenAI client with structured JSON judging.
- Three real-world tasks of increasing difficulty:
  - `customer_support_response`
  - `email_triage`
  - `code_review`
- Deterministic fallback graders for every task so the environment still runs when the judge model is unavailable.
- A root-level `inference.py` that follows the required `[START]`, `[STEP]`, `[END]` stdout format.
- A working Dockerfile, Hugging Face Spaces-friendly FastAPI app, tests, and a validator script.

## Architecture

The environment has two reward layers:

1. `nl_reward_env/reward_interpreter.py`
   Reads the natural-language reward instruction, agent output, and current state, then asks an LLM judge for a structured JSON assessment.
2. `nl_reward_env/graders/deterministic.py`
   Provides task-specific heuristics used as a fallback and as an anchoring signal for dense reward shaping.

The final reward is a shaped combination of:

- deterministic fallback score
- optional LLM judge score
- improvement bonus over the previous best attempt
- penalties for repetition, empty outputs, and low-quality behavior

This produces dense rewards on every step while still penalizing unsafe or low-signal behavior.

## Task design

### 1. Customer support response

- Goal: write an empathetic, policy-accurate support reply for a delayed order complaint.
- Reward emphasizes empathy, honest policy handling, and actionable next steps.
- Deterministic penalties trigger on false promises like guaranteed delivery or instant refunds.

### 2. Email triage

- Goal: triage a likely business-email-compromise message.
- Output format: JSON with `priority`, `category`, `assignee`, `justification`, `response_draft`.
- Reward emphasizes correct routing, safe operations handling, and machine-readable output.

### 3. Code review

- Goal: review a diff and identify the privilege-escalation bug.
- Reward emphasizes severity, concrete risk explanation, line references, and a fix.
- The fallback grader penalizes style-only comments and missing the authorization issue.

## Project layout

```text
.
├── Dockerfile
├── README.md
├── inference.py
├── openenv.yaml
├── pyproject.toml
├── nl_reward_env/
│   ├── baseline.py
│   ├── client.py
│   ├── config.py
│   ├── environment.py
│   ├── models.py
│   ├── reward_interpreter.py
│   ├── graders/
│   └── tasks/
├── server/
│   ├── app.py
│   └── natural_language_reward_environment.py
└── tests/
```

## Setup

### 1. Install

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

### 2. Required environment variables

Before running the LLM-based judge or baseline, define:

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your-token"
```

The code also accepts `API_KEY` as a fallback for local testing.

### 3. Run the FastAPI / OpenEnv server

```bash
python -m server.app
```

Or:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

## Using the environment

### In-process

```python
from nl_reward_env.environment import NaturalLanguageRewardEnvironment
from nl_reward_env.models import NaturalLanguageRewardAction

env = NaturalLanguageRewardEnvironment()
obs = env.reset(task_id="customer_support_response")
result = env.step(NaturalLanguageRewardAction(response="..."))
print(result.reward)
print(env.state.best_reward)
```

### OpenEnv client

```python
import asyncio
from nl_reward_env.client import NaturalLanguageRewardEnv
from nl_reward_env.models import NaturalLanguageRewardAction

async def main():
    async with NaturalLanguageRewardEnv(base_url="http://127.0.0.1:8000") as env:
        result = await env.reset(task_id="email_triage")
        result = await env.step(
            NaturalLanguageRewardAction(response='{"priority":"high","category":"security","assignee":"security@acme.co","justification":"...","response_draft":"..."}')
        )
        print(result.reward)

asyncio.run(main())
```

## Inference

The required submission script is at the repo root:

```bash
python inference.py
```

Behavior:

- Runs all three tasks by default.
- Emits only the required `[START]`, `[STEP]`, and `[END]` log lines.
- Uses the OpenAI client when credentials are available.
- Falls back to deterministic baseline actions if no model credentials are present, so the script still completes.

To run a single task:

```bash
export NLRDE_TASK="code_review"
python inference.py
```

## Docker

Build:

```bash
docker build -t nl-reward-env .
```

Run:

```bash
docker run -p 8000:8000 \
  -e API_BASE_URL="$API_BASE_URL" \
  -e MODEL_NAME="$MODEL_NAME" \
  -e HF_TOKEN="$HF_TOKEN" \
  nl-reward-env
```

## Hugging Face Spaces compatibility

This repository is designed for Docker Spaces:

- `openenv.yaml` points to `server.app:app`
- the FastAPI server exposes `/reset`, `/step`, `/state`, `/schema`, `/health`
- the root Dockerfile launches `python -m server.app` with configurable `WORKERS`, `MAX_CONCURRENT_ENVS`, `PORT`, and `HOST`

For a Docker Space:

1. Create a new Docker Space.
2. Push this repository.
3. Add `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN` as Space secrets.
4. Wait for the build to finish, then verify `POST /reset` returns HTTP 200.

## Scaling

This environment now supports the same single-container scaling knobs described in the OpenEnv scaling guide:

- `WORKERS`
- `MAX_CONCURRENT_ENVS`
- `PORT`
- `HOST`

Recommended Hugging Face free-tier settings:

- `WORKERS=2`
- `MAX_CONCURRENT_ENVS=100`

These defaults are already set in the Dockerfile so the Space behaves sensibly on CPU Basic without extra changes. For local development, you can override them:

```bash
WORKERS=2 MAX_CONCURRENT_ENVS=100 uv run server --port 8000
```

Or with Docker:

```bash
docker run -p 8000:8000 \
  -e WORKERS=2 \
  -e MAX_CONCURRENT_ENVS=100 \
  -e PORT=8000 \
  -e HOST=0.0.0.0 \
  -e API_BASE_URL="$API_BASE_URL" \
  -e MODEL_NAME="$MODEL_NAME" \
  -e HF_TOKEN="$HF_TOKEN" \
  nl-reward-env
```

Free-tier interpretation for this project:

- Development and testing: single container with WebSocket sessions
- Moderate load on HF Spaces free tier: keep a single container and use `WORKERS=2`
- Session cap: `MAX_CONCURRENT_ENVS=100` per deployment
- If you need sustained throughput far beyond free-tier limits, scale horizontally with multiple containers and a load balancer

The server entrypoint also supports:

```bash
uv run server --port 8001 --workers 2
```

If the default port is occupied locally, the server automatically falls back to the next available port unless you explicitly request a busy one.

## Validation and tests

Run the local checks:

```bash
pytest
openenv validate
python inference.py
```

The repo also includes the pre-submission helper:

```bash
bash scripts/validate-submission.sh https://your-space-url.hf.space
```

## Example results

Local fallback-baseline run from `python inference.py`:

| Task | Score |
| --- | --- |
| `customer_support_response` | `1.000` |
| `email_triage` | `1.000` |
| `code_review` | `0.941` |

All task scores are normalized into `[0.0, 1.0]`.

## Sources used

- [OpenEnv tutorial / repository](https://github.com/meta-pytorch/OpenEnv/tree/main/tutorial)
- [OpenEnv README](https://github.com/meta-pytorch/OpenEnv)
- [OpenAI structured outputs guide](https://platform.openai.com/docs/guides/structured-outputs)
- [Hackathon dashboard and checklist](https://www.scaler.com/school-of-technology/meta-pytorch-hackathon/dashboard?utm_source=midfunnel&utm_medium=whatsapp&utm_campaign=23pmmmr)

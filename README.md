# 🤖 TaskAgent-Flow

**A lightweight, zero-bloat Python micro-framework for running multi-step AI agent workflows on your own local LLM — no cloud API keys, no LangChain, no CrewAI.**

Define a task as a simple YAML or JSON file. TaskAgent-Flow runs each step through a local model via [Ollama](https://ollama.com), threads context from one step into the next, and **automatically retries and self-corrects** failed steps before giving up.

```yaml
task_name: "Research and Summarize"
steps:
  - id: 1
    action: "Generate 3 potential titles for a blog post about Edge AI."
    expected_output: "A list of 3 titles."
  - id: 2
    action: "Take the best title from step 1 and write a 2-sentence summary."
    expected_output: "A short 2-sentence summary."
```

```bash
python run.py example_task.yaml
```

That's it. No API keys, no telemetry, nothing leaves your machine.

---

## ✨ Why TaskAgent-Flow?

| | |
|---|---|
| 🖥️ **100% Local** | Runs entirely against your local [Ollama](https://ollama.com) server. Your prompts and data never leave your machine. |
| 🪶 **Zero Bloat** | Three dependencies: `pyyaml`, `requests`, `pydantic`. No LangChain, no CrewAI, no 40-package dependency tree. |
| 🔁 **Self-Healing Steps** | Failed or empty outputs automatically trigger a fallback loop — the model is told *why* it failed and asked to retry, up to 3 times, before raising a clear error. |
| 🧠 **Context-Aware** | Every step automatically receives the output of every previous step, so multi-step reasoning chains actually work. |
| 🎯 **Smart Model Fallback** | Defaults to `llama3`, automatically falls back to `mistral` if `llama3` isn't pulled — no manual config needed. |
| 📝 **YAML or JSON** | Define tasks in whichever format you prefer. |
| ✅ **Fully Typed** | Pydantic models + type hints throughout. Editor-friendly, easy to extend. |

---

## 📦 Installation

**Prerequisites:**
- Python 3.9+
- [Ollama](https://ollama.com) installed and running locally

```bash
# 1. Clone the repo
git clone https://github.com/your-username/TaskAgent-Flow.git
cd TaskAgent-Flow

# 2. Install dependencies
pip install -r requirements.txt

# 3. Pull a local model (if you haven't already)
ollama pull llama3
# or: ollama pull mistral

# 4. Make sure Ollama is running
ollama serve
```

---

## 🚀 Quick Start

Run the included example task:

```bash
python run.py example_task.yaml
```

You should see each step execute with live progress logs, followed by the final output of every step.

### Define your own task

Create a `.yaml` (or `.json`) file describing your workflow:

```yaml
task_name: "Plan a Product Launch"
model: "mistral"  # optional — overrides the default model for this task

steps:
  - id: 1
    action: "List 5 key risks for launching a new SaaS product."
    expected_output: "A numbered list of 5 risks."

  - id: 2
    action: "For the most critical risk from step 1, propose a mitigation plan."
    expected_output: "A short, actionable mitigation plan."
```

Then run it:

```bash
python run.py my_task.yaml --model llama3 --retries 3
```

### CLI options

```text
usage: run.py [-h] [--model MODEL] [--host HOST] [--retries RETRIES] [--quiet] task_file

positional arguments:
  task_file          Path to a .yaml/.yml/.json task definition file.

options:
  -h, --help         show this help message and exit
  --model MODEL      Override the Ollama model (default: llama3, falls back to mistral).
  --host HOST        Ollama API host URL. (default: http://localhost:11434)
  --retries RETRIES  Max retries per step on validation failure. (default: 3)
  --quiet            Suppress progress logging.
```

---

## 🛠 How It Works

```
┌─────────────────┐      ┌──────────────────┐      ┌───────────────────┐
│  task.yaml/json  │ ───▶ │   TaskEngine     │ ───▶ │   OllamaClient     │
│ (steps + config) │      │ (orchestration)  │      │ (local LLM calls)  │
└─────────────────┘      └──────────────────┘      └───────────────────┘
                                   │
                                   ▼
                       ┌───────────────────────┐
                       │  Validation Heuristic  │
                       │  (empty / refusal /    │
                       │   too-short checks)    │
                       └───────────────────────┘
                                   │
                     ┌─────────────┴──────────────┐
                     ▼                            ▼
              ✅ Step passes               ❌ Step fails
              → move to next step          → retry with failure
                                              reason injected into
                                              the next prompt
                                            (up to `--retries` times,
                                             then raises a clear error)
```

1. **Load** — `TaskEngine` reads and validates your task file against a Pydantic schema.
2. **Resolve model** — checks Ollama for available models, preferring your explicit choice, then `llama3`, then `mistral`.
3. **Execute steps in order** — each step's prompt includes the full output history of every prior step.
4. **Validate** — a fast heuristic checks for empty, too-short, or refusal-style outputs.
5. **Self-heal** — on failure, the model is re-prompted with the specific reason it failed (*"Your previous attempt failed because: the output was empty. Please try again."*), up to `max_retries` times.
6. **Fail loudly** — if a step never passes validation, a `TaskExecutionError` is raised with full context, instead of silently continuing with bad data.

---

## 📁 Project Layout

```
TaskAgent-Flow/
├── agent_flow/
│   ├── __init__.py        # package exports
│   ├── engine.py           # TaskEngine — orchestration + retry/fallback loop
│   ├── llm_client.py       # OllamaClient — talks to the local Ollama REST API
│   ├── models.py           # Pydantic schemas for tasks, steps, and results
│   └── validators.py       # lightweight output validation heuristic
├── tests/
│   ├── test_engine.py
│   ├── test_llm_client.py
│   └── test_validators.py
├── example_task.yaml        # ready-to-run sample task
├── run.py                    # CLI entrypoint
├── requirements.txt
└── README.md
```

---

## 🧩 Using It as a Library

```python
from agent_flow.engine import TaskEngine

engine = TaskEngine(task_path="my_task.yaml", model="llama3")
results = engine.run()

for result in results:
    print(f"Step {result.step_id} ({result.attempts} attempt(s)): {result.output}")
```

---

## 🧪 Running Tests

The test suite mocks all network calls, so it runs instantly with no Ollama server required:

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

---

## 🗺 Roadmap Ideas

- [ ] Parallel/branching step execution (DAG-style task graphs)
- [ ] Pluggable validators (regex, JSON-schema, embedding similarity)
- [ ] Streaming token output in the CLI
- [ ] Support for additional local runtimes (LM Studio, llama.cpp server)

Contributions and issues welcome — this project intentionally stays small, so PRs that add a new *optional* capability without growing the core dependency list are especially appreciated.

---

## 📄 License

MIT — see [LICENSE](LICENSE).

---

<p align="center">If this saved you from pulling in a 200MB agent framework for a 3-step script, consider giving it a ⭐</p>

#!/usr/bin/env python3
"""CLI entrypoint for TaskAgent-Flow.

Usage:
    python run.py path/to/task.yaml
    python run.py path/to/task.json --model mistral
"""

from __future__ import annotations

import argparse
import logging
import sys

from agent_flow.engine import DEFAULT_MAX_RETRIES, TaskEngine, TaskExecutionError
from agent_flow.llm_client import DEFAULT_HOST


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Execute a multi-step TaskAgent-Flow task definition using a local Ollama LLM.",
    )
    parser.add_argument("task_file", help="Path to a .yaml/.yml/.json task definition file.")
    parser.add_argument(
        "--model",
        default=None,
        help="Override the Ollama model (default: llama3, falls back to mistral).",
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="Ollama API host URL.")
    parser.add_argument(
        "--retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help="Max retries per step on validation failure.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress logging.")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    try:
        engine = TaskEngine(
            task_path=args.task_file,
            model=args.model,
            host=args.host,
            max_retries=args.retries,
            verbose=not args.quiet,
        )
        results = engine.run()
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except (ValueError, RuntimeError, ConnectionError, TimeoutError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except TaskExecutionError as exc:
        print(f"Task failed: {exc}", file=sys.stderr)
        return 2

    print(f"\nTask '{engine.task.task_name}' completed successfully.\n")
    for result in results:
        print(f"--- Step {result.step_id} (attempts: {result.attempts}) ---")
        print(result.output)
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())

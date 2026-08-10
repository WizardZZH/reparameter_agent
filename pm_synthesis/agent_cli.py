from __future__ import annotations

import argparse
import asyncio
import json
import traceback
from pathlib import Path

from .agent_runtime import run_agent_role
from .io_utils import atomic_write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one PM synthesis sub-agent role")
    parser.add_argument("--role", choices=("domain", "coding", "verifier"), required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--round", type=int, required=True, dest="round_number")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, default=None)
    return parser


async def _async_main(args: argparse.Namespace) -> int:
    role_dir = args.workspace / "rounds" / f"round-{args.round_number:03d}" / "agents" / args.role
    try:
        result = await run_agent_role(
            role=args.role,
            workspace=args.workspace,
            round_number=args.round_number,
            config_path=args.config,
            prompts_path=args.prompts,
        )
    except BaseException as exc:
        role_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            role_dir / "status.json",
            {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            },
        )
        print(json.dumps({"status": "error", "role": args.role, "error": str(exc)}))
        return 1
    print(json.dumps({"status": "success", "role": args.role, "result": str(result)}))
    return 0


def main() -> int:
    return asyncio.run(_async_main(build_parser().parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())

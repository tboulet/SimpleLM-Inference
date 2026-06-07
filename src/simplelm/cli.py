"""`simplelm serve …` CLI."""
from __future__ import annotations

import argparse
import sys

from simplelm.backends.hf import HuggingFaceBackend
from simplelm.server import serve


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="simplelm")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="Start an OpenAI-compatible server.")
    s.add_argument("--model-path", required=True,
                   help="HF snapshot path or hub id.")
    s.add_argument("--model-name", default=None,
                   help="Served model id; defaults to basename of --model-path.")
    s.add_argument("--port", type=int, default=9876)
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--torch-dtype",
                   choices=("bfloat16", "float16", "float32"), default="bfloat16")
    s.add_argument("--device-map", default="auto",
                   help="Passed to AutoModelForCausalLM.from_pretrained.")
    s.add_argument("--tool-parser", default="noop",
                   help="Tool-call parser: noop | gemma4 | qwen3_coder | kimi_k2 | glm45 | minimax-m2 | deepseek-v4")
    s.add_argument("--no-trust-remote-code", action="store_true",
                   help="Disable trust_remote_code (default: enabled).")
    s.add_argument("--no-processor", action="store_true",
                   help="Skip AutoProcessor (text-only models, slight speedup).")
    s.add_argument("--log-level", default="info")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv if argv is not None else sys.argv[1:])
    if args.cmd != "serve":
        return 2
    backend = HuggingFaceBackend(
        args.model_path,
        model_name=args.model_name,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
        trust_remote_code=not args.no_trust_remote_code,
        prefer_processor=not args.no_processor,
    )
    serve(
        backend,
        host=args.host,
        port=args.port,
        tool_parser=args.tool_parser,
        log_level=args.log_level,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

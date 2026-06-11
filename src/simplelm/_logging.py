"""Centralised logging for SimpleLM.

Usage:

    from simplelm._logging import logger, warning_once

    logger.info("loaded model %s", name)
    warning_once("unknown content type %r — dropping", t)   # logs ONCE per
                                                            # distinct message

`warning_once` exists for messages that sit on a hot path (per request,
per message, per token) where a plain `logger.warning` would flood the
log. The dedup set lives for the process lifetime, i.e. "once per
experiment / per served process" — which is what we want.

Call `setup_logging(level)` once at startup (the CLI / `serve()` does).
Nothing in SimpleLM should `print()` or silently `pass` on a dropped /
fallback path: emit a log line at the right level instead.

Levels we use:
    debug   — verbose internals (per-request prompt sizes, head probing)
    info    — lifecycle (model loaded, arch detected, server up)
    warning — a fallback was taken / an input was coerced / something
              dropped. The caller still gets a result, but not the ideal one.
    error   — a request failed.
"""
from __future__ import annotations

import logging

logger = logging.getLogger("simplelm")

# Messages already emitted via warning_once, for process-lifetime dedup.
_warned_once: set[str] = set()


def warning_once(msg: str, *args: object) -> None:
    """`logger.warning`, but emit each distinct (formatted) message only once.

    Args follow %-style formatting, like the stdlib logger:
        warning_once("bad dtype %r, using bfloat16", dtype)
    """
    formatted = msg % args if args else msg
    if formatted not in _warned_once:
        _warned_once.add(formatted)
        logger.warning(formatted)


def setup_logging(level: str = "info") -> None:
    """Configure the root handler + the `simplelm` logger level. Idempotent."""
    lvl = getattr(logging, str(level).upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger.setLevel(lvl)


def reset_warn_once() -> None:
    """Clear the dedup set (e.g. between tests). Rarely needed in prod."""
    _warned_once.clear()

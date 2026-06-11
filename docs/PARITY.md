# SimpleLM ↔ SGLang/vLLM feature parity

What an agentic client (AlanCode driving GameAgents) needs from an
OpenAI-compatible inference engine, and where SimpleLM stands. Source of
truth for the engine-maintenance backlog.

Verified against MI250X on Adastra (CINES) — see the Phase-0 run below.

## Matrix

| Capability | SimpleLM today | Priority |
|---|---|---|
| `/v1/chat/completions` (messages, max_tokens, temperature, top_p) | ✅ | — |
| `max_model_len` / context window in `/v1/models` | ✅ derived from the model config (`text_config`-aware), emitted in `/v1/models` | — (AlanCode reads it to size compaction) |
| Turn-boundary stop + OpenAI `stop` | ✅ inherits the model's `generation_config` eos (e.g. `<\|im_end\|>`); `stop` → transformers `stop_strings` | — |
| Tool calls (function calling) | ✅ via a **server-global** parser (`--tool-parser`), default **`noop`** (extracts nothing) | **P1 (config)** — must pass `--tool-parser universal`/per-family; 11 family parsers exist |
| Tool calls inside `<think>` | ✅ scans full decoded text (better than SGLang's content-only scan) | — (strength) |
| `reasoning_content` separation | ❌ `<think>…</think>` stays in `content` | P2 |
| `repetition_penalty` / `top_k` / `top_p` | ✅ inherited from the model's `generation_config` | — |
| `seed` / `n` / `logprobs` | ❌ none | P3 (minor) |
| `tool_choice` (force a tool) | ❌ accepted, ignored | P3 |
| `response_format` / JSON mode | ❌ accepted, ignored | P3 |
| Streaming (SSE) | ⚠️ pseudo — whole message as one chunk | P3 (works for LiteLLM) |
| Vision (`image_url`: data / file / http) | ✅ PIL load + processor, VLM auto-head | — |
| Context-overflow guard / truncation | ❌ a too-long prompt just errors | P2 (ties to `max_model_len`) |
| Param validation (bad temp/top_p/max_tokens) | ✅ clamp + warn-once (`hf.py`) | done |
| Throughput (paged-KV / batching / flash-attn) | ❌ single-thread `.generate()` | not correctness; ~10× slower |

## Backlog

1. `reasoning_content` separation — split `<think>…</think>` into its own
   field for reasoning models.
2. Context-overflow guard — reject or truncate prompts longer than
   `max_model_len` rather than erroring.
3. `tool_choice` (force a tool) / `response_format` (JSON mode) — accepted
   but ignored today.
4. Real token streaming (currently pseudo: the whole message as one chunk).
5. Throughput — paged-KV / batching / flash-attn (the ~10× gap vs SGLang).

Out of SimpleLM's scope: the Adastra mem-efficient-SDP caveat is an
*environment* concern (handled, if needed, by a `venv_ada` sitecustomize),
not the engine — SimpleLM is cluster-agnostic and on NVIDIA that backend is
the fast path.

## Phase-0 evidence (Adastra, 2026-06-11)

GameAgents `minigrid_empty_5x5` via AlanCode → SimpleLM (`venv_ada`),
`--tool-parser universal`.

- **Qwen2.5-3B-Instruct** (JID 5055692): integration ran 10 turns / 131k
  tokens, no crash — **the plumbing works**. But the 3B model emits
  `<tool_call>\nassistant\n…` and `<tool_call>\nuser\n<tool_response>{…}` —
  it **hallucinates the whole multi-turn transcript** instead of a clean
  `<tool_call>{json}</tool_call>`, so zero structured tool calls are
  parsed, `solution.py` is never written, both episodes hit the placeholder
  `NotImplementedError`. Two causes: (a) model too weak; (b) **no
  turn-boundary stop** lets it ramble (backlog #1).
- Qwen2.5-32B-Instruct: capability re-run (confirms whether a stronger
  model closes the loop) — result pending.

Takeaway: SimpleLM's OpenAI surface + tool parsing are functionally wired;
the gating gaps are **stop-at-turn-boundary** and **context-length
exposure**, plus model capability.

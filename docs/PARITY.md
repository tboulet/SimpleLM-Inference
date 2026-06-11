# SimpleLM ↔ SGLang/vLLM feature parity

What an agentic client (AlanCode driving GameAgents) needs from an
OpenAI-compatible inference engine, and where SimpleLM stands. Source of
truth for the engine-maintenance backlog.

Verified against MI250X on Adastra (CINES) — see the Phase-0 run below.

## Matrix

| Capability | SimpleLM today | Priority |
|---|---|---|
| `/v1/chat/completions` (messages, max_tokens, temperature, top_p) | ✅ | — |
| **`max_model_len` / context window in `/v1/models`** | ❌ `ModelInfo` has no context length | **P1** — AlanCode reads it to size compaction; absent → compaction can't trigger correctly |
| **Stop at chat turn boundary (`<|im_end|>` / `</tool_call>` / custom `stop`)** | ❌ only the tokenizer `eos` | **P1** — weak models run past their turn and hallucinate fake `assistant`/`user`/`<tool_response>` turns (see Phase 0) |
| Tool calls (function calling) | ✅ via a **server-global** parser (`--tool-parser`), default **`noop`** (extracts nothing) | **P1 (config)** — must pass `--tool-parser universal`/per-family; 11 family parsers exist |
| Tool calls inside `<think>` | ✅ scans full decoded text (better than SGLang's content-only scan) | — (strength) |
| `reasoning_content` separation | ❌ `<think>…</think>` stays in `content` | P2 |
| `stop` / `repetition_penalty` / `seed` / `n` / `logprobs` | ❌ none | P2 (`stop`, `repetition_penalty` matter for agent-loop robustness) |
| `tool_choice` (force a tool) | ❌ accepted, ignored | P3 |
| `response_format` / JSON mode | ❌ accepted, ignored | P3 |
| Streaming (SSE) | ⚠️ pseudo — whole message as one chunk | P3 (works for LiteLLM) |
| Vision (`image_url`: data / file / http) | ✅ PIL load + processor, VLM auto-head | — |
| Context-overflow guard / truncation | ❌ a too-long prompt just errors | P2 (ties to `max_model_len`) |
| Param validation (bad temp/top_p/max_tokens) | ✅ clamp + warn-once (`hf.py`) | done |
| Throughput (paged-KV / batching / flash-attn) | ❌ single-thread `.generate()` | not correctness; ~10× slower |
| `enable_mem_efficient_sdp(False)` on Adastra MI250 | ❌ not set (colleague flagged misbehavior) | P2 (one-liner) |

## Prioritised backlog

1. **Stop sequences / turn-boundary stopping** — honor `generation_config`
   eos (e.g. Qwen's `<|im_end|>`), accept OpenAI `stop`, stop at
   `</tool_call>`. Highest leverage: fixes the rambling failure for *every*
   model.
2. **`max_model_len` in `/v1/models`** — read `max_position_embeddings`
   (or `text_config.*`) from the model config, expose it. Unblocks AlanCode
   compaction.
3. **`repetition_penalty` + `stop` passthrough** to `.generate()`.
4. **`mem_efficient_sdp(False)`** on ROCm/Adastra.
5. `reasoning_content` split (reasoning models), then `tool_choice` /
   `response_format` / real streaming (P3).

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

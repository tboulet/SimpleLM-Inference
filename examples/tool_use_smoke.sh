#!/bin/bash
# Clean tool-use smoke: send an OpenAI Chat Completion with a tool
# definition, verify the model returns a structured `tool_calls` array.
#
# Designed to isolate the SimpleLM tool-call parser from any specific
# agent framework — if this passes for a given (model, parser) pair,
# the parser is correctly extracting that model's tool-call markup.

set -uo pipefail

PORT=${PORT:-9876}
LOG_DIR="${OUTPUT_DIR:-${SCRATCH}/experiments_logs}"
mkdir -p "${LOG_DIR}"

simplelm serve \
    --model-path "${MODEL_PATH}" \
    --model-name "${MODEL_NAME}" \
    --port "${PORT}" \
    --tool-parser "${TOOL_PARSER:-gemma4}" \
    --torch-dtype "${TORCH_DTYPE:-bfloat16}" \
    > "${LOG_DIR}/simplelm.${SLURM_JOB_ID}.log" 2>&1 &
SERVER_PID=$!
BASE_URL="http://127.0.0.1:${PORT}/v1"

for i in $(seq 1 120); do
    curl -sf "${BASE_URL}/models" >/dev/null 2>&1 && break
    kill -0 "${SERVER_PID}" 2>/dev/null || { tail -60 "${LOG_DIR}/simplelm.${SLURM_JOB_ID}.log"; exit 1; }
    sleep 10
done
echo "[tool_smoke] server up after $((i * 10))s"

# Send a chat completion with a single tool def. The prompt asks the
# model to use it; that's a strong nudge.
RESP=$(curl -sf -X POST "${BASE_URL}/chat/completions" \
    -H "Content-Type: application/json" \
    -d "$(jq -nc \
        --arg model "${MODEL_NAME}" \
        '{
          model: $model,
          messages: [
            {role: "system", content: "You are a helpful assistant. When asked about the weather, call the get_weather function. Do NOT answer in prose. Only emit a tool call."},
            {role: "user", content: "What is the weather in Paris right now?"}
          ],
          tools: [{
            type: "function",
            function: {
              name: "get_weather",
              description: "Get current weather for a city",
              parameters: {
                type: "object",
                properties: { city: { type: "string", description: "city name" } },
                required: ["city"]
              }
            }
          }],
          tool_choice: "auto",
          max_tokens: 200,
          temperature: 0.0
        }')")

echo "[tool_smoke] raw response:"
echo "${RESP}" | head -c 1500; echo

# Verify tool_calls in response
RESULT=$(echo "${RESP}" | python3 -c "
import sys, json
d = json.load(sys.stdin)
msg = d['choices'][0]['message']
tcs = msg.get('tool_calls', []) or []
if tcs:
    fn = tcs[0]['function']
    print(f'PASS: tool_call name={fn[\"name\"]!r} args={fn[\"arguments\"]!r}')
else:
    print(f'FAIL: no tool_calls. content={msg.get(\"content\",\"\")!r}')
")
echo "[tool_smoke] ${RESULT}"

# cleanup
kill "${SERVER_PID}" 2>/dev/null || true
sleep 2
kill -9 "${SERVER_PID}" 2>/dev/null || true
wait 2>/dev/null || true

case "${RESULT}" in
    PASS:*) exit 0 ;;
    *) exit 2 ;;
esac

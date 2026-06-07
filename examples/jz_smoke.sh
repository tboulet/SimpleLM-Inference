#!/bin/bash
# SimpleLM smoke on JZ via the ClusterControl pipeline.
#
# Run via run_inference_job.sh as CLIENT_COMMAND. Assumes:
#   - venv with simplelm installed (`pip install -e $WORK/SimpleLM-Inference`)
#   - $MODEL_PATH / $MODEL_NAME set by the launcher

set -uo pipefail

PORT=${PORT:-9876}
TOOL_PARSER=${TOOL_PARSER:-noop}
LOG_DIR="${OUTPUT_DIR:-${SCRATCH}/experiments_logs}"
mkdir -p "${LOG_DIR}"

echo "[simplelm_smoke] starting on :${PORT}  model=${MODEL_PATH}  parser=${TOOL_PARSER}"

simplelm serve \
    --model-path "${MODEL_PATH}" \
    --model-name "${MODEL_NAME}" \
    --port "${PORT}" \
    --tool-parser "${TOOL_PARSER}" \
    > "${LOG_DIR}/simplelm.${SLURM_JOB_ID}.log" 2>&1 &
SERVER_PID=$!
echo "[simplelm_smoke] PID=${SERVER_PID}"
BASE_URL="http://127.0.0.1:${PORT}/v1"

for i in $(seq 1 120); do
    if curl -sf "${BASE_URL}/models" >/dev/null 2>&1; then
        echo "[simplelm_smoke] ✓ ready after $((i * 10))s"
        break
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "[simplelm_smoke] ✗ server died at ~$((i * 10))s"
        tail -60 "${LOG_DIR}/simplelm.${SLURM_JOB_ID}.log"
        exit 1
    fi
    sleep 10
done

RESP=$(curl -sf -X POST "${BASE_URL}/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"${MODEL_NAME}\",
        \"messages\": [{\"role\": \"user\", \"content\": \"In ONE short sentence: what is the capital of France?\"}],
        \"max_tokens\": 60,
        \"temperature\": 0.1
    }")
echo "[simplelm_smoke] response:"
echo "${RESP}" | head -c 800; echo
CONTENT=$(echo "${RESP}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'])" 2>&1)
echo "[simplelm_smoke] content: ${CONTENT}"

kill "${SERVER_PID}" 2>/dev/null || true
sleep 2
kill -9 "${SERVER_PID}" 2>/dev/null || true
wait 2>/dev/null || true

if echo "${CONTENT}" | grep -qiE "paris"; then
    echo "[simplelm_smoke] PASS"
    exit 0
fi
echo "[simplelm_smoke] FAIL"
exit 2

# Example: curl a SimpleLM server

Start the server (assumes a HF snapshot of Qwen2.5-3B at the path):

```bash
simplelm serve \
    --model-path $HF_HOME/Qwen2.5-3B-Instruct \
    --port 9876 \
    --tool-parser noop
```

Wait for `INFO: Application startup complete.` then:

## 1. List models

```bash
curl -s http://127.0.0.1:9876/v1/models | jq
```

```json
{
  "object": "list",
  "data": [{"id": "Qwen2.5-3B-Instruct", "object": "model", "owned_by": "simplelm", "created": 1780854000}]
}
```

## 2. Chat completion (text-only)

```bash
curl -s -X POST http://127.0.0.1:9876/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen2.5-3B-Instruct",
        "messages": [{"role": "user", "content": "In ONE sentence: what is the capital of France?"}],
        "max_tokens": 60,
        "temperature": 0.1
    }' | jq
```

## 3. Chat completion with image (multimodal)

For a vision-capable model (gemma-3, Qwen2.5-VL, etc), pass the image
as a `data:` URL. Compute-node-safe: no outbound HTTPS needed.

```bash
# Base64-encode a local image
B64=$(base64 -w0 cat.jpg)
curl -s -X POST http://127.0.0.1:9876/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "$(jq -nc \
        --arg b64 "$B64" \
        '{
          model: "gemma-3-12b-it",
          messages: [{
            role: "user",
            content: [
              {type: "text", text: "What animal do you see? One sentence."},
              {type: "image_url", image_url: {url: ("data:image/jpeg;base64," + $b64)}}
            ]
          }],
          max_tokens: 80,
          temperature: 0.1
        }')"
```

## 4. With tool definitions (no actual tool execution)

```bash
curl -s -X POST http://127.0.0.1:9876/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "gemma-4-31B-it",
        "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"]
                }
            }
        }]
    }' | jq .choices[0].message
```

If the model emits a tool call in its native format, the configured
`--tool-parser` (e.g. `--tool-parser gemma4`) extracts it into
`tool_calls`. With `--tool-parser noop` the call appears in
`message.content` verbatim.

## 5. With Python OpenAI client

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:9876/v1", api_key="anything")
resp = client.chat.completions.create(
    model="Qwen2.5-3B-Instruct",
    messages=[{"role": "user", "content": "Hello?"}],
    max_tokens=40,
)
print(resp.choices[0].message.content)
```

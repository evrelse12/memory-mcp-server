# Installation

## Requirements
- Python 3.10–3.14 with a SQLite build that allows `enable_load_extension`.
  **macOS note:** python.org builds disable extension loading, so `sqlite-vec`
  cannot load; use a Homebrew/conda Python or run on Linux/Windows.
- An OpenAI-compatible **embedding** server and **chat** server (llama.cpp
  `llama-server` works well). Both can run on CPU.

## Install
```bash
git clone https://github.com/evrelse12/memory-mcp-server.git
cd memory-mcp-server
python3 -m venv venv
./venv/bin/pip install -e .          # add ".[dev]" for tests/lint
```

## Embedding server (example: nomic, CPU)
```bash
llama-server -m nomic-embed-text-v2-moe-q8_0.gguf \
  --embedding --pooling mean -ngl 0 --host 127.0.0.1 --port 8081
```
Match `MEMORY_EMBEDDING_MODEL` / `MEMORY_EMBEDDING_DIM` to this model (see
[CONFIGURATION.md](CONFIGURATION.md)). A systemd unit example:
```ini
[Unit]
Description=nomic embeddings
After=network.target
[Service]
User=blake
Environment=GGML_VULKAN=0 GGML_SYCL=0 GGML_CUDA=0
ExecStart=/home/blake/llama/bin/llama-server -m /home/blake/models/nomic-embed-text-v2-moe-q8_0.gguf \
  -c 512 -b 512 -ngl 0 --host 127.0.0.1 --port 8081 --embedding --pooling mean
Restart=always
[Install]
WantedBy=multi-user.target
```

## Chat server (example)
```bash
llama-server -m <chat-model>.gguf -a <alias> --jinja --host 127.0.0.1 --port 9931
```
Reasoning models are supported: empty `content` with text in `reasoning_content`
is handled automatically.

## Run as MCP server
stdio (default):
```bash
./venv/bin/memory-mcp
```
HTTP + bearer: see [HTTP.md](HTTP.md).

## Wire into a host (llama.cpp example)
`mcp-config.json`:
```json
{ "mcpServers": { "memory": {
  "command": "/path/to/venv/bin/python3",
  "args": ["/path/to/main.py"],
  "env": {
    "MEMORY_DATA_DIR": "/path/to/project",
    "MEMORY_LLM_API_URL": "http://localhost:9931",
    "MEMORY_EMBEDDING_API_URL": "http://localhost:8081",
    "FASTMCP_CHECK_FOR_UPDATES": "off",
    "FASTMCP_TELEMETRY_MODE": "off"
  }
}}}
```
Full env reference: [CONFIGURATION.md](CONFIGURATION.md).

## Verify
```bash
./venv/bin/pytest -q                 # offline test-suite
./venv/bin/python main.py < /dev/null   # should start and list DB path
```
If a host reports "0 tools", see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

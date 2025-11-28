# Pensieve Local LLM Server

**Run Claude Code with a local LLM on Apple Silicon.**

Pensieve provides a local "brain" for Claude Code, replacing the Anthropic API with a local Phi-3 model running on your Mac's GPU (via MLX).

## Prerequisites
1. **macOS with Apple Silicon** (M1/M2/M3/M4)
2. **Python 3.9+**
3. **Claude Code** (`npm install -g @anthropic-ai/claude-code`)

## Usage

There is only one script you need: `pensieve`.

### Flow 1: The Happy Path (Daily Usage)
When you want to work with Claude locally:

```bash
./pensieve
```
1. The script checks that everything is ready.
2. It starts the server (if not already running).
3. It launches a `claude` session connected to your local model.
4. **When you are done**, simply press `Ctrl-C` to exit Claude. The server shuts down automatically.

### Flow 2: First Run (Installation)
The first time you run `./pensieve`, it handles the setup automatically:

1. It detects missing dependencies and installs them (`mlx-lm`, `fastapi`, etc.).
2. It detects the missing model and downloads `Phi-3-mini-128k-instruct-4bit` (~2.5GB).
3. Once downloaded, it proceeds to start the server and launch Claude.

## Architecture
The project has been simplified for distribution:

- **`pensieve`**: The master launcher script. Handles lifecycle, updates, and execution.
- **`src/`**: The Python source code for the inference server.
- **`zz-archive/`**: Legacy Rust code and experiments.

## Troubleshooting

**"command not found: claude"**
Install Claude Code: `npm install -g @anthropic-ai/claude-code`

**"Server failed to start"**
Check `~/.local/share/pensieve/server.log` for details.

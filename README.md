# Pensieve Local LLM Server

**Run Claude Code with a local LLM on Apple Silicon - no API fees, complete privacy.**

Pensieve replaces the "brain" of the `claude` CLI with a local model running on your Mac's GPU (via MLX).

---

## Quick Start (The "Smart Launcher" Way)

We provide a single script that handles everything: checks dependencies, downloads the model (if missing), starts the background server, and launches Claude.

### 1. Prerequisites
- **macOS with Apple Silicon** (M1/M2/M3/M4)
- **Python 3.9+**
- **Claude Code** installed (`npm install -g @anthropic-ai/claude-code`)

### 2. Run it

```bash
# 1. Get the code
git clone https://github.com/that-in-rust/pensieve-local-llm-server.git
cd pensieve-local-llm-server

# 2. Launch (Handles setup, server start, and Claude session)
./scripts/pensieve
```

*That's it. The script will download the model (first time only), start the server, and drop you into Claude.*

---

## Manual Setup (Under the Hood)

If you prefer to do things manually:

1.  **Install Deps**: `pip3 install -r python_bridge/requirements.txt huggingface_hub`
2.  **Download Model**: `python3 -m huggingface_hub.cli download mlx-community/Phi-3-mini-128k-instruct-4bit --local-dir models/Phi-3-mini-128k-instruct-4bit`
3.  **Start Server**: `./scripts/start-mlx-server.sh`
4.  **Run Claude**: `./scripts/claude-local`

## Troubleshooting

**"Connection Refused"**
- The smart launcher `./scripts/pensieve` attempts to start the server automatically. If it fails, check `server.log`.

**"Model not found"**
- The script should download it automatically. Check your internet connection.

**"claude command not found"**
- You need to install Claude Code first: `npm install -g @anthropic-ai/claude-code`

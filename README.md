# Pensieve Local LLM Server

**Run Claude Code with a local LLM on Apple Silicon - no API fees, complete privacy.**

Pensieve replaces the "brain" of the `claude` CLI with a local model running on your Mac's GPU (via MLX). It tricks Claude Code into thinking it's talking to the Anthropic API, while actually routing everything to a local Phi-3 model.

---

## Quick Start

### 1. Prerequisites
- **macOS with Apple Silicon** (M1/M2/M3/M4)
- **Python 3.9+**
- **Claude Code** installed (`npm install -g @anthropic-ai/claude-code` or similar)

### 2. Setup (Run Once)

```bash
# 1. Install Python dependencies
pip3 install -r python_bridge/requirements.txt huggingface_hub

# 2. Download the Model (Phi-3 Mini 4-bit)
python3 -m huggingface_hub.cli download mlx-community/Phi-3-mini-128k-instruct-4bit --local-dir models/Phi-3-mini-128k-instruct-4bit
```

### 3. Start the Server

Open a terminal window and run the persistent server:

```bash
./scripts/start-mlx-server.sh
```
*You will see "Server Ready - Model Resident in Memory". Keep this window open.*

### 4. Run Claude Code (Session-Specific)

In a **new terminal window**, run the wrapper script. This configures Claude to use your local server *just for this session*:

```bash
./scripts/claude-local
```

You can now use Claude as normal, but it's running locally!

---

## How It Works

1.  **Native Compatibility**: The Pensieve server (`mlx_server.py`) implements the Anthropic `/v1/messages` API directly.
2.  **Session Isolation**: `claude-local` sets `ANTHROPIC_BASE_URL=http://127.0.0.1:8765` only for the current process. It does **not** modify your global `~/.claude/config.json`.
3.  **Memory Efficient**: The server loads the model once (~2.5GB RAM) and keeps it resident, avoiding the memory spikes of typical CLI tools.

## Performance
- **Model**: Phi-3-mini-128k-instruct-4bit
- **Memory Usage**: ~2.5 GB (Verified)
- **Speed**: ~25+ Tokens/Second on M-series chips

## Troubleshooting

**"Connection Refused"**
- Ensure `./scripts/start-mlx-server.sh` is running in a separate terminal.
- Check if port 8765 is free.

**"Model not found"**
- Ensure you ran the download command in Step 2 exactly as written.
- Check `ls -la models/` to verify files exist.

**Claude is "dumb"**
- Remember: You are replacing Claude 3.5 Sonnet (Huge) with Phi-3 Mini (Tiny). It will be faster and free, but less capable at complex reasoning or large coding tasks.

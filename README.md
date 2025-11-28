# Pensieve Local LLM Server

**Run Claude Code with a local LLM on Apple Silicon - no API fees, complete privacy.**

Pensieve replaces the "brain" of the `claude` CLI with a local model running on your Mac's GPU (via MLX).

---

## Quick Start (One-Line Install)

Copy and run this command. It handles everything: cloning, dependencies, model download, server startup, and launching Claude.

```bash
curl -sL https://raw.githubusercontent.com/that-in-rust/pensieve-local-llm-server/main/scripts/install.sh | bash
```

*Note: Requires macOS (Apple Silicon), Python 3.9+, and Claude Code.*

---

## Usage

After installation, you can run it anytime using the alias installed to `~/.local/bin/pensieve`:

```bash
# Make sure ~/.local/bin is in your PATH
pensieve
```

Or manually run the script from the install directory:
`~/.local/share/pensieve-server/scripts/pensieve`

---

## How It Works

1.  **Installs** to `~/.local/share/pensieve-server`.
2.  **Downloads** the Phi-3 model (first run only).
3.  **Starts** the local inference server in the background.
4.  **Launches** `claude` configured to talk to localhost.

## Troubleshooting

**"command not found: pensieve"**
- Add `export PATH=$PATH:~/.local/bin` to your `~/.zshrc`.

**"claude command not found"**
- Install Claude Code: `npm install -g @anthropic-ai/claude-code`

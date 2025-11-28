
# pensieve-local-llm-server

Ultra-minimal Apple Silicon binary that ships a preconfigured Claude-compatible endpoint powered by Phi-4 advanced reasoning model.

## Quick Start (Raw GitHub shell plan)

```bash
# From any folder
curl -sSf https://raw.githubusercontent.com/that-in-rust/pensieve-local-llm-server/main/pensieve.sh | bash
```

The `pensieve.sh` helper script:
1. Fetches the latest `pensieve-local-llm-server` binary from GitHub releases.
2. Ensures the prebuilt `Phi-4-reasoning-plus-4bit` MLX bundle is present (downloads with resume + checksum if missing).
3. Runs the binary with the baked-in configuration so no extra flags are needed.

## Zero-Config Behavior

The binary still uses the clap-based CLI stack, but every flag is hard-coded:
- **Model**: `Phi-4-reasoning-plus-4bit` (advanced reasoning model with MLX optimization in `models/phi-4-reasoning-plus-4bit/`).
- **Port**: `528491` on localhost.
- **API surface**: `/v1/messages` (Anthropic-compatible) + `/health`.

Running `pensieve-local-llm-server` directly is equivalent to `pensieve.sh`’s final step.

## User Journey

### First Run
1. Execute the curl | bash snippet above.
2. Script verifies Apple Silicon + MLX prerequisites.
3. Script downloads/releases the binary, fetches the MLX bundle if absent, then launches the server.
4. Console logs show download progress and conclude with `http://127.0.0.1:528491` ready for requests.

### Subsequent Runs
1. Re-run `pensieve-local-llm-server` (or the script) from any folder.
2. Cache hit skips downloads; startup completes in <10 s before serving Anthropic-compatible traffic.

## Implementation Snapshot
1. **Single Binary Architecture** – p01–p09 crates wired into one binary.
2. **Hard-Coded CLI Inputs** – clap wiring retained but only the baked-in model + port are honored.
3. **Prebuilt Model Cache** – Phi-4-reasoning-plus-4bit bundle pulled from mlx-community with checksum verification; advanced reasoning capabilities out-of-the-box.
4. **Apple Silicon MLX Acceleration** – All inference logic runs in Rust, driving MLX through Rust↔C bindings for Metal execution.
5. **Anthropic Compatibility** – `/v1/messages` endpoint mirrors Claude API semantics; `/health` for readiness checks.
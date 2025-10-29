# 🚀 Pensieve + Claude Code - Dead Simple Setup

## ⚡ 30-Second Setup

### 1. Login to Claude Code (One Time)

```bash
claude
# Select login method → Complete auth → type "exit"
```

### 2. Run Setup

```bash
cd /path/to/pensieve-local-llm-server
./scripts/setup-pensieve
```

### 3. Done! Now use it:

```bash
claude-local     # Chat with local Pensieve (private, free)
claude-cloud     # Chat with Anthropic cloud (latest models)
```

That's it! Both commands are ready to use immediately.

---

## 🎯 Daily Usage

After setup, it's dead simple:

### Use Local Server (Private, Free, Your Mac)

```bash
claude-local
```

- Automatically starts server if needed
- Remembers this choice for next time
- All data stays on your machine
- Zero API costs

### Use Cloud API (Latest Models, Full Power)

```bash
claude-cloud
```

- Uses Anthropic's cloud infrastructure
- Access to latest Claude models
- Remembers this choice for next time
- Standard API billing applies

### Switch Anytime

```bash
# Use local today
claude-local

# Exit and use cloud tomorrow
claude-cloud
```

No configuration, no restarts, no hassle.

---

## 🔧 Server Management

### Check Status

```bash
pensieve-server status
```

Shows if server is running, health check, memory usage.

### Manual Control

```bash
pensieve-server start      # Start server
pensieve-server stop       # Stop server
pensieve-server restart    # Restart server
pensieve-server logs       # View live logs
```

---

## 💡 How It Works

### `claude-local`
1. Checks if Claude Code is authenticated
2. Starts Pensieve server if not running
3. Updates `~/.claude/settings.json` to point to `127.0.0.1:7777`
4. Launches Claude Code
5. Done!

### `claude-cloud`
1. Checks if Claude Code is authenticated
2. Restores original cloud configuration
3. Launches Claude Code
4. Done!

### Smart Features
- ✅ Auto-detects best install location (/opt/homebrew/bin, /usr/local/bin, ~/.local/bin)
- ✅ Backs up your settings before any changes
- ✅ Preserves your preferences (like alwaysThinkingEnabled)
- ✅ Server starts automatically when using `claude-local`
- ✅ Configuration switches automatically - no manual edits
- ✅ Works from any directory

---

## 🐛 Troubleshooting

### "Claude Code not logged in"

You need to login once:
```bash
claude
# Complete authentication
exit
```

Then run `claude-local` or `claude-cloud` again.

### "Address already in use"

Another server is on port 7777:
```bash
pensieve-server stop
claude-local
```

### "Want to reset everything"

Your original settings are backed up:
```bash
ls ~/.claude/settings.json.backup-*
cp ~/.claude/settings.json.backup-<timestamp> ~/.claude/settings.json
```

### "Commands not found"

Add to PATH (setup script will tell you):
```bash
# Add to ~/.zshrc
export PATH="/opt/homebrew/bin:$PATH"
```

---

## 📁 What Gets Installed

```
/opt/homebrew/bin/          (or /usr/local/bin or ~/.local/bin)
├── claude-local            → Smart wrapper for local server
├── claude-cloud            → Smart wrapper for cloud API
└── pensieve-server         → Server management tool

~/.claude/
├── settings.json           → Current configuration (auto-managed)
├── settings.json.cloud     → Cloud backup (for quick restore)
└── settings.json.backup-*  → Timestamped backups
```

---

## 🎉 Why This Is Better

### Before
1. Set environment variables
2. Edit ~/.claude/settings.json manually
3. Restart Claude Code
4. Hope it works
5. Repeat for every switch

### After
1. Run `setup-pensieve` once
2. Use `claude-local` or `claude-cloud`
3. Done!

**5 manual steps → 1 command**

---

## 🔒 Privacy Note

When using `claude-local`:
- All inference happens on your Mac
- No data sent to external APIs
- Model runs locally via MLX framework
- API key is dummy value (server doesn't validate)
- Complete privacy and control

When using `claude-cloud`:
- Standard Anthropic API communication
- Subject to Anthropic's privacy policy
- Latest models and features
- Billed through your Anthropic account

---

## 🆘 Need Help?

**Check what's running:**
```bash
pensieve-server status
cat ~/.claude/settings.json
```

**Reset to cloud:**
```bash
claude-cloud
```

**Reset to local:**
```bash
claude-local
```

**See server logs:**
```bash
pensieve-server logs
```

---

## ✨ Advanced: What the Scripts Do

### `setup-pensieve`
- Detects if Claude Code is authenticated
- Chooses best install location automatically
- Creates backups of current settings
- Installs wrapper scripts
- Shows clear next steps

### `claude-local`
- Validates authentication
- Auto-starts Pensieve server if needed
- Configures settings for local endpoint
- Launches Claude Code

### `claude-cloud`
- Validates authentication
- Restores cloud configuration from backup
- Launches Claude Code

### `pensieve-server`
- Comprehensive server management
- Health monitoring
- Handles "port in use" errors
- Process tracking via PID file
- Real-time log viewing

---

**That's it! Maximum convenience, minimum hassle.** 🚀

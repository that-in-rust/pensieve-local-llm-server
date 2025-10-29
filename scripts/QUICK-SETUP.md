# 🚀 QUICK SETUP - Pensieve + Claude Code

## ⚡ Super Fast Setup (2 Minutes)

### Step 1: Login to Claude Code (One Time Only)

```bash
claude
# Select "Anthropic Console account" → Complete auth → type "exit"
```

### Step 2: Run Installer

```bash
cd /path/to/pensieve-local-llm-server
./scripts/install.sh
```

That's it! The installer will:
- Configure settings for local server
- Install server management tools
- Start the server
- You're ready to use `claude` with Pensieve

---

## 🎯 Daily Usage

After setup, it's super simple:

```bash
# Just run Claude normally - it uses local server
claude
```

```bash
# Server management (if needed)
pensieve-server status
pensieve-server restart
pensieve-server logs
```

```bash
# Switch between local and cloud
pensieve-switch local   # Use Pensieve
pensieve-switch cloud   # Use Anthropic API
exit                    # Exit Claude
claude                  # Relaunch with new config
```

---

## ❓ Why Do I Need to Login First?

Claude Code requires base authentication before it accepts any configuration. Think of it like:

1. **Base Auth** (one time): "Prove you have access to Claude"
2. **Endpoint Override** (installer): "Now use this local server instead"

Without step 1, Claude won't even look at the configuration in step 2.

---

## 🔥 The Cool Part

Once setup is complete, you just run:

```bash
claude
```

And it automatically uses your local Pensieve server! No wrappers needed, no special commands - just `claude`.

Want to try the latest Claude models from Anthropic?

```bash
pensieve-switch cloud
exit
claude
```

Done! That simple.

---

## 🐛 Troubleshooting

**"Still seeing login screen"**
- You need to complete the login once
- Select "Anthropic Console account"
- Complete the browser auth
- Type `exit` to return

**"Address already in use"**
```bash
pensieve-server stop
pensieve-server start
```

**"Want to reset everything"**
```bash
mv ~/.claude/settings.json.cloud-backup ~/.claude/settings.json
```

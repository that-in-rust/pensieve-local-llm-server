#!/bin/bash
# Pensieve + Claude Code Integration Installer
# ULTRATHINK: Comprehensive setup with login handling, backups, and testing

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CLAUDE_DIR="$HOME/.claude"
SETTINGS_FILE="$CLAUDE_DIR/settings.json"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

# Banner
echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║                                                              ║${NC}"
echo -e "${CYAN}║    ${GREEN}Pensieve Local LLM Server + Claude Code Installer${CYAN}       ║${NC}"
echo -e "${CYAN}║    ${BLUE}Comprehensive Setup with ULTRATHINK Logic${CYAN}               ║${NC}"
echo -e "${CYAN}║                                                              ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Step 1: Check Prerequisites
echo -e "${BLUE}[1/7]${NC} Checking prerequisites..."

# Check if Claude Code is installed
if ! command -v claude &> /dev/null; then
    echo -e "${RED}❌ Claude Code is not installed${NC}"
    echo -e "${YELLOW}💡 Install from: https://claude.com/download${NC}"
    exit 1
fi
echo -e "${GREEN}✓${NC} Claude Code is installed"

# Check if project is built
if [ ! -f "$PROJECT_DIR/target/debug/pensieve" ]; then
    echo -e "${YELLOW}⚠️  Pensieve binary not found${NC}"
    echo -e "${YELLOW}💡 Building project...${NC}"
    cd "$PROJECT_DIR"
    cargo build --workspace
    echo -e "${GREEN}✓${NC} Project built successfully"
else
    echo -e "${GREEN}✓${NC} Pensieve binary exists"
fi

# Check if model exists
MODEL_PATH="$PROJECT_DIR/models/Phi-3-mini-128k-instruct-4bit/model.safetensors"
if [ ! -f "$MODEL_PATH" ]; then
    echo -e "${YELLOW}⚠️  Model file not found: $MODEL_PATH${NC}"
    echo -e "${YELLOW}💡 Please download the model first${NC}"
fi

echo ""

# Step 2: Check Claude Code Login Status
echo -e "${BLUE}[2/7]${NC} Checking Claude Code authentication..."

NEEDS_LOGIN=false

if [ ! -d "$CLAUDE_DIR" ]; then
    echo -e "${YELLOW}⚠️  Claude directory not found${NC}"
    NEEDS_LOGIN=true
elif [ ! -f "$SETTINGS_FILE" ]; then
    echo -e "${YELLOW}⚠️  Settings file not found${NC}"
    NEEDS_LOGIN=true
else
    # Check if settings has any auth (cloud or local)
    if grep -q "ANTHROPIC_BASE_URL.*127.0.0.1" "$SETTINGS_FILE" 2>/dev/null; then
        echo -e "${GREEN}✓${NC} Local server config found (will update if needed)"
    else
        echo -e "${GREEN}✓${NC} Claude Code settings found"
    fi
fi

if [ "$NEEDS_LOGIN" = true ]; then
    echo ""
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║  ${YELLOW}FIRST-TIME SETUP: Claude Code Login Required${CYAN}            ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${YELLOW}You need to login to Claude Code once before we can configure it.${NC}"
    echo ""
    echo -e "${BLUE}Steps:${NC}"
    echo -e "  1. We'll launch Claude Code login"
    echo -e "  2. Complete the login process"
    echo -e "  3. Type ${GREEN}exit${NC} to return here"
    echo -e "  4. Installation will continue automatically"
    echo ""
    read -p "Press ENTER to launch Claude Code login..."

    # Launch Claude for login
    echo ""
    echo -e "${BLUE}→ Launching Claude Code...${NC}"
    claude || true

    echo ""
    echo -e "${GREEN}✓${NC} Welcome back! Continuing installation..."
    echo ""
    sleep 1
fi

# Step 3: Determine Install Location for Scripts
echo -e "${BLUE}[3/7]${NC} Determining installation location..."

# Use INSTALL_DIR from environment, or auto-detect best location
if [ -n "$INSTALL_DIR" ]; then
    INSTALL_DIR="$INSTALL_DIR"
elif [ -w "/opt/homebrew/bin" ] 2>/dev/null; then
    INSTALL_DIR="/opt/homebrew/bin"
elif [[ ":$PATH:" == *":/usr/local/bin:"* ]] && [ -w "/usr/local/bin" ] 2>/dev/null; then
    INSTALL_DIR="/usr/local/bin"
else
    INSTALL_DIR="$HOME/.local/bin"
fi

echo -e "${GREEN}✓${NC} Install location: ${CYAN}$INSTALL_DIR${NC}"

# Create directory if needed
if [ ! -d "$INSTALL_DIR" ]; then
    mkdir -p "$INSTALL_DIR"
    echo -e "${GREEN}✓${NC} Created directory: $INSTALL_DIR"
fi

# Check if in PATH
if [[ ":$PATH:" != *":$INSTALL_DIR:"* ]]; then
    echo -e "${YELLOW}⚠️  $INSTALL_DIR is not in your PATH${NC}"
    echo -e "${YELLOW}💡 Add this to your ~/.zshrc:${NC}"
    echo -e "   ${CYAN}export PATH=\"$INSTALL_DIR:\$PATH\"${NC}"
    echo ""
fi

echo ""

# Step 4: Install Scripts
echo -e "${BLUE}[4/7]${NC} Installing management scripts..."

ln -sf "$SCRIPT_DIR/pensieve-server" "$INSTALL_DIR/pensieve-server"
echo -e "${GREEN}✓${NC} Installed: ${CYAN}pensieve-server${NC}"

# Note: We're NOT installing claude-local/claude-cloud wrappers anymore
# Instead, we'll configure settings.json directly

echo ""

# Step 5: Backup and Configure settings.json
echo -e "${BLUE}[5/7]${NC} Configuring Claude Code for local server..."

# Always backup existing settings
if [ -f "$SETTINGS_FILE" ]; then
    BACKUP_FILE="$CLAUDE_DIR/settings.json.backup-$TIMESTAMP"
    cp "$SETTINGS_FILE" "$BACKUP_FILE"
    echo -e "${GREEN}✓${NC} Backed up settings: ${CYAN}settings.json.backup-$TIMESTAMP${NC}"

    # Check if already configured for local
    if grep -q "ANTHROPIC_BASE_URL.*127.0.0.1:7777" "$SETTINGS_FILE" 2>/dev/null; then
        echo -e "${YELLOW}⚠️  Settings already configured for local server${NC}"
        echo ""
        read -p "Reconfigure anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo -e "${BLUE}ℹ️  Skipping settings configuration${NC}"
            SKIP_SETTINGS=true
        fi
    fi
fi

if [ "$SKIP_SETTINGS" != true ]; then
    # Read existing settings to preserve other options
    if [ -f "$SETTINGS_FILE" ]; then
        # Try to preserve alwaysThinkingEnabled setting
        ALWAYS_THINKING=$(grep -o '"alwaysThinkingEnabled"[[:space:]]*:[[:space:]]*[^,}]*' "$SETTINGS_FILE" 2>/dev/null | grep -o 'true\|false' || echo "true")
    else
        ALWAYS_THINKING="true"
    fi

    # Write new settings with local server config
    cat > "$SETTINGS_FILE" << EOF
{
  "env": {
    "ANTHROPIC_API_KEY": "test-api-key-12345",
    "ANTHROPIC_BASE_URL": "http://127.0.0.1:7777"
  },
  "alwaysThinkingEnabled": $ALWAYS_THINKING
}
EOF

    echo -e "${GREEN}✓${NC} Configured settings.json for local server"
fi

# Create cloud backup for easy switching
CLOUD_BACKUP="$CLAUDE_DIR/settings.json.cloud-backup"
if [ -f "$BACKUP_FILE" ] && [ "$SKIP_SETTINGS" != true ]; then
    if ! grep -q "127.0.0.1:7777" "$BACKUP_FILE" 2>/dev/null; then
        cp "$BACKUP_FILE" "$CLOUD_BACKUP"
        echo -e "${GREEN}✓${NC} Saved cloud config: ${CYAN}settings.json.cloud-backup${NC}"
    fi
fi

echo ""

# Step 6: Create Switcher Script
echo -e "${BLUE}[6/7]${NC} Creating configuration switcher..."

cat > "$INSTALL_DIR/pensieve-switch" << 'EOF'
#!/bin/bash
# Switch between local and cloud Claude Code configuration

CLAUDE_DIR="$HOME/.claude"
SETTINGS_FILE="$CLAUDE_DIR/settings.json"
LOCAL_BACKUP="$CLAUDE_DIR/settings.json.local-backup"
CLOUD_BACKUP="$CLAUDE_DIR/settings.json.cloud-backup"

case "${1:-}" in
    local)
        if [ -f "$LOCAL_BACKUP" ]; then
            cp "$LOCAL_BACKUP" "$SETTINGS_FILE"
            echo "✅ Switched to LOCAL server (127.0.0.1:7777)"
        else
            echo "❌ Local backup not found. Run install.sh again."
            exit 1
        fi
        ;;
    cloud)
        if [ -f "$CLOUD_BACKUP" ]; then
            cp "$CLOUD_BACKUP" "$SETTINGS_FILE"
            echo "✅ Switched to CLOUD API (api.anthropic.com)"
        else
            echo "❌ Cloud backup not found."
            exit 1
        fi
        ;;
    status)
        if grep -q "127.0.0.1:7777" "$SETTINGS_FILE" 2>/dev/null; then
            echo "🚀 Currently using: LOCAL server (127.0.0.1:7777)"
        else
            echo "☁️  Currently using: CLOUD API"
        fi
        ;;
    *)
        echo "Pensieve Configuration Switcher"
        echo ""
        echo "Usage: pensieve-switch {local|cloud|status}"
        echo ""
        echo "Commands:"
        echo "  local   - Switch to local Pensieve server"
        echo "  cloud   - Switch to Anthropic cloud API"
        echo "  status  - Show current configuration"
        exit 1
        ;;
esac
EOF

chmod +x "$INSTALL_DIR/pensieve-switch"

# Save local config as backup
if [ "$SKIP_SETTINGS" != true ]; then
    cp "$SETTINGS_FILE" "$CLAUDE_DIR/settings.json.local-backup"
    echo -e "${GREEN}✓${NC} Installed: ${CYAN}pensieve-switch${NC}"
fi

echo ""

# Step 7: Test and Summary
echo -e "${BLUE}[7/7]${NC} Testing installation..."

# Test if scripts are accessible
if command -v pensieve-server &> /dev/null; then
    echo -e "${GREEN}✓${NC} pensieve-server is in PATH"
else
    echo -e "${YELLOW}⚠️  pensieve-server not in PATH yet${NC}"
    echo -e "${YELLOW}💡 Add $INSTALL_DIR to PATH or run: export PATH=\"$INSTALL_DIR:\$PATH\"${NC}"
fi

if command -v pensieve-switch &> /dev/null; then
    echo -e "${GREEN}✓${NC} pensieve-switch is in PATH"
fi

echo ""

# Installation Summary
echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║                                                              ║${NC}"
echo -e "${CYAN}║  ${GREEN}✅ Installation Complete!${CYAN}                                ║${NC}"
echo -e "${CYAN}║                                                              ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

echo -e "${BLUE}📚 What was installed:${NC}"
echo -e "  ${GREEN}•${NC} pensieve-server  → Server management (start/stop/status)"
echo -e "  ${GREEN}•${NC} pensieve-switch  → Switch between local/cloud configs"
echo -e "  ${GREEN}•${NC} Settings backup  → ${CYAN}$BACKUP_FILE${NC}"
echo ""

echo -e "${BLUE}🚀 Quick Start (3 steps):${NC}"
echo ""
echo -e "  ${YELLOW}1.${NC} Start the server:"
echo -e "     ${CYAN}pensieve-server start${NC}"
echo ""
echo -e "  ${YELLOW}2.${NC} Launch Claude Code:"
echo -e "     ${CYAN}claude${NC}"
echo ""
echo -e "  ${YELLOW}3.${NC} Test it:"
echo -e "     ${CYAN}Ask: \"What is 2+2?\" (should get response from local Phi-3)${NC}"
echo ""

echo -e "${BLUE}🔧 Server Management:${NC}"
echo -e "  ${CYAN}pensieve-server start${NC}     # Start the server"
echo -e "  ${CYAN}pensieve-server stop${NC}      # Stop the server"
echo -e "  ${CYAN}pensieve-server status${NC}    # Check health"
echo -e "  ${CYAN}pensieve-server logs${NC}      # View logs"
echo -e "  ${CYAN}pensieve-server restart${NC}   # Restart"
echo ""

echo -e "${BLUE}🔀 Switching Configurations:${NC}"
echo -e "  ${CYAN}pensieve-switch local${NC}     # Use local Pensieve server"
echo -e "  ${CYAN}pensieve-switch cloud${NC}     # Use Anthropic cloud API"
echo -e "  ${CYAN}pensieve-switch status${NC}    # Show current config"
echo ""

echo -e "${BLUE}💡 Tips:${NC}"
echo -e "  ${GREEN}•${NC} No restart needed when switching configs"
echo -e "  ${GREEN}•${NC} Server auto-stops \"port in use\" errors"
echo -e "  ${GREEN}•${NC} Your original settings are backed up safely"
echo -e "  ${GREEN}•${NC} Switch back to cloud anytime: ${CYAN}pensieve-switch cloud${NC}"
echo ""

echo -e "${YELLOW}⚠️  Important:${NC} Claude Code needs to be ${GREEN}restarted${NC} after switching configs"
echo -e "   Just type ${CYAN}exit${NC} and run ${CYAN}claude${NC} again"
echo ""

# Offer to start server now
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
read -p "🚀 Start Pensieve server now? (y/n) " -n 1 -r
echo
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    "$INSTALL_DIR/pensieve-server" start || true
    echo ""
    echo -e "${GREEN}✨ All set! Now run: ${CYAN}claude${NC}"
else
    echo -e "${BLUE}👍 No problem! Start when ready: ${CYAN}pensieve-server start${NC}"
fi

echo ""
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

#!/bin/bash
# Install claude-local and claude-cloud wrapper scripts
#
# This script creates symlinks in ~/.local/bin for easy access

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Use INSTALL_DIR from environment, or auto-detect best location
if [ -n "$INSTALL_DIR" ]; then
    INSTALL_DIR="$INSTALL_DIR"
# Prefer /opt/homebrew/bin if it exists and is writable (no sudo needed)
elif [ -w "/opt/homebrew/bin" ] 2>/dev/null; then
    INSTALL_DIR="/opt/homebrew/bin"
# Fall back to /usr/local/bin if in PATH (may need sudo)
elif [[ ":$PATH:" == *":/usr/local/bin:"* ]]; then
    INSTALL_DIR="/usr/local/bin"
# Last resort: ~/.local/bin
else
    INSTALL_DIR="$HOME/.local/bin"
fi

echo "🔧 Installing Claude Code wrapper scripts..."
echo "📍 Install location: $INSTALL_DIR"
echo ""

# Create install directory if it doesn't exist
if [ ! -d "$INSTALL_DIR" ]; then
    echo "📁 Creating $INSTALL_DIR..."
    mkdir -p "$INSTALL_DIR"
fi

# Check if INSTALL_DIR is in PATH
if [[ ":$PATH:" != *":$INSTALL_DIR:"* ]]; then
    echo "⚠️  WARNING: $INSTALL_DIR is not in your PATH"
    echo ""
    if [[ "$INSTALL_DIR" == "$HOME/.local/bin" ]]; then
        echo "Add this to your ~/.zshrc or ~/.bashrc:"
        echo "    export PATH=\"\$HOME/.local/bin:\$PATH\""
    else
        echo "Add this to your ~/.zshrc or ~/.bashrc:"
        echo "    export PATH=\"$INSTALL_DIR:\$PATH\""
    fi
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "❌ Installation cancelled"
        exit 1
    fi
fi

# Create symlinks
echo "🔗 Creating symlinks..."
ln -sf "$SCRIPT_DIR/claude-local" "$INSTALL_DIR/claude-local"
ln -sf "$SCRIPT_DIR/claude-cloud" "$INSTALL_DIR/claude-cloud"
ln -sf "$SCRIPT_DIR/pensieve-server" "$INSTALL_DIR/pensieve-server"

echo "✅ Installation complete!"
echo ""
echo "📚 Usage:"
echo "    pensieve-server start     # Start Pensieve server"
echo "    pensieve-server stop      # Stop Pensieve server"
echo "    pensieve-server status    # Check server status"
echo "    pensieve-server restart   # Restart server"
echo "    pensieve-server logs      # View server logs"
echo ""
echo "    claude-local              # Launch Claude with local server (auto-starts server)"
echo "    claude-cloud              # Launch Claude with cloud API"
echo ""
echo "🎯 You can now open MULTIPLE terminals and run different sessions:"
echo "    Terminal 1: claude-local   (uses local server)"
echo "    Terminal 2: claude-cloud   (uses cloud API)"
echo ""
echo "💡 Both sessions can run simultaneously without conflicts!"
echo ""
echo "🚀 Quick start:"
echo "    pensieve-server start     # Start the server once"
echo "    claude-local              # Then use Claude with local server"

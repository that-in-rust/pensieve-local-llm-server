#!/bin/bash
#
# Pensieve Installer & Launcher
# Usage: curl -sL https://raw.githubusercontent.com/that-in-rust/pensieve-local-llm-server/main/scripts/install.sh | bash
#

set -e

INSTALL_DIR="${HOME}/.local/share/pensieve-server"
REPO_URL="https://github.com/that-in-rust/pensieve-local-llm-server.git"
BIN_DIR="${HOME}/.local/bin"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}[Pensieve Installer]${NC}"

# 1. Clone/Update Repo
if [ -d "$INSTALL_DIR" ]; then
    echo "Updating existing installation in $INSTALL_DIR..."
    cd "$INSTALL_DIR"
    git pull --quiet
else
    echo "Installing to $INSTALL_DIR..."
    mkdir -p "$(dirname "$INSTALL_DIR")"
    git clone --quiet "$REPO_URL" "$INSTALL_DIR"
fi

# 2. Setup Symlink (Optional convenience)
mkdir -p "$BIN_DIR"
ln -sf "$INSTALL_DIR/scripts/pensieve" "$BIN_DIR/pensieve"

# 3. Run the Launcher
echo -e "${GREEN}Installation complete!${NC}"
echo "Running Pensieve..."
exec "$INSTALL_DIR/scripts/pensieve"

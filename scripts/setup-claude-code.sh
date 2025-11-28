#!/bin/bash
# Pensieve Claude Code Integration Setup
#
# This script configures Claude Code to use the local Pensieve server
# instead of the cloud Anthropic API.
#
# Usage: ./scripts/setup-claude-code.sh
#
# What it does:
# 1. Creates onboarding bypass file (~/.claude.json)
# 2. Updates Claude Code settings (~/.claude/settings.json) using Node.js
# 3. Sets required environment variables for local inference

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔧 Pensieve: Setting up Claude Code for local inference${NC}"
echo ""

# Step 1: Create onboarding bypass
echo -e "${GREEN}  → Creating onboarding bypass...${NC}"
cat > "$HOME/.claude.json" << 'EOF'
{
  "hasCompletedOnboarding": true
}
EOF

# Step 2: Ensure .claude directory exists
mkdir -p "$HOME/.claude"

# Step 3: Update settings.json using Node.js (safe JSON manipulation)
echo -e "${GREEN}  → Updating Claude Code settings...${NC}"
node --eval "
const fs = require('fs');
const path = require('path');

const settingsPath = path.join(process.env.HOME, '.claude', 'settings.json');

// Load existing settings or create new
let settings;
try {
    const content = fs.readFileSync(settingsPath, 'utf-8');
    settings = JSON.parse(content);
} catch {
    settings = {};
}

// Update with Pensieve configuration
const updated = {
    ...settings,
    env: {
        ...(settings.env || {}),
        ANTHROPIC_AUTH_TOKEN: 'pensieve-local-token',
        ANTHROPIC_BASE_URL: 'http://127.0.0.1:7777',
        API_TIMEOUT_MS: '3000000'
    },
    alwaysThinkingEnabled: true
};

// Write back safely with pretty formatting
fs.writeFileSync(settingsPath, JSON.stringify(updated, null, 2), 'utf-8');
console.log('  ✅ settings.json updated');
"

echo ""
echo -e "${GREEN}✅ Setup complete!${NC}"
echo ""
echo "Next steps:"
echo "  1. Start Pensieve server:"
echo "     ${BLUE}cargo run -p pensieve-09-anthropic-proxy${NC}"
echo ""
echo "  2. Or use the CLI (when available):"
echo "     ${BLUE}pensieve start${NC}"
echo ""
echo "  3. Test Claude Code integration:"
echo "     ${BLUE}claude --print 'Say hello in 5 words'${NC}"
echo ""
echo "Settings configured:"
echo "  • Base URL: http://127.0.0.1:7777"
echo "  • Auth Token: pensieve-local-token"
echo "  • Timeout: 50 minutes (for local inference)"
echo ""

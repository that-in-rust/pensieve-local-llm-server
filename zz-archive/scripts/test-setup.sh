#!/bin/bash
# Test script for setup-claude-code.sh
#
# This script verifies that setup-claude-code.sh correctly configures
# Claude Code for local Pensieve usage.
#
# Usage: ./scripts/test-setup.sh

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🧪 Testing Pensieve Claude Code setup script${NC}"
echo ""

# Backup existing files if they exist
BACKUP_DIR="/tmp/pensieve-setup-test-backup-$$"
mkdir -p "$BACKUP_DIR"

if [ -f "$HOME/.claude.json" ]; then
    echo -e "${BLUE}  → Backing up existing ~/.claude.json${NC}"
    cp "$HOME/.claude.json" "$BACKUP_DIR/.claude.json"
fi

if [ -f "$HOME/.claude/settings.json" ]; then
    echo -e "${BLUE}  → Backing up existing ~/.claude/settings.json${NC}"
    cp "$HOME/.claude/settings.json" "$BACKUP_DIR/settings.json"
fi

# Run setup script
echo -e "${BLUE}  → Running setup script...${NC}"
./scripts/setup-claude-code.sh

# Test 1: Verify .claude.json was created
echo -e "${BLUE}  → Test 1: Checking ~/.claude.json exists${NC}"
if [ ! -f "$HOME/.claude.json" ]; then
    echo -e "${RED}FAIL: ~/.claude.json not created${NC}"
    exit 1
fi
echo -e "${GREEN}  ✅ ~/.claude.json exists${NC}"

# Test 2: Verify settings.json was created
echo -e "${BLUE}  → Test 2: Checking ~/.claude/settings.json exists${NC}"
if [ ! -f "$HOME/.claude/settings.json" ]; then
    echo -e "${RED}FAIL: ~/.claude/settings.json not created${NC}"
    exit 1
fi
echo -e "${GREEN}  ✅ ~/.claude/settings.json exists${NC}"

# Test 3: Verify settings.json has required fields
echo -e "${BLUE}  → Test 3: Verifying settings.json content${NC}"
node --eval "
const fs = require('fs');
const path = require('path');

const settingsPath = path.join(process.env.HOME, '.claude', 'settings.json');
const settings = JSON.parse(fs.readFileSync(settingsPath, 'utf-8'));

let failed = false;

// Test: ANTHROPIC_AUTH_TOKEN
if (!settings.env || !settings.env.ANTHROPIC_AUTH_TOKEN) {
    console.error('FAIL: Missing ANTHROPIC_AUTH_TOKEN');
    failed = true;
} else if (settings.env.ANTHROPIC_AUTH_TOKEN !== 'pensieve-local-token') {
    console.error('FAIL: ANTHROPIC_AUTH_TOKEN has wrong value:', settings.env.ANTHROPIC_AUTH_TOKEN);
    failed = true;
}

// Test: ANTHROPIC_BASE_URL
if (!settings.env || !settings.env.ANTHROPIC_BASE_URL) {
    console.error('FAIL: Missing ANTHROPIC_BASE_URL');
    failed = true;
} else if (!settings.env.ANTHROPIC_BASE_URL.includes('127.0.0.1:7777')) {
    console.error('FAIL: ANTHROPIC_BASE_URL incorrect:', settings.env.ANTHROPIC_BASE_URL);
    failed = true;
}

// Test: API_TIMEOUT_MS
if (!settings.env || !settings.env.API_TIMEOUT_MS) {
    console.error('FAIL: Missing API_TIMEOUT_MS');
    failed = true;
} else if (parseInt(settings.env.API_TIMEOUT_MS) < 3000000) {
    console.error('FAIL: API_TIMEOUT_MS too low:', settings.env.API_TIMEOUT_MS);
    failed = true;
}

// Test: alwaysThinkingEnabled
if (!settings.alwaysThinkingEnabled) {
    console.error('FAIL: alwaysThinkingEnabled not set to true');
    failed = true;
}

if (failed) {
    process.exit(1);
}

console.log('  ✅ All required fields present and correct');
"

# Test 4: Verify JSON is valid
echo -e "${BLUE}  → Test 4: Verifying JSON validity${NC}"
if ! jq empty "$HOME/.claude/settings.json" 2>/dev/null; then
    echo -e "${RED}FAIL: settings.json is not valid JSON${NC}"
    exit 1
fi
echo -e "${GREEN}  ✅ settings.json is valid JSON${NC}"

# Restore backups
echo ""
echo -e "${BLUE}  → Restoring original files...${NC}"
if [ -f "$BACKUP_DIR/.claude.json" ]; then
    cp "$BACKUP_DIR/.claude.json" "$HOME/.claude.json"
    echo -e "${GREEN}  ✅ Restored ~/.claude.json${NC}"
fi

if [ -f "$BACKUP_DIR/settings.json" ]; then
    cp "$BACKUP_DIR/settings.json" "$HOME/.claude/settings.json"
    echo -e "${GREEN}  ✅ Restored ~/.claude/settings.json${NC}"
fi

# Clean up backup directory
rm -rf "$BACKUP_DIR"

echo ""
echo -e "${GREEN}✅ All tests passed!${NC}"
echo ""
echo "The setup script correctly configures Claude Code for Pensieve."

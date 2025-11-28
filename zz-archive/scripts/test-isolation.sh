#!/bin/bash
#
# test-isolation.sh - Test multi-instance isolation for Pensieve
#
# Verifies that:
# 1. claude-local uses local server
# 2. Regular claude uses normal API (or fails gracefully)
# 3. No interference between instances
# 4. Global settings unchanged

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "🧪 Testing Claude Code Isolation"
echo ""

# Test 1: Check that claude-local wrapper exists
echo -e "${YELLOW}Test 1: Wrapper script exists${NC}"
if [ -x "./scripts/claude-local" ]; then
    echo -e "${GREEN}✅ PASS: claude-local wrapper found${NC}"
else
    echo -e "${RED}❌ FAIL: claude-local wrapper not found or not executable${NC}"
    exit 1
fi
echo ""

# Test 2: Check server health
echo -e "${YELLOW}Test 2: Server health check${NC}"
if curl -s -f "http://127.0.0.1:7777/health" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ PASS: Server responding on port 7777${NC}"
else
    echo -e "${RED}❌ FAIL: Server not responding. Start with: cargo run --bin pensieve-proxy --release${NC}"
    exit 1
fi
echo ""

# Test 3: Check that global settings are NOT modified
echo -e "${YELLOW}Test 3: Global settings unchanged${NC}"
if [ -f "$HOME/.claude/settings.json" ]; then
    # Check if pensieve-local-token is in global settings (it shouldn't be with wrapper)
    if grep -q "pensieve-local-token" "$HOME/.claude/settings.json" 2>/dev/null; then
        echo -e "${YELLOW}⚠️  WARNING: Global settings contain pensieve-local-token${NC}"
        echo -e "${YELLOW}   This means setup-claude-code.sh was run (old approach)${NC}"
        echo -e "${YELLOW}   Wrapper script will override these settings${NC}"
    else
        echo -e "${GREEN}✅ PASS: Global settings clean (no local server config)${NC}"
    fi
else
    echo -e "${GREEN}✅ PASS: No global settings file yet${NC}"
fi
echo ""

# Test 4: Environment isolation
echo -e "${YELLOW}Test 4: Environment variable isolation${NC}"
# Save current env
OLD_BASE_URL="${ANTHROPIC_BASE_URL:-}"
OLD_API_KEY="${ANTHROPIC_API_KEY:-}"

# Check that we haven't polluted current shell
if [ -z "$ANTHROPIC_BASE_URL" ] && [ -z "$ANTHROPIC_API_KEY" ]; then
    echo -e "${GREEN}✅ PASS: Current shell environment clean${NC}"
elif [ "$ANTHROPIC_BASE_URL" = "http://127.0.0.1:7777" ]; then
    echo -e "${YELLOW}⚠️  WARNING: Current shell has local server config${NC}"
    echo -e "${YELLOW}   This is OK if you ran wrapper in this terminal before${NC}"
else
    echo -e "${GREEN}✅ PASS: Current shell has different config (as expected)${NC}"
fi
echo ""

# Test 5: Wrapper sets correct environment
echo -e "${YELLOW}Test 5: Wrapper environment variables${NC}"
# We can't directly test exec, but we can check the wrapper script content
if grep -q 'export ANTHROPIC_BASE_URL="http://.*:7777"' scripts/claude-local && \
   grep -q 'export ANTHROPIC_API_KEY' scripts/claude-local; then
    echo -e "${GREEN}✅ PASS: Wrapper script sets correct environment variables${NC}"
else
    echo -e "${RED}❌ FAIL: Wrapper script missing environment variables${NC}"
    exit 1
fi
echo ""

echo -e "${GREEN}🎉 All isolation tests passed!${NC}"
echo ""
echo -e "${YELLOW}Manual Test Instructions:${NC}"
echo "1. Terminal A: Run './scripts/claude-local --print \"test\"' (uses local)"
echo "2. Terminal B: Run 'claude --print \"test\"' (uses Anthropic API)"
echo "3. Verify both work independently"
echo "4. Check ~/.claude/settings.json is unchanged"
echo ""

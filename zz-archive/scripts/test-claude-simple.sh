#!/bin/bash
# Simple test script for Claude Code with Pensieve
# Usage: ./scripts/test-claude-simple.sh --print "your prompt"
# Or: ./scripts/test-claude-simple.sh (interactive mode)

export ANTHROPIC_BASE_URL="http://127.0.0.1:7777"
export ANTHROPIC_AUTH_TOKEN="pensieve-local-token"
export API_TIMEOUT_MS="3000000"

echo "🚀 Launching Claude Code with Pensieve..."
echo "📍 Base URL: $ANTHROPIC_BASE_URL"
echo ""

claude "$@"

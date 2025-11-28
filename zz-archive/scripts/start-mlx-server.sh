#!/bin/bash
#
# Start Pensieve MLX Persistent Server
#
# This script starts the FastAPI server that keeps the MLX model loaded in memory.
# Run this BEFORE starting the Pensieve main server or running tests.

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Configuration
DEFAULT_MODEL_PATH="./models/Phi-3-mini-128k-instruct-4bit"
DEFAULT_PORT=8765
DEFAULT_HOST="127.0.0.1"
DEFAULT_MAX_CONCURRENT=2

# Parse arguments
MODEL_PATH="${1:-$DEFAULT_MODEL_PATH}"
PORT="${2:-$DEFAULT_PORT}"
HOST="${3:-$DEFAULT_HOST}"
MAX_CONCURRENT="${4:-$DEFAULT_MAX_CONCURRENT}"

echo -e "${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     Pensieve MLX Persistent Server Launcher          ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo -e "${YELLOW}⚠️  Model not found at: $MODEL_PATH${NC}"
    echo ""
    echo "Please provide the model path as the first argument:"
    echo "  $0 /path/to/model"
    echo ""
    echo "Or ensure the default model is available at:"
    echo "  $DEFAULT_MODEL_PATH"
    exit 1
fi

echo -e "${GREEN}✓ Model found: $MODEL_PATH${NC}"

# Check if Python dependencies are installed
echo ""
echo "Checking Python dependencies..."

if ! python3 -c "import fastapi, uvicorn, mlx" 2>/dev/null; then
    echo -e "${YELLOW}⚠️  Missing Python dependencies${NC}"
    echo ""
    echo "Installing dependencies..."
    pip install -r python_bridge/requirements.txt
    echo ""
fi

echo -e "${GREEN}✓ Dependencies installed${NC}"

# Check if port is already in use
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo ""
    echo -e "${YELLOW}⚠️  Port $PORT is already in use${NC}"
    echo ""
    read -p "Kill existing process and restart? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Killing process on port $PORT..."
        lsof -ti:$PORT | xargs kill -9 2>/dev/null || true
        sleep 1
    else
        echo "Exiting. Please stop the existing server or choose a different port."
        exit 1
    fi
fi

# Start the server
echo ""
echo -e "${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║              Starting MLX Server                      ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Configuration:"
echo "  Model:           $MODEL_PATH"
echo "  Address:         http://$HOST:$PORT"
echo "  Max Concurrent:  $MAX_CONCURRENT"
echo ""
echo -e "${GREEN}Starting server...${NC}"
echo ""

python3 python_bridge/mlx_server.py \
    --model-path "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --max-concurrent "$MAX_CONCURRENT"

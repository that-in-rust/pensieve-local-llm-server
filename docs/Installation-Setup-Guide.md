# Installation Setup Guide

## Executive Summary

This guide provides comprehensive installation and setup instructions for the Pensieve Local LLM Server, covering both development and production deployment scenarios. The installation process is streamlined through automated scripts while providing manual configuration options for advanced users. The server is specifically optimized for macOS Apple Silicon systems and requires specific hardware and software prerequisites.

## Architecture Analysis

### System Requirements Architecture

**Hardware Prerequisites**:
- **Platform**: macOS with Apple Silicon (M1, M2, or M3 chips)
- **Memory**: Minimum 8GB RAM, 16GB+ recommended for optimal performance
- **Storage**: 5GB available space for model and dependencies
- **GPU**: Integrated Apple Neural Engine and GPU cores (automatic)

**Software Prerequisites**:
```bash
# Verify system compatibility
sw_vers                    # macOS version (13.0+ recommended)
uname -m                   # Should return "arm64" for Apple Silicon
python3 --version          # Python 3.8+ required
xcode-select --version     # Xcode Command Line Tools
```

### Installation Architecture

**Installation Methods**:
1. **One-Line Installer**: Automated installation for production use
2. **Development Setup**: Manual installation with development tools
3. **Source Installation**: Build from source for customization
4. **Docker Installation**: Containerized deployment (experimental)

**Directory Structure**:
```
~/.pensieve/                    # Configuration directory
├── config/                     # Configuration files
├── logs/                       # Application logs
├── models/                     # Downloaded models
└── data/                       # Runtime data

/opt/pensieve/                  # Installation directory
├── bin/                        # Executables
├── lib/                        # Libraries
├── scripts/                    # Management scripts
└── server/                     # Server components
```

## Key Components

### One-Line Installation Script

**Automated Installer** (`install.sh`):
```bash
#!/bin/bash
# One-line installer for Pensieve Local LLM Server

set -e

INSTALL_DIR="/opt/pensieve"
CONFIG_DIR="$HOME/.pensieve"
REPO_URL="https://github.com/your-org/pensieve-local-llm-server"

echo "🚀 Installing Pensieve Local LLM Server..."

# 1. System compatibility check
check_system_compatibility() {
    echo "📋 Checking system compatibility..."

    # Check Apple Silicon
    if [[ $(uname -m) != "arm64" ]]; then
        echo "❌ This installation requires Apple Silicon (M1/M2/M3)"
        exit 1
    fi

    # Check macOS version
    macOS_VERSION=$(sw_vers -productVersion)
    echo "✅ macOS version: $macOS_VERSION"

    # Check Python
    if ! command -v python3 &> /dev/null; then
        echo "❌ Python 3 is required but not installed"
        exit 1
    fi

    echo "✅ System compatibility verified"
}

# 2. Install system dependencies
install_system_dependencies() {
    echo "📦 Installing system dependencies..."

    # Install Xcode Command Line Tools
    if ! xcode-select -p &> /dev/null; then
        echo "Installing Xcode Command Line Tools..."
        xcode-select --install
        echo "Please complete the Xcode installation and press Enter to continue..."
        read -r
    fi

    # Install Homebrew if not present
    if ! command -v brew &> /dev/null; then
        echo "Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi

    # Install Rust
    if ! command -v cargo &> /dev/null; then
        echo "Installing Rust..."
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
        source "$HOME/.cargo/env"
    fi

    echo "✅ System dependencies installed"
}

# 3. Clone and build repository
clone_and_build() {
    echo "🔧 Building Pensieve server..."

    # Create installation directory
    sudo mkdir -p "$INSTALL_DIR"
    sudo chown "$USER" "$INSTALL_DIR"

    # Clone repository
    git clone "$REPO_URL" "$INSTALL_DIR/src"
    cd "$INSTALL_DIR/src"

    # Build Rust components
    echo "Building Rust workspace..."
    cargo build --release

    # Install Python dependencies
    echo "Installing Python dependencies..."
    pip3 install -r requirements.txt

    echo "✅ Build completed"
}

# 4. Download model
download_model() {
    echo "🧠 Downloading Phi-3 model..."

    mkdir -p "$CONFIG_DIR/models"

    # Download model (example with progress bar)
    if [ ! -d "$CONFIG_DIR/models/Phi-3-mini-128k-instruct-4bit" ]; then
        ./scripts/download-model.sh "$CONFIG_DIR/models"
    fi

    echo "✅ Model download completed"
}

# 5. Create configuration
create_configuration() {
    echo "⚙️ Creating configuration..."

    mkdir -p "$CONFIG_DIR/config"
    mkdir -p "$CONFIG_DIR/logs"

    # Create default configuration
    cat > "$CONFIG_DIR/config/pensieve.conf" << EOF
# Pensieve Local LLM Server Configuration
[server]
host = "127.0.0.1"
port = 8000
max_concurrent_requests = 4
log_level = "INFO"

[model]
path = "$CONFIG_DIR/models/Phi-3-mini-128k-instruct-4bit"
context_length = 128000
max_tokens = 4096

[performance]
gpu_memory_limit = "6GB"
enable_metal_acceleration = true
batch_size = 1
EOF

    echo "✅ Configuration created"
}

# 6. Setup Claude Code integration
setup_claude_integration() {
    echo "🔗 Setting up Claude Code integration..."

    if command -v claude &> /dev/null; then
        echo "Adding Pensieve integration to Claude Code..."

        # Create integration script
        cat > "$HOME/.pensieve/claude-integration.sh" << EOF
#!/bin/bash
export ANTHROPIC_API_URL="http://localhost:8000"
export ANTHROPIC_API_KEY="local-development"
exec claude "\$@"
EOF

        chmod +x "$HOME/.pensieve/claude-integration.sh"

        echo "Add this alias to your shell profile:"
        echo "alias claude-local='$HOME/.pensieve/claude-integration.sh'"
    else
        echo "Claude Code not found. Install with: npm install -g @anthropic-ai/claude-3-cli"
    fi

    echo "✅ Claude Code integration configured"
}

# 7. Install launcher script
install_launcher() {
    echo "🚀 Installing launcher script..."

    sudo cp "$INSTALL_DIR/src/scripts/pensieve" "/usr/local/bin/"
    sudo chmod +x "/usr/local/bin/pensieve"

    echo "✅ Launcher script installed"
}

# Execute installation
check_system_compatibility
install_system_dependencies
clone_and_build
download_model
create_configuration
setup_claude_integration
install_launcher

echo ""
echo "🎉 Installation completed successfully!"
echo ""
echo "To start the server:"
echo "  pensieve start"
echo ""
echo "To use with Claude Code:"
echo "  export ANTHROPIC_API_URL=http://localhost:8000"
echo "  export ANTHROPIC_API_KEY=local-development"
echo "  claude"
echo ""
echo "For help:"
echo "  pensieve --help"
```

### Development Setup Script

**Development Installation** (`scripts/dev-setup.sh`):
```bash
#!/bin/bash
# Development environment setup

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "🛠️ Setting up development environment..."

# 1. Install development dependencies
install_dev_dependencies() {
    echo "📦 Installing development dependencies..."

    # Python development dependencies
    pip3 install -r requirements.txt
    pip3 install -r requirements-dev.txt

    # Rust development dependencies
    cargo install cargo-watch
    cargo install cargo-tarpaulin
    cargo install cargo-criterion

    # Node.js for tooling (if needed)
    if command -v brew &> /dev/null; then
        brew install node
    fi

    echo "✅ Development dependencies installed"
}

# 2. Setup pre-commit hooks
setup_git_hooks() {
    echo "🔧 Setting up Git hooks..."

    mkdir -p "$PROJECT_ROOT/.git/hooks"

    cat > "$PROJECT_ROOT/.git/hooks/pre-commit" << 'EOF'
#!/bin/bash
# Pre-commit hook for code quality

echo "Running pre-commit checks..."

# Rust formatting and linting
echo "Checking Rust code formatting..."
cargo fmt --all -- --check
cargo clippy -- -D warnings

# Python formatting and linting
echo "Checking Python code formatting..."
black --check src/ tests/
flake8 src/ tests/

# Run tests
echo "Running tests..."
cargo test
pytest tests/

echo "✅ Pre-commit checks passed"
EOF

    chmod +x "$PROJECT_ROOT/.git/hooks/pre-commit"

    echo "✅ Git hooks configured"
}

# 3. Create development configuration
create_dev_config() {
    echo "⚙️ Creating development configuration..."

    cat > "$PROJECT_ROOT/config/dev.toml" << EOF
[server]
host = "127.0.0.1"
port = 8000
log_level = "DEBUG"
auto_reload = true

[model]
path = "./models/Phi-3-mini-128k-instruct-4bit"
context_length = 4096  # Reduced for development

[development]
enable_debug_mode = true
enable_profiling = true
print_request_responses = true
cache_responses = true
EOF

    echo "✅ Development configuration created"
}

# 4. Setup VS Code workspace (optional)
setup_vscode() {
    if command -v code &> /dev/null && [ -d "$PROJECT_ROOT/.vscode" ]; then
        echo "💻 Setting up VS Code workspace..."

        cat > "$PROJECT_ROOT/.vscode/settings.json" << EOF
{
    "rust-analyzer.checkOnSave.command": "clippy",
    "rust-analyzer.cargo.features": "all",
    "python.defaultInterpreterPath": "/usr/bin/python3",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black",
    "files.exclude": {
        "**/target": true,
        "**/__pycache__": true,
        "**/*.pyc": true
    }
}
EOF

        echo "✅ VS Code workspace configured"
    fi
}

# Execute development setup
install_dev_dependencies
setup_git_hooks
create_dev_config
setup_vscode

echo ""
echo "🎉 Development environment ready!"
echo ""
echo "Development commands:"
echo "  cargo watch -x run                    # Run with hot reload"
echo "  cargo test                            # Run Rust tests"
echo "  pytest tests/                        # Run Python tests"
echo "  cargo clippy                          # Lint Rust code"
echo "  black src/ tests/                     # Format Python code"
```

## Integration Points

### Claude Code Integration Setup

**Environment Configuration**:
```bash
# Setup script for Claude Code integration
setup_claude_code_integration() {
    echo "🔗 Configuring Claude Code integration..."

    # Check if Claude Code is installed
    if ! command -v claude &> /dev/null; then
        echo "Installing Claude Code..."
        npm install -g @anthropic-ai/claude-3-cli
    fi

    # Create wrapper script
    cat > "$HOME/.local/bin/claude-local" << 'EOF'
#!/bin/bash

# Pensieve Local LLM Server wrapper for Claude Code
export ANTHROPIC_API_URL="http://localhost:8000"
export ANTHROPIC_API_KEY="local-development"

# Start Pensieve server if not running
if ! pgrep -f "mlx-server\|pensieve" > /dev/null; then
    echo "Starting Pensieve server..."
    pensieve start --daemon
    sleep 5
fi

# Check server health
if ! curl -s http://localhost:8000/health > /dev/null; then
    echo "❌ Pensieve server is not responding"
    exit 1
fi

echo "🤖 Connecting to Claude Code via local Pensieve server..."
exec claude "$@"
EOF

    chmod +x "$HOME/.local/bin/claude-local"

    # Add to shell profile if not already present
    if ! grep -q "claude-local" "$HOME/.zshrc" 2>/dev/null && ! grep -q "claude-local" "$HOME/.bashrc" 2>/dev/null; then
        echo "" >> "$HOME/.zshrc"
        echo "# Claude Code with Pensieve integration" >> "$HOME/.zshrc"
        echo "export PATH=\"\$HOME/.local/bin:\$PATH\"" >> "$HOME/.zshrc"
        echo "alias claude=claude-local" >> "$HOME/.zshrc"
    fi

    echo "✅ Claude Code integration configured"
    echo "Use 'claude-local' or 'claude' to start Claude Code with local LLM"
}
```

### System Service Installation

**macOS LaunchAgent Setup**:
```bash
# Create macOS LaunchAgent for automatic startup
create_launch_agent() {
    echo "🚀 Creating system service..."

    LAUNCH_AGENT_DIR="$HOME/Library/LaunchAgents"
    LAUNCH_AGENT_PLIST="$LAUNCH_AGENT_DIR/com.pensieve.server.plist"

    mkdir -p "$LAUNCH_AGENT_DIR"

    cat > "$LAUNCH_AGENT_PLIST" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.pensieve.server</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/local/bin/pensieve</string>
        <string>start</string>
        <string>--daemon</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>WorkingDirectory</key>
    <string>$INSTALL_DIR/src</string>
    <key>StandardOutPath</key>
    <string>$HOME/.pensieve/logs/server.log</string>
    <key>StandardErrorPath</key>
    <string>$HOME/.pensieve/logs/server.error.log</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>MODEL_PATH</key>
        <string>$CONFIG_DIR/models/Phi-3-mini-128k-instruct-4bit</string>
        <key>RUST_LOG</key>
        <string>info</string>
    </dict>
</dict>
</plist>
EOF

    # Load the LaunchAgent
    launchctl load "$LAUNCH_AGENT_PLIST"

    echo "✅ System service created"
    echo "Pensieve server will start automatically on login"
}
```

## Implementation Details

### Model Download and Setup

**Automated Model Acquisition** (`scripts/download-model.sh`):
```bash
#!/bin/bash
# Download and setup Phi-3 model

set -e

MODEL_DIR="${1:-$HOME/.pensieve/models}"
MODEL_NAME="Phi-3-mini-128k-instruct-4bit"
BASE_URL="https://huggingface.co/microsoft/Phi-3-mini-128k-instruct/resolve/main"

echo "🧠 Downloading Phi-3 model to $MODEL_DIR..."

mkdir -p "$MODEL_DIR/$MODEL_NAME"

# Download model files
model_files=(
    "model-00001-of-00002.safetensors"
    "model-00002-of-00002.safetensors"
    "config.json"
    "tokenizer.json"
    "tokenizer_config.json"
    "special_tokens_map.json"
    "generation_config.json"
)

for file in "${model_files[@]}"; do
    if [ ! -f "$MODEL_DIR/$MODEL_NAME/$file" ]; then
        echo "Downloading $file..."
        curl -L "$BASE_URL/$file" -o "$MODEL_DIR/$MODEL_NAME/$file" --progress-bar
    else
        echo "✅ $file already exists"
    fi
done

# Convert to MLX format if needed
if [ ! -f "$MODEL_DIR/$MODEL_NAME/weights.npz" ]; then
    echo "Converting model to MLX format..."
    python3 -c "
import mlx.core as mx
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    '$MODEL_DIR/$MODEL_NAME',
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained('$MODEL_DIR/$MODEL_NAME')

# Save in MLX format
model.save_pretrained('$MODEL_DIR/$MODEL_NAME-mlx')
tokenizer.save_pretrained('$MODEL_DIR/$MODEL_NAME-mlx')
"
fi

echo "✅ Model setup completed"
echo "Model location: $MODEL_DIR/$MODEL_NAME-mlx"
```

### Configuration Management

**Configuration File Template**:
```toml
# config/pensieve.toml - Default configuration template

[server]
# Network configuration
host = "127.0.0.1"
port = 8000
max_concurrent_requests = 4
request_timeout = 300  # 5 minutes
enable_cors = true

# Logging configuration
log_level = "INFO"
log_file = "~/.pensieve/logs/server.log"
log_rotation = "daily"
max_log_files = 7

[model]
# Model configuration
path = "~/.pensieve/models/Phi-3-mini-128k-instruct-4bit"
context_length = 128000
max_tokens = 4096
temperature = 0.7
top_p = 0.9

[performance]
# Performance tuning
gpu_memory_limit = "6GB"
enable_metal_acceleration = true
batch_size = 1
enable_kv_cache = true
max_cache_size = "1GB"

[security]
# Security settings
enable_authentication = false
api_key = ""
allowed_origins = ["http://localhost:*", "http://127.0.0.1:*"]
rate_limit = 100  # requests per minute

[development]
# Development settings
debug_mode = false
profiling = false
print_requests = false
auto_reload = false
```

**Configuration Validation**:
```python
# config/validator.py
import os
import toml
from pathlib import Path
from typing import Dict, Any

class ConfigValidator:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self.load_config()

    def load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        try:
            with open(self.config_path, 'r') as f:
                return toml.load(f)
        except toml.TomlDecodeError as e:
            raise ValueError(f"Invalid TOML configuration: {e}")

    def validate_model_config(self) -> bool:
        model_config = self.config.get('model', {})

        # Check model path
        model_path = Path(model_config.get('path', '')).expanduser()
        if not model_path.exists():
            raise ValueError(f"Model path does not exist: {model_path}")

        # Check model files
        required_files = ['config.json', 'tokenizer.json', 'weights.npz']
        for file in required_files:
            if not (model_path / file).exists():
                raise ValueError(f"Required model file missing: {file}")

        # Validate numeric parameters
        if model_config.get('context_length', 0) <= 0:
            raise ValueError("context_length must be positive")

        if model_config.get('max_tokens', 0) <= 0:
            raise ValueError("max_tokens must be positive")

        return True

    def validate_server_config(self) -> bool:
        server_config = self.config.get('server', {})

        # Validate port
        port = server_config.get('port', 8000)
        if not (1 <= port <= 65535):
            raise ValueError(f"Invalid port number: {port}")

        # Validate concurrent requests
        max_requests = server_config.get('max_concurrent_requests', 4)
        if max_requests <= 0:
            raise ValueError("max_concurrent_requests must be positive")

        return True

    def validate_all(self) -> bool:
        try:
            self.validate_model_config()
            self.validate_server_config()
            return True
        except Exception as e:
            print(f"Configuration validation failed: {e}")
            return False
```

## Performance Characteristics

### Installation Performance

**Download and Setup Times**:
- **Model Download**: 5-15 minutes depending on internet speed (2.5GB model)
- **Build Time**: 5-10 minutes for Rust workspace compilation
- **Dependency Installation**: 2-5 minutes for Python packages
- **Total Installation**: 15-30 minutes on typical systems

**Resource Usage During Setup**:
- **Peak Memory**: <4GB during compilation
- **Disk Space**: 5GB total (2.5GB model + 2.5GB dependencies)
- **Network Usage**: 2.5GB for model download
- **CPU Usage**: High during Rust compilation

### Post-Installation Performance

**Startup Performance**:
```bash
# Benchmark startup times
time pensieve start

# Expected results:
# Cold start: 3-5 seconds (model loading)
# Warm start: <1 second (model already loaded)
# Memory usage: 2.5-4GB after startup
```

**Runtime Performance**:
- **Memory Baseline**: 2.5GB for model residency
- **Request Latency**: 500-1500ms first token, 50-200ms subsequent tokens
- **Throughput**: 5-20 tokens per second depending on complexity
- **Concurrent Requests**: Linear scaling up to hardware limits

## Testing Strategy

### Installation Validation

**Automated Verification Script** (`scripts/verify-installation.sh`):
```bash
#!/bin/bash
# Verify installation completeness

set -e

echo "🔍 Verifying Pensieve installation..."

# 1. Check system prerequisites
check_prerequisites() {
    echo "Checking system prerequisites..."

    # Check Apple Silicon
    if [[ $(uname -m) != "arm64" ]]; then
        echo "❌ Requires Apple Silicon"
        exit 1
    fi

    # Check Rust
    if ! command -v cargo &> /dev/null; then
        echo "❌ Rust not installed"
        exit 1
    fi

    # Check Python
    if ! command -v python3 &> /dev/null; then
        echo "❌ Python 3 not installed"
        exit 1
    fi

    echo "✅ System prerequisites OK"
}

# 2. Verify installation files
verify_installation_files() {
    echo "Checking installation files..."

    # Check launcher script
    if [ ! -f "/usr/local/bin/pensieve" ]; then
        echo "❌ Launcher script not found"
        exit 1
    fi

    # Check configuration directory
    if [ ! -d "$HOME/.pensieve" ]; then
        echo "❌ Configuration directory not found"
        exit 1
    fi

    echo "✅ Installation files OK"
}

# 3. Verify model download
verify_model() {
    echo "Checking model files..."

    MODEL_DIR="$HOME/.pensieve/models/Phi-3-mini-128k-instruct-4bit"

    if [ ! -d "$MODEL_DIR" ]; then
        echo "❌ Model directory not found"
        exit 1
    fi

    # Check essential model files
    required_files=("config.json" "tokenizer.json")
    for file in "${required_files[@]}"; do
        if [ ! -f "$MODEL_DIR/$file" ]; then
            echo "❌ Model file missing: $file"
            exit 1
        fi
    done

    echo "✅ Model files OK"
}

# 4. Test server functionality
test_server_functionality() {
    echo "Testing server functionality..."

    # Start server in background
    pensieve start --daemon &
    SERVER_PID=$!

    # Wait for server to start
    sleep 10

    # Test health endpoint
    if curl -s http://localhost:8000/health > /dev/null; then
        echo "✅ Server health check OK"
    else
        echo "❌ Server health check failed"
        kill $SERVER_PID
        exit 1
    fi

    # Test API endpoint
    if curl -s -X POST http://localhost:8000/v1/messages \
        -H "Content-Type: application/json" \
        -d '{"model":"claude-3-sonnet","max_tokens":10,"messages":[{"role":"user","content":"Hi"}]}' \
        > /dev/null; then
        echo "✅ API endpoint test OK"
    else
        echo "❌ API endpoint test failed"
        kill $SERVER_PID
        exit 1
    fi

    # Stop server
    kill $SERVER_PID
    echo "✅ Server functionality test OK"
}

# Execute verification
check_prerequisites
verify_installation_files
verify_model
test_server_functionality

echo ""
echo "🎉 Installation verification completed successfully!"
echo "Your Pensieve Local LLM Server is ready to use."
```

### Post-Installation Testing

**Smoke Test Script**:
```python
# tests/smoke_test.py
import requests
import time
import json

def smoke_test():
    """Basic functionality test after installation"""

    print("🧪 Running smoke test...")

    # Test 1: Health check
    try:
        response = requests.get("http://localhost:8000/health", timeout=10)
        assert response.status_code == 200
        print("✅ Health check passed")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

    # Test 2: Simple inference
    try:
        test_request = {
            "model": "claude-3-sonnet",
            "max_tokens": 20,
            "messages": [{"role": "user", "content": "What is 2+2?"}]
        }

        response = requests.post(
            "http://localhost:8000/v1/messages",
            json=test_request,
            timeout=30
        )

        assert response.status_code == 200
        print("✅ Inference test passed")

        # Validate response contains expected content
        response_text = response.text
        assert any(word in response_text.lower() for word in ["4", "four"])
        print("✅ Response validation passed")

    except Exception as e:
        print(f"❌ Inference test failed: {e}")
        return False

    print("🎉 Smoke test completed successfully!")
    return True

if __name__ == "__main__":
    smoke_test()
```

## Development Considerations

### Troubleshooting Common Issues

**Installation Problems**:
```bash
# Rust installation issues
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# Python dependency conflicts
python3 -m venv ~/.pensieve/venv
source ~/.pensieve/venv/bin/activate
pip install -r requirements.txt

# Permission issues
sudo chown -R $USER ~/.pensieve
sudo chmod +x /usr/local/bin/pensieve
```

**Runtime Issues**:
```bash
# Model loading problems
export MODEL_PATH="$HOME/.pensieve/models/Phi-3-mini-128k-instruct-4bit"
pensieve validate-config

# Memory issues
export GPU_MEMORY_LIMIT="4GB"
pensieve start

# Port conflicts
pensieve start --port 8001
export ANTHROPIC_API_URL="http://localhost:8001"
```

### Performance Tuning

**Environment Optimization**:
```bash
# GPU memory optimization
export METAL_DEVICE_WRAPPER_TYPE=1
export GPU_MEMORY_TARGET=6000  # 6GB

# CPU optimization
export OMP_NUM_THREADS=8  # Match CPU core count
export MKL_NUM_THREADS=8

# Python optimization
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
```

This comprehensive installation guide provides multiple installation paths, automated setup scripts, troubleshooting assistance, and performance optimization guidance for users of all technical levels.
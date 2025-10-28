# Pensieve Local LLM Server

**A modular local LLM server with Anthropic API compatibility and authentication**

## 🎯 Current Status: **Foundation Complete, Ready for Real Model Integration**

Pensieve provides a **working HTTP API server** with Anthropic API compatibility, proper authentication, and streaming support. The foundation is solid and ready for integrating real LLM models.

### ✅ **What Actually Works (Verified)**

- ✅ **HTTP API Server**: Starts reliably with proper error handling
- ✅ **Authentication**: Bearer token validation (supports test tokens and Anthropic-style keys)
- ✅ **SSE Streaming**: Real Server-Sent Events with proper headers
- ✅ **Mock Responses**: Realistic responses for testing and development
- ✅ **Health Endpoint**: Basic health monitoring
- ✅ **CLI Interface**: Server start/stop with configuration management
- ✅ **Modular Architecture**: 8-crates with clean separation of concerns

### ⚠️ **What's Next (Not Yet Implemented)**

- ⚠️ **Real LLM Inference**: Currently uses mock responses only
- ⚠️ **Model Loading**: Framework ready, but no real model integration yet
- ⚠️ **GPU Acceleration**: Metal framework integrated but not used for inference

## 🚀 Quick Start (Verified Steps)

### Prerequisites

- **Rust 1.75+** (tested with stable)
- **macOS or Linux** (Apple Silicon recommended)
- **4GB+ RAM** (for mock server operation)

### Installation & Testing

**Step 1: Clone and Build**
```bash
# Clone the repository
git clone https://github.com/that-in-rust/pensieve-local-llm-server
cd pensieve-local-llm-server

# Build the project (should compile with only warnings)
cargo build --workspace
```

**Step 2: Create Dummy Model File**
```bash
# The CLI requires a model file for validation
touch model.gguf
```

**Step 3: Start the Server**
```bash
# Start the server on port 8080
cargo run -p pensieve-01 -- start --model ./model.gguf --host 127.0.0.1 --port 8080
```

Expected output:
```
Starting Pensieve server...
Starting server on 127.0.0.1:8080
Server started successfully on 127.0.0.1:8080
Press Ctrl+C to stop the server
```

**Step 4: Test Health Endpoint**
```bash
curl http://127.0.0.1:8080/health
```

Expected response:
```json
{"status":"healthy","timestamp":"2024-01-01T00:00:00Z"}
```

**Step 5: Test Authentication**
```bash
# This should fail with 401 (no authentication)
curl -X POST http://127.0.0.1:8080/v1/messages \
  -H "Content-Type: application/json" \
  -d '{"model": "claude-3-sonnet-20240229", "max_tokens": 10, "messages": [{"role":"user","content":[{"type":"text","text":"Hello"}]}]}'
```

Expected error:
```
Missing request header "authorization"
```

**Step 6: Test API with Authentication**
```bash
# Use valid test token
curl -X POST http://127.0.0.1:8080/v1/messages \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer test-api-key-12345" \
  -d '{"model": "claude-3-sonnet-20240229", "max_tokens": 50, "messages": [{"role":"user","content":[{"type":"text","text":"Hello Pensieve"}]}]}'
```

Expected response:
```json
{
  "id": "...",
  "type": "message",
  "role": "assistant",
  "content": [{"type": "text", "text": "Mock response to: Hello Pensieve"}],
  "model": "claude-3-sonnet-20240229",
  "stop_reason": "end_turn",
  "stop_sequence": null,
  "usage": {"input_tokens": 10, "output_tokens": 5}
}
```

**Step 7: Test Streaming**
```bash
# Test streaming with authentication
curl -N -X POST http://127.0.0.1:8080/v1/messages \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-ant-api-test123" \
  -H "x-stream: true" \
  -d '{"model": "claude-3-sonnet-20240229", "max_tokens": 10, "messages": [{"role":"user","content":[{"type":"text","text":"Stream test"}]}]}'
```

Expected streaming output:
```
data: {"type": "message_start"}
data: {"type": "content_block_delta", "delta": {"text": "Mock"}}
data: {"type": "message_stop"}
```

## 📋 API Reference

### Authentication

The server requires Bearer token authentication via the `Authorization` header:

**Supported Tokens:**
- `test-api-key-12345` (for development/testing)
- `sk-ant-api-*` (Anthropic-style API keys)

### Endpoints

#### `POST /v1/messages` (Non-Streaming)

**Request:**
```bash
curl -X POST http://127.0.0.1:8080/v1/messages \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer test-api-key-12345" \
  -d '{
    "model": "claude-3-sonnet-20240229",
    "max_tokens": 100,
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "Hello, world!"
          }
        ]
      }
    ]
  }'
```

#### `POST /v1/messages` (Streaming)

**Request:**
```bash
curl -N -X POST http://127.0.0.1:8080/v1/messages \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer test-api-key-12345" \
  -H "x-stream: true" \
  -d '{
    "model": "claude-3-sonnet-20240229",
    "max_tokens": 50,
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "Stream this response"
          }
        ]
      }
    ]
  }'
```

#### `GET /health`

**Request:**
```bash
curl http://127.0.0.1:8080/health
```

## 🏗️ Architecture

Pensieve follows a **layered architecture** with 8 independent crates:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   pensieve-01   │    │   pensieve-02   │    │   pensieve-03   │
│     CLI Layer   │◄──►│  HTTP Server   │◄──►│  API Models     │
│                 │    │                 │    │                 │
│ • Config Mgmt   │    │ • Auth Headers │    │ • Anthropic API  │
│ • Commands      │    │ • Request Routing│    │ • JSON Serde     │
│ • Lifecycle     │    │ • Streaming     │    │ • Validation    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                │
                    ┌─────────────────┐
                    │   pensieve-04   │
                    │ Inference Engine│
                    │                 │
                    │ • Candle ML     │
                    │ • Mock Handler  │
                    │ • Streaming     │
                    │ • Performance   │
                    └─────────────────┘
                                │
                    ┌─────────────────┐
                    │   pensieve-05   │
                    │  Model Support  │
                    │                 │
                    │ • GGUF Format   │
                    │ • Data Models   │
                    │ • Validation    │
                    └─────────────────┘
                                │
                    ┌─────────────────┐
                    │   pensieve-06   │
                    │  Metal Support  │
                    │                 │
                    │ • GPU Framework │
                    │ • Device Mgmt   │
                    │ • Acceleration  │
                    └─────────────────┘
                                │
                    ┌─────────────────┐
                    │   pensieve-07   │
                    │ Core Foundation │
                    │                 │
                    │ • Traits        │
                    │ • Error Types   │
                    │ • Resources     │
                    └─────────────────┘
                                │
                    ┌─────────────────┐
                    │pensieve-08_claude│
                    │ Claude Core     │
                    │                 │
                    │ • Claude Types  │
                    │ • Integration   │
                    └─────────────────┘
```

### Dependency Layers

- **L1 (Core)**: `pensieve-07` - Foundation traits and error types
- **L2 (Engine)**: `pensieve-04`, `pensieve-05`, `pensieve-06` - Core functionality
- **L3 (Application)**: `pensieve-01`, `pensieve-02`, `pensieve-03` - User-facing features

## 🛠️ Development

### Building

```bash
# Development build
cargo build

# Release build (optimized)
cargo build --release

# Run specific crate tests
cargo test -p pensieve-01
cargo test -p pensieve-02
```

### CLI Commands

```bash
# Start server with custom configuration
cargo run -p pensieve-01 -- start --model ./model.gguf --host 127.0.0.1 --port 8080

# Show help
cargo run -p pensieve-01 -- --help

# Validate configuration
cargo run -p pensieve-01 -- validate --config config.json
```

## 🧪 Testing

### Current Test Status

The workspace compiles with warnings but has some test compilation issues. The core functionality is verified through manual testing.

```bash
# Build check (works)
cargo check --workspace

# Manual testing recommended (see Quick Start section)
```

## 🚧 Next Steps (Roadmap)

### ✅ Completed

- ✅ **Foundation Architecture**: 8-crates with clean dependency hierarchy
- ✅ **HTTP API Server**: Complete with authentication and streaming
- ✅ **Authentication System**: Bearer token validation
- ✅ **Mock Responses**: Realistic behavior for testing
- ✅ **SSE Streaming**: Proper Server-Sent Events implementation
- ✅ **CLI Interface**: Server management and configuration
- ✅ **Error Handling**: Comprehensive error responses

### 🔄 In Progress

- 🔄 **Test Suite**: Fixing compilation errors in test modules
- 🔄 **Documentation**: Improving accuracy and completeness

### 🎯 Future Development

- **Real Model Integration**: Connect to actual GGUF models using Candle
- **Performance Optimization**: GPU acceleration and memory management
- **Advanced Features**: Model switching, configuration management
- **Production Tools**: Monitoring, metrics, deployment guides

## 🤝 Contributing

This project follows **TDD-first principles**:

1. **Write tests first** (when adding new features)
2. **Implement minimal working solution**
3. **Refactor and optimize**
4. **Test manually** (until test suite is fixed)
5. **Update documentation**

### Development Workflow

```bash
# 1. Create feature branch
git checkout -b feature-name

# 2. Write failing test (if applicable)
# 3. Implement minimal solution
# 4. Test manually using curl examples
# 5. Refactor and optimize
# 6. Update README with verified functionality
# 7. Submit pull request
```

## 📄 License

MIT OR Apache-2.0

## 🙏 Acknowledgments

- **Candle ML Framework**: For excellent Rust ML infrastructure
- **Anthropic**: For API compatibility standards
- **Apple Silicon Community**: For Metal optimization guidance

---

## 🎯 Summary

Pensieve Local LLM Server provides a **solid foundation** for local LLM development with:

- ✅ **Working HTTP API** with full Anthropic compatibility
- ✅ **Proper authentication** for secure access
- ✅ **Streaming support** for real-time responses
- ✅ **Modular architecture** for maintainability
- ✅ **Mock responses** for development and testing

The server is **ready for real model integration** and can serve as a foundation for building complete local AI development tools.

---

**Current Status: Foundation Complete, Ready for Model Integration**
**Last Updated: October 28, 2025**
**Version: 0.1.0**
# Pensieve High-Level Design (HLD)

**Document Version**: 1.0
**Last Updated**: October 29, 2025
**Current Status**: Foundation Complete, MLX Integration Ready

## Executive Summary

The Pensieve Local LLM Server provides a **complete modular foundation** with HTTP API server, authentication, and streaming capabilities. The system has a **working 8-crate architecture** that is ready for MLX integration. Currently using mock responses for development and testing while the MLX framework integration is planned.

### Current Architecture Status
- **Foundation Complete**: All 8 crates compiled and functional
- **API Server Working**: HTTP server with authentication and streaming
- **CLI Interface**: Basic commands implemented (start, stop, status, config, validate)
- **Mock Responses**: Development and testing functionality in place

### Target Architecture Vision (To Be Implemented)
- **MLX Integration**: Real model inference with Apple Silicon optimization
- **Performance Target**: 25-40 TPS with Metal acceleration
- **Production Ready**: Complete local LLM server with automatic model management

## System Overview

### Architecture Principles
1. **Modular Design**: 8-crate architecture with clean separation of concerns
2. **Anthropic Compatibility**: Native API compatibility for drop-in Claude Code integration
3. **Apple Silicon Focus**: MLX framework optimization for M1/M2/M3 hardware
4. **Performance First**: Optimized for local inference speed and memory efficiency
5. **Extensible Foundation**: Clean interfaces ready for future enhancements

## Component Relationships

### System Architecture Diagram
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
                    │ • Mock Handler  │
                    │ • MLX Ready     │
                    │ • Streaming     │
                    │ • Performance   │
                    └─────────────────�
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

### Data Flow Overview
1. **Request Flow**: CLI → HTTP Server → Inference Engine → Response
2. **Authentication**: Bearer token validation at HTTP layer
3. **Model Processing**: Request parsing → Inference → Response formatting
4. **Streaming**: Real-time response streaming via Server-Sent Events

## Key Architectural Decisions

### **1. Why 8 Crates?**

**Decision**: Modular 8-cate architecture instead of monolithic design

**Reasoning**:
- **Separation of Concerns**: Each crate has a single, well-defined responsibility
- **Independent Testing**: Components can be tested in isolation
- **Future Extensibility**: Easy to add new features without affecting existing code
- **Team Development**: Multiple developers can work on different crates simultaneously

**Evidence**: Foundation compiles and functions correctly with clean interfaces

### **2. Why MLX Framework?**

**Decision**: Apple's official MLX framework for Apple Silicon

**Reasoning**:
- **Performance Superiority**: MLX delivers 25-40 TPS vs 15-30 TPS with alternatives
- **Metal Integration**: Direct access to Apple Metal for GPU acceleration
- **Memory Efficiency**: Optimized for Apple Silicon memory architecture
- **Future-Proofing**: Official Apple support ensures continued development and optimization
- **Ecosystem Maturity**: Growing community and comprehensive documentation

**Risk Assessment**: Limited to Apple Silicon ecosystem, but this aligns with target user base

### **3. Why Anthropic API Compatibility?**

**Decision**: Implement Anthropic API specification instead of custom API

**Reasoning**:
- **Drop-in Compatibility**: Works seamlessly with Claude Code and existing tools
- **User Familiarity**: Developers already know Anthropic API patterns
- **Tool Integration**: Immediate compatibility with development workflows
- **Market Standard**: Established format with extensive community support
- **Documentation**: Comprehensive examples and community resources

**Risk Assessment**: Dependency on external standard, but widely adopted in the industry

### **4. Why Phi-3 Mini 4-bit?**

**Decision**: Phi-3-mini-128k-instruct-4bit as default model

**Reasoning**:
- **Memory Efficiency**: ~1.5GB memory usage fits comfortably in 16GB+ systems
- **Context Capability**: 128K token context window for complex conversations
- **Performance**: Strong reasoning capabilities with fast inference speeds
- **Availability**: Readily available on Hugging Face with MLX community support
- **Quality**: Proven performance in instruction-following and reasoning tasks

**Alternative Considered**: Larger models (7B+ parameters) - rejected due to memory constraints

### **5. Why Port 7777?**

**Decision**: Fixed port 7777 for magical user experience

**Reasoning**:
- **Memorability**: Easy to remember magical number
- **Theme Alignment**: Perfect fit with Pensieve magical concept
- **Uniqueness**: Distinct from common ports (8080, 3000, etc.)
- **User Experience**: Adds magical aura to local AI interactions

## Integration Points

### **External Integrations**
1. **Claude Code**: Via Anthropic API compatibility
2. **Hugging Face**: For model downloading and community models
3. **Apple MLX**: For inference engine capabilities
4. **Apple Metal**: For GPU acceleration

### **Internal Interfaces**
1. **CLI ↔ HTTP Server**: Command execution and status reporting
2. **HTTP Server ↔ Inference Engine**: Request processing and response generation
3. **Inference Engine ↔ Model Support**: Model loading and management
4. **All Components ↔ Core Foundation**: Shared traits and error handling

### **Data Flow Integration**
1. **Request Processing**: HTTP → API Models → Inference
2. **Authentication**: Bearer token validation → Authorization
3. **Model Management**: Hugging Face → Local Cache → MLX Loading
4. **Response Generation**: MLX Inference → API Formatting → HTTP Response

## Performance Considerations

### **Memory Management**
- **Model Size**: Target <2GB for 4-bit Phi-3 Mini
- **System Overhead**: HTTP server and Rust runtime ~500MB
- **Total Target**: <2.5GB on 16GB systems
- **Cache Strategy**: Model persistence between sessions

### **Performance Targets**
- **First Token**: <300ms latency
- **Throughput**: 25-40 tokens per second
- **Concurrency**: Support for multiple simultaneous requests
- **Memory Efficiency**: Optimize for 16GB+ systems

### **Optimization Strategies**
- **Metal Acceleration**: Full GPU utilization for inference
- **Batch Processing**: Efficient token generation
- **Memory Pooling**: Reduce allocation overhead
- **Caching**: Model and response caching where appropriate

## Security Considerations

### **Authentication (Local Development Model)**
- **Bearer Token**: Fixed token "pensieve-local-key" for local development
- **Localhost Security**: Designed for single-user localhost environment
- **Simplicity Focus**: Zero configuration for development workflow
- **Future Extension**: Environment variable support for production scenarios

### **Local Privacy**
- **Data Isolation**: All processing happens locally
- **No Cloud Dependencies**: No external API calls for inference
- **User Control**: Complete control over model and data

### **Network Security (Local-First Design)**
- **Localhost by Default**: Server runs on 127.0.0.1:7777 for local security
- **Single User Model**: Designed for individual developer workstations
- **Optional Remote**: Future configuration for network access if needed
- **TLS Support**: HTTPS encryption for remote access scenarios

## Deployment Architecture

### **Single Binary Deployment**
- **Rust Compilation**: Static linking for easy distribution
- **No External Dependencies**: Self-contained deployment package
- **Cross-Platform**: Support for macOS with Apple Silicon

### **Configuration Management**
- **Default Settings**: Sensible defaults for immediate use
- **Environment Variables**: Override options for customization
- **Configuration Files**: JSON-based configuration for advanced users

### **Monitoring and Logging**
- **Structured Logging**: Comprehensive logging for debugging
- **Health Endpoints**: Status monitoring and health checks
- **Performance Metrics**: Request timing and throughput monitoring

---

**Next Steps**: Proceed to Arch02-LLD.md for detailed implementation specifications
**Dependencies**: All 8 crates compiled and tested, MLX research complete
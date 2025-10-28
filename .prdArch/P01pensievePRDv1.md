# User Journey for Pensieve Local LLM Server

## Section 0: Complete User Journey Visualization

```mermaid
---
title: Pensieve Local LLM Server - Complete User Journey
accTitle: Pensieve User Journey Flowchart
accDescr: "A comprehensive flowchart showing the complete user journey from installation to Claude Code integration with the Pensieve Local LLM Server, including automatic model download, server startup, and configuration steps."
config:
  flowchart:
    defaultRenderer: "elk"
  theme: "base"
  themeVariables:
    primaryColor: "#ECECFF"
    primaryTextColor: "#363636"
    primaryBorderColor: "#363636"
    lineColor: "#363636"
    secondaryColor: "#f8f8ff"
    tertiaryColor: "#f0f7ff"
    fontFamily: "system-ui, -apple-system, sans-serif"
    fontSize: "14px"
    nodeSpacing: 75
    rankSpacing: 75
    wrappingWidth: 150
---
flowchart TD
    Start([User Starts Journey]) --> Decision1{Has<br/>Apple Silicon?}

    Decision1 -- No --> Error1[⚠️ Unsupported Platform<br/>MLX requires Apple Silicon<br/>M1/M2/M3 Macs]
    Error1 --> End[❌ Journey Ends]

    Decision1 -- Yes --> Step1[📦 Install Dependencies<br/>• MLX Framework<br/>• Python 3.9+<br/>• Rust Toolchain]

    Step1 --> Step2[🚀 Single Command Setup<br/><code>cargo run -p pensieve-01 -- start</code>]

    Step2 --> AutoModel[📥 Auto-Download Model<br/>Phi-3-mini-128k-instruct-4bit<br/>from Hugging Face]

    AutoModel --> AutoServer[🔧 Auto-Configure Server<br/>• MLX Acceleration<br/>• Metal GPU Support<br/>• Port 7777]

    AutoServer --> Step3[✅ Server Running<br/>🌐 http://127.0.0.1:7777<br/>⚡ MLX + Metal Acceleration]

    Step3 --> Step4[⚙️ Configure Claude Code<br/><code>export ANTHROPIC_BASE_URL=http://127.0.0.1:7777</code><br/><code>export ANTHROPIC_API_KEY=pensieve-local-key</code>]

    Step4 --> Step5[🎯 Launch Claude Code<br/>• Full 128k Context<br/>• 25-40 tokens/second<br/>• Local Privacy]

    Step5 --> Success[🎉 Success!<br/>Magical Local LLM Experience<br/>Identical to Cloud Claude]

    Success --> End

    %% Styling for different node types
    classDef startNode fill:#90EE90,stroke:#2E8B57,stroke-width:3px
    classDef processNode fill:#ECECFF,stroke:#363636,stroke-width:2px
    classDef decisionNode fill:#FFF8DC,stroke:#DAA520,stroke-width:2px
    classDef successNode fill:#98FB98,stroke:#228B22,stroke-width:2px
    classDef errorNode fill:#FFB6C1,stroke:#DC143C,stroke-width:2px

    class Start startNode
    class Step1,Step2,Step3,Step4,Step5,AutoModel,AutoServer processNode
    class Decision1 decisionNode
    class Success successNode
    class Error1,End errorNode
```

## Architecture Decision: MLX-Powered

**Framework**: Apple MLX (optimized for Apple Silicon)
**Default Model**: mlx-community/Phi-3-mini-128k-instruct-4bit
**Target Platform**: macOS with Apple Silicon (M1/M2/M3)

## Simplified User Journey

### One-Command Setup Experience

1. **User runs the following command**:
    ```bash
    cargo run -p pensieve-01 -- start
    ```
    - Pensieve automatically downloads and configures Phi-3-mini-128k-instruct-4bit model
    - Server starts on default port 7777 with MLX acceleration
    - Zero manual configuration required

2. **User configures Claude Code**:
    ```bash
    export ANTHROPIC_BASE_URL=http://127.0.0.1:7777
    export ANTHROPIC_API_KEY=pensieve-local-key
    ```
    - Opens Claude Code with local LLM backend
    - Experience identical to cloud-based Claude
    - Full 128k context window support
    - Fast generation speeds on Apple Silicon

## Key Benefits

- **Instant Setup**: No model downloading or manual configuration
- **Optimized Performance**: MLX provides superior Apple Silicon acceleration
- **Memory Efficient**: 4-bit quantization allows running on 16GB+ Macs
- **Transparent Integration**: Claude Code works unchanged
- **Local Privacy**: All processing happens locally on device

## Technical Implementation

- **MLX Framework**: Apple's machine learning framework for Silicon
- **Automatic Model Management**: Built-in Hugging Face integration
- **Memory Optimization**: Smart KV cache management for large context
- **Metal Acceleration**: Full GPU utilization for fast inference
- **Rust + Python/C++**: Mixed architecture for optimal performance
# Do Not Commit Directory (.doNotCommit/)

This directory contains files that should not be committed to version control, but are tracked by git to enable Claude conversation storage and local development.

## Directory Structure

```
.doNotCommit/
├── README.md                    # This file
├── .refGitHubRepo/             # Reference repositories (gitignored)
│   ├── repo1/                  # Cloned reference repositories
│   ├── repo2/                  # Additional reference repos
│   └── ...                     # More reference repositories
├── conversations/              # Claude conversation history
├── local-config/               # Local development configuration
├── experiments/                # Experimental code and prototypes
└── temp/                       # Temporary working files
```

## Purpose

### .refGitHubRepo/ (Gitignored)
This subdirectory contains cloned reference repositories that are useful for research and development but should not be committed:

- Reference implementations of LLM servers
- Open-source model libraries
- Benchmark repositories
- Example projects and tutorials

**Note:** This directory is gitignored to avoid committing large binary files and external repositories.

### Tracked Files
The parent `.doNotCommit/` directory itself is tracked by git to enable:

- **Claude Conversation Storage**: Allow Claude to store conversation history
- **Local Development**: Keep development-specific files organized
- **Team Coordination**: Share local configurations while ignoring large binaries

## Usage Guidelines

### Reference Repositories (.refGitHubRepo/)
```bash
# Clone reference repositories for research
git clone https://github.com/example/llm-server.git .doNotCommit/.refGitHubRepo/llm-server
git clone https://github.com/example/model-library.git .doNotCommit/.refGitHubRepo/model-library
```

### Recommended Reference Repositories for LLM Server Development

#### Core LLM Libraries
- `ggerganov/llama.cpp` - C++ implementation for reference
- `huggingface/candle` - Rust ML framework (used in this project)
- `mozilla/sccache` - Compiler cache for faster builds

#### Reference Servers
- `ollama/ollama` - Popular LLM server implementation
- `lm-sys/FastChat` - Open serving system for LLMs
- `vllm-project/vllm` - High-throughput LLM inference engine

#### Model Repositories
- `huggingface/transformers` - Model implementations
- `ggerganov/ggml` - Model format and quantization
- Various model-specific repositories for format research

### Agent Integration with Reference Repositories

When using the `agent-explore-code` tool with local reference repositories:

1. **Clone Repository**: Clone the reference repo to `.doNotCommit/.refGitHubRepo/`
2. **Explore Code**: Use agent-explore-code to understand implementation patterns
3. **Extract Insights**: Document findings in `.domainDocs/`
4. **Apply Learnings**: Implement insights in the main project

Example workflow:
```bash
# Clone reference implementation
git clone https://github.com/ollama/ollama.git .doNotCommit/.refGitHubRepo/ollama

# Explore with agent-explore-code
agent-explore-code .doNotCommit/.refGitHubRepo/ollama --focus server-architecture
```

## File Management

### Claude Conversations
- Conversation history may be stored here for continuity
- Use descriptive filenames for conversation topics
- Clean up regularly to maintain organization

### Local Configuration
- Development-specific configuration files
- Environment variable templates
- Local build scripts
- Development tool configurations

### Experimental Code
- Prototype implementations
- Performance experiments
- Proof of concepts
- Sandbox code for testing ideas

## Cleanup and Maintenance

- Regular cleanup of outdated reference repositories
- Remove large binary files that might accidentally get committed
- Update reference repositories periodically with `git pull`
- Archive old experimental code when no longer needed

## Security Considerations

- Never commit API keys or secrets to this directory
- Be careful with sensitive configuration files
- Review contents before sharing with team members
- Use appropriate file permissions for sensitive data

---

**Last Updated:** [Date]
**Maintainer:** [Name]
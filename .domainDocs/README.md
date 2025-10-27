# Domain Documentation Directory (.domainDocs/)

This directory contains research documentation and knowledge about the domains relevant to the Pensieve local LLM server project.

## Purpose

The `.domainDocs/` folder is designed to store domain-specific research, technical documentation, and reference materials that inform the development of the local LLM server. This includes research on:

- Large Language Model architectures and implementations
- Model formats and optimization techniques
- Hardware acceleration and performance optimization
- Local LLM server best practices
- Related open-source projects and benchmarks

## File Organization

Files in this directory follow a sequential naming convention: `DXX*.md` where XX represents a two-digit number.

```
.domainDocs/
├── README.md                    # This file
├── D01-llm-architectures.md     # Research on LLM architectures
├── D02-model-formats.md         # Documentation on model file formats
├── D03-hardware-optimization.md # Hardware acceleration research
├── D04-api-design-patterns.md   # API design best practices
└── DXX-...                      # Additional domain research files
```

## Suggested Research Topics

### Core LLM Research
- Transformer architectures and variants
- Model quantization and compression
- Inference optimization techniques
- Memory management for large models

### Model Formats
- GGUF (GPT-Generated Unified Format)
- SafeTensors
- Hugging Face model formats
- Custom format considerations

### Hardware Acceleration
- CUDA backend implementation
- Apple Metal Performance Shaders
- CPU optimization (AVX, SIMD)
- Memory hierarchy optimization

### Server Architecture
- REST API design patterns
- WebSocket streaming protocols
- Load balancing and scaling
- Security considerations

## Usage Guidelines

1. **Sequential Numbering**: Use the next available number when creating new research documents
2. **Descriptive Titles**: Include both the number and a clear description in the filename
3. **Cross-References**: Link between related documents using relative paths
4. **Regular Updates**: Keep documentation current with the latest research
5. **Source Attribution**: Always cite sources and references for external research

## Integration with Development

- Use these documents to inform architecture decisions
- Reference domain knowledge during implementation
- Update documentation as new research emerges
- Share findings with the development team

## Tools and Workflow

- **Research**: Use WebSearch tool to gather latest information
- **Documentation**: Edit files directly or use external editors
- **References**: Store supporting materials in `.doNotCommit/.refGitHubRepo/`
- **Collaboration**: Share findings via team discussions and reviews

---

**Last Updated:** [Date]
**Maintainer:** [Name]
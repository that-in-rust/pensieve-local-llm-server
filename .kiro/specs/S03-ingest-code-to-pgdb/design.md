# Design Document

## Overview

The S03-ingest-code-to-pgdb system is a Rust-based code ingestion and analysis platform that transforms GitHub repositories and local codebases into searchable PostgreSQL databases. The system enables developers to systematically analyze large codebases using structured LLM workflows through IDE integration.

## Architecture

### High-Level Architecture

```mermaid
graph TB
    subgraph "Input Sources"
        A[GitHub Repository URL]
        B[Local Folder Path]
    end
    
    subgraph "Core System"
        C[CLI Interface]
        D[Ingestion Engine]
        E[File Processor]
        F[Query Engine]
        G[Task Generator]
    end
    
    subgraph "Storage Layer"
        H[(PostgreSQL Database)]
        I[INGEST_* Tables]
        J[QUERYRESULT_* Tables]
        K[ingestion_meta Table]
    end
    
    subgraph "Output Layer"
        L[Temporary Files]
        M[Task Markdown Files]
        N[Individual MD Files]
    end
    
    subgraph "IDE Integration"
        O[Kiro IDE]
        P[Task Execution]
        Q[LLM Analysis]
    end
    
    A --> C
    B --> C
    C --> D
    D --> E
    E --> H
    H --> I
    H --> K
    C --> F
    F --> H
    F --> L
    C --> G
    G --> M
    H --> J
    L --> O
    M --> O
    O --> P
    P --> Q
    Q --> N
    N --> C
# Code Ingest: High-Performance Rust Ingestion Engine

**Transform any codebase into queryable PostgreSQL intelligence in seconds.**

Code Ingest is a production-ready Rust tool that ingests GitHub repositories, local folders, and documents into PostgreSQL databases, enabling systematic code analysis through SQL queries, full-text search, and structured data exploration.

## Core Value Proposition

**Problem**: Analyzing large codebases manually is time-consuming and inconsistent.  
**Solution**: Automated ingestion + PostgreSQL storage + structured task generation = systematic code intelligence.  
**Result**: Complete workflow from raw code to ready-to-execute analysis tasks with A/B/C content files and structured task lists.

## Architecture Overview

```mermaid
%%{init: {
  "theme": "base",
  "themeVariables": {
    "primaryColor": "#F5F5F5",
    "secondaryColor": "#E0E0E0",
    "lineColor": "#616161",
    "textColor": "#212121",
    "fontSize": "16px",
    "fontFamily": "Helvetica, Arial, sans-serif"
  },
  "flowchart": {
    "nodeSpacing": 70,
    "rankSpacing": 80,
    "wrappingWidth": 160,
    "curve": "basis"
  },
  "useMaxWidth": false
}}%%

flowchart TD
    subgraph "Input Sources"
        A[GitHub Repository]
        B[Local Folder]
    end
    
    subgraph "Processing Engine"
        C[Git Cloner]
        D[File Classifier]
        E[Content Processor]
        F[Multi-Scale Context]
    end
    
    subgraph "Storage Layer"
        G[(PostgreSQL Database)]
        H[Timestamped Tables]
        I[Multi-Scale Windows]
    end
    
    subgraph "Query Layer"
        J[SQL Interface]
        K[Full-Text Search]
        L[Metadata Queries]
    end
    
    A --> C
    B --> D
    C --> D
    D --> E
    E --> F
    F --> G
    G --> H
    H --> I
    I --> J
    J --> K
    K --> L
```
- **CLI**: Command-line interface with clap
- **Core**: Ingestion engine with async processing
- **Database**: PostgreSQL with sqlx
- **Processing**: Multi-threaded file processing
- **Queries**: SQL interface with full-text search

## License

MIT License - see LICENSE file for details.

---

**Made with ⚡ by the Code Ingest Team**

*Transforming codebases into queryable intelligence, one repository at a time.*
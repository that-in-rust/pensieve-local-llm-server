# Pensieve

**Transform document collections into LLM-ready knowledge bases with intelligent deduplication.**

A high-performance Rust CLI tool that ingests text files, eliminates duplicates at file and paragraph levels, and creates optimized databases for AI processing.

## Quick Start

```bash
# Process documents into database
pensieve --input ~/Documents --database knowledge.db

# View results
pensieve stats --database knowledge.db
```

## Core Value

Pensieve solves the token waste problem in LLM workflows by:
- **Eliminating redundancy**: File and paragraph-level deduplication
- **Maximizing signal**: Only unique content reaches your LLM
- **Preserving provenance**: Track content back to source files

## Architecture Overview

```mermaid
graph TD
    Input[Document Collection] --> Scanner[File Scanner]
    Scanner --> Dedup[Deduplication Engine]
    Dedup --> Extract[Content Extractor]
    Extract --> Database[(SQLite Database)]
    
    subgraph "Processing Phases"
        direction LR
        P1[Scan] --> P2[Dedupe] --> P3[Extract] --> P4[Store]
    end
    
    subgraph "Output Benefits"
        direction TB
        Database --> Benefit1[Unique Content Only]
        Database --> Benefit2[Source Traceability]
        Database --> Benefit3[Token Optimization]
    end
    
    classDef input fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px
    classDef process fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
    classDef output fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    
    class Input input
    class Scanner,Dedup,Extract process
    class Database,Benefit1,Benefit2,Benefit3 output
```

## Processing Workflow

```mermaid
graph TD
    Start[Start] --> Phase1[Phase 1: Metadata Scan]
    Phase1 --> Phase2[Phase 2: File Deduplication]
    Phase2 --> Phase3[Phase 3: Content Extraction]
    Phase3 --> Complete[Complete]
    
    subgraph "Phase 1: Discovery"
        direction TB
        P1A[Recursive Directory Scan]
        P1B[File Type Detection]
        P1C[SHA-256 Hash Calculation]
    end
    
    subgraph "Phase 2: File-Level Dedup"
        direction TB
        P2A[Group by Content Hash]
        P2B[Mark Canonical Files]
        P2C[Store Metadata]
    end
    
    subgraph "Phase 3: Content Processing"
        direction TB
        P3A[Extract Text Content]
        P3B[Split into Paragraphs]
        P3C[Paragraph Deduplication]
        P3D[Store Unique Content]
    end
    
    Phase1 --> P1A
    P1A --> P1B
    P1B --> P1C
    
    Phase2 --> P2A
    P2A --> P2B
    P2B --> P2C
    
    Phase3 --> P3A
    P3A --> P3B
    P3B --> P3C
    P3C --> P3D
    
    classDef phase fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,font-weight:bold
    classDef step fill:#e8f5e8,stroke:#2e7d32,stroke-width:1px
    
    class Phase1,Phase2,Phase3 phase
    class P1A,P1B,P1C,P2A,P2B,P2C,P3A,P3B,P3C,P3D step
```

## Installation

```bash
# Build from source
git clone <repository-url>
cd pensieve
cargo build --release

# Binary available at target/release/pensieve
```

## Usage

### Basic Commands
```bash
# Initialize database
pensieve init --database my.db

# Process documents
pensieve --input /path/to/docs --database my.db

# View statistics
pensieve stats --database my.db

# Check dependencies
pensieve check-deps
```

### Example Output
```
Phase 1: Scanning files... 15,432 files found
Phase 2: Deduplication... 3,421 duplicates (22.2% savings)
Phase 3: Content processing... 95,847 unique paragraphs stored
Phase 4: Complete! 2,847,392 tokens ready for LLM processing
```

## Supported Formats

**Text & Documentation**: `.txt`, `.md`, `.rst`, `.org`, `.adoc`, `.wiki`, `.tex`, `.bib`

**Source Code**: `.rs`, `.py`, `.js`, `.ts`, `.java`, `.go`, `.c`, `.cpp`, `.php`, `.rb`, `.swift`

**Web & Markup**: `.html`, `.css`, `.xml`

**Configuration**: `.json`, `.yaml`, `.toml`, `.ini`, `.cfg`, `.env`

**Documents**: `.pdf`, `.docx` (basic text extraction)

**Data**: `.csv`, `.log`, `.sql`

## Database Schema

```mermaid
graph TD
    subgraph "Core Tables"
        direction TB
        Files[files<br/>- metadata<br/>- dedup status<br/>- token counts]
        Paragraphs[paragraphs<br/>- unique content<br/>- content hash<br/>- token estimates]
        Sources[paragraph_sources<br/>- file relationships<br/>- provenance tracking]
    end
    
    subgraph "Support Tables"
        direction TB
        Errors[processing_errors<br/>- error logging<br/>- debugging info]
        Schema[schema_version<br/>- migration tracking]
    end
    
    Files --> Paragraphs
    Files --> Sources
    Paragraphs --> Sources
    Files --> Errors
    
    classDef core fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef support fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
    
    class Files,Paragraphs,Sources core
    class Errors,Schema support
```

## Performance

- **File Scanning**: 10,000+ files/sec
- **Hash Calculation**: 500+ MB/sec  
- **Database Writes**: 50,000+ records/sec
- **Memory Usage**: <16GB for 500GB corpus
- **Scalability**: Tested with 500GB+ collections

## Configuration

Generate default config:
```bash
pensieve config --output pensieve.toml
```

Key settings:
- Thread count for parallel processing
- Batch sizes for database operations
- File type inclusion/exclusion rules
- Deduplication parameters

## Troubleshooting

**Database locked**: Ensure no other Pensieve processes running
**Permission denied**: Check file/directory permissions
**Slow processing**: Increase thread count, use SSD storage
**High memory**: Reduce batch sizes, process smaller chunks

## Development

```bash
# Run tests
cargo test

# Build debug version
cargo build

# Run with sample data
cargo run -- --input test_data --database test.db --verbose
```

### Project Structure
- `src/cli.rs` - Command-line interface
- `src/scanner.rs` - File system scanning
- `src/extractor.rs` - Content extraction
- `src/database.rs` - Database operations
- `src/deduplication.rs` - Duplicate detection

---

**Pensieve** - Efficiently transform document collections into LLM-ready knowledge bases with intelligent deduplication and optimized token usage.
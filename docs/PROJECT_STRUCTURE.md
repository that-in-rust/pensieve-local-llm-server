# Project Structure

## 📁 Repository Organization

```
pensieve/
├── 📂 code-ingest/                    # Main Rust application
│   ├── 📂 src/                        # Source code
│   │   ├── 📂 cli/                    # Command-line interface
│   │   ├── 📂 database/               # Database operations
│   │   ├── 📂 ingestion/              # Code ingestion engine
│   │   ├── 📂 processing/             # File processing
│   │   ├── 📂 tasks/                  # Task generation system
│   │   │   ├── 📄 simple_task_generator.rs      # Kiro-compatible task generator
│   │   │   ├── 📄 windowed_task_manager.rs      # Large task volume management
│   │   │   └── 📄 ...                 # Other task modules
│   │   └── 📄 main.rs                 # Application entry point
│   ├── 📄 Cargo.toml                  # Rust dependencies
│   └── 📄 README.md                   # Code-ingest specific docs
├── 📂 .kiro/                          # Kiro IDE configuration
│   ├── 📂 docs/                       # Reference documentation
│   ├── 📂 scripts/                    # Utility scripts
│   ├── 📂 specs/                      # Task specifications
│   ├── 📂 steering/                   # Steering rules
│   └── 📄 spec-S04-steering-doc-analysis.md    # Main analysis prompt
├── 📂 docs/                           # Project documentation
│   ├── 📂 analysis/                   # Technical analysis documents
│   ├── 📂 testing/                    # Test files and validation
│   └── 📄 README.md                   # Documentation index
├── 📂 examples/                       # Usage examples
├── 📂 gringotts/                      # Output workspace
├── 📂 scripts/                        # Build and utility scripts
├── 📄 README.md                       # Main project README
├── 📄 READMELongForm20250929.md       # Comprehensive documentation
└── 📄 Cargo.toml                      # Workspace configuration
```

## 🎯 Key Components

### Core Application (`code-ingest/`)
- **CLI Interface**: Command-line tool for code ingestion and task generation
- **Task System**: Windowed task management for large-scale code analysis
- **Database Engine**: PostgreSQL integration for code storage and querying
- **Processing Pipeline**: Multi-threaded file processing with context windows

### Task Generation System
- **SimpleTaskGenerator**: Produces Kiro-compatible checkbox markdown
- **WindowedTaskManager**: Handles large task volumes (1,551 tasks → 32 windows)
- **Progress Tracking**: Automatic state management and resumability
- **Format Compliance**: Exact match with Kiro parser requirements

### Documentation (`docs/`)
- **Analysis Documents**: Technical deep-dives and problem-solving approaches
- **Testing Files**: Validation scripts and format compliance tests
- **Implementation Guides**: Step-by-step usage and workflow documentation

### Configuration (`.kiro/`)
- **Steering Rules**: AI assistant guidance and analysis frameworks
- **Task Specifications**: Generated task files and examples
- **Reference Docs**: Mermaid guides, Rust patterns, analysis frameworks

## 🚀 Workflow Integration

### Development Workflow
1. **Code Ingestion**: `code-ingest ingest <source> --db-path <path>`
2. **Task Generation**: `code-ingest generate-hierarchical-tasks <table> --windowed`
3. **Analysis Execution**: Work through windowed tasks in Kiro IDE
4. **Progress Management**: Automatic advancement and tracking

### File Organization Principles
- **Clean Separation**: Source code, documentation, and configuration clearly separated
- **Logical Grouping**: Related files organized in appropriate directories
- **Version Control**: Temporary files excluded, important artifacts preserved
- **Accessibility**: Clear naming and structure for easy navigation

## 📊 Recent Improvements

### Task Generator Fix (Major Achievement)
- **Problem**: 19,497-line complex markdown files that Kiro couldn't parse
- **Solution**: Windowed system with simple checkbox format
- **Impact**: 99.5% file size reduction, 100% Kiro compatibility

### Repository Cleanup
- **Organized**: Moved analysis docs to `docs/analysis/`
- **Cleaned**: Removed temporary test executables
- **Structured**: Clear separation of concerns
- **Documented**: Comprehensive documentation index

This structure supports efficient development, clear documentation, and seamless integration with Kiro IDE workflows.
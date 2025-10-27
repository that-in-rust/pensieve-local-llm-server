# Code Ingest Examples

This directory contains practical examples of how to use code-ingest for various use cases.

## 📁 Examples

- [`basic_usage.md`](basic_usage.md) - Getting started with basic ingestion and querying
- [`security_analysis.md`](security_analysis.md) - Finding security issues in codebases
- [`architecture_analysis.md`](architecture_analysis.md) - Understanding codebase architecture
- [`performance_analysis.md`](performance_analysis.md) - Analyzing code performance patterns
- [`ide_integration.md`](ide_integration.md) - Using with IDE for systematic analysis
- [`batch_processing.md`](batch_processing.md) - Processing multiple repositories
- [`custom_queries.md`](custom_queries.md) - Advanced SQL queries for code analysis

## 🚀 Quick Start

1. **Install code-ingest**
   ```bash
   cargo install code-ingest
   ```

2. **Set up PostgreSQL**
   ```bash
   code-ingest pg-start
   ```

3. **Try the basic example**
   ```bash
   # Follow the steps in basic_usage.md
   code-ingest ingest https://github.com/rust-lang/mdBook --db-path ./analysis
   ```

## 💡 Use Case Categories

### 🔍 Code Discovery
- Finding specific functions or patterns
- Understanding code organization
- Locating configuration files

### 🛡️ Security Analysis
- Identifying potential vulnerabilities
- Finding hardcoded secrets
- Analyzing authentication patterns

### 🏗️ Architecture Analysis
- Understanding system dependencies
- Mapping data flow
- Identifying design patterns

### 📊 Code Quality
- Measuring code complexity
- Finding code duplication
- Analyzing test coverage patterns

### 🔧 Maintenance
- Finding deprecated APIs
- Locating TODO comments
- Identifying technical debt

Each example includes:
- **Objective**: What you're trying to achieve
- **Setup**: Required preparation steps
- **Commands**: Exact commands to run
- **Expected Output**: What results to expect
- **Analysis**: How to interpret the results
- **Next Steps**: Follow-up actions or deeper analysis
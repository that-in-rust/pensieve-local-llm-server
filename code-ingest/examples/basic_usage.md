# Basic Usage Example

Learn the fundamentals of code-ingest with a simple repository analysis.

## 🎯 Objective

Ingest a small Rust project and perform basic code exploration to understand:
- How many files of each type exist
- Where the main functionality is located
- Basic code metrics and structure

## 📋 Prerequisites

- code-ingest installed (`cargo install code-ingest`)
- PostgreSQL running locally
- Internet connection for GitHub access

## 🚀 Step-by-Step Walkthrough

### 1. Set Up Your Workspace

```bash
# Create a dedicated analysis directory
mkdir ~/code-analysis
cd ~/code-analysis

# Verify PostgreSQL is running
code-ingest pg-start
```

### 2. Ingest a Sample Repository

We'll use the popular `mdBook` project as our example:

```bash
# Ingest the mdBook repository
code-ingest ingest https://github.com/rust-lang/mdBook --db-path ./mdbook-analysis

# Expected output:
# 🚀 Starting ingestion...
# 📁 Source: https://github.com/rust-lang/mdBook
# 🗄️  Database: ./mdbook-analysis
# ⏳ Cloning repository...
# ✅ Repository cloned successfully
# 📊 Processing files...
# [████████████████████████████████] 847/847 files (100%)
# ✅ Ingestion completed successfully!
# 
# 📈 Results:
#    Table: INGEST_20240928143022
#    Files processed: 847
#    Duration: 2m 15s
#    Throughput: 6.3 files/sec
```

### 3. Explore the Ingested Data

#### List Available Tables
```bash
code-ingest list-tables --db-path ./mdbook-analysis

# Expected output:
# 📊 Ingestion Tables
# 
# ┌─────────────────────────┬─────────────────────────────────────┬───────────────┬──────────────┐
# │ Table Name              │ Repository URL                      │ Files         │ Created      │
# ├─────────────────────────┼─────────────────────────────────────┼───────────────┼──────────────┤
# │ INGEST_20240928143022   │ https://github.com/rust-lang/mdBook │ 847           │ 2 hours ago  │
# └─────────────────────────┴─────────────────────────────────────┴───────────────┴──────────────┘
```

#### Sample the Data
```bash
# Look at a few sample files to understand the data structure
code-ingest sample --table INGEST_20240928143022 --limit 5 --db-path ./mdbook-analysis

# Expected output:
# 📋 Sample from INGEST_20240928143022 (5 rows)
# 
# ┌─────────────────────────────────┬──────────────┬───────────┬──────────────┬────────────┐
# │ filepath                        │ filename     │ extension │ file_type    │ line_count │
# ├─────────────────────────────────┼──────────────┼───────────┼──────────────┼────────────┤
# │ src/main.rs                     │ main.rs      │ rs        │ direct_text  │ 45         │
# │ src/lib.rs                      │ lib.rs       │ rs        │ direct_text  │ 123        │
# │ Cargo.toml                      │ Cargo.toml   │ toml      │ direct_text  │ 18         │
# │ README.md                       │ README.md    │ md        │ direct_text  │ 67         │
# │ book-example/src/SUMMARY.md     │ SUMMARY.md   │ md        │ direct_text  │ 15         │
# └─────────────────────────────────┴──────────────┴───────────┴──────────────┴────────────┘
```

### 4. Basic Analysis Queries

#### File Type Distribution
```bash
code-ingest sql "SELECT file_type, COUNT(*) as count, 
                        ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 1) as percentage 
                 FROM INGEST_20240928143022 
                 GROUP BY file_type 
                 ORDER BY count DESC" --db-path ./mdbook-analysis

# Expected output:
# ┌──────────────┬───────┬────────────┐
# │ file_type    │ count │ percentage │
# ├──────────────┼───────┼────────────┤
# │ direct_text  │ 789   │ 93.2       │
# │ non_text     │ 46    │ 5.4        │
# │ convertible  │ 12    │ 1.4        │
# └──────────────┴───────┴────────────┘
```

#### Most Common File Extensions
```bash
code-ingest sql "SELECT extension, COUNT(*) as count 
                 FROM INGEST_20240928143022 
                 WHERE extension IS NOT NULL 
                 GROUP BY extension 
                 ORDER BY count DESC 
                 LIMIT 10" --db-path ./mdbook-analysis

# Expected output:
# ┌───────────┬───────┐
# │ extension │ count │
# ├───────────┼───────┤
# │ rs        │ 234   │
# │ md        │ 156   │
# │ toml      │ 45    │
# │ json      │ 23    │
# │ yml       │ 18    │
# │ txt       │ 12    │
# │ html      │ 8     │
# │ css       │ 6     │
# │ js        │ 4     │
# │ png       │ 89    │
# └───────────┴───────┘
```

#### Largest Files by Line Count
```bash
code-ingest sql "SELECT filepath, line_count, word_count 
                 FROM INGEST_20240928143022 
                 WHERE line_count IS NOT NULL 
                 ORDER BY line_count DESC 
                 LIMIT 10" --db-path ./mdbook-analysis

# Expected output:
# ┌─────────────────────────────────────┬────────────┬────────────┐
# │ filepath                            │ line_count │ word_count │
# ├─────────────────────────────────────┼────────────┼────────────┤
# │ src/renderer/html/mod.rs            │ 1,234      │ 8,567      │
# │ src/book/mod.rs                     │ 987        │ 6,234      │
# │ src/config.rs                       │ 756        │ 4,123      │
# │ src/preprocess/mod.rs               │ 654        │ 3,456      │
# │ tests/integration_tests.rs          │ 543        │ 2,987      │
# └─────────────────────────────────────┴────────────┴────────────┘
```

### 5. Content-Based Searches

#### Find Configuration Files
```bash
code-ingest sql "SELECT filepath, filename 
                 FROM INGEST_20240928143022 
                 WHERE filename LIKE '%.toml' OR filename LIKE '%.json' OR filename LIKE '%.yml' 
                 ORDER BY filepath" --db-path ./mdbook-analysis

# Expected output:
# ┌─────────────────────────────────────┬──────────────────┐
# │ filepath                            │ filename         │
# ├─────────────────────────────────────┼──────────────────┤
# │ Cargo.toml                          │ Cargo.toml       │
# │ book-example/book.toml              │ book.toml        │
# │ .github/workflows/main.yml          │ main.yml         │
# │ guide/book.toml                     │ book.toml        │
# └─────────────────────────────────────┴──────────────────┘
```

#### Search for Specific Functions
```bash
code-ingest sql "SELECT filepath, 
                        LENGTH(content_text) - LENGTH(REPLACE(content_text, 'fn ', '')) as fn_count
                 FROM INGEST_20240928143022 
                 WHERE content_text LIKE '%fn %' 
                   AND extension = 'rs'
                 ORDER BY fn_count DESC 
                 LIMIT 5" --db-path ./mdbook-analysis

# Expected output:
# ┌─────────────────────────────────────┬──────────┐
# │ filepath                            │ fn_count │
# ├─────────────────────────────────────┼──────────┤
# │ src/renderer/html/mod.rs            │ 45       │
# │ src/book/mod.rs                     │ 32       │
# │ src/config.rs                       │ 28       │
# │ tests/integration_tests.rs          │ 24       │
# │ src/preprocess/mod.rs               │ 19       │
# └─────────────────────────────────────┴──────────┘
```

#### Find Error Handling Patterns
```bash
code-ingest sql "SELECT filepath, 
                        (LENGTH(content_text) - LENGTH(REPLACE(LOWER(content_text), 'result<', ''))) / 7 as result_count,
                        (LENGTH(content_text) - LENGTH(REPLACE(LOWER(content_text), 'option<', ''))) / 7 as option_count
                 FROM INGEST_20240928143022 
                 WHERE content_text LIKE '%Result<%' OR content_text LIKE '%Option<%'
                   AND extension = 'rs'
                 ORDER BY (result_count + option_count) DESC 
                 LIMIT 5" --db-path ./mdbook-analysis

# Expected output:
# ┌─────────────────────────────────────┬──────────────┬──────────────┐
# │ filepath                            │ result_count │ option_count │
# ├─────────────────────────────────────┼──────────────┼──────────────┤
# │ src/renderer/html/mod.rs            │ 23           │ 12           │
# │ src/book/mod.rs                     │ 18           │ 8            │
# │ src/config.rs                       │ 15           │ 6            │
# │ src/preprocess/mod.rs               │ 12           │ 4            │
# │ tests/integration_tests.rs          │ 8            │ 3            │
# └─────────────────────────────────────┴──────────────┴──────────────┘
```

## 📊 Analysis Results

From this basic analysis, we can see that mdBook:

1. **File Composition**: 93.2% text files, mostly Rust source code and Markdown documentation
2. **Primary Language**: Rust (.rs files) with extensive Markdown documentation
3. **Architecture**: Modular structure with separate renderer, book, config, and preprocessing modules
4. **Code Quality**: Heavy use of Rust's Result and Option types for error handling
5. **Size**: Medium-sized project with ~800 files, largest modules around 1,000+ lines

## 🎯 Next Steps

Now that you understand the basics, try these follow-up examples:

1. **[Security Analysis](security_analysis.md)** - Look for potential security issues
2. **[Architecture Analysis](architecture_analysis.md)** - Understand the system design
3. **[IDE Integration](ide_integration.md)** - Use with your IDE for deeper analysis

## 💡 Key Takeaways

- **Start Small**: Begin with a manageable repository to understand the data structure
- **Explore First**: Use `sample` and basic queries to understand what data is available
- **Build Gradually**: Start with simple queries and add complexity as needed
- **Use Patterns**: Look for common patterns like function counts, error handling, etc.
- **Combine Metrics**: Use multiple metrics together for richer insights

## 🔧 Troubleshooting

**If ingestion fails:**
- Check PostgreSQL is running: `pg_isready`
- Verify network connectivity to GitHub
- Try a smaller repository first

**If queries are slow:**
- Add `LIMIT` clauses to large queries
- Use `EXPLAIN ANALYZE` to understand query performance
- Consider adding indexes for frequently queried columns

**If results seem incomplete:**
- Check the ingestion logs for skipped files
- Verify file type classification is working correctly
- Look for conversion tool errors (pdftotext, pandoc)
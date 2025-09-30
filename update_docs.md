# Documentation Update Summary

## Current Status: Commands Verified and Updated

After testing the actual commands, here's what's working vs what's deprecated:

## ✅ Working Commands:
1. **Ingestion**: `code-ingest ingest` (GitHub repos and local folders)
2. **Content Extraction**: `code-ingest extract-content` (creates A/B/C files)
3. **Task Generation**: `code-ingest generate-hierarchical-tasks` (creates structured task lists)

## ❌ Dead/Non-Working Commands:
1. **chunk-level-task-generator** - Command defined in CLI but not implemented (the actual working command is `generate-hierarchical-tasks`)

## Files Updated:
- README.md ✅ (removed dead generate-hierarchical-tasks command)
- READMELongForm20250929.md ✅ (updated with correct two-step workflow)

## Current Working Workflow:
```bash
# Step 1: Ingest data
./target/release/code-ingest ingest /path/to/data --folder-flag --db-path ./analysis

# Step 2: Extract content (creates A/B/C files)
./target/release/code-ingest extract-content TABLE_NAME --chunk-size 300 --output-dir .wipToBeDeletedFolder --db-path ./analysis
```

## Note:
Task generation currently requires manual creation or external tools since the CLI commands for task generation are not implemented despite being defined in the CLI interface.
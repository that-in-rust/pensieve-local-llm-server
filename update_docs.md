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

# Step 3: Generate hierarchical tasks (creates structured task list)
./target/release/code-ingest generate-hierarchical-tasks TABLE_NAME --chunks 300 --output TABLE_NAME_tasks.md --prompt-file .kiro/steering/analysis.md --db-path ./analysis
```

## Note:
The `generate-hierarchical-tasks` command IS working and creates structured task lists like the one we saw in `INGEST_20250930105036_tasks.md`. The confusion was about the `chunk-level-task-generator` command which is defined but not implemented.
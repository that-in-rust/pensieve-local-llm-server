# Migration Guide: From GenerateHierarchicalTasks to ChunkLevelTaskGenerator

## Overview

The `generate-hierarchical-tasks` command has been deprecated in favor of the simpler and more efficient `chunk-level-task-generator` command. This guide will help you migrate your workflows.

## Why the Change?

The old hierarchical task generation system was complex and difficult to maintain. The new `chunk-level-task-generator` provides:

- **Simpler workflow**: Two modes instead of complex hierarchical structures
- **Better performance**: Direct database operations without complex hierarchy building
- **Easier maintenance**: Fewer moving parts and clearer code paths
- **More reliable**: Fewer edge cases and error conditions

## Migration Examples

### Basic Task Generation

**OLD (Deprecated):**
```bash
code-ingest generate-hierarchical-tasks INGEST_20250928101039 \
  --output tasks.md --db-path /path/to/db
```

**NEW (Recommended):**
```bash
code-ingest chunk-level-task-generator INGEST_20250928101039 \
  --db-path /path/to/db
```

### Task Generation with Chunking

**OLD (Deprecated):**
```bash
code-ingest generate-hierarchical-tasks INGEST_20250928101039 \
  --chunks 500 --max-tasks 20 --output tasks.md \
  --prompt-file .kiro/steering/analysis.md --db-path /path/to/db
```

**NEW (Recommended):**
```bash
code-ingest chunk-level-task-generator INGEST_20250928101039 500 \
  --db-path /path/to/db --output-dir ./output
```

## Key Differences

### Output Format

| Aspect | Old Command | New Command |
|--------|-------------|-------------|
| **Output Files** | Single markdown file | Multiple content files + task list |
| **File Naming** | Custom output path | Standardized naming (content_N.txt, etc.) |
| **Task Format** | Hierarchical structure | Simple numbered list |
| **Content Files** | Referenced in tasks | Generated separately (content, contentL1, contentL2) |

### Workflow Changes

#### Old Workflow:
1. Run `generate-hierarchical-tasks`
2. Get single markdown file with hierarchical tasks
3. Manually reference content files

#### New Workflow:
1. Run `chunk-level-task-generator`
2. Get content files (content_N.txt, contentL1_N.txt, contentL2_N.txt)
3. Get task list file referencing content files
4. Execute tasks directly in Kiro

### Command Options

| Old Option | New Equivalent | Notes |
|------------|----------------|-------|
| `--output` | `--output-dir` | Directory instead of single file |
| `--chunks` | `chunk_size` (positional) | Second positional argument |
| `--max-tasks` | Not needed | Automatic based on table size |
| `--levels` | Not applicable | Simplified to file/chunk modes |
| `--groups` | Not applicable | Simplified to file/chunk modes |
| `--windowed` | Not applicable | Simplified workflow |
| `--prompt-file` | Not applicable | Content files are self-contained |

## Migration Steps

### Step 1: Update Your Scripts

Replace any scripts or documentation that use the old command:

```bash
# Find and replace in your scripts
sed -i 's/generate-hierarchical-tasks/chunk-level-task-generator/g' your-script.sh

# Update command structure
# OLD: generate-hierarchical-tasks TABLE --chunks SIZE --output FILE
# NEW: chunk-level-task-generator TABLE SIZE --output-dir DIR
```

### Step 2: Update Your Workflows

1. **File-level mode** (no chunking):
   ```bash
   code-ingest chunk-level-task-generator YOUR_TABLE --db-path /path/to/db
   ```

2. **Chunk-level mode** (with chunking):
   ```bash
   code-ingest chunk-level-task-generator YOUR_TABLE 500 --db-path /path/to/db
   ```

### Step 3: Verify Output

The new command generates:
- `content_1.txt`, `content_2.txt`, ... (individual row content)
- `contentL1_1.txt`, `contentL1_2.txt`, ... (row + next row)
- `contentL2_1.txt`, `contentL2_2.txt`, ... (row + next + next2)
- `task_list.txt` (references to content files)

### Step 4: Update Kiro Integration

The new task list format is simpler and more Kiro-friendly:

```markdown
# Task List

- [ ] 1. Process content_1.txt
- [ ] 2. Process content_2.txt
- [ ] 3. Process content_3.txt
```

## Troubleshooting

### Common Issues

1. **"Command not found"**: Make sure you're using the latest version of code-ingest
2. **Different output format**: The new command generates multiple files instead of one
3. **Missing chunk size**: For chunking, provide chunk size as second positional argument
4. **Output directory**: Use `--output-dir` instead of `--output`

### Getting Help

```bash
# Show help for new command
code-ingest chunk-level-task-generator --help

# Show examples
code-ingest examples
```

## Timeline

- **Current**: Both commands available, old command shows deprecation warnings
- **Next release**: Old command will be removed
- **Migration deadline**: Please migrate before the next major release

## Benefits of Migration

1. **Simpler maintenance**: Fewer complex hierarchical structures
2. **Better performance**: Direct database operations
3. **Clearer output**: Separate content files and task lists
4. **Future-proof**: Active development and support
5. **Better Kiro integration**: Optimized for Kiro workflows

## Support

If you encounter issues during migration:

1. Check this guide for common patterns
2. Use `code-ingest examples` for usage examples
3. Run with `--verbose` for detailed logging
4. File issues on the project repository

The new `chunk-level-task-generator` provides a cleaner, more maintainable approach to task generation while preserving the core functionality you need.
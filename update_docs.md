# Documentation Update Summary

The following files have been updated to remove references to the deprecated `generate-hierarchical-tasks` command and replace them with the new `chunk-level-task-generator` command:

## Files Updated:
- code-ingest/examples/chunked_analysis_workflow.md ✅
- code-ingest/examples/git_repository_workflow.md (needs update)
- code-ingest/examples/local_folder_workflow.md (needs update)
- code-ingest/docs/MIGRATION_GUIDE.md (needs update)

## Migration Pattern:
OLD: `code-ingest generate-hierarchical-tasks TABLE --chunks SIZE --output FILE --levels N --groups M`
NEW: `code-ingest chunk-level-task-generator TABLE SIZE --output-dir DIR`

The new command is simpler and more focused on the core functionality.
#!/bin/bash

# Unified Progress Tracker Script
# Consolidates repository snapshots, file tracking, and session context updates
# Replaces multiple overlapping hooks with single intelligent system

set -e

# Get git repository root to ensure all paths are relative to repo
GIT_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
cd "$GIT_ROOT"

TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S IST')
COMMIT_TIMESTAMP=$(date '+%Y%m%d %H%M IST')
SNAPSHOT_DIR="$GIT_ROOT/.kiro/file-snapshots"
SPEC_DIR="$GIT_ROOT/.kiro/specs/parseltongue-aim-daemon"
CONTEXT_FILE="$SPEC_DIR/SESSION_CONTEXT.md"
CURRENT_SNAPSHOT="$SNAPSHOT_DIR/current-snapshot.md"
PREVIOUS_SNAPSHOT="$SNAPSHOT_DIR/previous-snapshot.md"
CHANGE_LOG="$SNAPSHOT_DIR/change-log.md"
TEMP_SNAPSHOT="/tmp/unified_snapshot_$$.md"

# Ensure directories exist
mkdir -p "$SNAPSHOT_DIR"
mkdir -p "$SPEC_DIR"

echo "🔄 Unified Progress Tracker - $TIMESTAMP"

# Function to generate comprehensive repository snapshot
generate_repository_snapshot() {
    echo "# Repository Snapshot - $TIMESTAMP" > "$TEMP_SNAPSHOT"
    echo "" >> "$TEMP_SNAPSHOT"

    # Summary statistics - exclude .git but include other hidden files
    TOTAL_FILES=$(find "$GIT_ROOT" -type f ! -path "$GIT_ROOT/.git/*" ! -path "$GIT_ROOT/target/*" ! -path "$GIT_ROOT/node_modules/*" | wc -l)
    TOTAL_LINES=$(find "$GIT_ROOT" -type f ! -path "$GIT_ROOT/.git/*" -name "*.md" -o -name "*.rs" -o -name "*.toml" -o -name "*.json" -o -name "*.txt" -o -name "*.yml" -o -name "*.yaml" | xargs wc -l 2>/dev/null | tail -1 | awk '{print $1}' || echo "0")
    TOTAL_WORDS=$(find "$GIT_ROOT" -type f ! -path "$GIT_ROOT/.git/*" -name "*.md" -o -name "*.rs" -o -name "*.toml" -o -name "*.json" -o -name "*.txt" -o -name "*.yml" -o -name "*.yaml" | xargs wc -w 2>/dev/null | tail -1 | awk '{print $1}' || echo "0")

    echo "## Summary Statistics" >> "$TEMP_SNAPSHOT"
    echo "- **Total Files**: $(printf "%'d" $TOTAL_FILES)" >> "$TEMP_SNAPSHOT"
    echo "- **Total Lines**: $(printf "%'d" $TOTAL_LINES)" >> "$TEMP_SNAPSHOT"
    echo "- **Total Words**: $(printf "%'d" $TOTAL_WORDS)" >> "$TEMP_SNAPSHOT"
    echo "- **Snapshot Time**: $TIMESTAMP" >> "$TEMP_SNAPSHOT"
    echo "" >> "$TEMP_SNAPSHOT"

    # File inventory with counts
    echo "## File Inventory" >> "$TEMP_SNAPSHOT"
    echo "" >> "$TEMP_SNAPSHOT"
    echo "| File Path | Lines | Words | Size |" >> "$TEMP_SNAPSHOT"
    echo "|-----------|-------|-------|------|" >> "$TEMP_SNAPSHOT"

    find "$GIT_ROOT" -type f ! -path "$GIT_ROOT/.git/*" ! -path "$GIT_ROOT/target/*" ! -path "$GIT_ROOT/node_modules/*" | sort | while read -r file; do
        if [ -f "$file" ]; then
            # Convert absolute path to relative path from git root
            rel_path=$(realpath --relative-to="$GIT_ROOT" "$file")
            if file "$file" | grep -q "text"; then
                lines=$(wc -l < "$file" 2>/dev/null || echo "0")
                words=$(wc -w < "$file" 2>/dev/null || echo "0")
            else
                lines="[binary]"
                words="[binary]"
            fi
            size=$(ls -lh "$file" 2>/dev/null | awk '{print $5}' || echo "?")
            echo "| $rel_path | $lines | $words | $size |" >> "$TEMP_SNAPSHOT"
        fi
    done
}

# Function to update session context
update_session_context() {
    if [ -f "$CONTEXT_FILE" ]; then
        # Update timestamp
        sed -i "s/Last Updated: .*/Last Updated: $(date +%Y-%m-%d)/" "$CONTEXT_FILE"
        
        # Update task progress if requirements-tasks.md exists
        if [ -f "$SPEC_DIR/requirements-tasks.md" ]; then
            TOTAL_TASKS=$(grep -c "^- \[" "$SPEC_DIR/requirements-tasks.md" 2>/dev/null || echo "0")
            COMPLETED_TASKS=$(grep -c "^- \[x\]" "$SPEC_DIR/requirements-tasks.md" 2>/dev/null || echo "0")
            
            if [ "$TOTAL_TASKS" -gt 0 ]; then
                PROGRESS=$((COMPLETED_TASKS * 100 / TOTAL_TASKS))
                sed -i "s/Document Analysis [0-9]*% Complete/Document Analysis ${PROGRESS}% Complete/" "$CONTEXT_FILE"
                sed -i "s/Progress: [0-9]*%/Progress: ${PROGRESS}%/" "$CONTEXT_FILE"
            fi
        fi
        
        # Detect current phase
        if [ ! -f "$SPEC_DIR/design.md" ]; then
            PHASE="Requirements Analysis (Phase 1)"
        elif [ ! -f "$SPEC_DIR/implementation-plan.md" ]; then
            PHASE="Design Development (Phase 2)"
        else
            PHASE="Implementation Planning (Phase 3)"
        fi
        
        sed -i "s/Current Phase: .*/Current Phase: $PHASE/" "$CONTEXT_FILE"
        
        echo "✅ Session context updated"
    fi
}

# Function to generate delta report
generate_delta_report() {
    if [ ! -f "$PREVIOUS_SNAPSHOT" ]; then
        echo "📝 Initial snapshot created" >> "$CHANGE_LOG"
        return
    fi

    echo "" >> "$CHANGE_LOG"
    echo "## Delta Report - $TIMESTAMP" >> "$CHANGE_LOG"
    echo "" >> "$CHANGE_LOG"

    # Extract previous stats
    PREV_FILES=$(grep "Total Files" "$PREVIOUS_SNAPSHOT" | sed 's/.*: //' | tr -d ',' | tr -d '*' || echo "0")
    PREV_LINES=$(grep "Total Lines" "$PREVIOUS_SNAPSHOT" | sed 's/.*: //' | tr -d ',' | tr -d '*' || echo "0")
    PREV_WORDS=$(grep "Total Words" "$PREVIOUS_SNAPSHOT" | sed 's/.*: //' | tr -d ',' | tr -d '*' || echo "0")

    # Calculate changes
    FILE_DIFF=$((TOTAL_FILES - PREV_FILES))
    LINE_DIFF=$((TOTAL_LINES - PREV_LINES))
    WORD_DIFF=$((TOTAL_WORDS - PREV_WORDS))

    echo "### Summary Changes" >> "$CHANGE_LOG"
    echo "- **File Count**: $FILE_DIFF ($(printf "%'d" $TOTAL_FILES) total)" >> "$CHANGE_LOG"
    echo "- **Line Count**: $(printf "%'d" $LINE_DIFF) ($(printf "%'d" $TOTAL_LINES) total)" >> "$CHANGE_LOG"
    echo "- **Word Count**: $(printf "%'d" $WORD_DIFF) ($(printf "%'d" $TOTAL_WORDS) total)" >> "$CHANGE_LOG"
    echo "" >> "$CHANGE_LOG"

    # Detect specific file changes
    if [ -f "$PREVIOUS_SNAPSHOT" ] && [ -f "$TEMP_SNAPSHOT" ]; then
        echo "### File-Level Changes" >> "$CHANGE_LOG"
        
        # Extract file lists for comparison
        grep "^| \." "$PREVIOUS_SNAPSHOT" | awk -F'|' '{print $2}' | sed 's/^ *//;s/ *$//' | sort > /tmp/prev_files.txt 2>/dev/null || touch /tmp/prev_files.txt
        grep "^| \." "$TEMP_SNAPSHOT" | awk -F'|' '{print $2}' | sed 's/^ *//;s/ *$//' | sort > /tmp/curr_files.txt 2>/dev/null || touch /tmp/curr_files.txt
        
        # Find added files
        ADDED=$(comm -13 /tmp/prev_files.txt /tmp/curr_files.txt 2>/dev/null || echo "")
        if [ -n "$ADDED" ] && [ "$ADDED" != "" ]; then
            echo "**Added Files:**" >> "$CHANGE_LOG"
            echo "$ADDED" | head -10 | while read -r file; do
                echo "- $file" >> "$CHANGE_LOG"
            done
            echo "" >> "$CHANGE_LOG"
        fi
        
        # Find removed files
        REMOVED=$(comm -23 /tmp/prev_files.txt /tmp/curr_files.txt 2>/dev/null || echo "")
        if [ -n "$REMOVED" ] && [ "$REMOVED" != "" ]; then
            echo "**Removed Files:**" >> "$CHANGE_LOG"
            echo "$REMOVED" | head -10 | while read -r file; do
                echo "- $file" >> "$CHANGE_LOG"
            done
            echo "" >> "$CHANGE_LOG"
        fi
        
        # Cleanup temp files
        rm -f /tmp/prev_files.txt /tmp/curr_files.txt
    fi
    
    echo "---" >> "$CHANGE_LOG"
    echo "" >> "$CHANGE_LOG"
}

# Main execution
echo "📊 Generating repository snapshot..."

# Move current to previous if it exists
if [ -f "$CURRENT_SNAPSHOT" ]; then
    cp "$CURRENT_SNAPSHOT" "$PREVIOUS_SNAPSHOT"
fi

# Generate new snapshot
generate_repository_snapshot

# Update session context
update_session_context

# Initialize change log if needed
if [ ! -f "$CHANGE_LOG" ]; then
    echo "# Repository Change Log" > "$CHANGE_LOG"
    echo "" >> "$CHANGE_LOG"
    echo "Unified tracking of all repository changes with comprehensive delta reporting." >> "$CHANGE_LOG"
    echo "" >> "$CHANGE_LOG"
fi

# Generate delta report
generate_delta_report

# Move temp snapshot to current
mv "$TEMP_SNAPSHOT" "$CURRENT_SNAPSHOT"

# Git operations (only .kiro directory)
# Check for unstaged changes, staged changes, and untracked files in .kiro/
UNTRACKED_KIRO=$(git ls-files --others --exclude-standard .kiro/ | wc -l)
if git diff --quiet .kiro/ && git diff --cached --quiet .kiro/ && [ "$UNTRACKED_KIRO" -eq 0 ]; then
    echo "ℹ️  No .kiro changes to commit"
else
    git add .kiro/
    
    if ! git diff --cached --quiet .kiro/; then
        COMMIT_MSG="$COMMIT_TIMESTAMP"
        git commit -m "$COMMIT_MSG"
        
        if git push origin v01 2>/dev/null; then
            echo "✅ Changes committed and pushed to v01: $COMMIT_MSG"
        else
            echo "⚠️  Changes committed locally (push failed): $COMMIT_MSG"
        fi
    fi
fi

echo "✅ Unified progress tracking complete"
echo "📊 Files: $(printf "%'d" $TOTAL_FILES) | Lines: $(printf "%'d" $TOTAL_LINES) | Words: $(printf "%'d" $TOTAL_WORDS)"

echo "Is this not a cool sidekick script? Is it"
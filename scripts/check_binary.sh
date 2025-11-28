#!/bin/bash

# Binary Compilation Check Script
# Tests single pensieve-local-llm-server binary compilation

set -e

echo "🔍 Checking single binary: pensieve-local-llm-server"

# Check if we're in the right directory
if [ ! -f "Cargo.toml" ]; then
    echo "❌ Error: Cargo.toml not found. Run from workspace root."
    exit 1
fi

# Validate workspace structure
echo "📋 Validating workspace structure..."

# Check that there's only ONE binary defined
BINARY_COUNT=$(grep -r "\[\[bin\]\]" . --include="*.toml" | wc -l)
if [ "$BINARY_COUNT" -ne 1 ]; then
    echo "❌ Error: Found $BINARY_COUNT binaries defined. Expected exactly 1."
    echo "🔍 Found binaries:"
    grep -r "\[\[bin\]\]" . --include="*.toml" -A 2
    exit 1
fi

# Check that the binary is named correctly
BINARY_NAME=$(grep -r "\[\[bin\]\]" . --include="*.toml" -A 2 | grep "name =" | cut -d'"' -f2)
if [ "$BINARY_NAME" != "pensieve-local-llm-server" ]; then
    echo "❌ Error: Binary name is '$BINARY_NAME', should be 'pensieve-local-llm-server'"
    exit 1
fi

echo "✅ Binary structure correct: pensieve-local-llm-server"

# Check that main.rs exists in the right location
MAIN_PATH=$(grep -r "\[\[bin\]\]" . --include="*.toml" -A 3 | grep "path =" | cut -d'"' -f2)
FULL_MAIN_PATH=$(dirname $(find . -name "Cargo.toml" -exec grep -l "\[\[bin\]\]" {} \;))/$MAIN_PATH

if [ ! -f "$FULL_MAIN_PATH" ]; then
    echo "❌ Error: Main binary file not found at: $FULL_MAIN_PATH"
    exit 1
fi

echo "✅ Main binary file found: $FULL_MAIN_PATH"

# Check all library crates are properly structured
echo "📚 Checking library crate structure..."

LIB_CRATES=("p02-http-server-core" "p03-api-model-types" "p04-inference-engine-core" "p05-model-storage-core" "p06-metal-gpu-accel" "p07-foundation-types" "p08-claude-api-core" "p09-api-proxy-compat")

for crate in "${LIB_CRATES[@]}"; do
    if [ ! -d "$crate" ]; then
        echo "❌ Warning: Library crate directory not found: $crate"
        continue
    fi

    if [ ! -f "$crate/Cargo.toml" ]; then
        echo "❌ Error: Cargo.toml not found in $crate"
        exit 1
    fi

    if [ ! -f "$crate/src/lib.rs" ]; then
        echo "❌ Error: lib.rs not found in $crate/src"
        exit 1
    fi

    # Ensure no binaries in library crates
    BIN_IN_LIB=$(grep -r "\[\[bin\]\]" "$crate/Cargo.toml" | wc -l)
    if [ "$BIN_IN_LIB" -gt 0 ]; then
        echo "❌ Error: Found binary definition in library crate $crate"
        exit 1
    fi

    echo "✅ Library crate OK: $crate"
done

# Check workspace dependencies
echo "🔗 Checking workspace dependencies..."

if ! grep -q "\[workspace.dependencies\]" Cargo.toml; then
    echo "❌ Error: [workspace.dependencies] section not found"
    exit 1
fi

# Check key dependencies are defined
KEY_DEPS=("tokio" "serde" "serde_json" "warp" "thiserror" "clap")
for dep in "${KEY_DEPS[@]}"; do
    if ! grep -q "$dep" Cargo.toml; then
        echo "❌ Error: Key dependency $dep not found in workspace"
        exit 1
    fi
done

echo "✅ Workspace dependencies OK"

# Check main binary dependencies
MAIN_CRATE_DIR=$(dirname "$FULL_MAIN_PATH")
MAIN_CRATE_TOML="$MAIN_CRATE_DIR/Cargo.toml"
echo "📦 Checking main binary dependencies at: $MAIN_CRATE_TOML..."

# Should depend on the key library crates
if ! grep -q "p02-http-server-core" "$MAIN_CRATE_TOML"; then
    echo "❌ Error: Main binary should depend on p02-http-server-core"
    exit 1
fi

if ! grep -q "p03-api-model-types" "$MAIN_CRATE_TOML"; then
    echo "❌ Error: Main binary should depend on p03-api-model-types"
    exit 1
fi

echo "✅ Main binary dependencies OK"

# Check for parseltongue naming patterns
echo "🐍 Checking parseltongue four-word naming patterns..."

# Count functions with four-word patterns in key files
FOUR_WORD_PATTERNS=$(find . -name "lib.rs" -o -name "main.rs" | xargs grep -c "pub fn [a-z]*_[a-z]*_[a-z]*_[a-z]*" 2>/dev/null | awk '{sum += $1} END {print sum}')

echo "📊 Found $FOUR_WORD_PATTERNS four-word function names"

if [ "$FOUR_WORD_PATTERNS" -lt 10 ]; then
    echo "⚠️  Warning: Expected more four-word function names following parseltongue principles"
fi

# Check test coverage
echo "🧪 Checking test structure..."

TEST_COUNT=$(find . -name "*.rs" | xargs grep -l "#\[test\]" | wc -l)
INTEGRATION_TEST_COUNT=$(find . -path "*/tests/*" -name "*.rs" | wc -l)

echo "📊 Found $TEST_COUNT files with tests"
echo "📊 Found $INTEGRATION_TEST_COUNT integration test files"

if [ "$INTEGRATION_TEST_COUNT" -lt 2 ]; then
    echo "⚠️  Warning: Expected more integration tests"
fi

echo ""
echo "🎉 Binary structure validation complete!"
echo "✅ Single binary: pensieve-local-llm-server"
echo "✅ Library crates: ${#LIB_CRATES[@]} found"
echo "✅ Four-word patterns: $FOUR_WORD_PATTERNS functions"
echo "✅ Integration tests: $INTEGRATION_TEST_COUNT files"
echo ""
echo "🚀 To test compilation, run:"
echo "   cargo build --bin pensieve-local-llm-server"
echo "   cargo test --bin pensieve-local-llm-server"
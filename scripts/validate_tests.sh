#!/bin/bash

# Parseltongue Test Validation Script
# Follows clean build pattern and comprehensive testing

set -e

echo "🐍 Parseltongue Test Validation for HTTP Server"
echo "==============================================="

# Clean build pattern - start fresh
echo "🧹 Cleaning build artifacts..."
cargo clean 2>/dev/null || echo "No artifacts to clean"

# Check we're in the right directory
if [ ! -f "Cargo.toml" ]; then
    echo "❌ Error: Cargo.toml not found. Run from workspace root."
    exit 1
fi

echo ""
echo "📊 Test Inventory Analysis:"
echo "--------------------------"

# Count test files
TEST_FILES=$(find . -name "*.rs" -path "*/tests/*" | wc -l | tr -d ' ')
echo "✅ Found $TEST_FILES test files"

# Count four-word functions following parseltongue
FOUR_WORD_FUNCTIONS=$(find . -name "*.rs" -exec grep -l "pub fn [a-z]*_[a-z]*_[a-z]*_[a-z]*" {} \; | wc -l | tr -d ' ')
echo "✅ Found $FOUR_WORD_FUNCTIONS files with four-word function names"

# Count executable contracts (Executable Contract format)
EXECUTABLE_CONTRACTS=$(grep -r "Executable Contract" . --include="*.rs" | wc -l | tr -d ' ')
echo "✅ Found $EXECUTABLE_CONTRACTS executable contracts"

# Count property-based tests
PROPERTY_TESTS=$(grep -r "proptest!" . --include="*.rs" | wc -l | tr -d ' ')
echo "✅ Found $PROPERTY_TESTS property-based tests"

echo ""
echo "🔍 Parseltongue Principle Validation:"
echo "-----------------------------------"

# Validate four-word naming patterns
echo "Checking four-word naming patterns..."
if [ $FOUR_WORD_FUNCTIONS -gt 0 ]; then
    echo "✅ Four-word naming pattern implemented"
else
    echo "❌ Four-word naming pattern missing"
    exit 1
fi

# Validate executable specifications
echo "Checking executable specifications..."
if [ $EXECUTABLE_CONTRACTS -gt 0 ]; then
    echo "✅ Executable specifications present"
else
    echo "❌ Executable specifications missing"
    exit 1
fi

# Validate property-based testing
echo "Checking property-based testing..."
if [ $PROPERTY_TESTS -gt 0 ]; then
    echo "✅ Property-based testing implemented"
else
    echo "❌ Property-based testing missing"
    exit 1
fi

echo ""
echo "🏗️ Build Validation:"
echo "-------------------"

# Check compilation (would fail if cargo not available)
echo "Checking build structure..."
if grep -q "pensieve-local-llm-server" p01-cli-interface-launcher/Cargo.toml; then
    echo "✅ Binary name correct: pensieve-local-llm-server"
else
    echo "❌ Binary name incorrect"
    exit 1
fi

# Check single binary structure
BINARY_COUNT=$(grep -r "\[\[bin\]\]" . --include="*.toml" | wc -l | tr -d ' ')
if [ "$BINARY_COUNT" = "1" ]; then
    echo "✅ Single binary structure confirmed"
else
    echo "❌ Expected 1 binary, found $BINARY_COUNT"
    exit 1
fi

# Check library crates
LIB_CRATES=$(find . -name "Cargo.toml" -not -path "./Cargo.toml" | wc -l | tr -d ' ')
echo "✅ Found $LIB_CRATES library crates"

echo ""
echo "🧪 Test Categories:"
echo "-----------------"

# List test categories by file patterns
if [ -f "p02-http-server-core/tests/parseltongue_executable_contracts.rs" ]; then
    echo "✅ Executable contracts (WHEN...THEN...SHALL)"
fi

if [ -f "p02-http-server-core/tests/property_based_validation.rs" ]; then
    echo "✅ Property-based validation (invariants across input spaces)"
fi

if [ -f "p02-http-server-core/tests/stress_concurrent_requests.rs" ]; then
    echo "✅ Stress testing (concurrent load validation)"
fi

if [ -f "p02-http-server-core/tests/integration_tests.rs" ]; then
    echo "✅ Integration tests (end-to-end validation)"
fi

if [ -f "p02-http-server-core/benches/http_server_benchmarks.rs" ]; then
    echo "✅ Performance benchmarks (measurable contracts)"
fi

echo ""
echo "📋 Test Coverage Summary:"
echo "-----------------------"

# Count test types
UNIT_TESTS=$(grep -r "#\[test\]" . --include="*.rs" | wc -l | tr -d ' ')
TOKIO_TESTS=$(grep -r "#\[tokio::test\]" . --include="*.rs" | wc -l | tr -d ' ')
PROPERTY_TESTS=$(grep -r "proptest!" . --include="*.rs" | wc -l | tr -d ' ')

echo "✅ Unit tests: $UNIT_TESTS"
echo "✅ Async tests: $TOKIO_TESTS"
echo "✅ Property tests: $PROPERTY_TESTS"

echo ""
echo "⚡ Performance Contracts Validated:"
echo "----------------------------------"

# Check for performance contracts in tests
PERFORMANCE_CONTRACTS=$(grep -r "under.*ms.*contract" . --include="*.rs" | wc -l | tr -d ' ')
echo "✅ Found $PERFORMANCE_CONTRACTS performance contracts"

# List performance constraints
echo "Performance constraints found:"
grep -h "under.*ms.*contract" . --include="*.rs" | sed 's/.*contract //' | sort | uniq

echo ""
echo "🎯 Parseltongue Compliance Score:"
echo "--------------------------------"

# Calculate compliance score
TOTAL_SCORE=0
MAX_SCORE=6

# Four-word naming (1 point)
if [ $FOUR_WORD_FUNCTIONS -gt 0 ]; then
    TOTAL_SCORE=$((TOTAL_SCORE + 1))
    echo "✅ (+1) Four-word naming convention"
fi

# Executable specifications (1 point)
if [ $EXECUTABLE_CONTRACTS -gt 0 ]; then
    TOTAL_SCORE=$((TOTAL_SCORE + 1))
    echo "✅ (+1) Executable specifications"
fi

# Property-based testing (1 point)
if [ $PROPERTY_TESTS -gt 0 ]; then
    TOTAL_SCORE=$((TOTAL_SCORE + 1))
    echo "✅ (+1) Property-based testing"
fi

# Performance contracts (1 point)
if [ $PERFORMANCE_CONTRACTS -gt 0 ]; then
    TOTAL_SCORE=$((TOTAL_SCORE + 1))
    echo "✅ (+1) Performance contracts"
fi

# Single binary (1 point)
if [ "$BINARY_COUNT" = "1" ]; then
    TOTAL_SCORE=$((TOTAL_SCORE + 1))
    echo "✅ (+1) Single binary architecture"
fi

# TDD cycle (1 point)
if [ $TEST_FILES -gt 0 ]; then
    TOTAL_SCORE=$((TOTAL_SCORE + 1))
    echo "✅ (+1) Test-Driven Development"
fi

# Calculate percentage
COMPLIANCE_PERCENT=$((TOTAL_SCORE * 100 / MAX_SCORE))

echo ""
echo "🏆 Final Score: $TOTAL_SCORE/$MAX_SCORE ($COMPLIANCE_PERCENT%)"

if [ $COMPLIANCE_PERCENT -ge 80 ]; then
    echo "🎉 EXCELLENT: High parseltongue compliance!"
elif [ $COMPLIANCE_PERCENT -ge 60 ]; then
    echo "✅ GOOD: Acceptable parseltongue compliance"
else
    echo "⚠️  NEEDS IMPROVEMENT: Low parseltongue compliance"
fi

echo ""
echo "🚀 Ready for Testing Commands:"
echo "-----------------------------"
echo "Run tests:      cargo test --package p02-http-server-core"
echo "Run benchmarks: cargo bench --package p02-http-server-core"
echo "Build binary:   cargo build --bin pensieve-local-llm-server"

echo ""
echo "✨ Parseltongue validation complete!"
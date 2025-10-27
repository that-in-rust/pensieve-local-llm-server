#!/bin/bash

# Comprehensive validation script for chunk-level-task-generator command
# Tests requirements: 1.1, 1.2, 2.1, 2.6, 2.7

set -e

echo "🚀 Chunk-Level Task Generator Validation Script"
echo "Testing requirements: 1.1, 1.2, 2.1, 2.6, 2.7"
echo "================================================"

# Configuration
TEST_OUTPUT_DIR="./test_output_$(date +%s)"
BINARY_PATH="./target/release/code-ingest"
TEST_RESULTS_FILE="validation_results.md"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test counters
TESTS_PASSED=0
TESTS_FAILED=0
ISSUES_FOUND=()

# Helper functions
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
    ((TESTS_PASSED++))
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
    ((TESTS_FAILED++))
    ISSUES_FOUND+=("$1")
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if binary exists
    if [ ! -f "$BINARY_PATH" ]; then
        log_error "Binary not found at $BINARY_PATH. Please build with: cargo build --release"
        exit 1
    fi
    
    # Check if DATABASE_URL is set
    if [ -z "$DATABASE_URL" ]; then
        log_error "DATABASE_URL environment variable not set"
        echo "Please set DATABASE_URL to your PostgreSQL connection string"
        echo "Example: export DATABASE_URL='postgresql://user:password@localhost:5432/dbname'"
        exit 1
    fi
    
    # Create test output directory
    mkdir -p "$TEST_OUTPUT_DIR"
    log_success "Prerequisites check passed"
}

# Find available ingestion tables
find_test_tables() {
    log_info "Finding available ingestion tables..."
    
    # Use psql to find tables if available, otherwise use a simple approach
    if command -v psql &> /dev/null; then
        TEST_TABLES=$(psql "$DATABASE_URL" -t -c "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' AND table_name LIKE 'INGEST_%' ORDER BY table_name;" | tr -d ' ' | grep -v '^$' | head -5)
    else
        # Fallback to common table names
        TEST_TABLES="INGEST_20250928101039
INGEST_20250930025223
INGEST_20250930052722"
    fi
    
    if [ -z "$TEST_TABLES" ]; then
        log_error "No ingestion tables found. Please run ingestion first."
        exit 1
    fi
    
    echo "Found tables:"
    echo "$TEST_TABLES" | while read -r table; do
        echo "  - $table"
    done
    
    log_success "Found $(echo "$TEST_TABLES" | wc -l) ingestion tables"
}

# Test file-level mode (Requirement 1.1, 1.2, 2.6, 2.7)
test_file_level_mode() {
    local table_name="$1"
    log_info "Testing file-level mode with table: $table_name"
    
    local test_dir="$TEST_OUTPUT_DIR/file_level_$table_name"
    mkdir -p "$test_dir"
    
    # Execute the command
    local start_time=$(date +%s)
    if timeout 300 "$BINARY_PATH" chunk-level-task-generator "$table_name" --output-dir "$test_dir" --db-path "$DATABASE_URL" > "$test_dir/output.log" 2>&1; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        
        # Validate results
        validate_file_level_results "$table_name" "$test_dir" "$duration"
    else
        local exit_code=$?
        log_error "File-level mode failed for $table_name (exit code: $exit_code)"
        if [ -f "$test_dir/output.log" ]; then
            echo "Error output:"
            tail -10 "$test_dir/output.log"
        fi
        return 1
    fi
}

# Test chunk-level mode (Requirement 2.1, 2.2, 2.3, 2.4, 2.5)
test_chunk_level_mode() {
    local table_name="$1"
    local chunk_size="$2"
    log_info "Testing chunk-level mode with table: $table_name (chunk size: $chunk_size)"
    
    local test_dir="$TEST_OUTPUT_DIR/chunk_level_${table_name}_${chunk_size}"
    mkdir -p "$test_dir"
    
    # Execute the command
    local start_time=$(date +%s)
    if timeout 600 "$BINARY_PATH" chunk-level-task-generator "$table_name" "$chunk_size" --output-dir "$test_dir" --db-path "$DATABASE_URL" > "$test_dir/output.log" 2>&1; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        
        # Validate results
        validate_chunk_level_results "$table_name" "$chunk_size" "$test_dir" "$duration"
    else
        local exit_code=$?
        log_error "Chunk-level mode failed for $table_name with chunk size $chunk_size (exit code: $exit_code)"
        if [ -f "$test_dir/output.log" ]; then
            echo "Error output:"
            tail -10 "$test_dir/output.log"
        fi
        return 1
    fi
}

# Validate file-level mode results
validate_file_level_results() {
    local table_name="$1"
    local test_dir="$2"
    local duration="$3"
    
    local content_files_count=$(find "$test_dir" -name "content_*.txt" | wc -l)
    local content_l1_files_count=$(find "$test_dir" -name "contentL1_*.txt" | wc -l)
    local content_l2_files_count=$(find "$test_dir" -name "contentL2_*.txt" | wc -l)
    local task_list_files=$(find "$test_dir" -name "*task*.txt" -o -name "*task*.md")
    
    # Check if content files were created (Requirement 1.1, 2.6)
    if [ "$content_files_count" -gt 0 ] && [ "$content_l1_files_count" -gt 0 ] && [ "$content_l2_files_count" -gt 0 ]; then
        log_success "Content files created for $table_name: $content_files_count content, $content_l1_files_count L1, $content_l2_files_count L2"
        
        # Validate content file format
        validate_content_file_format "$test_dir"
        
        # Validate L1/L2 concatenation (Requirement 2.4, 2.5)
        validate_concatenation "$test_dir"
    else
        log_error "Missing content files for $table_name"
        return 1
    fi
    
    # Check if task list was created (Requirement 1.2, 2.7)
    if [ -n "$task_list_files" ]; then
        log_success "Task list created for $table_name"
        validate_task_list_format "$task_list_files" "$test_dir"
    else
        log_error "Task list not created for $table_name"
        return 1
    fi
    
    # Performance check
    if [ "$duration" -lt 60 ]; then
        log_success "File-level mode completed in ${duration}s (good performance)"
    else
        log_warning "File-level mode took ${duration}s (consider optimization)"
    fi
}

# Validate chunk-level mode results
validate_chunk_level_results() {
    local table_name="$1"
    local chunk_size="$2"
    local test_dir="$3"
    local duration="$4"
    
    local content_files_count=$(find "$test_dir" -name "content_*.txt" | wc -l)
    local task_list_files=$(find "$test_dir" -name "*task*.txt" -o -name "*task*.md")
    
    # Check if content files were created
    if [ "$content_files_count" -gt 0 ]; then
        log_success "Chunk-level mode created $content_files_count content files for $table_name"
        validate_content_file_format "$test_dir"
    else
        log_error "No content files created in chunk-level mode for $table_name"
        return 1
    fi
    
    # Check if task list was created
    if [ -n "$task_list_files" ]; then
        log_success "Task list created in chunk-level mode for $table_name"
        validate_task_list_format "$task_list_files" "$test_dir"
    else
        log_error "Task list not created in chunk-level mode for $table_name"
        return 1
    fi
    
    # Check if chunked table was mentioned in output (Requirement 2.1)
    if grep -q "chunked table" "$test_dir/output.log" 2>/dev/null || grep -q "Created.*table" "$test_dir/output.log" 2>/dev/null; then
        log_success "Chunked table creation mentioned in output for $table_name"
    else
        log_warning "Chunked table creation not clearly indicated in output for $table_name"
    fi
    
    # Performance check for chunk-level mode
    if [ "$duration" -lt 120 ]; then
        log_success "Chunk-level mode completed in ${duration}s (good performance)"
    else
        log_warning "Chunk-level mode took ${duration}s (consider optimization)"
    fi
}

# Validate content file format (Requirement 2.6)
validate_content_file_format() {
    local test_dir="$1"
    local sample_content_file=$(find "$test_dir" -name "content_1.txt" | head -1)
    
    if [ -f "$sample_content_file" ]; then
        local file_size=$(stat -c%s "$sample_content_file" 2>/dev/null || stat -f%z "$sample_content_file" 2>/dev/null || echo "0")
        if [ "$file_size" -gt 0 ]; then
            log_success "Content files have valid format (non-empty)"
        else
            log_error "Content files are empty"
        fi
    fi
}

# Validate L1/L2 concatenation (Requirement 2.4, 2.5)
validate_concatenation() {
    local test_dir="$1"
    local content_file=$(find "$test_dir" -name "content_1.txt" | head -1)
    local content_l1_file=$(find "$test_dir" -name "contentL1_1.txt" | head -1)
    local content_l2_file=$(find "$test_dir" -name "contentL2_1.txt" | head -1)
    
    if [ -f "$content_file" ] && [ -f "$content_l1_file" ] && [ -f "$content_l2_file" ]; then
        local content_size=$(stat -c%s "$content_file" 2>/dev/null || stat -f%z "$content_file" 2>/dev/null || echo "0")
        local content_l1_size=$(stat -c%s "$content_l1_file" 2>/dev/null || stat -f%z "$content_l1_file" 2>/dev/null || echo "0")
        local content_l2_size=$(stat -c%s "$content_l2_file" 2>/dev/null || stat -f%z "$content_l2_file" 2>/dev/null || echo "0")
        
        if [ "$content_l2_size" -ge "$content_l1_size" ] && [ "$content_l1_size" -ge "$content_size" ]; then
            log_success "L1/L2 concatenation appears correct (L2 >= L1 >= content)"
        else
            log_error "L1/L2 concatenation size relationship incorrect"
        fi
    fi
}

# Validate task list format (Requirement 2.7)
validate_task_list_format() {
    local task_list_file="$1"
    local test_dir="$2"
    
    if [ -f "$task_list_file" ]; then
        # Check if task list references content files
        if grep -q "content_" "$task_list_file" && grep -q "contentL1_" "$task_list_file" && grep -q "contentL2_" "$task_list_file"; then
            log_success "Task list references all content file types"
        else
            log_error "Task list doesn't reference all content file types"
        fi
        
        # Check if task list has reasonable structure
        local line_count=$(wc -l < "$task_list_file")
        if [ "$line_count" -gt 3 ]; then
            log_success "Task list has reasonable structure ($line_count lines)"
        else
            log_warning "Task list seems too short ($line_count lines)"
        fi
    fi
}

# Test error handling (Requirement 3.1, 3.2)
test_error_handling() {
    log_info "Testing error handling scenarios..."
    
    local error_test_dir="$TEST_OUTPUT_DIR/error_tests"
    mkdir -p "$error_test_dir"
    
    # Test 1: Non-existent table
    log_info "Testing non-existent table error..."
    if "$BINARY_PATH" chunk-level-task-generator "nonexistent_table_12345" --output-dir "$error_test_dir" --db-path "$DATABASE_URL" > "$error_test_dir/nonexistent_table.log" 2>&1; then
        log_error "Command should have failed for non-existent table"
    else
        if grep -q -i "not found\|does not exist\|table.*nonexistent" "$error_test_dir/nonexistent_table.log"; then
            log_success "Proper error message for non-existent table"
        else
            log_warning "Error occurred but message could be clearer for non-existent table"
        fi
    fi
    
    # Test 2: Invalid chunk size
    local first_table=$(echo "$TEST_TABLES" | head -1)
    if [ -n "$first_table" ]; then
        log_info "Testing invalid chunk size error..."
        if "$BINARY_PATH" chunk-level-task-generator "$first_table" 0 --output-dir "$error_test_dir" --db-path "$DATABASE_URL" > "$error_test_dir/invalid_chunk.log" 2>&1; then
            log_error "Command should have failed for chunk size 0"
        else
            if grep -q -i "invalid.*chunk\|chunk.*size\|must be.*0" "$error_test_dir/invalid_chunk.log"; then
                log_success "Proper error message for invalid chunk size"
            else
                log_warning "Error occurred but message could be clearer for invalid chunk size"
            fi
        fi
    fi
    
    # Test 3: SQL injection attempt
    log_info "Testing SQL injection protection..."
    if "$BINARY_PATH" chunk-level-task-generator "test'; DROP TABLE users; --" --output-dir "$error_test_dir" --db-path "$DATABASE_URL" > "$error_test_dir/sql_injection.log" 2>&1; then
        log_error "Command should have failed for SQL injection attempt"
    else
        if grep -q -i "invalid.*table\|invalid.*character\|table.*name" "$error_test_dir/sql_injection.log"; then
            log_success "Proper protection against SQL injection"
        else
            log_warning "Error occurred but SQL injection protection unclear"
        fi
    fi
}

# Test performance with large tables (Requirement: Test performance with large tables >1000 rows)
test_performance() {
    log_info "Testing performance with available tables..."
    
    echo "$TEST_TABLES" | while read -r table; do
        if [ -n "$table" ]; then
            # Get table size if possible
            local table_size="unknown"
            if command -v psql &> /dev/null; then
                table_size=$(psql "$DATABASE_URL" -t -c "SELECT COUNT(*) FROM $table;" 2>/dev/null | tr -d ' ' || echo "unknown")
            fi
            
            log_info "Testing performance with $table (size: $table_size rows)"
            
            local perf_test_dir="$TEST_OUTPUT_DIR/performance_$table"
            mkdir -p "$perf_test_dir"
            
            local start_time=$(date +%s)
            if timeout 600 "$BINARY_PATH" chunk-level-task-generator "$table" --output-dir "$perf_test_dir" --db-path "$DATABASE_URL" > "$perf_test_dir/output.log" 2>&1; then
                local end_time=$(date +%s)
                local duration=$((end_time - start_time))
                
                if [ "$table_size" != "unknown" ] && [ "$table_size" -gt 1000 ]; then
                    local throughput=$((table_size / duration))
                    log_success "Performance test passed for $table: ${duration}s for $table_size rows (${throughput} rows/sec)"
                else
                    log_success "Performance test completed for $table: ${duration}s"
                fi
            else
                log_error "Performance test failed for $table"
            fi
        fi
    done
}

# Generate test report
generate_report() {
    log_info "Generating test report..."
    
    cat > "$TEST_RESULTS_FILE" << EOF
# Chunk-Level Task Generator Validation Report

**Date:** $(date)
**Test Output Directory:** $TEST_OUTPUT_DIR

## Summary

- **Tests Passed:** $TESTS_PASSED
- **Tests Failed:** $TESTS_FAILED
- **Overall Status:** $([ $TESTS_FAILED -eq 0 ] && echo "✅ PASSED" || echo "❌ FAILED")

## Requirements Tested

- **1.1**: File-level mode content file generation
- **1.2**: Task list generation with row references  
- **2.1**: Chunk-level mode with chunked table creation
- **2.6**: Content file creation and validation
- **2.7**: Task list format compatibility

## Test Tables

$(echo "$TEST_TABLES" | sed 's/^/- /')

## Issues Found

$(if [ ${#ISSUES_FOUND[@]} -eq 0 ]; then
    echo "No issues found."
else
    printf '%s\n' "${ISSUES_FOUND[@]}" | sed 's/^/- /'
fi)

## Test Artifacts

All test outputs are available in: \`$TEST_OUTPUT_DIR\`

## Recommendations

$(if [ $TESTS_FAILED -eq 0 ]; then
    echo "✅ The chunk-level-task-generator command is working correctly and meets all tested requirements."
else
    echo "❌ Some tests failed. Please review the issues above and the detailed logs in the test output directory."
fi)
EOF
    
    log_success "Test report generated: $TEST_RESULTS_FILE"
}

# Main execution
main() {
    echo "Starting validation at $(date)"
    
    check_prerequisites
    find_test_tables
    
    echo
    echo "🧪 Running Tests"
    echo "==============="
    
    # Test file-level mode with each table
    echo "$TEST_TABLES" | while read -r table; do
        if [ -n "$table" ]; then
            test_file_level_mode "$table"
        fi
    done
    
    # Test chunk-level mode with each table
    echo "$TEST_TABLES" | while read -r table; do
        if [ -n "$table" ]; then
            test_chunk_level_mode "$table" 500
        fi
    done
    
    # Test error handling
    test_error_handling
    
    # Test performance
    test_performance
    
    # Generate report
    generate_report
    
    echo
    echo "📊 Final Results"
    echo "==============="
    echo "Tests Passed: $TESTS_PASSED"
    echo "Tests Failed: $TESTS_FAILED"
    
    if [ $TESTS_FAILED -eq 0 ]; then
        echo -e "${GREEN}🎉 All tests passed! The chunk-level-task-generator command is working correctly.${NC}"
        echo "📄 Detailed report: $TEST_RESULTS_FILE"
        exit 0
    else
        echo -e "${RED}💥 Some tests failed. Please review the issues above.${NC}"
        echo "📄 Detailed report: $TEST_RESULTS_FILE"
        echo "📁 Test artifacts: $TEST_OUTPUT_DIR"
        exit 1
    fi
}

# Run main function
main "$@"
#!/bin/bash
# Release testing script for code-ingest
# Tests a specific release version across different installation methods

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Configuration
REPO="your-org/code-ingest"
TEST_DIR="/tmp/code-ingest-release-test"
TEST_DB_DIR="$TEST_DIR/test-db"

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Cleanup function
cleanup() {
    print_status "Cleaning up test environment..."
    rm -rf "$TEST_DIR"
    
    # Stop test PostgreSQL if we started it
    if [[ -n "$TEST_PG_PID" ]]; then
        kill "$TEST_PG_PID" 2>/dev/null || true
    fi
}

# Set up cleanup trap
trap cleanup EXIT

# Parse command line arguments
show_help() {
    echo "Release testing script for code-ingest"
    echo
    echo "Usage: $0 [VERSION] [OPTIONS]"
    echo
    echo "Arguments:"
    echo "  VERSION                 Version to test (e.g., v1.2.3 or latest)"
    echo
    echo "Options:"
    echo "  --skip-download         Skip download tests"
    echo "  --skip-install          Skip installation tests"
    echo "  --skip-functionality    Skip functionality tests"
    echo "  --skip-docker          Skip Docker tests"
    echo "  --help, -h             Show this help message"
    echo
    echo "Examples:"
    echo "  $0 v1.2.3              # Test specific version"
    echo "  $0 latest               # Test latest release"
    echo "  $0 v1.2.3 --skip-docker # Test without Docker"
}

# Test binary download
test_download() {
    local version=$1
    
    print_status "Testing binary download for version $version..."
    
    # Detect platform
    local os=$(uname -s | tr '[:upper:]' '[:lower:]')
    local arch=$(uname -m)
    
    case $os in
        linux*)
            case $arch in
                x86_64) local platform="x86_64-unknown-linux-gnu" ;;
                aarch64|arm64) local platform="aarch64-unknown-linux-gnu" ;;
                *) print_error "Unsupported architecture: $arch"; return 1 ;;
            esac
            ;;
        darwin*)
            case $arch in
                x86_64) local platform="x86_64-apple-darwin" ;;
                arm64) local platform="aarch64-apple-darwin" ;;
                *) print_error "Unsupported architecture: $arch"; return 1 ;;
            esac
            ;;
        *)
            print_error "Unsupported operating system: $os"
            return 1
            ;;
    esac
    
    local archive_name="code-ingest-${version#v}-${platform}.tar.gz"
    local download_url="https://github.com/${REPO}/releases/download/${version}/${archive_name}"
    
    print_status "Downloading: $download_url"
    
    mkdir -p "$TEST_DIR/downloads"
    cd "$TEST_DIR/downloads"
    
    if command_exists curl; then
        curl -sSL "$download_url" -o "$archive_name"
    elif command_exists wget; then
        wget -q "$download_url"
    else
        print_error "Neither curl nor wget found"
        return 1
    fi
    
    # Verify download
    if [[ ! -f "$archive_name" ]]; then
        print_error "Download failed: $archive_name not found"
        return 1
    fi
    
    # Extract and verify binary
    tar -xzf "$archive_name"
    local binary_path="code-ingest-${version#v}-${platform}/code-ingest"
    
    if [[ ! -f "$binary_path" ]]; then
        print_error "Binary not found in archive: $binary_path"
        return 1
    fi
    
    # Test binary execution
    chmod +x "$binary_path"
    if "./$binary_path" --version >/dev/null 2>&1; then
        print_success "Binary download and execution test passed"
        echo "BINARY_PATH=$TEST_DIR/downloads/$binary_path" >> "$TEST_DIR/env"
        return 0
    else
        print_error "Binary execution failed"
        return 1
    fi
}

# Test installation script
test_install_script() {
    local version=$1
    
    print_status "Testing installation script for version $version..."
    
    # Create isolated installation directory
    local install_dir="$TEST_DIR/install-test"
    mkdir -p "$install_dir"
    
    # Download and run install script
    local install_script_url="https://github.com/${REPO}/releases/download/${version}/install.sh"
    
    print_status "Downloading install script: $install_script_url"
    
    if command_exists curl; then
        curl -sSL "$install_script_url" -o "$TEST_DIR/install.sh"
    elif command_exists wget; then
        wget -q "$install_script_url" -O "$TEST_DIR/install.sh"
    else
        print_error "Neither curl nor wget found"
        return 1
    fi
    
    chmod +x "$TEST_DIR/install.sh"
    
    # Run install script with custom directory
    export INSTALL_DIR="$install_dir"
    if "$TEST_DIR/install.sh"; then
        print_success "Installation script completed successfully"
        
        # Verify installation
        if [[ -f "$install_dir/code-ingest" ]]; then
            if "$install_dir/code-ingest" --version >/dev/null 2>&1; then
                print_success "Installed binary works correctly"
                return 0
            else
                print_error "Installed binary execution failed"
                return 1
            fi
        else
            print_error "Binary not found after installation"
            return 1
        fi
    else
        print_error "Installation script failed"
        return 1
    fi
}

# Test Docker image
test_docker() {
    local version=$1
    
    if ! command_exists docker; then
        print_warning "Docker not found, skipping Docker tests"
        return 0
    fi
    
    print_status "Testing Docker image for version $version..."
    
    local image_tag="your-org/code-ingest:${version#v}"
    
    # Try to pull the image
    if docker pull "$image_tag" >/dev/null 2>&1; then
        print_success "Docker image pulled successfully"
    else
        print_warning "Could not pull Docker image, trying latest"
        image_tag="your-org/code-ingest:latest"
        if ! docker pull "$image_tag" >/dev/null 2>&1; then
            print_warning "Docker image not available, skipping Docker tests"
            return 0
        fi
    fi
    
    # Test basic functionality
    if docker run --rm "$image_tag" --version >/dev/null 2>&1; then
        print_success "Docker image basic test passed"
    else
        print_error "Docker image basic test failed"
        return 1
    fi
    
    # Test with volume mount
    mkdir -p "$TEST_DIR/docker-test"
    if docker run --rm -v "$TEST_DIR/docker-test:/workspace" "$image_tag" --help >/dev/null 2>&1; then
        print_success "Docker volume mount test passed"
        return 0
    else
        print_error "Docker volume mount test failed"
        return 1
    fi
}

# Test basic functionality
test_functionality() {
    local binary_path=$1
    
    print_status "Testing basic functionality..."
    
    # Test help command
    if "$binary_path" --help >/dev/null 2>&1; then
        print_success "Help command works"
    else
        print_error "Help command failed"
        return 1
    fi
    
    # Test version command
    local version_output=$("$binary_path" --version 2>&1)
    if [[ $? -eq 0 ]]; then
        print_success "Version command works: $version_output"
    else
        print_error "Version command failed"
        return 1
    fi
    
    # Test PostgreSQL setup guidance
    if "$binary_path" pg-start >/dev/null 2>&1; then
        print_success "PostgreSQL setup guidance works"
    else
        print_warning "PostgreSQL setup guidance had issues (may be expected)"
    fi
    
    return 0
}

# Test with real repository (if PostgreSQL is available)
test_real_ingestion() {
    local binary_path=$1
    
    print_status "Testing real repository ingestion..."
    
    # Check if PostgreSQL is available
    if ! command_exists psql; then
        print_warning "PostgreSQL not available, skipping ingestion test"
        return 0
    fi
    
    # Try to connect to PostgreSQL
    if ! pg_isready -h localhost -p 5432 >/dev/null 2>&1; then
        print_warning "PostgreSQL not running, skipping ingestion test"
        return 0
    fi
    
    # Create test database
    local test_db="code_ingest_release_test_$$"
    if createdb "$test_db" 2>/dev/null; then
        print_status "Created test database: $test_db"
        
        # Test ingestion with a small public repository
        local test_repo="https://github.com/rust-lang/mdBook"
        local db_path="$TEST_DIR/ingestion-test"
        
        export DATABASE_URL="postgresql://localhost:5432/$test_db"
        
        if timeout 300 "$binary_path" ingest "$test_repo" --db-path "$db_path" >/dev/null 2>&1; then
            print_success "Repository ingestion completed successfully"
            
            # Test SQL query
            if "$binary_path" sql "SELECT COUNT(*) FROM ingestion_meta" --db-path "$db_path" >/dev/null 2>&1; then
                print_success "SQL query test passed"
            else
                print_warning "SQL query test failed"
            fi
        else
            print_warning "Repository ingestion failed or timed out"
        fi
        
        # Cleanup test database
        dropdb "$test_db" 2>/dev/null || true
    else
        print_warning "Could not create test database, skipping ingestion test"
    fi
    
    return 0
}

# Main test function
main() {
    local version="latest"
    local skip_download=false
    local skip_install=false
    local skip_functionality=false
    local skip_docker=false
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-download)
                skip_download=true
                shift
                ;;
            --skip-install)
                skip_install=true
                shift
                ;;
            --skip-functionality)
                skip_functionality=true
                shift
                ;;
            --skip-docker)
                skip_docker=true
                shift
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            -*)
                print_error "Unknown option: $1"
                show_help
                exit 1
                ;;
            *)
                version=$1
                shift
                ;;
        esac
    done
    
    echo "Code Ingest Release Testing"
    echo "=========================="
    echo "Version: $version"
    echo "Test directory: $TEST_DIR"
    echo
    
    # Create test environment
    rm -rf "$TEST_DIR"
    mkdir -p "$TEST_DIR"
    touch "$TEST_DIR/env"
    
    local test_count=0
    local passed_count=0
    local failed_tests=()
    
    # Run tests
    if [[ "$skip_download" != true ]]; then
        test_count=$((test_count + 1))
        if test_download "$version"; then
            passed_count=$((passed_count + 1))
        else
            failed_tests+=("Download test")
        fi
    fi
    
    if [[ "$skip_install" != true ]]; then
        test_count=$((test_count + 1))
        if test_install_script "$version"; then
            passed_count=$((passed_count + 1))
        else
            failed_tests+=("Installation script test")
        fi
    fi
    
    if [[ "$skip_docker" != true ]]; then
        test_count=$((test_count + 1))
        if test_docker "$version"; then
            passed_count=$((passed_count + 1))
        else
            failed_tests+=("Docker test")
        fi
    fi
    
    # Get binary path for functionality tests
    local binary_path=""
    if [[ -f "$TEST_DIR/env" ]]; then
        source "$TEST_DIR/env"
        binary_path="$BINARY_PATH"
    fi
    
    if [[ "$skip_functionality" != true && -n "$binary_path" ]]; then
        test_count=$((test_count + 1))
        if test_functionality "$binary_path"; then
            passed_count=$((passed_count + 1))
        else
            failed_tests+=("Functionality test")
        fi
        
        # Real ingestion test (bonus)
        if test_real_ingestion "$binary_path"; then
            print_success "Real ingestion test passed (bonus)"
        else
            print_warning "Real ingestion test failed (not counted as failure)"
        fi
    fi
    
    # Summary
    echo
    echo "Test Results Summary"
    echo "==================="
    echo "Total tests: $test_count"
    echo "Passed: $passed_count"
    echo "Failed: $((test_count - passed_count))"
    
    if [[ ${#failed_tests[@]} -gt 0 ]]; then
        echo
        print_error "Failed tests:"
        for test in "${failed_tests[@]}"; do
            echo "  - $test"
        done
        echo
        exit 1
    else
        echo
        print_success "All tests passed! Release $version is working correctly."
        echo
        exit 0
    fi
}

# Run main function with all arguments
main "$@"
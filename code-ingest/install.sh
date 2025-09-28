#!/bin/bash
# Code Ingest Installation Script
# Installs code-ingest and sets up the development environment

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

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Detect OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
        echo "windows"
    else
        echo "unknown"
    fi
}

# Install Rust if not present
install_rust() {
    if command_exists rustc; then
        print_success "Rust is already installed ($(rustc --version))"
        return 0
    fi

    print_status "Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source ~/.cargo/env
    print_success "Rust installed successfully"
}

# Install PostgreSQL
install_postgresql() {
    local os=$(detect_os)
    
    if command_exists psql; then
        print_success "PostgreSQL is already installed ($(psql --version | head -n1))"
        return 0
    fi

    print_status "Installing PostgreSQL..."
    
    case $os in
        "linux")
            if command_exists apt-get; then
                sudo apt-get update
                sudo apt-get install -y postgresql postgresql-contrib libpq-dev
                sudo systemctl start postgresql
                sudo systemctl enable postgresql
            elif command_exists yum; then
                sudo yum install -y postgresql-server postgresql-contrib postgresql-devel
                sudo postgresql-setup initdb
                sudo systemctl start postgresql
                sudo systemctl enable postgresql
            else
                print_error "Unsupported Linux distribution. Please install PostgreSQL manually."
                return 1
            fi
            ;;
        "macos")
            if command_exists brew; then
                brew install postgresql@15
                brew services start postgresql
            else
                print_error "Homebrew not found. Please install PostgreSQL manually or install Homebrew first."
                return 1
            fi
            ;;
        "windows")
            print_warning "Please download and install PostgreSQL from https://www.postgresql.org/download/windows/"
            return 1
            ;;
        *)
            print_error "Unsupported operating system. Please install PostgreSQL manually."
            return 1
            ;;
    esac
    
    print_success "PostgreSQL installed successfully"
}

# Install optional dependencies
install_optional_deps() {
    local os=$(detect_os)
    
    print_status "Installing optional dependencies for file conversion..."
    
    case $os in
        "linux")
            if command_exists apt-get; then
                sudo apt-get install -y poppler-utils pandoc
            elif command_exists yum; then
                sudo yum install -y poppler-utils pandoc
            fi
            ;;
        "macos")
            if command_exists brew; then
                brew install poppler pandoc
            fi
            ;;
        *)
            print_warning "Skipping optional dependencies for unsupported OS"
            ;;
    esac
}

# Install code-ingest
install_code_ingest() {
    print_status "Installing code-ingest..."
    
    if [[ -f "Cargo.toml" ]]; then
        # Install from source
        print_status "Installing from source..."
        cargo install --path .
    else
        # Install from crates.io
        print_status "Installing from crates.io..."
        cargo install code-ingest
    fi
    
    print_success "code-ingest installed successfully"
}

# Setup database
setup_database() {
    print_status "Setting up database..."
    
    # Check if PostgreSQL is running
    if ! pg_isready -h localhost -p 5432 >/dev/null 2>&1; then
        print_warning "PostgreSQL is not running. Please start it manually."
        return 1
    fi
    
    # Create database if it doesn't exist
    if ! psql -h localhost -U postgres -lqt | cut -d \| -f 1 | grep -qw code_analysis; then
        print_status "Creating code_analysis database..."
        createdb code_analysis 2>/dev/null || {
            print_warning "Could not create database. You may need to create it manually:"
            echo "  createdb code_analysis"
        }
    else
        print_success "Database code_analysis already exists"
    fi
}

# Verify installation
verify_installation() {
    print_status "Verifying installation..."
    
    if command_exists code-ingest; then
        print_success "code-ingest is installed: $(code-ingest --version)"
    else
        print_error "code-ingest installation failed"
        return 1
    fi
    
    if pg_isready -h localhost -p 5432 >/dev/null 2>&1; then
        print_success "PostgreSQL is running and accessible"
    else
        print_warning "PostgreSQL is not accessible. Please check your installation."
    fi
    
    # Test basic functionality
    print_status "Testing basic functionality..."
    if code-ingest --help >/dev/null 2>&1; then
        print_success "code-ingest is working correctly"
    else
        print_error "code-ingest is not working properly"
        return 1
    fi
}

# Print usage instructions
print_usage() {
    echo
    print_success "Installation completed successfully!"
    echo
    echo "Next steps:"
    echo "1. Set up your database connection (if needed):"
    echo "   export DATABASE_URL=\"postgresql://username:password@localhost:5432/code_analysis\""
    echo
    echo "2. Test the installation:"
    echo "   code-ingest pg-start"
    echo
    echo "3. Try ingesting a repository:"
    echo "   code-ingest ingest https://github.com/rust-lang/mdBook --db-path ./analysis"
    echo
    echo "4. Read the documentation:"
    echo "   - README.md for getting started"
    echo "   - docs/CLI_REFERENCE.md for command reference"
    echo "   - examples/ directory for usage examples"
    echo
    echo "For help and support:"
    echo "   - GitHub: https://github.com/your-org/code-ingest"
    echo "   - Documentation: https://docs.code-ingest.dev"
    echo
}

# Main installation function
main() {
    echo "Code Ingest Installation Script"
    echo "=============================="
    echo
    
    # Check for help flag
    if [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
        echo "Usage: $0 [OPTIONS]"
        echo
        echo "Options:"
        echo "  --skip-rust         Skip Rust installation"
        echo "  --skip-postgresql   Skip PostgreSQL installation"
        echo "  --skip-optional     Skip optional dependencies"
        echo "  --help, -h          Show this help message"
        echo
        exit 0
    fi
    
    # Parse command line arguments
    SKIP_RUST=false
    SKIP_POSTGRESQL=false
    SKIP_OPTIONAL=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-rust)
                SKIP_RUST=true
                shift
                ;;
            --skip-postgresql)
                SKIP_POSTGRESQL=true
                shift
                ;;
            --skip-optional)
                SKIP_OPTIONAL=true
                shift
                ;;
            *)
                print_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Run installation steps
    if [[ "$SKIP_RUST" != true ]]; then
        install_rust || exit 1
    fi
    
    if [[ "$SKIP_POSTGRESQL" != true ]]; then
        install_postgresql || print_warning "PostgreSQL installation failed, continuing anyway..."
    fi
    
    if [[ "$SKIP_OPTIONAL" != true ]]; then
        install_optional_deps || print_warning "Optional dependencies installation failed, continuing anyway..."
    fi
    
    install_code_ingest || exit 1
    
    if [[ "$SKIP_POSTGRESQL" != true ]]; then
        setup_database || print_warning "Database setup failed, you may need to set it up manually"
    fi
    
    verify_installation || exit 1
    
    print_usage
}

# Run main function with all arguments
main "$@"
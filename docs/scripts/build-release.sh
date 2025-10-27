#!/bin/bash
# Cross-platform release build script for code-ingest
# Builds optimized binaries for multiple platforms and creates distribution packages

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
PROJECT_NAME="code-ingest"
CARGO_MANIFEST="code-ingest/Cargo.toml"
BUILD_DIR="target/release-builds"
DIST_DIR="target/dist"

# Get version from Cargo.toml
get_version() {
    grep '^version = ' "$CARGO_MANIFEST" | sed 's/version = "\(.*\)"/\1/'
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Install cross-compilation targets
install_targets() {
    print_status "Installing cross-compilation targets..."
    
    local targets=(
        "x86_64-unknown-linux-gnu"
        "x86_64-unknown-linux-musl"
        "aarch64-unknown-linux-gnu"
        "x86_64-apple-darwin"
        "aarch64-apple-darwin"
        "x86_64-pc-windows-gnu"
    )
    
    for target in "${targets[@]}"; do
        print_status "Installing target: $target"
        rustup target add "$target" || print_warning "Failed to install target: $target"
    done
}

# Build for a specific target
build_target() {
    local target=$1
    local features=${2:-""}
    
    print_status "Building for target: $target"
    
    cd code-ingest
    
    local cargo_cmd="cargo build --release --target $target"
    if [[ -n "$features" ]]; then
        cargo_cmd="$cargo_cmd --features $features"
    fi
    
    if $cargo_cmd; then
        print_success "Built successfully for $target"
        return 0
    else
        print_error "Build failed for $target"
        return 1
    fi
    
    cd ..
}

# Create distribution archive
create_archive() {
    local target=$1
    local version=$2
    local binary_name="code-ingest"
    local archive_name="${PROJECT_NAME}-${version}-${target}"
    
    # Determine binary extension
    local binary_ext=""
    if [[ "$target" == *"windows"* ]]; then
        binary_ext=".exe"
    fi
    
    local binary_path="code-ingest/target/${target}/release/${binary_name}${binary_ext}"
    
    if [[ ! -f "$binary_path" ]]; then
        print_error "Binary not found: $binary_path"
        return 1
    fi
    
    print_status "Creating archive for $target..."
    
    # Create temporary directory for archive contents
    local temp_dir=$(mktemp -d)
    local archive_dir="$temp_dir/$archive_name"
    mkdir -p "$archive_dir"
    
    # Copy binary
    cp "$binary_path" "$archive_dir/${binary_name}${binary_ext}"
    
    # Copy documentation
    cp code-ingest/README.md "$archive_dir/"
    cp code-ingest/docs/CLI_REFERENCE.md "$archive_dir/" 2>/dev/null || true
    cp LICENSE* "$archive_dir/" 2>/dev/null || true
    
    # Create install script for Unix-like systems
    if [[ "$target" != *"windows"* ]]; then
        cat > "$archive_dir/install.sh" << 'EOF'
#!/bin/bash
# Installation script for code-ingest

set -e

INSTALL_DIR="${INSTALL_DIR:-/usr/local/bin}"
BINARY_NAME="code-ingest"

echo "Installing code-ingest to $INSTALL_DIR..."

# Check if we have write permissions
if [[ ! -w "$INSTALL_DIR" ]]; then
    echo "Error: No write permission to $INSTALL_DIR"
    echo "Try running with sudo or set INSTALL_DIR to a writable directory"
    exit 1
fi

# Copy binary
cp "$BINARY_NAME" "$INSTALL_DIR/"
chmod +x "$INSTALL_DIR/$BINARY_NAME"

echo "Installation completed successfully!"
echo "Run 'code-ingest --help' to get started"
EOF
        chmod +x "$archive_dir/install.sh"
    fi
    
    # Create archive
    mkdir -p "$DIST_DIR"
    
    if [[ "$target" == *"windows"* ]]; then
        # Create ZIP for Windows
        (cd "$temp_dir" && zip -r "$archive_name.zip" "$archive_name")
        mv "$temp_dir/$archive_name.zip" "$DIST_DIR/"
        print_success "Created: $DIST_DIR/$archive_name.zip"
    else
        # Create tar.gz for Unix-like systems
        (cd "$temp_dir" && tar -czf "$archive_name.tar.gz" "$archive_name")
        mv "$temp_dir/$archive_name.tar.gz" "$DIST_DIR/"
        print_success "Created: $DIST_DIR/$archive_name.tar.gz"
    fi
    
    # Cleanup
    rm -rf "$temp_dir"
}

# Create checksums file
create_checksums() {
    print_status "Creating checksums..."
    
    cd "$DIST_DIR"
    
    # Create SHA256 checksums
    if command_exists sha256sum; then
        sha256sum *.tar.gz *.zip 2>/dev/null > SHA256SUMS || true
    elif command_exists shasum; then
        shasum -a 256 *.tar.gz *.zip 2>/dev/null > SHA256SUMS || true
    fi
    
    # Create MD5 checksums for compatibility
    if command_exists md5sum; then
        md5sum *.tar.gz *.zip 2>/dev/null > MD5SUMS || true
    elif command_exists md5; then
        md5 *.tar.gz *.zip 2>/dev/null > MD5SUMS || true
    fi
    
    cd - > /dev/null
    
    if [[ -f "$DIST_DIR/SHA256SUMS" ]]; then
        print_success "Created checksums: $DIST_DIR/SHA256SUMS"
    fi
}

# Build Debian package
build_deb() {
    if ! command_exists cargo-deb; then
        print_status "Installing cargo-deb..."
        cargo install cargo-deb
    fi
    
    print_status "Building Debian package..."
    
    cd code-ingest
    
    if cargo deb --target x86_64-unknown-linux-gnu; then
        # Move .deb file to dist directory
        local deb_file=$(find target/x86_64-unknown-linux-gnu/debian -name "*.deb" | head -n1)
        if [[ -n "$deb_file" ]]; then
            mkdir -p "../$DIST_DIR"
            cp "$deb_file" "../$DIST_DIR/"
            print_success "Created Debian package: $(basename "$deb_file")"
        fi
    else
        print_warning "Failed to build Debian package"
    fi
    
    cd ..
}

# Build RPM package
build_rpm() {
    if ! command_exists cargo-generate-rpm; then
        print_status "Installing cargo-generate-rpm..."
        cargo install cargo-generate-rpm
    fi
    
    print_status "Building RPM package..."
    
    cd code-ingest
    
    if cargo generate-rpm --target x86_64-unknown-linux-gnu; then
        # Move .rpm file to dist directory
        local rpm_file=$(find target/generate-rpm -name "*.rpm" | head -n1)
        if [[ -n "$rpm_file" ]]; then
            mkdir -p "../$DIST_DIR"
            cp "$rpm_file" "../$DIST_DIR/"
            print_success "Created RPM package: $(basename "$rpm_file")"
        fi
    else
        print_warning "Failed to build RPM package"
    fi
    
    cd ..
}

# Create Homebrew formula
create_homebrew_formula() {
    local version=$1
    local sha256_macos_intel=""
    local sha256_macos_arm=""
    
    # Calculate SHA256 for macOS binaries
    if [[ -f "$DIST_DIR/${PROJECT_NAME}-${version}-x86_64-apple-darwin.tar.gz" ]]; then
        if command_exists sha256sum; then
            sha256_macos_intel=$(sha256sum "$DIST_DIR/${PROJECT_NAME}-${version}-x86_64-apple-darwin.tar.gz" | cut -d' ' -f1)
        elif command_exists shasum; then
            sha256_macos_intel=$(shasum -a 256 "$DIST_DIR/${PROJECT_NAME}-${version}-x86_64-apple-darwin.tar.gz" | cut -d' ' -f1)
        fi
    fi
    
    if [[ -f "$DIST_DIR/${PROJECT_NAME}-${version}-aarch64-apple-darwin.tar.gz" ]]; then
        if command_exists sha256sum; then
            sha256_macos_arm=$(sha256sum "$DIST_DIR/${PROJECT_NAME}-${version}-aarch64-apple-darwin.tar.gz" | cut -d' ' -f1)
        elif command_exists shasum; then
            sha256_macos_arm=$(shasum -a 256 "$DIST_DIR/${PROJECT_NAME}-${version}-aarch64-apple-darwin.tar.gz" | cut -d' ' -f1)
        fi
    fi
    
    print_status "Creating Homebrew formula..."
    
    mkdir -p "$DIST_DIR/homebrew"
    
    cat > "$DIST_DIR/homebrew/code-ingest.rb" << EOF
class CodeIngest < Formula
  desc "High-performance tool for ingesting GitHub repositories into PostgreSQL"
  homepage "https://github.com/your-org/code-ingest"
  version "$version"
  license "MIT OR Apache-2.0"

  if Hardware::CPU.intel?
    url "https://github.com/your-org/code-ingest/releases/download/v#{version}/code-ingest-#{version}-x86_64-apple-darwin.tar.gz"
    sha256 "$sha256_macos_intel"
  elsif Hardware::CPU.arm?
    url "https://github.com/your-org/code-ingest/releases/download/v#{version}/code-ingest-#{version}-aarch64-apple-darwin.tar.gz"
    sha256 "$sha256_macos_arm"
  end

  depends_on "postgresql"

  def install
    bin.install "code-ingest"
  end

  test do
    system "#{bin}/code-ingest", "--version"
  end
end
EOF
    
    print_success "Created Homebrew formula: $DIST_DIR/homebrew/code-ingest.rb"
}

# Main build function
main() {
    echo "Code Ingest Release Builder"
    echo "=========================="
    echo
    
    # Parse command line arguments
    local build_all=true
    local build_packages=false
    local targets=()
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --target)
                targets+=("$2")
                build_all=false
                shift 2
                ;;
            --packages)
                build_packages=true
                shift
                ;;
            --help|-h)
                echo "Usage: $0 [OPTIONS]"
                echo
                echo "Options:"
                echo "  --target TARGET     Build for specific target (can be used multiple times)"
                echo "  --packages          Build distribution packages (deb, rpm, homebrew)"
                echo "  --help, -h          Show this help message"
                echo
                echo "Available targets:"
                echo "  x86_64-unknown-linux-gnu"
                echo "  x86_64-unknown-linux-musl"
                echo "  aarch64-unknown-linux-gnu"
                echo "  x86_64-apple-darwin"
                echo "  aarch64-apple-darwin"
                echo "  x86_64-pc-windows-gnu"
                echo
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Get version
    local version=$(get_version)
    print_status "Building version: $version"
    
    # Clean previous builds
    print_status "Cleaning previous builds..."
    rm -rf "$BUILD_DIR" "$DIST_DIR"
    mkdir -p "$BUILD_DIR" "$DIST_DIR"
    
    # Install targets if building all
    if [[ "$build_all" == true ]]; then
        install_targets
        targets=(
            "x86_64-unknown-linux-gnu"
            "x86_64-unknown-linux-musl"
            "aarch64-unknown-linux-gnu"
            "x86_64-apple-darwin"
            "aarch64-apple-darwin"
            "x86_64-pc-windows-gnu"
        )
    fi
    
    # Build for each target
    local successful_targets=()
    for target in "${targets[@]}"; do
        if build_target "$target"; then
            successful_targets+=("$target")
            create_archive "$target" "$version"
        fi
    done
    
    # Create checksums
    if [[ ${#successful_targets[@]} -gt 0 ]]; then
        create_checksums
    fi
    
    # Build packages if requested
    if [[ "$build_packages" == true ]]; then
        # Only build packages if we have Linux builds
        for target in "${successful_targets[@]}"; do
            if [[ "$target" == "x86_64-unknown-linux-gnu" ]]; then
                build_deb
                build_rpm
                break
            fi
        done
        
        # Create Homebrew formula if we have macOS builds
        local has_macos=false
        for target in "${successful_targets[@]}"; do
            if [[ "$target" == *"apple-darwin"* ]]; then
                has_macos=true
                break
            fi
        done
        
        if [[ "$has_macos" == true ]]; then
            create_homebrew_formula "$version"
        fi
    fi
    
    # Summary
    echo
    print_success "Build completed successfully!"
    echo
    echo "Built targets:"
    for target in "${successful_targets[@]}"; do
        echo "  ✓ $target"
    done
    echo
    echo "Distribution files created in: $DIST_DIR"
    if [[ -d "$DIST_DIR" ]]; then
        ls -la "$DIST_DIR"
    fi
    echo
    echo "Next steps:"
    echo "1. Test the binaries on target platforms"
    echo "2. Create a GitHub release with these artifacts"
    echo "3. Update package managers (Homebrew, AUR, etc.)"
    echo "4. Update documentation with installation instructions"
}

# Run main function with all arguments
main "$@"
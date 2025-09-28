# Installation Guide

This guide covers all the ways to install `code-ingest` on different platforms.

## Quick Install (Recommended)

### Linux and macOS

```bash
curl -sSL https://github.com/your-org/code-ingest/releases/latest/download/install.sh | bash
```

This script will:
- Detect your platform automatically
- Download the appropriate binary
- Install it to `/usr/local/bin` (or `$INSTALL_DIR` if set)
- Verify the installation

### Custom Installation Directory

```bash
export INSTALL_DIR="$HOME/.local/bin"
curl -sSL https://github.com/your-org/code-ingest/releases/latest/download/install.sh | bash
```

## Package Managers

### Homebrew (macOS and Linux)

```bash
# Add the tap
brew tap your-org/tap

# Install code-ingest
brew install code-ingest

# Update to latest version
brew upgrade code-ingest
```

### Debian/Ubuntu (APT)

```bash
# Download and install the .deb package
wget https://github.com/your-org/code-ingest/releases/latest/download/code-ingest_VERSION_amd64.deb
sudo dpkg -i code-ingest_VERSION_amd64.deb

# Install dependencies if needed
sudo apt-get install -f
```

### Fedora/RHEL/CentOS (RPM)

```bash
# Download and install the .rpm package
wget https://github.com/your-org/code-ingest/releases/latest/download/code-ingest-VERSION-1.x86_64.rpm
sudo rpm -i code-ingest-VERSION-1.x86_64.rpm

# Or using dnf
sudo dnf install code-ingest-VERSION-1.x86_64.rpm
```

### Arch Linux (AUR)

```bash
# Using yay
yay -S code-ingest

# Using paru
paru -S code-ingest

# Manual installation
git clone https://aur.archlinux.org/code-ingest.git
cd code-ingest
makepkg -si
```

## Manual Installation

### Download Pre-built Binaries

1. Go to the [releases page](https://github.com/your-org/code-ingest/releases)
2. Download the appropriate archive for your platform:
   - **Linux x86_64**: `code-ingest-VERSION-x86_64-unknown-linux-gnu.tar.gz`
   - **Linux ARM64**: `code-ingest-VERSION-aarch64-unknown-linux-gnu.tar.gz`
   - **Linux (musl)**: `code-ingest-VERSION-x86_64-unknown-linux-musl.tar.gz`
   - **macOS Intel**: `code-ingest-VERSION-x86_64-apple-darwin.tar.gz`
   - **macOS Apple Silicon**: `code-ingest-VERSION-aarch64-apple-darwin.tar.gz`
   - **Windows**: `code-ingest-VERSION-x86_64-pc-windows-msvc.zip`

3. Extract the archive:
   ```bash
   # For .tar.gz files
   tar -xzf code-ingest-VERSION-PLATFORM.tar.gz
   
   # For .zip files (Windows)
   unzip code-ingest-VERSION-PLATFORM.zip
   ```

4. Move the binary to a directory in your PATH:
   ```bash
   # Linux/macOS
   sudo mv code-ingest-VERSION-PLATFORM/code-ingest /usr/local/bin/
   
   # Or to user directory
   mv code-ingest-VERSION-PLATFORM/code-ingest ~/.local/bin/
   ```

5. Make it executable (Linux/macOS):
   ```bash
   chmod +x /usr/local/bin/code-ingest
   ```

### Verify Installation

```bash
code-ingest --version
code-ingest --help
```

## Build from Source

### Prerequisites

- **Rust**: Install from [rustup.rs](https://rustup.rs/)
- **PostgreSQL**: Required for database operations
- **Git**: For cloning repositories
- **System dependencies**:
  - Linux: `pkg-config`, `libssl-dev`
  - macOS: Xcode Command Line Tools
  - Windows: Visual Studio Build Tools

### Optional Dependencies (for file conversion)

- **poppler-utils**: For PDF text extraction (`pdftotext`)
- **pandoc**: For document conversion
- **Python 3**: For Excel file processing

#### Install Optional Dependencies

**Ubuntu/Debian:**
```bash
sudo apt-get install poppler-utils pandoc python3-pip
pip3 install openpyxl pandas
```

**macOS:**
```bash
brew install poppler pandoc python3
pip3 install openpyxl pandas
```

**Fedora/RHEL:**
```bash
sudo dnf install poppler-utils pandoc python3-pip
pip3 install openpyxl pandas
```

### Build Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-org/code-ingest.git
   cd code-ingest
   ```

2. **Build the project:**
   ```bash
   # Debug build (faster compilation)
   make build
   
   # Release build (optimized)
   make build-release
   
   # Or using cargo directly
   cd code-ingest
   cargo build --release
   ```

3. **Install locally:**
   ```bash
   # Install to system
   make install
   
   # Install to ~/.cargo/bin
   make install-dev
   
   # Or using cargo
   cd code-ingest
   cargo install --path .
   ```

### Development Setup

If you want to contribute or modify the code:

```bash
# Set up development environment
make dev-setup

# Set up local databases
make db-setup

# Run tests
make test

# Run all checks (lint, format, test)
make dev-check
```

## Platform-Specific Instructions

### Windows

1. **Download the Windows binary** from the releases page
2. **Extract the ZIP file** to a folder (e.g., `C:\Program Files\code-ingest\`)
3. **Add to PATH**:
   - Open System Properties → Advanced → Environment Variables
   - Add the installation directory to your PATH
   - Or place the binary in an existing PATH directory

4. **Install PostgreSQL**:
   - Download from [postgresql.org](https://www.postgresql.org/download/windows/)
   - Follow the installation wizard
   - Note the connection details for later use

### Docker

Run code-ingest in a Docker container:

```bash
# Pull the image
docker pull your-org/code-ingest:latest

# Run with volume mounts
docker run -it --rm \
  -v $(pwd)/analysis:/workspace/analysis \
  -e DATABASE_URL="postgresql://user:pass@host:5432/db" \
  your-org/code-ingest:latest \
  ingest https://github.com/rust-lang/mdBook --db-path /workspace/analysis
```

### Build Docker Image

```bash
# Build the image
docker build -t code-ingest .

# Run locally built image
docker run -it --rm code-ingest --help
```

## Configuration

### Environment Variables

- `DATABASE_URL`: PostgreSQL connection string
- `GITHUB_TOKEN`: GitHub personal access token for private repositories
- `RUST_LOG`: Logging level (debug, info, warn, error)

### Example Configuration

```bash
# ~/.bashrc or ~/.zshrc
export DATABASE_URL="postgresql://username:password@localhost:5432/code_analysis"
export GITHUB_TOKEN="ghp_your_token_here"
export RUST_LOG="info"
```

## Verification

After installation, verify everything works:

```bash
# Check version
code-ingest --version

# Test PostgreSQL connection
code-ingest pg-start

# Test basic functionality
mkdir test-analysis
code-ingest ingest https://github.com/rust-lang/mdBook --db-path ./test-analysis

# Query the ingested data
code-ingest sql "SELECT COUNT(*) FROM ingestion_meta" --db-path ./test-analysis
```

## Troubleshooting

### Common Issues

1. **Binary not found after installation**
   - Check that the installation directory is in your PATH
   - Try running with full path: `/usr/local/bin/code-ingest --version`

2. **Permission denied**
   - Make sure the binary is executable: `chmod +x /path/to/code-ingest`
   - Check directory permissions

3. **PostgreSQL connection issues**
   - Verify PostgreSQL is running: `pg_isready`
   - Check connection string format
   - Ensure database exists

4. **Missing system dependencies**
   - Install development tools for your platform
   - Check error messages for specific missing libraries

### Getting Help

- **Documentation**: Check the [docs/](docs/) directory
- **Issues**: Report bugs on [GitHub Issues](https://github.com/your-org/code-ingest/issues)
- **Discussions**: Ask questions in [GitHub Discussions](https://github.com/your-org/code-ingest/discussions)

## Upgrading

### Package Managers

```bash
# Homebrew
brew upgrade code-ingest

# APT (Debian/Ubuntu)
sudo apt update && sudo apt upgrade code-ingest

# DNF (Fedora)
sudo dnf upgrade code-ingest
```

### Manual Upgrade

1. Download the new version using the same method as installation
2. Replace the existing binary
3. Check the changelog for any breaking changes

### From Source

```bash
git pull origin main
make clean
make build-release
make install
```

## Uninstallation

### Package Managers

```bash
# Homebrew
brew uninstall code-ingest

# APT
sudo apt remove code-ingest

# DNF
sudo dnf remove code-ingest
```

### Manual Uninstallation

```bash
# Remove binary
sudo rm /usr/local/bin/code-ingest

# Or use the Makefile
make uninstall

# Remove configuration (optional)
rm -rf ~/.config/code-ingest
```

## Next Steps

After installation:

1. **Read the [CLI Reference](docs/CLI_REFERENCE.md)** for command details
2. **Check out [examples/](examples/)** for usage examples
3. **Set up your first analysis** with the getting started guide
4. **Configure your development environment** for optimal performance

---

For more detailed information, see:
- [CLI Reference](docs/CLI_REFERENCE.md)
- [Developer Guide](docs/DEVELOPER_GUIDE.md)
- [Troubleshooting Guide](docs/TROUBLESHOOTING.md)
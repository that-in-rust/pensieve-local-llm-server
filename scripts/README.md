# Scripts Directory

This directory contains automation scripts for building, releasing, and managing the code-ingest project.

## Scripts Overview

### Build and Release Scripts

#### `build-release.sh`
Cross-platform release build script that creates optimized binaries for all supported platforms.

**Usage:**
```bash
# Build for all platforms
./scripts/build-release.sh

# Build for specific target
./scripts/build-release.sh --target x86_64-unknown-linux-gnu

# Build with distribution packages
./scripts/build-release.sh --packages
```

**Features:**
- Cross-compilation for Linux, macOS, and Windows
- Automatic archive creation (tar.gz, zip)
- Checksum generation (SHA256, MD5)
- Debian and RPM package building
- Homebrew formula generation

#### `version-bump.sh`
Version management script for semantic versioning.

**Usage:**
```bash
# Show current version
./scripts/version-bump.sh current

# Bump patch version (1.0.0 -> 1.0.1)
./scripts/version-bump.sh bump patch

# Bump minor version (1.0.0 -> 1.1.0)
./scripts/version-bump.sh bump minor

# Bump major version (1.0.0 -> 2.0.0)
./scripts/version-bump.sh bump major

# Set specific version
./scripts/version-bump.sh set 1.2.3

# Create git tag for current version
./scripts/version-bump.sh tag
```

**Features:**
- Semantic versioning compliance
- Automatic Cargo.toml updates
- Changelog generation
- Git tag creation

### Testing Scripts

#### `test-release.sh`
Comprehensive release testing script that validates releases across different installation methods.

**Usage:**
```bash
# Test specific version
./scripts/test-release.sh v1.2.3

# Test latest release
./scripts/test-release.sh latest

# Skip specific test types
./scripts/test-release.sh v1.2.3 --skip-docker --skip-install
```

**Test Coverage:**
- Binary download and execution
- Installation script functionality
- Docker image validation
- Basic CLI functionality
- Real repository ingestion (if PostgreSQL available)

## Release Process

### Quick Release Workflow

1. **Prepare Release:**
   ```bash
   # For patch release
   make release-prepare
   
   # For minor release
   make release-prepare-minor
   
   # For major release
   make release-prepare-major
   ```

2. **Push Tag:**
   ```bash
   git push origin main
   git push origin --tags
   ```

3. **Monitor GitHub Actions:**
   - Check the release workflow completion
   - Verify all artifacts are built
   - Confirm release is published

### Manual Release (Fallback)

If automated release fails:

1. **Build Artifacts:**
   ```bash
   ./scripts/build-release.sh --packages
   ```

2. **Test Release:**
   ```bash
   ./scripts/test-release.sh v1.2.3
   ```

3. **Create GitHub Release:**
   - Upload artifacts from `target/dist/`
   - Include checksums and install script
   - Write release notes

## Development Workflow

### Version Management

```bash
# Check current version
./scripts/version-bump.sh current

# Prepare for next development cycle
./scripts/version-bump.sh bump patch
git add -A
git commit -m "Bump version to $(./scripts/version-bump.sh current)"
```

### Build Testing

```bash
# Test local build
make build-release

# Test cross-platform builds
./scripts/build-release.sh --target x86_64-unknown-linux-musl

# Test packages
./scripts/build-release.sh --packages
```

## CI/CD Integration

### GitHub Actions

The scripts integrate with GitHub Actions workflows:

- **`build-release.sh`**: Used in release workflow for artifact creation
- **`version-bump.sh`**: Used for automated version management
- **`test-release.sh`**: Used for release validation

### Local CI Simulation

```bash
# Run the same checks as CI
make ci-test

# Build like CI
make ci-build

# Full CI simulation
make dev-full
```

## Platform Support

### Supported Targets

- **Linux x86_64**: `x86_64-unknown-linux-gnu`
- **Linux x86_64 (musl)**: `x86_64-unknown-linux-musl`
- **Linux ARM64**: `aarch64-unknown-linux-gnu`
- **macOS Intel**: `x86_64-apple-darwin`
- **macOS Apple Silicon**: `aarch64-apple-darwin`
- **Windows**: `x86_64-pc-windows-msvc`

### Package Formats

- **Archives**: `.tar.gz` (Unix), `.zip` (Windows)
- **Debian**: `.deb` packages
- **RPM**: `.rpm` packages
- **Homebrew**: Formula generation
- **Docker**: Multi-arch container images

## Configuration

### Environment Variables

Scripts respect these environment variables:

- `CARGO_TARGET_DIR`: Override cargo target directory
- `INSTALL_DIR`: Installation directory for install scripts
- `GITHUB_TOKEN`: GitHub API token for releases
- `DATABASE_URL`: PostgreSQL connection for testing

### Script Configuration

Most scripts have configuration sections at the top:

```bash
# Configuration
PROJECT_NAME="code-ingest"
CARGO_MANIFEST="code-ingest/Cargo.toml"
BUILD_DIR="target/release-builds"
DIST_DIR="target/dist"
```

## Troubleshooting

### Common Issues

1. **Cross-compilation failures:**
   ```bash
   # Install missing targets
   rustup target add x86_64-unknown-linux-musl
   
   # Install cross-compilation tools
   cargo install cross
   ```

2. **Package building failures:**
   ```bash
   # Install packaging tools
   cargo install cargo-deb cargo-generate-rpm
   ```

3. **Permission issues:**
   ```bash
   # Make scripts executable
   chmod +x scripts/*.sh
   ```

### Debug Mode

Enable debug output in scripts:

```bash
# Enable bash debug mode
bash -x ./scripts/build-release.sh

# Enable verbose cargo output
CARGO_TERM_VERBOSE=true ./scripts/build-release.sh
```

## Contributing

### Adding New Scripts

1. **Follow naming convention:** `action-target.sh`
2. **Include help text:** `--help` flag support
3. **Add error handling:** Use `set -e` and proper error messages
4. **Document in this README:** Add usage examples
5. **Test thoroughly:** Verify on multiple platforms

### Script Standards

- **Shebang:** `#!/bin/bash`
- **Error handling:** `set -e` at the top
- **Color output:** Use consistent color functions
- **Help text:** Always include `--help` option
- **Cleanup:** Use trap for cleanup on exit

### Testing Scripts

```bash
# Test script syntax
bash -n scripts/script-name.sh

# Test with shellcheck (if available)
shellcheck scripts/script-name.sh

# Test functionality
./scripts/script-name.sh --help
```

## Security Considerations

### Token Handling

- Never commit tokens to version control
- Use environment variables for sensitive data
- Validate token permissions before use

### Binary Verification

- Always generate and verify checksums
- Sign releases when possible
- Use HTTPS for all downloads

### Container Security

- Use minimal base images
- Run as non-root user
- Scan for vulnerabilities

## Maintenance

### Regular Tasks

- **Update dependencies:** Check for new Rust versions
- **Review scripts:** Ensure compatibility with new platforms
- **Test workflows:** Validate CI/CD pipelines regularly
- **Update documentation:** Keep README current

### Monitoring

- **GitHub Actions:** Monitor workflow success rates
- **Download metrics:** Track release adoption
- **Error reports:** Monitor for installation issues
- **Performance:** Track build times and artifact sizes

---

For more information, see:
- [Release Checklist](release-checklist.md)
- [Main README](../README.md)
- [Developer Guide](../docs/DEVELOPER_GUIDE.md)
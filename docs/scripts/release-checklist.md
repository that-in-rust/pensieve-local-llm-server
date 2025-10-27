# Release Checklist

This checklist ensures consistent and reliable releases of code-ingest.

## Pre-Release Preparation

### 1. Code Quality Checks
- [ ] All tests pass locally (`make test`)
- [ ] Code is properly formatted (`make check-format`)
- [ ] No clippy warnings (`make lint`)
- [ ] Security audit passes (`make security`)
- [ ] Performance benchmarks are acceptable (`make bench`)
- [ ] Documentation is up to date (`make docs-check`)

### 2. Version Management
- [ ] Determine version bump type (patch/minor/major)
- [ ] Update version in `code-ingest/Cargo.toml`
- [ ] Update CHANGELOG.md with release notes
- [ ] Commit version changes
- [ ] Create and push git tag

### 3. Testing
- [ ] Run full test suite (`make ci-test`)
- [ ] Test on multiple platforms (Linux, macOS, Windows)
- [ ] Verify database operations work correctly
- [ ] Test with different PostgreSQL versions
- [ ] Validate file processing with various file types

### 4. Documentation Updates
- [ ] Update README.md if needed
- [ ] Update CLI_REFERENCE.md for any new commands
- [ ] Update INSTALL.md for any installation changes
- [ ] Check all example files work with new version
- [ ] Update Docker image documentation

## Release Process

### 1. Automated Release (Recommended)
- [ ] Push git tag to trigger GitHub Actions
- [ ] Monitor GitHub Actions workflow completion
- [ ] Verify all artifacts are built successfully
- [ ] Check that checksums are generated
- [ ] Confirm release is created on GitHub

### 2. Manual Release (Fallback)
- [ ] Build release artifacts (`make release-build`)
- [ ] Generate checksums (`scripts/build-release.sh`)
- [ ] Create GitHub release manually
- [ ] Upload all artifacts to release
- [ ] Update package managers manually

### 3. Distribution Updates
- [ ] Homebrew formula is updated (automated)
- [ ] AUR package is updated (manual)
- [ ] Docker image is built and pushed
- [ ] Debian/RPM packages are available
- [ ] Installation script works with new version

## Post-Release Verification

### 1. Installation Testing
- [ ] Test quick install script on clean system
- [ ] Verify Homebrew installation works
- [ ] Test Debian package installation
- [ ] Test RPM package installation
- [ ] Verify Docker image works correctly

### 2. Functionality Testing
- [ ] Test basic ingestion workflow
- [ ] Verify SQL query functionality
- [ ] Test IDE integration features
- [ ] Check performance with large repositories
- [ ] Validate error handling and recovery

### 3. Documentation and Communication
- [ ] Update project website (if applicable)
- [ ] Announce release on social media
- [ ] Update package manager descriptions
- [ ] Notify users of breaking changes (if any)
- [ ] Update integration documentation

## Rollback Plan

If issues are discovered after release:

### 1. Immediate Actions
- [ ] Document the issue clearly
- [ ] Assess impact and severity
- [ ] Decide on rollback vs. hotfix
- [ ] Communicate with users if needed

### 2. Rollback Process
- [ ] Remove problematic release from GitHub
- [ ] Revert package manager updates
- [ ] Update installation scripts to use previous version
- [ ] Rebuild Docker image with previous version
- [ ] Notify users of the rollback

### 3. Fix and Re-release
- [ ] Fix the identified issues
- [ ] Follow full release process again
- [ ] Increment patch version for hotfix
- [ ] Document lessons learned

## Release Types

### Patch Release (x.y.Z)
- Bug fixes
- Security updates
- Documentation improvements
- Performance optimizations (non-breaking)

**Process:**
```bash
make version-bump-patch
make release-prepare
git push origin main
git push origin --tags
```

### Minor Release (x.Y.z)
- New features (backward compatible)
- New CLI commands
- Enhanced functionality
- Dependency updates

**Process:**
```bash
make version-bump-minor
make release-prepare
git push origin main
git push origin --tags
```

### Major Release (X.y.z)
- Breaking changes
- API modifications
- Major architectural changes
- Removed deprecated features

**Process:**
```bash
make version-bump-major
make release-prepare
git push origin main
git push origin --tags
```

## Automation Scripts

### Quick Release Commands
```bash
# Patch release
make release-prepare

# Minor release  
make release-prepare-minor

# Major release
make release-prepare-major

# Build all artifacts locally
make release-build

# Test installation
make demo-setup
```

### Manual Verification
```bash
# Test the release locally
./scripts/test-release.sh v1.2.3

# Verify checksums
./scripts/verify-checksums.sh

# Test Docker image
docker run --rm code-ingest:latest --version
```

## Common Issues and Solutions

### Build Failures
- **Cross-compilation issues**: Check target installation
- **Dependency conflicts**: Update Cargo.lock
- **Test failures**: Run tests locally first

### Distribution Issues
- **Package manager delays**: Contact maintainers
- **Checksum mismatches**: Rebuild artifacts
- **Installation script failures**: Test on clean systems

### Performance Regressions
- **Benchmark failures**: Compare with baseline
- **Memory issues**: Run with profiling tools
- **Speed degradation**: Identify bottlenecks

## Release Schedule

### Regular Releases
- **Patch releases**: As needed for critical fixes
- **Minor releases**: Monthly or bi-monthly
- **Major releases**: Quarterly or as needed

### Emergency Releases
- **Security vulnerabilities**: Within 24-48 hours
- **Critical bugs**: Within 1 week
- **Data corruption issues**: Immediate

## Success Criteria

A release is considered successful when:
- [ ] All automated tests pass
- [ ] Installation works on all supported platforms
- [ ] No critical bugs reported within 48 hours
- [ ] Performance benchmarks meet expectations
- [ ] User feedback is positive
- [ ] Documentation is accurate and complete

## Contact Information

**Release Manager**: [Your Name] <email@example.com>
**Backup**: [Backup Name] <backup@example.com>
**Security Contact**: security@code-ingest.dev

## Tools and Resources

- **GitHub Actions**: Automated CI/CD
- **Cargo**: Rust package manager
- **Cross**: Cross-compilation tool
- **Docker**: Container builds
- **Homebrew**: macOS package manager
- **AUR**: Arch Linux packages
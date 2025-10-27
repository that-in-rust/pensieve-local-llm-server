# Makefile for code-ingest project
# Provides convenient commands for building, testing, and releasing

.PHONY: help build test clean release install dev-setup lint format check-format security audit bench docs

# Default target
help: ## Show this help message
	@echo "Code Ingest - Build and Release Management"
	@echo "=========================================="
	@echo
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo "Environment variables:"
	@echo "  CARGO_TARGET_DIR    Override cargo target directory"
	@echo "  INSTALL_DIR         Installation directory (default: /usr/local/bin)"
	@echo "  VERSION             Version for release builds"

# Build targets
build: ## Build the project in debug mode
	cd code-ingest && cargo build

build-release: ## Build optimized release binary
	cd code-ingest && cargo build --release

build-all-targets: ## Build for all supported targets
	./scripts/build-release.sh

# Test targets
test: ## Run all tests
	cd code-ingest && cargo test

test-unit: ## Run unit tests only
	cd code-ingest && cargo test --lib --bins unit_

test-integration: ## Run integration tests only
	cd code-ingest && cargo test --tests integration_

test-e2e: ## Run end-to-end tests
	cd code-ingest && cargo test --tests end_to_end_

test-property: ## Run property-based tests
	cd code-ingest && cargo test --tests property_

test-coverage: ## Generate test coverage report
	cd code-ingest && cargo llvm-cov --all-features --workspace --lcov --output-path lcov.info
	@echo "Coverage report generated: code-ingest/lcov.info"

# Development targets
dev-setup: ## Set up development environment
	@echo "Setting up development environment..."
	rustup component add rustfmt clippy llvm-tools-preview
	cargo install cargo-llvm-cov cargo-audit cargo-deny cargo-deb cargo-generate-rpm
	@echo "Development environment ready!"

lint: ## Run clippy lints
	cd code-ingest && cargo clippy --all-targets --all-features -- -D warnings

format: ## Format code with rustfmt
	cd code-ingest && cargo fmt --all

check-format: ## Check code formatting
	cd code-ingest && cargo fmt --all -- --check

# Security and audit targets
security: ## Run security audit
	cd code-ingest && cargo audit

audit: security ## Alias for security

deny: ## Run cargo-deny checks
	cd code-ingest && cargo deny check

# Performance targets
bench: ## Run benchmarks
	cd code-ingest && cargo bench --bench performance_benchmarks

bench-baseline: ## Run benchmarks and save as baseline
	cd code-ingest && cargo bench --bench performance_benchmarks -- --save-baseline main

bench-compare: ## Compare benchmarks against baseline
	cd code-ingest && cargo bench --bench performance_benchmarks -- --baseline main

# Documentation targets
docs: ## Build documentation
	cd code-ingest && cargo doc --all-features --no-deps --open

docs-check: ## Check documentation for issues
	cd code-ingest && cargo doc --all-features --no-deps 2>&1 | grep -i "warning\|error" && exit 1 || exit 0

# Version management
version-current: ## Show current version
	./scripts/version-bump.sh current

version-bump-patch: ## Bump patch version (1.0.0 -> 1.0.1)
	./scripts/version-bump.sh bump patch

version-bump-minor: ## Bump minor version (1.0.0 -> 1.1.0)
	./scripts/version-bump.sh bump minor

version-bump-major: ## Bump major version (1.0.0 -> 2.0.0)
	./scripts/version-bump.sh bump major

version-set: ## Set specific version (usage: make version-set VERSION=1.2.3)
	@if [ -z "$(VERSION)" ]; then echo "Error: VERSION not specified. Usage: make version-set VERSION=1.2.3"; exit 1; fi
	./scripts/version-bump.sh set $(VERSION)

version-tag: ## Create git tag for current version
	./scripts/version-bump.sh tag

# Release targets
release-build: ## Build release artifacts for all platforms
	./scripts/build-release.sh --packages

release-local: ## Build release for current platform only
	./scripts/build-release.sh --target $(shell rustc -vV | grep host | cut -d' ' -f2)

release-prepare: version-bump-patch version-tag ## Prepare a patch release (bump version and create tag)

release-prepare-minor: version-bump-minor version-tag ## Prepare a minor release

release-prepare-major: version-bump-major version-tag ## Prepare a major release

# Installation targets
install: build-release ## Install to system (requires sudo for /usr/local/bin)
	@INSTALL_DIR=${INSTALL_DIR:-/usr/local/bin}; \
	echo "Installing code-ingest to $$INSTALL_DIR..."; \
	if [ ! -w "$$INSTALL_DIR" ]; then \
		sudo mkdir -p "$$INSTALL_DIR"; \
		sudo cp code-ingest/target/release/code-ingest "$$INSTALL_DIR/"; \
		sudo chmod +x "$$INSTALL_DIR/code-ingest"; \
	else \
		mkdir -p "$$INSTALL_DIR"; \
		cp code-ingest/target/release/code-ingest "$$INSTALL_DIR/"; \
		chmod +x "$$INSTALL_DIR/code-ingest"; \
	fi; \
	echo "Installation completed successfully!"

install-dev: build ## Install development build to ~/.cargo/bin
	cd code-ingest && cargo install --path . --force

uninstall: ## Uninstall from system
	@INSTALL_DIR=${INSTALL_DIR:-/usr/local/bin}; \
	if [ -f "$$INSTALL_DIR/code-ingest" ]; then \
		if [ ! -w "$$INSTALL_DIR" ]; then \
			sudo rm "$$INSTALL_DIR/code-ingest"; \
		else \
			rm "$$INSTALL_DIR/code-ingest"; \
		fi; \
		echo "code-ingest uninstalled from $$INSTALL_DIR"; \
	else \
		echo "code-ingest not found in $$INSTALL_DIR"; \
	fi

# Package building
package-deb: ## Build Debian package
	cd code-ingest && cargo deb --target x86_64-unknown-linux-gnu

package-rpm: ## Build RPM package
	cd code-ingest && cargo generate-rpm --target x86_64-unknown-linux-gnu

package-all: package-deb package-rpm ## Build all packages

# Cleanup targets
clean: ## Clean build artifacts
	cd code-ingest && cargo clean
	rm -rf target/release-builds target/dist

clean-all: clean ## Clean everything including caches
	cd code-ingest && cargo clean
	rm -rf target/
	rm -rf ~/.cargo/registry/cache/
	rm -rf ~/.cargo/git/

# CI/CD helpers
ci-setup: ## Set up CI environment
	rustup component add rustfmt clippy llvm-tools-preview
	cargo install cargo-llvm-cov

ci-test: lint check-format test security ## Run all CI checks

ci-build: ## Build for CI (all targets)
	./scripts/build-release.sh

# Development workflow helpers
dev-check: lint check-format test ## Quick development checks

dev-full: clean dev-check build-release test-coverage ## Full development workflow

# Quick commands
quick-test: ## Quick test (unit tests only)
	cd code-ingest && cargo test --lib unit_

quick-build: ## Quick build (debug mode)
	cd code-ingest && cargo build

# Database setup helpers
db-setup: ## Set up local PostgreSQL for development
	@echo "Setting up local PostgreSQL..."
	@if command -v psql >/dev/null 2>&1; then \
		createdb code_ingest_dev 2>/dev/null || echo "Database code_ingest_dev already exists"; \
		createdb code_ingest_test 2>/dev/null || echo "Database code_ingest_test already exists"; \
		echo "Databases created successfully!"; \
	else \
		echo "PostgreSQL not found. Please install PostgreSQL first."; \
		echo "On macOS: brew install postgresql"; \
		echo "On Ubuntu: sudo apt-get install postgresql postgresql-contrib"; \
	fi

db-clean: ## Clean test databases
	@echo "Cleaning test databases..."
	@dropdb code_ingest_test 2>/dev/null || echo "Test database doesn't exist"
	@createdb code_ingest_test 2>/dev/null || echo "Failed to recreate test database"

# Example and demo targets
demo-setup: ## Set up demo environment
	mkdir -p demo-workspace
	@echo "Demo workspace created in demo-workspace/"
	@echo "Run: cd demo-workspace && code-ingest --help"

demo-clean: ## Clean demo environment
	rm -rf demo-workspace

# Help for specific workflows
help-release: ## Show release workflow help
	@echo "Release Workflow:"
	@echo "=================="
	@echo
	@echo "1. Prepare release:"
	@echo "   make release-prepare        # Patch release (1.0.0 -> 1.0.1)"
	@echo "   make release-prepare-minor  # Minor release (1.0.0 -> 1.1.0)"
	@echo "   make release-prepare-major  # Major release (1.0.0 -> 2.0.0)"
	@echo
	@echo "2. Build release artifacts:"
	@echo "   make release-build          # Build for all platforms"
	@echo
	@echo "3. Push tag to trigger GitHub Actions:"
	@echo "   git push origin v<version>"
	@echo
	@echo "4. GitHub Actions will automatically:"
	@echo "   - Build binaries for all platforms"
	@echo "   - Create distribution packages"
	@echo "   - Generate checksums"
	@echo "   - Create GitHub release"
	@echo "   - Update Homebrew formula"

help-dev: ## Show development workflow help
	@echo "Development Workflow:"
	@echo "===================="
	@echo
	@echo "1. Initial setup:"
	@echo "   make dev-setup              # Install development tools"
	@echo "   make db-setup               # Set up local databases"
	@echo
	@echo "2. Development cycle:"
	@echo "   make dev-check              # Quick checks (lint, format, test)"
	@echo "   make build                  # Build in debug mode"
	@echo "   make test                   # Run all tests"
	@echo
	@echo "3. Before committing:"
	@echo "   make dev-full               # Full workflow with coverage"
	@echo "   make ci-test                # Same checks as CI"
	@echo
	@echo "4. Performance testing:"
	@echo "   make bench                  # Run benchmarks"
	@echo "   make test-coverage          # Generate coverage report"
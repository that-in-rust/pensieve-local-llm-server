# Multi-stage Dockerfile for code-ingest
# Builds a minimal container with all necessary dependencies

# Build stage
FROM rust:1.75-slim as builder

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    libpq-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy workspace configuration
COPY Cargo.toml Cargo.lock ./

# Copy source code
COPY code-ingest/ ./code-ingest/

# Build the application in release mode
RUN cd code-ingest && cargo build --release

# Runtime stage
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    libpq5 \
    postgresql-client \
    poppler-utils \
    pandoc \
    python3 \
    python3-pip \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages for Excel processing
RUN pip3 install --no-cache-dir openpyxl pandas

# Create non-root user
RUN useradd -m -u 1000 codeuser && \
    mkdir -p /workspace && \
    chown -R codeuser:codeuser /workspace

# Copy binary from builder stage
COPY --from=builder /app/code-ingest/target/release/code-ingest /usr/local/bin/

# Make binary executable
RUN chmod +x /usr/local/bin/code-ingest

# Set working directory
WORKDIR /workspace

# Switch to non-root user
USER codeuser

# Set environment variables
ENV RUST_LOG=info
ENV RUST_BACKTRACE=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD code-ingest --version || exit 1

# Default command
ENTRYPOINT ["code-ingest"]
CMD ["--help"]

# Metadata
LABEL org.opencontainers.image.title="code-ingest"
LABEL org.opencontainers.image.description="High-performance tool for ingesting GitHub repositories into PostgreSQL"
LABEL org.opencontainers.image.vendor="Code Ingest Team"
LABEL org.opencontainers.image.licenses="MIT OR Apache-2.0"
LABEL org.opencontainers.image.source="https://github.com/your-org/code-ingest"
LABEL org.opencontainers.image.documentation="https://github.com/your-org/code-ingest/blob/main/README.md"
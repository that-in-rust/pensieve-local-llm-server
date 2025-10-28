# Pensieve Deployment Guide

**Complete guide for deploying the Pensieve Local LLM Server in production**

## Overview

This guide covers deployment strategies for the Pensieve Local LLM Server, from local development to production environments. Pensieve is designed to be lightweight, secure, and easy to deploy.

## System Requirements

### Minimum Requirements
- **OS**: macOS 12+ (Apple Silicon recommended)
- **RAM**: 8GB (16GB+ recommended)
- **Storage**: 10GB free space
- **CPU**: Apple Silicon M1/M2/M3 (x86_64 with CPU fallback)
- **Network**: Local network access for API calls

### Recommended Production Requirements
- **OS**: macOS 13+ (Apple Silicon)
- **RAM**: 32GB+ for larger models
- **Storage**: 50GB+ SSD storage
- **CPU**: Apple Silicon M2 Pro/M3 Max
- **Network**: Gigabit Ethernet or WiFi 6

## Installation

### Method 1: Build from Source (Recommended)

```bash
# Clone the repository
git clone https://github.com/amuldotexe/pensieve-local-llm-server
cd pensieve-local-llm-server

# Install Rust if not already installed
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Build the release version
cargo build --release

# Verify installation
./target/release/pensieve --version
```

### Method 2: Download Binary (Future)

```bash
# Download the appropriate binary for your architecture
curl -L https://github.com/amuldotexe/pensieve/releases/download/v0.1.0/pensieve-macos-arm64.tar.gz -o pensieve.tar.gz
tar -xzf pensieve.tar.gz
chmod +x pensieve
./pensieve --version
```

## Configuration

### Basic Configuration

Create a configuration file `config.json`:

```json
{
  "server": {
    "host": "0.0.0.0",
    "port": 7777,
    "max_concurrent_requests": 100,
    "request_timeout_ms": 30000,
    "enable_cors": true
  },
  "logging": {
    "level": "info",
    "format": "json",
    "file": "/var/log/pensieve/pensieve.log"
  },
  "model": {
    "model_path": "/opt/pensieve/models/llama-2-7b-chat.gguf",
    "model_type": "llama",
    "context_size": 4096,
    "gpu_layers": 32
  }
}
```

### Production Configuration

For production deployments, use these enhanced settings:

```json
{
  "server": {
    "host": "0.0.0.0",
    "port": 7777,
    "max_concurrent_requests": 50,
    "request_timeout_ms": 60000,
    "enable_cors": false
  },
  "logging": {
    "level": "warn",
    "format": "json",
    "file": "/var/log/pensieve/pensieve.log"
  },
  "model": {
    "model_path": "/opt/pensieve/models/llama-2-13b-chat.gguf",
    "model_type": "llama",
    "context_size": 2048,
    "gpu_layers": 0
  }
}
```

## Deployment Methods

### Method 1: Standalone Service

#### 1. Create Service User

```bash
# Create a dedicated user for the service
sudo dscl . create /users/pensieve
sudo dscl . create /users/pensieve UserShell /usr/bin/false
sudo dscl . create /users/pensieve RealName "Pensieve Service"
sudo dscl . create /users/pensieve PrimaryGroupID 20
sudo dscl . create /users/pensieve NFSHomeDirectory /var/empty
sudo dscl . create /users/pensieve UniqueID 500
```

#### 2. Install Application

```bash
# Create directories
sudo mkdir -p /opt/pensieve/bin
sudo mkdir -p /opt/pensieve/config
sudo mkdir -p /opt/pensieve/models
sudo mkdir -p /var/log/pensieve

# Copy binary
sudo cp target/release/pensieve /opt/pensieve/bin/
sudo chmod +x /opt/pensieve/bin/pensieve

# Copy configuration
sudo cp config.json /opt/pensieve/config/
sudo chown -R pensieve:staff /opt/pensieve
sudo chown -R pensieve:staff /var/log/pensieve
```

#### 3. Create Launch Daemon

Create `/Library/LaunchDaemons/com.pensieve.server.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.pensieve.server</string>
    <key>ProgramArguments</key>
    <array>
        <string>/opt/pensieve/bin/pensieve</string>
        <string>-c</string>
        <string>/opt/pensieve/config/config.json</string>
        <string>start</string>
    </array>
    <key>WorkingDirectory</key>
    <string>/opt/pensieve</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/var/log/pensieve/pensieve.out.log</string>
    <key>StandardErrorPath</key>
    <string>/var/log/pensieve/pensieve.err.log</string>
    <key>UserName</key>
    <string>pensieve</string>
    <key>GroupName</key>
    <string>staff</string>
</dict>
</plist>
```

#### 4. Start Service

```bash
# Load the service
sudo launchctl load /Library/LaunchDaemons/com.pensieve.server.plist

# Check status
sudo launchctl list | grep pensieve

# View logs
tail -f /var/log/pensieve/pensieve.out.log
```

### Method 2: Docker Deployment

#### 1. Build Docker Image

```dockerfile
# Dockerfile
FROM --platform=linux/arm64 rust:1.75 as builder

WORKDIR /app
COPY . .
RUN cargo build --release

FROM --platform=linux/arm64 debian:bookworm-slim

RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/pensieve /usr/local/bin/
RUN chmod +x /usr/local/bin/pensieve

EXPOSE 7777
CMD ["pensieve", "start"]
```

```bash
# Build the image
docker build -t pensieve:latest .

# Run container
docker run -d \
  --name pensieve \
  -p 7777:7777 \
  -v $(pwd)/config.json:/config.json \
  -v $(pwd)/models:/models \
  pensieve:latest \
  -c /config.json start
```

#### 2. Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  pensieve:
    build: .
    ports:
      - "7777:7777"
    volumes:
      - ./config.json:/config.json:ro
      - ./models:/models:ro
      - ./logs:/var/log/pensieve
    environment:
      - RUST_LOG=info
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:7777/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - pensieve
    restart: unless-stopped
```

### Method 3: Systemd (Linux)

Create `/etc/systemd/system/pensieve.service`:

```ini
[Unit]
Description=Pensieve Local LLM Server
After=network.target

[Service]
Type=simple
User=pensieve
Group=pensieve
WorkingDirectory=/opt/pensieve
ExecStart=/opt/pensieve/bin/pensieve -c /opt/pensieve/config/config.json start
ExecReload=/bin/kill -HUP $MAINPID
Restart=always
RestartSec=5

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/log/pensieve

# Resource limits
LimitNOFILE=65536
MemoryLimit=8G

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
sudo systemctl enable pensieve
sudo systemctl start pensieve
sudo systemctl status pensieve
```

## Model Management

### Downloading Models

```bash
# Create models directory
mkdir -p /opt/pensieve/models
cd /opt/pensieve/models

# Example: Download Llama 2 7B Chat model
wget https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf

# Verify model
sha256sum llama-2-7b-chat.Q4_K_M.gguf
```

### Model Selection Guide

| Model Size | RAM Required | Performance | Use Case |
|------------|--------------|-------------|----------|
| 3B (Q4) | 4GB | Fast | Simple queries, development |
| 7B (Q4) | 6GB | Balanced | General purpose, documentation |
| 13B (Q4) | 10GB | Good quality | Complex reasoning, analysis |
| 70B (Q4) | 40GB | High quality | Advanced tasks (requires high-end hardware) |

### GPU Layer Configuration

```bash
# Test different GPU layer counts
./pensieve start --model llama-2-7b-chat.gguf --gpu-layers 0   # CPU only
./pensieve start --model llama-2-7b-chat.gguf --gpu-layers 16  # Some GPU
./pensieve start --model llama-2-7b-chat.gguf --gpu-layers 32  # More GPU
./pensieve start --model llama-2-7b-chat.gguf --gpu-layers -1  # All GPU
```

## Security

### Network Security

#### 1. Firewall Configuration

```bash
# Allow only local network access
sudo pfctl -e
echo "block in all\npass in on en0 from 192.168.0.0/16 to any port 7777" | sudo pfctl -f -
```

#### 2. Reverse Proxy with Nginx

Create `/etc/nginx/sites-available/pensieve`:

```nginx
server {
    listen 80;
    server_name localhost;

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;

    location / {
        limit_req zone=api burst=20 nodelay;

        proxy_pass http://127.0.0.1:7777;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    # Health check endpoint (no rate limiting)
    location /health {
        proxy_pass http://127.0.0.1:7777;
        access_log off;
    }
}
```

#### 3. SSL/TLS Configuration

```bash
# Generate self-signed certificate (for development)
sudo mkdir -p /etc/nginx/ssl
sudo openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout /etc/nginx/ssl/pensieve.key \
    -out /etc/nginx/ssl/pensieve.crt

# Update nginx config for HTTPS
# Add SSL configuration to server block
```

### Application Security

#### 1. File Permissions

```bash
# Secure configuration files
sudo chmod 600 /opt/pensieve/config/config.json
sudo chown pensieve:staff /opt/pensieve/config/config.json

# Secure model files
sudo chmod 644 /opt/pensieve/models/*.gguf
sudo chown pensieve:staff /opt/pensieve/models/*.gguf
```

#### 2. Logging and Monitoring

```bash
# Configure log rotation
sudo tee /etc/logrotate.d/pensieve << EOF
/var/log/pensieve/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 644 pensieve staff
}
EOF
```

## Monitoring and Maintenance

### Health Monitoring

#### 1. Health Check Script

Create `/opt/pensieve/scripts/health_check.sh`:

```bash
#!/bin/bash

HEALTH_URL="http://127.0.0.1:7777/health"
RESPONSE=$(curl -s "$HEALTH_URL")

if echo "$RESPONSE" | grep -q '"status":"healthy"'; then
    echo "✅ Pensieve is healthy"
    exit 0
else
    echo "❌ Pensieve is unhealthy: $RESPONSE"
    exit 1
fi
```

#### 2. Performance Monitoring

```bash
# Monitor memory usage
ps aux | grep pensieve

# Monitor network connections
netstat -an | grep 7777

# Monitor response times
curl -w "@curl-format.txt" -s -o /dev/null http://127.0.0.1:7777/health
```

Create `curl-format.txt`:

```
     time_namelookup:  %{time_namelookup}\n
        time_connect:  %{time_connect}\n
     time_appconnect:  %{time_appconnect}\n
    time_pretransfer:  %{time_pretransfer}\n
       time_redirect:  %{time_redirect}\n
  time_starttransfer:  %{time_starttransfer}\n
                     ----------\n
          time_total:  %{time_total}\n
```

### Backup and Recovery

#### 1. Configuration Backup

```bash
#!/bin/bash
# backup.sh

BACKUP_DIR="/backup/pensieve"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p "$BACKUP_DIR"

# Backup configuration
tar -czf "$BACKUP_DIR/config_$DATE.tar.gz" /opt/pensieve/config/

# Backup logs (last 7 days)
find /var/log/pensieve -name "*.log" -mtime -7 -exec tar -czf "$BACKUP_DIR/logs_$DATE.tar.gz" {} +

echo "Backup completed: $BACKUP_DIR"
```

#### 2. Model Backup

```bash
# Backup important models
rsync -av --progress /opt/pensieve/models/ /backup/pensieve/models/
```

## Troubleshooting

### Common Issues

#### 1. Server Won't Start

```bash
# Check configuration
pensieve validate -c /opt/pensieve/config/config.json

# Check logs
tail -f /var/log/pensieve/pensieve.err.log

# Check port availability
lsof -i :7777
```

#### 2. Model Loading Issues

```bash
# Verify model file
file /opt/pensieve/models/llama-2-7b-chat.gguf

# Check permissions
ls -la /opt/pensieve/models/

# Test with different GPU settings
pensieve start --model /opt/pensieve/models/model.gguf --gpu-layers 0
```

#### 3. Performance Issues

```bash
# Monitor system resources
top -o mem
top -o cpu

# Check GPU usage (if available)
sudo powermetrics --samplers gpu_power

# Adjust configuration
# Reduce context_size or gpu_layers if memory constrained
```

#### 4. Network Issues

```bash
# Test local connectivity
curl -v http://127.0.0.1:7777/health

# Check firewall rules
sudo pfctl -s rules

# Test from remote machine
curl -v http://SERVER_IP:7777/health
```

### Debug Mode

```bash
# Start with debug logging
pensieve --log-level debug start

# Enable verbose output
pensieve --verbose start
```

## Scaling Considerations

### Load Balancing

For high-availability deployments:

```nginx
upstream pensieve_backend {
    server 127.0.0.1:7777;
    server 127.0.0.1:8081;
    server 127.0.0.1:8082;
}

server {
    location / {
        proxy_pass http://pensieve_backend;
    }
}
```

### Resource Optimization

1. **Memory Management**: Adjust context size based on available RAM
2. **GPU Utilization**: Experiment with different GPU layer counts
3. **Concurrent Requests**: Limit based on hardware capabilities
4. **Model Selection**: Choose appropriate model size for workload

---

## Deployment Checklist

- [ ] System requirements met
- [ ] Binary installed and tested
- [ ] Configuration file created
- [ ] Models downloaded and verified
- [ ] Service configured and started
- [ ] Health monitoring set up
- [ ] Logging configured
- [ ] Security measures implemented
- [ ] Backup procedures established
- [ ] Performance tested
- [ ] Documentation updated

**Last Updated**: October 28, 2025
**Version**: 0.1.0
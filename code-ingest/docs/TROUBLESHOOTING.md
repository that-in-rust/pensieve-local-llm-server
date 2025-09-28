# Troubleshooting Guide

Common issues and solutions for code-ingest.

## 🔍 Quick Diagnostics

### Health Check Commands

```bash
# Check if code-ingest is installed correctly
code-ingest --version

# Test PostgreSQL connection
code-ingest list-tables --db-path postgresql://localhost/test

# Verify system dependencies
which pdftotext pandoc git

# Check system resources
free -h  # Linux
vm_stat  # macOS
```

### Debug Mode

Enable detailed logging to diagnose issues:

```bash
# Enable debug logging
export RUST_LOG=debug
code-ingest ingest <repo> --db-path ./analysis

# Trace SQL queries
export RUST_LOG=sqlx=debug,code_ingest=debug

# Full trace (very verbose)
export RUST_LOG=trace
```

## 🗄️ Database Issues

### PostgreSQL Connection Failed

**Error:**
```
Error: Database connection failed: connection to server at "localhost" (127.0.0.1), port 5432 failed
```

**Solutions:**

1. **Check if PostgreSQL is running:**
   ```bash
   # Linux/macOS
   pg_isready -h localhost -p 5432
   
   # If not running, start it:
   # macOS (Homebrew)
   brew services start postgresql
   
   # Linux (systemd)
   sudo systemctl start postgresql
   
   # Linux (service)
   sudo service postgresql start
   ```

2. **Verify database exists:**
   ```bash
   # List databases
   psql -h localhost -U postgres -l
   
   # Create database if missing
   createdb code_analysis
   ```

3. **Check connection parameters:**
   ```bash
   # Test connection manually
   psql -h localhost -U postgres -d code_analysis
   
   # If authentication fails, check pg_hba.conf
   sudo nano /etc/postgresql/15/main/pg_hba.conf  # Linux
   nano /usr/local/var/postgres/pg_hba.conf       # macOS
   ```

4. **Fix authentication issues:**
   ```bash
   # Set password for postgres user
   sudo -u postgres psql
   ALTER USER postgres PASSWORD 'your_password';
   \q
   
   # Update connection string
   code-ingest ingest <repo> --db-path "postgresql://postgres:your_password@localhost/code_analysis"
   ```

### Database Permission Denied

**Error:**
```
Error: permission denied for table ingestion_meta
```

**Solutions:**

1. **Grant necessary permissions:**
   ```sql
   -- Connect as superuser
   psql -h localhost -U postgres -d code_analysis
   
   -- Grant permissions to your user
   GRANT ALL PRIVILEGES ON DATABASE code_analysis TO your_username;
   GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO your_username;
   GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO your_username;
   ```

2. **Create user with proper permissions:**
   ```sql
   CREATE USER code_ingest_user WITH PASSWORD 'secure_password';
   GRANT ALL PRIVILEGES ON DATABASE code_analysis TO code_ingest_user;
   ALTER USER code_ingest_user CREATEDB;
   ```

### Table Already Exists Error

**Error:**
```
Error: relation "INGEST_20240928143022" already exists
```

**Solutions:**

1. **List existing tables:**
   ```bash
   code-ingest list-tables --db-path ./analysis
   ```

2. **Use existing table or drop it:**
   ```sql
   -- Drop existing table (CAUTION: data loss)
   DROP TABLE INGEST_20240928143022;
   
   -- Or query existing table
   SELECT COUNT(*) FROM INGEST_20240928143022;
   ```

3. **Clean up old ingestions:**
   ```sql
   -- List all ingestion tables
   SELECT table_name FROM information_schema.tables 
   WHERE table_name LIKE 'INGEST_%';
   
   -- Drop old tables (adjust date as needed)
   DROP TABLE INGEST_20240927120000;
   ```

## 🔐 Authentication Issues

### GitHub Authentication Failed

**Error:**
```
Error: GitHub authentication failed: Bad credentials
```

**Solutions:**

1. **Check token validity:**
   ```bash
   # Test token with curl
   curl -H "Authorization: token $GITHUB_TOKEN" https://api.github.com/user
   
   # Should return user information, not 401 Unauthorized
   ```

2. **Generate new token:**
   - Go to GitHub Settings → Developer settings → Personal access tokens
   - Generate new token with `repo` scope for private repositories
   - Set environment variable:
     ```bash
     export GITHUB_TOKEN="ghp_xxxxxxxxxxxxxxxxxxxx"
     ```

3. **Token scope issues:**
   ```bash
   # Check token scopes
   curl -H "Authorization: token $GITHUB_TOKEN" -I https://api.github.com/user
   # Look for X-OAuth-Scopes header
   
   # For private repos, ensure token has 'repo' scope
   # For public repos, 'public_repo' scope is sufficient
   ```

4. **Organization restrictions:**
   - Check if your organization has SSO enabled
   - Authorize token for SSO if required
   - Contact organization admin if access is restricted

### SSH Key Issues (for Git operations)

**Error:**
```
Error: Permission denied (publickey)
```

**Solutions:**

1. **Use HTTPS instead of SSH:**
   ```bash
   # Instead of: git@github.com:user/repo.git
   # Use: https://github.com/user/repo.git
   code-ingest ingest https://github.com/user/repo --token $GITHUB_TOKEN
   ```

2. **Fix SSH configuration:**
   ```bash
   # Test SSH connection
   ssh -T git@github.com
   
   # Add SSH key if needed
   ssh-add ~/.ssh/id_rsa
   
   # Generate new SSH key if needed
   ssh-keygen -t ed25519 -C "your_email@example.com"
   ```

## 📁 File Processing Issues

### File Too Large Error

**Error:**
```
Error: File exceeds maximum size limit: 15.2MB > 10MB
```

**Solutions:**

1. **Increase file size limit:**
   ```bash
   code-ingest ingest <repo> --max-file-size 52428800  # 50MB
   ```

2. **Skip large files:**
   ```bash
   # Files larger than limit are automatically skipped
   # Check processing logs for skipped files
   export RUST_LOG=info
   code-ingest ingest <repo> --db-path ./analysis
   ```

3. **Filter files before processing:**
   ```bash
   # Use .gitignore patterns to exclude large files
   echo "*.zip" >> .gitignore
   echo "*.tar.gz" >> .gitignore
   echo "node_modules/" >> .gitignore
   ```

### Conversion Tool Missing

**Error:**
```
Error: External command failed: pdftotext: command not found
```

**Solutions:**

1. **Install missing tools:**
   ```bash
   # PDF processing
   sudo apt-get install poppler-utils  # Linux
   brew install poppler               # macOS
   
   # Office documents
   sudo apt-get install pandoc        # Linux
   brew install pandoc               # macOS
   
   # Verify installation
   which pdftotext pandoc
   ```

2. **Skip convertible files:**
   ```bash
   # Convertible files will be marked as skipped if tools are missing
   # Check processing results for skipped files
   code-ingest sql "SELECT COUNT(*) FROM INGEST_20240928143022 WHERE file_type = 'convertible' AND content_text IS NULL"
   ```

### Encoding Issues

**Error:**
```
Error: Invalid UTF-8 sequence in file: src/legacy.c
```

**Solutions:**

1. **Check file encoding:**
   ```bash
   file -bi src/legacy.c
   # Should show charset information
   ```

2. **Convert file encoding:**
   ```bash
   # Convert to UTF-8
   iconv -f ISO-8859-1 -t UTF-8 src/legacy.c > src/legacy_utf8.c
   
   # Or use dos2unix for Windows line endings
   dos2unix src/legacy.c
   ```

3. **Skip problematic files:**
   ```bash
   # Files with encoding issues are automatically skipped
   # Check logs for details
   export RUST_LOG=warn
   code-ingest ingest <repo> --db-path ./analysis
   ```

## ⚡ Performance Issues

### Slow Ingestion Performance

**Symptoms:**
- Ingestion takes much longer than expected
- High CPU or memory usage
- System becomes unresponsive

**Solutions:**

1. **Adjust concurrency:**
   ```bash
   # Reduce concurrent processing
   code-ingest ingest <repo> --max-concurrency 2
   
   # Or set environment variable
   export CODE_INGEST_MAX_CONCURRENCY=4
   ```

2. **Monitor system resources:**
   ```bash
   # Check CPU and memory usage
   htop  # Linux
   top   # macOS/Linux
   
   # Check disk I/O
   iotop  # Linux
   
   # Check PostgreSQL performance
   SELECT * FROM pg_stat_activity;
   ```

3. **Optimize PostgreSQL:**
   ```sql
   -- Increase memory settings (adjust based on available RAM)
   -- Add to postgresql.conf:
   shared_buffers = 256MB
   work_mem = 4MB
   maintenance_work_mem = 64MB
   effective_cache_size = 1GB
   
   -- Restart PostgreSQL after changes
   ```

4. **Use SSD storage:**
   ```bash
   # Move database to SSD if using HDD
   # Update connection string to point to SSD location
   ```

### Memory Usage Too High

**Error:**
```
Error: System out of memory during processing
```

**Solutions:**

1. **Set memory limits:**
   ```bash
   export CODE_INGEST_MAX_MEMORY_MB=512  # Limit to 512MB
   code-ingest ingest <repo> --db-path ./analysis
   ```

2. **Process in smaller batches:**
   ```bash
   # Reduce concurrency to lower memory usage
   code-ingest ingest <repo> --max-concurrency 1 --db-path ./analysis
   ```

3. **Monitor memory usage:**
   ```bash
   # Linux
   watch -n 1 'free -h'
   
   # macOS
   while true; do vm_stat; sleep 1; done
   ```

### Slow Query Performance

**Symptoms:**
- SQL queries take a long time to execute
- Database becomes unresponsive during queries

**Solutions:**

1. **Add database indexes:**
   ```sql
   -- Index on file extension
   CREATE INDEX idx_extension ON INGEST_20240928143022(extension);
   
   -- Index on file type
   CREATE INDEX idx_file_type ON INGEST_20240928143022(file_type);
   
   -- Index on file path for pattern matching
   CREATE INDEX idx_filepath_pattern ON INGEST_20240928143022 USING gin(filepath gin_trgm_ops);
   ```

2. **Optimize queries:**
   ```sql
   -- Use EXPLAIN ANALYZE to understand query performance
   EXPLAIN ANALYZE SELECT * FROM INGEST_20240928143022 WHERE content_text LIKE '%function%';
   
   -- Use full-text search instead of LIKE for content searches
   SELECT * FROM INGEST_20240928143022 
   WHERE to_tsvector('english', content_text) @@ plainto_tsquery('english', 'function');
   ```

3. **Limit result sets:**
   ```bash
   # Always use LIMIT for large queries
   code-ingest sql "SELECT * FROM INGEST_20240928143022 LIMIT 100"
   
   # Use pagination for large result sets
   code-ingest sql "SELECT * FROM INGEST_20240928143022 ORDER BY file_id LIMIT 100 OFFSET 200"
   ```

## 🌐 Network Issues

### Repository Clone Failed

**Error:**
```
Error: Failed to clone repository: network unreachable
```

**Solutions:**

1. **Check network connectivity:**
   ```bash
   # Test GitHub connectivity
   ping github.com
   curl -I https://github.com
   
   # Test specific repository
   curl -I https://github.com/user/repo
   ```

2. **Use different clone method:**
   ```bash
   # Try HTTPS instead of SSH
   code-ingest ingest https://github.com/user/repo.git
   
   # Use token for private repos
   code-ingest ingest https://github.com/user/repo.git --token $GITHUB_TOKEN
   ```

3. **Configure proxy settings:**
   ```bash
   # Set HTTP proxy
   export HTTP_PROXY=http://proxy.company.com:8080
   export HTTPS_PROXY=http://proxy.company.com:8080
   
   # Configure Git proxy
   git config --global http.proxy http://proxy.company.com:8080
   ```

4. **Increase timeout:**
   ```bash
   # Git timeout settings
   git config --global http.lowSpeedLimit 0
   git config --global http.lowSpeedTime 999999
   ```

### Rate Limiting

**Error:**
```
Error: GitHub API rate limit exceeded
```

**Solutions:**

1. **Use authenticated requests:**
   ```bash
   # Authenticated requests have higher rate limits
   export GITHUB_TOKEN="your_token"
   code-ingest ingest https://github.com/user/repo
   ```

2. **Wait for rate limit reset:**
   ```bash
   # Check rate limit status
   curl -H "Authorization: token $GITHUB_TOKEN" https://api.github.com/rate_limit
   
   # Wait until reset time or use different token
   ```

3. **Use local clone:**
   ```bash
   # Clone manually and process local folder
   git clone https://github.com/user/repo.git
   code-ingest ingest ./repo --db-path ./analysis
   ```

## 🔧 System-Specific Issues

### macOS Issues

**Xcode Command Line Tools Missing:**
```bash
# Install Xcode command line tools
xcode-select --install
```

**Homebrew Permission Issues:**
```bash
# Fix Homebrew permissions
sudo chown -R $(whoami) /usr/local/var/homebrew
```

**PostgreSQL Service Issues:**
```bash
# Check Homebrew services
brew services list | grep postgresql

# Restart PostgreSQL
brew services restart postgresql
```

### Linux Issues

**Package Manager Issues:**
```bash
# Update package lists
sudo apt-get update

# Install build essentials
sudo apt-get install build-essential pkg-config libssl-dev

# Install PostgreSQL development headers
sudo apt-get install libpq-dev
```

**SystemD Service Issues:**
```bash
# Check PostgreSQL status
sudo systemctl status postgresql

# Enable auto-start
sudo systemctl enable postgresql

# Check logs
sudo journalctl -u postgresql
```

### Windows Issues

**PowerShell Execution Policy:**
```powershell
# Allow script execution
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Path Issues:**
```cmd
# Add PostgreSQL to PATH
set PATH=%PATH%;C:\Program Files\PostgreSQL\15\bin

# Make permanent
setx PATH "%PATH%;C:\Program Files\PostgreSQL\15\bin"
```

**WSL Integration:**
```bash
# Use WSL for better compatibility
wsl --install
# Then run code-ingest inside WSL
```

## 🚨 Emergency Recovery

### Corrupted Database

**Symptoms:**
- Database connection errors
- Data corruption messages
- Unexpected query results

**Recovery Steps:**

1. **Stop all connections:**
   ```sql
   SELECT pg_terminate_backend(pid) FROM pg_stat_activity 
   WHERE datname = 'code_analysis' AND pid <> pg_backend_pid();
   ```

2. **Check database integrity:**
   ```bash
   # Check for corruption
   pg_dump code_analysis > /dev/null
   
   # Vacuum and analyze
   psql -d code_analysis -c "VACUUM FULL ANALYZE;"
   ```

3. **Restore from backup:**
   ```bash
   # If you have a backup
   dropdb code_analysis
   createdb code_analysis
   psql code_analysis < backup.sql
   ```

4. **Recreate database:**
   ```bash
   # Last resort: start fresh
   dropdb code_analysis
   createdb code_analysis
   # Re-run ingestion
   ```

### Disk Space Full

**Error:**
```
Error: No space left on device
```

**Solutions:**

1. **Clean up old ingestions:**
   ```sql
   -- Drop old ingestion tables
   DROP TABLE INGEST_20240901120000;
   DROP TABLE INGEST_20240902120000;
   
   -- Vacuum to reclaim space
   VACUUM FULL;
   ```

2. **Move database to larger disk:**
   ```bash
   # Stop PostgreSQL
   sudo systemctl stop postgresql
   
   # Move data directory
   sudo mv /var/lib/postgresql/15/main /new/larger/disk/
   
   # Update configuration
   sudo nano /etc/postgresql/15/main/postgresql.conf
   # Change data_directory = '/new/larger/disk/main'
   
   # Start PostgreSQL
   sudo systemctl start postgresql
   ```

3. **Clean up temporary files:**
   ```bash
   # Clean up /tmp
   sudo rm -rf /tmp/code-ingest-*
   
   # Clean up logs
   sudo journalctl --vacuum-time=7d
   ```

## 📞 Getting Help

### Collect Debug Information

Before reporting issues, collect this information:

```bash
# System information
uname -a
code-ingest --version

# PostgreSQL version
psql --version
psql -c "SELECT version();"

# Environment variables
env | grep -E "(DATABASE_URL|GITHUB_TOKEN|RUST_LOG|CODE_INGEST_)"

# Recent logs
export RUST_LOG=debug
code-ingest ingest <repo> --db-path ./analysis 2>&1 | tail -100
```

### Report Issues

1. **GitHub Issues**: https://github.com/your-org/code-ingest/issues
2. **Discussions**: https://github.com/your-org/code-ingest/discussions
3. **Email**: support@your-org.com

### Include in Bug Reports

- Operating system and version
- code-ingest version
- PostgreSQL version
- Complete error message
- Steps to reproduce
- Debug logs (with sensitive information removed)

### Community Resources

- **Documentation**: https://docs.code-ingest.dev
- **Examples**: https://github.com/your-org/code-ingest/tree/main/examples
- **FAQ**: https://docs.code-ingest.dev/faq
- **Discord**: https://discord.gg/code-ingest
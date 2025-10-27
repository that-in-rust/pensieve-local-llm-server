# Security Analysis Example

Use code-ingest to identify potential security vulnerabilities and sensitive information in codebases.

## 🎯 Objective

Perform a comprehensive security analysis to find:
- Hardcoded secrets and credentials
- Potential SQL injection vulnerabilities
- Insecure cryptographic practices
- Authentication and authorization issues
- Input validation problems

## 🛡️ Security Patterns to Look For

### 1. Hardcoded Secrets

#### API Keys and Tokens
```bash
code-ingest sql "SELECT filepath, 
                        SUBSTRING(content_text FROM position('api[_-]?key' in lower(content_text)) FOR 100) as context
                 FROM INGEST_20240928143022 
                 WHERE lower(content_text) ~ 'api[_-]?key\s*[=:]\s*[\"''][a-zA-Z0-9]{20,}'
                 ORDER BY filepath" --db-path ./analysis
```

#### Database Credentials
```bash
code-ingest sql "SELECT filepath, line_count,
                        CASE 
                          WHEN lower(content_text) LIKE '%password%=%' THEN 'Password in config'
                          WHEN lower(content_text) LIKE '%mysql://%:%@%' THEN 'MySQL connection string'
                          WHEN lower(content_text) LIKE '%postgresql://%:%@%' THEN 'PostgreSQL connection string'
                          WHEN lower(content_text) LIKE '%mongodb://%:%@%' THEN 'MongoDB connection string'
                        END as credential_type
                 FROM INGEST_20240928143022 
                 WHERE lower(content_text) ~ '(password\s*[=:]\s*[\"''][^\"'']+|[a-z]+://[^:]+:[^@]+@)'
                   AND file_type = 'direct_text'
                 ORDER BY filepath" --db-path ./analysis
```

#### Private Keys and Certificates
```bash
code-ingest sql "SELECT filepath, filename,
                        CASE 
                          WHEN content_text LIKE '%BEGIN PRIVATE KEY%' THEN 'Private Key'
                          WHEN content_text LIKE '%BEGIN RSA PRIVATE KEY%' THEN 'RSA Private Key'
                          WHEN content_text LIKE '%BEGIN CERTIFICATE%' THEN 'Certificate'
                          WHEN lower(content_text) ~ 'private[_-]?key\s*[=:]\s*[\"''][a-zA-Z0-9+/=]{100,}' THEN 'Encoded Private Key'
                        END as key_type
                 FROM INGEST_20240928143022 
                 WHERE content_text ~ '(BEGIN (RSA )?PRIVATE KEY|BEGIN CERTIFICATE|private[_-]?key\s*[=:]\s*[\"''][a-zA-Z0-9+/=]{100,})'
                 ORDER BY filepath" --db-path ./analysis
```

### 2. SQL Injection Vulnerabilities

#### Dynamic Query Construction
```bash
code-ingest sql "SELECT filepath, 
                        COUNT(*) as potential_sql_injections,
                        string_agg(DISTINCT 
                          CASE 
                            WHEN content_text ~ 'query\s*\+\s*[a-zA-Z_]' THEN 'String concatenation'
                            WHEN content_text ~ 'format!\s*\(\s*[\"''].*SELECT.*\{' THEN 'Format macro with SELECT'
                            WHEN content_text ~ 'sprintf.*SELECT.*%' THEN 'sprintf with SELECT'
                          END, ', ') as vulnerability_types
                 FROM INGEST_20240928143022 
                 WHERE content_text ~ '(query\s*\+\s*|format!\s*\(\s*[\"''].*SELECT.*\{|sprintf.*SELECT.*%)'
                   AND extension IN ('rs', 'py', 'js', 'java', 'php', 'c', 'cpp')
                 GROUP BY filepath
                 ORDER BY potential_sql_injections DESC" --db-path ./analysis
```

#### Unsafe Database Queries
```bash
code-ingest sql "SELECT filepath,
                        SUBSTRING(content_text FROM position('execute' in lower(content_text)) - 20 FOR 100) as context
                 FROM INGEST_20240928143022 
                 WHERE lower(content_text) ~ '(execute|query)\s*\(\s*[\"''].*\$.*[\"'']'
                   AND extension IN ('rs', 'py', 'js', 'java', 'php')
                 ORDER BY filepath" --db-path ./analysis
```

### 3. Cryptographic Issues

#### Weak Cryptographic Algorithms
```bash
code-ingest sql "SELECT filepath,
                        string_agg(DISTINCT 
                          CASE 
                            WHEN lower(content_text) LIKE '%md5%' THEN 'MD5'
                            WHEN lower(content_text) LIKE '%sha1%' THEN 'SHA1'
                            WHEN lower(content_text) LIKE '%des%' THEN 'DES'
                            WHEN lower(content_text) LIKE '%rc4%' THEN 'RC4'
                          END, ', ') as weak_algorithms
                 FROM INGEST_20240928143022 
                 WHERE lower(content_text) ~ '(md5|sha1|\\bdes\\b|rc4)'
                   AND file_type = 'direct_text'
                 GROUP BY filepath
                 HAVING string_agg(DISTINCT 
                          CASE 
                            WHEN lower(content_text) LIKE '%md5%' THEN 'MD5'
                            WHEN lower(content_text) LIKE '%sha1%' THEN 'SHA1'
                            WHEN lower(content_text) LIKE '%des%' THEN 'DES'
                            WHEN lower(content_text) LIKE '%rc4%' THEN 'RC4'
                          END, ', ') IS NOT NULL
                 ORDER BY filepath" --db-path ./analysis
```

#### Hardcoded Cryptographic Keys
```bash
code-ingest sql "SELECT filepath,
                        CASE 
                          WHEN content_text ~ 'key\s*[=:]\s*[\"''][a-fA-F0-9]{32,}[\"'']' THEN 'Hex key'
                          WHEN content_text ~ 'secret\s*[=:]\s*[\"''][a-zA-Z0-9+/=]{20,}[\"'']' THEN 'Base64 secret'
                          WHEN content_text ~ 'iv\s*[=:]\s*[\"''][a-fA-F0-9]{16,}[\"'']' THEN 'Initialization Vector'
                        END as key_type
                 FROM INGEST_20240928143022 
                 WHERE content_text ~ '(key|secret|iv)\s*[=:]\s*[\"''][a-fA-F0-9A-Z+/=]{16,}[\"'']'
                   AND file_type = 'direct_text'
                 ORDER BY filepath" --db-path ./analysis
```

### 4. Authentication and Authorization

#### Weak Authentication Patterns
```bash
code-ingest sql "SELECT filepath,
                        string_agg(DISTINCT 
                          CASE 
                            WHEN lower(content_text) LIKE '%password == %' THEN 'Plain text password comparison'
                            WHEN lower(content_text) ~ 'auth.*=.*true' THEN 'Hardcoded auth bypass'
                            WHEN lower(content_text) LIKE '%admin == true%' THEN 'Hardcoded admin access'
                            WHEN lower(content_text) ~ 'if.*user.*==.*[\"'']admin[\"'']' THEN 'Hardcoded admin user'
                          END, ', ') as auth_issues
                 FROM INGEST_20240928143022 
                 WHERE lower(content_text) ~ '(password\s*==|auth.*=.*true|admin\s*==\s*true|if.*user.*==.*[\"'']admin[\"''])'
                   AND file_type = 'direct_text'
                 GROUP BY filepath
                 HAVING string_agg(DISTINCT 
                          CASE 
                            WHEN lower(content_text) LIKE '%password == %' THEN 'Plain text password comparison'
                            WHEN lower(content_text) ~ 'auth.*=.*true' THEN 'Hardcoded auth bypass'
                            WHEN lower(content_text) LIKE '%admin == true%' THEN 'Hardcoded admin access'
                            WHEN lower(content_text) ~ 'if.*user.*==.*[\"'']admin[\"'']' THEN 'Hardcoded admin user'
                          END, ', ') IS NOT NULL
                 ORDER BY filepath" --db-path ./analysis
```

#### Missing Authorization Checks
```bash
code-ingest sql "SELECT filepath,
                        COUNT(*) as endpoint_count,
                        SUM(CASE WHEN lower(content_text) ~ '(authorize|permission|role|access)' THEN 1 ELSE 0 END) as auth_checks
                 FROM INGEST_20240928143022 
                 WHERE lower(content_text) ~ '(route|endpoint|handler|controller)'
                   AND extension IN ('rs', 'py', 'js', 'java', 'php')
                 GROUP BY filepath
                 HAVING COUNT(*) > 0
                 ORDER BY (endpoint_count - auth_checks) DESC" --db-path ./analysis
```

### 5. Input Validation Issues

#### Unsafe Input Handling
```bash
code-ingest sql "SELECT filepath,
                        string_agg(DISTINCT 
                          CASE 
                            WHEN content_text ~ 'eval\s*\(' THEN 'eval() usage'
                            WHEN content_text ~ 'exec\s*\(' THEN 'exec() usage'
                            WHEN content_text ~ 'system\s*\(' THEN 'system() call'
                            WHEN content_text ~ 'shell_exec\s*\(' THEN 'shell_exec() usage'
                          END, ', ') as unsafe_functions
                 FROM INGEST_20240928143022 
                 WHERE content_text ~ '(eval|exec|system|shell_exec)\s*\('
                   AND extension IN ('py', 'js', 'php', 'rb')
                 GROUP BY filepath
                 HAVING string_agg(DISTINCT 
                          CASE 
                            WHEN content_text ~ 'eval\s*\(' THEN 'eval() usage'
                            WHEN content_text ~ 'exec\s*\(' THEN 'exec() usage'
                            WHEN content_text ~ 'system\s*\(' THEN 'system() call'
                            WHEN content_text ~ 'shell_exec\s*\(' THEN 'shell_exec() usage'
                          END, ', ') IS NOT NULL
                 ORDER BY filepath" --db-path ./analysis
```

## 🔍 Comprehensive Security Analysis Workflow

### Step 1: Prepare Security Analysis
```bash
# Create a comprehensive security query
code-ingest query-prepare "
  SELECT filepath, content_text, extension, line_count
  FROM INGEST_20240928143022 
  WHERE file_type = 'direct_text'
    AND (
      lower(content_text) ~ '(password|secret|key|token|credential)' OR
      lower(content_text) ~ '(sql|query|execute|prepare)' OR
      lower(content_text) ~ '(auth|login|admin|permission)' OR
      lower(content_text) ~ '(eval|exec|system|shell)' OR
      lower(content_text) ~ '(md5|sha1|des|rc4)'
    )
  ORDER BY filepath
" --temp-path ./security-temp.txt \
  --tasks-file ./security-tasks.md \
  --output-table QUERYRESULT_security_analysis \
  --db-path ./analysis
```

### Step 2: Create Security Analysis Prompt
```bash
cat > security-analysis-prompt.md << 'EOF'
# Security Analysis Prompt

Analyze the provided code files for security vulnerabilities. For each file, identify:

## 1. Credential and Secret Management
- Hardcoded passwords, API keys, tokens
- Database connection strings with credentials
- Private keys or certificates in code
- Weak or default credentials

## 2. Input Validation and Sanitization
- SQL injection vulnerabilities
- Command injection risks
- Cross-site scripting (XSS) potential
- Path traversal vulnerabilities
- Unsafe deserialization

## 3. Authentication and Authorization
- Weak authentication mechanisms
- Missing authorization checks
- Privilege escalation risks
- Session management issues

## 4. Cryptographic Issues
- Use of weak algorithms (MD5, SHA1, DES, RC4)
- Hardcoded cryptographic keys
- Improper random number generation
- Weak key derivation

## 5. Error Handling and Information Disclosure
- Verbose error messages
- Stack traces in production
- Debug information exposure
- Sensitive data in logs

For each issue found, provide:
- **Severity**: Critical/High/Medium/Low
- **Location**: File path and approximate line
- **Description**: What the vulnerability is
- **Impact**: Potential security impact
- **Recommendation**: How to fix it

Format as structured markdown with clear sections.
EOF
```

### Step 3: Generate Security Tasks
```bash
code-ingest generate-tasks \
  --sql "SELECT * FROM QUERYRESULT_security_analysis" \
  --prompt-file security-analysis-prompt.md \
  --output-table QUERYRESULT_security_findings \
  --tasks-file ./security-review-tasks.md \
  --db-path ./analysis
```

### Step 4: Execute Security Analysis
Open `security-review-tasks.md` in your IDE and execute the analysis tasks systematically.

### Step 5: Store and Export Results
```bash
# Store analysis results
code-ingest store-result \
  --output-table QUERYRESULT_security_findings \
  --result-file ./security-analysis-results.txt \
  --original-query "Security analysis of codebase" \
  --db-path ./analysis

# Export individual findings as markdown files
code-ingest print-to-md \
  --table QUERYRESULT_security_findings \
  --sql "SELECT * FROM QUERYRESULT_security_findings WHERE severity IN ('Critical', 'High')" \
  --prefix security-finding \
  --location ./security-reports/ \
  --db-path ./analysis
```

## 📊 Security Metrics Dashboard

### Overall Security Score
```bash
code-ingest sql "
  WITH security_metrics AS (
    SELECT 
      COUNT(*) as total_files,
      SUM(CASE WHEN lower(content_text) ~ '(password|secret|key)\s*[=:]' THEN 1 ELSE 0 END) as credential_files,
      SUM(CASE WHEN lower(content_text) ~ '(md5|sha1|des|rc4)' THEN 1 ELSE 0 END) as weak_crypto_files,
      SUM(CASE WHEN lower(content_text) ~ '(eval|exec|system)\s*\(' THEN 1 ELSE 0 END) as unsafe_exec_files,
      SUM(CASE WHEN lower(content_text) ~ 'query.*\+.*[a-zA-Z_]' THEN 1 ELSE 0 END) as sql_injection_files
    FROM INGEST_20240928143022 
    WHERE file_type = 'direct_text'
  )
  SELECT 
    total_files,
    credential_files,
    weak_crypto_files,
    unsafe_exec_files,
    sql_injection_files,
    ROUND(
      (1.0 - (credential_files + weak_crypto_files + unsafe_exec_files + sql_injection_files)::float / total_files) * 100, 
      2
    ) as security_score_percentage
  FROM security_metrics
" --db-path ./analysis
```

### Security Issues by Category
```bash
code-ingest sql "
  SELECT 
    'Credential Management' as category,
    COUNT(*) as issue_count
  FROM INGEST_20240928143022 
  WHERE lower(content_text) ~ '(password|secret|key|token)\s*[=:]'
    AND file_type = 'direct_text'
  
  UNION ALL
  
  SELECT 
    'Weak Cryptography' as category,
    COUNT(*) as issue_count
  FROM INGEST_20240928143022 
  WHERE lower(content_text) ~ '(md5|sha1|des|rc4)'
    AND file_type = 'direct_text'
  
  UNION ALL
  
  SELECT 
    'Unsafe Execution' as category,
    COUNT(*) as issue_count
  FROM INGEST_20240928143022 
  WHERE lower(content_text) ~ '(eval|exec|system|shell_exec)\s*\('
    AND file_type = 'direct_text'
  
  UNION ALL
  
  SELECT 
    'SQL Injection Risk' as category,
    COUNT(*) as issue_count
  FROM INGEST_20240928143022 
  WHERE lower(content_text) ~ '(query|execute).*\+.*[a-zA-Z_]'
    AND file_type = 'direct_text'
  
  ORDER BY issue_count DESC
" --db-path ./analysis
```

## 🚨 Critical Security Checklist

After running the analysis, verify these critical security aspects:

### ✅ Secrets Management
- [ ] No hardcoded API keys or tokens
- [ ] No database credentials in code
- [ ] No private keys or certificates in repository
- [ ] Secrets properly externalized to environment variables

### ✅ Input Validation
- [ ] All user inputs are validated and sanitized
- [ ] SQL queries use parameterized statements
- [ ] No direct execution of user-provided code
- [ ] File uploads are properly restricted

### ✅ Authentication & Authorization
- [ ] Strong authentication mechanisms in place
- [ ] Authorization checks on all protected resources
- [ ] No hardcoded admin credentials
- [ ] Proper session management

### ✅ Cryptography
- [ ] Strong cryptographic algorithms (AES, SHA-256+)
- [ ] No hardcoded cryptographic keys
- [ ] Proper random number generation
- [ ] Secure key derivation functions

### ✅ Error Handling
- [ ] No sensitive information in error messages
- [ ] Proper logging without exposing secrets
- [ ] Graceful error handling
- [ ] No debug information in production

## 🔧 Remediation Priorities

1. **Critical**: Hardcoded secrets and credentials
2. **High**: SQL injection vulnerabilities
3. **High**: Weak cryptographic algorithms
4. **Medium**: Missing authorization checks
5. **Medium**: Unsafe input handling
6. **Low**: Information disclosure in errors

## 📈 Continuous Security Monitoring

Set up regular security scans:

```bash
#!/bin/bash
# security-scan.sh - Run weekly security analysis

REPO_URL="$1"
ANALYSIS_DIR="./security-analysis-$(date +%Y%m%d)"

# Ingest latest code
code-ingest ingest "$REPO_URL" --db-path "$ANALYSIS_DIR"

# Run security analysis
code-ingest sql "/* Security queries here */" --db-path "$ANALYSIS_DIR" > security-report-$(date +%Y%m%d).txt

# Alert if critical issues found
if grep -q "Critical\|High" security-report-$(date +%Y%m%d).txt; then
    echo "ALERT: Critical security issues found!"
    # Send notification
fi
```

This comprehensive security analysis approach helps identify and prioritize security vulnerabilities in your codebase systematically.
# IDE Integration Example

Learn how to use code-ingest with your IDE (especially Kiro) for systematic code analysis workflows.

## 🎯 Objective

Set up a complete IDE integration workflow that:
- Prepares code analysis tasks systematically
- Leverages AI assistance for deep code understanding
- Stores and tracks analysis results
- Creates reusable analysis workflows

## 🔧 Prerequisites

- code-ingest installed and working
- Kiro IDE or compatible IDE with task execution support
- A repository already ingested into PostgreSQL

## 🚀 Complete IDE Integration Workflow

### Step 1: Prepare Analysis Query

First, identify what you want to analyze. Let's analyze authentication patterns in a codebase:

```bash
# Prepare authentication-related code for analysis
code-ingest query-prepare "
  SELECT filepath, content_text, line_count, word_count
  FROM INGEST_20240928143022 
  WHERE (
    lower(content_text) LIKE '%auth%' OR
    lower(content_text) LIKE '%login%' OR
    lower(content_text) LIKE '%password%' OR
    lower(content_text) LIKE '%token%' OR
    lower(content_text) LIKE '%session%'
  )
  AND file_type = 'direct_text'
  AND extension IN ('rs', 'py', 'js', 'ts', 'java', 'go')
  ORDER BY line_count DESC
" --temp-path ./auth-analysis-temp.txt \
  --tasks-file ./auth-analysis-tasks.md \
  --output-table QUERYRESULT_auth_analysis \
  --db-path ./analysis

# Expected output:
# 🔍 Preparing query for IDE analysis...
# 📊 Query executed: 23 results found
# 📝 Results written to: ./auth-analysis-temp.txt
# 📋 Tasks file created: ./auth-analysis-tasks.md
# 🗄️  Output table created: QUERYRESULT_auth_analysis
```

### Step 2: Create Analysis Prompt

Create a detailed prompt file for the AI analysis:

```bash
cat > auth-analysis-prompt.md << 'EOF'
# Authentication System Analysis

Analyze the provided code files to understand the authentication system. For each file, provide:

## 1. Authentication Mechanisms
- What authentication methods are implemented?
- How are credentials validated?
- What authentication libraries/frameworks are used?
- Are there any custom authentication implementations?

## 2. Security Assessment
- Are passwords properly hashed?
- Is there protection against brute force attacks?
- Are sessions managed securely?
- Are there any obvious security vulnerabilities?

## 3. Architecture Analysis
- How is authentication integrated into the application flow?
- What are the key authentication components?
- How do different parts of the system interact for auth?
- Are there any authentication middleware or decorators?

## 4. Token Management
- How are authentication tokens generated and validated?
- What is the token lifecycle (creation, refresh, expiration)?
- Are tokens stored securely?
- Is there proper token revocation?

## 5. User Management
- How are user accounts created and managed?
- What user roles and permissions exist?
- How is authorization handled after authentication?
- Are there admin/privileged account protections?

## 6. Integration Points
- How does authentication integrate with external systems?
- Are there OAuth, SAML, or other SSO implementations?
- How are API keys or service-to-service auth handled?

For each file analyzed, provide:
- **Purpose**: What role this file plays in authentication
- **Key Functions**: Important functions/methods and their purposes
- **Security Notes**: Any security considerations or concerns
- **Dependencies**: What other components this relies on
- **Recommendations**: Suggestions for improvements

Format the analysis as structured markdown with clear sections and code references.
EOF
```

### Step 3: Generate Systematic Analysis Tasks

```bash
code-ingest generate-tasks \
  --sql "SELECT filepath, content_text FROM QUERYRESULT_auth_analysis ORDER BY line_count DESC" \
  --prompt-file auth-analysis-prompt.md \
  --output-table QUERYRESULT_auth_detailed_analysis \
  --tasks-file ./auth-systematic-tasks.md \
  --db-path ./analysis

# Expected output:
# 📋 Generating analysis tasks...
# 🔍 Query executed: 23 files selected
# 📝 Prompt file: auth-analysis-prompt.md
# 🎯 Dividing into 7 task groups (3-4 files each)
# 📄 Tasks file created: ./auth-systematic-tasks.md
# 🗄️  Output table created: QUERYRESULT_auth_detailed_analysis
```

### Step 4: Execute Tasks in IDE

Open `auth-systematic-tasks.md` in Kiro IDE. The file will look like this:

```markdown
# Authentication System Analysis Tasks

## Task Overview
Systematic analysis of 23 authentication-related files divided into 7 groups for comprehensive review.

- [ ] 1. Core Authentication Components (Files 1-4)
  - [ ] 1.1 Analyze src/auth/mod.rs - Main authentication module
  - [ ] 1.2 Analyze src/auth/login.rs - Login functionality
  - [ ] 1.3 Analyze src/auth/token.rs - Token management
  - [ ] 1.4 Analyze src/auth/middleware.rs - Authentication middleware

- [ ] 2. User Management System (Files 5-7)
  - [ ] 2.1 Analyze src/user/model.rs - User data model
  - [ ] 2.2 Analyze src/user/service.rs - User service layer
  - [ ] 2.3 Analyze src/user/repository.rs - User data access

- [ ] 3. Session Management (Files 8-11)
  - [ ] 3.1 Analyze src/session/manager.rs - Session management
  - [ ] 3.2 Analyze src/session/store.rs - Session storage
  - [ ] 3.3 Analyze src/session/cookie.rs - Cookie handling
  - [ ] 3.4 Analyze src/session/redis.rs - Redis session backend

- [ ] 4. API Authentication (Files 12-15)
  - [ ] 4.1 Analyze src/api/auth.rs - API authentication
  - [ ] 4.2 Analyze src/api/jwt.rs - JWT token handling
  - [ ] 4.3 Analyze src/api/oauth.rs - OAuth implementation
  - [ ] 4.4 Analyze src/api/keys.rs - API key management

- [ ] 5. Security Components (Files 16-19)
  - [ ] 5.1 Analyze src/security/hash.rs - Password hashing
  - [ ] 5.2 Analyze src/security/crypto.rs - Cryptographic functions
  - [ ] 5.3 Analyze src/security/validation.rs - Input validation
  - [ ] 5.4 Analyze src/security/rate_limit.rs - Rate limiting

- [ ] 6. External Integrations (Files 20-21)
  - [ ] 6.1 Analyze src/integrations/ldap.rs - LDAP integration
  - [ ] 6.2 Analyze src/integrations/saml.rs - SAML implementation

- [ ] 7. Testing and Configuration (Files 22-23)
  - [ ] 7.1 Analyze tests/auth_integration_test.rs - Integration tests
  - [ ] 7.2 Analyze config/auth.toml - Authentication configuration
```

In Kiro IDE:
1. Click "Start Task" next to task 1
2. The IDE will read the temp file and apply the analysis prompt
3. Review and refine the AI analysis
4. Mark the task as complete
5. Repeat for all 7 task groups

### Step 5: Store Analysis Results

After completing the analysis in your IDE, store the results:

```bash
# Store the detailed analysis results
code-ingest store-result \
  --output-table QUERYRESULT_auth_detailed_analysis \
  --result-file ./auth-analysis-complete.txt \
  --original-query "Comprehensive authentication system analysis" \
  --db-path ./analysis

# Expected output:
# 💾 Storing analysis results...
# 📊 Result file: ./auth-analysis-complete.txt (15.7 KB)
# 🗄️  Output table: QUERYRESULT_auth_detailed_analysis
# ✅ Results stored successfully!
```

### Step 6: Export Individual Analysis Reports

```bash
# Export each analysis section as individual markdown files
code-ingest print-to-md \
  --table QUERYRESULT_auth_detailed_analysis \
  --sql "SELECT * FROM QUERYRESULT_auth_detailed_analysis ORDER BY created_at" \
  --prefix auth-analysis \
  --location ./auth-reports/ \
  --db-path ./analysis

# Expected output:
# 📄 Exporting query results to markdown files...
# 🔍 Query executed: 7 results found
# 📁 Output location: ./auth-reports/
# 📝 Generated files:
#    - auth-analysis-00001.md (Core Authentication Components)
#    - auth-analysis-00002.md (User Management System)
#    - auth-analysis-00003.md (Session Management)
#    - auth-analysis-00004.md (API Authentication)
#    - auth-analysis-00005.md (Security Components)
#    - auth-analysis-00006.md (External Integrations)
#    - auth-analysis-00007.md (Testing and Configuration)
```

## 🔄 Advanced IDE Integration Patterns

### Pattern 1: Multi-Layer Analysis

Analyze the same codebase from different perspectives:

```bash
# Layer 1: Architecture Analysis
code-ingest query-prepare "
  SELECT filepath, content_text 
  FROM INGEST_20240928143022 
  WHERE filepath LIKE '%/mod.rs' OR filepath LIKE '%/lib.rs'
" --temp-path ./architecture-temp.txt \
  --tasks-file ./architecture-tasks.md \
  --output-table QUERYRESULT_architecture \
  --db-path ./analysis

# Layer 2: Security Analysis
code-ingest query-prepare "
  SELECT filepath, content_text 
  FROM INGEST_20240928143022 
  WHERE lower(content_text) ~ '(unsafe|unwrap|expect|panic)'
" --temp-path ./security-temp.txt \
  --tasks-file ./security-tasks.md \
  --output-table QUERYRESULT_security \
  --db-path ./analysis

# Layer 3: Performance Analysis
code-ingest query-prepare "
  SELECT filepath, content_text 
  FROM INGEST_20240928143022 
  WHERE lower(content_text) ~ '(clone|collect|allocate|vec!|hashmap)'
" --temp-path ./performance-temp.txt \
  --tasks-file ./performance-tasks.md \
  --output-table QUERYRESULT_performance \
  --db-path ./analysis
```

### Pattern 2: Iterative Refinement

Use analysis results to drive deeper investigation:

```bash
# First pass: High-level analysis
code-ingest query-prepare "
  SELECT filepath, content_text 
  FROM INGEST_20240928143022 
  WHERE extension = 'rs' AND line_count > 100
  ORDER BY line_count DESC LIMIT 20
" --temp-path ./overview-temp.txt \
  --tasks-file ./overview-tasks.md \
  --output-table QUERYRESULT_overview \
  --db-path ./analysis

# After analysis, identify interesting patterns and dive deeper
# Second pass: Focus on specific components identified in first pass
code-ingest query-prepare "
  SELECT filepath, content_text 
  FROM INGEST_20240928143022 
  WHERE filepath IN (
    'src/core/engine.rs',
    'src/database/operations.rs',
    'src/processing/pipeline.rs'
  )
" --temp-path ./deep-dive-temp.txt \
  --tasks-file ./deep-dive-tasks.md \
  --output-table QUERYRESULT_deep_dive \
  --db-path ./analysis
```

### Pattern 3: Cross-Reference Analysis

Analyze relationships between different parts of the system:

```bash
# Find all files that import/use a specific module
code-ingest query-prepare "
  SELECT filepath, content_text 
  FROM INGEST_20240928143022 
  WHERE content_text LIKE '%use crate::auth%' OR content_text LIKE '%mod auth%'
" --temp-path ./auth-usage-temp.txt \
  --tasks-file ./auth-usage-tasks.md \
  --output-table QUERYRESULT_auth_usage \
  --db-path ./analysis
```

## 📊 IDE Integration Metrics

Track your analysis productivity:

```bash
# Analysis completion metrics
code-ingest sql "
  SELECT 
    DATE(created_at) as analysis_date,
    COUNT(*) as analyses_completed,
    AVG(LENGTH(analysis_result)) as avg_analysis_length
  FROM QUERYRESULT_auth_detailed_analysis 
  GROUP BY DATE(created_at)
  ORDER BY analysis_date DESC
" --db-path ./analysis

# Analysis quality metrics
code-ingest sql "
  SELECT 
    original_query,
    COUNT(*) as result_count,
    AVG(LENGTH(analysis_result)) as avg_depth,
    MAX(created_at) as last_updated
  FROM (
    SELECT 'Authentication Analysis' as original_query, analysis_result, created_at FROM QUERYRESULT_auth_detailed_analysis
    UNION ALL
    SELECT 'Security Analysis' as original_query, analysis_result, created_at FROM QUERYRESULT_security
    UNION ALL
    SELECT 'Architecture Analysis' as original_query, analysis_result, created_at FROM QUERYRESULT_architecture
  ) combined_analyses
  GROUP BY original_query
  ORDER BY result_count DESC
" --db-path ./analysis
```

## 🎯 IDE Integration Best Practices

### 1. Structured Analysis Approach
- Start with broad overview queries
- Use systematic task division (7 groups works well)
- Create specific, actionable prompts
- Store and version your analysis results

### 2. Prompt Engineering
- Be specific about what you want to learn
- Include context about the codebase
- Ask for structured output (markdown sections)
- Request actionable recommendations

### 3. Result Management
- Store all analysis results in the database
- Export important findings as individual files
- Create summary reports combining multiple analyses
- Track analysis history and evolution

### 4. Workflow Automation
```bash
#!/bin/bash
# automated-analysis.sh - Complete analysis workflow

REPO_URL="$1"
ANALYSIS_TYPE="$2"
DB_PATH="./analysis"

# Step 1: Ingest if needed
if [ ! -d "$DB_PATH" ]; then
    code-ingest ingest "$REPO_URL" --db-path "$DB_PATH"
fi

# Step 2: Prepare analysis based on type
case "$ANALYSIS_TYPE" in
    "auth")
        QUERY="SELECT filepath, content_text FROM INGEST_* WHERE lower(content_text) LIKE '%auth%'"
        PROMPT="auth-analysis-prompt.md"
        ;;
    "security")
        QUERY="SELECT filepath, content_text FROM INGEST_* WHERE lower(content_text) ~ '(password|secret|unsafe)'"
        PROMPT="security-analysis-prompt.md"
        ;;
    *)
        echo "Unknown analysis type: $ANALYSIS_TYPE"
        exit 1
        ;;
esac

# Step 3: Execute workflow
code-ingest query-prepare "$QUERY" \
    --temp-path "./${ANALYSIS_TYPE}-temp.txt" \
    --tasks-file "./${ANALYSIS_TYPE}-tasks.md" \
    --output-table "QUERYRESULT_${ANALYSIS_TYPE}" \
    --db-path "$DB_PATH"

echo "Analysis prepared. Open ${ANALYSIS_TYPE}-tasks.md in your IDE to continue."
```

### 5. Quality Assurance
- Review AI analysis results critically
- Cross-reference findings with actual code
- Validate recommendations before implementing
- Keep analysis results updated as code changes

## 🔧 Troubleshooting IDE Integration

### Common Issues

**Tasks file not generating properly:**
- Check that the query returns results
- Verify the prompt file exists and is readable
- Ensure the output table name is valid

**IDE not executing tasks:**
- Verify the temp file path is accessible
- Check that the tasks file format is correct
- Ensure the IDE has proper permissions

**Analysis results incomplete:**
- Review the prompt for clarity and specificity
- Check if the temp file contains expected data
- Verify the AI model has sufficient context

**Storage operations failing:**
- Ensure the database connection is working
- Check that the output table exists
- Verify file paths are correct and accessible

This IDE integration approach transforms code-ingest from a simple query tool into a powerful systematic analysis platform that leverages AI assistance for deep code understanding.
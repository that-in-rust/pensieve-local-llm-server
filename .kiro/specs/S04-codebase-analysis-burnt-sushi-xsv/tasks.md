# Implementation Plan

# Table Structure

column_name        | data_type               
-------------------+-------------------------
file_id            | bigint                  
ingestion_id       | bigint                  
filepath           | character varying       
filename           | character varying       
extension          | character varying       
file_size_bytes    | bigint                  
line_count         | integer   

- the repo - https://github.com/BurntSushi/xsv



Ingestion
``` bash
./target/release/code-ingest ingest https://github.com/BurntSushi/xsv --db-path /Users/neetipatni/desktop/PensieveDB01
```
Running SQL Query
``` bash
# 1. Explore your data: 
./target/release/code-ingest run -- list-tables --db-path /Users/neetipatni/desktop/PensieveDB01

# 2. Sample the data:
cargo run -- sample --table INGEST_20250928101039 --db-path /Users/neetipatni/desktop/PensieveDB01

# 3. Run queries:
cargo run -- sql 'SELECT filepath, filename FROM "INGEST_20250928101039" LIMIT 5' --db-path /Users/neetipatni/desktop/PensieveDB01

# 4. Export files:
cargo run -- print-to-md --table INGEST_20250928101039 --sql 'SELECT * FROM "INGEST_20250928101039" LIMIT 10' --prefix xsv --location ./exports --db-path /Users/neetipatni/desktop/PensieveDB01

# Working SQL Query Examples:
# Basic file listing
cargo run -- sql 'SELECT filepath, filename FROM "INGEST_20250928101039" LIMIT 3' --db-path /Users/neetipatni/desktop/PensieveDB01

# Count files by extension
cargo run -- sql 'SELECT extension, COUNT(*) FROM "INGEST_20250928101039" GROUP BY extension ORDER BY COUNT(*) DESC' --db-path /Users/neetipatni/desktop/PensieveDB01

# Find Rust files with specific patterns
cargo run -- sql 'SELECT filepath FROM "INGEST_20250928101039" WHERE extension = '\''rs'\'' AND content_text LIKE '\''%unsafe%'\''' --db-path /Users/neetipatni/desktop/PensieveDB01
```


## Phase 1: Database Setup and Verification

- [x] 1. Do a fresh ingestion of - the repo - https://github.com/BurntSushi/xsv + Ingest PG Database to be stored at - /Users/neetipatni/desktop/PensieveDB01 + Result PG database to be stored at -  /Users/neetipatni/desktop/PensieveDB01
  - Run
  ``` bash
  ./target/release/code-ingest ingest https://github.com/BurntSushi/xsv --db-path /Users/neetipatni/desktop/PensieveDB01
  ```
  - [ ] Note down the table name and validate the count of rows and column list


- [ ] 4. Generate systematic analysis tasks using code-ingest
  - Run: `code-ingest generate-tasks --sql "SELECT * FROM INGEST_20250928062949" --prompt-file ./xsv-l1-l8-analysis-prompt.md --output-table QUERYRESULT_xsv_knowledge_arbitrage --tasks-file ./xsv-analysis-tasks.md --db-path /Users/neetipatni/desktop/PensieveDB01`
  - Verify generated tasks.md file contains systematic analysis prompts for all 59 files
  - Confirm tasks are structured for chunked processing with 300-500 line segments and 10-20 line overlap
  - _Requirements: 6.1, 7.1, 7.2, 7.3, 7.4_


- [ ] 9. Validate analysis completeness and quality
  - Verify all 59 files have been processed through L1-L8 analysis
  - Confirm all requirements have corresponding analysis results
  - Validate export functionality and data integrity
  - Generate final analysis report with metrics and insights summary
  - _Requirements: 1.4, 2.4, 3.4_
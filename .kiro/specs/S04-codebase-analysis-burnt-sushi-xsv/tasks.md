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

- the repo : https://github.com/BurntSushi/xsv
- the prompt file : .kiro/steering/spec-S04-steering-doc-analysis.md



Ingestion
``` bash
./target/release/code-ingest ingest https://github.com/BurntSushi/xsv --db-path /Users/neetipatni/desktop/PensieveDB01
```
INGEST_20250928101039

Running SQL Query
``` bash
# 1. List tables - WORKING ✅
./target/release/code-ingest list-tables --db-path /Users/neetipatni/desktop/PensieveDB01

# 2. Sample data - WORKING ✅  
./target/release/code-ingest sample --table INGEST_20250928101039 --db-path /Users/neetipatni/desktop/PensieveDB01

# 3. SQL queries - WORKING ✅
./target/release/code-ingest sql 'SELECT filepath, filename FROM "INGEST_20250928101039" LIMIT 5' --db-path /Users/neetipatni/desktop/PensieveDB01

```


## Phase 1: Database Setup and Verification

- [x] 1. Do a fresh ingestion of - the repo - https://github.com/BurntSushi/xsv + Ingest PG Database to be stored at - /Users/neetipatni/desktop/PensieveDB01 + Result PG database to be stored at -  /Users/neetipatni/desktop/PensieveDB01
  - Run
  ``` bash
  ./target/release/code-ingest ingest https://github.com/BurntSushi/xsv --db-path /Users/neetipatni/desktop/PensieveDB01
  ```
  - [ ] Note down the table name and validate the count of rows and column list


- [ ] 4. Generate systematic analysis tasks using code-ingest
  - Run
  ``` bash
  ./target/release/code-ingest generate-tasks --sql 'SELECT * FROM "INGEST_20250928062949"' --prompt-file /Users/neetipatni/Desktop/Game20250927/pensieve/.kiro/steering/spec-S04-steering-doc-analysis.md --output-table QUERYRESULT_xsv_$%Y%M%D%H%S --tasks-file ./xsv-analysis-tasks.md --db-path /Users/neetipatni/desktop/PensieveDB01
  ``
  - Verify generated tasks.md file contains systematic analysis prompts for all 59 files
  - Confirm tasks are structured for chunked processing with 300-500 line segments and 10-20 line overlap
  - _Requirements: 6.1, 7.1, 7.2, 7.3, 7.4_


- [ ] 9. Validate analysis completeness and quality
  - Verify all 59 files have been processed through L1-L8 analysis
  - Confirm all requirements have corresponding analysis results
  - Validate export functionality and data integrity
  - Generate final analysis report with metrics and insights summary
  - _Requirements: 1.4, 2.4, 3.4_
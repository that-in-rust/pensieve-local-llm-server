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
- Ingested TableName


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
  ./target/release/code-ingest task-generator --table "INGEST_20250928101039" --db-path /Users/neetipatni/desktop/PensieveDB01 --output-file .kiro/specs/S04-codebase-analysis-burnt-sushi-xsv/xsv-analysis-tasks.md
  ```

- [x] 5. Analyze INGEST_20250928101039 row 35 
  - **Content**: `.raw_data_202509/INGEST_20250928101039_35_Content.txt` as A + `.raw_data_202509/INGEST_20250928101039_35_Content_L1.txt` as B + `.raw_data_202509/INGEST_20250928101039_35_Content_L2.txt` as C
  - **Prompt**: `.kiro/steering/spec-S04-steering-doc-analysis.md` where you try to find insights of A alone ; A in context of B ; B in cotext of C ; A in context B & C
  - **Output**: `gringotts/WorkArea/INGEST_20250928101039_35.md`

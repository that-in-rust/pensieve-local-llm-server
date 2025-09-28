# Requirements Document

## Introduction

Generate structured task lists from ingested codebase data to enable systematic analysis. Transform database query results into Kiro-compatible markdown files with hierarchical task numbering.

## Example Scenario

**Input:** Query `INGEST_20250928101039` table returns 35 files from xsv repository  
**Output:** Markdown file with 7 task groups, each containing ~5 analysis tasks  
**Source:** Database table with columns: file_id, filepath, filename, extension, line_count  
**Prompt:** Use `.kiro/steering/spec-S04-steering-doc-analysis.md` for analysis

Format e.g. of final task

- [x] 5. Analyze INGEST_20250928101039 row 35 
  - **Content**: `.raw_data_202509/INGEST_20250928101039_35_Content.txt` as A + `.raw_data_202509/INGEST_20250928101039_35_Content_L1.txt` as B + `.raw_data_202509/INGEST_20250928101039_35_Content_L2.txt` as C
  - **Prompt**: `.kiro/steering/spec-S04-steering-doc-analysis.md` where you try to find insights of A alone ; A in context of B ; B in cotext of C ; A in context B & C
  - **Output**: `gringotts/WorkArea/INGEST_20250928101039_35.md`


# Strict Scope

- the Table_NAME is known
- a terminal command with code-ingest tool to find SQL count of rows in TableName - then loop through all of the rows to create 3 rows of 
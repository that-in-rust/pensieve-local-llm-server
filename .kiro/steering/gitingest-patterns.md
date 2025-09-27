# Code Ingestion: The Shreyas Approach

## The Core Problem

**What developers actually need**: Understand unfamiliar codebases in minutes, not hours.

**Why current tools suck**: GitHub search is shallow, grep is slow, IDEs don't work across repos.

**The insight**: GitIngest solves the LLM prompt problem. We need to solve the persistent knowledge problem.

## The Solution

**Input**: GitHub URL + local folder path  
**Output**: Searchable PostgreSQL database  
**Success Metric**: 10x faster codebase understanding

## The Three Critical Decisions

1. **Rust over Python**: 100x faster processing, memory safety, true parallelism
2. **PostgreSQL over Vector DB**: Structured queries, full-text search, familiar to teams  
3. **Three file types**: Direct text (.rs, .py), convertible (.pdf), non-text (.jpg)

## The Non-Negotiables

- **30-second clone limit**: Anything slower kills adoption
- **1000 files/second processing**: Must handle large repos
- **User-specified local path**: No magic temp directories
- **One command**: `code-ingest <url> --local-path <path>`

## The 3-Week Plan

**Week 1**: Prove core value (clone → process → store)  
**Week 2**: Make it fast (parallel processing, batch inserts)  
**Week 3**: Make it useful (search, web interface, private repos)

## The Key Insight

**GitIngest**: GitHub repo → Text blob → LLM prompt  
**Our tool**: GitHub repo → Structured DB → Persistent queries

We're building a searchable knowledge base, not a one-time converter.

# Requirements Document: Understanding My Twitter Voice

## Introduction

Analyze 60,000+ tweets to understand authentic voice patterns and extract insights for potential book creation. Use code-ingest to load data into PostgreSQL, then apply voice analysis prompts to 1000-line chunks.

## Requirements

### Requirement 1: Data Ingestion

**User Story:** As a content creator, I want to ingest my Twitter data into PostgreSQL so that I can analyze it systematically.

#### Acceptance Criteria

1. WHEN I run code-ingest THEN the system SHALL execute `./target/release/code-ingest ingest /Users/neetipatni/Desktop/Wisdom20250929/pensieve/rawDataWillBeDeleted --folder-flag --db-path /Users/neetipatni/desktop/PensieveDB01`
2. WHEN ingestion completes THEN the system SHALL create a timestamped PostgreSQL table with all Twitter content and metadata

### Requirement 2: Voice Analysis Processing and Output Storage

**User Story:** As someone analyzing my writing voice, I want to process my Twitter data in 1000-line chunks using the voice analysis prompt and store the observations systematically so that I can build a comprehensive voice profile.

#### Acceptance Criteria

1. WHEN processing data THEN the system SHALL create 1000-line chunks from the ingested content
2. WHEN analyzing chunks THEN the system SHALL apply the voice analysis prompt from `.kiro/steering/non-technical-authentic-voice-prompt.md` 
3. WHEN generating outputs THEN the system SHALL create Mermaid diagrams following `.kiro/steering/ref-Mermaid-202509.md` specifications
4. WHEN storing analysis results THEN the system SHALL create a new timestamped PostgreSQL table `VOICE_ANALYSIS_OBSERVATIONS_YYYYMMDDHHMMSS` to store chunk analysis results, voice patterns, and insights with references back to source chunks for systematic querying and accumulation
5. WHEN analysis completes THEN the system SHALL produce consolidated voice profiles, gem curation, and book creation frameworks as specified in the voice analysis prompt
6. WHEN all analysis is stored in PostgreSQL THEN the system SHALL extract and consolidate the results into a readable markdown file in the gringotts folder for easy review and sharing
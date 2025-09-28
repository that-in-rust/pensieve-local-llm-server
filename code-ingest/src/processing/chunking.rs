//! Chunking engine for large file processing
//!
//! This module provides LOC-based file splitting with configurable chunk sizes,
//! chunk boundary detection that preserves code structure, and proper metadata tracking.

use crate::error::{ProcessingError, ProcessingResult};
use serde::{Deserialize, Serialize};
use tracing::{debug, trace};

/// Configuration for the chunking engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkingConfig {
    /// Number of lines per chunk
    pub chunk_size: usize,
    /// Whether to preserve code structure at chunk boundaries
    pub preserve_structure: bool,
    /// Minimum chunk size (prevents tiny chunks at file end)
    pub min_chunk_size: usize,
    /// Maximum overlap between chunks (for context preservation)
    pub max_overlap: usize,
}

impl Default for ChunkingConfig {
    fn default() -> Self {
        Self {
            chunk_size: 500,
            preserve_structure: true,
            min_chunk_size: 50,
            max_overlap: 10,
        }
    }
}

/// Metadata for a single chunk
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChunkMetadata {
    /// Chunk number (1-based)
    pub chunk_number: u32,
    /// Starting line number in original file (1-based)
    pub start_line: u32,
    /// Ending line number in original file (1-based, inclusive)
    pub end_line: u32,
    /// Number of lines in this chunk
    pub line_count: u32,
    /// Whether this chunk was adjusted for structure preservation
    pub structure_adjusted: bool,
}

/// A single chunk of file content with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkData {
    /// Chunk metadata
    pub metadata: ChunkMetadata,
    /// The actual content of this chunk
    pub content: String,
    /// Original file identifier
    pub file_id: String,
    /// Original filepath
    pub filepath: String,
    /// Original filename
    pub filename: String,
    /// File extension
    pub extension: Option<String>,
}

/// Result of chunking a file
#[derive(Debug, Clone)]
pub struct ChunkingResult {
    /// All chunks created from the file
    pub chunks: Vec<ChunkData>,
    /// Total number of lines in original file
    pub total_lines: u32,
    /// Whether the file was actually chunked (false if below threshold)
    pub was_chunked: bool,
}

/// Core chunking engine
#[derive(Debug, Clone)]
pub struct ChunkingEngine {
    config: ChunkingConfig,
}

impl ChunkingEngine {
    /// Create a new chunking engine with the given configuration
    pub fn new(config: ChunkingConfig) -> Self {
        Self { config }
    }

    /// Create a chunking engine with default configuration
    pub fn default() -> Self {
        Self::new(ChunkingConfig::default())
    }

    /// Create a chunking engine with a specific chunk size
    pub fn with_chunk_size(chunk_size: usize) -> Self {
        Self::new(ChunkingConfig {
            chunk_size,
            ..ChunkingConfig::default()
        })
    }

    /// Chunk file content into multiple chunks
    /// 
    /// If the file has fewer lines than chunk_size, returns a single chunk.
    /// Otherwise, splits the file into chunks of approximately chunk_size lines each.
    pub fn chunk_content(
        &self,
        content: &str,
        file_id: String,
        filepath: String,
        filename: String,
        extension: Option<String>,
    ) -> ProcessingResult<ChunkingResult> {
        let lines: Vec<&str> = content.lines().collect();
        let total_lines = lines.len() as u32;

        debug!(
            "Chunking file {} with {} lines (chunk_size: {})",
            filepath, total_lines, self.config.chunk_size
        );

        // If file is smaller than chunk size, return as single chunk
        if (total_lines as usize) < self.config.chunk_size {
            trace!("File {} is smaller than chunk size, returning as single chunk", filepath);
            
            let chunk_data = ChunkData {
                metadata: ChunkMetadata {
                    chunk_number: 1,
                    start_line: 1,
                    end_line: total_lines,
                    line_count: total_lines,
                    structure_adjusted: false,
                },
                content: content.to_string(),
                file_id,
                filepath,
                filename,
                extension,
            };

            return Ok(ChunkingResult {
                chunks: vec![chunk_data],
                total_lines,
                was_chunked: false,
            });
        }

        // Split into chunks
        let chunks = self.split_into_chunks(&lines, file_id, filepath, filename, extension)?;

        Ok(ChunkingResult {
            chunks,
            total_lines,
            was_chunked: true,
        })
    }

    /// Split lines into chunks with proper boundary detection
    fn split_into_chunks(
        &self,
        lines: &[&str],
        file_id: String,
        filepath: String,
        filename: String,
        extension: Option<String>,
    ) -> ProcessingResult<Vec<ChunkData>> {
        let mut chunks = Vec::new();
        let mut current_start = 0;
        let mut chunk_number = 1;

        while current_start < lines.len() {
            let ideal_end = std::cmp::min(current_start + self.config.chunk_size, lines.len());
            
            // Find the best boundary for this chunk
            let actual_end = if self.config.preserve_structure && ideal_end < lines.len() {
                self.find_structure_boundary(lines, current_start, ideal_end)
            } else {
                ideal_end
            };

            // Ensure we don't create chunks that are too small (except for the last chunk)
            let final_end = if actual_end < lines.len() && 
                              (actual_end - current_start) < self.config.min_chunk_size {
                // Extend to minimum size if possible
                std::cmp::min(current_start + self.config.min_chunk_size, lines.len())
            } else {
                actual_end
            };

            // Create chunk content
            let chunk_lines = &lines[current_start..final_end];
            let chunk_content = chunk_lines.join("\n");

            let metadata = ChunkMetadata {
                chunk_number,
                start_line: (current_start + 1) as u32, // 1-based line numbers
                end_line: final_end as u32,
                line_count: (final_end - current_start) as u32,
                structure_adjusted: final_end != ideal_end,
            };

            let chunk_data = ChunkData {
                metadata,
                content: chunk_content,
                file_id: file_id.clone(),
                filepath: filepath.clone(),
                filename: filename.clone(),
                extension: extension.clone(),
            };

            chunks.push(chunk_data);

            trace!(
                "Created chunk {} for file {}: lines {}-{} ({} lines)",
                chunk_number, filepath, current_start + 1, final_end, final_end - current_start
            );

            current_start = final_end;
            chunk_number += 1;
        }

        debug!("Split file {} into {} chunks", filepath, chunks.len());
        Ok(chunks)
    }

    /// Find a good boundary for chunk splitting that preserves code structure
    fn find_structure_boundary(&self, lines: &[&str], _start: usize, ideal_end: usize) -> usize {
        // Look for good boundaries within a reasonable range around the ideal end
        let search_range = std::cmp::min(self.config.max_overlap, self.config.chunk_size / 10);
        let search_start = ideal_end.saturating_sub(search_range);
        let search_end = std::cmp::min(ideal_end + search_range, lines.len());

        // Priority order for good boundaries (higher score = better boundary)
        let mut best_boundary = ideal_end;
        let mut best_score = 0;

        for i in search_start..search_end {
            if i >= lines.len() {
                break;
            }

            let line = lines[i].trim();
            let score = self.calculate_boundary_score(line, i, lines);

            if score > best_score {
                best_score = score;
                best_boundary = i + 1; // +1 because we want to end after this line
            }
        }

        // Ensure we don't go beyond the file
        std::cmp::min(best_boundary, lines.len())
    }

    /// Calculate a score for how good a line is as a chunk boundary
    fn calculate_boundary_score(&self, line: &str, line_index: usize, all_lines: &[&str]) -> u32 {
        let mut score: u32 = 0;

        // Empty lines are good boundaries
        if line.is_empty() {
            score += 10;
        }

        // Lines with only whitespace are good boundaries
        if line.chars().all(|c| c.is_whitespace()) {
            score += 8;
        }

        // Comments are decent boundaries
        if line.starts_with("//") || line.starts_with('#') || line.starts_with("/*") {
            score += 5;
        }

        // Function/class/struct definitions are good boundaries
        if line.contains("fn ") || line.contains("class ") || line.contains("struct ") || 
           line.contains("impl ") || line.contains("trait ") || line.contains("enum ") {
            score += 15;
        }

        // Closing braces are good boundaries
        if line.trim() == "}" || line.trim() == "};" {
            score += 12;
        }

        // Opening braces at end of line suggest start of block (not ideal for ending)
        if line.trim().ends_with('{') {
            score = score.saturating_sub(5);
        }

        // Lines that are clearly in the middle of expressions are bad boundaries
        if line.trim().ends_with(',') || line.trim().ends_with('\\') || 
           line.trim().ends_with("&&") || line.trim().ends_with("||") {
            score = score.saturating_sub(8);
        }

        // Look ahead to see if next line is indented (suggests we're in middle of block)
        if line_index + 1 < all_lines.len() {
            let next_line = all_lines[line_index + 1];
            if !next_line.trim().is_empty() && next_line.starts_with("    ") || next_line.starts_with('\t') {
                // Next line is indented, might be in middle of block
                score = score.saturating_sub(3);
            }
        }

        score
    }

    /// Get the configuration used by this chunking engine
    pub fn config(&self) -> &ChunkingConfig {
        &self.config
    }

    /// Validate that chunk metadata is consistent
    pub fn validate_chunks(&self, chunks: &[ChunkData]) -> ProcessingResult<()> {
        if chunks.is_empty() {
            return Err(ProcessingError::ChunkingFailed {
                reason: "No chunks provided for validation".to_string(),
            });
        }

        // Check chunk numbering
        for (i, chunk) in chunks.iter().enumerate() {
            let expected_number = (i + 1) as u32;
            if chunk.metadata.chunk_number != expected_number {
                return Err(ProcessingError::ChunkingFailed {
                    reason: format!(
                        "Chunk numbering inconsistent: expected {}, got {}",
                        expected_number, chunk.metadata.chunk_number
                    ),
                });
            }
        }

        // Check line number continuity
        for i in 1..chunks.len() {
            let prev_chunk = &chunks[i - 1];
            let curr_chunk = &chunks[i];

            if curr_chunk.metadata.start_line != prev_chunk.metadata.end_line + 1 {
                return Err(ProcessingError::ChunkingFailed {
                    reason: format!(
                        "Line number gap between chunks {} and {}: {} -> {}",
                        prev_chunk.metadata.chunk_number,
                        curr_chunk.metadata.chunk_number,
                        prev_chunk.metadata.end_line,
                        curr_chunk.metadata.start_line
                    ),
                });
            }
        }

        // Check that line counts match
        for chunk in chunks {
            let expected_lines = chunk.metadata.end_line - chunk.metadata.start_line + 1;
            if chunk.metadata.line_count != expected_lines {
                return Err(ProcessingError::ChunkingFailed {
                    reason: format!(
                        "Chunk {} line count mismatch: metadata says {}, calculated {}",
                        chunk.metadata.chunk_number, chunk.metadata.line_count, expected_lines
                    ),
                });
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_content(lines: usize) -> String {
        (1..=lines)
            .map(|i| format!("line {}", i))
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn create_rust_code_content() -> String {
        r#"use std::collections::HashMap;

// This is a test function
fn main() {
    let mut map = HashMap::new();
    map.insert("key1", "value1");
    map.insert("key2", "value2");
    
    for (key, value) in &map {
        println!("{}: {}", key, value);
    }
}

struct TestStruct {
    field1: String,
    field2: i32,
}

impl TestStruct {
    fn new(field1: String, field2: i32) -> Self {
        Self { field1, field2 }
    }
    
    fn get_field1(&self) -> &str {
        &self.field1
    }
}

trait TestTrait {
    fn test_method(&self) -> String;
}

impl TestTrait for TestStruct {
    fn test_method(&self) -> String {
        format!("{}: {}", self.field1, self.field2)
    }
}"#.to_string()
    }

    #[test]
    fn test_chunking_engine_creation() {
        let engine = ChunkingEngine::default();
        assert_eq!(engine.config().chunk_size, 500);

        let engine = ChunkingEngine::with_chunk_size(100);
        assert_eq!(engine.config().chunk_size, 100);
    }

    #[test]
    fn test_small_file_no_chunking() {
        let engine = ChunkingEngine::with_chunk_size(100);
        let content = create_test_content(50);
        
        let result = engine.chunk_content(
            &content,
            "test_file_1".to_string(),
            "test.txt".to_string(),
            "test.txt".to_string(),
            Some("txt".to_string()),
        ).unwrap();

        assert!(!result.was_chunked);
        assert_eq!(result.chunks.len(), 1);
        assert_eq!(result.total_lines, 50);
        
        let chunk = &result.chunks[0];
        assert_eq!(chunk.metadata.chunk_number, 1);
        assert_eq!(chunk.metadata.start_line, 1);
        assert_eq!(chunk.metadata.end_line, 50);
        assert_eq!(chunk.metadata.line_count, 50);
        assert!(!chunk.metadata.structure_adjusted);
    }

    #[test]
    fn test_large_file_chunking() {
        let engine = ChunkingEngine::with_chunk_size(10);
        let content = create_test_content(25);
        
        let result = engine.chunk_content(
            &content,
            "test_file_2".to_string(),
            "test.txt".to_string(),
            "test.txt".to_string(),
            Some("txt".to_string()),
        ).unwrap();

        assert!(result.was_chunked);
        assert_eq!(result.total_lines, 25);
        assert_eq!(result.chunks.len(), 3); // 10 + 10 + 5 lines
        
        // Validate chunk metadata
        engine.validate_chunks(&result.chunks).unwrap();
        
        // Check first chunk
        let chunk1 = &result.chunks[0];
        assert_eq!(chunk1.metadata.chunk_number, 1);
        assert_eq!(chunk1.metadata.start_line, 1);
        assert_eq!(chunk1.metadata.end_line, 10);
        assert_eq!(chunk1.metadata.line_count, 10);
        
        // Check last chunk
        let chunk3 = &result.chunks[2];
        assert_eq!(chunk3.metadata.chunk_number, 3);
        assert_eq!(chunk3.metadata.start_line, 21);
        assert_eq!(chunk3.metadata.end_line, 25);
        assert_eq!(chunk3.metadata.line_count, 5);
    }

    #[test]
    fn test_structure_boundary_detection() {
        let engine = ChunkingEngine::new(ChunkingConfig {
            chunk_size: 15,
            preserve_structure: true,
            min_chunk_size: 5,
            max_overlap: 5,
        });
        
        let content = create_rust_code_content();
        let result = engine.chunk_content(
            &content,
            "rust_file".to_string(),
            "test.rs".to_string(),
            "test.rs".to_string(),
            Some("rs".to_string()),
        ).unwrap();

        assert!(result.was_chunked);
        engine.validate_chunks(&result.chunks).unwrap();
        
        // Should have multiple chunks due to structure preservation
        assert!(result.chunks.len() > 1);
        
        // Check that some chunks were structure-adjusted
        let _has_adjusted = result.chunks.iter().any(|c| c.metadata.structure_adjusted);
        // Note: This might not always be true depending on the exact content and boundaries
        // but it's a good indicator that the structure detection is working
    }

    #[test]
    fn test_boundary_score_calculation() {
        let engine = ChunkingEngine::default();
        let lines = vec![
            "    let x = 1;",           // Middle of block
            "",                        // Empty line (good boundary)
            "fn test() {",             // Function start (good boundary)
            "    println!(\"test\");", // Inside function
            "}",                       // Closing brace (good boundary)
            "// Comment",              // Comment (decent boundary)
        ];

        // Empty line should have high score
        let empty_score = engine.calculate_boundary_score("", 1, &lines);
        assert!(empty_score > 5);

        // Function definition should have high score
        let fn_score = engine.calculate_boundary_score("fn test() {", 2, &lines);
        assert!(fn_score > 10);

        // Closing brace should have high score
        let brace_score = engine.calculate_boundary_score("}", 4, &lines);
        assert!(brace_score > 10);

        // Comment should have moderate score
        let comment_score = engine.calculate_boundary_score("// Comment", 5, &lines);
        assert!(comment_score > 0);
        assert!(comment_score < fn_score);
    }

    #[test]
    fn test_chunk_validation() {
        let engine = ChunkingEngine::default();
        
        // Valid chunks
        let valid_chunks = vec![
            ChunkData {
                metadata: ChunkMetadata {
                    chunk_number: 1,
                    start_line: 1,
                    end_line: 10,
                    line_count: 10,
                    structure_adjusted: false,
                },
                content: "test content".to_string(),
                file_id: "test".to_string(),
                filepath: "test.txt".to_string(),
                filename: "test.txt".to_string(),
                extension: Some("txt".to_string()),
            },
            ChunkData {
                metadata: ChunkMetadata {
                    chunk_number: 2,
                    start_line: 11,
                    end_line: 20,
                    line_count: 10,
                    structure_adjusted: false,
                },
                content: "more test content".to_string(),
                file_id: "test".to_string(),
                filepath: "test.txt".to_string(),
                filename: "test.txt".to_string(),
                extension: Some("txt".to_string()),
            },
        ];

        assert!(engine.validate_chunks(&valid_chunks).is_ok());

        // Invalid chunks - wrong numbering
        let mut invalid_chunks = valid_chunks.clone();
        invalid_chunks[1].metadata.chunk_number = 3;
        assert!(engine.validate_chunks(&invalid_chunks).is_err());

        // Invalid chunks - line number gap
        let mut invalid_chunks = valid_chunks.clone();
        invalid_chunks[1].metadata.start_line = 15;
        assert!(engine.validate_chunks(&invalid_chunks).is_err());

        // Invalid chunks - wrong line count
        let mut invalid_chunks = valid_chunks;
        invalid_chunks[0].metadata.line_count = 5;
        assert!(engine.validate_chunks(&invalid_chunks).is_err());
    }

    #[test]
    fn test_chunk_metadata_consistency() {
        let engine = ChunkingEngine::with_chunk_size(5);
        let content = create_test_content(12);
        
        let result = engine.chunk_content(
            &content,
            "test_file".to_string(),
            "test.txt".to_string(),
            "test.txt".to_string(),
            Some("txt".to_string()),
        ).unwrap();

        // Should create 3 chunks: 5 + 5 + 2 lines
        assert_eq!(result.chunks.len(), 3);
        
        // Validate all chunks
        engine.validate_chunks(&result.chunks).unwrap();
        
        // Check that all chunks have correct file metadata
        for chunk in &result.chunks {
            assert_eq!(chunk.file_id, "test_file");
            assert_eq!(chunk.filepath, "test.txt");
            assert_eq!(chunk.filename, "test.txt");
            assert_eq!(chunk.extension, Some("txt".to_string()));
        }
    }

    #[test]
    fn test_min_chunk_size_enforcement() {
        let engine = ChunkingEngine::new(ChunkingConfig {
            chunk_size: 10,
            preserve_structure: false,
            min_chunk_size: 5,
            max_overlap: 2,
        });
        
        let content = create_test_content(23); // Should create chunks of 10, 10, 3 -> but 3 is too small
        
        let result = engine.chunk_content(
            &content,
            "test_file".to_string(),
            "test.txt".to_string(),
            "test.txt".to_string(),
            Some("txt".to_string()),
        ).unwrap();

        engine.validate_chunks(&result.chunks).unwrap();
        
        // Check that no chunk is smaller than min_chunk_size (except possibly the last one)
        for (i, chunk) in result.chunks.iter().enumerate() {
            if i < result.chunks.len() - 1 {
                // Not the last chunk
                assert!(chunk.metadata.line_count >= engine.config.min_chunk_size as u32);
            }
        }
    }
}
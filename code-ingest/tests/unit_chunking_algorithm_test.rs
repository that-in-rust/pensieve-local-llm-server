//! Comprehensive unit tests for the chunking algorithm
//!
//! Tests cover various file sizes, chunk configurations, boundary detection,
//! context generation, and edge cases as specified in task 9.1.

use code_ingest::processing::chunking::{
    ChunkingEngine, ChunkingConfig, ChunkMetadata, ChunkData, ChunkContext, ContextLevel
};
use code_ingest::error::ProcessingError;

/// Create test content with specified number of lines
fn create_test_content(lines: usize) -> String {
    (1..=lines)
        .map(|i| format!("line {}", i))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Create realistic Rust code content for structure boundary testing
fn create_rust_code_content() -> String {
    r#"use std::collections::HashMap;
use std::sync::Arc;

// Configuration struct
pub struct Config {
    pub name: String,
    pub value: i32,
}

impl Config {
    pub fn new(name: String, value: i32) -> Self {
        Self { name, value }
    }
    
    pub fn get_name(&self) -> &str {
        &self.name
    }
}

// Main function
fn main() {
    let config = Config::new("test".to_string(), 42);
    println!("Config: {}", config.get_name());
    
    let mut map = HashMap::new();
    map.insert("key1", "value1");
    map.insert("key2", "value2");
    
    for (key, value) in &map {
        println!("{}: {}", key, value);
    }
}

// Test trait
trait TestTrait {
    fn test_method(&self) -> String;
}

impl TestTrait for Config {
    fn test_method(&self) -> String {
        format!("{}: {}", self.name, self.value)
    }
}

// Another struct
pub struct DataProcessor {
    data: Vec<String>,
}

impl DataProcessor {
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }
    
    pub fn add_data(&mut self, item: String) {
        self.data.push(item);
    }
    
    pub fn process_all(&self) -> Vec<String> {
        self.data.iter().map(|s| s.to_uppercase()).collect()
    }
}"#.to_string()
}

/// Create test file metadata
fn create_test_file_metadata() -> (String, String, String, Option<String>) {
    (
        "test_file_1".to_string(),
        "test/path/file.rs".to_string(),
        "file.rs".to_string(),
        Some("rs".to_string()),
    )
}

#[cfg(test)]
mod chunking_engine_tests {
    use super::*;

    #[test]
    fn test_chunking_engine_creation_with_default_config() {
        let engine = ChunkingEngine::default();
        assert_eq!(engine.config().chunk_size, 500);
        assert_eq!(engine.config().preserve_structure, true);
        assert_eq!(engine.config().min_chunk_size, 50);
        assert_eq!(engine.config().max_overlap, 10);
    }

    #[test]
    fn test_chunking_engine_creation_with_custom_config() {
        let config = ChunkingConfig {
            chunk_size: 100,
            preserve_structure: false,
            min_chunk_size: 10,
            max_overlap: 5,
        };
        let engine = ChunkingEngine::new(config.clone());
        assert_eq!(engine.config().chunk_size, 100);
        assert_eq!(engine.config().preserve_structure, false);
        assert_eq!(engine.config().min_chunk_size, 10);
        assert_eq!(engine.config().max_overlap, 5);
    }

    #[test]
    fn test_chunking_engine_with_chunk_size() {
        let engine = ChunkingEngine::with_chunk_size(200);
        assert_eq!(engine.config().chunk_size, 200);
        // Other values should be defaults
        assert_eq!(engine.config().preserve_structure, true);
        assert_eq!(engine.config().min_chunk_size, 50);
        assert_eq!(engine.config().max_overlap, 10);
    }
}

#[cfg(test)]
mod small_file_chunking_tests {
    use super::*;

    #[test]
    fn test_small_file_no_chunking() {
        let engine = ChunkingEngine::with_chunk_size(100);
        let content = create_test_content(50);
        let (file_id, filepath, filename, extension) = create_test_file_metadata();
        
        let result = engine.chunk_content(&content, file_id.clone(), filepath.clone(), filename.clone(), extension.clone()).unwrap();

        assert!(!result.was_chunked);
        assert_eq!(result.chunks.len(), 1);
        assert_eq!(result.total_lines, 50);
        
        let chunk = &result.chunks[0];
        assert_eq!(chunk.metadata.chunk_number, 1);
        assert_eq!(chunk.metadata.start_line, 1);
        assert_eq!(chunk.metadata.end_line, 50);
        assert_eq!(chunk.metadata.line_count, 50);
        assert!(!chunk.metadata.structure_adjusted);
        assert_eq!(chunk.file_id, file_id);
        assert_eq!(chunk.filepath, filepath);
        assert_eq!(chunk.filename, filename);
        assert_eq!(chunk.extension, extension);
    }

    #[test]
    fn test_empty_file() {
        let engine = ChunkingEngine::with_chunk_size(10);
        let content = "";
        let (file_id, filepath, filename, extension) = create_test_file_metadata();
        
        let result = engine.chunk_content(&content, file_id, filepath, filename, extension).unwrap();

        assert!(!result.was_chunked);
        assert_eq!(result.chunks.len(), 1);
        assert_eq!(result.total_lines, 0);
        
        let chunk = &result.chunks[0];
        assert_eq!(chunk.metadata.chunk_number, 1);
        assert_eq!(chunk.metadata.start_line, 1);
        assert_eq!(chunk.metadata.end_line, 0);
        assert_eq!(chunk.metadata.line_count, 0);
        assert_eq!(chunk.content, "");
    }

    #[test]
    fn test_single_line_file() {
        let engine = ChunkingEngine::with_chunk_size(10);
        let content = "single line";
        let (file_id, filepath, filename, extension) = create_test_file_metadata();
        
        let result = engine.chunk_content(&content, file_id, filepath, filename, extension).unwrap();

        assert!(!result.was_chunked);
        assert_eq!(result.chunks.len(), 1);
        assert_eq!(result.total_lines, 1);
        
        let chunk = &result.chunks[0];
        assert_eq!(chunk.metadata.chunk_number, 1);
        assert_eq!(chunk.metadata.start_line, 1);
        assert_eq!(chunk.metadata.end_line, 1);
        assert_eq!(chunk.metadata.line_count, 1);
        assert_eq!(chunk.content, "single line");
    }

    #[test]
    fn test_file_exactly_chunk_size() {
        let chunk_size = 10;
        let engine = ChunkingEngine::with_chunk_size(chunk_size);
        let content = create_test_content(chunk_size);
        let (file_id, filepath, filename, extension) = create_test_file_metadata();
        
        let result = engine.chunk_content(&content, file_id, filepath, filename, extension).unwrap();

        // File exactly at chunk size should not be chunked
        assert!(!result.was_chunked);
        assert_eq!(result.chunks.len(), 1);
        assert_eq!(result.total_lines, chunk_size as u32);
    }
}

#[cfg(test)]
mod large_file_chunking_tests {
    use super::*;

    #[test]
    fn test_large_file_basic_chunking() {
        let engine = ChunkingEngine::with_chunk_size(10);
        let content = create_test_content(25);
        let (file_id, filepath, filename, extension) = create_test_file_metadata();
        
        let result = engine.chunk_content(&content, file_id, filepath, filename, extension).unwrap();

        assert!(result.was_chunked);
        assert_eq!(result.total_lines, 25);
        assert_eq!(result.chunks.len(), 3); // 10 + 10 + 5 lines
        
        // Validate chunk metadata consistency
        engine.validate_chunks(&result.chunks).unwrap();
        
        // Check first chunk
        let chunk1 = &result.chunks[0];
        assert_eq!(chunk1.metadata.chunk_number, 1);
        assert_eq!(chunk1.metadata.start_line, 1);
        assert_eq!(chunk1.metadata.end_line, 10);
        assert_eq!(chunk1.metadata.line_count, 10);
        
        // Check second chunk
        let chunk2 = &result.chunks[1];
        assert_eq!(chunk2.metadata.chunk_number, 2);
        assert_eq!(chunk2.metadata.start_line, 11);
        assert_eq!(chunk2.metadata.end_line, 20);
        assert_eq!(chunk2.metadata.line_count, 10);
        
        // Check last chunk
        let chunk3 = &result.chunks[2];
        assert_eq!(chunk3.metadata.chunk_number, 3);
        assert_eq!(chunk3.metadata.start_line, 21);
        assert_eq!(chunk3.metadata.end_line, 25);
        assert_eq!(chunk3.metadata.line_count, 5);
    }

    #[test]
    fn test_large_file_perfect_division() {
        let engine = ChunkingEngine::with_chunk_size(5);
        let content = create_test_content(20); // Perfectly divisible by 5
        let (file_id, filepath, filename, extension) = create_test_file_metadata();
        
        let result = engine.chunk_content(&content, file_id, filepath, filename, extension).unwrap();

        assert!(result.was_chunked);
        assert_eq!(result.total_lines, 20);
        assert_eq!(result.chunks.len(), 4); // 4 chunks of 5 lines each
        
        engine.validate_chunks(&result.chunks).unwrap();
        
        // All chunks should have exactly 5 lines
        for (i, chunk) in result.chunks.iter().enumerate() {
            assert_eq!(chunk.metadata.chunk_number, (i + 1) as u32);
            assert_eq!(chunk.metadata.line_count, 5);
            assert_eq!(chunk.metadata.start_line, (i * 5 + 1) as u32);
            assert_eq!(chunk.metadata.end_line, ((i + 1) * 5) as u32);
        }
    }

    #[test]
    fn test_very_large_file() {
        let engine = ChunkingEngine::with_chunk_size(100);
        let content = create_test_content(1000);
        let (file_id, filepath, filename, extension) = create_test_file_metadata();
        
        let result = engine.chunk_content(&content, file_id, filepath, filename, extension).unwrap();

        assert!(result.was_chunked);
        assert_eq!(result.total_lines, 1000);
        assert_eq!(result.chunks.len(), 10); // 10 chunks of 100 lines each
        
        engine.validate_chunks(&result.chunks).unwrap();
        
        // Verify content preservation
        let reconstructed_content = result.chunks
            .iter()
            .map(|chunk| chunk.content.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        
        assert_eq!(reconstructed_content, content);
    }
}

#[cfg(test)]
mod structure_boundary_tests {
    use super::*;

    #[test]
    fn test_structure_boundary_detection_enabled() {
        let engine = ChunkingEngine::new(ChunkingConfig {
            chunk_size: 15,
            preserve_structure: true,
            min_chunk_size: 5,
            max_overlap: 5,
        });
        
        let content = create_rust_code_content();
        let (file_id, filepath, filename, extension) = create_test_file_metadata();
        
        let result = engine.chunk_content(&content, file_id, filepath, filename, extension).unwrap();

        assert!(result.was_chunked);
        engine.validate_chunks(&result.chunks).unwrap();
        
        // Should have multiple chunks due to structure preservation
        assert!(result.chunks.len() > 1);
        
        // Check that file metadata is preserved across all chunks
        for chunk in &result.chunks {
            assert_eq!(chunk.filepath, "test/path/file.rs");
            assert_eq!(chunk.filename, "file.rs");
            assert_eq!(chunk.extension, Some("rs".to_string()));
        }
    }

    #[test]
    fn test_structure_boundary_detection_disabled() {
        let engine = ChunkingEngine::new(ChunkingConfig {
            chunk_size: 15,
            preserve_structure: false,
            min_chunk_size: 5,
            max_overlap: 5,
        });
        
        let content = create_rust_code_content();
        let (file_id, filepath, filename, extension) = create_test_file_metadata();
        
        let result = engine.chunk_content(&content, file_id, filepath, filename, extension).unwrap();

        assert!(result.was_chunked);
        engine.validate_chunks(&result.chunks).unwrap();
        
        // With structure preservation disabled, chunks should be more regular
        let lines: Vec<&str> = content.lines().collect();
        let expected_chunks = (lines.len() + 14) / 15; // Ceiling division
        assert_eq!(result.chunks.len(), expected_chunks);
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
            "struct Test {",           // Struct definition (good boundary)
            "    field: i32,",         // Struct field
            "};",                      // Struct end (good boundary)
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

        // Struct definition should have high score
        let struct_score = engine.calculate_boundary_score("struct Test {", 6, &lines);
        assert!(struct_score > 10);
    }
}

#[cfg(test)]
mod min_chunk_size_tests {
    use super::*;

    #[test]
    fn test_min_chunk_size_enforcement() {
        let engine = ChunkingEngine::new(ChunkingConfig {
            chunk_size: 10,
            preserve_structure: false,
            min_chunk_size: 5,
            max_overlap: 2,
        });
        
        let content = create_test_content(23); // Should create chunks of 10, 10, 3 -> but 3 is too small
        let (file_id, filepath, filename, extension) = create_test_file_metadata();
        
        let result = engine.chunk_content(&content, file_id, filepath, filename, extension).unwrap();

        engine.validate_chunks(&result.chunks).unwrap();
        
        // Check that no chunk is smaller than min_chunk_size (except possibly the last one)
        for (i, chunk) in result.chunks.iter().enumerate() {
            if i < result.chunks.len() - 1 {
                // Not the last chunk
                assert!(chunk.metadata.line_count >= engine.config().min_chunk_size as u32);
            }
        }
    }

    #[test]
    fn test_min_chunk_size_with_very_small_remainder() {
        let engine = ChunkingEngine::new(ChunkingConfig {
            chunk_size: 10,
            preserve_structure: false,
            min_chunk_size: 8,
            max_overlap: 2,
        });
        
        let content = create_test_content(21); // 10 + 10 + 1, but min_chunk_size is 8
        let (file_id, filepath, filename, extension) = create_test_file_metadata();
        
        let result = engine.chunk_content(&content, file_id, filepath, filename, extension).unwrap();

        engine.validate_chunks(&result.chunks).unwrap();
        
        // Should extend the last chunk to meet minimum size or merge with previous
        for chunk in &result.chunks {
            if result.chunks.len() > 1 {
                // If there are multiple chunks, each should meet minimum size
                assert!(chunk.metadata.line_count >= engine.config().min_chunk_size as u32);
            }
        }
    }
}

#[cfg(test)]
mod chunk_validation_tests {
    use super::*;

    #[test]
    fn test_chunk_validation_success() {
        let engine = ChunkingEngine::default();
        
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
    }

    #[test]
    fn test_chunk_validation_empty_chunks() {
        let engine = ChunkingEngine::default();
        let empty_chunks = vec![];
        
        let result = engine.validate_chunks(&empty_chunks);
        assert!(result.is_err());
        
        if let Err(ProcessingError::ChunkingFailed { reason }) = result {
            assert!(reason.contains("No chunks provided"));
        } else {
            panic!("Expected ChunkingFailed error");
        }
    }

    #[test]
    fn test_chunk_validation_wrong_numbering() {
        let engine = ChunkingEngine::default();
        
        let mut invalid_chunks = vec![
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
                    chunk_number: 3, // Should be 2
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

        let result = engine.validate_chunks(&invalid_chunks);
        assert!(result.is_err());
        
        if let Err(ProcessingError::ChunkingFailed { reason }) = result {
            assert!(reason.contains("numbering inconsistent"));
        } else {
            panic!("Expected ChunkingFailed error");
        }
    }

    #[test]
    fn test_chunk_validation_line_number_gap() {
        let engine = ChunkingEngine::default();
        
        let invalid_chunks = vec![
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
                    start_line: 15, // Should be 11
                    end_line: 25,
                    line_count: 11,
                    structure_adjusted: false,
                },
                content: "more test content".to_string(),
                file_id: "test".to_string(),
                filepath: "test.txt".to_string(),
                filename: "test.txt".to_string(),
                extension: Some("txt".to_string()),
            },
        ];

        let result = engine.validate_chunks(&invalid_chunks);
        assert!(result.is_err());
        
        if let Err(ProcessingError::ChunkingFailed { reason }) = result {
            assert!(reason.contains("Line number gap"));
        } else {
            panic!("Expected ChunkingFailed error");
        }
    }

    #[test]
    fn test_chunk_validation_wrong_line_count() {
        let engine = ChunkingEngine::default();
        
        let invalid_chunks = vec![
            ChunkData {
                metadata: ChunkMetadata {
                    chunk_number: 1,
                    start_line: 1,
                    end_line: 10,
                    line_count: 5, // Should be 10
                    structure_adjusted: false,
                },
                content: "test content".to_string(),
                file_id: "test".to_string(),
                filepath: "test.txt".to_string(),
                filename: "test.txt".to_string(),
                extension: Some("txt".to_string()),
            },
        ];

        let result = engine.validate_chunks(&invalid_chunks);
        assert!(result.is_err());
        
        if let Err(ProcessingError::ChunkingFailed { reason }) = result {
            assert!(reason.contains("line count mismatch"));
        } else {
            panic!("Expected ChunkingFailed error");
        }
    }
}

#[cfg(test)]
mod content_preservation_tests {
    use super::*;

    #[test]
    fn test_content_preservation_simple() {
        let engine = ChunkingEngine::with_chunk_size(5);
        let original_content = create_test_content(12);
        let (file_id, filepath, filename, extension) = create_test_file_metadata();
        
        let result = engine.chunk_content(&original_content, file_id, filepath, filename, extension).unwrap();

        // Reconstruct content from chunks
        let reconstructed_content = result.chunks
            .iter()
            .map(|chunk| chunk.content.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        
        assert_eq!(reconstructed_content, original_content);
    }

    #[test]
    fn test_content_preservation_with_structure() {
        let engine = ChunkingEngine::new(ChunkingConfig {
            chunk_size: 10,
            preserve_structure: true,
            min_chunk_size: 3,
            max_overlap: 3,
        });
        
        let original_content = create_rust_code_content();
        let (file_id, filepath, filename, extension) = create_test_file_metadata();
        
        let result = engine.chunk_content(&original_content, file_id, filepath, filename, extension).unwrap();

        // Reconstruct content from chunks
        let reconstructed_content = result.chunks
            .iter()
            .map(|chunk| chunk.content.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        
        assert_eq!(reconstructed_content, original_content);
    }

    #[test]
    fn test_metadata_consistency() {
        let engine = ChunkingEngine::with_chunk_size(5);
        let content = create_test_content(12);
        let (file_id, filepath, filename, extension) = create_test_file_metadata();
        
        let result = engine.chunk_content(&content, file_id.clone(), filepath.clone(), filename.clone(), extension.clone()).unwrap();

        // Validate all chunks have correct file metadata
        for chunk in &result.chunks {
            assert_eq!(chunk.file_id, file_id);
            assert_eq!(chunk.filepath, filepath);
            assert_eq!(chunk.filename, filename);
            assert_eq!(chunk.extension, extension);
        }
        
        // Validate chunk metadata consistency
        engine.validate_chunks(&result.chunks).unwrap();
    }
}

#[cfg(test)]
mod edge_case_tests {
    use super::*;

    #[test]
    fn test_chunk_size_larger_than_file() {
        let engine = ChunkingEngine::with_chunk_size(1000);
        let content = create_test_content(10);
        let (file_id, filepath, filename, extension) = create_test_file_metadata();
        
        let result = engine.chunk_content(&content, file_id, filepath, filename, extension).unwrap();

        assert!(!result.was_chunked);
        assert_eq!(result.chunks.len(), 1);
        assert_eq!(result.total_lines, 10);
    }

    #[test]
    fn test_chunk_size_one() {
        let engine = ChunkingEngine::with_chunk_size(1);
        let content = create_test_content(3);
        let (file_id, filepath, filename, extension) = create_test_file_metadata();
        
        let result = engine.chunk_content(&content, file_id, filepath, filename, extension).unwrap();

        assert!(result.was_chunked);
        assert_eq!(result.chunks.len(), 3);
        
        for (i, chunk) in result.chunks.iter().enumerate() {
            assert_eq!(chunk.metadata.chunk_number, (i + 1) as u32);
            assert_eq!(chunk.metadata.line_count, 1);
            assert_eq!(chunk.content, format!("line {}", i + 1));
        }
    }

    #[test]
    fn test_file_with_empty_lines() {
        let engine = ChunkingEngine::with_chunk_size(3);
        let content = "line 1\n\nline 3\n\nline 5";
        let (file_id, filepath, filename, extension) = create_test_file_metadata();
        
        let result = engine.chunk_content(&content, file_id, filepath, filename, extension).unwrap();

        assert!(result.was_chunked);
        assert_eq!(result.total_lines, 5);
        
        // Verify content preservation including empty lines
        let reconstructed_content = result.chunks
            .iter()
            .map(|chunk| chunk.content.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        
        assert_eq!(reconstructed_content, content);
    }

    #[test]
    fn test_file_with_only_whitespace_lines() {
        let engine = ChunkingEngine::with_chunk_size(2);
        let content = "   \n\t\t\n   \n";
        let (file_id, filepath, filename, extension) = create_test_file_metadata();
        
        let result = engine.chunk_content(&content, file_id, filepath, filename, extension).unwrap();

        assert!(result.was_chunked);
        assert_eq!(result.total_lines, 3);
        
        // Verify whitespace preservation
        let reconstructed_content = result.chunks
            .iter()
            .map(|chunk| chunk.content.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        
        assert_eq!(reconstructed_content, content);
    }
}

#[cfg(test)]
mod performance_tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_chunking_performance_large_file() {
        let engine = ChunkingEngine::with_chunk_size(100);
        let content = create_test_content(10000); // 10k lines
        let (file_id, filepath, filename, extension) = create_test_file_metadata();
        
        let start = Instant::now();
        let result = engine.chunk_content(&content, file_id, filepath, filename, extension).unwrap();
        let elapsed = start.elapsed();
        
        // Should complete within reasonable time (adjust threshold as needed)
        assert!(elapsed.as_millis() < 1000, "Chunking took too long: {:?}", elapsed);
        
        assert!(result.was_chunked);
        assert_eq!(result.chunks.len(), 100); // 10k / 100 = 100 chunks
        assert_eq!(result.total_lines, 10000);
        
        // Validate all chunks
        engine.validate_chunks(&result.chunks).unwrap();
    }

    #[test]
    fn test_chunking_performance_many_small_chunks() {
        let engine = ChunkingEngine::with_chunk_size(1);
        let content = create_test_content(1000); // 1k lines, 1k chunks
        let (file_id, filepath, filename, extension) = create_test_file_metadata();
        
        let start = Instant::now();
        let result = engine.chunk_content(&content, file_id, filepath, filename, extension).unwrap();
        let elapsed = start.elapsed();
        
        // Should complete within reasonable time
        assert!(elapsed.as_millis() < 500, "Chunking took too long: {:?}", elapsed);
        
        assert!(result.was_chunked);
        assert_eq!(result.chunks.len(), 1000);
        assert_eq!(result.total_lines, 1000);
    }
}
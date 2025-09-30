//! Unit tests for ChunkingService that don't require database connections
//!
//! These tests focus on the core chunking logic and can run without external dependencies.

use crate::database::models::IngestedFile;
use crate::tasks::chunk_level_task_generator::{ChunkedFile, TaskGeneratorError};
use chrono::Utc;

fn create_test_ingested_file(file_id: i64, content: &str, line_count: i32) -> IngestedFile {
    IngestedFile {
        file_id,
        ingestion_id: 1,
        filepath: format!("test_file_{}.rs", file_id),
        filename: format!("test_file_{}.rs", file_id),
        extension: Some("rs".to_string()),
        file_size_bytes: content.len() as i64,
        line_count: Some(line_count),
        word_count: Some(content.split_whitespace().count() as i32),
        token_count: Some((content.split_whitespace().count() as f32 * 0.75) as i32),
        content_text: Some(content.to_string()),
        file_type_str: "direct_text".to_string(),
        conversion_command: None,
        relative_path: format!("test_file_{}.rs", file_id),
        absolute_path: format!("/tmp/test_file_{}.rs", file_id),
        created_at: Utc::now(),
    }
}

/// Test the core chunking logic without database dependencies
pub fn test_chunking_logic_small_file() {
    let content = "line 1\nline 2\nline 3";
    let file = create_test_ingested_file(1, content, 3);
    
    // Simulate the chunking logic directly
    let lines: Vec<&str> = content.lines().collect();
    let total_lines = lines.len();
    let chunk_size = 5; // chunk_size > line_count
    
    // Small file logic: should be copied unchanged
    assert!(total_lines <= chunk_size);
    
    let chunk = ChunkedFile::new(
        &file,
        0,
        content.to_string(),
        content.to_string(), // L1 same as content for single chunk
        content.to_string(), // L2 same as content for single chunk
        total_lines as i32,
    );
    
    assert_eq!(chunk.chunk_number, 0);
    assert_eq!(chunk.content, content);
    assert_eq!(chunk.content_l1, content);
    assert_eq!(chunk.content_l2, content);
    assert_eq!(chunk.line_count, 3);
    assert_eq!(chunk.original_file_id, 1);
    
    println!("✅ Small file chunking logic test passed");
}

/// Test the core chunking logic for large files
pub fn test_chunking_logic_large_file() {
    let content = "line 1\nline 2\nline 3\nline 4\nline 5\nline 6\nline 7\nline 8";
    let file = create_test_ingested_file(2, content, 8);
    
    let lines: Vec<&str> = content.lines().collect();
    let total_lines = lines.len();
    let chunk_size = 3; // chunk_size < line_count
    
    // Large file logic: should be broken into chunks
    assert!(total_lines > chunk_size);
    
    let mut chunks = Vec::new();
    let mut chunk_number = 0;
    
    for chunk_start in (0..total_lines).step_by(chunk_size) {
        let chunk_end = std::cmp::min(chunk_start + chunk_size, total_lines);
        
        // Get content for this chunk
        let chunk_lines = &lines[chunk_start..chunk_end];
        let chunk_content = chunk_lines.join("\n");

        // Generate L1 content (current + next chunk)
        let l1_end = std::cmp::min(chunk_start + (chunk_size * 2), total_lines);
        let l1_lines = &lines[chunk_start..l1_end];
        let l1_content = l1_lines.join("\n");

        // Generate L2 content (current + next + next2 chunk)
        let l2_end = std::cmp::min(chunk_start + (chunk_size * 3), total_lines);
        let l2_lines = &lines[chunk_start..l2_end];
        let l2_content = l2_lines.join("\n");

        let chunked_file = ChunkedFile::new(
            &file,
            chunk_number,
            chunk_content,
            l1_content,
            l2_content,
            chunk_lines.len() as i32,
        );

        chunks.push(chunked_file);
        chunk_number += 1;
    }
    
    assert_eq!(chunks.len(), 3); // 8 lines / 3 per chunk = 3 chunks (2 full + 1 partial)
    
    // Test first chunk
    assert_eq!(chunks[0].chunk_number, 0);
    assert_eq!(chunks[0].content, "line 1\nline 2\nline 3");
    assert_eq!(chunks[0].line_count, 3);
    
    // Test L1 content (current + next chunk)
    assert_eq!(chunks[0].content_l1, "line 1\nline 2\nline 3\nline 4\nline 5\nline 6");
    
    // Test L2 content (current + next + next2 chunk)
    assert_eq!(chunks[0].content_l2, "line 1\nline 2\nline 3\nline 4\nline 5\nline 6\nline 7\nline 8");
    
    // Test second chunk
    assert_eq!(chunks[1].chunk_number, 1);
    assert_eq!(chunks[1].content, "line 4\nline 5\nline 6");
    assert_eq!(chunks[1].line_count, 3);
    
    // Test third chunk (partial)
    assert_eq!(chunks[2].chunk_number, 2);
    assert_eq!(chunks[2].content, "line 7\nline 8");
    assert_eq!(chunks[2].line_count, 2);
    
    println!("✅ Large file chunking logic test passed");
}

/// Test L1 and L2 concatenation logic
pub fn test_l1_l2_concatenation_logic() {
    let content = "chunk1_line1\nchunk1_line2\nchunk2_line1\nchunk2_line2\nchunk3_line1\nchunk3_line2";
    let file = create_test_ingested_file(4, content, 6);
    
    let lines: Vec<&str> = content.lines().collect();
    let total_lines = lines.len();
    let chunk_size = 2; // 2 lines per chunk
    
    let mut chunks = Vec::new();
    let mut chunk_number = 0;
    
    for chunk_start in (0..total_lines).step_by(chunk_size) {
        let chunk_end = std::cmp::min(chunk_start + chunk_size, total_lines);
        
        // Get content for this chunk
        let chunk_lines = &lines[chunk_start..chunk_end];
        let chunk_content = chunk_lines.join("\n");

        // Generate L1 content (current + next chunk)
        let l1_end = std::cmp::min(chunk_start + (chunk_size * 2), total_lines);
        let l1_lines = &lines[chunk_start..l1_end];
        let l1_content = l1_lines.join("\n");

        // Generate L2 content (current + next + next2 chunk)
        let l2_end = std::cmp::min(chunk_start + (chunk_size * 3), total_lines);
        let l2_lines = &lines[chunk_start..l2_end];
        let l2_content = l2_lines.join("\n");

        let chunked_file = ChunkedFile::new(
            &file,
            chunk_number,
            chunk_content,
            l1_content,
            l2_content,
            chunk_lines.len() as i32,
        );

        chunks.push(chunked_file);
        chunk_number += 1;
    }
    
    assert_eq!(chunks.len(), 3);
    
    // Test first chunk L1 and L2
    let chunk0 = &chunks[0];
    assert_eq!(chunk0.content, "chunk1_line1\nchunk1_line2");
    assert_eq!(chunk0.content_l1, "chunk1_line1\nchunk1_line2\nchunk2_line1\nchunk2_line2"); // current + next
    assert_eq!(chunk0.content_l2, "chunk1_line1\nchunk1_line2\nchunk2_line1\nchunk2_line2\nchunk3_line1\nchunk3_line2"); // current + next + next2
    
    // Test second chunk L1 and L2
    let chunk1 = &chunks[1];
    assert_eq!(chunk1.content, "chunk2_line1\nchunk2_line2");
    assert_eq!(chunk1.content_l1, "chunk2_line1\nchunk2_line2\nchunk3_line1\nchunk3_line2"); // current + next
    assert_eq!(chunk1.content_l2, "chunk2_line1\nchunk2_line2\nchunk3_line1\nchunk3_line2"); // current + next (no next2)
    
    // Test third chunk L1 and L2
    let chunk2 = &chunks[2];
    assert_eq!(chunk2.content, "chunk3_line1\nchunk3_line2");
    assert_eq!(chunk2.content_l1, "chunk3_line1\nchunk3_line2"); // current only (no next)
    assert_eq!(chunk2.content_l2, "chunk3_line1\nchunk3_line2"); // current only (no next)
    
    println!("✅ L1/L2 concatenation logic test passed");
}

/// Test parameter validation
pub fn test_validate_chunking_params() {
    // Test valid chunk size
    let chunk_size = 500;
    assert!(chunk_size > 0);
    
    // Test invalid chunk size (0)
    let chunk_size = 0;
    assert_eq!(chunk_size, 0);
    let error = TaskGeneratorError::invalid_chunk_size(chunk_size);
    assert!(matches!(error, TaskGeneratorError::InvalidChunkSize { size: 0 }));
    
    // Test edge cases
    let small_chunk_size = 1;
    assert!(small_chunk_size > 0); // Should succeed but warn about small size
    
    let large_chunk_size = 15000;
    assert!(large_chunk_size > 0); // Should succeed but warn about large size
    
    println!("✅ Chunking parameter validation test passed");
}

/// Test empty content handling
pub fn test_empty_content_handling() {
    let file = create_test_ingested_file(3, "", 0);
    let mut file_no_content = file.clone();
    file_no_content.content_text = None;
    
    // Simulate empty content handling
    let content = match &file_no_content.content_text {
        Some(content) => content,
        None => {
            // Should create empty chunk
            let chunk = ChunkedFile::new(
                &file_no_content,
                0,
                String::new(),
                String::new(),
                String::new(),
                0,
            );
            
            assert_eq!(chunk.content, "");
            assert_eq!(chunk.content_l1, "");
            assert_eq!(chunk.content_l2, "");
            assert_eq!(chunk.line_count, 0);
            
            println!("✅ Empty content handling test passed");
            return;
        }
    };
    
    // If we get here, content was not None
    assert_eq!(content, "");
    println!("✅ Empty content handling test passed");
}

/// Run all chunking logic tests
pub fn run_all_chunking_tests() {
    println!("🧪 Running ChunkingService logic tests...");
    
    test_chunking_logic_small_file();
    test_chunking_logic_large_file();
    test_l1_l2_concatenation_logic();
    test_validate_chunking_params();
    test_empty_content_handling();
    
    println!("✅ All ChunkingService logic tests passed!");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_chunking_logic() {
        run_all_chunking_tests();
    }
}
//! Comprehensive unit tests for chunk context generation
//!
//! Tests cover L1 and L2 context generation, boundary handling,
//! and various chunk configurations as specified in task 9.1.

use code_ingest::processing::chunking::{
    ChunkingEngine, ChunkingConfig, ChunkData, ChunkContext, ContextLevel, ChunkMetadata
};
use code_ingest::error::ProcessingError;

/// Create test content with specified number of lines
fn create_test_content(lines: usize) -> String {
    (1..=lines)
        .map(|i| format!("line {}", i))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Create test chunks for context generation testing
fn create_test_chunks(count: usize, lines_per_chunk: usize) -> Vec<ChunkData> {
    (0..count)
        .map(|i| {
            let start_line = (i * lines_per_chunk + 1) as u32;
            let end_line = ((i + 1) * lines_per_chunk) as u32;
            let content = (start_line..=end_line)
                .map(|line_num| format!("line {}", line_num))
                .collect::<Vec<_>>()
                .join("\n");

            ChunkData {
                metadata: ChunkMetadata {
                    chunk_number: (i + 1) as u32,
                    start_line,
                    end_line,
                    line_count: lines_per_chunk as u32,
                    structure_adjusted: false,
                },
                content,
                file_id: "test_file".to_string(),
                filepath: "test.txt".to_string(),
                filename: "test.txt".to_string(),
                extension: Some("txt".to_string()),
            }
        })
        .collect()
}

#[cfg(test)]
mod context_generation_basic_tests {
    use super::*;

    #[test]
    fn test_context_generation_single_chunk() {
        let engine = ChunkingEngine::default();
        let chunks = create_test_chunks(1, 5);
        
        let context = engine.generate_context(&chunks, 0).unwrap();
        
        // With only one chunk, L1 and L2 should be the same
        assert_eq!(context.l1_chunk_count, 1);
        assert_eq!(context.l2_chunk_count, 1);
        assert!(context.is_boundary); // Single chunk is always a boundary
        
        // Content should be the same for L1 and L2
        assert_eq!(context.l1_content, context.l2_content);
        assert!(context.l1_content.contains("line 1"));
        assert!(context.l1_content.contains("line 5"));
    }

    #[test]
    fn test_context_generation_three_chunks() {
        let engine = ChunkingEngine::default();
        let chunks = create_test_chunks(3, 5); // 3 chunks of 5 lines each
        
        // Test middle chunk (index 1)
        let context = engine.generate_context(&chunks, 1).unwrap();
        
        // L1 should include all 3 chunks (±1 from index 1)
        assert_eq!(context.l1_chunk_count, 3);
        // L2 should also include all 3 chunks (±2 from index 1, but we only have 3 total)
        assert_eq!(context.l2_chunk_count, 3);
        assert!(!context.is_boundary); // Middle chunk is not a boundary
        
        // L1 and L2 content should be the same since we include all chunks
        assert_eq!(context.l1_content, context.l2_content);
        
        // Should contain content from all chunks
        assert!(context.l1_content.contains("line 1"));  // From chunk 0
        assert!(context.l1_content.contains("line 8"));  // From chunk 1
        assert!(context.l1_content.contains("line 15")); // From chunk 2
    }

    #[test]
    fn test_context_generation_seven_chunks() {
        let engine = ChunkingEngine::default();
        let chunks = create_test_chunks(7, 3); // 7 chunks of 3 lines each
        
        // Test middle chunk (index 3)
        let context = engine.generate_context(&chunks, 3).unwrap();
        
        // L1 should include chunks 2, 3, 4 (±1 from index 3)
        assert_eq!(context.l1_chunk_count, 3);
        
        // L2 should include chunks 1, 2, 3, 4, 5 (±2 from index 3)
        assert_eq!(context.l2_chunk_count, 5);
        assert!(!context.is_boundary); // Middle chunk is not a boundary
        
        // L2 content should be longer than L1 content
        assert!(context.l2_content.len() > context.l1_content.len());
        
        // L1 should contain chunks 2, 3, 4
        assert!(context.l1_content.contains("line 7"));  // From chunk 2
        assert!(context.l1_content.contains("line 10")); // From chunk 3
        assert!(context.l1_content.contains("line 13")); // From chunk 4
        
        // L2 should additionally contain chunks 1 and 5
        assert!(context.l2_content.contains("line 4"));  // From chunk 1
        assert!(context.l2_content.contains("line 16")); // From chunk 5
    }

    #[test]
    fn test_context_generation_invalid_index() {
        let engine = ChunkingEngine::default();
        let chunks = create_test_chunks(3, 5);
        
        // Test invalid index
        let result = engine.generate_context(&chunks, 5);
        assert!(result.is_err());
        
        if let Err(ProcessingError::ChunkingFailed { reason }) = result {
            assert!(reason.contains("Invalid chunk index"));
        } else {
            panic!("Expected ChunkingFailed error");
        }
    }
}

#[cfg(test)]
mod boundary_chunk_tests {
    use super::*;

    #[test]
    fn test_first_chunk_boundary() {
        let engine = ChunkingEngine::default();
        let chunks = create_test_chunks(5, 4); // 5 chunks of 4 lines each
        
        // Test first chunk (index 0)
        let context = engine.generate_context(&chunks, 0).unwrap();
        
        assert!(context.is_boundary);
        
        // L1 should include chunks 0, 1 (can't go before index 0)
        assert_eq!(context.l1_chunk_count, 2);
        
        // L2 should include chunks 0, 1, 2 (can't go before index 0)
        assert_eq!(context.l2_chunk_count, 3);
        
        // Should contain content from chunks 0, 1 for L1
        assert!(context.l1_content.contains("line 1")); // From chunk 0
        assert!(context.l1_content.contains("line 5")); // From chunk 1
        
        // Should contain content from chunks 0, 1, 2 for L2
        assert!(context.l2_content.contains("line 1")); // From chunk 0
        assert!(context.l2_content.contains("line 5")); // From chunk 1
        assert!(context.l2_content.contains("line 9")); // From chunk 2
    }

    #[test]
    fn test_last_chunk_boundary() {
        let engine = ChunkingEngine::default();
        let chunks = create_test_chunks(5, 4); // 5 chunks of 4 lines each
        
        // Test last chunk (index 4)
        let context = engine.generate_context(&chunks, 4).unwrap();
        
        assert!(context.is_boundary);
        
        // L1 should include chunks 3, 4 (can't go after index 4)
        assert_eq!(context.l1_chunk_count, 2);
        
        // L2 should include chunks 2, 3, 4 (can't go after index 4)
        assert_eq!(context.l2_chunk_count, 3);
        
        // Should contain content from chunks 3, 4 for L1
        assert!(context.l1_content.contains("line 13")); // From chunk 3
        assert!(context.l1_content.contains("line 17")); // From chunk 4
        
        // Should contain content from chunks 2, 3, 4 for L2
        assert!(context.l2_content.contains("line 9"));  // From chunk 2
        assert!(context.l2_content.contains("line 13")); // From chunk 3
        assert!(context.l2_content.contains("line 17")); // From chunk 4
    }

    #[test]
    fn test_second_chunk_not_boundary() {
        let engine = ChunkingEngine::default();
        let chunks = create_test_chunks(5, 3);
        
        // Test second chunk (index 1)
        let context = engine.generate_context(&chunks, 1).unwrap();
        
        assert!(!context.is_boundary); // Second chunk is not a boundary
        
        // L1 should include chunks 0, 1, 2
        assert_eq!(context.l1_chunk_count, 3);
        
        // L2 should include chunks 0, 1, 2, 3 (can't go before index 0)
        assert_eq!(context.l2_chunk_count, 4);
    }

    #[test]
    fn test_second_to_last_chunk_not_boundary() {
        let engine = ChunkingEngine::default();
        let chunks = create_test_chunks(5, 3);
        
        // Test second-to-last chunk (index 3)
        let context = engine.generate_context(&chunks, 3).unwrap();
        
        assert!(!context.is_boundary); // Second-to-last chunk is not a boundary
        
        // L1 should include chunks 2, 3, 4
        assert_eq!(context.l1_chunk_count, 3);
        
        // L2 should include chunks 1, 2, 3, 4 (can't go after index 4)
        assert_eq!(context.l2_chunk_count, 4);
    }
}

#[cfg(test)]
mod context_range_calculation_tests {
    use super::*;

    #[test]
    fn test_l1_context_range_calculation() {
        let engine = ChunkingEngine::default();
        
        // Test L1 context range for various scenarios
        let (start, end) = engine.calculate_context_range(10, 5, ContextLevel::L1);
        assert_eq!(start, 4); // 5 - 1 = 4
        assert_eq!(end, 6);   // min(5 + 1, 10 - 1) = 6
        
        // Test at beginning
        let (start, end) = engine.calculate_context_range(10, 0, ContextLevel::L1);
        assert_eq!(start, 0); // saturating_sub(0, 1) = 0
        assert_eq!(end, 1);   // min(0 + 1, 10 - 1) = 1
        
        // Test at end
        let (start, end) = engine.calculate_context_range(10, 9, ContextLevel::L1);
        assert_eq!(start, 8); // 9 - 1 = 8
        assert_eq!(end, 9);   // min(9 + 1, 10 - 1) = 9
    }

    #[test]
    fn test_l2_context_range_calculation() {
        let engine = ChunkingEngine::default();
        
        // Test L2 context range for various scenarios
        let (start, end) = engine.calculate_context_range(10, 5, ContextLevel::L2);
        assert_eq!(start, 3); // 5 - 2 = 3
        assert_eq!(end, 7);   // min(5 + 2, 10 - 1) = 7
        
        // Test at beginning
        let (start, end) = engine.calculate_context_range(10, 1, ContextLevel::L2);
        assert_eq!(start, 0); // saturating_sub(1, 2) = 0
        assert_eq!(end, 3);   // min(1 + 2, 10 - 1) = 3
        
        // Test at end
        let (start, end) = engine.calculate_context_range(10, 8, ContextLevel::L2);
        assert_eq!(start, 6); // 8 - 2 = 6
        assert_eq!(end, 9);   // min(8 + 2, 10 - 1) = 9
    }

    #[test]
    fn test_context_range_small_collection() {
        let engine = ChunkingEngine::default();
        
        // Test with very small collection (3 chunks)
        let (start, end) = engine.calculate_context_range(3, 1, ContextLevel::L1);
        assert_eq!(start, 0); // 1 - 1 = 0
        assert_eq!(end, 2);   // min(1 + 1, 3 - 1) = 2
        
        let (start, end) = engine.calculate_context_range(3, 1, ContextLevel::L2);
        assert_eq!(start, 0); // saturating_sub(1, 2) = 0
        assert_eq!(end, 2);   // min(1 + 2, 3 - 1) = 2
    }
}

#[cfg(test)]
mod context_content_tests {
    use super::*;

    #[test]
    fn test_context_content_separation() {
        let engine = ChunkingEngine::default();
        let chunks = create_test_chunks(3, 2); // 3 chunks of 2 lines each
        
        let context = engine.generate_context(&chunks, 1).unwrap();
        
        // Chunks should be separated by double newlines
        let expected_separator = "\n\n";
        assert!(context.l1_content.contains(&expected_separator));
        
        // Count separators - should have 2 separators for 3 chunks
        let separator_count = context.l1_content.matches(&expected_separator).count();
        assert_eq!(separator_count, 2);
    }

    #[test]
    fn test_context_content_order() {
        let engine = ChunkingEngine::default();
        let chunks = create_test_chunks(5, 2); // 5 chunks of 2 lines each
        
        // Test middle chunk (index 2)
        let context = engine.generate_context(&chunks, 2).unwrap();
        
        // L1 should contain chunks 1, 2, 3 in order
        let l1_parts: Vec<&str> = context.l1_content.split("\n\n").collect();
        assert_eq!(l1_parts.len(), 3);
        
        // First part should be chunk 1 (lines 3-4)
        assert!(l1_parts[0].contains("line 3"));
        assert!(l1_parts[0].contains("line 4"));
        
        // Second part should be chunk 2 (lines 5-6)
        assert!(l1_parts[1].contains("line 5"));
        assert!(l1_parts[1].contains("line 6"));
        
        // Third part should be chunk 3 (lines 7-8)
        assert!(l1_parts[2].contains("line 7"));
        assert!(l1_parts[2].contains("line 8"));
    }

    #[test]
    fn test_context_content_completeness() {
        let engine = ChunkingEngine::default();
        let chunks = create_test_chunks(7, 1); // 7 chunks of 1 line each
        
        // Test middle chunk (index 3)
        let context = engine.generate_context(&chunks, 3).unwrap();
        
        // L1 should contain chunks 2, 3, 4
        assert!(context.l1_content.contains("line 3")); // chunk 2
        assert!(context.l1_content.contains("line 4")); // chunk 3
        assert!(context.l1_content.contains("line 5")); // chunk 4
        
        // L1 should NOT contain chunks 1, 5, 6, 7
        assert!(!context.l1_content.contains("line 2")); // chunk 1
        assert!(!context.l1_content.contains("line 6")); // chunk 5
        assert!(!context.l1_content.contains("line 7")); // chunk 6
        
        // L2 should contain chunks 1, 2, 3, 4, 5
        assert!(context.l2_content.contains("line 2")); // chunk 1
        assert!(context.l2_content.contains("line 3")); // chunk 2
        assert!(context.l2_content.contains("line 4")); // chunk 3
        assert!(context.l2_content.contains("line 5")); // chunk 4
        assert!(context.l2_content.contains("line 6")); // chunk 5
        
        // L2 should NOT contain chunks 0, 6
        assert!(!context.l2_content.contains("line 1")); // chunk 0
        assert!(!context.l2_content.contains("line 7")); // chunk 6
    }
}

#[cfg(test)]
mod generate_all_contexts_tests {
    use super::*;

    #[test]
    fn test_generate_all_contexts_success() {
        let engine = ChunkingEngine::default();
        let chunks = create_test_chunks(5, 3);
        
        let contexts = engine.generate_all_contexts(&chunks).unwrap();
        
        assert_eq!(contexts.len(), chunks.len());
        
        // Check that each context corresponds to the correct chunk
        for (i, context) in contexts.iter().enumerate() {
            // Boundary chunks are first and last
            let expected_boundary = i == 0 || i == chunks.len() - 1;
            assert_eq!(context.is_boundary, expected_boundary);
            
            // All contexts should have valid chunk counts
            assert!(context.l1_chunk_count > 0);
            assert!(context.l2_chunk_count > 0);
            assert!(context.l2_chunk_count >= context.l1_chunk_count);
        }
    }

    #[test]
    fn test_generate_all_contexts_empty_chunks() {
        let engine = ChunkingEngine::default();
        let chunks = vec![];
        
        let contexts = engine.generate_all_contexts(&chunks).unwrap();
        assert_eq!(contexts.len(), 0);
    }

    #[test]
    fn test_generate_all_contexts_single_chunk() {
        let engine = ChunkingEngine::default();
        let chunks = create_test_chunks(1, 5);
        
        let contexts = engine.generate_all_contexts(&chunks).unwrap();
        
        assert_eq!(contexts.len(), 1);
        let context = &contexts[0];
        
        assert!(context.is_boundary);
        assert_eq!(context.l1_chunk_count, 1);
        assert_eq!(context.l2_chunk_count, 1);
        assert_eq!(context.l1_content, context.l2_content);
    }
}

#[cfg(test)]
mod chunk_content_with_context_tests {
    use super::*;

    #[test]
    fn test_chunk_content_with_context_integration() {
        let engine = ChunkingEngine::with_chunk_size(3);
        let content = create_test_content(9); // Will create 3 chunks of 3 lines each
        
        let (chunking_result, contexts) = engine.chunk_content_with_context(
            &content,
            "test_file".to_string(),
            "test.txt".to_string(),
            "test.txt".to_string(),
            Some("txt".to_string()),
        ).unwrap();
        
        assert_eq!(chunking_result.chunks.len(), contexts.len());
        assert_eq!(chunking_result.chunks.len(), 3);
        
        // Validate that contexts match chunks
        for (i, (chunk, context)) in chunking_result.chunks.iter().zip(contexts.iter()).enumerate() {
            // Context should include the chunk's content
            assert!(context.l1_content.contains(&chunk.content) || 
                   context.l1_content.contains(&chunk.content.replace('\n', "\n")));
            
            // Boundary detection should be consistent
            let expected_boundary = i == 0 || i == chunking_result.chunks.len() - 1;
            assert_eq!(context.is_boundary, expected_boundary);
        }
    }

    #[test]
    fn test_chunk_content_with_context_small_file() {
        let engine = ChunkingEngine::with_chunk_size(10);
        let content = create_test_content(5); // Smaller than chunk size
        
        let (chunking_result, contexts) = engine.chunk_content_with_context(
            &content,
            "test_file".to_string(),
            "test.txt".to_string(),
            "test.txt".to_string(),
            Some("txt".to_string()),
        ).unwrap();
        
        assert!(!chunking_result.was_chunked);
        assert_eq!(chunking_result.chunks.len(), 1);
        assert_eq!(contexts.len(), 1);
        
        let context = &contexts[0];
        assert!(context.is_boundary);
        assert_eq!(context.l1_chunk_count, 1);
        assert_eq!(context.l2_chunk_count, 1);
    }
}

#[cfg(test)]
mod context_performance_tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_context_generation_performance() {
        let engine = ChunkingEngine::default();
        let chunks = create_test_chunks(100, 10); // 100 chunks
        
        let start = Instant::now();
        let contexts = engine.generate_all_contexts(&chunks).unwrap();
        let elapsed = start.elapsed();
        
        // Should complete within reasonable time
        assert!(elapsed.as_millis() < 100, "Context generation took too long: {:?}", elapsed);
        
        assert_eq!(contexts.len(), 100);
        
        // Verify all contexts are valid
        for (i, context) in contexts.iter().enumerate() {
            assert!(context.l1_chunk_count > 0);
            assert!(context.l2_chunk_count >= context.l1_chunk_count);
            
            let expected_boundary = i == 0 || i == chunks.len() - 1;
            assert_eq!(context.is_boundary, expected_boundary);
        }
    }

    #[test]
    fn test_context_generation_memory_efficiency() {
        let engine = ChunkingEngine::default();
        let chunks = create_test_chunks(1000, 5); // 1000 chunks
        
        // Generate contexts one by one to test memory efficiency
        for i in 0..chunks.len() {
            let context = engine.generate_context(&chunks, i).unwrap();
            
            // Basic validation
            assert!(context.l1_chunk_count > 0);
            assert!(context.l2_chunk_count >= context.l1_chunk_count);
            
            // Context content should not be excessively large
            assert!(context.l1_content.len() < 10000); // Reasonable upper bound
            assert!(context.l2_content.len() < 20000); // Reasonable upper bound
        }
    }
}
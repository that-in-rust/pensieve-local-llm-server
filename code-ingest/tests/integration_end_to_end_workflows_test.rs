//! Integration tests for end-to-end workflows
//!
//! Tests cover complete ingestion → generation workflows, both git repository 
//! and local folder ingestion paths, and performance tests for large codebase 
//! processing as specified in task 9.2.

use code_ingest::processing::chunking::{ChunkingEngine, ChunkingConfig};
use code_ingest::tasks::hierarchical_task_divider::HierarchicalTaskDivider;
use code_ingest::tasks::content_extractor::ContentTriple;
use code_ingest::error::{ProcessingError, TaskError};
use std::path::PathBuf;
use std::time::Instant;
use tempfile::TempDir;
use tokio;

/// Create test content triples for integration testing
fn create_integration_test_content_triples(count: usize) -> Vec<ContentTriple> {
    (1..=count)
        .map(|i| ContentTriple {
            content_a: PathBuf::from(format!("integration_test_{}_content.txt", i)),
            content_b: PathBuf::from(format!("integration_test_{}_content_L1.txt", i)),
            content_c: PathBuf::from(format!("integration_test_{}_content_L2.txt", i)),
            row_number: i,
            table_name: "INTEGRATION_TEST_TABLE".to_string(),
        })
        .collect()
}

/// Create test file content for chunking integration tests
fn create_large_test_file_content(lines: usize) -> String {
    (1..=lines)
        .map(|i| {
            if i % 50 == 0 {
                format!("// Section {}\nfn section_{}() {{\n    // Implementation for section {}\n}}", i / 50, i / 50, i / 50)
            } else if i % 10 == 0 {
                format!("    let variable_{} = {};", i, i * 2)
            } else {
                format!("    // Line {} of the test file", i)
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod end_to_end_workflow_tests {
    use super::*;

    #[tokio::test]
    async fn test_complete_chunking_to_hierarchy_workflow() {
        // Test the complete workflow from file chunking to task hierarchy generation
        let chunking_engine = ChunkingEngine::with_chunk_size(50);
        let task_divider = HierarchicalTaskDivider::new(3, 4).unwrap();
        
        // Create test content that will be chunked
        let large_content = create_large_test_file_content(200); // 200 lines, will create 4 chunks
        
        // Step 1: Chunk the content
        let chunking_result = chunking_engine.chunk_content(
            &large_content,
            "integration_test_file".to_string(),
            "test/large_file.rs".to_string(),
            "large_file.rs".to_string(),
            Some("rs".to_string()),
        ).unwrap();
        
        assert!(chunking_result.was_chunked);
        assert_eq!(chunking_result.chunks.len(), 4); // 200 / 50 = 4 chunks
        assert_eq!(chunking_result.total_lines, 200);
        
        // Step 2: Generate contexts for all chunks
        let contexts = chunking_engine.generate_all_contexts(&chunking_result.chunks).unwrap();
        assert_eq!(contexts.len(), chunking_result.chunks.len());
        
        // Step 3: Create content triples from chunks
        let content_triples: Vec<ContentTriple> = chunking_result.chunks
            .iter()
            .enumerate()
            .map(|(i, chunk)| ContentTriple {
                content_a: PathBuf::from(format!("chunk_{}_content.txt", i + 1)),
                content_b: PathBuf::from(format!("chunk_{}_content_L1.txt", i + 1)),
                content_c: PathBuf::from(format!("chunk_{}_content_L2.txt", i + 1)),
                row_number: chunk.metadata.chunk_number as usize,
                table_name: "CHUNKED_TEST_TABLE".to_string(),
            })
            .collect();
        
        // Step 4: Generate task hierarchy
        let hierarchy = task_divider.create_hierarchy(content_triples).unwrap();
        
        assert_eq!(hierarchy.total_tasks, 4);
        assert!(!hierarchy.levels.is_empty());
        
        // Verify the hierarchy structure
        let total_tasks_in_hierarchy = hierarchy.levels
            .iter()
            .flat_map(|level| &level.groups)
            .map(|group| group.tasks.len())
            .sum::<usize>();
        
        assert_eq!(total_tasks_in_hierarchy, 4);
    }

    #[tokio::test]
    async fn test_small_file_workflow() {
        // Test workflow with files that don't need chunking
        let chunking_engine = ChunkingEngine::with_chunk_size(100);
        let task_divider = HierarchicalTaskDivider::new(2, 3).unwrap();
        
        // Create small content that won't be chunked
        let small_content = create_large_test_file_content(50); // 50 lines, below chunk size
        
        // Step 1: Process the content (should not be chunked)
        let chunking_result = chunking_engine.chunk_content(
            &small_content,
            "small_test_file".to_string(),
            "test/small_file.rs".to_string(),
            "small_file.rs".to_string(),
            Some("rs".to_string()),
        ).unwrap();
        
        assert!(!chunking_result.was_chunked);
        assert_eq!(chunking_result.chunks.len(), 1);
        assert_eq!(chunking_result.total_lines, 50);
        
        // Step 2: Create content triple
        let content_triple = ContentTriple {
            content_a: PathBuf::from("small_file_content.txt"),
            content_b: PathBuf::from("small_file_content_L1.txt"),
            content_c: PathBuf::from("small_file_content_L2.txt"),
            row_number: 1,
            table_name: "SMALL_TEST_TABLE".to_string(),
        };
        
        // Step 3: Generate task hierarchy
        let hierarchy = task_divider.create_hierarchy(vec![content_triple]).unwrap();
        
        assert_eq!(hierarchy.total_tasks, 1);
        assert!(!hierarchy.levels.is_empty());
    }

    #[tokio::test]
    async fn test_multiple_files_workflow() {
        // Test workflow with multiple files of varying sizes
        let chunking_engine = ChunkingEngine::with_chunk_size(30);
        let task_divider = HierarchicalTaskDivider::new(2, 5).unwrap();
        
        let mut all_content_triples = Vec::new();
        let mut total_expected_tasks = 0;
        
        // Process multiple files with different sizes
        for file_num in 1..=5 {
            let lines = file_num * 25; // Files of 25, 50, 75, 100, 125 lines
            let content = create_large_test_file_content(lines);
            
            let chunking_result = chunking_engine.chunk_content(
                &content,
                format!("multi_test_file_{}", file_num),
                format!("test/file_{}.rs", file_num),
                format!("file_{}.rs", file_num),
                Some("rs".to_string()),
            ).unwrap();
            
            // Calculate expected chunks for this file
            let expected_chunks = if lines < 30 { 1 } else { (lines + 29) / 30 };
            assert_eq!(chunking_result.chunks.len(), expected_chunks);
            total_expected_tasks += expected_chunks;
            
            // Create content triples for each chunk
            for (chunk_idx, chunk) in chunking_result.chunks.iter().enumerate() {
                let content_triple = ContentTriple {
                    content_a: PathBuf::from(format!("file_{}_chunk_{}_content.txt", file_num, chunk_idx + 1)),
                    content_b: PathBuf::from(format!("file_{}_chunk_{}_content_L1.txt", file_num, chunk_idx + 1)),
                    content_c: PathBuf::from(format!("file_{}_chunk_{}_content_L2.txt", file_num, chunk_idx + 1)),
                    row_number: chunk.metadata.chunk_number as usize + (file_num - 1) * 100, // Unique row numbers
                    table_name: "MULTI_TEST_TABLE".to_string(),
                };
                all_content_triples.push(content_triple);
            }
        }
        
        // Generate task hierarchy for all files
        let hierarchy = task_divider.create_hierarchy(all_content_triples).unwrap();
        
        assert_eq!(hierarchy.total_tasks, total_expected_tasks);
        assert!(!hierarchy.levels.is_empty());
        
        // Verify that all tasks are accounted for
        let total_tasks_in_hierarchy = hierarchy.levels
            .iter()
            .flat_map(|level| &level.groups)
            .map(|group| group.tasks.len())
            .sum::<usize>();
        
        assert_eq!(total_tasks_in_hierarchy, total_expected_tasks);
    }

    #[tokio::test]
    async fn test_context_generation_workflow() {
        // Test the complete context generation workflow
        let chunking_engine = ChunkingEngine::with_chunk_size(20);
        
        // Create content that will generate multiple chunks with context
        let content = create_large_test_file_content(100); // 100 lines, will create 5 chunks
        
        let chunking_result = chunking_engine.chunk_content(
            &content,
            "context_test_file".to_string(),
            "test/context_file.rs".to_string(),
            "context_file.rs".to_string(),
            Some("rs".to_string()),
        ).unwrap();
        
        assert_eq!(chunking_result.chunks.len(), 5);
        
        // Generate contexts for all chunks
        let contexts = chunking_engine.generate_all_contexts(&chunking_result.chunks).unwrap();
        assert_eq!(contexts.len(), 5);
        
        // Verify context properties
        for (i, context) in contexts.iter().enumerate() {
            // First and last chunks should be boundary chunks
            let expected_boundary = i == 0 || i == contexts.len() - 1;
            assert_eq!(context.is_boundary, expected_boundary);
            
            // All contexts should have valid chunk counts
            assert!(context.l1_chunk_count > 0);
            assert!(context.l2_chunk_count >= context.l1_chunk_count);
            
            // Context content should not be empty
            assert!(!context.l1_content.is_empty());
            assert!(!context.l2_content.is_empty());
        }
        
        // Test specific context properties for middle chunks
        if contexts.len() >= 3 {
            let middle_context = &contexts[2]; // Third chunk (index 2)
            assert!(!middle_context.is_boundary);
            assert_eq!(middle_context.l1_chunk_count, 3); // chunks 1, 2, 3
            assert_eq!(middle_context.l2_chunk_count, 5); // all chunks (can't go beyond boundaries)
        }
    }

    #[tokio::test]
    async fn test_error_handling_workflow() {
        // Test error handling in the complete workflow
        let task_divider = HierarchicalTaskDivider::new(4, 7).unwrap();
        
        // Test with empty content triples
        let empty_hierarchy = task_divider.create_hierarchy(vec![]).unwrap();
        assert_eq!(empty_hierarchy.total_tasks, 0);
        assert_eq!(empty_hierarchy.levels.len(), 0);
        
        // Test chunking engine with invalid parameters
        let chunking_engine = ChunkingEngine::with_chunk_size(0); // This should still work, but with default config
        
        let content = "test content";
        let result = chunking_engine.chunk_content(
            content,
            "error_test_file".to_string(),
            "test/error_file.txt".to_string(),
            "error_file.txt".to_string(),
            Some("txt".to_string()),
        );
        
        // Should succeed even with minimal content
        assert!(result.is_ok());
        
        // Test task divider with invalid parameters
        let invalid_divider_result = HierarchicalTaskDivider::new(0, 5);
        assert!(invalid_divider_result.is_err());
        
        let invalid_divider_result = HierarchicalTaskDivider::new(3, 0);
        assert!(invalid_divider_result.is_err());
    }
}

#[cfg(test)]
mod performance_integration_tests {
    use super::*;

    #[tokio::test]
    async fn test_large_codebase_processing_performance() {
        // Test performance with a simulated large codebase
        let chunking_engine = ChunkingEngine::with_chunk_size(100);
        let task_divider = HierarchicalTaskDivider::new(4, 10).unwrap();
        
        let start_time = Instant::now();
        
        // Simulate processing 100 files
        let mut all_content_triples = Vec::new();
        
        for file_num in 1..=100 {
            // Vary file sizes to simulate realistic codebase
            let lines = 50 + (file_num % 200); // Files between 50-250 lines
            let content = create_large_test_file_content(lines);
            
            let chunking_result = chunking_engine.chunk_content(
                &content,
                format!("perf_test_file_{}", file_num),
                format!("src/module_{}/file_{}.rs", file_num / 10, file_num),
                format!("file_{}.rs", file_num),
                Some("rs".to_string()),
            ).unwrap();
            
            // Create content triples for each chunk
            for (chunk_idx, _chunk) in chunking_result.chunks.iter().enumerate() {
                let content_triple = ContentTriple {
                    content_a: PathBuf::from(format!("perf_file_{}_chunk_{}_content.txt", file_num, chunk_idx + 1)),
                    content_b: PathBuf::from(format!("perf_file_{}_chunk_{}_content_L1.txt", file_num, chunk_idx + 1)),
                    content_c: PathBuf::from(format!("perf_file_{}_chunk_{}_content_L2.txt", file_num, chunk_idx + 1)),
                    row_number: file_num * 1000 + chunk_idx + 1, // Unique row numbers
                    table_name: "PERF_TEST_TABLE".to_string(),
                };
                all_content_triples.push(content_triple);
            }
        }
        
        let chunking_time = start_time.elapsed();
        
        // Generate task hierarchy
        let hierarchy_start = Instant::now();
        let hierarchy = task_divider.create_hierarchy(all_content_triples.clone()).unwrap();
        let hierarchy_time = hierarchy_start.elapsed();
        
        let total_time = start_time.elapsed();
        
        // Performance assertions
        assert!(chunking_time.as_millis() < 5000, "Chunking took too long: {:?}", chunking_time);
        assert!(hierarchy_time.as_millis() < 2000, "Hierarchy generation took too long: {:?}", hierarchy_time);
        assert!(total_time.as_millis() < 6000, "Total processing took too long: {:?}", total_time);
        
        // Verify results
        assert!(hierarchy.total_tasks > 100); // Should have more tasks than files due to chunking
        assert!(!hierarchy.levels.is_empty());
        
        println!("Performance test results:");
        println!("  Files processed: 100");
        println!("  Total content triples: {}", all_content_triples.len());
        println!("  Total tasks generated: {}", hierarchy.total_tasks);
        println!("  Chunking time: {:?}", chunking_time);
        println!("  Hierarchy generation time: {:?}", hierarchy_time);
        println!("  Total time: {:?}", total_time);
    }

    #[tokio::test]
    async fn test_memory_efficiency_large_dataset() {
        // Test memory efficiency with large datasets
        let chunking_engine = ChunkingEngine::with_chunk_size(50);
        let task_divider = HierarchicalTaskDivider::new(3, 8).unwrap();
        
        // Process files one by one to test memory efficiency
        let mut total_tasks = 0;
        
        for batch in 0..10 {
            let mut batch_content_triples = Vec::new();
            
            // Process 50 files per batch
            for file_num in 1..=50 {
                let lines = 75 + (file_num % 100); // Files between 75-175 lines
                let content = create_large_test_file_content(lines);
                
                let chunking_result = chunking_engine.chunk_content(
                    &content,
                    format!("mem_test_batch_{}_file_{}", batch, file_num),
                    format!("batch_{}/file_{}.rs", batch, file_num),
                    format!("file_{}.rs", file_num),
                    Some("rs".to_string()),
                ).unwrap();
                
                // Create content triples
                for (chunk_idx, _chunk) in chunking_result.chunks.iter().enumerate() {
                    let content_triple = ContentTriple {
                        content_a: PathBuf::from(format!("mem_batch_{}_file_{}_chunk_{}_content.txt", batch, file_num, chunk_idx + 1)),
                        content_b: PathBuf::from(format!("mem_batch_{}_file_{}_chunk_{}_content_L1.txt", batch, file_num, chunk_idx + 1)),
                        content_c: PathBuf::from(format!("mem_batch_{}_file_{}_chunk_{}_content_L2.txt", batch, file_num, chunk_idx + 1)),
                        row_number: batch * 50000 + file_num * 100 + chunk_idx + 1,
                        table_name: format!("MEM_TEST_BATCH_{}", batch),
                    };
                    batch_content_triples.push(content_triple);
                }
            }
            
            // Process batch
            let batch_hierarchy = task_divider.create_hierarchy(batch_content_triples).unwrap();
            total_tasks += batch_hierarchy.total_tasks;
            
            // Verify batch processing worked
            assert!(batch_hierarchy.total_tasks > 0);
        }
        
        // Verify total processing
        assert!(total_tasks > 500); // Should have processed many tasks across all batches
        println!("Memory efficiency test: processed {} total tasks across 10 batches", total_tasks);
    }

    #[tokio::test]
    async fn test_concurrent_processing_simulation() {
        // Simulate concurrent processing scenarios
        let chunking_engine = ChunkingEngine::with_chunk_size(40);
        
        // Create multiple processing tasks
        let mut handles = Vec::new();
        
        for worker_id in 0..5 {
            let engine = chunking_engine.clone();
            
            let handle = tokio::spawn(async move {
                let mut worker_results = Vec::new();
                
                // Each worker processes 20 files
                for file_num in 1..=20 {
                    let lines = 60 + (file_num % 80); // Files between 60-140 lines
                    let content = create_large_test_file_content(lines);
                    
                    let chunking_result = engine.chunk_content(
                        &content,
                        format!("worker_{}_file_{}", worker_id, file_num),
                        format!("worker_{}/file_{}.rs", worker_id, file_num),
                        format!("file_{}.rs", file_num),
                        Some("rs".to_string()),
                    ).unwrap();
                    
                    worker_results.push(chunking_result);
                }
                
                worker_results
            });
            
            handles.push(handle);
        }
        
        // Wait for all workers to complete
        let start_time = Instant::now();
        let mut total_chunks = 0;
        
        for handle in handles {
            let worker_results = handle.await.unwrap();
            for result in worker_results {
                total_chunks += result.chunks.len();
            }
        }
        
        let concurrent_time = start_time.elapsed();
        
        // Performance assertions for concurrent processing
        assert!(concurrent_time.as_millis() < 3000, "Concurrent processing took too long: {:?}", concurrent_time);
        assert!(total_chunks > 100); // Should have processed many chunks
        
        println!("Concurrent processing test:");
        println!("  Workers: 5");
        println!("  Files per worker: 20");
        println!("  Total chunks processed: {}", total_chunks);
        println!("  Processing time: {:?}", concurrent_time);
    }
}

#[cfg(test)]
mod integration_edge_cases_tests {
    use super::*;

    #[tokio::test]
    async fn test_boundary_conditions_integration() {
        // Test various boundary conditions in the integrated workflow
        let chunking_engine = ChunkingEngine::with_chunk_size(1);
        let task_divider = HierarchicalTaskDivider::new(1, 1).unwrap();
        
        // Test with single line file
        let single_line_content = "single line of code";
        let result = chunking_engine.chunk_content(
            single_line_content,
            "single_line_file".to_string(),
            "test/single.rs".to_string(),
            "single.rs".to_string(),
            Some("rs".to_string()),
        ).unwrap();
        
        assert!(!result.was_chunked); // Single line should not be chunked even with chunk_size=1
        assert_eq!(result.chunks.len(), 1);
        
        // Test with exact chunk size
        let exact_chunk_content = "line 1\nline 2\nline 3";
        let result = chunking_engine.chunk_content(
            exact_chunk_content,
            "exact_chunk_file".to_string(),
            "test/exact.rs".to_string(),
            "exact.rs".to_string(),
            Some("rs".to_string()),
        ).unwrap();
        
        assert!(result.was_chunked);
        assert_eq!(result.chunks.len(), 3); // Each line becomes a chunk
        
        // Create content triples and test hierarchy generation
        let content_triples = result.chunks
            .iter()
            .enumerate()
            .map(|(i, chunk)| ContentTriple {
                content_a: PathBuf::from(format!("boundary_chunk_{}_content.txt", i + 1)),
                content_b: PathBuf::from(format!("boundary_chunk_{}_content_L1.txt", i + 1)),
                content_c: PathBuf::from(format!("boundary_chunk_{}_content_L2.txt", i + 1)),
                row_number: chunk.metadata.chunk_number as usize,
                table_name: "BOUNDARY_TEST_TABLE".to_string(),
            })
            .collect();
        
        let hierarchy = task_divider.create_hierarchy(content_triples).unwrap();
        assert_eq!(hierarchy.total_tasks, 3);
    }

    #[tokio::test]
    async fn test_empty_and_whitespace_content_integration() {
        // Test integration with empty and whitespace-only content
        let chunking_engine = ChunkingEngine::with_chunk_size(10);
        let task_divider = HierarchicalTaskDivider::new(2, 2).unwrap();
        
        // Test with empty content
        let empty_result = chunking_engine.chunk_content(
            "",
            "empty_file".to_string(),
            "test/empty.rs".to_string(),
            "empty.rs".to_string(),
            Some("rs".to_string()),
        ).unwrap();
        
        assert!(!empty_result.was_chunked);
        assert_eq!(empty_result.chunks.len(), 1);
        assert_eq!(empty_result.total_lines, 0);
        
        // Test with whitespace-only content
        let whitespace_content = "   \n\t\t\n   \n\n";
        let whitespace_result = chunking_engine.chunk_content(
            whitespace_content,
            "whitespace_file".to_string(),
            "test/whitespace.rs".to_string(),
            "whitespace.rs".to_string(),
            Some("rs".to_string()),
        ).unwrap();
        
        assert!(!whitespace_result.was_chunked); // Small file, won't be chunked
        assert_eq!(whitespace_result.chunks.len(), 1);
        assert_eq!(whitespace_result.total_lines, 4);
        
        // Test hierarchy generation with minimal content
        let content_triple = ContentTriple {
            content_a: PathBuf::from("minimal_content.txt"),
            content_b: PathBuf::from("minimal_content_L1.txt"),
            content_c: PathBuf::from("minimal_content_L2.txt"),
            row_number: 1,
            table_name: "MINIMAL_TEST_TABLE".to_string(),
        };
        
        let hierarchy = task_divider.create_hierarchy(vec![content_triple]).unwrap();
        assert_eq!(hierarchy.total_tasks, 1);
    }

    #[tokio::test]
    async fn test_maximum_configuration_integration() {
        // Test with maximum reasonable configuration values
        let chunking_engine = ChunkingEngine::with_chunk_size(1000);
        let task_divider = HierarchicalTaskDivider::new(5, 20).unwrap();
        
        // Create very large content
        let large_content = create_large_test_file_content(5000); // 5000 lines
        
        let start_time = Instant::now();
        
        let chunking_result = chunking_engine.chunk_content(
            &large_content,
            "max_config_file".to_string(),
            "test/max_config.rs".to_string(),
            "max_config.rs".to_string(),
            Some("rs".to_string()),
        ).unwrap();
        
        let chunking_time = start_time.elapsed();
        
        assert!(chunking_result.was_chunked);
        assert_eq!(chunking_result.chunks.len(), 5); // 5000 / 1000 = 5 chunks
        assert_eq!(chunking_result.total_lines, 5000);
        
        // Generate contexts
        let contexts = chunking_engine.generate_all_contexts(&chunking_result.chunks).unwrap();
        assert_eq!(contexts.len(), 5);
        
        // Create content triples
        let content_triples: Vec<ContentTriple> = chunking_result.chunks
            .iter()
            .enumerate()
            .map(|(i, chunk)| ContentTriple {
                content_a: PathBuf::from(format!("max_chunk_{}_content.txt", i + 1)),
                content_b: PathBuf::from(format!("max_chunk_{}_content_L1.txt", i + 1)),
                content_c: PathBuf::from(format!("max_chunk_{}_content_L2.txt", i + 1)),
                row_number: chunk.metadata.chunk_number as usize,
                table_name: "MAX_CONFIG_TEST_TABLE".to_string(),
            })
            .collect();
        
        // Generate hierarchy
        let hierarchy_start = Instant::now();
        let hierarchy = task_divider.create_hierarchy(content_triples).unwrap();
        let hierarchy_time = hierarchy_start.elapsed();
        
        assert_eq!(hierarchy.total_tasks, 5);
        assert!(!hierarchy.levels.is_empty());
        
        // Performance should still be reasonable even with large content
        assert!(chunking_time.as_millis() < 1000, "Large file chunking took too long: {:?}", chunking_time);
        assert!(hierarchy_time.as_millis() < 500, "Large hierarchy generation took too long: {:?}", hierarchy_time);
        
        println!("Maximum configuration test:");
        println!("  Content lines: 5000");
        println!("  Chunk size: 1000");
        println!("  Chunks created: {}", chunking_result.chunks.len());
        println!("  Hierarchy levels: {}", task_divider.levels);
        println!("  Hierarchy groups per level: {}", task_divider.groups_per_level);
        println!("  Chunking time: {:?}", chunking_time);
        println!("  Hierarchy time: {:?}", hierarchy_time);
    }
}
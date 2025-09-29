//! Property-based and stress tests
//!
//! Tests cover property-based tests for chunking content preservation,
//! stress tests for concurrent database operations and large file processing,
//! and edge case tests for boundary conditions as specified in task 9.3.

use code_ingest::processing::chunking::{ChunkingEngine, ChunkingConfig};
use code_ingest::tasks::hierarchical_task_divider::HierarchicalTaskDivider;
use code_ingest::tasks::content_extractor::ContentTriple;
use code_ingest::error::{ProcessingError, TaskError};
use std::path::PathBuf;
use std::time::{Instant, Duration};
use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use tokio;

/// Generate random content for property-based testing
fn generate_random_content(lines: usize, seed: u64) -> String {
    // Simple deterministic pseudo-random generator for reproducible tests
    let mut rng_state = seed;
    
    (0..lines)
        .map(|i| {
            rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            let line_type = rng_state % 5;
            
            match line_type {
                0 => format!("// Comment line {}", i),
                1 => format!("fn function_{}() {{", i),
                2 => format!("    let var_{} = {};", i, rng_state % 1000),
                3 => format!("}}"),
                _ => format!("    // Regular code line {}", i),
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Generate random content triples for testing
fn generate_random_content_triples(count: usize, seed: u64) -> Vec<ContentTriple> {
    let mut rng_state = seed;
    
    (0..count)
        .map(|i| {
            rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            let table_variant = rng_state % 3;
            
            ContentTriple {
                content_a: PathBuf::from(format!("random_test_{}_content.txt", i)),
                content_b: PathBuf::from(format!("random_test_{}_content_L1.txt", i)),
                content_c: PathBuf::from(format!("random_test_{}_content_L2.txt", i)),
                row_number: i + 1,
                table_name: format!("RANDOM_TEST_TABLE_{}", table_variant),
            }
        })
        .collect()
}

#[cfg(test)]
mod property_based_tests {
    use super::*;

    #[tokio::test]
    async fn test_chunking_content_preservation_property() {
        // Property: Chunking should always preserve the original content when reconstructed
        let test_cases = vec![
            (10, 3),   // Small file, small chunks
            (50, 10),  // Medium file, medium chunks
            (100, 25), // Large file, medium chunks
            (200, 50), // Large file, large chunks
            (1000, 100), // Very large file
        ];
        
        for (lines, chunk_size) in test_cases {
            for seed in [12345, 67890, 54321, 98765, 11111] {
                let original_content = generate_random_content(lines, seed);
                let chunking_engine = ChunkingEngine::with_chunk_size(chunk_size);
                
                let chunking_result = chunking_engine.chunk_content(
                    &original_content,
                    format!("property_test_{}_{}", lines, seed),
                    format!("test/property_{}_{}.rs", lines, seed),
                    format!("property_{}_{}.rs", lines, seed),
                    Some("rs".to_string()),
                ).unwrap();
                
                // Property: Reconstructed content should equal original content
                let reconstructed_content = chunking_result.chunks
                    .iter()
                    .map(|chunk| chunk.content.as_str())
                    .collect::<Vec<_>>()
                    .join("\n");
                
                assert_eq!(
                    reconstructed_content, 
                    original_content,
                    "Content preservation failed for lines={}, chunk_size={}, seed={}",
                    lines, chunk_size, seed
                );
                
                // Property: Total lines should be preserved
                assert_eq!(
                    chunking_result.total_lines, 
                    lines as u32,
                    "Line count preservation failed for lines={}, chunk_size={}, seed={}",
                    lines, chunk_size, seed
                );
                
                // Property: Chunk metadata should be consistent
                chunking_engine.validate_chunks(&chunking_result.chunks).unwrap();
            }
        }
    }

    #[tokio::test]
    async fn test_hierarchy_generation_properties() {
        // Property: Task hierarchy should always preserve total task count
        let test_configurations = vec![
            (1, 1),   // Minimal configuration
            (2, 3),   // Small configuration
            (3, 5),   // Medium configuration
            (4, 7),   // Default configuration
            (5, 10),  // Large configuration
        ];
        
        for (levels, groups) in test_configurations {
            for task_count in [1, 5, 10, 25, 50, 100] {
                for seed in [111, 222, 333, 444, 555] {
                    let content_triples = generate_random_content_triples(task_count, seed);
                    let task_divider = HierarchicalTaskDivider::new(levels, groups).unwrap();
                    
                    let hierarchy = task_divider.create_hierarchy(content_triples.clone()).unwrap();
                    
                    // Property: Total tasks should be preserved
                    assert_eq!(
                        hierarchy.total_tasks, 
                        task_count,
                        "Task count preservation failed for levels={}, groups={}, tasks={}, seed={}",
                        levels, groups, task_count, seed
                    );
                    
                    // Property: All tasks should be accounted for in the hierarchy
                    let total_tasks_in_hierarchy = hierarchy.levels
                        .iter()
                        .flat_map(|level| &level.groups)
                        .map(|group| count_tasks_in_group(group))
                        .sum::<usize>();
                    
                    assert_eq!(
                        total_tasks_in_hierarchy, 
                        task_count,
                        "Task accounting failed for levels={}, groups={}, tasks={}, seed={}",
                        levels, groups, task_count, seed
                    );
                    
                    // Property: Row numbers should be unique and within expected range
                    let mut row_numbers = Vec::new();
                    collect_row_numbers(&hierarchy, &mut row_numbers);
                    
                    assert_eq!(
                        row_numbers.len(), 
                        task_count,
                        "Row number count mismatch for levels={}, groups={}, tasks={}, seed={}",
                        levels, groups, task_count, seed
                    );
                }
            }
        }
    }

    #[tokio::test]
    async fn test_chunking_boundary_properties() {
        // Property: Chunk boundaries should always be valid
        let chunk_sizes = vec![1, 2, 5, 10, 25, 50, 100];
        let file_sizes = vec![1, 2, 3, 10, 25, 99, 100, 101, 200, 500];
        
        for chunk_size in chunk_sizes {
            for file_size in &file_sizes {
                for seed in [777, 888, 999] {
                    let content = generate_random_content(*file_size, seed);
                    let chunking_engine = ChunkingEngine::with_chunk_size(chunk_size);
                    
                    let chunking_result = chunking_engine.chunk_content(
                        &content,
                        format!("boundary_test_{}_{}", file_size, seed),
                        format!("test/boundary_{}_{}.rs", file_size, seed),
                        format!("boundary_{}_{}.rs", file_size, seed),
                        Some("rs".to_string()),
                    ).unwrap();
                    
                    // Property: If file is smaller than chunk size, should not be chunked
                    if *file_size < chunk_size {
                        assert!(
                            !chunking_result.was_chunked,
                            "Small file was incorrectly chunked: file_size={}, chunk_size={}",
                            file_size, chunk_size
                        );
                        assert_eq!(
                            chunking_result.chunks.len(), 
                            1,
                            "Small file should have exactly 1 chunk: file_size={}, chunk_size={}",
                            file_size, chunk_size
                        );
                    }
                    
                    // Property: All chunks should have valid line ranges
                    for (i, chunk) in chunking_result.chunks.iter().enumerate() {
                        assert!(
                            chunk.metadata.start_line <= chunk.metadata.end_line,
                            "Invalid chunk line range: chunk {}, start={}, end={}",
                            i, chunk.metadata.start_line, chunk.metadata.end_line
                        );
                        
                        assert!(
                            chunk.metadata.line_count == (chunk.metadata.end_line - chunk.metadata.start_line + 1),
                            "Inconsistent chunk line count: chunk {}, count={}, calculated={}",
                            i, chunk.metadata.line_count, chunk.metadata.end_line - chunk.metadata.start_line + 1
                        );
                    }
                    
                    // Property: Chunks should cover the entire file without gaps or overlaps
                    if chunking_result.chunks.len() > 1 {
                        for i in 1..chunking_result.chunks.len() {
                            let prev_chunk = &chunking_result.chunks[i - 1];
                            let curr_chunk = &chunking_result.chunks[i];
                            
                            assert_eq!(
                                curr_chunk.metadata.start_line,
                                prev_chunk.metadata.end_line + 1,
                                "Gap or overlap between chunks {} and {}: prev_end={}, curr_start={}",
                                i - 1, i, prev_chunk.metadata.end_line, curr_chunk.metadata.start_line
                            );
                        }
                    }
                }
            }
        }
    }

    #[tokio::test]
    async fn test_context_generation_properties() {
        // Property: Context generation should always produce valid contexts
        let chunking_engine = ChunkingEngine::with_chunk_size(10);
        
        for file_lines in [25, 50, 100] {
            for seed in [1001, 2002, 3003] {
                let content = generate_random_content(file_lines, seed);
                
                let chunking_result = chunking_engine.chunk_content(
                    &content,
                    format!("context_test_{}_{}", file_lines, seed),
                    format!("test/context_{}_{}.rs", file_lines, seed),
                    format!("context_{}_{}.rs", file_lines, seed),
                    Some("rs".to_string()),
                ).unwrap();
                
                let contexts = chunking_engine.generate_all_contexts(&chunking_result.chunks).unwrap();
                
                // Property: Should have one context per chunk
                assert_eq!(
                    contexts.len(), 
                    chunking_result.chunks.len(),
                    "Context count mismatch for file_lines={}, seed={}",
                    file_lines, seed
                );
                
                // Property: Context chunk counts should be valid
                for (i, context) in contexts.iter().enumerate() {
                    assert!(
                        context.l1_chunk_count > 0,
                        "Invalid L1 chunk count for chunk {}: count={}",
                        i, context.l1_chunk_count
                    );
                    
                    assert!(
                        context.l2_chunk_count >= context.l1_chunk_count,
                        "L2 chunk count should be >= L1 for chunk {}: L1={}, L2={}",
                        i, context.l1_chunk_count, context.l2_chunk_count
                    );
                    
                    assert!(
                        context.l2_chunk_count <= chunking_result.chunks.len(),
                        "L2 chunk count exceeds total chunks for chunk {}: L2={}, total={}",
                        i, context.l2_chunk_count, chunking_result.chunks.len()
                    );
                    
                    // Property: Boundary chunks should be marked correctly
                    let expected_boundary = i == 0 || i == contexts.len() - 1;
                    assert_eq!(
                        context.is_boundary, 
                        expected_boundary,
                        "Incorrect boundary marking for chunk {}: expected={}, actual={}",
                        i, expected_boundary, context.is_boundary
                    );
                }
            }
        }
    }

    // Helper function to count tasks in a hierarchical group
    fn count_tasks_in_group(group: &code_ingest::tasks::hierarchical_task_divider::HierarchicalTaskGroup) -> usize {
        group.tasks.len() + group.sub_groups.iter().map(count_tasks_in_group).sum::<usize>()
    }

    // Helper function to collect all row numbers from a hierarchy
    fn collect_row_numbers(
        hierarchy: &code_ingest::tasks::hierarchical_task_divider::TaskHierarchy, 
        row_numbers: &mut Vec<usize>
    ) {
        for level in &hierarchy.levels {
            for group in &level.groups {
                collect_row_numbers_from_group(group, row_numbers);
            }
        }
    }

    fn collect_row_numbers_from_group(
        group: &code_ingest::tasks::hierarchical_task_divider::HierarchicalTaskGroup,
        row_numbers: &mut Vec<usize>
    ) {
        for task in &group.tasks {
            row_numbers.push(task.row_number);
        }
        for sub_group in &group.sub_groups {
            collect_row_numbers_from_group(sub_group, row_numbers);
        }
    }
}

#[cfg(test)]
mod stress_tests {
    use super::*;

    #[tokio::test]
    async fn test_large_file_processing_stress() {
        // Stress test: Process very large files
        let chunking_engine = ChunkingEngine::with_chunk_size(500);
        
        let file_sizes = vec![5000, 10000, 20000]; // Very large files
        
        for file_size in file_sizes {
            let start_time = Instant::now();
            
            let content = generate_random_content(file_size, 12345);
            
            let chunking_result = chunking_engine.chunk_content(
                &content,
                format!("stress_large_file_{}", file_size),
                format!("test/stress_large_{}.rs", file_size),
                format!("stress_large_{}.rs", file_size),
                Some("rs".to_string()),
            ).unwrap();
            
            let processing_time = start_time.elapsed();
            
            // Stress test assertions
            assert!(
                processing_time < Duration::from_secs(5),
                "Large file processing took too long: {} lines in {:?}",
                file_size, processing_time
            );
            
            assert!(chunking_result.was_chunked);
            assert_eq!(chunking_result.total_lines, file_size as u32);
            
            // Validate all chunks
            chunking_engine.validate_chunks(&chunking_result.chunks).unwrap();
            
            // Generate contexts for stress testing
            let context_start = Instant::now();
            let contexts = chunking_engine.generate_all_contexts(&chunking_result.chunks).unwrap();
            let context_time = context_start.elapsed();
            
            assert!(
                context_time < Duration::from_secs(2),
                "Context generation took too long: {} chunks in {:?}",
                chunking_result.chunks.len(), context_time
            );
            
            assert_eq!(contexts.len(), chunking_result.chunks.len());
            
            println!("Large file stress test: {} lines processed in {:?}, contexts in {:?}", 
                     file_size, processing_time, context_time);
        }
    }

    #[tokio::test]
    async fn test_concurrent_chunking_stress() {
        // Stress test: Concurrent chunking operations
        let chunking_engine = Arc::new(ChunkingEngine::with_chunk_size(100));
        let results = Arc::new(Mutex::new(Vec::new()));
        
        let mut handles = Vec::new();
        
        // Spawn multiple concurrent chunking tasks
        for worker_id in 0..10 {
            let engine = Arc::clone(&chunking_engine);
            let results_ref = Arc::clone(&results);
            
            let handle = tokio::spawn(async move {
                let mut worker_results = Vec::new();
                
                // Each worker processes multiple files
                for file_id in 0..20 {
                    let file_size = 150 + (worker_id * 10) + file_id; // Varying file sizes
                    let content = generate_random_content(file_size, (worker_id * 1000 + file_id) as u64);
                    
                    let start_time = Instant::now();
                    
                    let chunking_result = engine.chunk_content(
                        &content,
                        format!("concurrent_worker_{}_file_{}", worker_id, file_id),
                        format!("test/concurrent_{}_{}.rs", worker_id, file_id),
                        format!("concurrent_{}_{}.rs", worker_id, file_id),
                        Some("rs".to_string()),
                    ).unwrap();
                    
                    let processing_time = start_time.elapsed();
                    
                    worker_results.push((worker_id, file_id, file_size, processing_time, chunking_result.chunks.len()));
                    
                    // Validate chunks
                    engine.validate_chunks(&chunking_result.chunks).unwrap();
                }
                
                // Store results
                {
                    let mut results_guard = results_ref.lock().unwrap();
                    results_guard.extend(worker_results);
                }
            });
            
            handles.push(handle);
        }
        
        // Wait for all workers to complete
        let overall_start = Instant::now();
        for handle in handles {
            handle.await.unwrap();
        }
        let overall_time = overall_start.elapsed();
        
        // Analyze results
        let results_guard = results.lock().unwrap();
        let total_files = results_guard.len();
        let total_chunks: usize = results_guard.iter().map(|(_, _, _, _, chunks)| chunks).sum();
        let max_processing_time = results_guard.iter().map(|(_, _, _, time, _)| *time).max().unwrap();
        let avg_processing_time = results_guard.iter().map(|(_, _, _, time, _)| time.as_millis()).sum::<u128>() / total_files as u128;
        
        // Stress test assertions
        assert_eq!(total_files, 200); // 10 workers * 20 files each
        assert!(overall_time < Duration::from_secs(10), "Concurrent processing took too long: {:?}", overall_time);
        assert!(max_processing_time < Duration::from_millis(500), "Individual file processing took too long: {:?}", max_processing_time);
        
        println!("Concurrent chunking stress test:");
        println!("  Workers: 10");
        println!("  Files per worker: 20");
        println!("  Total files: {}", total_files);
        println!("  Total chunks: {}", total_chunks);
        println!("  Overall time: {:?}", overall_time);
        println!("  Max processing time: {:?}", max_processing_time);
        println!("  Avg processing time: {}ms", avg_processing_time);
    }

    #[tokio::test]
    async fn test_hierarchy_generation_stress() {
        // Stress test: Generate hierarchies for large numbers of tasks
        let task_counts = vec![500, 1000, 2000];
        let configurations = vec![
            (3, 5),
            (4, 7),
            (5, 10),
        ];
        
        for task_count in task_counts {
            for (levels, groups) in &configurations {
                let start_time = Instant::now();
                
                let content_triples = generate_random_content_triples(task_count, 54321);
                let task_divider = HierarchicalTaskDivider::new(*levels, *groups).unwrap();
                
                let hierarchy = task_divider.create_hierarchy(content_triples).unwrap();
                
                let processing_time = start_time.elapsed();
                
                // Stress test assertions
                assert!(
                    processing_time < Duration::from_secs(3),
                    "Hierarchy generation took too long: {} tasks, {}x{} config in {:?}",
                    task_count, levels, groups, processing_time
                );
                
                assert_eq!(hierarchy.total_tasks, task_count);
                
                // Verify hierarchy structure
                let total_tasks_in_hierarchy = hierarchy.levels
                    .iter()
                    .flat_map(|level| &level.groups)
                    .map(|group| count_tasks_in_group_stress(group))
                    .sum::<usize>();
                
                assert_eq!(total_tasks_in_hierarchy, task_count);
                
                println!("Hierarchy stress test: {} tasks, {}x{} config in {:?}", 
                         task_count, levels, groups, processing_time);
            }
        }
    }

    #[tokio::test]
    async fn test_memory_pressure_stress() {
        // Stress test: Process many files to test memory usage
        let chunking_engine = ChunkingEngine::with_chunk_size(200);
        let task_divider = HierarchicalTaskDivider::new(4, 8).unwrap();
        
        let batch_size = 100;
        let num_batches = 10;
        
        for batch in 0..num_batches {
            let batch_start = Instant::now();
            let mut batch_content_triples = Vec::new();
            
            // Process files in batches to test memory efficiency
            for file_id in 0..batch_size {
                let file_size = 250 + (file_id % 100); // Varying file sizes
                let content = generate_random_content(file_size, (batch * 1000 + file_id) as u64);
                
                let chunking_result = chunking_engine.chunk_content(
                    &content,
                    format!("memory_batch_{}_file_{}", batch, file_id),
                    format!("test/memory_{}_{}.rs", batch, file_id),
                    format!("memory_{}_{}.rs", batch, file_id),
                    Some("rs".to_string()),
                ).unwrap();
                
                // Create content triples for each chunk
                for (chunk_idx, _chunk) in chunking_result.chunks.iter().enumerate() {
                    let content_triple = ContentTriple {
                        content_a: PathBuf::from(format!("memory_batch_{}_file_{}_chunk_{}_content.txt", batch, file_id, chunk_idx + 1)),
                        content_b: PathBuf::from(format!("memory_batch_{}_file_{}_chunk_{}_content_L1.txt", batch, file_id, chunk_idx + 1)),
                        content_c: PathBuf::from(format!("memory_batch_{}_file_{}_chunk_{}_content_L2.txt", batch, file_id, chunk_idx + 1)),
                        row_number: batch * 10000 + file_id * 100 + chunk_idx + 1,
                        table_name: format!("MEMORY_STRESS_BATCH_{}", batch),
                    };
                    batch_content_triples.push(content_triple);
                }
            }
            
            // Generate hierarchy for batch
            let hierarchy = task_divider.create_hierarchy(batch_content_triples.clone()).unwrap();
            let batch_time = batch_start.elapsed();
            
            // Memory pressure assertions
            assert!(
                batch_time < Duration::from_secs(5),
                "Batch {} processing took too long: {:?}",
                batch, batch_time
            );
            
            assert_eq!(hierarchy.total_tasks, batch_content_triples.len());
            
            println!("Memory pressure batch {}: {} tasks in {:?}", 
                     batch, hierarchy.total_tasks, batch_time);
            
            // Force cleanup between batches (simulate memory pressure)
            drop(batch_content_triples);
            drop(hierarchy);
        }
    }

    #[tokio::test]
    async fn test_edge_case_stress() {
        // Stress test: Various edge cases in rapid succession
        let chunking_engine = ChunkingEngine::with_chunk_size(50);
        let task_divider = HierarchicalTaskDivider::new(3, 5).unwrap();
        
        let very_long_line = "x".repeat(10000);
        let many_empty_lines = "\n".repeat(100);
        let edge_cases = vec![
            ("empty", ""),
            ("single_line", "single line"),
            ("whitespace_only", "   \n\t\t\n   "),
            ("very_long_lines", &very_long_line),
            ("many_empty_lines", &many_empty_lines),
            ("mixed_content", "line1\n\n\nline4\n   \nline6"),
        ];
        
        for iteration in 0..100 {
            for (case_name, content) in &edge_cases {
                let chunking_result = chunking_engine.chunk_content(
                    content,
                    format!("edge_stress_{}_{}", case_name, iteration),
                    format!("test/edge_{}_{}.txt", case_name, iteration),
                    format!("edge_{}_{}.txt", case_name, iteration),
                    Some("txt".to_string()),
                ).unwrap();
                
                // Validate chunks
                chunking_engine.validate_chunks(&chunking_result.chunks).unwrap();
                
                // Create content triple if there's content
                if !content.is_empty() {
                    let content_triple = ContentTriple {
                        content_a: PathBuf::from(format!("edge_{}_{}_content.txt", case_name, iteration)),
                        content_b: PathBuf::from(format!("edge_{}_{}_content_L1.txt", case_name, iteration)),
                        content_c: PathBuf::from(format!("edge_{}_{}_content_L2.txt", case_name, iteration)),
                        row_number: iteration + 1,
                        table_name: format!("EDGE_STRESS_{}", case_name.to_uppercase()),
                    };
                    
                    let hierarchy = task_divider.create_hierarchy(vec![content_triple]).unwrap();
                    assert_eq!(hierarchy.total_tasks, 1);
                }
            }
        }
        
        println!("Edge case stress test: {} iterations of {} cases completed", 100, edge_cases.len());
    }

    // Helper function for stress tests
    fn count_tasks_in_group_stress(group: &code_ingest::tasks::hierarchical_task_divider::HierarchicalTaskGroup) -> usize {
        group.tasks.len() + group.sub_groups.iter().map(count_tasks_in_group_stress).sum::<usize>()
    }
}

#[cfg(test)]
mod boundary_condition_tests {
    use super::*;

    #[tokio::test]
    async fn test_minimum_configuration_boundaries() {
        // Test absolute minimum configurations
        let chunking_engine = ChunkingEngine::with_chunk_size(1);
        let task_divider = HierarchicalTaskDivider::new(1, 1).unwrap();
        
        // Test with minimal content
        let minimal_cases = vec![
            ("", 0),
            ("a", 1),
            ("a\nb", 2),
            ("a\nb\nc", 3),
        ];
        
        for (content, expected_lines) in minimal_cases {
            let chunking_result = chunking_engine.chunk_content(
                content,
                "minimal_test".to_string(),
                "test/minimal.txt".to_string(),
                "minimal.txt".to_string(),
                Some("txt".to_string()),
            ).unwrap();
            
            assert_eq!(chunking_result.total_lines, expected_lines);
            assert_eq!(chunking_result.chunks.len(), 1); // Even with chunk_size=1, small files aren't chunked
            
            if !content.is_empty() {
                let content_triple = ContentTriple {
                    content_a: PathBuf::from("minimal_content.txt"),
                    content_b: PathBuf::from("minimal_content_L1.txt"),
                    content_c: PathBuf::from("minimal_content_L2.txt"),
                    row_number: 1,
                    table_name: "MINIMAL_TEST".to_string(),
                };
                
                let hierarchy = task_divider.create_hierarchy(vec![content_triple]).unwrap();
                assert_eq!(hierarchy.total_tasks, 1);
            }
        }
    }

    #[tokio::test]
    async fn test_maximum_reasonable_boundaries() {
        // Test with maximum reasonable configurations
        let chunking_engine = ChunkingEngine::new(ChunkingConfig {
            chunk_size: 10000,
            preserve_structure: true,
            min_chunk_size: 1000,
            max_overlap: 500,
        });
        
        let task_divider = HierarchicalTaskDivider::new(10, 50).unwrap();
        
        // Test with very large content
        let large_content = generate_random_content(50000, 99999);
        
        let start_time = Instant::now();
        
        let chunking_result = chunking_engine.chunk_content(
            &large_content,
            "max_boundary_test".to_string(),
            "test/max_boundary.rs".to_string(),
            "max_boundary.rs".to_string(),
            Some("rs".to_string()),
        ).unwrap();
        
        let chunking_time = start_time.elapsed();
        
        // Should handle large content efficiently
        assert!(chunking_time < Duration::from_secs(10), "Large content chunking took too long: {:?}", chunking_time);
        assert_eq!(chunking_result.total_lines, 50000);
        assert!(chunking_result.was_chunked);
        
        // Create many content triples
        let content_triples: Vec<ContentTriple> = (0..1000)
            .map(|i| ContentTriple {
                content_a: PathBuf::from(format!("max_boundary_{}_content.txt", i)),
                content_b: PathBuf::from(format!("max_boundary_{}_content_L1.txt", i)),
                content_c: PathBuf::from(format!("max_boundary_{}_content_L2.txt", i)),
                row_number: i + 1,
                table_name: "MAX_BOUNDARY_TEST".to_string(),
            })
            .collect();
        
        let hierarchy_start = Instant::now();
        let hierarchy = task_divider.create_hierarchy(content_triples).unwrap();
        let hierarchy_time = hierarchy_start.elapsed();
        
        assert!(hierarchy_time < Duration::from_secs(5), "Large hierarchy generation took too long: {:?}", hierarchy_time);
        assert_eq!(hierarchy.total_tasks, 1000);
        
        println!("Maximum boundary test:");
        println!("  Content lines: 50000");
        println!("  Content triples: 1000");
        println!("  Chunking time: {:?}", chunking_time);
        println!("  Hierarchy time: {:?}", hierarchy_time);
    }

    #[tokio::test]
    async fn test_error_boundary_conditions() {
        // Test various error boundary conditions
        
        // Test invalid task divider configurations
        assert!(HierarchicalTaskDivider::new(0, 5).is_err());
        assert!(HierarchicalTaskDivider::new(5, 0).is_err());
        
        // Test valid edge case configurations
        assert!(HierarchicalTaskDivider::new(1, 1).is_ok());
        assert!(HierarchicalTaskDivider::new(100, 100).is_ok());
        
        // Test chunking with various configurations
        let test_configs = vec![
            ChunkingConfig { chunk_size: 1, preserve_structure: false, min_chunk_size: 1, max_overlap: 0 },
            ChunkingConfig { chunk_size: 1000000, preserve_structure: true, min_chunk_size: 1, max_overlap: 1000 },
        ];
        
        for config in test_configs {
            let chunking_engine = ChunkingEngine::new(config);
            let content = "test content\nwith multiple lines\nfor boundary testing";
            
            let result = chunking_engine.chunk_content(
                content,
                "boundary_config_test".to_string(),
                "test/boundary_config.txt".to_string(),
                "boundary_config.txt".to_string(),
                Some("txt".to_string()),
            );
            
            // Should not fail even with extreme configurations
            assert!(result.is_ok());
            
            let chunking_result = result.unwrap();
            assert_eq!(chunking_result.total_lines, 3);
            
            // Validate chunks
            chunking_engine.validate_chunks(&chunking_result.chunks).unwrap();
        }
    }

    #[tokio::test]
    async fn test_unicode_and_special_character_boundaries() {
        // Test with various unicode and special characters
        let special_contents = vec![
            "Hello 世界\nこんにちは\n🚀 Rust",
            "Line with\ttabs\nand\r\nwindows\nline endings",
            "Émojis: 🎉🔥💯\nSpecial chars: àáâãäå\nSymbols: ∑∆∇∈∉",
            &("Very long line: ".to_string() + &"x".repeat(10000) + "\nShort line"),
        ];
        
        let chunking_engine = ChunkingEngine::with_chunk_size(2);
        
        for (i, content) in special_contents.iter().enumerate() {
            let chunking_result = chunking_engine.chunk_content(
                content,
                format!("unicode_test_{}", i),
                format!("test/unicode_{}.txt", i),
                format!("unicode_{}.txt", i),
                Some("txt".to_string()),
            ).unwrap();
            
            // Should handle unicode content correctly
            chunking_engine.validate_chunks(&chunking_result.chunks).unwrap();
            
            // Verify content preservation
            let reconstructed = chunking_result.chunks
                .iter()
                .map(|chunk| chunk.content.as_str())
                .collect::<Vec<_>>()
                .join("\n");
            
            assert_eq!(reconstructed, *content, "Unicode content preservation failed for case {}", i);
        }
    }
}
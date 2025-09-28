//! Integration tests for ingestion paths
//!
//! Tests cover both git repository and local folder ingestion paths,
//! simulating real-world ingestion scenarios as specified in task 9.2.

use code_ingest::processing::chunking::{ChunkingEngine, ChunkingConfig};
use code_ingest::tasks::hierarchical_task_divider::HierarchicalTaskDivider;
use code_ingest::tasks::content_extractor::ContentTriple;
use code_ingest::processing::{ProcessedFile, FileType};
use std::path::{Path, PathBuf};
use std::time::Instant;
use tempfile::TempDir;
use tokio;

/// Simulate processed files from git repository ingestion
fn create_git_repository_simulation(repo_name: &str, file_count: usize) -> Vec<ProcessedFile> {
    (1..=file_count)
        .map(|i| {
            let file_type = match i % 4 {
                0 => FileType::NonText,
                1 => FileType::Convertible,
                _ => FileType::DirectText,
            };
            
            let extension = match file_type {
                FileType::DirectText => "rs",
                FileType::Convertible => "md",
                FileType::NonText => "bin",
            };
            
            ProcessedFile {
                filepath: format!("{}/src/module_{}/file_{}.{}", repo_name, i / 10, i, extension),
                filename: format!("file_{}.{}", i, extension),
                extension: extension.to_string(),
                file_size_bytes: (i * 1024) as i64,
                line_count: if file_type == FileType::NonText { None } else { Some((i * 20) as i32) },
                word_count: if file_type == FileType::NonText { None } else { Some((i * 100) as i32) },
                token_count: if file_type == FileType::NonText { None } else { Some((i * 150) as i32) },
                content_text: if file_type == FileType::NonText { 
                    None 
                } else { 
                    Some(format!("Content of file {} from repository {}", i, repo_name)) 
                },
                file_type,
                conversion_command: if file_type == FileType::Convertible { 
                    Some(format!("pandoc -f markdown -t plain")) 
                } else { 
                    None 
                },
                relative_path: format!("src/module_{}/file_{}.{}", i / 10, i, extension),
                absolute_path: format!("/tmp/repos/{}/src/module_{}/file_{}.{}", repo_name, i / 10, i, extension),
                skipped: false,
                skip_reason: None,
            }
        })
        .collect()
}

/// Simulate processed files from local folder ingestion
fn create_local_folder_simulation(folder_path: &str, file_count: usize) -> Vec<ProcessedFile> {
    (1..=file_count)
        .map(|i| {
            let file_type = match i % 3 {
                0 => FileType::Convertible,
                1 => FileType::NonText,
                _ => FileType::DirectText,
            };
            
            let extension = match file_type {
                FileType::DirectText => "py",
                FileType::Convertible => "txt",
                FileType::NonText => "exe",
            };
            
            ProcessedFile {
                filepath: format!("{}/subfolder_{}/local_file_{}.{}", folder_path, i / 5, i, extension),
                filename: format!("local_file_{}.{}", i, extension),
                extension: extension.to_string(),
                file_size_bytes: (i * 512) as i64,
                line_count: if file_type == FileType::NonText { None } else { Some((i * 15) as i32) },
                word_count: if file_type == FileType::NonText { None } else { Some((i * 75) as i32) },
                token_count: if file_type == FileType::NonText { None } else { Some((i * 120) as i32) },
                content_text: if file_type == FileType::NonText { 
                    None 
                } else { 
                    Some(format!("Local file {} content from folder {}", i, folder_path)) 
                },
                file_type,
                conversion_command: if file_type == FileType::Convertible { 
                    Some("cat".to_string()) 
                } else { 
                    None 
                },
                relative_path: format!("subfolder_{}/local_file_{}.{}", i / 5, i, extension),
                absolute_path: format!("{}/subfolder_{}/local_file_{}.{}", folder_path, i / 5, i, extension),
                skipped: false,
                skip_reason: None,
            }
        })
        .collect()
}

/// Create content triples from processed files
fn create_content_triples_from_processed_files(
    processed_files: &[ProcessedFile],
    table_name: &str,
) -> Vec<ContentTriple> {
    processed_files
        .iter()
        .enumerate()
        .filter(|(_, file)| file.content_text.is_some()) // Only process files with content
        .map(|(i, file)| ContentTriple {
            content_a: PathBuf::from(format!("{}_row_{}_content.txt", table_name, i + 1)),
            content_b: PathBuf::from(format!("{}_row_{}_content_L1.txt", table_name, i + 1)),
            content_c: PathBuf::from(format!("{}_row_{}_content_L2.txt", table_name, i + 1)),
            row_number: i + 1,
            table_name: table_name.to_string(),
        })
        .collect()
}

#[cfg(test)]
mod git_repository_ingestion_tests {
    use super::*;

    #[tokio::test]
    async fn test_git_repository_ingestion_workflow() {
        // Simulate ingesting a git repository
        let repo_name = "test-rust-project";
        let processed_files = create_git_repository_simulation(repo_name, 50);
        
        // Verify file distribution by type
        let direct_text_count = processed_files.iter().filter(|f| f.file_type == FileType::DirectText).count();
        let convertible_count = processed_files.iter().filter(|f| f.file_type == FileType::Convertible).count();
        let non_text_count = processed_files.iter().filter(|f| f.file_type == FileType::NonText).count();
        
        assert!(direct_text_count > 0);
        assert!(convertible_count > 0);
        assert!(non_text_count > 0);
        assert_eq!(direct_text_count + convertible_count + non_text_count, 50);
        
        // Create content triples from files with content
        let table_name = "INGEST_GIT_TEST_REPO";
        let content_triples = create_content_triples_from_processed_files(&processed_files, table_name);
        
        // Should have fewer content triples than total files (excluding non-text files)
        assert!(content_triples.len() < processed_files.len());
        assert!(content_triples.len() > 0);
        
        // Generate task hierarchy
        let task_divider = HierarchicalTaskDivider::new(3, 5).unwrap();
        let hierarchy = task_divider.create_hierarchy(content_triples.clone()).unwrap();
        
        assert_eq!(hierarchy.total_tasks, content_triples.len());
        assert!(!hierarchy.levels.is_empty());
        
        // Verify all content triples reference the correct table
        for triple in &content_triples {
            assert_eq!(triple.table_name, table_name);
        }
    }

    #[tokio::test]
    async fn test_git_repository_with_chunking_workflow() {
        // Simulate git repository ingestion with large files that need chunking
        let repo_name = "large-rust-project";
        let mut processed_files = create_git_repository_simulation(repo_name, 30);
        
        // Modify some files to have large line counts (simulate large files)
        for (i, file) in processed_files.iter_mut().enumerate() {
            if i % 5 == 0 && file.file_type == FileType::DirectText {
                file.line_count = Some(500 + (i * 100) as i32); // Large files
                file.content_text = Some(format!("Large file {} with {} lines", i, file.line_count.unwrap()));
            }
        }
        
        // Simulate chunking for large files
        let chunking_engine = ChunkingEngine::with_chunk_size(200);
        let mut all_content_triples = Vec::new();
        
        for (file_idx, file) in processed_files.iter().enumerate() {
            if let Some(content) = &file.content_text {
                if let Some(line_count) = file.line_count {
                    if line_count > 200 {
                        // Simulate chunking large file
                        let simulated_content = (1..=line_count)
                            .map(|line| format!("Line {} of file {}", line, file.filename))
                            .collect::<Vec<_>>()
                            .join("\n");
                        
                        let chunking_result = chunking_engine.chunk_content(
                            &simulated_content,
                            format!("git_file_{}", file_idx),
                            file.filepath.clone(),
                            file.filename.clone(),
                            Some(file.extension.clone()),
                        ).unwrap();
                        
                        // Create content triples for each chunk
                        for (chunk_idx, _chunk) in chunking_result.chunks.iter().enumerate() {
                            let content_triple = ContentTriple {
                                content_a: PathBuf::from(format!("git_chunked_file_{}_chunk_{}_content.txt", file_idx, chunk_idx + 1)),
                                content_b: PathBuf::from(format!("git_chunked_file_{}_chunk_{}_content_L1.txt", file_idx, chunk_idx + 1)),
                                content_c: PathBuf::from(format!("git_chunked_file_{}_chunk_{}_content_L2.txt", file_idx, chunk_idx + 1)),
                                row_number: file_idx * 100 + chunk_idx + 1,
                                table_name: "INGEST_GIT_CHUNKED_REPO".to_string(),
                            };
                            all_content_triples.push(content_triple);
                        }
                    } else {
                        // Small file, no chunking needed
                        let content_triple = ContentTriple {
                            content_a: PathBuf::from(format!("git_small_file_{}_content.txt", file_idx)),
                            content_b: PathBuf::from(format!("git_small_file_{}_content_L1.txt", file_idx)),
                            content_c: PathBuf::from(format!("git_small_file_{}_content_L2.txt", file_idx)),
                            row_number: file_idx + 1,
                            table_name: "INGEST_GIT_CHUNKED_REPO".to_string(),
                        };
                        all_content_triples.push(content_triple);
                    }
                }
            }
        }
        
        // Generate task hierarchy
        let task_divider = HierarchicalTaskDivider::new(4, 7).unwrap();
        let hierarchy = task_divider.create_hierarchy(all_content_triples.clone()).unwrap();
        
        assert_eq!(hierarchy.total_tasks, all_content_triples.len());
        assert!(hierarchy.total_tasks > 30); // Should have more tasks than files due to chunking
        assert!(!hierarchy.levels.is_empty());
    }

    #[tokio::test]
    async fn test_git_repository_performance() {
        // Test performance with a large simulated git repository
        let start_time = Instant::now();
        
        let repo_name = "performance-test-repo";
        let processed_files = create_git_repository_simulation(repo_name, 200);
        
        let simulation_time = start_time.elapsed();
        
        // Create content triples
        let content_creation_start = Instant::now();
        let table_name = "INGEST_GIT_PERF_REPO";
        let content_triples = create_content_triples_from_processed_files(&processed_files, table_name);
        let content_creation_time = content_creation_start.elapsed();
        
        // Generate task hierarchy
        let hierarchy_start = Instant::now();
        let task_divider = HierarchicalTaskDivider::new(4, 10).unwrap();
        let hierarchy = task_divider.create_hierarchy(content_triples.clone()).unwrap();
        let hierarchy_time = hierarchy_start.elapsed();
        
        let total_time = start_time.elapsed();
        
        // Performance assertions
        assert!(simulation_time.as_millis() < 1000, "File simulation took too long: {:?}", simulation_time);
        assert!(content_creation_time.as_millis() < 500, "Content triple creation took too long: {:?}", content_creation_time);
        assert!(hierarchy_time.as_millis() < 1000, "Hierarchy generation took too long: {:?}", hierarchy_time);
        assert!(total_time.as_millis() < 2000, "Total git repository workflow took too long: {:?}", total_time);
        
        // Verify results
        assert_eq!(processed_files.len(), 200);
        assert!(content_triples.len() > 100); // Should have many content triples
        assert_eq!(hierarchy.total_tasks, content_triples.len());
        
        println!("Git repository performance test:");
        println!("  Simulated files: {}", processed_files.len());
        println!("  Content triples: {}", content_triples.len());
        println!("  Generated tasks: {}", hierarchy.total_tasks);
        println!("  Simulation time: {:?}", simulation_time);
        println!("  Content creation time: {:?}", content_creation_time);
        println!("  Hierarchy time: {:?}", hierarchy_time);
        println!("  Total time: {:?}", total_time);
    }
}

#[cfg(test)]
mod local_folder_ingestion_tests {
    use super::*;

    #[tokio::test]
    async fn test_local_folder_ingestion_workflow() {
        // Simulate ingesting a local folder
        let folder_path = "/home/user/local-project";
        let processed_files = create_local_folder_simulation(folder_path, 40);
        
        // Verify file distribution by type
        let direct_text_count = processed_files.iter().filter(|f| f.file_type == FileType::DirectText).count();
        let convertible_count = processed_files.iter().filter(|f| f.file_type == FileType::Convertible).count();
        let non_text_count = processed_files.iter().filter(|f| f.file_type == FileType::NonText).count();
        
        assert!(direct_text_count > 0);
        assert!(convertible_count > 0);
        assert!(non_text_count > 0);
        assert_eq!(direct_text_count + convertible_count + non_text_count, 40);
        
        // Verify local folder specific characteristics
        for file in &processed_files {
            assert!(file.absolute_path.starts_with(folder_path));
            assert!(!file.relative_path.starts_with("/"));
            
            // Check file type consistency
            match file.file_type {
                FileType::DirectText => assert_eq!(file.extension, "py"),
                FileType::Convertible => assert_eq!(file.extension, "txt"),
                FileType::NonText => assert_eq!(file.extension, "exe"),
            }
        }
        
        // Create content triples and generate hierarchy
        let table_name = "INGEST_LOCAL_FOLDER";
        let content_triples = create_content_triples_from_processed_files(&processed_files, table_name);
        
        let task_divider = HierarchicalTaskDivider::new(3, 6).unwrap();
        let hierarchy = task_divider.create_hierarchy(content_triples.clone()).unwrap();
        
        assert_eq!(hierarchy.total_tasks, content_triples.len());
        assert!(!hierarchy.levels.is_empty());
    }

    #[tokio::test]
    async fn test_local_folder_with_mixed_file_types() {
        // Test local folder ingestion with various file types and sizes
        let folder_path = "/project/mixed-content";
        let mut processed_files = create_local_folder_simulation(folder_path, 60);
        
        // Add some files with special characteristics
        processed_files.push(ProcessedFile {
            filepath: format!("{}/README.md", folder_path),
            filename: "README.md".to_string(),
            extension: "md".to_string(),
            file_size_bytes: 2048,
            line_count: Some(100),
            word_count: Some(500),
            token_count: Some(600),
            content_text: Some("README file content with project documentation".to_string()),
            file_type: FileType::Convertible,
            conversion_command: Some("pandoc -f markdown -t plain".to_string()),
            relative_path: "README.md".to_string(),
            absolute_path: format!("{}/README.md", folder_path),
            skipped: false,
            skip_reason: None,
        });
        
        // Add a skipped file
        processed_files.push(ProcessedFile {
            filepath: format!("{}/large_binary.bin", folder_path),
            filename: "large_binary.bin".to_string(),
            extension: "bin".to_string(),
            file_size_bytes: 1024 * 1024 * 100, // 100MB
            line_count: None,
            word_count: None,
            token_count: None,
            content_text: None,
            file_type: FileType::NonText,
            conversion_command: None,
            relative_path: "large_binary.bin".to_string(),
            absolute_path: format!("{}/large_binary.bin", folder_path),
            skipped: true,
            skip_reason: Some("File too large".to_string()),
        });
        
        // Verify mixed file handling
        let skipped_files = processed_files.iter().filter(|f| f.skipped).count();
        let non_skipped_files = processed_files.iter().filter(|f| !f.skipped).count();
        
        assert_eq!(skipped_files, 1);
        assert_eq!(non_skipped_files, 61);
        
        // Create content triples only from non-skipped files with content
        let content_triples: Vec<ContentTriple> = processed_files
            .iter()
            .enumerate()
            .filter(|(_, file)| !file.skipped && file.content_text.is_some())
            .map(|(i, _)| ContentTriple {
                content_a: PathBuf::from(format!("mixed_folder_row_{}_content.txt", i + 1)),
                content_b: PathBuf::from(format!("mixed_folder_row_{}_content_L1.txt", i + 1)),
                content_c: PathBuf::from(format!("mixed_folder_row_{}_content_L2.txt", i + 1)),
                row_number: i + 1,
                table_name: "INGEST_MIXED_FOLDER".to_string(),
            })
            .collect();
        
        // Generate hierarchy
        let task_divider = HierarchicalTaskDivider::new(3, 8).unwrap();
        let hierarchy = task_divider.create_hierarchy(content_triples.clone()).unwrap();
        
        assert_eq!(hierarchy.total_tasks, content_triples.len());
        assert!(hierarchy.total_tasks > 0);
        assert!(!hierarchy.levels.is_empty());
    }

    #[tokio::test]
    async fn test_local_folder_chunking_integration() {
        // Test local folder ingestion with chunking for large files
        let folder_path = "/project/large-files";
        let mut processed_files = create_local_folder_simulation(folder_path, 25);
        
        // Create some large files that need chunking
        for i in 0..5 {
            processed_files[i * 5].line_count = Some(300 + (i * 50) as i32);
            processed_files[i * 5].content_text = Some(format!(
                "Large local file {} with {} lines of content", 
                i, 
                processed_files[i * 5].line_count.unwrap()
            ));
        }
        
        // Process files with chunking
        let chunking_engine = ChunkingEngine::with_chunk_size(150);
        let mut all_content_triples = Vec::new();
        
        for (file_idx, file) in processed_files.iter().enumerate() {
            if let Some(content) = &file.content_text {
                if let Some(line_count) = file.line_count {
                    if line_count > 150 {
                        // Simulate chunking
                        let simulated_content = (1..=line_count)
                            .map(|line| format!("Local file line {} content", line))
                            .collect::<Vec<_>>()
                            .join("\n");
                        
                        let chunking_result = chunking_engine.chunk_content(
                            &simulated_content,
                            format!("local_file_{}", file_idx),
                            file.filepath.clone(),
                            file.filename.clone(),
                            Some(file.extension.clone()),
                        ).unwrap();
                        
                        // Create content triples for chunks
                        for (chunk_idx, _chunk) in chunking_result.chunks.iter().enumerate() {
                            let content_triple = ContentTriple {
                                content_a: PathBuf::from(format!("local_chunked_file_{}_chunk_{}_content.txt", file_idx, chunk_idx + 1)),
                                content_b: PathBuf::from(format!("local_chunked_file_{}_chunk_{}_content_L1.txt", file_idx, chunk_idx + 1)),
                                content_c: PathBuf::from(format!("local_chunked_file_{}_chunk_{}_content_L2.txt", file_idx, chunk_idx + 1)),
                                row_number: file_idx * 100 + chunk_idx + 1,
                                table_name: "INGEST_LOCAL_CHUNKED".to_string(),
                            };
                            all_content_triples.push(content_triple);
                        }
                    } else {
                        // Small file, no chunking
                        let content_triple = ContentTriple {
                            content_a: PathBuf::from(format!("local_small_file_{}_content.txt", file_idx)),
                            content_b: PathBuf::from(format!("local_small_file_{}_content_L1.txt", file_idx)),
                            content_c: PathBuf::from(format!("local_small_file_{}_content_L2.txt", file_idx)),
                            row_number: file_idx + 1,
                            table_name: "INGEST_LOCAL_CHUNKED".to_string(),
                        };
                        all_content_triples.push(content_triple);
                    }
                }
            }
        }
        
        // Generate task hierarchy
        let task_divider = HierarchicalTaskDivider::new(3, 7).unwrap();
        let hierarchy = task_divider.create_hierarchy(all_content_triples.clone()).unwrap();
        
        assert_eq!(hierarchy.total_tasks, all_content_triples.len());
        assert!(hierarchy.total_tasks > 25); // Should have more tasks due to chunking
        assert!(!hierarchy.levels.is_empty());
    }
}

#[cfg(test)]
mod ingestion_comparison_tests {
    use super::*;

    #[tokio::test]
    async fn test_git_vs_local_folder_ingestion_comparison() {
        // Compare git repository vs local folder ingestion workflows
        let start_time = Instant::now();
        
        // Simulate git repository ingestion
        let git_start = Instant::now();
        let git_files = create_git_repository_simulation("comparison-repo", 100);
        let git_content_triples = create_content_triples_from_processed_files(&git_files, "INGEST_GIT_COMPARISON");
        let git_time = git_start.elapsed();
        
        // Simulate local folder ingestion
        let local_start = Instant::now();
        let local_files = create_local_folder_simulation("/comparison/folder", 100);
        let local_content_triples = create_content_triples_from_processed_files(&local_files, "INGEST_LOCAL_COMPARISON");
        let local_time = local_start.elapsed();
        
        // Generate hierarchies for both
        let task_divider = HierarchicalTaskDivider::new(4, 8).unwrap();
        
        let git_hierarchy_start = Instant::now();
        let git_hierarchy = task_divider.create_hierarchy(git_content_triples.clone()).unwrap();
        let git_hierarchy_time = git_hierarchy_start.elapsed();
        
        let local_hierarchy_start = Instant::now();
        let local_hierarchy = task_divider.create_hierarchy(local_content_triples.clone()).unwrap();
        let local_hierarchy_time = local_hierarchy_start.elapsed();
        
        let total_time = start_time.elapsed();
        
        // Compare results
        assert_eq!(git_files.len(), local_files.len()); // Same number of files
        assert!(git_content_triples.len() > 0);
        assert!(local_content_triples.len() > 0);
        
        // Both should generate valid hierarchies
        assert_eq!(git_hierarchy.total_tasks, git_content_triples.len());
        assert_eq!(local_hierarchy.total_tasks, local_content_triples.len());
        
        // Performance should be similar
        let time_difference = if git_time > local_time {
            git_time - local_time
        } else {
            local_time - git_time
        };
        
        // Time difference should not be too large (within 50% of each other)
        let max_time = std::cmp::max(git_time, local_time);
        assert!(time_difference < max_time / 2, "Git and local processing times too different: git={:?}, local={:?}", git_time, local_time);
        
        println!("Git vs Local folder comparison:");
        println!("  Files processed: {} each", git_files.len());
        println!("  Git content triples: {}", git_content_triples.len());
        println!("  Local content triples: {}", local_content_triples.len());
        println!("  Git processing time: {:?}", git_time);
        println!("  Local processing time: {:?}", local_time);
        println!("  Git hierarchy time: {:?}", git_hierarchy_time);
        println!("  Local hierarchy time: {:?}", local_hierarchy_time);
        println!("  Total comparison time: {:?}", total_time);
    }

    #[tokio::test]
    async fn test_mixed_ingestion_sources_workflow() {
        // Test workflow that combines both git and local folder sources
        let task_divider = HierarchicalTaskDivider::new(4, 10).unwrap();
        
        // Create files from both sources
        let git_files = create_git_repository_simulation("mixed-git-repo", 30);
        let local_files = create_local_folder_simulation("/mixed/local", 20);
        
        // Combine content triples from both sources
        let mut all_content_triples = Vec::new();
        
        // Add git content triples
        let git_triples = create_content_triples_from_processed_files(&git_files, "INGEST_MIXED_GIT");
        all_content_triples.extend(git_triples);
        
        // Add local content triples with different row numbering
        let local_triples: Vec<ContentTriple> = create_content_triples_from_processed_files(&local_files, "INGEST_MIXED_LOCAL")
            .into_iter()
            .enumerate()
            .map(|(i, mut triple)| {
                triple.row_number += 1000; // Offset to avoid conflicts
                triple
            })
            .collect();
        all_content_triples.extend(local_triples);
        
        // Generate unified hierarchy
        let hierarchy = task_divider.create_hierarchy(all_content_triples.clone()).unwrap();
        
        assert_eq!(hierarchy.total_tasks, all_content_triples.len());
        assert!(hierarchy.total_tasks > 30); // Should have tasks from both sources
        assert!(!hierarchy.levels.is_empty());
        
        // Verify mixed sources are handled correctly
        let git_tasks = all_content_triples.iter().filter(|t| t.table_name.contains("GIT")).count();
        let local_tasks = all_content_triples.iter().filter(|t| t.table_name.contains("LOCAL")).count();
        
        assert!(git_tasks > 0);
        assert!(local_tasks > 0);
        assert_eq!(git_tasks + local_tasks, all_content_triples.len());
        
        println!("Mixed ingestion sources test:");
        println!("  Git files: {}", git_files.len());
        println!("  Local files: {}", local_files.len());
        println!("  Git tasks: {}", git_tasks);
        println!("  Local tasks: {}", local_tasks);
        println!("  Total tasks: {}", hierarchy.total_tasks);
    }
}
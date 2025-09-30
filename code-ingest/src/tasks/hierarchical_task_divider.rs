//! [DEPRECATED] Hierarchical task division for creating 4-level task hierarchies
//! 
//! ⚠️  DEPRECATION WARNING: This module is deprecated and will be removed in a future version.
//! Please use `chunk_level_task_generator` instead for simpler, more maintainable task generation.
//! 
//! This module implements the hierarchical task division algorithm that creates
//! structured task hierarchies with configurable levels and groups per level.
//! The default configuration creates 4 levels with 7 groups per level.
//! 
//! Migration path: Use `ChunkLevelTaskGenerator` for file-level or chunk-level task generation.

use crate::error::{TaskError, TaskResult};
use crate::tasks::content_extractor::{ContentTriple, ProcessingConfig, ProcessingProgress, CancellationToken};
use serde::{Deserialize, Serialize};
use std::time::Instant;
use tracing::{debug, info};



/// Hierarchical task divider that creates multi-level task structures
#[derive(Debug, Clone)]
pub struct HierarchicalTaskDivider {
    /// Number of hierarchy levels (default: 4)
    pub levels: usize,
    /// Number of groups per level (default: 7)
    pub groups_per_level: usize,
}

impl HierarchicalTaskDivider {
    /// Create a new hierarchical task divider
    pub fn new(levels: usize, groups_per_level: usize) -> TaskResult<Self> {
        if levels == 0 {
            return Err(TaskError::InvalidTaskConfiguration {
                cause: "Levels must be greater than 0".to_string(),
                suggestion: "Set levels to a value between 1 and 10".to_string(),
            });
        }
        
        if groups_per_level == 0 {
            return Err(TaskError::InvalidTaskConfiguration {
                cause: "Groups per level must be greater than 0".to_string(),
                suggestion: "Set groups_per_level to a value between 1 and 20".to_string(),
            });
        }
        
        Ok(Self {
            levels,
            groups_per_level,
        })
    }

    /// Create a task hierarchy from content triples
    pub fn create_hierarchy(&self, content_triples: Vec<ContentTriple>) -> TaskResult<TaskHierarchy> {
        if content_triples.is_empty() {
            return Ok(TaskHierarchy {
                levels: vec![],
                total_tasks: 0,
            });
        }

        let total_tasks = content_triples.len();
        let levels = self.distribute_across_levels_with_ids(content_triples, 1, String::new())?;

        Ok(TaskHierarchy {
            levels,
            total_tasks,
        })
    }

    /// Create a task hierarchy with performance optimizations for large datasets
    ///
    /// This method is optimized for large numbers of content triples (10,000+) and provides:
    /// - Memory-efficient processing with streaming
    /// - Progress reporting and cancellation support
    /// - Batch processing to prevent memory exhaustion
    /// - Concurrent task creation where possible
    ///
    /// # Arguments
    /// * `content_triples` - List of content triples to organize into hierarchy
    /// * `config` - Processing configuration for optimization
    /// * `progress_callback` - Optional callback for progress updates
    /// * `cancellation_token` - Optional token for operation cancellation
    ///
    /// # Returns
    /// * `TaskResult<TaskHierarchy>` - Hierarchical task structure
    pub async fn create_hierarchy_streaming<F>(
        &self,
        content_triples: Vec<ContentTriple>,
        config: ProcessingConfig,
        mut progress_callback: Option<F>,
        cancellation_token: Option<CancellationToken>,
    ) -> TaskResult<TaskHierarchy>
    where
        F: FnMut(ProcessingProgress) + Send + 'static,
    {
        let start_time = Instant::now();
        let total_items = content_triples.len();
        
        debug!("Starting streaming hierarchy creation for {} items with config: {:?}", 
               total_items, config);

        if content_triples.is_empty() {
            return Ok(TaskHierarchy {
                levels: vec![],
                total_tasks: 0,
            });
        }

        // Check memory usage before processing
        let estimated_memory_mb = self.estimate_hierarchy_memory_usage(total_items);
        if estimated_memory_mb > config.memory_limit_mb {
            return Err(TaskError::MemoryLimitExceeded {
                operation: "hierarchy creation".to_string(),
                used_mb: estimated_memory_mb,
                limit_mb: config.memory_limit_mb,
                suggestion: format!("Increase memory limit to {} MB or process in smaller batches", 
                                  estimated_memory_mb + 100),
            });
        }

        // Process in batches if the dataset is very large
        if total_items > config.batch_size * 10 {
            return self.create_hierarchy_batched(
                content_triples,
                config,
                progress_callback,
                cancellation_token,
            ).await;
        }

        // For smaller datasets, use the optimized single-pass method
        let mut processed = 0;
        let levels = self.distribute_across_levels_with_progress(
            content_triples,
            1,
            String::new(),
            &mut processed,
            total_items,
            &config,
            &mut progress_callback,
            &cancellation_token,
            start_time,
        ).await?;

        let elapsed = start_time.elapsed();
        info!("Completed hierarchy creation: {} items in {:.2}s ({:.1} items/sec)", 
              total_items, elapsed.as_secs_f64(), total_items as f64 / elapsed.as_secs_f64());

        Ok(TaskHierarchy {
            levels,
            total_tasks: total_items,
        })
    }

    /// Create hierarchy using batched processing for very large datasets
    async fn create_hierarchy_batched<F>(
        &self,
        content_triples: Vec<ContentTriple>,
        config: ProcessingConfig,
        mut progress_callback: Option<F>,
        cancellation_token: Option<CancellationToken>,
    ) -> TaskResult<TaskHierarchy>
    where
        F: FnMut(ProcessingProgress) + Send + 'static,
    {
        let start_time = Instant::now();
        let total_items = content_triples.len();
        let total_batches = (total_items + config.batch_size - 1) / config.batch_size;
        
        info!("Processing {} items in {} batches of size {}", 
              total_items, total_batches, config.batch_size);

        // Process the first level in batches to create top-level groups
        let mut all_levels = Vec::new();
        let mut processed = 0;

        for batch_num in 0..total_batches {
            // Check for cancellation
            if let Some(ref token) = cancellation_token {
                if token.is_cancelled().await {
                    return Err(TaskError::AsyncCancelled {
                        operation: "hierarchy creation (batched)".to_string(),
                        suggestion: "Operation was cancelled by user request".to_string(),
                    });
                }
            }

            let start_idx = batch_num * config.batch_size;
            let end_idx = std::cmp::min(start_idx + config.batch_size, total_items);
            let batch_items = content_triples[start_idx..end_idx].to_vec();

            debug!("Processing batch {}/{}: {} items", 
                   batch_num + 1, total_batches, batch_items.len());

            // Process batch with timeout
            let batch_result = tokio::time::timeout(
                std::time::Duration::from_secs(config.operation_timeout_seconds),
                self.process_hierarchy_batch(batch_items, batch_num + 1)
            ).await;

            let batch_levels = match batch_result {
                Ok(Ok(levels)) => levels,
                Ok(Err(e)) => {
                    return Err(TaskError::BatchProcessingFailed {
                        processed,
                        total: total_items,
                        cause: e.to_string(),
                        suggestion: "Reduce batch size or increase timeout".to_string(),
                        source: Some(Box::new(e)),
                    });
                }
                Err(_) => {
                    return Err(TaskError::AsyncTimeout {
                        operation: "hierarchy batch processing".to_string(),
                        timeout_seconds: config.operation_timeout_seconds,
                        suggestion: "Increase timeout or reduce batch size".to_string(),
                    });
                }
            };

            // Merge batch levels into overall structure
            if all_levels.is_empty() {
                all_levels = batch_levels;
            } else {
                self.merge_hierarchy_levels(&mut all_levels, batch_levels)?;
            }

            processed += end_idx - start_idx;

            // Report progress
            if config.enable_progress_reporting && processed % config.progress_interval == 0 {
                if let Some(ref mut callback) = progress_callback {
                    let elapsed = start_time.elapsed().as_secs();
                    let estimated_remaining = if processed > 0 {
                        Some((elapsed * (total_items - processed) as u64) / processed as u64)
                    } else {
                        None
                    };

                    let progress = ProcessingProgress {
                        processed,
                        total: total_items,
                        current_batch: batch_num + 1,
                        total_batches,
                        elapsed_seconds: elapsed,
                        estimated_remaining_seconds: estimated_remaining,
                        current_memory_usage_mb: Some(self.estimate_hierarchy_memory_usage(processed)),
                    };

                    callback(progress);
                }
            }
        }

        let elapsed = start_time.elapsed();
        info!("Completed batched hierarchy creation: {} items in {:.2}s ({:.1} items/sec)", 
              total_items, elapsed.as_secs_f64(), total_items as f64 / elapsed.as_secs_f64());

        Ok(TaskHierarchy {
            levels: all_levels,
            total_tasks: total_items,
        })
    }

    /// Process a single batch for hierarchy creation
    async fn process_hierarchy_batch(
        &self,
        batch_items: Vec<ContentTriple>,
        batch_id: usize,
    ) -> TaskResult<Vec<TaskLevel>> {
        // Create a mini-hierarchy for this batch
        let batch_divider = HierarchicalTaskDivider::new(self.levels, self.groups_per_level)?;
        let batch_levels = batch_divider.distribute_across_levels_with_ids(
            batch_items,
            1,
            format!("B{}", batch_id), // Prefix with batch ID
        )?;

        Ok(batch_levels)
    }

    /// Merge hierarchy levels from batches
    fn merge_hierarchy_levels(
        &self,
        target_levels: &mut Vec<TaskLevel>,
        source_levels: Vec<TaskLevel>,
    ) -> TaskResult<()> {
        for source_level in source_levels {
            // Find or create corresponding level in target
            let target_level = target_levels
                .iter_mut()
                .find(|level| level.level == source_level.level);

            if let Some(target_level) = target_level {
                // Merge groups
                target_level.groups.extend(source_level.groups);
            } else {
                // Add new level
                target_levels.push(source_level);
            }
        }

        // Sort levels by level number
        target_levels.sort_by_key(|level| level.level);

        Ok(())
    }

    /// Estimate memory usage for hierarchy creation in MB
    fn estimate_hierarchy_memory_usage(&self, item_count: usize) -> usize {
        // Rough estimation based on hierarchy structure:
        // - Each AnalysisTask: ~500 bytes (paths, strings, metadata)
        // - Each HierarchicalTaskGroup: ~200 bytes + tasks
        // - Each TaskLevel: ~100 bytes + groups
        // - Hierarchy overhead: ~1000 bytes
        
        let task_memory = item_count * 500; // bytes per task
        let group_memory = (item_count / self.groups_per_level + 1) * 200; // estimated groups
        let level_memory = self.levels * 100; // bytes per level
        let overhead = 1000; // general overhead
        
        let total_bytes = task_memory + group_memory + level_memory + overhead;
        let total_mb = total_bytes / (1024 * 1024);
        
        std::cmp::max(1, total_mb) // Minimum 1 MB
    }

    /// Distribute items across levels with progress reporting and cancellation support
    async fn distribute_across_levels_with_progress<F>(
        &self,
        items: Vec<ContentTriple>,
        current_level: usize,
        parent_id: String,
        processed: &mut usize,
        total: usize,
        config: &ProcessingConfig,
        progress_callback: &mut Option<F>,
        cancellation_token: &Option<CancellationToken>,
        start_time: Instant,
    ) -> TaskResult<Vec<TaskLevel>>
    where
        F: FnMut(ProcessingProgress) + Send + 'static,
    {
        if items.is_empty() {
            return Ok(vec![]);
        }

        // Check for cancellation
        if let Some(ref token) = cancellation_token {
            if token.is_cancelled().await {
                return Err(TaskError::AsyncCancelled {
                    operation: "hierarchy level distribution".to_string(),
                    suggestion: "Operation was cancelled by user request".to_string(),
                });
            }
        }

        // If we've reached the maximum level, create leaf tasks
        if current_level > self.levels {
            let groups = self.create_leaf_groups_with_ids(items, current_level, &parent_id)?;
            *processed += groups.iter().map(|g| g.tasks.len()).sum::<usize>();
            
            // Report progress
            if config.enable_progress_reporting {
                if let Some(ref mut callback) = progress_callback {
                    let elapsed = start_time.elapsed().as_secs();
                    let estimated_remaining = if *processed > 0 {
                        Some((elapsed * (total - *processed) as u64) / *processed as u64)
                    } else {
                        None
                    };

                    let progress = ProcessingProgress {
                        processed: *processed,
                        total,
                        current_batch: 1,
                        total_batches: 1,
                        elapsed_seconds: elapsed,
                        estimated_remaining_seconds: estimated_remaining,
                        current_memory_usage_mb: Some(self.estimate_hierarchy_memory_usage(*processed)),
                    };

                    callback(progress);
                }
            }

            return Ok(vec![TaskLevel {
                level: current_level,
                groups,
            }]);
        }

        // Use the existing synchronous method for now
        // In a full implementation, this could be made async with concurrent processing
        self.distribute_across_levels_with_ids(items, current_level, parent_id)
    }

    /// Distribute content triples across hierarchy levels with proper ID generation
    /// 
    /// This method implements mathematical distribution with remainder handling and
    /// generates hierarchical IDs in the format 1.2.3.4:
    /// - Level 1: 1, 2, 3, ...
    /// - Level 2: 1.1, 1.2, 1.3, ..., 2.1, 2.2, ...
    /// - Level 3: 1.1.1, 1.1.2, ..., 1.2.1, 1.2.2, ...
    /// - Level 4: 1.1.1.1, 1.1.1.2, ..., (leaf tasks)
    fn distribute_across_levels_with_ids(
        &self,
        items: Vec<ContentTriple>,
        current_level: usize,
        parent_id: String,
    ) -> TaskResult<Vec<TaskLevel>> {
        if items.is_empty() {
            return Ok(vec![]);
        }

        // If we've reached the maximum level, create leaf tasks
        if current_level > self.levels {
            let groups = self.create_leaf_groups_with_ids(items, current_level, &parent_id)?;
            return Ok(vec![TaskLevel {
                level: current_level,
                groups,
            }]);
        }

        // Mathematical distribution with remainder handling
        let total_items = items.len();
        let base_items_per_group = total_items / self.groups_per_level;
        let remainder = total_items % self.groups_per_level;
        
        let mut groups = Vec::new();
        let mut item_index = 0;

        for group_id in 1..=self.groups_per_level {
            // First 'remainder' groups get one extra item
            let items_in_this_group = if group_id <= remainder {
                base_items_per_group + 1
            } else {
                base_items_per_group
            };

            // Break if no more items to distribute
            if items_in_this_group == 0 || item_index >= total_items {
                break;
            }

            let end_index = std::cmp::min(item_index + items_in_this_group, total_items);
            let group_items: Vec<ContentTriple> = items[item_index..end_index].to_vec();

            // Generate hierarchical ID
            let hierarchical_id = if parent_id.is_empty() {
                group_id.to_string()
            } else {
                format!("{}.{}", parent_id, group_id)
            };

            // Create hierarchical task group
            let group = if current_level < self.levels {
                // Intermediate level - create sub-groups recursively
                let sub_levels = self.distribute_across_levels_with_ids(
                    group_items, 
                    current_level + 1, 
                    hierarchical_id.clone()
                )?;
                let sub_groups = if let Some(next_level) = sub_levels.first() {
                    next_level.groups.clone()
                } else {
                    vec![]
                };

                HierarchicalTaskGroup {
                    id: hierarchical_id.clone(),
                    title: format!("Task Group {} (Level {})", hierarchical_id, current_level),
                    tasks: vec![], // Intermediate groups don't have direct tasks
                    sub_groups,
                }
            } else {
                // Leaf level - create actual analysis tasks
                let tasks = self.create_analysis_tasks_with_ids(group_items, &hierarchical_id)?;
                HierarchicalTaskGroup {
                    id: hierarchical_id.clone(),
                    title: format!("Analysis Group {} (Level {})", hierarchical_id, current_level),
                    tasks,
                    sub_groups: vec![],
                }
            };

            groups.push(group);
            item_index = end_index;
        }

        Ok(vec![TaskLevel {
            level: current_level,
            groups,
        }])
    }

    /// Legacy method for backward compatibility
    #[allow(dead_code)]
    fn distribute_across_levels(
        &self,
        items: Vec<ContentTriple>,
        current_level: usize,
    ) -> TaskResult<Vec<TaskLevel>> {
        self.distribute_across_levels_with_ids(items, current_level, String::new())
    }

    /// Create leaf groups for the final level using mathematical distribution with hierarchical IDs
    fn create_leaf_groups_with_ids(
        &self,
        items: Vec<ContentTriple>,
        _level: usize,
        parent_id: &str,
    ) -> TaskResult<Vec<HierarchicalTaskGroup>> {
        if items.is_empty() {
            return Ok(vec![]);
        }

        // Mathematical distribution with remainder handling
        let total_items = items.len();
        let base_items_per_group = total_items / self.groups_per_level;
        let remainder = total_items % self.groups_per_level;
        
        let mut groups = Vec::new();
        let mut item_index = 0;

        for group_id in 1..=self.groups_per_level {
            // First 'remainder' groups get one extra item
            let items_in_this_group = if group_id <= remainder {
                base_items_per_group + 1
            } else {
                base_items_per_group
            };

            // Break if no more items to distribute
            if items_in_this_group == 0 || item_index >= total_items {
                break;
            }

            let end_index = std::cmp::min(item_index + items_in_this_group, total_items);
            let group_items: Vec<ContentTriple> = items[item_index..end_index].to_vec();

            // Generate hierarchical ID
            let hierarchical_id = if parent_id.is_empty() {
                group_id.to_string()
            } else {
                format!("{}.{}", parent_id, group_id)
            };

            let tasks = self.create_analysis_tasks_with_ids(group_items, &hierarchical_id)?;

            let group = HierarchicalTaskGroup {
                id: hierarchical_id.clone(),
                title: format!("Analysis Group {}", hierarchical_id),
                tasks,
                sub_groups: vec![],
            };

            groups.push(group);
            item_index = end_index;
        }

        Ok(groups)
    }

    /// Legacy method for backward compatibility
    #[allow(dead_code)]
    fn create_leaf_groups(
        &self,
        items: Vec<ContentTriple>,
        level: usize,
    ) -> TaskResult<Vec<HierarchicalTaskGroup>> {
        self.create_leaf_groups_with_ids(items, level, "")
    }

    /// Create analysis tasks from content triples with hierarchical ID generation
    fn create_analysis_tasks_with_ids(
        &self,
        items: Vec<ContentTriple>,
        group_id: &str,
    ) -> TaskResult<Vec<AnalysisTask>> {
        let tasks = items
            .into_iter()
            .enumerate()
            .map(|(idx, content_triple)| {
                let task_id = format!("{}.{}", group_id, idx + 1);
                AnalysisTask {
                    id: task_id.clone(),
                    table_name: "UNKNOWN".to_string(), // Will be set by caller
                    row_number: content_triple.row_number,
                    content_files: content_triple,
                    prompt_file: std::path::PathBuf::from(".kiro/steering/spec-S04-steering-doc-analysis.md"),
                    output_file: std::path::PathBuf::from(format!("gringotts/WorkArea/task_{}.md", task_id.replace('.', "_"))),
                    analysis_stages: vec![
                        AnalysisStage::AnalyzeA,
                        AnalysisStage::AnalyzeAInContextB,
                        AnalysisStage::AnalyzeBInContextC,
                        AnalysisStage::AnalyzeAInContextBC,
                    ],
                }
            })
            .collect();

        Ok(tasks)
    }

    /// Legacy method for backward compatibility
    #[allow(dead_code)]
    fn create_analysis_tasks_from_items(
        &self,
        items: Vec<ContentTriple>,
        group_id: &str,
    ) -> TaskResult<Vec<AnalysisTask>> {
        self.create_analysis_tasks_with_ids(items, group_id)
    }
}

/// Complete task hierarchy with multiple levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskHierarchy {
    /// All levels in the hierarchy
    pub levels: Vec<TaskLevel>,
    /// Total number of analysis tasks across all levels
    pub total_tasks: usize,
}

/// A single level in the task hierarchy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskLevel {
    /// Level number (1-based)
    pub level: usize,
    /// Groups at this level
    pub groups: Vec<HierarchicalTaskGroup>,
}

/// A hierarchical task group that can contain sub-groups or analysis tasks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchicalTaskGroup {
    /// Group identifier (e.g., "1", "2.3", "1.2.3")
    pub id: String,
    /// Human-readable group title
    pub title: String,
    /// Analysis tasks at this group (only for leaf groups)
    pub tasks: Vec<AnalysisTask>,
    /// Sub-groups (for intermediate levels)
    pub sub_groups: Vec<HierarchicalTaskGroup>,
}

/// Individual analysis task for L1-L8 methodology
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisTask {
    /// Task identifier (e.g., "1.2.3.4")
    pub id: String,
    /// Source table name
    pub table_name: String,
    /// Row number from the source table
    pub row_number: usize,
    /// Content files (A, B, C)
    pub content_files: ContentTriple,
    /// Prompt file for analysis
    pub prompt_file: std::path::PathBuf,
    /// Output file for results
    pub output_file: std::path::PathBuf,
    /// Analysis stages to perform
    pub analysis_stages: Vec<AnalysisStage>,
}

/// Analysis stages for L1-L8 methodology
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnalysisStage {
    /// Analyze content A alone
    AnalyzeA,
    /// Analyze A in context of B
    AnalyzeAInContextB,
    /// Analyze B in context of C
    AnalyzeBInContextC,
    /// Analyze A in context of both B and C
    AnalyzeAInContextBC,
}

impl Default for HierarchicalTaskDivider {
    fn default() -> Self {
        Self {
            levels: 4,
            groups_per_level: 7,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::Arc;

    fn create_test_content_triples(count: usize) -> Vec<ContentTriple> {
        (1..=count)
            .map(|i| ContentTriple {
                content_a: PathBuf::from(format!("test_{}_content.txt", i)),
                content_b: PathBuf::from(format!("test_{}_content_L1.txt", i)),
                content_c: PathBuf::from(format!("test_{}_content_L2.txt", i)),
                row_number: i,
                table_name: "TEST_TABLE".to_string(),
            })
            .collect()
    }

    #[test]
    fn test_hierarchical_task_divider_creation() {
        let divider = HierarchicalTaskDivider::new(4, 7).unwrap();
        assert_eq!(divider.levels, 4);
        assert_eq!(divider.groups_per_level, 7);
    }

    #[test]
    fn test_hierarchical_task_divider_invalid_params() {
        // Test zero levels
        let result = HierarchicalTaskDivider::new(0, 7);
        assert!(result.is_err());
        
        // Test zero groups per level
        let result = HierarchicalTaskDivider::new(4, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_default_hierarchical_task_divider() {
        let divider = HierarchicalTaskDivider::default();
        assert_eq!(divider.levels, 4);
        assert_eq!(divider.groups_per_level, 7);
    }

    #[test]
    fn test_create_hierarchy_empty_input() {
        let divider = HierarchicalTaskDivider::new(4, 7).unwrap();
        let content_triples = vec![];
        
        let hierarchy = divider.create_hierarchy(content_triples).unwrap();
        assert_eq!(hierarchy.levels.len(), 0);
        assert_eq!(hierarchy.total_tasks, 0);
    }

    #[test]
    fn test_estimate_hierarchy_memory_usage() {
        let divider = HierarchicalTaskDivider::new(4, 7).unwrap();
        
        // Test memory estimation for different sizes
        assert_eq!(divider.estimate_hierarchy_memory_usage(0), 1); // Minimum 1 MB
        assert!(divider.estimate_hierarchy_memory_usage(1000) > 1);
        assert!(divider.estimate_hierarchy_memory_usage(10000) > divider.estimate_hierarchy_memory_usage(1000));
        
        // Memory usage should scale roughly linearly with item count
        let small_memory = divider.estimate_hierarchy_memory_usage(1000);
        let large_memory = divider.estimate_hierarchy_memory_usage(10000);
        assert!(large_memory > small_memory * 5); // Should be significantly larger
    }

    #[tokio::test]
    async fn test_create_hierarchy_streaming_small_dataset() {
        let divider = HierarchicalTaskDivider::new(2, 3).unwrap();
        let content_triples = create_test_content_triples(9); // 3^2 = 9 items for perfect distribution
        
        let config = ProcessingConfig {
            batch_size: 100, // Large batch size to avoid batching
            max_concurrent: 2,
            memory_limit_mb: 100,
            operation_timeout_seconds: 30,
            enable_progress_reporting: false,
            progress_interval: 10,
        };

        let hierarchy = divider.create_hierarchy_streaming(
            content_triples,
            config,
            None::<fn(ProcessingProgress)>,
            None,
        ).await.unwrap();

        assert_eq!(hierarchy.total_tasks, 9);
        assert!(!hierarchy.levels.is_empty());
    }

    #[tokio::test]
    async fn test_create_hierarchy_streaming_with_progress() {
        let divider = HierarchicalTaskDivider::new(2, 2).unwrap();
        let content_triples = create_test_content_triples(4);
        
        let config = ProcessingConfig {
            batch_size: 10,
            max_concurrent: 2,
            memory_limit_mb: 50,
            operation_timeout_seconds: 30,
            enable_progress_reporting: true,
            progress_interval: 1, // Report progress frequently
        };

        let progress_reports = Arc::new(std::sync::Mutex::new(Vec::new()));
        let progress_reports_clone = Arc::clone(&progress_reports);
        let progress_callback = move |progress: ProcessingProgress| {
            progress_reports_clone.lock().unwrap().push(progress);
        };

        let hierarchy = divider.create_hierarchy_streaming(
            content_triples,
            config,
            Some(progress_callback),
            None,
        ).await.unwrap();

        assert_eq!(hierarchy.total_tasks, 4);
        // Note: Progress reports might be empty for small datasets processed quickly
    }

    #[tokio::test]
    async fn test_create_hierarchy_streaming_with_cancellation() {
        let divider = HierarchicalTaskDivider::new(3, 5).unwrap();
        let content_triples = create_test_content_triples(25);
        
        let config = ProcessingConfig::default();
        let cancellation_token = CancellationToken::new();
        
        // Cancel immediately
        cancellation_token.cancel().await.unwrap();

        let result = divider.create_hierarchy_streaming(
            content_triples,
            config,
            None::<fn(ProcessingProgress)>,
            Some(cancellation_token),
        ).await;

        // Should be cancelled
        assert!(result.is_err());
        match result.unwrap_err() {
            TaskError::AsyncCancelled { operation, .. } => {
                assert!(operation.contains("hierarchy"));
            }
            _ => panic!("Expected AsyncCancelled error"),
        }
    }

    #[tokio::test]
    async fn test_create_hierarchy_streaming_memory_limit() {
        let divider = HierarchicalTaskDivider::new(4, 7).unwrap();
        let content_triples = create_test_content_triples(1000); // Large dataset
        
        let config = ProcessingConfig {
            batch_size: 100,
            max_concurrent: 2,
            memory_limit_mb: 1, // Very low memory limit
            operation_timeout_seconds: 30,
            enable_progress_reporting: false,
            progress_interval: 100,
        };

        let result = divider.create_hierarchy_streaming(
            content_triples,
            config,
            None::<fn(ProcessingProgress)>,
            None,
        ).await;

        // Should fail due to memory limit
        assert!(result.is_err());
        match result.unwrap_err() {
            TaskError::MemoryLimitExceeded { operation, used_mb, limit_mb, .. } => {
                assert!(operation.contains("hierarchy"));
                assert!(used_mb > limit_mb);
            }
            _ => panic!("Expected MemoryLimitExceeded error"),
        }
    }

    #[tokio::test]
    async fn test_process_hierarchy_batch() {
        let divider = HierarchicalTaskDivider::new(2, 3).unwrap();
        let batch_items = create_test_content_triples(6);
        
        let batch_levels = divider.process_hierarchy_batch(batch_items, 1).await.unwrap();
        
        assert!(!batch_levels.is_empty());
        // Verify that batch ID is included in group IDs
        for level in &batch_levels {
            for group in &level.groups {
                assert!(group.id.contains("B1")); // Should contain batch ID
            }
        }
    }

    #[test]
    fn test_merge_hierarchy_levels() {
        let divider = HierarchicalTaskDivider::new(2, 3).unwrap();
        
        // Create target levels
        let mut target_levels = vec![
            TaskLevel {
                level: 1,
                groups: vec![
                    HierarchicalTaskGroup {
                        id: "1".to_string(),
                        title: "Group 1".to_string(),
                        tasks: vec![],
                        sub_groups: vec![],
                    }
                ],
            }
        ];

        // Create source levels to merge
        let source_levels = vec![
            TaskLevel {
                level: 1,
                groups: vec![
                    HierarchicalTaskGroup {
                        id: "2".to_string(),
                        title: "Group 2".to_string(),
                        tasks: vec![],
                        sub_groups: vec![],
                    }
                ],
            },
            TaskLevel {
                level: 2,
                groups: vec![
                    HierarchicalTaskGroup {
                        id: "2.1".to_string(),
                        title: "Group 2.1".to_string(),
                        tasks: vec![],
                        sub_groups: vec![],
                    }
                ],
            }
        ];

        divider.merge_hierarchy_levels(&mut target_levels, source_levels).unwrap();

        // Should have 2 levels now
        assert_eq!(target_levels.len(), 2);
        
        // Level 1 should have 2 groups
        let level_1 = target_levels.iter().find(|l| l.level == 1).unwrap();
        assert_eq!(level_1.groups.len(), 2);
        
        // Level 2 should have 1 group
        let level_2 = target_levels.iter().find(|l| l.level == 2).unwrap();
        assert_eq!(level_2.groups.len(), 1);
    }

    #[test]
    fn test_create_hierarchy_basic_structure() {
        let divider = HierarchicalTaskDivider::new(4, 7).unwrap();
        let content_triples = create_test_content_triples(35);
        
        let hierarchy = divider.create_hierarchy(content_triples).unwrap();
        assert_eq!(hierarchy.total_tasks, 35);
        assert!(!hierarchy.levels.is_empty());
        
        // Should have at least one level
        let first_level = &hierarchy.levels[0];
        assert_eq!(first_level.level, 1);
        assert!(!first_level.groups.is_empty());
    }

    #[test]
    fn test_create_hierarchy_small_dataset() {
        let divider = HierarchicalTaskDivider::new(2, 3).unwrap();
        let content_triples = create_test_content_triples(5);
        
        let hierarchy = divider.create_hierarchy(content_triples).unwrap();
        assert_eq!(hierarchy.total_tasks, 5);
        
        // Verify structure
        assert!(!hierarchy.levels.is_empty());
        let first_level = &hierarchy.levels[0];
        assert_eq!(first_level.level, 1);
        
        // Should distribute across groups
        let total_groups = first_level.groups.len();
        assert!(total_groups > 0);
        assert!(total_groups <= 3); // Should not exceed groups_per_level
    }

    #[test]
    fn test_analysis_task_structure() {
        let divider = HierarchicalTaskDivider::new(1, 2).unwrap(); // Single level for simplicity
        let content_triples = create_test_content_triples(3);
        
        let hierarchy = divider.create_hierarchy(content_triples).unwrap();
        
        // Find a leaf group with tasks
        let mut found_tasks = false;
        for level in &hierarchy.levels {
            for group in &level.groups {
                if !group.tasks.is_empty() {
                    found_tasks = true;
                    let task = &group.tasks[0];
                    
                    // Verify task structure
                    assert!(!task.id.is_empty());
                    assert_eq!(task.analysis_stages.len(), 4);
                    assert!(task.content_files.content_a.to_string_lossy().contains("content.txt"));
                    assert!(task.content_files.content_b.to_string_lossy().contains("L1.txt"));
                    assert!(task.content_files.content_c.to_string_lossy().contains("L2.txt"));
                    break;
                }
            }
            if found_tasks {
                break;
            }
        }
        
        assert!(found_tasks, "Should have found at least one analysis task");
    }

    #[test]
    fn test_analysis_stages() {
        use AnalysisStage::*;
        
        let stages = vec![AnalyzeA, AnalyzeAInContextB, AnalyzeBInContextC, AnalyzeAInContextBC];
        assert_eq!(stages.len(), 4);
        
        // Test serialization
        let serialized = serde_json::to_string(&stages).unwrap();
        let deserialized: Vec<AnalysisStage> = serde_json::from_str(&serialized).unwrap();
        assert_eq!(stages.len(), deserialized.len());
    }

    #[test]
    fn test_mathematical_distribution_35_rows() {
        // Test case: 35 rows with 7 groups per level, 4 levels
        let divider = HierarchicalTaskDivider::new(4, 7).unwrap();
        let content_triples = create_test_content_triples(35);
        
        let hierarchy = divider.create_hierarchy(content_triples).unwrap();
        assert_eq!(hierarchy.total_tasks, 35);
        
        // Verify first level has proper distribution
        assert!(!hierarchy.levels.is_empty());
        let first_level = &hierarchy.levels[0];
        assert_eq!(first_level.level, 1);
        
        // 35 ÷ 7 = 5 remainder 0, so each group should have 5 items
        let total_items_in_groups: usize = first_level.groups.iter()
            .map(|g| count_items_in_group(g))
            .sum();
        assert_eq!(total_items_in_groups, 35);
    }

    #[test]
    fn test_mathematical_distribution_100_rows() {
        // Test case: 100 rows with 7 groups per level, 4 levels
        let divider = HierarchicalTaskDivider::new(4, 7).unwrap();
        let content_triples = create_test_content_triples(100);
        
        let hierarchy = divider.create_hierarchy(content_triples).unwrap();
        assert_eq!(hierarchy.total_tasks, 100);
        
        // Verify distribution
        assert!(!hierarchy.levels.is_empty());
        let first_level = &hierarchy.levels[0];
        
        // 100 ÷ 7 = 14 remainder 2
        // First 2 groups should have 15 items, remaining 5 groups should have 14 items
        let group_sizes: Vec<usize> = first_level.groups.iter()
            .map(|g| count_items_in_group(g))
            .collect();
        
        // Verify total items
        let total_items: usize = group_sizes.iter().sum();
        assert_eq!(total_items, 100);
        
        // Verify remainder distribution (first groups get extra items)
        if group_sizes.len() >= 2 {
            assert!(group_sizes[0] >= group_sizes[2]); // First group >= third group
            assert!(group_sizes[1] >= group_sizes[2]); // Second group >= third group
        }
    }

    #[test]
    fn test_mathematical_distribution_1000_rows() {
        // Test case: 1000 rows with 7 groups per level, 4 levels
        let divider = HierarchicalTaskDivider::new(4, 7).unwrap();
        let content_triples = create_test_content_triples(1000);
        
        let hierarchy = divider.create_hierarchy(content_triples).unwrap();
        assert_eq!(hierarchy.total_tasks, 1000);
        
        // Verify distribution
        assert!(!hierarchy.levels.is_empty());
        let first_level = &hierarchy.levels[0];
        
        // 1000 ÷ 7 = 142 remainder 6
        // First 6 groups should have 143 items, last group should have 142 items
        let group_sizes: Vec<usize> = first_level.groups.iter()
            .map(|g| count_items_in_group(g))
            .collect();
        
        // Verify total items
        let total_items: usize = group_sizes.iter().sum();
        assert_eq!(total_items, 1000);
        
        // Verify remainder distribution
        if group_sizes.len() == 7 {
            // First 6 groups should have 143 items each
            for i in 0..6 {
                assert_eq!(group_sizes[i], 143);
            }
            // Last group should have 142 items
            assert_eq!(group_sizes[6], 142);
        }
    }

    #[test]
    fn test_groups_per_level_constraint() {
        // Test that we never exceed groups_per_level at any level
        let divider = HierarchicalTaskDivider::new(3, 5).unwrap(); // 3 levels, 5 groups per level
        let content_triples = create_test_content_triples(73); // Prime number for interesting distribution
        
        let hierarchy = divider.create_hierarchy(content_triples).unwrap();
        
        // Check each level respects the constraint
        for level in &hierarchy.levels {
            assert!(level.groups.len() <= 5, "Level {} has {} groups, expected <= 5", level.level, level.groups.len());
            
            // Check sub-groups recursively
            for group in &level.groups {
                check_group_constraint(group, 5);
            }
        }
    }

    #[test]
    fn test_remainder_distribution_edge_cases() {
        let divider = HierarchicalTaskDivider::new(2, 7).unwrap();
        
        // Test with exactly divisible number
        let content_triples = create_test_content_triples(14); // 14 ÷ 7 = 2 remainder 0
        let hierarchy = divider.create_hierarchy(content_triples).unwrap();
        
        if let Some(first_level) = hierarchy.levels.first() {
            let group_sizes: Vec<usize> = first_level.groups.iter()
                .map(|g| count_items_in_group(g))
                .collect();
            
            // All groups should have equal size (2 items each)
            for size in &group_sizes {
                assert_eq!(*size, 2);
            }
        }
        
        // Test with remainder
        let content_triples = create_test_content_triples(16); // 16 ÷ 7 = 2 remainder 2
        let hierarchy = divider.create_hierarchy(content_triples).unwrap();
        
        if let Some(first_level) = hierarchy.levels.first() {
            let group_sizes: Vec<usize> = first_level.groups.iter()
                .map(|g| count_items_in_group(g))
                .collect();
            
            // First 2 groups should have 3 items, remaining should have 2 items
            if group_sizes.len() >= 3 {
                assert_eq!(group_sizes[0], 3); // First group gets extra
                assert_eq!(group_sizes[1], 3); // Second group gets extra
                assert_eq!(group_sizes[2], 2); // Third group gets base amount
            }
        }
    }

    // Helper function to count items in a group recursively
    fn count_items_in_group(group: &HierarchicalTaskGroup) -> usize {
        if !group.tasks.is_empty() {
            // Leaf group - count tasks
            group.tasks.len()
        } else {
            // Intermediate group - count items in sub-groups
            group.sub_groups.iter()
                .map(|sub_group| count_items_in_group(sub_group))
                .sum()
        }
    }

    // Helper function to check group constraint recursively
    fn check_group_constraint(group: &HierarchicalTaskGroup, max_groups: usize) {
        assert!(group.sub_groups.len() <= max_groups, 
                "Group {} has {} sub-groups, expected <= {}", 
                group.id, group.sub_groups.len(), max_groups);
        
        for sub_group in &group.sub_groups {
            check_group_constraint(sub_group, max_groups);
        }
    }

    #[test]
    fn test_hierarchical_id_generation() {
        let divider = HierarchicalTaskDivider::new(3, 2).unwrap(); // 3 levels, 2 groups per level
        let content_triples = create_test_content_triples(8); // 8 items for interesting distribution
        
        let hierarchy = divider.create_hierarchy(content_triples).unwrap();
        
        // Collect all IDs from the hierarchy
        let mut all_ids = std::collections::HashSet::new();
        collect_all_ids(&hierarchy.levels, &mut all_ids);
        
        // Verify ID format and uniqueness
        for id in &all_ids {
            // Check ID format (should be numbers separated by dots)
            let parts: Vec<&str> = id.split('.').collect();
            assert!(!parts.is_empty(), "ID should not be empty: {}", id);
            assert!(parts.len() <= 4, "ID should have at most 4 levels: {}", id); // 3 levels + task number
            
            // Each part should be a positive number
            for part in parts {
                let num: usize = part.parse().expect(&format!("ID part should be a number: {}", part));
                assert!(num > 0, "ID parts should be positive: {}", id);
            }
        }
        
        // Verify uniqueness (set size should equal number of unique IDs)
        let total_ids = count_total_ids(&hierarchy.levels);
        assert_eq!(all_ids.len(), total_ids, "All IDs should be unique");
    }

    #[test]
    fn test_hierarchical_id_format_compliance() {
        let divider = HierarchicalTaskDivider::new(4, 7).unwrap(); // Standard 4 levels, 7 groups
        let content_triples = create_test_content_triples(35);
        
        let hierarchy = divider.create_hierarchy(content_triples).unwrap();
        
        // Check that level 1 groups have IDs like "1", "2", "3", etc.
        if let Some(level1) = hierarchy.levels.first() {
            for (idx, group) in level1.groups.iter().enumerate() {
                let expected_id = (idx + 1).to_string();
                assert_eq!(group.id, expected_id, "Level 1 group ID should be {}", expected_id);
            }
        }
        
        // Verify hierarchical structure in IDs
        verify_hierarchical_id_structure(&hierarchy.levels, "");
    }

    #[test]
    fn test_task_id_uniqueness() {
        let divider = HierarchicalTaskDivider::new(4, 7).unwrap();
        let content_triples = create_test_content_triples(100);
        
        let hierarchy = divider.create_hierarchy(content_triples).unwrap();
        
        // Collect all task IDs
        let mut task_ids = std::collections::HashSet::new();
        collect_all_task_ids(&hierarchy.levels, &mut task_ids);
        
        // Verify uniqueness
        let total_tasks = count_total_tasks(&hierarchy.levels);
        assert_eq!(task_ids.len(), total_tasks, "All task IDs should be unique");
        
        // Verify task ID format (should be group_id.task_number)
        for task_id in &task_ids {
            let parts: Vec<&str> = task_id.split('.').collect();
            assert!(parts.len() >= 2, "Task ID should have at least 2 parts: {}", task_id);
            
            // Last part should be the task number within the group
            let task_num: usize = parts.last().unwrap().parse()
                .expect(&format!("Last part of task ID should be a number: {}", task_id));
            assert!(task_num > 0, "Task number should be positive: {}", task_id);
        }
    }

    #[test]
    fn test_parent_child_relationships() {
        let divider = HierarchicalTaskDivider::new(3, 3).unwrap(); // 3 levels, 3 groups per level
        let content_triples = create_test_content_triples(18); // 18 items for good distribution
        
        let hierarchy = divider.create_hierarchy(content_triples).unwrap();
        
        // Verify parent-child ID relationships
        verify_parent_child_relationships(&hierarchy.levels);
    }

    #[test]
    fn test_hierarchy_integrity() {
        let divider = HierarchicalTaskDivider::new(4, 5).unwrap();
        let content_triples = create_test_content_triples(200);
        
        let hierarchy = divider.create_hierarchy(content_triples).unwrap();
        
        // Verify hierarchy integrity
        assert_eq!(hierarchy.total_tasks, 200);
        
        // Count tasks at leaf level should equal total tasks
        let leaf_task_count = count_leaf_tasks(&hierarchy.levels);
        assert_eq!(leaf_task_count, 200, "Leaf task count should match total tasks");
        
        // Verify no orphaned groups or tasks
        verify_no_orphaned_elements(&hierarchy.levels);
    }

    // Helper functions for testing

    fn collect_all_ids(levels: &[TaskLevel], ids: &mut std::collections::HashSet<String>) {
        for level in levels {
            for group in &level.groups {
                ids.insert(group.id.clone());
                collect_group_ids(group, ids);
            }
        }
    }

    fn collect_group_ids(group: &HierarchicalTaskGroup, ids: &mut std::collections::HashSet<String>) {
        for sub_group in &group.sub_groups {
            ids.insert(sub_group.id.clone());
            collect_group_ids(sub_group, ids);
        }
        
        for task in &group.tasks {
            ids.insert(task.id.clone());
        }
    }

    fn count_total_ids(levels: &[TaskLevel]) -> usize {
        let mut count = 0;
        for level in levels {
            for group in &level.groups {
                count += 1; // Count the group itself
                count += count_group_ids(group);
            }
        }
        count
    }

    fn count_group_ids(group: &HierarchicalTaskGroup) -> usize {
        let mut count = 0;
        for sub_group in &group.sub_groups {
            count += 1; // Count the sub-group
            count += count_group_ids(sub_group);
        }
        count += group.tasks.len(); // Count tasks
        count
    }

    fn collect_all_task_ids(levels: &[TaskLevel], task_ids: &mut std::collections::HashSet<String>) {
        for level in levels {
            for group in &level.groups {
                collect_task_ids_from_group(group, task_ids);
            }
        }
    }

    fn collect_task_ids_from_group(group: &HierarchicalTaskGroup, task_ids: &mut std::collections::HashSet<String>) {
        for task in &group.tasks {
            task_ids.insert(task.id.clone());
        }
        
        for sub_group in &group.sub_groups {
            collect_task_ids_from_group(sub_group, task_ids);
        }
    }

    fn count_total_tasks(levels: &[TaskLevel]) -> usize {
        let mut count = 0;
        for level in levels {
            for group in &level.groups {
                count += count_tasks_in_group(group);
            }
        }
        count
    }

    fn count_tasks_in_group(group: &HierarchicalTaskGroup) -> usize {
        let mut count = group.tasks.len();
        for sub_group in &group.sub_groups {
            count += count_tasks_in_group(sub_group);
        }
        count
    }

    fn verify_hierarchical_id_structure(levels: &[TaskLevel], parent_prefix: &str) {
        for level in levels {
            for (idx, group) in level.groups.iter().enumerate() {
                let expected_prefix = if parent_prefix.is_empty() {
                    (idx + 1).to_string()
                } else {
                    format!("{}.{}", parent_prefix, idx + 1)
                };
                
                assert!(group.id.starts_with(&expected_prefix), 
                        "Group ID {} should start with {}", group.id, expected_prefix);
                
                // Recursively verify sub-groups
                verify_sub_group_id_structure(&group.sub_groups, &group.id);
            }
        }
    }

    fn verify_sub_group_id_structure(sub_groups: &[HierarchicalTaskGroup], parent_id: &str) {
        for (idx, sub_group) in sub_groups.iter().enumerate() {
            let expected_id = format!("{}.{}", parent_id, idx + 1);
            assert_eq!(sub_group.id, expected_id, 
                      "Sub-group ID should be {}", expected_id);
            
            // Recursively verify nested sub-groups
            verify_sub_group_id_structure(&sub_group.sub_groups, &sub_group.id);
        }
    }

    fn verify_parent_child_relationships(levels: &[TaskLevel]) {
        for level in levels {
            for group in &level.groups {
                verify_group_parent_child_relationships(group);
            }
        }
    }

    fn verify_group_parent_child_relationships(group: &HierarchicalTaskGroup) {
        // Verify that all sub-groups have IDs that start with this group's ID
        for sub_group in &group.sub_groups {
            assert!(sub_group.id.starts_with(&group.id), 
                   "Sub-group ID {} should start with parent ID {}", 
                   sub_group.id, group.id);
            
            // Verify the sub-group ID is exactly parent_id + "." + number
            let expected_prefix = format!("{}.", group.id);
            assert!(sub_group.id.starts_with(&expected_prefix),
                   "Sub-group ID {} should have format {}<number>", 
                   sub_group.id, expected_prefix);
            
            // Recursively verify nested relationships
            verify_group_parent_child_relationships(sub_group);
        }
        
        // Verify that all tasks have IDs that start with this group's ID
        for task in &group.tasks {
            assert!(task.id.starts_with(&group.id),
                   "Task ID {} should start with group ID {}", 
                   task.id, group.id);
        }
    }

    fn count_leaf_tasks(levels: &[TaskLevel]) -> usize {
        let mut count = 0;
        for level in levels {
            for group in &level.groups {
                count += count_leaf_tasks_in_group(group);
            }
        }
        count
    }

    fn count_leaf_tasks_in_group(group: &HierarchicalTaskGroup) -> usize {
        if !group.tasks.is_empty() {
            // This is a leaf group with actual tasks
            group.tasks.len()
        } else {
            // This is an intermediate group, count tasks in sub-groups
            let mut count = 0;
            for sub_group in &group.sub_groups {
                count += count_leaf_tasks_in_group(sub_group);
            }
            count
        }
    }

    fn verify_no_orphaned_elements(levels: &[TaskLevel]) {
        for level in levels {
            for group in &level.groups {
                verify_no_orphaned_elements_in_group(group);
            }
        }
    }

    fn verify_no_orphaned_elements_in_group(group: &HierarchicalTaskGroup) {
        // A group should either have tasks OR sub-groups, but not both
        if !group.tasks.is_empty() {
            assert!(group.sub_groups.is_empty(), 
                   "Leaf group {} should not have sub-groups", group.id);
        } else if !group.sub_groups.is_empty() {
            // Intermediate group should have sub-groups
            for sub_group in &group.sub_groups {
                verify_no_orphaned_elements_in_group(sub_group);
            }
        } else {
            // Empty group is not allowed
            panic!("Group {} should have either tasks or sub-groups", group.id);
        }
    }
}
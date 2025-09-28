//! Hierarchical task division for creating 4-level task hierarchies
//! 
//! This module implements the hierarchical task division algorithm that creates
//! structured task hierarchies with configurable levels and groups per level.
//! The default configuration creates 4 levels with 7 groups per level.

use crate::error::{TaskError, TaskResult};
use crate::tasks::content_extractor::ContentTriple;
use serde::{Deserialize, Serialize};

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
            });
        }
        
        if groups_per_level == 0 {
            return Err(TaskError::InvalidTaskConfiguration {
                cause: "Groups per level must be greater than 0".to_string(),
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
        let levels = self.distribute_across_levels(content_triples, 1)?;

        Ok(TaskHierarchy {
            levels,
            total_tasks,
        })
    }

    /// Distribute content triples across hierarchy levels recursively
    fn distribute_across_levels(
        &self,
        items: Vec<ContentTriple>,
        current_level: usize,
    ) -> TaskResult<Vec<TaskLevel>> {
        if items.is_empty() {
            return Ok(vec![]);
        }

        // If we've reached the maximum level, create leaf tasks
        if current_level >= self.levels {
            let groups = self.create_leaf_groups(items, current_level)?;
            return Ok(vec![TaskLevel {
                level: current_level,
                groups,
            }]);
        }

        // Calculate items per group at this level
        let items_per_group = (items.len() + self.groups_per_level - 1) / self.groups_per_level;
        let mut groups = Vec::new();
        let mut item_index = 0;

        for group_id in 1..=self.groups_per_level {
            let end_index = std::cmp::min(item_index + items_per_group, items.len());
            
            if item_index >= items.len() {
                break;
            }

            let group_items: Vec<ContentTriple> = items[item_index..end_index].to_vec();
            
            if group_items.is_empty() {
                break;
            }

            // Create hierarchical task group
            let group = if current_level < self.levels - 1 {
                // Intermediate level - create sub-groups
                let sub_levels = self.distribute_across_levels(group_items, current_level + 1)?;
                let sub_groups = if let Some(next_level) = sub_levels.first() {
                    next_level.groups.clone()
                } else {
                    vec![]
                };

                HierarchicalTaskGroup {
                    id: group_id.to_string(),
                    title: format!("Task Group {} (Level {})", group_id, current_level),
                    tasks: vec![], // Intermediate groups don't have direct tasks
                    sub_groups,
                }
            } else {
                // Leaf level - create actual analysis tasks
                let tasks = group_items
                    .into_iter()
                    .enumerate()
                    .map(|(idx, content_triple)| AnalysisTask {
                        id: format!("{}.{}", group_id, idx + 1),
                        table_name: "UNKNOWN".to_string(), // Will be set by caller
                        row_number: content_triple.row_number,
                        content_files: content_triple,
                        prompt_file: std::path::PathBuf::from(".kiro/steering/spec-S04-steering-doc-analysis.md"),
                        output_file: std::path::PathBuf::from(format!("gringotts/WorkArea/task_{}.md", group_id)),
                        analysis_stages: vec![
                            AnalysisStage::AnalyzeA,
                            AnalysisStage::AnalyzeAInContextB,
                            AnalysisStage::AnalyzeBInContextC,
                            AnalysisStage::AnalyzeAInContextBC,
                        ],
                    })
                    .collect();

                HierarchicalTaskGroup {
                    id: group_id.to_string(),
                    title: format!("Analysis Group {} (Level {})", group_id, current_level),
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

    /// Create leaf groups for the final level
    fn create_leaf_groups(
        &self,
        items: Vec<ContentTriple>,
        level: usize,
    ) -> TaskResult<Vec<HierarchicalTaskGroup>> {
        let items_per_group = (items.len() + self.groups_per_level - 1) / self.groups_per_level;
        let mut groups = Vec::new();
        let mut item_index = 0;

        for group_id in 1..=self.groups_per_level {
            let end_index = std::cmp::min(item_index + items_per_group, items.len());
            
            if item_index >= items.len() {
                break;
            }

            let group_items: Vec<ContentTriple> = items[item_index..end_index].to_vec();
            
            if group_items.is_empty() {
                break;
            }

            let tasks = group_items
                .into_iter()
                .enumerate()
                .map(|(idx, content_triple)| AnalysisTask {
                    id: format!("{}.{}", group_id, idx + 1),
                    table_name: "UNKNOWN".to_string(),
                    row_number: content_triple.row_number,
                    content_files: content_triple,
                    prompt_file: std::path::PathBuf::from(".kiro/steering/spec-S04-steering-doc-analysis.md"),
                    output_file: std::path::PathBuf::from(format!("gringotts/WorkArea/task_{}_{}.md", group_id, idx + 1)),
                    analysis_stages: vec![
                        AnalysisStage::AnalyzeA,
                        AnalysisStage::AnalyzeAInContextB,
                        AnalysisStage::AnalyzeBInContextC,
                        AnalysisStage::AnalyzeAInContextBC,
                    ],
                })
                .collect();

            let group = HierarchicalTaskGroup {
                id: group_id.to_string(),
                title: format!("Analysis Group {}", group_id),
                tasks,
                sub_groups: vec![],
            };

            groups.push(group);
            item_index = end_index;
        }

        Ok(groups)
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
}
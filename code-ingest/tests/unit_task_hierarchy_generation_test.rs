//! Comprehensive unit tests for task hierarchy generation
//!
//! Tests cover different level/group combinations, mathematical distribution,
//! hierarchical ID generation, and edge cases as specified in task 9.1.

use code_ingest::tasks::hierarchical_task_divider::{
    HierarchicalTaskDivider, TaskHierarchy, HierarchicalTaskGroup
};
use code_ingest::tasks::content_extractor::ContentTriple;
use code_ingest::error::TaskError;
use std::path::PathBuf;

/// Create test content triples for hierarchy testing
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

/// Validate hierarchical ID format (e.g., "1", "1.2", "1.2.3", "1.2.3.4")
fn validate_hierarchical_id(id: &str, expected_depth: usize) -> bool {
    let parts: Vec<&str> = id.split('.').collect();
    if parts.len() != expected_depth {
        return false;
    }
    
    // Each part should be a positive integer
    parts.iter().all(|part| {
        part.parse::<u32>().map(|n| n > 0).unwrap_or(false)
    })
}

/// Count total tasks in a hierarchy
fn count_total_tasks(hierarchy: &TaskHierarchy) -> usize {
    hierarchy.levels.iter()
        .flat_map(|level| &level.groups)
        .map(|group| count_tasks_in_group(group))
        .sum()
}

/// Count tasks in a hierarchical group (recursive)
fn count_tasks_in_group(group: &HierarchicalTaskGroup) -> usize {
    group.tasks.len() + group.sub_groups.iter().map(count_tasks_in_group).sum::<usize>()
}

#[cfg(test)]
mod hierarchical_task_divider_creation_tests {
    use super::*;

    #[test]
    fn test_hierarchical_task_divider_creation_valid() {
        let divider = HierarchicalTaskDivider::new(4, 7).unwrap();
        assert_eq!(divider.levels, 4);
        assert_eq!(divider.groups_per_level, 7);
    }

    #[test]
    fn test_hierarchical_task_divider_creation_zero_levels() {
        let result = HierarchicalTaskDivider::new(0, 7);
        assert!(result.is_err());
        
        if let Err(TaskError::InvalidTaskConfiguration { cause, suggestion }) = result {
            assert!(cause.contains("Levels must be greater than 0"));
            assert!(suggestion.contains("Set levels to a value between 1 and 10"));
        } else {
            panic!("Expected InvalidTaskConfiguration error");
        }
    }

    #[test]
    fn test_hierarchical_task_divider_creation_zero_groups() {
        let result = HierarchicalTaskDivider::new(4, 0);
        assert!(result.is_err());
        
        if let Err(TaskError::InvalidTaskConfiguration { cause, suggestion }) = result {
            assert!(cause.contains("Groups per level must be greater than 0"));
            assert!(suggestion.contains("Set groups_per_level to a value between 1 and 20"));
        } else {
            panic!("Expected InvalidTaskConfiguration error");
        }
    }

    #[test]
    fn test_hierarchical_task_divider_default() {
        let divider = HierarchicalTaskDivider::default();
        assert_eq!(divider.levels, 4);
        assert_eq!(divider.groups_per_level, 7);
    }

    #[test]
    fn test_hierarchical_task_divider_extreme_values() {
        // Test with very large values
        let divider = HierarchicalTaskDivider::new(10, 100).unwrap();
        assert_eq!(divider.levels, 10);
        assert_eq!(divider.groups_per_level, 100);
        
        // Test with minimal values
        let divider = HierarchicalTaskDivider::new(1, 1).unwrap();
        assert_eq!(divider.levels, 1);
        assert_eq!(divider.groups_per_level, 1);
    }
}

#[cfg(test)]
mod hierarchy_creation_basic_tests {
    use super::*;

    #[test]
    fn test_create_hierarchy_empty_input() {
        let divider = HierarchicalTaskDivider::new(4, 7).unwrap();
        let content_triples = vec![];
        
        let hierarchy = divider.create_hierarchy(content_triples).unwrap();
        assert_eq!(hierarchy.levels.len(), 0);
        assert_eq!(hierarchy.total_tasks, 0);
    }

    #[test]
    fn test_create_hierarchy_single_item() {
        let divider = HierarchicalTaskDivider::new(2, 3).unwrap();
        let content_triples = create_test_content_triples(1);
        
        let hierarchy = divider.create_hierarchy(content_triples).unwrap();
        assert_eq!(hierarchy.total_tasks, 1);
        assert!(!hierarchy.levels.is_empty());
        
        // Should have created at least one level
        assert!(hierarchy.levels.len() >= 1);
        
        // Total tasks should match input
        assert_eq!(count_total_tasks(&hierarchy), 1);
    }

    #[test]
    fn test_create_hierarchy_perfect_distribution() {
        let divider = HierarchicalTaskDivider::new(2, 3).unwrap();
        let content_triples = create_test_content_triples(9); // 3^2 = 9 items for perfect distribution
        
        let hierarchy = divider.create_hierarchy(content_triples).unwrap();
        assert_eq!(hierarchy.total_tasks, 9);
        
        // Should have 2 levels
        assert_eq!(hierarchy.levels.len(), 2);
        
        // Total tasks should be preserved
        assert_eq!(count_total_tasks(&hierarchy), 9);
    }

    #[test]
    fn test_create_hierarchy_imperfect_distribution() {
        let divider = HierarchicalTaskDivider::new(2, 3).unwrap();
        let content_triples = create_test_content_triples(10); // Not perfectly divisible
        
        let hierarchy = divider.create_hierarchy(content_triples).unwrap();
        assert_eq!(hierarchy.total_tasks, 10);
        
        // Total tasks should be preserved
        assert_eq!(count_total_tasks(&hierarchy), 10);
    }
}

#[cfg(test)]
mod mathematical_distribution_tests {
    use super::*;

    #[test]
    fn test_mathematical_distribution_even_split() {
        let divider = HierarchicalTaskDivider::new(1, 4).unwrap();
        let content_triples = create_test_content_triples(12); // 12 / 4 = 3 items per group
        
        let hierarchy = divider.create_hierarchy(content_triples).unwrap();
        assert_eq!(hierarchy.total_tasks, 12);
        assert_eq!(hierarchy.levels.len(), 1);
        
        let level = &hierarchy.levels[0];
        assert_eq!(level.groups.len(), 4);
        
        // Each group should have exactly 3 tasks
        for group in &level.groups {
            assert_eq!(group.tasks.len(), 3);
        }
    }

    #[test]
    fn test_mathematical_distribution_with_remainder() {
        let divider = HierarchicalTaskDivider::new(1, 3).unwrap();
        let content_triples = create_test_content_triples(10); // 10 / 3 = 3 remainder 1
        
        let hierarchy = divider.create_hierarchy(content_triples).unwrap();
        assert_eq!(hierarchy.total_tasks, 10);
        assert_eq!(hierarchy.levels.len(), 1);
        
        let level = &hierarchy.levels[0];
        assert_eq!(level.groups.len(), 3);
        
        // First group should have 4 tasks (3 + 1 remainder)
        assert_eq!(level.groups[0].tasks.len(), 4);
        
        // Other groups should have 3 tasks each
        assert_eq!(level.groups[1].tasks.len(), 3);
        assert_eq!(level.groups[2].tasks.len(), 3);
        
        // Total should be preserved
        let total_tasks: usize = level.groups.iter().map(|g| g.tasks.len()).sum();
        assert_eq!(total_tasks, 10);
    }

    #[test]
    fn test_mathematical_distribution_large_remainder() {
        let divider = HierarchicalTaskDivider::new(1, 5).unwrap();
        let content_triples = create_test_content_triples(23); // 23 / 5 = 4 remainder 3
        
        let hierarchy = divider.create_hierarchy(content_triples).unwrap();
        assert_eq!(hierarchy.total_tasks, 23);
        
        let level = &hierarchy.levels[0];
        assert_eq!(level.groups.len(), 5);
        
        // First 3 groups should have 5 tasks each (4 + 1 remainder)
        for i in 0..3 {
            assert_eq!(level.groups[i].tasks.len(), 5);
        }
        
        // Last 2 groups should have 4 tasks each
        for i in 3..5 {
            assert_eq!(level.groups[i].tasks.len(), 4);
        }
        
        // Total should be preserved: 3*5 + 2*4 = 15 + 8 = 23
        let total_tasks: usize = level.groups.iter().map(|g| g.tasks.len()).sum();
        assert_eq!(total_tasks, 23);
    }

    #[test]
    fn test_mathematical_distribution_more_groups_than_items() {
        let divider = HierarchicalTaskDivider::new(1, 10).unwrap();
        let content_triples = create_test_content_triples(3); // Only 3 items for 10 groups
        
        let hierarchy = divider.create_hierarchy(content_triples).unwrap();
        assert_eq!(hierarchy.total_tasks, 3);
        
        let level = &hierarchy.levels[0];
        // Should only create 3 groups (one for each item)
        assert_eq!(level.groups.len(), 3);
        
        // Each group should have exactly 1 task
        for group in &level.groups {
            assert_eq!(group.tasks.len(), 1);
        }
    }
}

#[cfg(test)]
mod hierarchical_id_generation_tests {
    use super::*;

    #[test]
    fn test_hierarchical_id_single_level() {
        let divider = HierarchicalTaskDivider::new(1, 3).unwrap();
        let content_triples = create_test_content_triples(6);
        
        let hierarchy = divider.create_hierarchy(content_triples).unwrap();
        let level = &hierarchy.levels[0];
        
        // Groups should have IDs: "1", "2", "3"
        assert_eq!(level.groups[0].id, "1");
        assert_eq!(level.groups[1].id, "2");
        assert_eq!(level.groups[2].id, "3");
        
        // Validate ID format
        for group in &level.groups {
            assert!(validate_hierarchical_id(&group.id, 1));
        }
    }

    #[test]
    fn test_hierarchical_id_two_levels() {
        let divider = HierarchicalTaskDivider::new(2, 2).unwrap();
        let content_triples = create_test_content_triples(8); // 2^2 * 2 = 8 items
        
        let hierarchy = divider.create_hierarchy(content_triples).unwrap();
        
        // Should have 2 levels
        assert_eq!(hierarchy.levels.len(), 2);
        
        // Level 1 groups should have IDs: "1", "2"
        let level1 = &hierarchy.levels[0];
        assert_eq!(level1.groups[0].id, "1");
        assert_eq!(level1.groups[1].id, "2");
        
        // Validate level 1 ID format
        for group in &level1.groups {
            assert!(validate_hierarchical_id(&group.id, 1));
        }
        
        // Level 2 should have sub-groups with IDs like "1.1", "1.2", "2.1", "2.2"
        if hierarchy.levels.len() > 1 {
            let level2 = &hierarchy.levels[1];
            for group in &level2.groups {
                assert!(validate_hierarchical_id(&group.id, 2));
                assert!(group.id.contains('.'));
            }
        }
    }

    #[test]
    fn test_hierarchical_id_four_levels() {
        let divider = HierarchicalTaskDivider::new(4, 2).unwrap();
        let content_triples = create_test_content_triples(16); // 2^4 = 16 items for perfect distribution
        
        let hierarchy = divider.create_hierarchy(content_triples).unwrap();
        
        // Should create a 4-level hierarchy
        assert!(hierarchy.levels.len() <= 4); // May be fewer if items fit in fewer levels
        
        // Check that all group IDs have valid hierarchical format
        for level in &hierarchy.levels {
            for group in &level.groups {
                let depth = group.id.split('.').count();
                assert!(depth >= 1 && depth <= 4);
                assert!(validate_hierarchical_id(&group.id, depth));
            }
        }
    }

    #[test]
    fn test_task_id_generation() {
        let divider = HierarchicalTaskDivider::new(1, 2).unwrap();
        let content_triples = create_test_content_triples(4);
        
        let hierarchy = divider.create_hierarchy(content_triples).unwrap();
        let level = &hierarchy.levels[0];
        
        // Check task IDs within groups
        for (group_idx, group) in level.groups.iter().enumerate() {
            let expected_group_id = (group_idx + 1).to_string();
            assert_eq!(group.id, expected_group_id);
            
            for (task_idx, task) in group.tasks.iter().enumerate() {
                let expected_task_id = format!("{}.{}", expected_group_id, task_idx + 1);
                assert_eq!(task.id, expected_task_id);
            }
        }
    }
}

#[cfg(test)]
mod level_group_combination_tests {
    use super::*;

    #[test]
    fn test_levels_2_groups_3() {
        let divider = HierarchicalTaskDivider::new(2, 3).unwrap();
        let content_triples = create_test_content_triples(18); // 3^2 * 2 = 18 items
        
        let hierarchy = divider.create_hierarchy(content_triples).unwrap();
        assert_eq!(hierarchy.total_tasks, 18);
        
        // Should preserve all tasks
        assert_eq!(count_total_tasks(&hierarchy), 18);
    }

    #[test]
    fn test_levels_3_groups_5() {
        let divider = HierarchicalTaskDivider::new(3, 5).unwrap();
        let content_triples = create_test_content_triples(25);
        
        let hierarchy = divider.create_hierarchy(content_triples).unwrap();
        assert_eq!(hierarchy.total_tasks, 25);
        
        // Should preserve all tasks
        assert_eq!(count_total_tasks(&hierarchy), 25);
    }

    #[test]
    fn test_levels_1_groups_10() {
        let divider = HierarchicalTaskDivider::new(1, 10).unwrap();
        let content_triples = create_test_content_triples(50);
        
        let hierarchy = divider.create_hierarchy(content_triples).unwrap();
        assert_eq!(hierarchy.total_tasks, 50);
        assert_eq!(hierarchy.levels.len(), 1);
        
        let level = &hierarchy.levels[0];
        assert_eq!(level.groups.len(), 10);
        
        // Each group should have 5 tasks (50 / 10)
        for group in &level.groups {
            assert_eq!(group.tasks.len(), 5);
        }
    }

    #[test]
    fn test_levels_5_groups_2() {
        let divider = HierarchicalTaskDivider::new(5, 2).unwrap();
        let content_triples = create_test_content_triples(32); // 2^5 = 32 items
        
        let hierarchy = divider.create_hierarchy(content_triples).unwrap();
        assert_eq!(hierarchy.total_tasks, 32);
        
        // Should preserve all tasks
        assert_eq!(count_total_tasks(&hierarchy), 32);
    }

    #[test]
    fn test_default_configuration() {
        let divider = HierarchicalTaskDivider::default(); // 4 levels, 7 groups
        let content_triples = create_test_content_triples(49); // 7^2 = 49 for good distribution
        
        let hierarchy = divider.create_hierarchy(content_triples).unwrap();
        assert_eq!(hierarchy.total_tasks, 49);
        
        // Should preserve all tasks
        assert_eq!(count_total_tasks(&hierarchy), 49);
    }
}

#[cfg(test)]
mod task_metadata_tests {
    use super::*;

    #[test]
    fn test_analysis_task_creation() {
        let divider = HierarchicalTaskDivider::new(1, 2).unwrap();
        let content_triples = create_test_content_triples(4);
        
        let hierarchy = divider.create_hierarchy(content_triples).unwrap();
        let level = &hierarchy.levels[0];
        
        // Check that tasks have correct metadata
        for group in &level.groups {
            for task in &group.tasks {
                // Task should have valid ID
                assert!(!task.id.is_empty());
                
                // Task should have row number
                assert!(task.row_number > 0);
                
                // Task should have content files
                assert!(!task.content_files.content_a.as_os_str().is_empty());
                assert!(!task.content_files.content_b.as_os_str().is_empty());
                assert!(!task.content_files.content_c.as_os_str().is_empty());
                
                // Task should have prompt file
                assert!(!task.prompt_file.as_os_str().is_empty());
                
                // Task should have output file
                assert!(!task.output_file.as_os_str().is_empty());
                
                // Task should have analysis stages
                assert!(!task.analysis_stages.is_empty());
            }
        }
    }

    #[test]
    fn test_task_row_number_preservation() {
        let divider = HierarchicalTaskDivider::new(1, 3).unwrap();
        let content_triples = create_test_content_triples(9);
        
        let hierarchy = divider.create_hierarchy(content_triples).unwrap();
        
        // Collect all row numbers from tasks
        let mut row_numbers = Vec::new();
        for level in &hierarchy.levels {
            for group in &level.groups {
                for task in &group.tasks {
                    row_numbers.push(task.row_number);
                }
            }
        }
        
        // Sort row numbers
        row_numbers.sort();
        
        // Should have all row numbers from 1 to 9
        assert_eq!(row_numbers, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn test_task_content_file_paths() {
        let divider = HierarchicalTaskDivider::new(1, 2).unwrap();
        let content_triples = create_test_content_triples(2);
        
        let hierarchy = divider.create_hierarchy(content_triples).unwrap();
        let level = &hierarchy.levels[0];
        
        // Check that content file paths are preserved correctly
        for (group_idx, group) in level.groups.iter().enumerate() {
            for (task_idx, task) in group.tasks.iter().enumerate() {
                let expected_row = group_idx * (2 / level.groups.len()) + task_idx + 1;
                
                // Content files should match the original content triple
                let expected_content_a = format!("test_{}_content.txt", task.row_number);
                let expected_content_b = format!("test_{}_content_L1.txt", task.row_number);
                let expected_content_c = format!("test_{}_content_L2.txt", task.row_number);
                
                assert_eq!(task.content_files.content_a.to_string_lossy(), expected_content_a);
                assert_eq!(task.content_files.content_b.to_string_lossy(), expected_content_b);
                assert_eq!(task.content_files.content_c.to_string_lossy(), expected_content_c);
            }
        }
    }
}

#[cfg(test)]
mod edge_case_tests {
    use super::*;

    #[test]
    fn test_single_item_single_group() {
        let divider = HierarchicalTaskDivider::new(1, 1).unwrap();
        let content_triples = create_test_content_triples(1);
        
        let hierarchy = divider.create_hierarchy(content_triples).unwrap();
        assert_eq!(hierarchy.total_tasks, 1);
        assert_eq!(hierarchy.levels.len(), 1);
        
        let level = &hierarchy.levels[0];
        assert_eq!(level.groups.len(), 1);
        assert_eq!(level.groups[0].tasks.len(), 1);
        assert_eq!(level.groups[0].id, "1");
        assert_eq!(level.groups[0].tasks[0].id, "1.1");
    }

    #[test]
    fn test_many_levels_few_items() {
        let divider = HierarchicalTaskDivider::new(10, 2).unwrap();
        let content_triples = create_test_content_triples(3);
        
        let hierarchy = divider.create_hierarchy(content_triples).unwrap();
        assert_eq!(hierarchy.total_tasks, 3);
        
        // Should preserve all tasks regardless of deep hierarchy
        assert_eq!(count_total_tasks(&hierarchy), 3);
    }

    #[test]
    fn test_few_levels_many_items() {
        let divider = HierarchicalTaskDivider::new(1, 2).unwrap();
        let content_triples = create_test_content_triples(100);
        
        let hierarchy = divider.create_hierarchy(content_triples).unwrap();
        assert_eq!(hierarchy.total_tasks, 100);
        assert_eq!(hierarchy.levels.len(), 1);
        
        let level = &hierarchy.levels[0];
        assert_eq!(level.groups.len(), 2);
        
        // Should distribute 100 items across 2 groups: 50 each
        assert_eq!(level.groups[0].tasks.len(), 50);
        assert_eq!(level.groups[1].tasks.len(), 50);
    }

    #[test]
    fn test_large_hierarchy() {
        let divider = HierarchicalTaskDivider::new(3, 10).unwrap();
        let content_triples = create_test_content_triples(1000);
        
        let hierarchy = divider.create_hierarchy(content_triples).unwrap();
        assert_eq!(hierarchy.total_tasks, 1000);
        
        // Should preserve all tasks
        assert_eq!(count_total_tasks(&hierarchy), 1000);
    }
}

// Note: Memory estimation tests would require access to private methods
// These tests focus on public API functionality

#[cfg(test)]
mod performance_tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_hierarchy_creation_performance() {
        let divider = HierarchicalTaskDivider::new(4, 7).unwrap();
        let content_triples = create_test_content_triples(1000);
        
        let start = Instant::now();
        let hierarchy = divider.create_hierarchy(content_triples).unwrap();
        let elapsed = start.elapsed();
        
        // Should complete within reasonable time
        assert!(elapsed.as_millis() < 500, "Hierarchy creation took too long: {:?}", elapsed);
        
        assert_eq!(hierarchy.total_tasks, 1000);
        assert_eq!(count_total_tasks(&hierarchy), 1000);
    }

    #[test]
    fn test_hierarchy_creation_large_dataset() {
        let divider = HierarchicalTaskDivider::new(3, 5).unwrap();
        let content_triples = create_test_content_triples(5000);
        
        let start = Instant::now();
        let hierarchy = divider.create_hierarchy(content_triples).unwrap();
        let elapsed = start.elapsed();
        
        // Should complete within reasonable time for large dataset
        assert!(elapsed.as_millis() < 2000, "Large hierarchy creation took too long: {:?}", elapsed);
        
        assert_eq!(hierarchy.total_tasks, 5000);
        assert_eq!(count_total_tasks(&hierarchy), 5000);
    }
}
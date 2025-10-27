//! Example demonstrating the usage of the new task generation data models
//! 
//! This example shows how to:
//! 1. Create ingestion sources (Git and local folder)
//! 2. Set up generation configurations
//! 3. Create task hierarchies with chunked content
//! 4. Generate content file references

use code_ingest::tasks::models::{
    IngestionSource, ChunkMetadata, GenerationConfig, 
    TaskHierarchy, Task, ContentFileReference, ContentFileType
};
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Task Generation Data Models Example");
    println!("=====================================\n");

    // 1. Create different ingestion sources
    println!("1. Creating Ingestion Sources:");
    
    // Git repository source
    let git_source = IngestionSource::git_repository("https://github.com/BurntSushi/xsv");
    println!("   Git Source: {}", git_source.display_name());
    git_source.validate()?;
    
    // Local folder source
    let temp_dir = std::env::temp_dir();
    let local_source = IngestionSource::local_folder(&temp_dir);
    println!("   Local Source: {}", local_source.display_name());
    local_source.validate()?;
    
    println!("   ✅ All sources validated successfully\n");

    // 2. Create chunk metadata for a large file
    println!("2. Creating Chunk Metadata:");
    
    let file_path = PathBuf::from("src/large_file.rs");
    let chunk1 = ChunkMetadata::new(
        file_path.clone(),
        "large_file.rs".to_string(),
        1,
        1,
        300,
        "// This is chunk 1 content\nfn main() {\n    println!(\"Hello, world!\");\n}".to_string(),
    );
    
    println!("   Chunk ID: {}", chunk1.chunk_id);
    println!("   File: {}", chunk1.file_path.display());
    println!("   Lines: {}-{} ({} lines)", chunk1.start_line, chunk1.end_line, chunk1.line_count);
    println!("   Size: {} bytes", chunk1.size_bytes);
    println!("   Identifier: {}", chunk1.chunk_identifier());
    println!("   ✅ Chunk metadata created successfully\n");

    // 3. Create generation configuration
    println!("3. Creating Generation Configuration:");
    
    let config = GenerationConfig::new(
        "INGEST_20250928101039".to_string(),
        4,
        7,
        PathBuf::from("tasks.md"),
    )
    .with_chunking(300)
    .with_prompt_file(PathBuf::from(".kiro/steering/spec-S04-steering-doc-analysis.md"))
    .with_max_tasks(35);
    
    println!("   Table: {}", config.table_name);
    println!("   Effective Table: {}", config.effective_table_name());
    println!("   Levels: {}, Groups: {}", config.levels, config.groups);
    println!("   Chunking: {} (size: {:?})", config.enable_chunking, config.chunk_size);
    println!("   Output: {}", config.output_file.display());
    println!("   Work Area: {}", config.work_area_dir.display());
    config.validate()?;
    println!("   ✅ Configuration validated successfully\n");

    // 4. Create task hierarchy
    println!("4. Creating Task Hierarchy:");
    
    let mut hierarchy = TaskHierarchy::new(3, 5, 25);
    
    // Create multiple root tasks to match the expected structure
    for i in 1..=5 {
        let mut root_task = Task::new(
            format!("task_{}", i),
            format!("Analyze INGEST_20250928101039 group {}", i),
            i.to_string(),
            0,
        );
        
        // Add content files to the first task as an example
        if i == 1 {
            let content_file_a = ContentFileReference::new(
                PathBuf::from(".raw_data_202509/INGEST_20250928101039_1_Content.txt"),
                "A".to_string(),
                "Primary content file".to_string(),
                ContentFileType::Content,
            );
            
            let content_file_b = ContentFileReference::new(
                PathBuf::from(".raw_data_202509/INGEST_20250928101039_1_Content_L1.txt"),
                "B".to_string(),
                "L1 context file".to_string(),
                ContentFileType::L1Context,
            );
            
            let content_file_c = ContentFileReference::new(
                PathBuf::from(".raw_data_202509/INGEST_20250928101039_1_Content_L2.txt"),
                "C".to_string(),
                "L2 context file".to_string(),
                ContentFileType::L2Context,
            );
            
            root_task.add_content_file(content_file_a);
            root_task.add_content_file(content_file_b);
            root_task.add_content_file(content_file_c);
            
            root_task.set_prompt_file(PathBuf::from(".kiro/steering/spec-S04-steering-doc-analysis.md"));
            root_task.set_output_file(PathBuf::from("gringotts/WorkArea/INGEST_20250928101039_1.md"));
            
            // Add a subtask
            let subtask = Task::new(
                "task_1_1".to_string(),
                "Analyze core components".to_string(),
                "1.1".to_string(),
                1,
            );
            root_task.add_subtask(subtask);
        }
        
        hierarchy.add_root_task(root_task);
    }
    
    println!("   Hierarchy Levels: {}", hierarchy.levels);
    println!("   Groups per Level: {}", hierarchy.groups_per_level);
    println!("   Total Tasks: {}", hierarchy.total_tasks);
    println!("   Root Tasks: {}", hierarchy.root_tasks.len());
    
    hierarchy.validate()?;
    println!("   ✅ Task hierarchy validated successfully\n");

    // 5. Display task hierarchy statistics
    println!("5. Task Hierarchy Statistics:");
    
    let stats = hierarchy.statistics();
    println!("   Total Tasks: {}", stats.total_tasks);
    println!("   Leaf Tasks: {}", stats.leaf_tasks);
    println!("   Branch Tasks: {}", stats.branch_tasks);
    println!("   Max Depth: {}", stats.max_depth);
    println!("   Avg Children per Branch: {:.2}", stats.avg_children_per_branch);
    println!("   ✅ Statistics calculated successfully\n");

    // 6. Demonstrate task operations
    println!("6. Task Operations:");
    
    let task = &hierarchy.root_tasks[0];
    println!("   Task ID: {}", task.id);
    println!("   Description: {}", task.description);
    println!("   Task Number: {}", task.task_number);
    println!("   Level: {}", task.level);
    println!("   Content Files: {}", task.content_files.len());
    println!("   Subtasks: {}", task.subtasks.len());
    println!("   Is Leaf: {}", task.is_leaf());
    println!("   Total Subtasks: {}", task.total_subtasks());
    println!("   Completed: {}", task.completed);
    
    for (i, content_file) in task.content_files.iter().enumerate() {
        println!("     Content File {}: {} ({})", 
                 i + 1, 
                 content_file.role, 
                 content_file.file_type.display_name());
    }
    
    println!("   ✅ Task operations demonstrated successfully\n");

    // 7. Show content file paths for chunked processing
    println!("7. Content File Path Generation:");
    
    let base_dir = PathBuf::from(".raw_data_202509");
    let table_name = "INGEST_20250928101039_300";
    
    println!("   Content File: {}", 
             chunk1.content_file_path(&base_dir, table_name).display());
    println!("   L1 File: {}", 
             chunk1.l1_content_file_path(&base_dir, table_name).display());
    println!("   L2 File: {}", 
             chunk1.l2_content_file_path(&base_dir, table_name).display());
    println!("   ✅ File paths generated successfully\n");

    println!("🎉 All examples completed successfully!");
    println!("The core data models are ready for use in the task generation system.");
    
    Ok(())
}
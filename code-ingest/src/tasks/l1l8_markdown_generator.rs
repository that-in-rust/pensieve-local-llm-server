//! L1L8 Markdown Generator for Task List Generation
//!
//! This module provides the L1L8MarkdownGenerator struct that creates markdown files
//! with hierarchical task structures for L1-L8 analysis methodology.

use crate::error::{TaskError, TaskResult};
use crate::tasks::hierarchical_task_divider::{TaskHierarchy, HierarchicalTaskGroup, AnalysisTask, AnalysisStage};
use std::path::{Path, PathBuf};
use tracing::{debug, info};

/// L1L8 Markdown generator for creating analysis task files
#[derive(Clone, Debug)]
pub struct L1L8MarkdownGenerator {
    /// Path to the prompt file for analysis
    prompt_file: PathBuf,
    /// Output directory for generated markdown files
    output_dir: PathBuf,
}

impl L1L8MarkdownGenerator {
    /// Create a new L1L8MarkdownGenerator
    ///
    /// # Arguments
    /// * `prompt_file` - Path to the prompt file for analysis methodology
    /// * `output_dir` - Directory where markdown files will be created
    ///
    /// # Returns
    /// * `Self` - New L1L8MarkdownGenerator instance
    pub fn new(prompt_file: PathBuf, output_dir: PathBuf) -> Self {
        Self {
            prompt_file,
            output_dir,
        }
    }

    /// Generate hierarchical markdown for a complete task structure
    ///
    /// # Arguments
    /// * `hierarchy` - Task hierarchy to generate markdown for
    /// * `table_name` - Name of the source table
    ///
    /// # Returns
    /// * `TaskResult<String>` - Generated markdown content
    ///
    /// # Examples
    /// ```rust
    /// let generator = L1L8MarkdownGenerator::new(
    ///     PathBuf::from(".kiro/steering/spec-S04-steering-doc-analysis.md"),
    ///     PathBuf::from("gringotts/WorkArea")
    /// );
    /// let markdown = generator.generate_hierarchical_markdown(&hierarchy, "INGEST_20250928101039").await?;
    /// ```
    pub async fn generate_hierarchical_markdown(
        &self,
        hierarchy: &TaskHierarchy,
        table_name: &str,
    ) -> TaskResult<String> {
        debug!("Generating hierarchical markdown for table: {}", table_name);

        let mut markdown = String::new();

        // Add header
        markdown.push_str(&format!("# L1-L8 Analysis Tasks for {}\n\n", table_name));

        // Add metadata section
        self.add_metadata_section(&mut markdown, hierarchy, table_name)?;

        // Add L1-L8 methodology section
        self.add_l1l8_methodology_section(&mut markdown)?;

        // Add task hierarchy
        self.add_task_hierarchy(&mut markdown, hierarchy, table_name).await?;

        // Add footer with instructions
        self.add_footer_section(&mut markdown, table_name)?;

        info!("Generated hierarchical markdown with {} total tasks", hierarchy.total_tasks);
        Ok(markdown)
    }

    /// Create an individual analysis task in markdown format
    ///
    /// # Arguments
    /// * `task` - Analysis task to format
    /// * `table_name` - Name of the source table
    ///
    /// # Returns
    /// * `String` - Formatted task markdown
    pub fn create_analysis_task(&self, task: &AnalysisTask, table_name: &str) -> String {
        let mut task_md = String::new();

        // Task title with ID
        task_md.push_str(&format!("- [ ] {}. Analyze {} row {}\n", 
            task.id, table_name, task.row_number));

        // Content section with A/B/C file references
        task_md.push_str(&format!("  - **Content**: `{}` as A + `{}` as B + `{}` as C\n",
            task.content_files.content_a.display(),
            task.content_files.content_b.display(),
            task.content_files.content_c.display()
        ));

        // Prompt section with L1-L8 analysis instructions
        task_md.push_str(&format!("  - **Prompt**: `{}` where you try to find insights of A alone ; A in context of B ; B in context of C ; A in context B & C\n",
            self.prompt_file.display()
        ));

        // Output section
        let output_file = format!("gringotts/WorkArea/{}_{}.md", table_name, task.row_number);
        task_md.push_str(&format!("  - **Output**: `{}`\n", output_file));

        // Analysis stages
        task_md.push_str("  - **Analysis Stages**:\n");
        for stage in &task.analysis_stages {
            let stage_description = self.format_analysis_stage(stage);
            task_md.push_str(&format!("    - {}\n", stage_description));
        }

        task_md
    }

    /// Format L1-L8 analysis instructions for methodology integration
    ///
    /// # Returns
    /// * `String` - Formatted L1-L8 analysis instructions
    pub fn format_l1l8_analysis_instructions(&self) -> String {
        let mut instructions = String::new();

        instructions.push_str("## L1-L8 Analysis Methodology\n\n");
        instructions.push_str("This task structure implements the L1-L8 extraction hierarchy for systematic codebase analysis:\n\n");

        instructions.push_str("### Horizon 1: Tactical Implementation (The \"How\")\n");
        instructions.push_str("- **L1: Idiomatic Patterns & Micro-Optimizations**: Efficiency, bug reduction, raw performance, mechanical sympathy\n");
        instructions.push_str("- **L2: Design Patterns & Composition**: Abstraction boundaries, API ergonomics, RAII variants, advanced trait usage\n");
        instructions.push_str("- **L3: Micro-Library Opportunities**: High-utility components under ~2000 LOC\n\n");

        instructions.push_str("### Horizon 2: Strategic Architecture (The \"What\")\n");
        instructions.push_str("- **L4: Macro-Library & Platform Opportunities**: High-PMF ideas offering ecosystem dominance\n");
        instructions.push_str("- **L5: LLD Architecture Decisions & Invariants**: Concurrency models, state management, internal modularity\n");
        instructions.push_str("- **L6: Domain-Specific Architecture & Hardware Interaction**: Kernel bypass, GPU pipelines, OS abstractions\n\n");

        instructions.push_str("### Horizon 3: Foundational Evolution (The \"Future\" and \"Why\")\n");
        instructions.push_str("- **L7: Language Capability & Evolution**: Identifying limitations of Rust itself\n");
        instructions.push_str("- **L8: The Meta-Context**: The archaeology of intent from commit history and constraints\n\n");

        instructions.push_str("### Analysis Process\n");
        instructions.push_str("Each task follows a 4-stage analysis process:\n");
        instructions.push_str("1. **Analyze A alone**: Extract insights from the raw content\n");
        instructions.push_str("2. **A in context of B**: Understand A within its immediate file context (L1)\n");
        instructions.push_str("3. **B in context of C**: Understand the immediate context within architectural context (L2)\n");
        instructions.push_str("4. **A in context of B & C**: Synthesize insights across all contextual layers\n\n");

        instructions
    }

    /// Write hierarchical markdown to a file
    ///
    /// # Arguments
    /// * `hierarchy` - Task hierarchy to write
    /// * `table_name` - Name of the source table
    /// * `output_file` - Path to the output file
    ///
    /// # Returns
    /// * `TaskResult<()>` - Success or error
    pub async fn write_hierarchical_markdown_to_file(
        &self,
        hierarchy: &TaskHierarchy,
        table_name: &str,
        output_file: &Path,
    ) -> TaskResult<()> {
        debug!("Writing hierarchical markdown to file: {}", output_file.display());

        // Generate markdown content
        let markdown = self.generate_hierarchical_markdown(hierarchy, table_name).await?;

        // Ensure parent directory exists
        if let Some(parent) = output_file.parent() {
            tokio::fs::create_dir_all(parent).await.map_err(|e| {
                TaskError::TaskFileCreationFailed {
                    path: parent.display().to_string(),
                    cause: format!("Failed to create parent directory: {}", e),
                    suggestion: "Check directory permissions and available disk space".to_string(),
                    source: Some(Box::new(e)),
                }
            })?;
        }

        // Write markdown to file
        tokio::fs::write(output_file, markdown).await.map_err(|e| {
            TaskError::TaskFileCreationFailed {
                path: output_file.display().to_string(),
                cause: e.to_string(),
                suggestion: "Check file permissions and available disk space".to_string(),
                source: Some(Box::new(e)),
            }
        })?;

        info!("Successfully wrote hierarchical markdown to: {}", output_file.display());
        Ok(())
    }

    /// Add metadata section to markdown
    fn add_metadata_section(
        &self,
        markdown: &mut String,
        hierarchy: &TaskHierarchy,
        table_name: &str,
    ) -> TaskResult<()> {
        markdown.push_str("## Task Generation Metadata\n\n");
        markdown.push_str(&format!("- **Source Table**: `{}`\n", table_name));
        markdown.push_str(&format!("- **Total Tasks**: {}\n", hierarchy.total_tasks));
        markdown.push_str(&format!("- **Hierarchy Levels**: {}\n", hierarchy.levels.len()));
        markdown.push_str(&format!("- **Prompt File**: `{}`\n", self.prompt_file.display()));
        markdown.push_str(&format!("- **Output Directory**: `{}`\n", self.output_dir.display()));
        markdown.push_str(&format!("- **Generated At**: {}\n", chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")));
        markdown.push_str("\n");

        // Add level summary
        markdown.push_str("### Hierarchy Structure\n\n");
        for level in &hierarchy.levels {
            markdown.push_str(&format!("- **Level {}**: {} groups\n", level.level, level.groups.len()));
        }
        markdown.push_str("\n");

        Ok(())
    }

    /// Add L1-L8 methodology section to markdown
    fn add_l1l8_methodology_section(&self, markdown: &mut String) -> TaskResult<()> {
        let methodology = self.format_l1l8_analysis_instructions();
        markdown.push_str(&methodology);
        Ok(())
    }

    /// Add task hierarchy to markdown
    async fn add_task_hierarchy(
        &self,
        markdown: &mut String,
        hierarchy: &TaskHierarchy,
        table_name: &str,
    ) -> TaskResult<()> {
        markdown.push_str("## Task Hierarchy\n\n");

        // Process each level
        for level in &hierarchy.levels {
            markdown.push_str(&format!("### Level {} Groups\n\n", level.level));
            
            for group in &level.groups {
                self.add_hierarchical_group(markdown, group, table_name, 0).await?;
            }
            
            markdown.push_str("\n");
        }

        Ok(())
    }

    /// Add a hierarchical group to markdown recursively
    fn add_hierarchical_group<'a>(
        &'a self,
        markdown: &'a mut String,
        group: &'a HierarchicalTaskGroup,
        table_name: &'a str,
        indent_level: usize,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = TaskResult<()>> + 'a>> {
        Box::pin(async move {
            let indent = "  ".repeat(indent_level);

            // Add group header
            markdown.push_str(&format!("{}### {}\n\n", indent, group.title));

            // Add tasks if this is a leaf group
            if !group.tasks.is_empty() {
                for task in &group.tasks {
                    let task_markdown = self.create_analysis_task(task, table_name);
                    // Add proper indentation to each line
                    for line in task_markdown.lines() {
                        markdown.push_str(&format!("{}{}\n", indent, line));
                    }
                }
                markdown.push_str("\n");
            }

            // Add sub-groups recursively
            if !group.sub_groups.is_empty() {
                for sub_group in &group.sub_groups {
                    self.add_hierarchical_group(markdown, sub_group, table_name, indent_level + 1).await?;
                }
            }

            Ok(())
        })
    }

    /// Add footer section with processing instructions
    fn add_footer_section(&self, markdown: &mut String, table_name: &str) -> TaskResult<()> {
        markdown.push_str("## Processing Instructions\n\n");
        markdown.push_str("### How to Execute These Tasks\n\n");
        markdown.push_str("1. **Sequential Processing**: Work through tasks in hierarchical order (1.1.1.1, 1.1.1.2, etc.)\n");
        markdown.push_str("2. **Context Integration**: Use the A/B/C file structure for multi-layered analysis\n");
        markdown.push_str("3. **L1-L8 Methodology**: Apply the extraction hierarchy to identify patterns and insights\n");
        markdown.push_str("4. **Output Generation**: Store results in the specified WorkArea directory\n\n");

        markdown.push_str("### File Structure\n\n");
        markdown.push_str("```\n");
        markdown.push_str(".raw_data_202509/\n");
        markdown.push_str(&format!("├── {}_*_Content.txt     # A files (raw content)\n", table_name));
        markdown.push_str(&format!("├── {}_*_Content_L1.txt  # B files (L1 context)\n", table_name));
        markdown.push_str(&format!("└── {}_*_Content_L2.txt  # C files (L2 context)\n", table_name));
        markdown.push_str("\n");
        markdown.push_str("gringotts/WorkArea/\n");
        markdown.push_str(&format!("└── {}_{}.md            # Analysis outputs\n", table_name, "*"));
        markdown.push_str("```\n\n");

        markdown.push_str("### Analysis Commands\n\n");
        markdown.push_str("```bash\n");
        markdown.push_str("# Count rows in source table\n");
        markdown.push_str(&format!("code-ingest count-rows {}\n", table_name));
        markdown.push_str("\n# Extract content files\n");
        markdown.push_str(&format!("code-ingest chunk-level-task-generator {} --output-dir .raw_data_202509\n", table_name));
        markdown.push_str("\n# Generate hierarchical tasks\n");
        markdown.push_str(&format!("code-ingest generate-hierarchical-tasks {} --levels 4 --groups 7 --output {}_tasks.md\n", table_name, table_name));
        markdown.push_str("```\n\n");

        markdown.push_str("### Notes\n\n");
        markdown.push_str("- Each task represents systematic analysis of one database row\n");
        markdown.push_str("- The hierarchical structure enables parallel processing of task groups\n");
        markdown.push_str("- L1-L8 methodology ensures comprehensive knowledge extraction\n");
        markdown.push_str("- Output files follow consistent naming for easy reference\n\n");

        Ok(())
    }

    /// Format an analysis stage for display
    pub fn format_analysis_stage(&self, stage: &AnalysisStage) -> String {
        match stage {
            AnalysisStage::AnalyzeA => "Analyze A alone: Extract insights from raw content".to_string(),
            AnalysisStage::AnalyzeAInContextB => "A in context of B: Understand content within immediate file context".to_string(),
            AnalysisStage::AnalyzeBInContextC => "B in context of C: Understand immediate context within architectural context".to_string(),
            AnalysisStage::AnalyzeAInContextBC => "A in context of B & C: Synthesize insights across all contextual layers".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tasks::content_extractor::ContentTriple;
    use crate::tasks::hierarchical_task_divider::{HierarchicalTaskDivider, AnalysisStage};
    use std::path::PathBuf;

    fn create_test_hierarchy() -> TaskHierarchy {
        let divider = HierarchicalTaskDivider::new(2, 2).unwrap();
        let content_triples = vec![
            ContentTriple {
                content_a: PathBuf::from(".raw_data_202509/TEST_1_Content.txt"),
                content_b: PathBuf::from(".raw_data_202509/TEST_1_Content_L1.txt"),
                content_c: PathBuf::from(".raw_data_202509/TEST_1_Content_L2.txt"),
                row_number: 1,
                table_name: "TEST_TABLE".to_string(),
            },
            ContentTriple {
                content_a: PathBuf::from(".raw_data_202509/TEST_2_Content.txt"),
                content_b: PathBuf::from(".raw_data_202509/TEST_2_Content_L1.txt"),
                content_c: PathBuf::from(".raw_data_202509/TEST_2_Content_L2.txt"),
                row_number: 2,
                table_name: "TEST_TABLE".to_string(),
            },
        ];
        divider.create_hierarchy(content_triples).unwrap()
    }

    #[test]
    fn test_l1l8_markdown_generator_creation() {
        let generator = L1L8MarkdownGenerator::new(
            PathBuf::from(".kiro/steering/spec-S04-steering-doc-analysis.md"),
            PathBuf::from("gringotts/WorkArea"),
        );
        
        assert_eq!(generator.prompt_file, PathBuf::from(".kiro/steering/spec-S04-steering-doc-analysis.md"));
        assert_eq!(generator.output_dir, PathBuf::from("gringotts/WorkArea"));
    }

    #[tokio::test]
    async fn test_generate_hierarchical_markdown() {
        let generator = L1L8MarkdownGenerator::new(
            PathBuf::from(".kiro/steering/spec-S04-steering-doc-analysis.md"),
            PathBuf::from("gringotts/WorkArea"),
        );
        
        let hierarchy = create_test_hierarchy();
        let markdown = generator.generate_hierarchical_markdown(&hierarchy, "TEST_TABLE").await.unwrap();
        
        // Verify markdown structure
        assert!(markdown.contains("# L1-L8 Analysis Tasks for TEST_TABLE"));
        assert!(markdown.contains("## Task Generation Metadata"));
        assert!(markdown.contains("## L1-L8 Analysis Methodology"));
        assert!(markdown.contains("## Task Hierarchy"));
        assert!(markdown.contains("## Processing Instructions"));
        
        // Verify metadata content
        assert!(markdown.contains("**Source Table**: `TEST_TABLE`"));
        assert!(markdown.contains("**Total Tasks**: 2"));
        
        // Verify methodology content
        assert!(markdown.contains("L1: Idiomatic Patterns"));
        assert!(markdown.contains("L8: The Meta-Context"));
        
        // Verify task content
        assert!(markdown.contains("Analyze TEST_TABLE row"));
        assert!(markdown.contains("Content**: `.raw_data_202509/TEST_"));
        assert!(markdown.contains("Output**: `gringotts/WorkArea/TEST_TABLE_"));
    }

    #[test]
    fn test_create_analysis_task() {
        let generator = L1L8MarkdownGenerator::new(
            PathBuf::from(".kiro/steering/spec-S04-steering-doc-analysis.md"),
            PathBuf::from("gringotts/WorkArea"),
        );
        
        let task = AnalysisTask {
            id: "1.1.1.1".to_string(),
            table_name: "TEST_TABLE".to_string(),
            row_number: 5,
            content_files: ContentTriple {
                content_a: PathBuf::from(".raw_data_202509/TEST_5_Content.txt"),
                content_b: PathBuf::from(".raw_data_202509/TEST_5_Content_L1.txt"),
                content_c: PathBuf::from(".raw_data_202509/TEST_5_Content_L2.txt"),
                row_number: 5,
                table_name: "TEST_TABLE".to_string(),
            },
            prompt_file: PathBuf::from(".kiro/steering/spec-S04-steering-doc-analysis.md"),
            output_file: PathBuf::from("gringotts/WorkArea/TEST_TABLE_5.md"),
            analysis_stages: vec![
                AnalysisStage::AnalyzeA,
                AnalysisStage::AnalyzeAInContextB,
                AnalysisStage::AnalyzeBInContextC,
                AnalysisStage::AnalyzeAInContextBC,
            ],
        };
        
        let task_markdown = generator.create_analysis_task(&task, "TEST_TABLE");
        
        // Verify task format
        assert!(task_markdown.contains("- [ ] 1.1.1.1. Analyze TEST_TABLE row 5"));
        assert!(task_markdown.contains("**Content**: `.raw_data_202509/TEST_5_Content.txt` as A"));
        assert!(task_markdown.contains("**Prompt**: `.kiro/steering/spec-S04-steering-doc-analysis.md`"));
        assert!(task_markdown.contains("**Output**: `gringotts/WorkArea/TEST_TABLE_5.md`"));
        assert!(task_markdown.contains("**Analysis Stages**:"));
        
        // Verify analysis stages
        assert!(task_markdown.contains("Analyze A alone"));
        assert!(task_markdown.contains("A in context of B"));
        assert!(task_markdown.contains("B in context of C"));
        assert!(task_markdown.contains("A in context of B & C"));
    }

    #[test]
    fn test_format_l1l8_analysis_instructions() {
        let generator = L1L8MarkdownGenerator::new(
            PathBuf::from(".kiro/steering/spec-S04-steering-doc-analysis.md"),
            PathBuf::from("gringotts/WorkArea"),
        );
        
        let instructions = generator.format_l1l8_analysis_instructions();
        
        // Verify methodology structure
        assert!(instructions.contains("## L1-L8 Analysis Methodology"));
        assert!(instructions.contains("### Horizon 1: Tactical Implementation"));
        assert!(instructions.contains("### Horizon 2: Strategic Architecture"));
        assert!(instructions.contains("### Horizon 3: Foundational Evolution"));
        assert!(instructions.contains("### Analysis Process"));
        
        // Verify L1-L8 levels
        assert!(instructions.contains("L1: Idiomatic Patterns"));
        assert!(instructions.contains("L2: Design Patterns"));
        assert!(instructions.contains("L3: Micro-Library"));
        assert!(instructions.contains("L4: Macro-Library"));
        assert!(instructions.contains("L5: LLD Architecture"));
        assert!(instructions.contains("L6: Domain-Specific"));
        assert!(instructions.contains("L7: Language Capability"));
        assert!(instructions.contains("L8: The Meta-Context"));
        
        // Verify analysis process
        assert!(instructions.contains("1. **Analyze A alone**"));
        assert!(instructions.contains("2. **A in context of B**"));
        assert!(instructions.contains("3. **B in context of C**"));
        assert!(instructions.contains("4. **A in context of B & C**"));
    }

    #[test]
    fn test_format_analysis_stage() {
        let generator = L1L8MarkdownGenerator::new(
            PathBuf::from(".kiro/steering/spec-S04-steering-doc-analysis.md"),
            PathBuf::from("gringotts/WorkArea"),
        );
        
        // Test each analysis stage
        let stage_a = generator.format_analysis_stage(&AnalysisStage::AnalyzeA);
        assert!(stage_a.contains("Analyze A alone"));
        assert!(stage_a.contains("raw content"));
        
        let stage_ab = generator.format_analysis_stage(&AnalysisStage::AnalyzeAInContextB);
        assert!(stage_ab.contains("A in context of B"));
        assert!(stage_ab.contains("immediate file context"));
        
        let stage_bc = generator.format_analysis_stage(&AnalysisStage::AnalyzeBInContextC);
        assert!(stage_bc.contains("B in context of C"));
        assert!(stage_bc.contains("architectural context"));
        
        let stage_abc = generator.format_analysis_stage(&AnalysisStage::AnalyzeAInContextBC);
        assert!(stage_abc.contains("A in context of B & C"));
        assert!(stage_abc.contains("all contextual layers"));
    }

    #[tokio::test]
    async fn test_write_hierarchical_markdown_to_file() {
        let temp_dir = tempfile::tempdir().unwrap();
        let output_file = temp_dir.path().join("test_tasks.md");
        
        let generator = L1L8MarkdownGenerator::new(
            PathBuf::from(".kiro/steering/spec-S04-steering-doc-analysis.md"),
            PathBuf::from("gringotts/WorkArea"),
        );
        
        let hierarchy = create_test_hierarchy();
        
        // Write markdown to file
        generator.write_hierarchical_markdown_to_file(&hierarchy, "TEST_TABLE", &output_file).await.unwrap();
        
        // Verify file was created
        assert!(output_file.exists());
        
        // Verify file content
        let content = tokio::fs::read_to_string(&output_file).await.unwrap();
        assert!(content.contains("# L1-L8 Analysis Tasks for TEST_TABLE"));
        assert!(content.contains("**Total Tasks**: 2"));
    }

    #[test]
    fn test_task_format_compliance() {
        let generator = L1L8MarkdownGenerator::new(
            PathBuf::from(".kiro/steering/spec-S04-steering-doc-analysis.md"),
            PathBuf::from("gringotts/WorkArea"),
        );
        
        let task = AnalysisTask {
            id: "2.3.1.4".to_string(),
            table_name: "INGEST_20250928101039".to_string(),
            row_number: 35,
            content_files: ContentTriple {
                content_a: PathBuf::from(".raw_data_202509/INGEST_20250928101039_35_Content.txt"),
                content_b: PathBuf::from(".raw_data_202509/INGEST_20250928101039_35_Content_L1.txt"),
                content_c: PathBuf::from(".raw_data_202509/INGEST_20250928101039_35_Content_L2.txt"),
                row_number: 35,
                table_name: "INGEST_20250928101039".to_string(),
            },
            prompt_file: PathBuf::from(".kiro/steering/spec-S04-steering-doc-analysis.md"),
            output_file: PathBuf::from("gringotts/WorkArea/INGEST_20250928101039_35.md"),
            analysis_stages: vec![
                AnalysisStage::AnalyzeA,
                AnalysisStage::AnalyzeAInContextB,
                AnalysisStage::AnalyzeBInContextC,
                AnalysisStage::AnalyzeAInContextBC,
            ],
        };
        
        let task_markdown = generator.create_analysis_task(&task, "INGEST_20250928101039");
        
        // Verify exact format compliance with requirements
        assert!(task_markdown.contains("- [ ] 2.3.1.4. Analyze INGEST_20250928101039 row 35"));
        assert!(task_markdown.contains("**Content**: `.raw_data_202509/INGEST_20250928101039_35_Content.txt` as A + `.raw_data_202509/INGEST_20250928101039_35_Content_L1.txt` as B + `.raw_data_202509/INGEST_20250928101039_35_Content_L2.txt` as C"));
        assert!(task_markdown.contains("**Prompt**: `.kiro/steering/spec-S04-steering-doc-analysis.md` where you try to find insights of A alone ; A in context of B ; B in context of C ; A in context B & C"));
        assert!(task_markdown.contains("**Output**: `gringotts/WorkArea/INGEST_20250928101039_35.md`"));
    }
}
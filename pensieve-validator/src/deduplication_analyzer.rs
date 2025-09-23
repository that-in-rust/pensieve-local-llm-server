use crate::errors::{ValidationError, Result};
use crate::types::{
    DeduplicationROI, ParagraphDeduplicationSavings, DuplicateGroup, ROIRecommendation,
    CanonicalSelectionLogic, DeduplicationConfig,
};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use sha2::{Digest, Sha256};

/// Analyzer for measuring deduplication return on investment
pub struct DeduplicationAnalyzer {
    config: DeduplicationConfig,
}

/// File hash and metadata for deduplication analysis
#[derive(Debug, Clone)]
struct FileInfo {
    path: PathBuf,
    size: u64,
    hash: String,
    content_hash: Option<String>, // For content-based deduplication
    last_modified: std::time::SystemTime,
}

/// Paragraph content for deduplication analysis
#[derive(Debug, Clone)]
struct ParagraphInfo {
    content: String,
    hash: String,
    token_count: usize,
    source_files: Vec<PathBuf>,
}

/// Timing measurements for ROI calculation
#[derive(Debug, Default)]
struct TimingMeasurements {
    file_scanning_duration: Duration,
    hash_calculation_duration: Duration,
    duplicate_detection_duration: Duration,
    paragraph_analysis_duration: Duration,
    total_deduplication_overhead: Duration,
}

impl DeduplicationAnalyzer {
    /// Create a new deduplication analyzer with default configuration
    pub fn new() -> Self {
        Self {
            config: DeduplicationConfig::default(),
        }
    }

    /// Create a new deduplication analyzer with custom configuration
    pub fn with_config(config: DeduplicationConfig) -> Self {
        Self { config }
    }

    /// Analyze deduplication ROI for a directory
    pub async fn analyze_deduplication_roi(&self, directory: &Path) -> Result<DeduplicationROI> {
        let start_time = Instant::now();
        let mut timing = TimingMeasurements::default();

        // Step 1: Scan files and calculate hashes
        let scan_start = Instant::now();
        let file_infos = self.scan_files_for_deduplication(directory).await?;
        timing.file_scanning_duration = scan_start.elapsed();

        // Step 2: Detect file-level duplicates
        let duplicate_start = Instant::now();
        let duplicate_groups = self.detect_file_duplicates(&file_infos)?;
        timing.duplicate_detection_duration = duplicate_start.elapsed();

        // Step 3: Analyze paragraph-level deduplication (if enabled)
        let paragraph_start = Instant::now();
        let paragraph_savings = if self.config.enable_paragraph_deduplication {
            self.analyze_paragraph_deduplication(&file_infos).await?
        } else {
            ParagraphDeduplicationSavings {
                total_paragraphs: 0,
                unique_paragraphs: 0,
                duplicate_paragraphs: 0,
                token_savings: 0,
                token_savings_percentage: 0.0,
                processing_time_saved_seconds: 0.0,
            }
        };
        timing.paragraph_analysis_duration = paragraph_start.elapsed();

        // Step 4: Calculate ROI metrics
        timing.total_deduplication_overhead = start_time.elapsed();
        let roi = self.calculate_roi_metrics(
            &duplicate_groups,
            &paragraph_savings,
            &timing,
            &file_infos,
        )?;

        Ok(roi)
    }

    /// Scan directory for files and calculate hashes for deduplication analysis
    fn scan_files_for_deduplication<'a>(&'a self, directory: &'a Path) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Vec<FileInfo>>> + 'a>> {
        Box::pin(async move {
            let mut file_infos = Vec::new();
            let mut entries = tokio::fs::read_dir(directory).await
                .map_err(|e| ValidationError::FileSystem(e))?;

            while let Some(entry) = entries.next_entry().await
                .map_err(|e| ValidationError::FileSystem(e))? {
                
                let path = entry.path();
                
                if path.is_file() {
                    if let Ok(file_info) = self.analyze_file_for_deduplication(&path).await {
                        if file_info.size >= self.config.min_file_size_for_analysis {
                            file_infos.push(file_info);
                        }
                    }
                } else if path.is_dir() {
                    // Recursively scan subdirectories
                    let mut sub_files = self.scan_files_for_deduplication(&path).await?;
                    file_infos.append(&mut sub_files);
                }
            }

            Ok(file_infos)
        })
    }

    /// Analyze a single file for deduplication
    async fn analyze_file_for_deduplication(&self, path: &Path) -> Result<FileInfo> {
        let metadata = tokio::fs::metadata(path).await
            .map_err(|e| ValidationError::FileSystem(e))?;

        let size = metadata.len();
        let last_modified = metadata.modified()
            .map_err(|e| ValidationError::FileSystem(e))?;

        // Calculate file hash (for exact duplicates)
        let hash = self.calculate_file_hash(path).await?;

        // Calculate content hash (for content-based deduplication)
        let content_hash = if self.is_text_file(path) {
            Some(self.calculate_content_hash(path).await?)
        } else {
            None
        };

        Ok(FileInfo {
            path: path.to_path_buf(),
            size,
            hash,
            content_hash,
            last_modified,
        })
    }

    /// Calculate SHA-256 hash of file
    async fn calculate_file_hash(&self, path: &Path) -> Result<String> {
        let content = tokio::fs::read(path).await
            .map_err(|e| ValidationError::FileSystem(e))?;
        
        let mut hasher = Sha256::new();
        hasher.update(&content);
        Ok(format!("{:x}", hasher.finalize()))
    }

    /// Calculate content-based hash (normalized for whitespace, etc.)
    async fn calculate_content_hash(&self, path: &Path) -> Result<String> {
        let content = tokio::fs::read_to_string(path).await
            .map_err(|e| ValidationError::FileSystem(e))?;
        
        // Normalize content for better deduplication
        let normalized = self.normalize_content(&content);
        
        let mut hasher = Sha256::new();
        hasher.update(normalized.as_bytes());
        Ok(format!("{:x}", hasher.finalize()))
    }

    /// Normalize content for deduplication analysis
    fn normalize_content(&self, content: &str) -> String {
        content
            .lines()
            .map(|line| line.trim())
            .filter(|line| !line.is_empty())
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Check if file is likely a text file
    fn is_text_file(&self, path: &Path) -> bool {
        if let Some(extension) = path.extension() {
            let ext = extension.to_string_lossy().to_lowercase();
            matches!(ext.as_str(), 
                "txt" | "md" | "rs" | "py" | "js" | "ts" | "html" | "css" | 
                "json" | "xml" | "yaml" | "yml" | "toml" | "cfg" | "ini" |
                "log" | "csv" | "sql" | "sh" | "bat" | "ps1"
            )
        } else {
            false
        }
    }

    /// Detect file-level duplicates and group them
    fn detect_file_duplicates(&self, file_infos: &[FileInfo]) -> Result<Vec<DuplicateGroup>> {
        let mut hash_to_files: HashMap<String, Vec<&FileInfo>> = HashMap::new();
        
        // Group files by hash
        for file_info in file_infos {
            hash_to_files.entry(file_info.hash.clone())
                .or_insert_with(Vec::new)
                .push(file_info);
        }

        let mut duplicate_groups = Vec::new();

        // Process groups with duplicates
        for (_, files) in hash_to_files {
            if files.len() > 1 {
                let canonical_file = self.select_canonical_file(&files)?;
                let duplicate_files: Vec<PathBuf> = files
                    .iter()
                    .filter(|f| f.path != canonical_file.path)
                    .map(|f| f.path.clone())
                    .collect();

                let total_savings_bytes = (files.len() - 1) as u64 * canonical_file.size;

                duplicate_groups.push(DuplicateGroup {
                    canonical_file: canonical_file.path.clone(),
                    duplicate_files,
                    file_size_bytes: canonical_file.size,
                    total_savings_bytes,
                    selection_reason: self.explain_canonical_selection(&canonical_file, &files),
                });
            }
        }

        // Sort by savings (largest first)
        duplicate_groups.sort_by(|a, b| b.total_savings_bytes.cmp(&a.total_savings_bytes));

        Ok(duplicate_groups)
    }

    /// Select the canonical file from a group of duplicates
    fn select_canonical_file<'a>(&self, files: &[&'a FileInfo]) -> Result<&'a FileInfo> {
        // Selection criteria (in order of priority):
        // 1. Shortest path (likely in a more organized location)
        // 2. Most recent modification time
        // 3. Lexicographically first path (for consistency)

        let canonical = files
            .iter()
            .min_by(|a, b| {
                // Primary: shortest path
                let path_len_cmp = a.path.to_string_lossy().len().cmp(&b.path.to_string_lossy().len());
                if path_len_cmp != std::cmp::Ordering::Equal {
                    return path_len_cmp;
                }

                // Secondary: most recent modification
                let mod_time_cmp = b.last_modified.cmp(&a.last_modified);
                if mod_time_cmp != std::cmp::Ordering::Equal {
                    return mod_time_cmp;
                }

                // Tertiary: lexicographic order
                a.path.cmp(&b.path)
            })
            .ok_or_else(|| ValidationError::Analysis("No files in duplicate group".to_string()))?;

        Ok(canonical)
    }

    /// Explain why a particular file was selected as canonical
    fn explain_canonical_selection(&self, canonical: &FileInfo, all_files: &[&FileInfo]) -> String {
        let path_len = canonical.path.to_string_lossy().len();
        let shortest_path = all_files.iter()
            .map(|f| f.path.to_string_lossy().len())
            .min()
            .unwrap_or(path_len);

        if path_len == shortest_path {
            format!("Selected due to shortest path ({} characters)", path_len)
        } else {
            format!("Selected due to most recent modification time ({:?})", canonical.last_modified)
        }
    }

    /// Analyze paragraph-level deduplication opportunities
    async fn analyze_paragraph_deduplication(&self, file_infos: &[FileInfo]) -> Result<ParagraphDeduplicationSavings> {
        let mut all_paragraphs = Vec::new();
        let mut paragraph_to_files: HashMap<String, Vec<PathBuf>> = HashMap::new();

        // Extract paragraphs from text files
        for file_info in file_infos {
            if file_info.content_hash.is_some() {
                if let Ok(paragraphs) = self.extract_paragraphs(&file_info.path).await {
                    for paragraph in paragraphs {
                        paragraph_to_files.entry(paragraph.hash.clone())
                            .or_insert_with(Vec::new)
                            .push(file_info.path.clone());
                        all_paragraphs.push(paragraph);
                    }
                }
            }
        }

        // Calculate deduplication savings
        let total_paragraphs = all_paragraphs.len() as u64;
        let unique_hashes: HashSet<String> = all_paragraphs.iter()
            .map(|p| p.hash.clone())
            .collect();
        let unique_paragraphs = unique_hashes.len() as u64;
        let duplicate_paragraphs = total_paragraphs - unique_paragraphs;

        let total_tokens: usize = all_paragraphs.iter()
            .map(|p| p.token_count)
            .sum();
        
        let unique_tokens: usize = unique_hashes.iter()
            .filter_map(|hash| {
                all_paragraphs.iter()
                    .find(|p| &p.hash == hash)
                    .map(|p| p.token_count)
            })
            .sum();

        let token_savings = (total_tokens - unique_tokens) as u64;
        let token_savings_percentage = if total_tokens > 0 {
            (token_savings as f64 / total_tokens as f64) * 100.0
        } else {
            0.0
        };

        // Estimate processing time saved (rough heuristic: 1ms per token)
        let processing_time_saved_seconds = (token_savings as f64) / 1000.0;

        Ok(ParagraphDeduplicationSavings {
            total_paragraphs,
            unique_paragraphs,
            duplicate_paragraphs,
            token_savings,
            token_savings_percentage,
            processing_time_saved_seconds,
        })
    }

    /// Extract paragraphs from a text file
    async fn extract_paragraphs(&self, path: &Path) -> Result<Vec<ParagraphInfo>> {
        let content = tokio::fs::read_to_string(path).await
            .map_err(|e| ValidationError::FileSystem(e))?;

        let mut paragraphs = Vec::new();
        
        // Split content into paragraphs (separated by double newlines)
        for paragraph_text in content.split("\n\n") {
            let trimmed = paragraph_text.trim();
            if !trimmed.is_empty() && trimmed.len() > 50 { // Only consider substantial paragraphs
                let normalized = self.normalize_content(trimmed);
                let token_count = self.estimate_token_count(&normalized);
                
                let mut hasher = Sha256::new();
                hasher.update(normalized.as_bytes());
                let hash = format!("{:x}", hasher.finalize());

                paragraphs.push(ParagraphInfo {
                    content: normalized,
                    hash,
                    token_count,
                    source_files: vec![path.to_path_buf()],
                });
            }
        }

        Ok(paragraphs)
    }

    /// Estimate token count for a text (rough approximation)
    fn estimate_token_count(&self, text: &str) -> usize {
        // Simple approximation: split by whitespace and punctuation
        text.split_whitespace().count()
    }

    /// Calculate comprehensive ROI metrics
    fn calculate_roi_metrics(
        &self,
        duplicate_groups: &[DuplicateGroup],
        paragraph_savings: &ParagraphDeduplicationSavings,
        timing: &TimingMeasurements,
        file_infos: &[FileInfo],
    ) -> Result<DeduplicationROI> {
        // File-level savings
        let file_level_duplicates = duplicate_groups.iter()
            .map(|g| g.duplicate_files.len() as u64)
            .sum();
        
        let storage_saved_bytes = duplicate_groups.iter()
            .map(|g| g.total_savings_bytes)
            .sum();

        let total_storage_bytes: u64 = file_infos.iter()
            .map(|f| f.size)
            .sum();

        let storage_saved_percentage = if total_storage_bytes > 0 {
            (storage_saved_bytes as f64 / total_storage_bytes as f64) * 100.0
        } else {
            0.0
        };

        // Time calculations
        let deduplication_overhead_seconds = timing.total_deduplication_overhead.as_secs_f64();
        
        // Estimate time saved in downstream processing
        // Heuristic: saved files would take 10ms each to process
        let file_processing_time_saved = (file_level_duplicates as f64) * 0.01;
        let total_processing_time_saved = file_processing_time_saved + paragraph_savings.processing_time_saved_seconds;
        
        let net_benefit_seconds = total_processing_time_saved - deduplication_overhead_seconds;

        // ROI recommendation
        let roi_recommendation = self.calculate_roi_recommendation(
            storage_saved_percentage,
            net_benefit_seconds,
            deduplication_overhead_seconds,
        );

        // Canonical selection logic
        let canonical_selection_logic = CanonicalSelectionLogic {
            primary_criteria: "Shortest file path".to_string(),
            secondary_criteria: vec![
                "Most recent modification time".to_string(),
                "Lexicographic ordering".to_string(),
            ],
            explanation: "Files with shorter paths are typically in more organized locations and easier to find".to_string(),
        };

        Ok(DeduplicationROI {
            file_level_duplicates,
            storage_saved_bytes,
            storage_saved_percentage,
            processing_time_saved_seconds: total_processing_time_saved,
            deduplication_overhead_seconds,
            net_benefit_seconds,
            paragraph_level_savings: paragraph_savings.clone(),
            duplicate_groups: duplicate_groups.to_vec(),
            roi_recommendation,
            canonical_selection_logic,
        })
    }

    /// Calculate ROI recommendation based on metrics
    fn calculate_roi_recommendation(
        &self,
        storage_saved_percentage: f64,
        net_benefit_seconds: f64,
        overhead_seconds: f64,
    ) -> ROIRecommendation {
        // Negative ROI: overhead exceeds benefits
        if net_benefit_seconds < 0.0 {
            return ROIRecommendation::Negative;
        }

        // High value: >50% storage savings or >10x time benefit
        if storage_saved_percentage > 50.0 || (net_benefit_seconds / overhead_seconds) > 10.0 {
            return ROIRecommendation::HighValue;
        }

        // Moderate value: 20-50% storage savings or 3-10x time benefit
        if storage_saved_percentage > 20.0 || (net_benefit_seconds / overhead_seconds) > 3.0 {
            return ROIRecommendation::ModerateValue;
        }

        // Low value: 5-20% storage savings or positive but small time benefit
        if storage_saved_percentage > 5.0 || net_benefit_seconds > 0.0 {
            return ROIRecommendation::LowValue;
        }

        ROIRecommendation::Negative
    }
}

impl Default for DeduplicationAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use tokio::fs;

    async fn create_test_directory_with_duplicates() -> Result<TempDir> {
        let temp_dir = TempDir::new().map_err(|e| ValidationError::FileSystem(e))?;
        let base_path = temp_dir.path();

        // Create subdirectory first
        fs::create_dir_all(base_path.join("subdir")).await?;
        
        // Create identical files
        let content1 = "This is the same content in multiple files.";
        fs::write(base_path.join("file1.txt"), content1).await?;
        fs::write(base_path.join("copy_of_file1.txt"), content1).await?;
        fs::write(base_path.join("subdir/another_copy.txt"), content1).await?;

        // Create files with similar content (for paragraph deduplication)
        let content2 = "First paragraph with some content that is long enough to be considered for deduplication analysis.\n\nSecond paragraph with different content that is also long enough for analysis.";
        let content3 = "First paragraph with some content that is long enough to be considered for deduplication analysis.\n\nThird paragraph with unique content that is also long enough for analysis.";
        
        fs::write(base_path.join("doc1.txt"), content2).await?;
        fs::write(base_path.join("doc2.txt"), content3).await?;

        // Create unique files
        fs::write(base_path.join("unique1.txt"), "Completely unique content here.").await?;
        fs::write(base_path.join("unique2.txt"), "Another unique file with different content.").await?;

        Ok(temp_dir)
    }

    #[tokio::test]
    async fn test_file_duplicate_detection() -> Result<()> {
        let temp_dir = create_test_directory_with_duplicates().await?;
        // Use a config with lower minimum file size for testing
        let config = DeduplicationConfig {
            min_file_size_for_analysis: 1, // 1 byte minimum for testing
            ..Default::default()
        };
        let analyzer = DeduplicationAnalyzer::with_config(config);

        let roi = analyzer.analyze_deduplication_roi(temp_dir.path()).await?;

        // Should detect at least one duplicate group
        assert!(!roi.duplicate_groups.is_empty());
        
        // Should have detected file-level duplicates
        assert!(roi.file_level_duplicates > 0);
        
        // Should have some storage savings
        assert!(roi.storage_saved_bytes > 0);
        assert!(roi.storage_saved_percentage > 0.0);

        Ok(())
    }

    #[tokio::test]
    async fn test_paragraph_deduplication() -> Result<()> {
        let temp_dir = create_test_directory_with_duplicates().await?;
        // Use a config with lower minimum file size for testing
        let config = DeduplicationConfig {
            min_file_size_for_analysis: 1, // 1 byte minimum for testing
            ..Default::default()
        };
        let analyzer = DeduplicationAnalyzer::with_config(config);

        let roi = analyzer.analyze_deduplication_roi(temp_dir.path()).await?;

        // Should have analyzed paragraphs
        assert!(roi.paragraph_level_savings.total_paragraphs > 0);
        
        // Should have found some duplicate paragraphs
        assert!(roi.paragraph_level_savings.duplicate_paragraphs > 0);
        
        // Should have some token savings
        assert!(roi.paragraph_level_savings.token_savings > 0);

        Ok(())
    }

    #[tokio::test]
    async fn test_roi_recommendation() -> Result<()> {
        let temp_dir = create_test_directory_with_duplicates().await?;
        // Use a config with lower minimum file size for testing
        let config = DeduplicationConfig {
            min_file_size_for_analysis: 1, // 1 byte minimum for testing
            ..Default::default()
        };
        let analyzer = DeduplicationAnalyzer::with_config(config);

        let roi = analyzer.analyze_deduplication_roi(temp_dir.path()).await?;

        // Should have a valid ROI recommendation
        assert!(matches!(roi.roi_recommendation, 
            ROIRecommendation::HighValue | 
            ROIRecommendation::ModerateValue | 
            ROIRecommendation::LowValue | 
            ROIRecommendation::Negative
        ));

        Ok(())
    }

    #[tokio::test]
    async fn test_canonical_file_selection() -> Result<()> {
        let temp_dir = create_test_directory_with_duplicates().await?;
        // Use a config with lower minimum file size for testing
        let config = DeduplicationConfig {
            min_file_size_for_analysis: 1, // 1 byte minimum for testing
            ..Default::default()
        };
        let analyzer = DeduplicationAnalyzer::with_config(config);

        let roi = analyzer.analyze_deduplication_roi(temp_dir.path()).await?;

        // Should have selected canonical files
        for group in &roi.duplicate_groups {
            assert!(!group.canonical_file.as_os_str().is_empty());
            assert!(!group.duplicate_files.is_empty());
            assert!(!group.selection_reason.is_empty());
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_timing_measurements() -> Result<()> {
        let temp_dir = create_test_directory_with_duplicates().await?;
        // Use a config with lower minimum file size for testing
        let config = DeduplicationConfig {
            min_file_size_for_analysis: 1, // 1 byte minimum for testing
            ..Default::default()
        };
        let analyzer = DeduplicationAnalyzer::with_config(config);

        let roi = analyzer.analyze_deduplication_roi(temp_dir.path()).await?;

        // Should have measured overhead
        assert!(roi.deduplication_overhead_seconds > 0.0);
        
        // Net benefit can be positive or negative
        assert!(roi.net_benefit_seconds != 0.0);

        Ok(())
    }
}
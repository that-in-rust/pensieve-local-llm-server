use crate::chaos_detector::ChaosDetector;
use crate::errors::{ValidationError, Result};
use crate::types::{DirectoryAnalysis, FileTypeStats, SizeDistribution, DepthAnalysis, ChaosIndicators};
use crate::types::*;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

/// Pre-flight analysis of target directory
pub struct DirectoryAnalyzer {
    chaos_detector: ChaosDetector,
}

impl Default for DirectoryAnalyzer {
    fn default() -> Self {
        Self {
            chaos_detector: ChaosDetector::new(),
        }
    }
}

impl DirectoryAnalyzer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_chaos_detector(chaos_detector: ChaosDetector) -> Self {
        Self { chaos_detector }
    }

    /// Perform comprehensive directory analysis
    pub fn analyze_directory(&self, directory: &Path) -> Result<DirectoryAnalysis> {
        if !directory.exists() {
            return Err(ValidationError::DirectoryNotAccessible {
                path: directory.to_path_buf(),
                cause: "Directory does not exist".to_string(),
            });
        }

        if !directory.is_dir() {
            return Err(ValidationError::DirectoryNotAccessible {
                path: directory.to_path_buf(),
                cause: "Path is not a directory".to_string(),
            });
        }

        let mut total_files = 0u64;
        let mut total_directories = 0u64;
        let mut total_size_bytes = 0u64;
        let mut file_type_distribution: HashMap<String, FileTypeStats> = HashMap::new();
        let mut size_distribution = SizeDistribution {
            zero_byte_files: 0,
            small_files: 0,
            medium_files: 0,
            large_files: 0,
            very_large_files: 0,
            largest_file_size: 0,
            largest_file_path: PathBuf::new(),
        };
        let mut depth_analysis = DepthAnalysis {
            max_depth: 0,
            average_depth: 0.0,
            files_by_depth: HashMap::new(),
            deepest_path: PathBuf::new(),
        };

        let mut total_depth = 0usize;
        let mut file_count_for_depth = 0u64;

        // Walk through directory
        for entry in WalkDir::new(directory).into_iter() {
            match entry {
                Ok(entry) => {
                    let path = entry.path();
                    let depth = entry.depth();

                    if entry.file_type().is_dir() {
                        total_directories += 1;
                        
                        // Update depth analysis for directories
                        if depth > depth_analysis.max_depth {
                            depth_analysis.max_depth = depth;
                            depth_analysis.deepest_path = path.to_path_buf();
                        }
                    } else if entry.file_type().is_file() {
                        total_files += 1;
                        file_count_for_depth += 1;
                        total_depth += depth;

                        // Update files by depth
                        *depth_analysis.files_by_depth.entry(depth).or_insert(0) += 1;

                        // Analyze file
                        if let Err(e) = self.analyze_file(path, &mut file_type_distribution, &mut size_distribution, &mut total_size_bytes) {
                            eprintln!("Warning: Failed to analyze file {}: {}", path.display(), e);
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Warning: Error walking directory: {}", e);
                }
            }
        }

        // Calculate average depth
        depth_analysis.average_depth = if file_count_for_depth > 0 {
            total_depth as f64 / file_count_for_depth as f64
        } else {
            0.0
        };

        // Perform chaos detection
        let chaos_report = self.chaos_detector.detect_chaos_files(directory)?;
        let chaos_indicators = chaos_report.calculate_chaos_metrics(total_files);

        Ok(DirectoryAnalysis {
            total_files,
            total_directories,
            total_size_bytes,
            file_type_distribution,
            size_distribution,
            depth_analysis,
            chaos_indicators,
        })
    }

    fn analyze_file(
        &self,
        path: &Path,
        file_type_distribution: &mut HashMap<String, FileTypeStats>,
        size_distribution: &mut SizeDistribution,
        total_size_bytes: &mut u64,
    ) -> Result<()> {
        let metadata = fs::metadata(path).map_err(|e| ValidationError::FileSystem(e))?;
        let file_size = metadata.len();
        *total_size_bytes += file_size;

        // Update size distribution
        self.update_size_distribution(path, file_size, size_distribution);

        // Determine file type
        let file_type = self.determine_file_type(path);
        let processing_complexity = self.assess_processing_complexity(&file_type, path);

        // Update file type statistics
        let stats = file_type_distribution.entry(file_type).or_insert(FileTypeStats {
            count: 0,
            total_size_bytes: 0,
            average_size_bytes: 0,
            largest_file: path.to_path_buf(),
            processing_complexity,
        });

        stats.count += 1;
        stats.total_size_bytes += file_size;
        stats.average_size_bytes = stats.total_size_bytes / stats.count;

        // Update largest file for this type
        if file_size > fs::metadata(&stats.largest_file).map(|m| m.len()).unwrap_or(0) {
            stats.largest_file = path.to_path_buf();
        }

        Ok(())
    }

    fn update_size_distribution(&self, path: &Path, file_size: u64, size_distribution: &mut SizeDistribution) {
        match file_size {
            0 => size_distribution.zero_byte_files += 1,
            1..=1_023 => size_distribution.small_files += 1,           // < 1KB
            1_024..=1_048_575 => size_distribution.medium_files += 1, // 1KB - 1MB
            1_048_576..=104_857_599 => size_distribution.large_files += 1, // 1MB - 100MB
            _ => size_distribution.very_large_files += 1,             // > 100MB
        }

        // Update largest file
        if file_size > size_distribution.largest_file_size {
            size_distribution.largest_file_size = file_size;
            size_distribution.largest_file_path = path.to_path_buf();
        }
    }

    fn determine_file_type(&self, path: &Path) -> String {
        // First try by extension
        if let Some(extension) = path.extension() {
            let ext_str = extension.to_string_lossy().to_lowercase();
            
            // Map common extensions to categories
            match ext_str.as_str() {
                "txt" | "md" | "rst" | "log" => "text".to_string(),
                "rs" | "py" | "js" | "ts" | "java" | "cpp" | "c" | "h" => "source_code".to_string(),
                "json" | "xml" | "yaml" | "yml" | "toml" | "csv" => "structured_data".to_string(),
                "pdf" | "doc" | "docx" | "odt" => "document".to_string(),
                "jpg" | "jpeg" | "png" | "gif" | "bmp" | "svg" => "image".to_string(),
                "mp3" | "wav" | "flac" | "ogg" => "audio".to_string(),
                "mp4" | "avi" | "mkv" | "mov" => "video".to_string(),
                "zip" | "tar" | "gz" | "bz2" | "xz" | "7z" => "archive".to_string(),
                "exe" | "dll" | "so" | "dylib" => "binary".to_string(),
                _ => format!("other_{}", ext_str),
            }
        } else {
            // Try to detect by content for extensionless files
            self.detect_type_by_content(path).unwrap_or_else(|| "unknown".to_string())
        }
    }

    fn detect_type_by_content(&self, path: &Path) -> Option<String> {
        let mut buffer = [0u8; 512];
        if let Ok(mut file) = std::fs::File::open(path) {
            use std::io::Read;
            if let Ok(bytes_read) = file.read(&mut buffer) {
                if bytes_read >= 4 {
                    // Check magic numbers
                    match &buffer[0..4] {
                        [0x89, 0x50, 0x4E, 0x47] => return Some("image".to_string()),
                        [0xFF, 0xD8, 0xFF, _] => return Some("image".to_string()),
                        [0x47, 0x49, 0x46, 0x38] => return Some("image".to_string()),
                        [0x25, 0x50, 0x44, 0x46] => return Some("document".to_string()),
                        [0x50, 0x4B, 0x03, 0x04] => return Some("archive".to_string()),
                        _ => {}
                    }
                }

                // Check if it's likely text
                let text_chars = buffer[0..bytes_read].iter()
                    .filter(|&&b| b.is_ascii_graphic() || b.is_ascii_whitespace())
                    .count();
                
                if bytes_read > 0 && text_chars as f64 / bytes_read as f64 > 0.8 {
                    return Some("text".to_string());
                }
            }
        }
        None
    }

    fn assess_processing_complexity(&self, file_type: &str, path: &Path) -> ProcessingComplexity {
        match file_type {
            "text" | "source_code" => ProcessingComplexity::Low,
            "structured_data" | "document" => ProcessingComplexity::Medium,
            "image" | "audio" | "video" | "archive" | "binary" => ProcessingComplexity::High,
            _ => {
                // For unknown types, try to assess by file size
                if let Ok(metadata) = fs::metadata(path) {
                    match metadata.len() {
                        0..=1_048_576 => ProcessingComplexity::Low,      // < 1MB
                        1_048_577..=10_485_760 => ProcessingComplexity::Medium, // 1-10MB
                        _ => ProcessingComplexity::High,                 // > 10MB
                    }
                } else {
                    ProcessingComplexity::Medium
                }
            }
        }
    }

    /// Get chaos detection report separately
    pub fn get_chaos_report(&self, directory: &Path) -> Result<ChaosReport> {
        self.chaos_detector.detect_chaos_files(directory)
    }
}
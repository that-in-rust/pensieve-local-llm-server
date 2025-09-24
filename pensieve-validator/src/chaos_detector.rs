use crate::errors::Result;
use crate::types::{ChaosReport, PermissionIssue, PermissionIssueType, LargeFile, SizeCategory, MisleadingFile, UnicodeFile, UnusualCharacterFile, DeepNestedFile, CorruptedFile};
use crate::types::*;
use std::fs;
use std::path::Path;
use walkdir::WalkDir;

/// Identifies problematic files that commonly cause issues
pub struct ChaosDetector {
    max_symlink_depth: usize,
    large_file_threshold: u64,
    very_large_file_threshold: u64,
    enormous_file_threshold: u64,
    max_path_depth: usize,
    max_path_length: usize,
}

impl Default for ChaosDetector {
    fn default() -> Self {
        Self {
            max_symlink_depth: 10,
            large_file_threshold: 100_000_000,      // 100MB
            very_large_file_threshold: 1_000_000_000, // 1GB
            enormous_file_threshold: 10_000_000_000,   // 10GB
            max_path_depth: 20,
            max_path_length: 260, // Windows MAX_PATH limit
        }
    }
}

impl ChaosDetector {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_config(
        max_symlink_depth: usize,
        large_file_threshold: u64,
        max_path_depth: usize,
        max_path_length: usize,
    ) -> Self {
        Self {
            max_symlink_depth,
            large_file_threshold,
            very_large_file_threshold: large_file_threshold * 10,
            enormous_file_threshold: large_file_threshold * 100,
            max_path_depth,
            max_path_length,
        }
    }

    /// Detect files that commonly cause issues
    pub fn detect_chaos_files(&self, directory: &Path) -> Result<ChaosReport> {
        let mut report = ChaosReport {
            files_without_extensions: Vec::new(),
            misleading_extensions: Vec::new(),
            unicode_filenames: Vec::new(),
            extremely_large_files: Vec::new(),
            zero_byte_files: Vec::new(),
            permission_issues: Vec::new(),
            symlink_chains: Vec::new(),
            corrupted_files: Vec::new(),
            unusual_characters: Vec::new(),
            deep_nesting: Vec::new(),
        };

        for entry in WalkDir::new(directory)
            .follow_links(false) // Don't follow symlinks to avoid infinite loops
            .into_iter()
        {
            match entry {
                Ok(entry) => {
                    if let Err(e) = self.analyze_entry(&entry, &mut report) {
                        eprintln!("Warning: Failed to analyze {}: {}", entry.path().display(), e);
                    }
                }
                Err(e) => {
                    // Handle walkdir errors (permission issues, etc.)
                    if let Some(path) = e.path() {
                        if let Some(io_error) = e.io_error() {
                            match io_error.kind() {
                                std::io::ErrorKind::PermissionDenied => {
                                    report.permission_issues.push(PermissionIssue {
                                        path: path.to_path_buf(),
                                        issue_type: PermissionIssueType::ReadDenied,
                                        details: format!("Permission denied during directory traversal: {}", io_error),
                                    });
                                }
                                _ => {
                                    eprintln!("Warning: IO error accessing {}: {}", path.display(), io_error);
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(report)
    }

    fn analyze_entry(&self, entry: &walkdir::DirEntry, report: &mut ChaosReport) -> Result<()> {
        let path = entry.path();
        
        // Skip directories for most checks
        if entry.file_type().is_dir() {
            self.check_deep_nesting(path, report);
            self.check_unicode_filename(path, report);
            self.check_unusual_characters(path, report);
            return Ok(());
        }

        // File-specific checks
        if entry.file_type().is_file() {
            self.check_file_extension(path, report)?;
            self.check_file_size(path, report)?;
            self.check_misleading_extension(path, report)?;
            self.check_corrupted_file(path, report);
        }

        // Symlink checks
        if entry.file_type().is_symlink() {
            self.check_symlink_chain(path, report)?;
        }

        // Common checks for all entry types
        self.check_unicode_filename(path, report);
        self.check_unusual_characters(path, report);
        self.check_deep_nesting(path, report);
        self.check_permissions(path, report);

        Ok(())
    }

    fn check_file_extension(&self, path: &Path, report: &mut ChaosReport) -> Result<()> {
        if let Some(file_name) = path.file_name() {
            let file_name_str = file_name.to_string_lossy();
            
            // Check for files without extensions
            if !file_name_str.contains('.') || file_name_str.starts_with('.') {
                // Hidden files starting with . are not considered extensionless
                if !file_name_str.starts_with('.') {
                    report.files_without_extensions.push(path.to_path_buf());
                }
            }
        }
        Ok(())
    }

    fn check_file_size(&self, path: &Path, report: &mut ChaosReport) -> Result<()> {
        match fs::metadata(path) {
            Ok(metadata) => {
                let size = metadata.len();
                
                if size == 0 {
                    report.zero_byte_files.push(path.to_path_buf());
                } else if size >= self.large_file_threshold {
                    let category = if size >= self.enormous_file_threshold {
                        SizeCategory::Enormous
                    } else if size >= self.very_large_file_threshold {
                        SizeCategory::VeryLarge
                    } else {
                        SizeCategory::Large
                    };

                    report.extremely_large_files.push(LargeFile {
                        path: path.to_path_buf(),
                        size_bytes: size,
                        size_category: category,
                    });
                }
            }
            Err(e) => {
                report.permission_issues.push(PermissionIssue {
                    path: path.to_path_buf(),
                    issue_type: PermissionIssueType::ReadDenied,
                    details: format!("Cannot read file metadata: {}", e),
                });
            }
        }
        Ok(())
    }

    fn check_misleading_extension(&self, path: &Path, report: &mut ChaosReport) -> Result<()> {
        if let Some(extension) = path.extension() {
            let extension_str = extension.to_string_lossy().to_lowercase();
            let mime_from_extension = mime_guess::from_ext(&extension_str);
            
            // Try to detect actual file type by reading file header
            if let Ok(actual_mime) = self.detect_file_type_by_content(path) {
                let extension_mime = mime_from_extension.first();
                
                if let Some(ext_mime) = extension_mime {
                    // Compare MIME types
                    if ext_mime.type_() != actual_mime.type_() {
                        report.misleading_extensions.push(MisleadingFile {
                            path: path.to_path_buf(),
                            claimed_type: ext_mime.to_string(),
                            actual_type: actual_mime.to_string(),
                            confidence: 0.8, // Basic confidence score
                        });
                    }
                }
            }
        }
        Ok(())
    }

    fn detect_file_type_by_content(&self, path: &Path) -> Result<mime::Mime> {
        let mut buffer = [0u8; 512];
        match std::fs::File::open(path) {
            Ok(mut file) => {
                use std::io::Read;
                let bytes_read = file.read(&mut buffer).unwrap_or(0);
                
                // Basic magic number detection
                if bytes_read >= 4 {
                    match &buffer[0..4] {
                        [0x89, 0x50, 0x4E, 0x47] => return Ok("image/png".parse().unwrap()),
                        [0xFF, 0xD8, 0xFF, _] => return Ok("image/jpeg".parse().unwrap()),
                        [0x47, 0x49, 0x46, 0x38] => return Ok("image/gif".parse().unwrap()),
                        [0x25, 0x50, 0x44, 0x46] => return Ok("application/pdf".parse().unwrap()),
                        [0x50, 0x4B, 0x03, 0x04] => return Ok("application/zip".parse().unwrap()),
                        _ => {}
                    }
                }

                // Check if it's likely text
                let text_chars = buffer[0..bytes_read].iter()
                    .filter(|&&b| b.is_ascii_graphic() || b.is_ascii_whitespace())
                    .count();
                
                if bytes_read > 0 && text_chars as f64 / bytes_read as f64 > 0.8 {
                    return Ok("text/plain".parse().unwrap());
                }

                // Default to binary
                Ok("application/octet-stream".parse().unwrap())
            }
            Err(_) => Ok("application/octet-stream".parse().unwrap()),
        }
    }

    fn check_unicode_filename(&self, path: &Path, report: &mut ChaosReport) {
        if let Some(file_name) = path.file_name() {
            let file_name_str = file_name.to_string_lossy();
            let mut unicode_categories = Vec::new();
            let mut problematic_chars = Vec::new();

            for ch in file_name_str.chars() {
                if !ch.is_ascii() {
                    problematic_chars.push(ch);
                    
                    // Categorize unicode characters
                    match unicode_general_category::get_general_category(ch) {
                        unicode_general_category::GeneralCategory::SymbolOther => {
                            unicode_categories.push("Symbol".to_string());
                        }
                        unicode_general_category::GeneralCategory::MarkNonspacing => {
                            unicode_categories.push("NonspacingMark".to_string());
                        }
                        unicode_general_category::GeneralCategory::OtherControl => {
                            unicode_categories.push("Control".to_string());
                        }
                        _ => {
                            unicode_categories.push("Unicode".to_string());
                        }
                    }
                }
            }

            if !problematic_chars.is_empty() {
                unicode_categories.sort();
                unicode_categories.dedup();
                
                report.unicode_filenames.push(UnicodeFile {
                    path: path.to_path_buf(),
                    unicode_categories,
                    problematic_chars,
                });
            }
        }
    }

    fn check_unusual_characters(&self, path: &Path, report: &mut ChaosReport) {
        if let Some(file_name) = path.file_name() {
            let file_name_str = file_name.to_string_lossy();
            let mut unusual_chars = Vec::new();
            let mut char_categories = Vec::new();

            for ch in file_name_str.chars() {
                // Check for problematic characters that might cause issues
                match ch {
                    // Control characters
                    '\x00'..='\x1F' | '\x7F' => {
                        unusual_chars.push(ch);
                        char_categories.push("Control".to_string());
                    }
                    // Problematic punctuation
                    '<' | '>' | ':' | '"' | '|' | '?' | '*' => {
                        unusual_chars.push(ch);
                        char_categories.push("ProblematicPunctuation".to_string());
                    }
                    // Zero-width characters
                    '\u{200B}' | '\u{200C}' | '\u{200D}' | '\u{FEFF}' => {
                        unusual_chars.push(ch);
                        char_categories.push("ZeroWidth".to_string());
                    }
                    _ => {}
                }
            }

            if !unusual_chars.is_empty() {
                char_categories.sort();
                char_categories.dedup();
                
                report.unusual_characters.push(UnusualCharacterFile {
                    path: path.to_path_buf(),
                    unusual_chars,
                    char_categories,
                });
            }
        }
    }

    fn check_deep_nesting(&self, path: &Path, report: &mut ChaosReport) {
        let depth = path.components().count();
        let path_length = path.to_string_lossy().len();

        if depth > self.max_path_depth || path_length > self.max_path_length {
            report.deep_nesting.push(DeepNestedFile {
                path: path.to_path_buf(),
                depth,
                path_length,
            });
        }
    }

    fn check_permissions(&self, path: &Path, report: &mut ChaosReport) {
        match fs::metadata(path) {
            Ok(metadata) => {
                // On Unix systems, check permissions
                #[cfg(unix)]
                {
                    use std::os::unix::fs::PermissionsExt;
                    let mode = metadata.permissions().mode();
                    
                    // Check for unusual permission combinations
                    if mode & 0o777 == 0 {
                        report.permission_issues.push(PermissionIssue {
                            path: path.to_path_buf(),
                            issue_type: PermissionIssueType::ReadDenied,
                            details: "File has no permissions set".to_string(),
                        });
                    }
                }
            }
            Err(e) => {
                let issue_type = match e.kind() {
                    std::io::ErrorKind::PermissionDenied => PermissionIssueType::ReadDenied,
                    _ => PermissionIssueType::OwnershipIssue,
                };

                report.permission_issues.push(PermissionIssue {
                    path: path.to_path_buf(),
                    issue_type,
                    details: e.to_string(),
                });
            }
        }
    }

    fn check_symlink_chain(&self, path: &Path, report: &mut ChaosReport) -> Result<()> {
        let mut chain = Vec::new();
        let mut current_path = path.to_path_buf();
        let mut visited = std::collections::HashSet::new();

        for _ in 0..self.max_symlink_depth {
            if visited.contains(&current_path) {
                // Circular symlink detected
                report.symlink_chains.push(SymlinkChain {
                    start_path: path.to_path_buf(),
                    chain: chain.clone(),
                    chain_length: chain.len(),
                    is_circular: true,
                    final_target: None,
                });
                return Ok(());
            }

            visited.insert(current_path.clone());
            chain.push(current_path.clone());

            match fs::read_link(&current_path) {
                Ok(target) => {
                    current_path = if target.is_absolute() {
                        target
                    } else {
                        current_path.parent()
                            .unwrap_or_else(|| Path::new("/"))
                            .join(target)
                    };
                }
                Err(_) => {
                    // End of chain or broken symlink
                    break;
                }
            }
        }

        if chain.len() > 1 {
            let final_target = if current_path.exists() {
                Some(current_path)
            } else {
                None
            };

            let chain_length = chain.len();
            report.symlink_chains.push(SymlinkChain {
                start_path: path.to_path_buf(),
                chain,
                chain_length,
                is_circular: false,
                final_target,
            });
        }

        Ok(())
    }

    fn check_corrupted_file(&self, path: &Path, report: &mut ChaosReport) {
        // Basic corruption detection
        match fs::File::open(path) {
            Ok(mut file) => {
                use std::io::Read;
                let mut buffer = [0u8; 1024];
                
                match file.read(&mut buffer) {
                    Ok(0) => {
                        // Empty file - already handled in size check
                    }
                    Ok(bytes_read) => {
                        // Check for common corruption patterns
                        if self.detect_corruption_patterns(&buffer[0..bytes_read]) {
                            report.corrupted_files.push(CorruptedFile {
                                path: path.to_path_buf(),
                                corruption_type: CorruptionType::MalformedStructure,
                                details: "Detected corruption patterns in file content".to_string(),
                            });
                        }
                    }
                    Err(e) => {
                        report.corrupted_files.push(CorruptedFile {
                            path: path.to_path_buf(),
                            corruption_type: CorruptionType::UnreadableContent,
                            details: format!("Cannot read file content: {}", e),
                        });
                    }
                }
            }
            Err(e) => {
                if e.kind() != std::io::ErrorKind::PermissionDenied {
                    report.corrupted_files.push(CorruptedFile {
                        path: path.to_path_buf(),
                        corruption_type: CorruptionType::UnreadableContent,
                        details: format!("Cannot open file: {}", e),
                    });
                }
            }
        }
    }

    fn detect_corruption_patterns(&self, buffer: &[u8]) -> bool {
        // Simple heuristics for detecting corruption
        
        // Check for excessive null bytes (might indicate corruption)
        let null_count = buffer.iter().filter(|&&b| b == 0).count();
        if null_count > buffer.len() / 2 {
            return true;
        }

        // Check for repeated patterns that might indicate corruption
        if buffer.len() >= 16 {
            let pattern = &buffer[0..4];
            let mut pattern_count = 0;
            for chunk in buffer.chunks(4) {
                if chunk == pattern {
                    pattern_count += 1;
                }
            }
            // If more than 75% of the file is the same 4-byte pattern, it might be corrupted
            if pattern_count > (buffer.len() / 4) * 3 / 4 {
                return true;
            }
        }

        false
    }
}

// Helper module for unicode categorization
mod unicode_general_category {
    #[derive(Debug, Clone, Copy)]
    pub enum GeneralCategory {
        SymbolOther,
        MarkNonspacing,
        OtherControl,
        Other,
    }

    pub fn get_general_category(ch: char) -> GeneralCategory {
        // Simplified categorization - in a real implementation, 
        // you'd use the unicode-general-category crate
        match ch {
            '\u{0000}'..='\u{001F}' | '\u{007F}'..='\u{009F}' => GeneralCategory::OtherControl,
            '\u{0300}'..='\u{036F}' => GeneralCategory::MarkNonspacing,
            '\u{2000}'..='\u{206F}' => GeneralCategory::SymbolOther,
            _ => GeneralCategory::Other,
        }
    }
}
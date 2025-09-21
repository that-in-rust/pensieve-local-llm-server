//! File system scanning and metadata extraction

use crate::prelude::*;
use crate::types::{FileMetadata, FileType, ProcessingStatus, DuplicateStatus};
use crate::errors::ErrorContext;
use std::path::Path;
use std::time::{SystemTime, Instant};
use std::os::unix::fs::PermissionsExt;
use walkdir::WalkDir;
use rayon::prelude::*;
use sha2::{Sha256, Digest};
use tokio::fs::File;
use tokio::io::{AsyncReadExt, BufReader};
use std::collections::HashMap;
use uuid::Uuid;

/// File scanner for directory traversal and metadata extraction
pub struct FileScanner {
    /// Root directory being scanned
    root_path: std::path::PathBuf,
    /// Whether to follow symbolic links
    follow_symlinks: bool,
    /// Maximum directory depth to traverse
    max_depth: Option<usize>,
}

impl FileScanner {
    /// Create a new file scanner
    pub fn new(root_path: impl AsRef<Path>) -> Self {
        Self {
            root_path: root_path.as_ref().to_path_buf(),
            follow_symlinks: false,
            max_depth: None,
        }
    }

    /// Configure whether to follow symbolic links
    pub fn follow_symlinks(mut self, follow: bool) -> Self {
        self.follow_symlinks = follow;
        self
    }

    /// Set maximum directory depth to traverse
    pub fn max_depth(mut self, depth: usize) -> Self {
        self.max_depth = Some(depth);
        self
    }

    /// Scan directory and return file metadata with parallel processing
    pub async fn scan(&self) -> Result<Vec<FileMetadata>> {
        let start_time = Instant::now();
        let mut progress = ScanProgress::new();
        
        // First pass: discover all files using walkdir
        let discovered_files = self.discover_files()?;
        progress.total_files = discovered_files.len();
        
        println!("Discovered {} files, starting metadata extraction...", discovered_files.len());
        
        // Second pass: extract metadata in parallel using rayon
        let file_detector = FileTypeDetector::new();
        let metadata_results: Vec<Result<FileMetadata>> = discovered_files
            .into_par_iter()
            .map(|path| {
                // Extract metadata for each file
                let metadata = self.extract_metadata_sync(&path, &file_detector)?;
                Ok(metadata)
            })
            .collect();
        
        // Collect successful results and handle errors
        let mut successful_metadata = Vec::new();
        let mut error_count = 0;
        
        for result in metadata_results {
            match result {
                Ok(metadata) => {
                    progress.update(metadata.size);
                    successful_metadata.push(metadata);
                }
                Err(e) => {
                    error_count += 1;
                    eprintln!("Error processing file: {}", e);
                }
            }
        }
        
        // Third pass: detect duplicates by hash
        let deduplicated_metadata = self.detect_duplicates(successful_metadata);
        
        let elapsed = start_time.elapsed();
        println!(
            "Metadata scanning complete: {} files processed, {} errors, {:.2} files/sec",
            progress.processed_files,
            error_count,
            progress.processed_files as f64 / elapsed.as_secs_f64()
        );
        
        Ok(deduplicated_metadata)
    }

    /// Extract metadata for a single file (async version)
    pub async fn extract_metadata(&self, path: &Path) -> Result<FileMetadata> {
        let file_detector = FileTypeDetector::new();
        self.extract_metadata_with_detector(path, &file_detector).await
    }
    
    /// Extract metadata for a single file with provided detector
    pub async fn extract_metadata_with_detector(
        &self, 
        path: &Path, 
        file_detector: &FileTypeDetector
    ) -> Result<FileMetadata> {
        let std_metadata = tokio::fs::metadata(path).await
            .with_file_context(path)?;
        
        // Calculate relative path from root
        let relative_path = path.strip_prefix(&self.root_path)
            .unwrap_or(path)
            .to_path_buf();
        
        // Calculate directory depth
        let depth_level = relative_path.components().count() as u32;
        
        // Extract path components
        let folder_path = path.parent().unwrap_or(Path::new("")).to_path_buf();
        let filename = path.file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        let file_extension = path.extension()
            .map(|ext| ext.to_string_lossy().to_lowercase());
        
        // Check if file is hidden (starts with dot on Unix)
        let is_hidden = filename.starts_with('.');
        
        // Check if file is a symbolic link
        let is_symlink = std_metadata.file_type().is_symlink();
        let symlink_target = if is_symlink {
            tokio::fs::read_link(path).await.ok()
        } else {
            None
        };
        
        // Get file permissions (Unix-style)
        #[cfg(unix)]
        let permissions = std_metadata.permissions().mode();
        #[cfg(not(unix))]
        let permissions = 0; // Default for non-Unix systems
        
        // Convert system times to UTC
        let creation_date = std_metadata.created()
            .unwrap_or(SystemTime::UNIX_EPOCH)
            .into();
        let modification_date = std_metadata.modified()
            .unwrap_or(SystemTime::UNIX_EPOCH)
            .into();
        let access_date = std_metadata.accessed()
            .unwrap_or(SystemTime::UNIX_EPOCH)
            .into();
        
        // Determine file type
        let file_type = if std_metadata.is_dir() {
            FileType::Directory
        } else {
            FileType::File
        };
        
        // Calculate file hash (only for regular files)
        let hash = if file_type == FileType::File && !file_detector.should_exclude(path).await {
            HashCalculator::calculate_hash(path).await?
        } else {
            String::new() // Empty hash for directories or excluded files
        };
        
        Ok(FileMetadata {
            full_filepath: path.to_path_buf(),
            folder_path,
            filename,
            file_extension,
            file_type,
            size: std_metadata.len(),
            hash,
            creation_date,
            modification_date,
            access_date,
            permissions,
            depth_level,
            relative_path,
            is_hidden,
            is_symlink,
            symlink_target,
            duplicate_status: DuplicateStatus::Unique, // Will be updated in deduplication pass
            duplicate_group_id: None,
            processing_status: ProcessingStatus::Pending,
            estimated_tokens: None,
            processed_at: None,
            error_message: None,
        })
    }
    
    /// Synchronous version of metadata extraction for use with rayon
    fn extract_metadata_sync(&self, path: &Path, file_detector: &FileTypeDetector) -> Result<FileMetadata> {
        let std_metadata = std::fs::metadata(path)
            .with_file_context(path)?;
        
        // Calculate relative path from root
        let relative_path = path.strip_prefix(&self.root_path)
            .unwrap_or(path)
            .to_path_buf();
        
        // Calculate directory depth
        let depth_level = relative_path.components().count() as u32;
        
        // Extract path components
        let folder_path = path.parent().unwrap_or(Path::new("")).to_path_buf();
        let filename = path.file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        let file_extension = path.extension()
            .map(|ext| ext.to_string_lossy().to_lowercase());
        
        // Check if file is hidden (starts with dot on Unix)
        let is_hidden = filename.starts_with('.');
        
        // Check if file is a symbolic link
        let is_symlink = std_metadata.file_type().is_symlink();
        let symlink_target = if is_symlink {
            std::fs::read_link(path).ok()
        } else {
            None
        };
        
        // Get file permissions (Unix-style)
        #[cfg(unix)]
        let permissions = std_metadata.permissions().mode();
        #[cfg(not(unix))]
        let permissions = 0; // Default for non-Unix systems
        
        // Convert system times to UTC
        let creation_date = std_metadata.created()
            .unwrap_or(SystemTime::UNIX_EPOCH)
            .into();
        let modification_date = std_metadata.modified()
            .unwrap_or(SystemTime::UNIX_EPOCH)
            .into();
        let access_date = std_metadata.accessed()
            .unwrap_or(SystemTime::UNIX_EPOCH)
            .into();
        
        // Determine file type
        let file_type = if std_metadata.is_dir() {
            FileType::Directory
        } else {
            FileType::File
        };
        
        // Calculate file hash (only for regular files, synchronously)
        let hash = if file_type == FileType::File {
            // Use synchronous hash calculation for parallel processing
            HashCalculator::calculate_hash_sync(path)?
        } else {
            String::new() // Empty hash for directories
        };
        
        Ok(FileMetadata {
            full_filepath: path.to_path_buf(),
            folder_path,
            filename,
            file_extension,
            file_type,
            size: std_metadata.len(),
            hash,
            creation_date,
            modification_date,
            access_date,
            permissions,
            depth_level,
            relative_path,
            is_hidden,
            is_symlink,
            symlink_target,
            duplicate_status: DuplicateStatus::Unique, // Will be updated in deduplication pass
            duplicate_group_id: None,
            processing_status: ProcessingStatus::Pending,
            estimated_tokens: None,
            processed_at: None,
            error_message: None,
        })
    }
    
    /// Discover all files in the directory tree
    fn discover_files(&self) -> Result<Vec<std::path::PathBuf>> {
        let mut walker = WalkDir::new(&self.root_path);
        
        if let Some(max_depth) = self.max_depth {
            walker = walker.max_depth(max_depth);
        }
        
        if self.follow_symlinks {
            walker = walker.follow_links(true);
        }
        
        let files: Result<Vec<_>> = walker
            .into_iter()
            .filter_map(|entry| {
                match entry {
                    Ok(entry) => {
                        // Only include regular files, not directories
                        if entry.file_type().is_file() {
                            Some(Ok(entry.path().to_path_buf()))
                        } else {
                            None
                        }
                    }
                    Err(e) => Some(Err(crate::errors::PensieveError::Io(e.into()))),
                }
            })
            .collect();
        
        files
    }
    
    /// Detect duplicate files by hash and assign duplicate status
    fn detect_duplicates(&self, mut metadata: Vec<FileMetadata>) -> Vec<FileMetadata> {
        let mut hash_to_files: HashMap<String, Vec<usize>> = HashMap::new();
        
        // Group files by hash
        for (index, file_metadata) in metadata.iter().enumerate() {
            if !file_metadata.hash.is_empty() {
                hash_to_files
                    .entry(file_metadata.hash.clone())
                    .or_insert_with(Vec::new)
                    .push(index);
            }
        }
        
        let mut unique_count = 0;
        let mut duplicate_count = 0;
        
        // Assign duplicate status and group IDs
        for (hash, indices) in hash_to_files {
            if indices.len() == 1 {
                // Unique file
                metadata[indices[0]].duplicate_status = DuplicateStatus::Unique;
                unique_count += 1;
            } else {
                // Duplicate files - first one is canonical, rest are duplicates
                let group_id = Uuid::new_v4();
                
                for (i, &index) in indices.iter().enumerate() {
                    metadata[index].duplicate_group_id = Some(group_id);
                    if i == 0 {
                        metadata[index].duplicate_status = DuplicateStatus::Canonical;
                        unique_count += 1;
                    } else {
                        metadata[index].duplicate_status = DuplicateStatus::Duplicate;
                        duplicate_count += 1;
                    }
                }
            }
        }
        
        println!(
            "Deduplication complete: {} unique files, {} duplicates identified",
            unique_count, duplicate_count
        );
        
        metadata
    }
}

/// File type detector for classifying files
pub struct FileTypeDetector {
    /// MIME type detector for magic number analysis
    mime_detector: MimeDetector,
}

impl FileTypeDetector {
    /// Create a new file type detector
    pub fn new() -> Self {
        Self {
            mime_detector: MimeDetector::new(),
        }
    }

    /// Detect file type based on extension and content
    pub async fn detect_type(&self, path: &Path) -> Result<FileClassification> {
        // First check by file extension
        if let Some(classification) = self.classify_by_extension(path) {
            // For binary files, we trust the extension
            if matches!(classification, FileClassification::Binary) {
                return Ok(classification);
            }
            
            // For text files, verify with MIME type detection to catch mislabeled files
            if let Ok(mime_type) = self.mime_detector.detect_mime_type(path).await {
                if self.is_binary_mime_type(&mime_type) {
                    return Ok(FileClassification::Binary);
                }
            }
            
            return Ok(classification);
        }

        // If no extension match, use MIME type detection
        match self.mime_detector.detect_mime_type(path).await {
            Ok(mime_type) => {
                if self.is_binary_mime_type(&mime_type) {
                    Ok(FileClassification::Binary)
                } else if self.is_text_mime_type(&mime_type) {
                    // Default text files to Tier 1 native processing
                    Ok(FileClassification::Tier1Native)
                } else {
                    // Unknown MIME type, default to binary for safety
                    Ok(FileClassification::Binary)
                }
            }
            Err(_) => {
                // If MIME detection fails, default to binary for safety
                Ok(FileClassification::Binary)
            }
        }
    }

    /// Check if file should be excluded from processing
    pub async fn should_exclude(&self, path: &Path) -> bool {
        match self.detect_type(path).await {
            Ok(FileClassification::Binary) => true,
            Ok(_) => false,
            Err(_) => true, // Exclude files we can't classify
        }
    }

    /// Classify file based on extension
    fn classify_by_extension(&self, path: &Path) -> Option<FileClassification> {
        let extension = path.extension()?.to_str()?.to_lowercase();
        
        // Tier 1: Native Rust processing
        if TIER1_EXTENSIONS.contains(&extension.as_str()) {
            return Some(FileClassification::Tier1Native);
        }
        
        // Tier 2: External tool processing
        if TIER2_EXTENSIONS.contains(&extension.as_str()) {
            return Some(FileClassification::Tier2External);
        }
        
        // Binary exclusions
        if BINARY_EXTENSIONS.contains(&extension.as_str()) {
            return Some(FileClassification::Binary);
        }
        
        None
    }

    /// Check if MIME type indicates binary content
    fn is_binary_mime_type(&self, mime_type: &str) -> bool {
        BINARY_MIME_TYPES.iter().any(|&binary_mime| {
            mime_type.starts_with(binary_mime)
        })
    }

    /// Check if MIME type indicates text content
    fn is_text_mime_type(&self, mime_type: &str) -> bool {
        TEXT_MIME_TYPES.iter().any(|&text_mime| {
            mime_type.starts_with(text_mime)
        })
    }

    /// Get access to the MIME detector for testing
    pub fn mime_detector(&self) -> &MimeDetector {
        &self.mime_detector
    }
}

impl Default for FileTypeDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// MIME type detector using magic number analysis
pub struct MimeDetector;

impl MimeDetector {
    /// Create a new MIME detector
    pub fn new() -> Self {
        Self
    }

    /// Detect MIME type by reading file magic numbers
    pub async fn detect_mime_type(&self, path: &Path) -> Result<String> {
        use tokio::fs::File;
        use tokio::io::AsyncReadExt;

        // Read first 512 bytes for magic number detection
        let mut file = File::open(path).await
            .with_file_context(path)?;
        
        let mut buffer = [0u8; 512];
        let bytes_read = file.read(&mut buffer).await
            .with_file_context(path)?;
        
        // Use mime_guess as fallback for extension-based detection
        let extension_guess = mime_guess::from_path(path).first_or_octet_stream();
        
        // Perform magic number analysis
        let magic_mime = self.detect_by_magic_numbers(&buffer[..bytes_read]);
        
        // Prefer magic number detection over extension guess
        Ok(magic_mime.unwrap_or_else(|| extension_guess.to_string()))
    }

    /// Detect MIME type by analyzing magic numbers
    fn detect_by_magic_numbers(&self, data: &[u8]) -> Option<String> {
        if data.is_empty() {
            return None;
        }

        // PDF files
        if data.starts_with(b"%PDF") {
            return Some("application/pdf".to_string());
        }

        // ZIP-based formats (DOCX, XLSX, etc.)
        if data.starts_with(b"PK\x03\x04") || data.starts_with(b"PK\x05\x06") || data.starts_with(b"PK\x07\x08") {
            // Could be ZIP, DOCX, XLSX, etc. - need more analysis
            if data.len() > 30 {
                // Look for Office Open XML signatures
                let data_str = String::from_utf8_lossy(data);
                if data_str.contains("word/") {
                    return Some("application/vnd.openxmlformats-officedocument.wordprocessingml.document".to_string());
                }
                if data_str.contains("xl/") {
                    return Some("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet".to_string());
                }
                if data_str.contains("ppt/") {
                    return Some("application/vnd.openxmlformats-officedocument.presentationml.presentation".to_string());
                }
            }
            return Some("application/zip".to_string());
        }

        // Image formats
        if data.starts_with(b"\xFF\xD8\xFF") {
            return Some("image/jpeg".to_string());
        }
        if data.starts_with(b"\x89PNG\r\n\x1A\n") {
            return Some("image/png".to_string());
        }
        if data.starts_with(b"GIF87a") || data.starts_with(b"GIF89a") {
            return Some("image/gif".to_string());
        }
        if data.starts_with(b"RIFF") && data.len() > 8 && &data[8..12] == b"WEBP" {
            return Some("image/webp".to_string());
        }

        // Audio/Video formats
        if data.starts_with(b"ID3") || (data.len() > 2 && data[0] == 0xFF && (data[1] & 0xE0) == 0xE0) {
            return Some("audio/mpeg".to_string());
        }
        if data.starts_with(b"RIFF") && data.len() > 8 && &data[8..12] == b"WAVE" {
            return Some("audio/wav".to_string());
        }
        if data.starts_with(b"\x00\x00\x00\x18ftypmp4") || data.starts_with(b"\x00\x00\x00\x20ftypmp4") {
            return Some("video/mp4".to_string());
        }

        // Archive formats
        if data.starts_with(b"\x1F\x8B") {
            return Some("application/gzip".to_string());
        }
        if data.starts_with(b"Rar!\x1A\x07\x00") || data.starts_with(b"Rar!\x1A\x07\x01\x00") {
            return Some("application/x-rar-compressed".to_string());
        }
        if data.starts_with(b"7z\xBC\xAF\x27\x1C") {
            return Some("application/x-7z-compressed".to_string());
        }

        // Executable formats
        if data.starts_with(b"MZ") {
            return Some("application/x-msdownload".to_string());
        }
        if data.starts_with(b"\x7FELF") {
            return Some("application/x-executable".to_string());
        }
        if data.starts_with(b"\xFE\xED\xFA\xCE") || data.starts_with(b"\xFE\xED\xFA\xCF") ||
           data.starts_with(b"\xCE\xFA\xED\xFE") || data.starts_with(b"\xCF\xFA\xED\xFE") {
            return Some("application/x-mach-binary".to_string());
        }

        // Text formats - check for UTF-8 BOM or high ratio of printable characters
        if data.starts_with(b"\xEF\xBB\xBF") {
            return Some("text/plain".to_string());
        }

        // Check if content appears to be text
        if self.is_likely_text(data) {
            return Some("text/plain".to_string());
        }

        None
    }

    /// Heuristic to determine if data is likely text
    fn is_likely_text(&self, data: &[u8]) -> bool {
        if data.is_empty() {
            return false;
        }

        let mut printable_count = 0;
        let mut control_count = 0;

        for &byte in data.iter().take(512) {
            match byte {
                // Printable ASCII
                0x20..=0x7E => printable_count += 1,
                // Common whitespace
                0x09 | 0x0A | 0x0D => printable_count += 1,
                // Control characters
                0x00..=0x08 | 0x0B | 0x0C | 0x0E..=0x1F => control_count += 1,
                // High-bit characters (could be UTF-8)
                0x80..=0xFF => {
                    // Don't count against text, but don't count as printable either
                }
                _ => {}
            }
        }

        // Consider it text if:
        // 1. At least 70% of sampled bytes are printable
        // 2. Control characters are less than 5% of total
        let total_sampled = (printable_count + control_count).min(data.len());
        if total_sampled == 0 {
            return false;
        }

        let printable_ratio = printable_count as f64 / total_sampled as f64;
        let control_ratio = control_count as f64 / total_sampled as f64;

        printable_ratio >= 0.7 && control_ratio < 0.05
    }
}

impl Default for MimeDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// File classification for processing strategy
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FileClassification {
    /// Tier 1: Native Rust processing
    Tier1Native,
    /// Tier 2: External tool processing
    Tier2External,
    /// Binary file - should be excluded
    Binary,
}

// File extension mappings for different tiers

/// Tier 1 extensions: Native Rust processing
const TIER1_EXTENSIONS: &[&str] = &[
    // Plain text formats
    "txt", "md", "rst", "org", "adoc", "wiki",
    
    // Source code files
    "rs", "py", "js", "ts", "java", "go", "c", "cpp", "h", "hpp", "cc", "cxx",
    "php", "rb", "swift", "kt", "scala", "clj", "hs", "elm", "lua", "pl", "r", "m",
    
    // Configuration and data files
    "json", "yaml", "yml", "toml", "ini", "cfg", "env", "properties", "conf",
    
    // Web formats
    "html", "htm", "css", "xml", "svg",
    
    // Scripts
    "sh", "bash", "zsh", "fish", "ps1", "bat", "cmd",
    
    // Data formats
    "csv", "tsv", "log", "sql",
    
    // Documentation formats
    "tex", "bib",
    
    // Special files
    "dockerfile", "gitignore", "gitattributes", "makefile",
];

/// Tier 2 extensions: External tool processing
const TIER2_EXTENSIONS: &[&str] = &[
    // PDF documents
    "pdf",
    
    // Microsoft Office formats
    "docx", "xlsx", "pptx",
    
    // OpenDocument formats
    "odt", "ods", "odp",
    
    // Rich text formats
    "rtf",
    
    // E-book formats
    "epub", "mobi", "azw", "azw3", "fb2", "lit", "pdb", "tcr", "prc",
    
    // Legacy Office formats (if tools available)
    "doc", "xls", "ppt",
    
    // Apple formats
    "pages", "numbers", "key",
];

/// Binary extensions: Should be excluded
const BINARY_EXTENSIONS: &[&str] = &[
    // Image formats (note: svg is handled as text in Tier 1)
    "jpg", "jpeg", "png", "gif", "bmp", "tiff", "tif", "webp", "ico",
    "raw", "cr2", "nef", "arw", "dng", "psd", "ai", "eps",
    
    // Video formats (note: ts conflicts with TypeScript, prioritizing TS files)
    "mp4", "avi", "mov", "mkv", "wmv", "flv", "webm", "m4v", "3gp", "ogv",
    "mpg", "mpeg", "m2v", "vob", "mts", "m2ts",
    
    // Audio formats
    "mp3", "wav", "flac", "ogg", "aac", "wma", "m4a", "opus", "ape", "ac3",
    "dts", "amr", "au", "ra", "aiff",
    
    // Archive formats
    "zip", "tar", "gz", "bz2", "xz", "7z", "rar", "cab", "iso", "dmg",
    "pkg", "deb", "rpm", "msi", "appx",
    
    // Executable formats (note: bat/cmd conflicts with scripts, handling via MIME detection)
    "exe", "bin", "app", "run", "com", "scr", "vbs", "jar",
    
    // Library formats
    "dll", "so", "dylib", "a", "lib", "framework",
    
    // Database formats
    "db", "sqlite", "sqlite3", "mdb", "accdb",
    
    // Font formats
    "ttf", "otf", "woff", "woff2", "eot",
    
    // 3D and CAD formats
    "obj", "fbx", "dae", "3ds", "blend", "max", "dwg", "dxf",
    
    // Backup and temporary files
    "bak", "tmp", "temp", "cache", "swp", "swo", "orig", "rej",
];

/// MIME types that indicate binary content
const BINARY_MIME_TYPES: &[&str] = &[
    "image/",
    "video/",
    "audio/",
    "application/octet-stream",
    "application/x-executable",
    "application/x-msdownload",
    "application/x-mach-binary",
    "application/zip",
    "application/x-rar-compressed",
    "application/x-7z-compressed",
    "application/gzip",
    "application/x-tar",
    "application/x-bzip2",
    "application/x-xz",
    "font/",
];

/// MIME types that indicate text content
const TEXT_MIME_TYPES: &[&str] = &[
    "text/",
    "application/json",
    "application/xml",
    "application/javascript",
    "application/x-sh",
    "application/x-shellscript",
];

/// Hash calculator for file content
pub struct HashCalculator;

impl HashCalculator {
    /// Calculate SHA-256 hash of file content (async version with buffered I/O)
    pub async fn calculate_hash(path: &Path) -> Result<String> {
        let file = File::open(path).await
            .with_file_context(path)?;
        
        let mut reader = BufReader::with_capacity(64 * 1024, file); // 64KB buffer
        let mut hasher = Sha256::new();
        let mut buffer = vec![0u8; 64 * 1024]; // 64KB chunks
        
        loop {
            let bytes_read = reader.read(&mut buffer).await
                .with_file_context(path)?;
            
            if bytes_read == 0 {
                break; // End of file
            }
            
            hasher.update(&buffer[..bytes_read]);
        }
        
        let hash_bytes = hasher.finalize();
        Ok(format!("{:x}", hash_bytes))
    }
    
    /// Calculate SHA-256 hash of file content (synchronous version for parallel processing)
    pub fn calculate_hash_sync(path: &Path) -> Result<String> {
        use std::fs::File;
        use std::io::{BufReader, Read};
        
        let file = File::open(path)
            .with_file_context(path)?;
        
        let mut reader = BufReader::with_capacity(64 * 1024, file); // 64KB buffer
        let mut hasher = Sha256::new();
        let mut buffer = vec![0u8; 64 * 1024]; // 64KB chunks
        
        loop {
            let bytes_read = reader.read(&mut buffer)
                .with_file_context(path)?;
            
            if bytes_read == 0 {
                break; // End of file
            }
            
            hasher.update(&buffer[..bytes_read]);
        }
        
        let hash_bytes = hasher.finalize();
        Ok(format!("{:x}", hash_bytes))
    }
}

/// Progress reporter for scanning operations
pub struct ScanProgress {
    /// Total files discovered
    pub total_files: usize,
    /// Files processed so far
    pub processed_files: usize,
    /// Total bytes processed
    pub total_bytes: u64,
    /// Files per second processing rate
    pub files_per_second: f64,
    /// Estimated time remaining
    pub eta_seconds: Option<u64>,
    /// Start time for rate calculation
    start_time: Instant,
}

impl ScanProgress {
    /// Create new progress tracker
    pub fn new() -> Self {
        Self {
            total_files: 0,
            processed_files: 0,
            total_bytes: 0,
            files_per_second: 0.0,
            eta_seconds: None,
            start_time: Instant::now(),
        }
    }

    /// Update progress with new file processed
    pub fn update(&mut self, file_size: u64) {
        self.processed_files += 1;
        self.total_bytes += file_size;
        
        // Calculate processing rate
        let elapsed = self.start_time.elapsed().as_secs_f64();
        if elapsed > 0.0 {
            self.files_per_second = self.processed_files as f64 / elapsed;
            
            // Calculate ETA
            if self.total_files > 0 && self.files_per_second > 0.0 {
                let remaining_files = self.total_files - self.processed_files;
                self.eta_seconds = Some((remaining_files as f64 / self.files_per_second) as u64);
            }
        }
    }

    /// Get completion percentage
    pub fn completion_percentage(&self) -> f64 {
        if self.total_files == 0 {
            0.0
        } else {
            (self.processed_files as f64 / self.total_files as f64) * 100.0
        }
    }
    
    /// Format progress as human-readable string
    pub fn format_progress(&self) -> String {
        let percentage = self.completion_percentage();
        let mb_processed = self.total_bytes as f64 / (1024.0 * 1024.0);
        
        let eta_str = if let Some(eta) = self.eta_seconds {
            if eta < 60 {
                format!("{}s", eta)
            } else if eta < 3600 {
                format!("{}m{}s", eta / 60, eta % 60)
            } else {
                format!("{}h{}m", eta / 3600, (eta % 3600) / 60)
            }
        } else {
            "unknown".to_string()
        };
        
        format!(
            "{}/{} files ({:.1}%) | {:.1} MB | {:.1} files/sec | ETA: {}",
            self.processed_files,
            self.total_files,
            percentage,
            mb_processed,
            self.files_per_second,
            eta_str
        )
    }
}

impl Default for ScanProgress {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[tokio::test]
    async fn test_file_classification_by_extension() {
        let detector = FileTypeDetector::new();

        // Test Tier 1 extensions
        let rust_file = PathBuf::from("test.rs");
        assert_eq!(
            detector.classify_by_extension(&rust_file),
            Some(FileClassification::Tier1Native)
        );

        let python_file = PathBuf::from("script.py");
        assert_eq!(
            detector.classify_by_extension(&python_file),
            Some(FileClassification::Tier1Native)
        );

        let json_file = PathBuf::from("config.json");
        assert_eq!(
            detector.classify_by_extension(&json_file),
            Some(FileClassification::Tier1Native)
        );

        // Test Tier 2 extensions
        let pdf_file = PathBuf::from("document.pdf");
        assert_eq!(
            detector.classify_by_extension(&pdf_file),
            Some(FileClassification::Tier2External)
        );

        let docx_file = PathBuf::from("document.docx");
        assert_eq!(
            detector.classify_by_extension(&docx_file),
            Some(FileClassification::Tier2External)
        );

        // Test binary extensions
        let image_file = PathBuf::from("photo.jpg");
        assert_eq!(
            detector.classify_by_extension(&image_file),
            Some(FileClassification::Binary)
        );

        let video_file = PathBuf::from("movie.mp4");
        assert_eq!(
            detector.classify_by_extension(&video_file),
            Some(FileClassification::Binary)
        );

        // Test unknown extension
        let unknown_file = PathBuf::from("file.unknown");
        assert_eq!(detector.classify_by_extension(&unknown_file), None);
    }

    #[tokio::test]
    async fn test_mime_detection_magic_numbers() {
        let detector = MimeDetector::new();

        // Test PDF magic number
        let pdf_data = b"%PDF-1.4\n%\xE2\xE3\xCF\xD3";
        assert_eq!(
            detector.detect_by_magic_numbers(pdf_data),
            Some("application/pdf".to_string())
        );

        // Test JPEG magic number
        let jpeg_data = b"\xFF\xD8\xFF\xE0\x00\x10JFIF";
        assert_eq!(
            detector.detect_by_magic_numbers(jpeg_data),
            Some("image/jpeg".to_string())
        );

        // Test PNG magic number
        let png_data = b"\x89PNG\r\n\x1A\n\x00\x00\x00\rIHDR";
        assert_eq!(
            detector.detect_by_magic_numbers(png_data),
            Some("image/png".to_string())
        );

        // Test ZIP magic number (could be DOCX)
        let zip_data = b"PK\x03\x04\x14\x00\x00\x00";
        assert_eq!(
            detector.detect_by_magic_numbers(zip_data),
            Some("application/zip".to_string())
        );

        // Test text content
        let text_data = b"This is plain text content with normal characters.";
        assert_eq!(
            detector.detect_by_magic_numbers(text_data),
            Some("text/plain".to_string())
        );
    }

    #[tokio::test]
    async fn test_text_detection_heuristic() {
        let detector = MimeDetector::new();

        // Test clearly text content
        let text_data = b"Hello, world! This is a test file with normal text.";
        assert!(detector.is_likely_text(text_data));

        // Test content with some control characters but mostly text
        let mixed_data = b"Hello\tworld\nThis is text\r\nwith whitespace.";
        assert!(detector.is_likely_text(mixed_data));

        // Test binary content
        let binary_data = b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0A\x0B\x0C\x0D\x0E\x0F";
        assert!(!detector.is_likely_text(binary_data));

        // Test empty data
        let empty_data = b"";
        assert!(!detector.is_likely_text(empty_data));

        // Test high control character ratio
        let control_heavy = b"\x00\x01\x02text\x03\x04\x05";
        assert!(!detector.is_likely_text(control_heavy));
    }

    #[tokio::test]
    async fn test_file_type_detection_with_real_files() -> Result<()> {
        let detector = FileTypeDetector::new();

        // Create a temporary text file
        let mut text_file = NamedTempFile::new()?;
        text_file.write_all(b"This is a test text file with normal content.")?;
        let text_path = text_file.path();

        let classification = detector.detect_type(text_path).await?;
        // Should be classified as Tier1Native since it's a text file
        assert_eq!(classification, FileClassification::Tier1Native);

        // Create a temporary binary file
        let mut binary_file = NamedTempFile::new()?;
        binary_file.write_all(b"\x89PNG\r\n\x1A\n\x00\x00\x00\rIHDR\x00\x00")?;
        let binary_path = binary_file.path();

        let classification = detector.detect_type(binary_path).await?;
        // Should be classified as Binary due to PNG magic number
        assert_eq!(classification, FileClassification::Binary);

        Ok(())
    }

    #[tokio::test]
    async fn test_should_exclude_logic() -> Result<()> {
        let detector = FileTypeDetector::new();

        // Create a text file that should not be excluded
        let mut text_file = NamedTempFile::new()?;
        text_file.write_all(b"This is normal text content.")?;
        let text_path = text_file.path();

        assert!(!detector.should_exclude(text_path).await);

        // Create a binary file that should be excluded
        let mut binary_file = NamedTempFile::new()?;
        binary_file.write_all(b"\xFF\xD8\xFF\xE0\x00\x10JFIF")?; // JPEG magic
        let binary_path = binary_file.path();

        assert!(detector.should_exclude(binary_path).await);

        Ok(())
    }

    #[test]
    fn test_extension_mappings_completeness() {
        // Verify that our extension lists don't overlap
        let tier1_set: std::collections::HashSet<_> = TIER1_EXTENSIONS.iter().collect();
        let tier2_set: std::collections::HashSet<_> = TIER2_EXTENSIONS.iter().collect();
        let binary_set: std::collections::HashSet<_> = BINARY_EXTENSIONS.iter().collect();

        // Check for overlaps
        let tier1_tier2_overlap: Vec<_> = tier1_set.intersection(&tier2_set).collect();
        let tier1_binary_overlap: Vec<_> = tier1_set.intersection(&binary_set).collect();
        let tier2_binary_overlap: Vec<_> = tier2_set.intersection(&binary_set).collect();

        assert!(tier1_tier2_overlap.is_empty(), "Tier1 and Tier2 extensions overlap: {:?}", tier1_tier2_overlap);
        assert!(tier1_binary_overlap.is_empty(), "Tier1 and Binary extensions overlap: {:?}", tier1_binary_overlap);
        assert!(tier2_binary_overlap.is_empty(), "Tier2 and Binary extensions overlap: {:?}", tier2_binary_overlap);

        // Verify we have reasonable coverage
        assert!(TIER1_EXTENSIONS.len() > 20, "Should have substantial Tier1 coverage");
        assert!(TIER2_EXTENSIONS.len() > 5, "Should have reasonable Tier2 coverage");
        assert!(BINARY_EXTENSIONS.len() > 30, "Should have comprehensive binary exclusions");
    }

    #[test]
    fn test_mime_type_mappings() {
        // Verify MIME type arrays don't overlap inappropriately
        let binary_prefixes: Vec<_> = BINARY_MIME_TYPES.iter().collect();
        let text_prefixes: Vec<_> = TEXT_MIME_TYPES.iter().collect();

        // These should be mutually exclusive
        for &binary_prefix in &binary_prefixes {
            for &text_prefix in &text_prefixes {
                assert!(
                    !binary_prefix.starts_with(text_prefix) && !text_prefix.starts_with(binary_prefix),
                    "MIME type conflict: '{}' and '{}'", binary_prefix, text_prefix
                );
            }
        }
    }
}
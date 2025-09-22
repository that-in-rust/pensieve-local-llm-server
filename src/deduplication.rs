//! File-level deduplication logic and services

use crate::prelude::*;
use crate::types::{FileMetadata, DuplicateStatus};
use crate::database::Database;
use std::collections::HashMap;
use uuid::Uuid;
use chrono::{DateTime, Utc};

/// Service for handling file-level deduplication
pub struct DeduplicationService {
    /// Database connection for persistence
    database: Database,
}

impl DeduplicationService {
    /// Create a new deduplication service
    pub fn new(database: Database) -> Self {
        Self { database }
    }

    /// Process files for deduplication and assign duplicate status
    pub async fn process_duplicates(&self, mut files: Vec<FileMetadata>) -> Result<Vec<FileMetadata>> {
        if files.is_empty() {
            return Ok(files);
        }

        println!("Processing {} files for deduplication...", files.len());
        
        // Group files by hash
        let hash_groups = self.group_files_by_hash(&files);
        
        // Assign duplicate status
        self.assign_duplicate_status(&mut files, &hash_groups);
        
        // Generate statistics
        let stats = self.calculate_statistics(&files);
        self.print_statistics(&stats);
        
        Ok(files)
    }

    /// Group files by their content hash
    fn group_files_by_hash(&self, files: &[FileMetadata]) -> HashMap<String, Vec<usize>> {
        let mut hash_to_indices: HashMap<String, Vec<usize>> = HashMap::new();
        
        for (index, file) in files.iter().enumerate() {
            // Only group files with non-empty hashes (regular files)
            if !file.hash.is_empty() && file.file_type == crate::types::FileType::File {
                hash_to_indices
                    .entry(file.hash.clone())
                    .or_insert_with(Vec::new)
                    .push(index);
            }
        }
        
        hash_to_indices
    }

    /// Assign duplicate status to files based on hash groups
    fn assign_duplicate_status(
        &self, 
        files: &mut [FileMetadata], 
        hash_groups: &HashMap<String, Vec<usize>>
    ) {
        for (_hash, indices) in hash_groups {
            if indices.len() == 1 {
                // Single file with this hash - mark as unique
                files[indices[0]].duplicate_status = DuplicateStatus::Unique;
            } else {
                // Multiple files with same hash - mark as duplicates
                let group_id = Uuid::new_v4();
                
                // Choose canonical file using deterministic criteria
                let canonical_index = self.choose_canonical_file(files, indices);
                
                for &index in indices {
                    files[index].duplicate_group_id = Some(group_id);
                    
                    if index == canonical_index {
                        files[index].duplicate_status = DuplicateStatus::Canonical;
                    } else {
                        files[index].duplicate_status = DuplicateStatus::Duplicate;
                    }
                }
            }
        }
    }

    /// Choose the canonical file from a group of duplicates
    /// Criteria (in order of preference):
    /// 1. Shortest path (fewer directory levels)
    /// 2. Alphabetically first path (for deterministic results)
    /// 3. Most recent modification time (as tiebreaker)
    fn choose_canonical_file(&self, files: &[FileMetadata], indices: &[usize]) -> usize {
        let mut best_index = indices[0];
        let mut best_file = &files[best_index];
        
        for &index in indices.iter().skip(1) {
            let current_file = &files[index];
            
            // Compare by path depth (prefer shallower paths)
            let best_depth = best_file.depth_level;
            let current_depth = current_file.depth_level;
            
            if current_depth < best_depth {
                best_index = index;
                best_file = current_file;
                continue;
            } else if current_depth > best_depth {
                continue;
            }
            
            // Same depth - compare alphabetically
            let path_comparison = current_file.full_filepath.cmp(&best_file.full_filepath);
            match path_comparison {
                std::cmp::Ordering::Less => {
                    best_index = index;
                    best_file = current_file;
                }
                std::cmp::Ordering::Equal => {
                    // Identical paths shouldn't happen, but use modification time as tiebreaker
                    if current_file.modification_date > best_file.modification_date {
                        best_index = index;
                        best_file = current_file;
                    }
                }
                std::cmp::Ordering::Greater => {
                    // Keep current best
                }
            }
        }
        
        best_index
    }

    /// Calculate deduplication statistics
    fn calculate_statistics(&self, files: &[FileMetadata]) -> DeduplicationStats {
        let mut stats = DeduplicationStats::default();
        let mut duplicate_groups: std::collections::HashSet<Uuid> = std::collections::HashSet::new();
        
        for file in files {
            stats.total_files += 1;
            stats.total_size += file.size;
            
            match file.duplicate_status {
                DuplicateStatus::Unique => {
                    stats.unique_files += 1;
                }
                DuplicateStatus::Canonical => {
                    stats.canonical_files += 1;
                    if let Some(group_id) = file.duplicate_group_id {
                        duplicate_groups.insert(group_id);
                    }
                }
                DuplicateStatus::Duplicate => {
                    stats.duplicate_files += 1;
                    stats.duplicate_size += file.size;
                    if let Some(group_id) = file.duplicate_group_id {
                        duplicate_groups.insert(group_id);
                    }
                }
            }
        }
        
        stats.duplicate_groups = duplicate_groups.len() as u64;
        stats.effective_files = stats.unique_files + stats.canonical_files;
        stats.deduplication_ratio = if stats.total_files > 0 {
            stats.duplicate_files as f64 / stats.total_files as f64
        } else {
            0.0
        };
        
        stats
    }

    /// Print deduplication statistics
    fn print_statistics(&self, stats: &DeduplicationStats) {
        println!("\nDeduplication Results:");
        println!("  Total files processed: {}", stats.total_files);
        println!("  Unique files: {}", stats.unique_files);
        println!("  Canonical files: {}", stats.canonical_files);
        println!("  Duplicate files: {}", stats.duplicate_files);
        println!("  Duplicate groups: {}", stats.duplicate_groups);
        println!("  Effective files (after deduplication): {}", stats.effective_files);
        
        if stats.total_files > 0 {
            println!("  Deduplication rate: {:.1}%", stats.deduplication_ratio * 100.0);
        }
        
        if stats.duplicate_size > 0 {
            println!("  Space savings: {:.2} MB", stats.duplicate_size as f64 / 1_048_576.0);
            
            if stats.total_size > 0 {
                let space_savings_percentage = (stats.duplicate_size as f64 / stats.total_size as f64) * 100.0;
                println!("  Space savings percentage: {:.1}%", space_savings_percentage);
            }
        }
    }

    /// Get duplicate files by group ID
    pub async fn get_duplicate_group(&self, group_id: Uuid) -> Result<Vec<FileMetadata>> {
        let group_id_str = group_id.to_string();
        
        let rows = sqlx::query!(
            r#"
            SELECT file_id, full_filepath, folder_path, filename, file_extension, file_type,
                   size, hash, creation_date, modification_date, access_date, permissions,
                   depth_level, relative_path, is_hidden, is_symlink, symlink_target,
                   duplicate_status, duplicate_group_id, processing_status, estimated_tokens,
                   processed_at, error_message
            FROM files 
            WHERE duplicate_group_id = ?
            ORDER BY duplicate_status, full_filepath
            "#,
            group_id_str
        )
        .fetch_all(self.database.pool())
        .await
        .map_err(|e| PensieveError::Database(e))?;
        
        let mut files = Vec::new();
        for row in rows {
            let duplicate_status = match row.duplicate_status.as_str() {
                "unique" => DuplicateStatus::Unique,
                "canonical" => DuplicateStatus::Canonical,
                "duplicate" => DuplicateStatus::Duplicate,
                _ => DuplicateStatus::Unique,
            };
            
            let processing_status = match row.processing_status.as_str() {
                "pending" => crate::types::ProcessingStatus::Pending,
                "processed" => crate::types::ProcessingStatus::Processed,
                "error" => crate::types::ProcessingStatus::Error,
                "skipped_binary" => crate::types::ProcessingStatus::SkippedBinary,
                "skipped_dependency" => crate::types::ProcessingStatus::SkippedDependency,
                "deleted" => crate::types::ProcessingStatus::Deleted,
                _ => crate::types::ProcessingStatus::Pending,
            };
            
            let file_type = match row.file_type.as_str() {
                "file" => crate::types::FileType::File,
                "directory" => crate::types::FileType::Directory,
                _ => crate::types::FileType::File,
            };
            
            // Convert NaiveDateTime to DateTime<Utc>
            let creation_date = row.creation_date
                .map(|dt| DateTime::<Utc>::from_naive_utc_and_offset(dt, Utc))
                .unwrap_or_else(|| chrono::Utc::now());
            let modification_date = row.modification_date
                .map(|dt| DateTime::<Utc>::from_naive_utc_and_offset(dt, Utc))
                .unwrap_or_else(|| chrono::Utc::now());
            let access_date = row.access_date
                .map(|dt| DateTime::<Utc>::from_naive_utc_and_offset(dt, Utc))
                .unwrap_or_else(|| chrono::Utc::now());
            let processed_at = row.processed_at
                .map(|dt| DateTime::<Utc>::from_naive_utc_and_offset(dt, Utc));

            let metadata = FileMetadata {
                full_filepath: std::path::PathBuf::from(row.full_filepath),
                folder_path: std::path::PathBuf::from(row.folder_path),
                filename: row.filename,
                file_extension: row.file_extension,
                file_type,
                size: row.size as u64,
                hash: row.hash,
                creation_date,
                modification_date,
                access_date,
                permissions: row.permissions.unwrap_or(0) as u32,
                depth_level: row.depth_level as u32,
                relative_path: std::path::PathBuf::from(row.relative_path),
                is_hidden: row.is_hidden,
                is_symlink: row.is_symlink,
                symlink_target: row.symlink_target.map(std::path::PathBuf::from),
                duplicate_status,
                duplicate_group_id: row.duplicate_group_id.as_ref().and_then(|s| Uuid::parse_str(s).ok()),
                processing_status,
                estimated_tokens: row.estimated_tokens.map(|t| t as u32),
                processed_at,
                error_message: row.error_message,
            };
            
            files.push(metadata);
        }
        
        Ok(files)
    }

    /// List all duplicate groups with summary information
    pub async fn list_duplicate_groups(&self) -> Result<Vec<DuplicateGroupSummary>> {
        let rows = sqlx::query_as::<_, (Option<String>, i64, i64, String, String)>(
            r#"
            SELECT 
                duplicate_group_id,
                COUNT(*) as file_count,
                SUM(size) as total_size,
                MIN(full_filepath) as canonical_path,
                hash
            FROM files 
            WHERE duplicate_group_id IS NOT NULL
            GROUP BY duplicate_group_id, hash
            HAVING COUNT(*) > 1
            ORDER BY total_size DESC
            "#
        )
        .fetch_all(self.database.pool())
        .await
        .map_err(|e| PensieveError::Database(e))?;
        
        let mut groups = Vec::new();
        for (group_id_str, file_count, total_size, canonical_path, hash) in rows {
            if let Some(group_id_str) = group_id_str {
                if let Ok(group_id) = Uuid::parse_str(&group_id_str) {
                    let summary = DuplicateGroupSummary {
                        group_id,
                        file_count: file_count as u32,
                        total_size: total_size as u64,
                        canonical_path: std::path::PathBuf::from(canonical_path),
                        hash,
                    };
                    groups.push(summary);
                }
            }
        }
        
        Ok(groups)
    }

    /// Get database reference for advanced operations
    pub fn database(&self) -> &Database {
        &self.database
    }
}

/// Deduplication statistics
#[derive(Debug, Default)]
pub struct DeduplicationStats {
    /// Total number of files processed
    pub total_files: u64,
    /// Number of unique files (no duplicates)
    pub unique_files: u64,
    /// Number of canonical files (first in duplicate groups)
    pub canonical_files: u64,
    /// Number of duplicate files
    pub duplicate_files: u64,
    /// Number of duplicate groups
    pub duplicate_groups: u64,
    /// Effective number of files after deduplication
    pub effective_files: u64,
    /// Total size of all files
    pub total_size: u64,
    /// Total size of duplicate files
    pub duplicate_size: u64,
    /// Deduplication ratio (0.0 to 1.0)
    pub deduplication_ratio: f64,
}

/// Summary information for a duplicate group
#[derive(Debug, Clone)]
pub struct DuplicateGroupSummary {
    /// Unique group identifier
    pub group_id: Uuid,
    /// Number of files in the group
    pub file_count: u32,
    /// Total size of all files in the group
    pub total_size: u64,
    /// Path to the canonical file
    pub canonical_path: std::path::PathBuf,
    /// Content hash of the files
    pub hash: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{FileType, ProcessingStatus};
    use tempfile::NamedTempFile;
    use std::path::PathBuf;
    use chrono::Utc;

    async fn create_test_database() -> Result<Database> {
        let temp_file = NamedTempFile::new().unwrap();
        let db_path = temp_file.path();
        let db = Database::new(db_path).await?;
        db.initialize_schema().await?;
        Ok(db)
    }

    fn create_test_file(path: &str, hash: &str, size: u64) -> FileMetadata {
        let now = Utc::now();
        FileMetadata {
            full_filepath: PathBuf::from(path),
            folder_path: PathBuf::from("/test"),
            filename: path.split('/').last().unwrap_or(path).to_string(),
            file_extension: Some("txt".to_string()),
            file_type: FileType::File,
            size,
            hash: hash.to_string(),
            creation_date: now,
            modification_date: now,
            access_date: now,
            permissions: 644,
            depth_level: 1,
            relative_path: PathBuf::from(path),
            is_hidden: false,
            is_symlink: false,
            symlink_target: None,
            duplicate_status: DuplicateStatus::Unique,
            duplicate_group_id: None,
            processing_status: ProcessingStatus::Pending,
            estimated_tokens: None,
            processed_at: None,
            error_message: None,
        }
    }

    #[tokio::test]
    async fn test_unique_files_detection() {
        let db = create_test_database().await.unwrap();
        let service = DeduplicationService::new(db);
        
        let files = vec![
            create_test_file("/test/file1.txt", "hash1", 100),
            create_test_file("/test/file2.txt", "hash2", 200),
            create_test_file("/test/file3.txt", "hash3", 300),
        ];
        
        let result = service.process_duplicates(files).await.unwrap();
        
        assert_eq!(result.len(), 3);
        for file in &result {
            assert_eq!(file.duplicate_status, DuplicateStatus::Unique);
            assert!(file.duplicate_group_id.is_none());
        }
    }

    #[tokio::test]
    async fn test_duplicate_detection() {
        let db = create_test_database().await.unwrap();
        let service = DeduplicationService::new(db);
        
        let files = vec![
            create_test_file("/test/file1.txt", "hash1", 100),
            create_test_file("/test/subdir/file2.txt", "hash1", 100), // Duplicate of file1
            create_test_file("/test/file3.txt", "hash2", 200),
            create_test_file("/test/another/file4.txt", "hash1", 100), // Another duplicate of file1
        ];
        
        let result = service.process_duplicates(files).await.unwrap();
        
        assert_eq!(result.len(), 4);
        
        // Count by status
        let unique_count = result.iter().filter(|f| f.duplicate_status == DuplicateStatus::Unique).count();
        let canonical_count = result.iter().filter(|f| f.duplicate_status == DuplicateStatus::Canonical).count();
        let duplicate_count = result.iter().filter(|f| f.duplicate_status == DuplicateStatus::Duplicate).count();
        
        assert_eq!(unique_count, 1); // file3.txt
        assert_eq!(canonical_count, 1); // file1.txt (shortest path)
        assert_eq!(duplicate_count, 2); // file2.txt and file4.txt
        
        // Check that duplicates have the same group ID
        let duplicates: Vec<_> = result.iter()
            .filter(|f| f.duplicate_status != DuplicateStatus::Unique)
            .collect();
        
        let group_id = duplicates[0].duplicate_group_id.unwrap();
        for duplicate in &duplicates {
            assert_eq!(duplicate.duplicate_group_id, Some(group_id));
        }
    }

    #[tokio::test]
    async fn test_canonical_file_selection() {
        let db = create_test_database().await.unwrap();
        let service = DeduplicationService::new(db);
        
        let files = vec![
            create_test_file("/test/deep/nested/path/file.txt", "hash1", 100),
            create_test_file("/test/file.txt", "hash1", 100), // Should be canonical (shorter path)
            create_test_file("/test/another/file.txt", "hash1", 100),
        ];
        
        let result = service.process_duplicates(files).await.unwrap();
        
        // Find the canonical file
        let canonical = result.iter()
            .find(|f| f.duplicate_status == DuplicateStatus::Canonical)
            .unwrap();
        
        // Should be the file with the shortest path
        assert_eq!(canonical.full_filepath, PathBuf::from("/test/file.txt"));
    }

    #[tokio::test]
    async fn test_statistics_calculation() {
        let db = create_test_database().await.unwrap();
        let service = DeduplicationService::new(db);
        
        let files = vec![
            create_test_file("/test/file1.txt", "hash1", 100),
            create_test_file("/test/file2.txt", "hash1", 100), // Duplicate
            create_test_file("/test/file3.txt", "hash2", 200), // Unique
            create_test_file("/test/file4.txt", "hash3", 300),
            create_test_file("/test/file5.txt", "hash3", 300), // Duplicate
        ];
        
        let result = service.process_duplicates(files).await.unwrap();
        let stats = service.calculate_statistics(&result);
        
        assert_eq!(stats.total_files, 5);
        assert_eq!(stats.unique_files, 1); // file3.txt
        assert_eq!(stats.canonical_files, 2); // file1.txt, file4.txt
        assert_eq!(stats.duplicate_files, 2); // file2.txt, file5.txt
        assert_eq!(stats.duplicate_groups, 2);
        assert_eq!(stats.effective_files, 3); // unique + canonical
        assert_eq!(stats.duplicate_size, 400); // 100 + 300
        assert_eq!(stats.total_size, 1000); // 100 + 100 + 200 + 300 + 300
    }
}
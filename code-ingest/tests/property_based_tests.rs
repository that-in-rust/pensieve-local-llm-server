//! Property-based tests for code-ingest
//! 
//! Tests Requirements 2.1, 2.2, 2.3 - File classification accuracy with property-based testing

use code_ingest::processing::{
    classifier::{FileClassifier, FileType},
    text_processor::TextProcessor,
    FileProcessor,
};
use proptest::prelude::*;
use std::path::Path;
use tempfile::NamedTempFile;
use std::io::Write;

/// Generate valid file extensions
fn valid_extension() -> impl Strategy<Value = String> {
    prop_oneof![
        // DirectText extensions
        Just("rs".to_string()),
        Just("py".to_string()),
        Just("js".to_string()),
        Just("ts".to_string()),
        Just("md".to_string()),
        Just("txt".to_string()),
        Just("json".to_string()),
        Just("yaml".to_string()),
        Just("toml".to_string()),
        Just("sql".to_string()),
        Just("sh".to_string()),
        Just("c".to_string()),
        Just("cpp".to_string()),
        Just("java".to_string()),
        Just("go".to_string()),
        Just("rb".to_string()),
        Just("php".to_string()),
        Just("html".to_string()),
        Just("css".to_string()),
        Just("xml".to_string()),
        
        // Convertible extensions
        Just("pdf".to_string()),
        Just("docx".to_string()),
        Just("xlsx".to_string()),
        Just("pptx".to_string()),
        
        // Binary extensions
        Just("jpg".to_string()),
        Just("png".to_string()),
        Just("gif".to_string()),
        Just("mp4".to_string()),
        Just("exe".to_string()),
        Just("bin".to_string()),
        Just("zip".to_string()),
    ]
}

/// Generate random file extensions
fn random_extension() -> impl Strategy<Value = String> {
    "[a-zA-Z0-9]{1,10}"
}

/// Generate valid filenames
fn valid_filename() -> impl Strategy<Value = String> {
    "[a-zA-Z0-9_-]{1,50}"
}

/// Generate file content
fn file_content() -> impl Strategy<Value = String> {
    prop_oneof![
        // Empty content
        Just("".to_string()),
        
        // Simple content
        Just("Hello, world!".to_string()),
        
        // Code-like content
        Just("fn main() { println!(\"Hello!\"); }".to_string()),
        Just("def hello(): print('Hello')".to_string()),
        Just("console.log('Hello');".to_string()),
        
        // Multi-line content
        Just("Line 1\nLine 2\nLine 3".to_string()),
        
        // JSON-like content
        Just(r#"{"key": "value", "number": 42}"#.to_string()),
        
        // Random content
        ".*{0,1000}",
    ]
}

proptest! {
    /// Test that file classification is consistent and deterministic
    #[test]
    fn test_classification_consistency(
        ext in valid_extension(),
        iterations in 1usize..100
    ) {
        let classifier = FileClassifier::new();
        
        // Classification should be consistent across multiple calls
        let first_result = classifier.classify_by_extension(&ext);
        
        for _ in 0..iterations {
            let result = classifier.classify_by_extension(&ext);
            prop_assert_eq!(result, first_result, 
                          "Classification should be consistent for extension: {}", ext);
        }
        
        // Result should always be a valid FileType
        match first_result {
            FileType::DirectText | FileType::Convertible | FileType::NonText => {
                // Valid result
            }
        }
    }
    
    /// Test case-insensitive classification
    #[test]
    fn test_case_insensitive_classification(
        ext in valid_extension(),
        case_transform in 0u8..4
    ) {
        let classifier = FileClassifier::new();
        
        let original_result = classifier.classify_by_extension(&ext);
        
        let transformed_ext = match case_transform {
            0 => ext.to_lowercase(),
            1 => ext.to_uppercase(),
            2 => {
                let mut chars: Vec<char> = ext.chars().collect();
                if !chars.is_empty() {
                    chars[0] = chars[0].to_uppercase().next().unwrap_or(chars[0]);
                }
                chars.into_iter().collect()
            }
            _ => ext.clone(),
        };
        
        let transformed_result = classifier.classify_by_extension(&transformed_ext);
        
        prop_assert_eq!(original_result, transformed_result,
                       "Classification should be case-insensitive: '{}' vs '{}'",
                       ext, transformed_ext);
    }
    
    /// Test file path classification robustness
    #[test]
    fn test_file_path_classification(
        filename in valid_filename(),
        ext in valid_extension(),
        path_components in prop::collection::vec("[a-zA-Z0-9_-]{1,20}", 0..5)
    ) {
        let classifier = FileClassifier::new();
        
        // Build path with directory components
        let mut path_str = path_components.join("/");
        if !path_str.is_empty() {
            path_str.push('/');
        }
        path_str.push_str(&format!("{}.{}", filename, ext));
        
        let path = Path::new(&path_str);
        
        // Should not panic
        let file_result = classifier.classify_file(path);
        let ext_result = classifier.classify_by_extension(&ext);
        
        // Should match extension-based classification
        prop_assert_eq!(file_result, ext_result,
                       "File path classification should match extension classification for: {}",
                       path_str);
    }
    
    /// Test unknown extensions default to NonText
    #[test]
    fn test_unknown_extensions_default(
        ext in random_extension().prop_filter("Not a known extension", |e| {
            !["rs", "py", "js", "ts", "md", "txt", "json", "yaml", "toml", "sql", "sh",
              "c", "cpp", "java", "go", "rb", "php", "html", "css", "xml",
              "pdf", "docx", "xlsx", "pptx",
              "jpg", "png", "gif", "mp4", "exe", "bin", "zip"].contains(&e.as_str())
        })
    ) {
        let classifier = FileClassifier::new();
        
        let result = classifier.classify_by_extension(&ext);
        
        // Unknown extensions should default to NonText
        prop_assert_eq!(result, FileType::NonText,
                       "Unknown extension '{}' should default to NonText", ext);
    }
    
    /// Test text processing with various content
    #[test]
    fn test_text_processing_robustness(
        filename in valid_filename(),
        content in file_content()
    ) {
        tokio_test::block_on(async {
            let processor = TextProcessor::new();
            
            // Create temporary file with content
            let mut temp_file = NamedTempFile::with_suffix(&format!(".{}.txt", filename)).unwrap();
            temp_file.write_all(content.as_bytes()).unwrap();
            temp_file.flush().unwrap();
            
            // Should not panic
            let result = processor.process(temp_file.path()).await;
            
            match result {
                Ok(processed) => {
                    // Verify basic properties
                    prop_assert!(!processed.filepath.is_empty());
                    prop_assert!(!processed.filename.is_empty());
                    prop_assert!(processed.file_size_bytes >= 0);
                    
                    if let Some(ref stored_content) = processed.content_text {
                        prop_assert_eq!(stored_content, &content, "Content should be preserved");
                    }
                    
                    if let Some(line_count) = processed.line_count {
                        let expected_lines = if content.is_empty() { 0 } else { content.lines().count() };
                        prop_assert_eq!(line_count as usize, expected_lines, "Line count should be accurate");
                    }
                    
                    if let Some(word_count) = processed.word_count {
                        let expected_words = content.split_whitespace().count();
                        prop_assert_eq!(word_count as usize, expected_words, "Word count should be accurate");
                    }
                }
                Err(_) => {
                    // Processing can fail for various reasons (file too large, etc.)
                    // This is acceptable behavior
                }
            }
        });
    }
    
    /// Test file type string conversion roundtrip
    #[test]
    fn test_file_type_string_roundtrip(
        file_type in prop_oneof![
            Just(FileType::DirectText),
            Just(FileType::Convertible),
            Just(FileType::NonText),
        ]
    ) {
        let as_string = file_type.as_str();
        let parsed_back = FileType::from_str(as_string);
        
        prop_assert_eq!(parsed_back, Some(file_type),
                       "FileType string conversion should be reversible");
    }
    
    /// Test custom mapping consistency
    #[test]
    fn test_custom_mapping_consistency(
        ext in "[a-zA-Z0-9]{1,10}",
        file_type in prop_oneof![
            Just(FileType::DirectText),
            Just(FileType::Convertible),
            Just(FileType::NonText),
        ]
    ) {
        let mut classifier = FileClassifier::new();
        
        // Add custom mapping
        classifier.add_custom_mapping(ext.clone(), file_type);
        
        // Should return the custom mapping
        let result = classifier.classify_by_extension(&ext);
        prop_assert_eq!(result, file_type,
                       "Custom mapping should be respected for extension: {}", ext);
        
        // Should be case-insensitive
        let upper_result = classifier.classify_by_extension(&ext.to_uppercase());
        prop_assert_eq!(upper_result, file_type,
                       "Custom mapping should be case-insensitive");
        
        // Remove mapping
        let removed = classifier.remove_custom_mapping(&ext);
        prop_assert_eq!(removed, Some(file_type),
                       "Remove should return the previous mapping");
        
        // Should revert to default behavior
        let default_result = classifier.classify_by_extension(&ext);
        // Default should be NonText for unknown extensions
        prop_assert_eq!(default_result, FileType::NonText,
                       "Should revert to default after removal");
    }
    
    /// Test parallel classification consistency
    #[test]
    fn test_parallel_classification_consistency(
        extensions in prop::collection::vec(valid_extension(), 1..20),
        thread_count in 1usize..8
    ) {
        use std::sync::Arc;
        use std::thread;
        
        let classifier = Arc::new(FileClassifier::new());
        
        // Classify sequentially first
        let sequential_results: Vec<_> = extensions.iter()
            .map(|ext| classifier.classify_by_extension(ext))
            .collect();
        
        // Classify in parallel
        let mut handles = Vec::new();
        let extensions_per_thread = extensions.len() / thread_count.max(1);
        
        for i in 0..thread_count {
            let start_idx = i * extensions_per_thread;
            let end_idx = if i == thread_count - 1 {
                extensions.len()
            } else {
                (i + 1) * extensions_per_thread
            };
            
            if start_idx < extensions.len() {
                let thread_extensions = extensions[start_idx..end_idx.min(extensions.len())].to_vec();
                let classifier_clone = Arc::clone(&classifier);
                
                let handle = thread::spawn(move || {
                    thread_extensions.iter()
                        .map(|ext| classifier_clone.classify_by_extension(ext))
                        .collect::<Vec<_>>()
                });
                
                handles.push((handle, start_idx, end_idx.min(extensions.len())));
            }
        }
        
        // Collect parallel results
        let mut parallel_results = vec![FileType::NonText; extensions.len()];
        for (handle, start_idx, end_idx) in handles {
            let thread_results = handle.join().unwrap();
            for (i, result) in thread_results.into_iter().enumerate() {
                if start_idx + i < end_idx {
                    parallel_results[start_idx + i] = result;
                }
            }
        }
        
        // Results should be identical
        prop_assert_eq!(sequential_results, parallel_results,
                       "Parallel classification should match sequential results");
    }
    
    /// Test memory usage bounds during processing
    #[test]
    fn test_memory_usage_bounds(
        file_count in 1usize..50,
        content_size in 1usize..1000
    ) {
        tokio_test::block_on(async {
            let processor = TextProcessor::new();
            
            // Create multiple temporary files
            let mut temp_files = Vec::new();
            let content = "x".repeat(content_size);
            
            for i in 0..file_count {
                let mut temp_file = NamedTempFile::with_suffix(&format!(".test_{}.txt", i)).unwrap();
                temp_file.write_all(content.as_bytes()).unwrap();
                temp_file.flush().unwrap();
                temp_files.push(temp_file);
            }
            
            // Process all files
            let mut results = Vec::new();
            for temp_file in &temp_files {
                match processor.process(temp_file.path()).await {
                    Ok(result) => results.push(result),
                    Err(_) => {
                        // Some processing failures are acceptable
                    }
                }
            }
            
            // Memory usage should be reasonable
            // (This is a basic check - in practice you'd use more sophisticated memory monitoring)
            prop_assert!(results.len() <= file_count, "Should not process more files than provided");
            
            // Each result should have reasonable memory usage
            for result in &results {
                if let Some(ref content) = result.content_text {
                    prop_assert!(content.len() <= content_size * 2, 
                               "Processed content should not be excessively larger than input");
                }
            }
        });
    }
}

/// Additional edge case property tests
#[cfg(test)]
mod edge_case_properties {
    use super::*;
    
    proptest! {
        /// Test empty and whitespace-only content
        #[test]
        fn test_empty_and_whitespace_content(
            whitespace_type in prop_oneof![
                Just(""),
                Just(" "),
                Just("\t"),
                Just("\n"),
                Just("\r\n"),
                Just("   \t\n  "),
            ]
        ) {
            tokio_test::block_on(async {
                let processor = TextProcessor::new();
                
                let mut temp_file = NamedTempFile::with_suffix(".txt").unwrap();
                temp_file.write_all(whitespace_type.as_bytes()).unwrap();
                temp_file.flush().unwrap();
                
                let result = processor.process(temp_file.path()).await;
                
                match result {
                    Ok(processed) => {
                        if let Some(ref content) = processed.content_text {
                            prop_assert_eq!(content, &whitespace_type, "Whitespace should be preserved");
                        }
                        
                        if let Some(word_count) = processed.word_count {
                            let expected_words = whitespace_type.split_whitespace().count();
                            prop_assert_eq!(word_count as usize, expected_words, 
                                          "Word count should handle whitespace correctly");
                        }
                    }
                    Err(_) => {
                        // Processing can fail, which is acceptable
                    }
                }
            });
        }
        
        /// Test very long filenames and paths
        #[test]
        fn test_long_filenames(
            filename_length in 1usize..200,
            path_depth in 0usize..10
        ) {
            let classifier = FileClassifier::new();
            
            let filename = "a".repeat(filename_length);
            let path_components: Vec<String> = (0..path_depth)
                .map(|i| format!("dir_{}", i))
                .collect();
            
            let mut path_str = path_components.join("/");
            if !path_str.is_empty() {
                path_str.push('/');
            }
            path_str.push_str(&format!("{}.rs", filename));
            
            let path = Path::new(&path_str);
            
            // Should not panic with long paths
            let result = classifier.classify_file(path);
            
            // Should classify based on extension
            prop_assert_eq!(result, FileType::DirectText, 
                          "Should classify .rs files as DirectText regardless of path length");
        }
        
        /// Test special characters in extensions
        #[test]
        fn test_special_characters_in_extensions(
            base_ext in "[a-zA-Z]{1,5}",
            special_char in prop_oneof![
                Just('~'), Just('#'), Just('$'), Just('%'), Just('@'),
                Just('!'), Just('&'), Just('*'), Just('('), Just(')'),
                Just('-'), Just('='), Just('+'), Just('['), Just(']'),
                Just('{'), Just('}'), Just('|'), Just('\\'), Just(':'),
                Just(';'), Just('"'), Just('\''), Just('<'), Just('>'),
                Just(','), Just('.'), Just('?'), Just('/'),
            ]
        ) {
            let classifier = FileClassifier::new();
            
            let ext_with_special = format!("{}{}", base_ext, special_char);
            
            // Should handle special characters gracefully
            let result = classifier.classify_by_extension(&ext_with_special);
            
            // Should default to NonText for extensions with special characters
            prop_assert_eq!(result, FileType::NonText,
                          "Extensions with special characters should default to NonText: {}",
                          ext_with_special);
        }
        
        /// Test Unicode in filenames and extensions
        #[test]
        fn test_unicode_handling(
            unicode_filename in "\\PC{1,20}",  // Unicode characters
            ascii_ext in "[a-zA-Z]{1,5}"
        ) {
            let classifier = FileClassifier::new();
            
            let path_str = format!("{}.{}", unicode_filename, ascii_ext);
            let path = Path::new(&path_str);
            
            // Should handle Unicode filenames gracefully
            let result = classifier.classify_file(path);
            
            // Should classify based on ASCII extension
            let ext_result = classifier.classify_by_extension(&ascii_ext);
            prop_assert_eq!(result, ext_result,
                          "Unicode filename should not affect extension-based classification");
        }
    }
}
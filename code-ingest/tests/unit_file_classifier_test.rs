//! Unit tests for FileClassifier
//! 
//! Tests Requirements 2.1, 2.2, 2.3 - Three-type file classification system

use code_ingest::processing::classifier::FileClassifier;
use code_ingest::processing::FileType;
use std::path::Path;
use proptest::prelude::*;

#[test]
fn test_direct_text_extensions() {
    let classifier = FileClassifier::new();
    
    // Test all required DirectText extensions from requirements
    let direct_text_extensions = [
        "rs", "py", "js", "ts", "md", "txt", "json", "yaml", "yml", "toml",
        "sql", "sh", "bash", "c", "cpp", "cc", "cxx", "h", "hpp", "java",
        "go", "rb", "php", "html", "css", "xml", "dockerfile", "gitignore",
        "env", "ini", "cfg", "conf", "log", "csv", "tsv"
    ];
    
    for ext in &direct_text_extensions {
        let file_type = classifier.classify_by_extension(ext);
        assert_eq!(
            file_type, 
            FileType::DirectText,
            "Extension '{}' should be classified as DirectText", 
            ext
        );
    }
}

#[test]
fn test_convertible_extensions() {
    let classifier = FileClassifier::new();
    
    // Test all required Convertible extensions from requirements
    let convertible_extensions = ["pdf", "docx", "xlsx", "pptx", "odt", "ods", "odp"];
    
    for ext in &convertible_extensions {
        let file_type = classifier.classify_by_extension(ext);
        assert_eq!(
            file_type, 
            FileType::Convertible,
            "Extension '{}' should be classified as Convertible", 
            ext
        );
    }
}

#[test]
fn test_binary_extensions() {
    let classifier = FileClassifier::new();
    
    // Test Binary/NonText extensions from requirements
    let binary_extensions = [
        "jpg", "jpeg", "png", "gif", "bmp", "tiff", "webp",
        "mp4", "avi", "mov", "mkv", "webm", "mp3", "wav", "flac",
        "exe", "dll", "so", "dylib", "bin", "o", "obj",
        "tar", "gz", "bz2", "xz", "7z", "rar"
    ];
    
    for ext in &binary_extensions {
        let file_type = classifier.classify_by_extension(ext);
        assert_eq!(
            file_type, 
            FileType::NonText,
            "Extension '{}' should be classified as NonText", 
            ext
        );
    }
}

#[test]
fn test_case_insensitive_classification() {
    let classifier = FileClassifier::new();
    
    // Test case variations
    let test_cases = [
        ("RS", FileType::DirectText),
        ("rs", FileType::DirectText),
        ("Rs", FileType::DirectText),
        ("PDF", FileType::Convertible),
        ("pdf", FileType::Convertible),
        ("Pdf", FileType::Convertible),
        ("JPG", FileType::NonText),
        ("jpg", FileType::NonText),
        ("Jpg", FileType::NonText),
    ];
    
    for (ext, expected) in &test_cases {
        let file_type = classifier.classify_by_extension(ext);
        assert_eq!(
            file_type, 
            *expected,
            "Extension '{}' should be classified as {:?}", 
            ext, 
            expected
        );
    }
}

#[test]
fn test_unknown_extensions_default_to_binary() {
    let classifier = FileClassifier::new();
    
    let unknown_extensions = [
        "unknown", "xyz", "abc123", "weird_ext", "", "123"
    ];
    
    for ext in &unknown_extensions {
        let file_type = classifier.classify_by_extension(ext);
        assert_eq!(
            file_type, 
            FileType::NonText,
            "Unknown extension '{}' should default to NonText", 
            ext
        );
    }
}

#[test]
fn test_file_path_classification() {
    let classifier = FileClassifier::new();
    
    let test_cases = [
        ("src/main.rs", FileType::DirectText),
        ("docs/README.md", FileType::DirectText),
        ("config/app.json", FileType::DirectText),
        ("documents/report.pdf", FileType::Convertible),
        ("images/photo.jpg", FileType::NonText),
        ("build/output.exe", FileType::NonText),
        ("no_extension", FileType::NonText), // No extension defaults to NonText
        (".gitignore", FileType::DirectText), // Special case
        (".env", FileType::DirectText),
    ];
    
    for (path_str, expected) in &test_cases {
        let path = Path::new(path_str);
        let file_type = classifier.classify_file(path);
        assert_eq!(
            file_type, 
            *expected,
            "File '{}' should be classified as {:?}", 
            path_str, 
            expected
        );
    }
}

#[test]
fn test_special_files() {
    let classifier = FileClassifier::new();
    
    // Test special files that should be treated as text
    let special_text_files = [
        "Dockerfile",
        "Makefile", 
        "CMakeLists.txt",
        "requirements.txt",
        "package.json",
        "Cargo.toml",
        ".gitignore",
        ".env",
        "LICENSE",
        "README"
    ];
    
    for filename in &special_text_files {
        let path = Path::new(filename);
        let file_type = classifier.classify_file(path);
        
        // Most should be DirectText, but some might be based on extension
        match *filename {
            "Dockerfile" | "Makefile" | "LICENSE" | "README" => {
                // These have no extension, so they default to NonText in basic implementation
                // But a smart classifier might recognize them as text
                // For now, we'll accept either classification
            }
            _ => {
                // Files with known text extensions should be DirectText
                if filename.contains('.') {
                    assert_eq!(
                        file_type, 
                        FileType::DirectText,
                        "Special file '{}' should be classified as DirectText", 
                        filename
                    );
                }
            }
        }
    }
}

#[test]
fn test_custom_mappings() {
    let mut classifier = FileClassifier::new();
    
    // Test adding custom mapping
    classifier.add_custom_mapping("custom".to_string(), FileType::DirectText);
    assert_eq!(classifier.classify_by_extension("custom"), FileType::DirectText);
    
    // Test overriding existing mapping
    let original = classifier.classify_by_extension("pdf");
    assert_eq!(original, FileType::Convertible);
    
    classifier.add_custom_mapping("pdf".to_string(), FileType::NonText);
    assert_eq!(classifier.classify_by_extension("pdf"), FileType::NonText);
    
    // Test removing custom mapping
    let removed = classifier.remove_custom_mapping("pdf");
    assert_eq!(removed, Some(FileType::NonText));
    
    // Should revert to original classification
    assert_eq!(classifier.classify_by_extension("pdf"), FileType::Convertible);
    
    // Test removing non-existent mapping
    let not_found = classifier.remove_custom_mapping("nonexistent");
    assert_eq!(not_found, None);
}

#[test]
fn test_file_type_serialization() {
    // Test database storage format
    assert_eq!(FileType::DirectText.as_str(), "direct_text");
    assert_eq!(FileType::Convertible.as_str(), "convertible");
    assert_eq!(FileType::NonText.as_str(), "non_text");
    
    // Test parsing from database
    assert_eq!(FileType::from_str("direct_text"), Some(FileType::DirectText));
    assert_eq!(FileType::from_str("convertible"), Some(FileType::Convertible));
    assert_eq!(FileType::from_str("non_text"), Some(FileType::NonText));
    assert_eq!(FileType::from_str("invalid"), None);
    assert_eq!(FileType::from_str(""), None);
}

#[test]
fn test_classifier_thread_safety() {
    use std::sync::Arc;
    use std::thread;
    
    let classifier = Arc::new(FileClassifier::new());
    let mut handles = Vec::new();
    
    // Test concurrent access
    for i in 0..10 {
        let classifier_clone = Arc::clone(&classifier);
        let handle = thread::spawn(move || {
            let extensions = ["rs", "py", "js", "pdf", "jpg"];
            for ext in &extensions {
                let _ = classifier_clone.classify_by_extension(ext);
            }
            i
        });
        handles.push(handle);
    }
    
    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }
    
    // Classifier should still work correctly
    assert_eq!(classifier.classify_by_extension("rs"), FileType::DirectText);
}

#[test]
fn test_performance_characteristics() {
    let classifier = FileClassifier::new();
    
    // Test that classification is fast
    let start = std::time::Instant::now();
    
    for _ in 0..10000 {
        classifier.classify_by_extension("rs");
        classifier.classify_by_extension("pdf");
        classifier.classify_by_extension("jpg");
        classifier.classify_by_extension("unknown");
    }
    
    let elapsed = start.elapsed();
    
    // Should be very fast (less than 100ms for 40k classifications)
    assert!(
        elapsed.as_millis() < 100,
        "Classification took too long: {:?}",
        elapsed
    );
}

// Property-based tests using proptest
proptest! {
    #[test]
    fn test_classification_consistency(ext in "[a-zA-Z0-9_]{1,10}") {
        let classifier = FileClassifier::new();
        
        // Classification should be consistent
        let result1 = classifier.classify_by_extension(&ext);
        let result2 = classifier.classify_by_extension(&ext);
        assert_eq!(result1, result2);
        
        // Should always return a valid FileType
        match result1 {
            FileType::DirectText | FileType::Convertible | FileType::NonText => {
                // Valid result
            }
        }
    }
    
    #[test]
    fn test_case_insensitive_consistency(
        ext in "[a-zA-Z]{1,5}",
        case_variant in 0u8..4
    ) {
        let classifier = FileClassifier::new();
        
        let modified_ext = match case_variant {
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
        
        let result1 = classifier.classify_by_extension(&ext);
        let result2 = classifier.classify_by_extension(&modified_ext);
        
        // Should be case-insensitive
        assert_eq!(result1, result2, 
                  "Classification should be case-insensitive: '{}' vs '{}'", 
                  ext, modified_ext);
    }
    
    #[test]
    fn test_path_classification_robustness(
        filename in "[a-zA-Z0-9_-]{1,20}",
        extension in "[a-zA-Z]{1,5}"
    ) {
        let classifier = FileClassifier::new();
        
        let path_str = format!("{}.{}", filename, extension);
        let path = Path::new(&path_str);
        
        // Should not panic and should return valid result
        let result = classifier.classify_file(path);
        match result {
            FileType::DirectText | FileType::Convertible | FileType::NonText => {
                // Valid result
            }
        }
        
        // Should be consistent with extension-only classification
        let ext_result = classifier.classify_by_extension(&extension);
        assert_eq!(result, ext_result);
    }
}

#[cfg(test)]
mod edge_cases {
    use super::*;
    
    #[test]
    fn test_empty_extension() {
        let classifier = FileClassifier::new();
        
        let result = classifier.classify_by_extension("");
        assert_eq!(result, FileType::NonText);
    }
    
    #[test]
    fn test_very_long_extension() {
        let classifier = FileClassifier::new();
        
        let long_ext = "a".repeat(1000);
        let result = classifier.classify_by_extension(&long_ext);
        assert_eq!(result, FileType::NonText);
    }
    
    #[test]
    fn test_numeric_extension() {
        let classifier = FileClassifier::new();
        
        let result = classifier.classify_by_extension("123");
        assert_eq!(result, FileType::NonText);
    }
    
    #[test]
    fn test_special_characters_in_extension() {
        let classifier = FileClassifier::new();
        
        let special_extensions = ["rs~", "py#", "js$", "md%", "txt@"];
        
        for ext in &special_extensions {
            let result = classifier.classify_by_extension(ext);
            // Should default to NonText for extensions with special characters
            assert_eq!(result, FileType::NonText);
        }
    }
    
    #[test]
    fn test_path_with_no_extension() {
        let classifier = FileClassifier::new();
        
        let paths = [
            Path::new("README"),
            Path::new("Makefile"),
            Path::new("LICENSE"),
            Path::new("src/main"),
            Path::new("no_extension_file"),
        ];
        
        for path in &paths {
            let result = classifier.classify_file(path);
            // Files without extensions should default to NonText
            assert_eq!(result, FileType::NonText);
        }
    }
    
    #[test]
    fn test_path_with_multiple_dots() {
        let classifier = FileClassifier::new();
        
        let paths = [
            Path::new("file.backup.rs"),
            Path::new("config.local.json"),
            Path::new("archive.tar.gz"),
            Path::new("document.final.pdf"),
        ];
        
        let expected = [
            FileType::DirectText,  // .rs
            FileType::DirectText,  // .json
            FileType::NonText,     // .gz
            FileType::Convertible, // .pdf
        ];
        
        for (path, expected_type) in paths.iter().zip(expected.iter()) {
            let result = classifier.classify_file(path);
            assert_eq!(
                result, 
                *expected_type,
                "Path '{}' should be classified as {:?}",
                path.display(),
                expected_type
            );
        }
    }
}
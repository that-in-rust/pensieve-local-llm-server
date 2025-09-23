//! Integration tests for file type detection system

use pensieve::scanner::{FileTypeDetector, FileClassification};
use std::io::Write;
use tempfile::NamedTempFile;

#[tokio::test]
async fn test_comprehensive_file_type_detection() -> Result<(), Box<dyn std::error::Error>> {
    let detector = FileTypeDetector::new();

    // Test 1: Rust source file (Tier 1)
    let mut rust_file = NamedTempFile::with_suffix(".rs")?;
    rust_file.write_all(b"fn main() { println!(\"Hello, world!\"); }")?;
    
    let classification = detector.detect_type(rust_file.path()).await?;
    assert_eq!(classification, FileClassification::Tier1Native);
    assert!(!detector.should_exclude(rust_file.path()).await);

    // Test 2: JSON configuration file (Tier 1)
    let mut json_file = NamedTempFile::with_suffix(".json")?;
    json_file.write_all(b"{\"name\": \"test\", \"version\": \"1.0.0\"}")?;
    
    let classification = detector.detect_type(json_file.path()).await?;
    assert_eq!(classification, FileClassification::Tier1Native);
    assert!(!detector.should_exclude(json_file.path()).await);

    // Test 3: PDF file (Tier 2)
    let mut pdf_file = NamedTempFile::with_suffix(".pdf")?;
    pdf_file.write_all(b"%PDF-1.4\n%\xE2\xE3\xCF\xD3\nSome PDF content here")?;
    
    let classification = detector.detect_type(pdf_file.path()).await?;
    assert_eq!(classification, FileClassification::Tier2External);
    assert!(!detector.should_exclude(pdf_file.path()).await);

    // Test 4: JPEG image (Binary - should be excluded)
    let mut jpeg_file = NamedTempFile::with_suffix(".jpg")?;
    jpeg_file.write_all(b"\xFF\xD8\xFF\xE0\x00\x10JFIF\x00\x01\x01\x01")?;
    
    let classification = detector.detect_type(jpeg_file.path()).await?;
    assert_eq!(classification, FileClassification::Binary);
    assert!(detector.should_exclude(jpeg_file.path()).await);

    // Test 5: Text file with binary magic number (should detect as binary)
    let mut fake_text_file = NamedTempFile::with_suffix(".txt")?;
    fake_text_file.write_all(b"\x89PNG\r\n\x1A\n\x00\x00\x00\rIHDR")?;
    
    let classification = detector.detect_type(fake_text_file.path()).await?;
    assert_eq!(classification, FileClassification::Binary);
    assert!(detector.should_exclude(fake_text_file.path()).await);

    // Test 6: Unknown extension with text content
    let mut unknown_file = NamedTempFile::with_suffix(".unknown")?;
    unknown_file.write_all(b"This is clearly text content with normal characters.")?;
    
    let classification = detector.detect_type(unknown_file.path()).await?;
    assert_eq!(classification, FileClassification::Tier1Native);
    assert!(!detector.should_exclude(unknown_file.path()).await);

    // Test 7: Unknown extension with binary content
    let mut unknown_binary = NamedTempFile::with_suffix(".mystery")?;
    unknown_binary.write_all(b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09")?;
    
    let classification = detector.detect_type(unknown_binary.path()).await?;
    assert_eq!(classification, FileClassification::Binary);
    assert!(detector.should_exclude(unknown_binary.path()).await);

    println!("✅ All file type detection tests passed!");
    Ok(())
}

#[tokio::test]
async fn test_mime_type_detection_accuracy() -> Result<(), Box<dyn std::error::Error>> {
    let detector = FileTypeDetector::new();

    // Test various file formats with their magic numbers
    let test_cases = vec![
        // (content, expected_mime_prefix, description)
        (b"%PDF-1.4".as_slice(), "application/pdf", "PDF file"),
        (b"\xFF\xD8\xFF\xE0".as_slice(), "image/jpeg", "JPEG image"),
        (b"\x89PNG\r\n\x1A\n".as_slice(), "image/png", "PNG image"),
        (b"GIF89a".as_slice(), "image/gif", "GIF image"),
        (b"PK\x03\x04".as_slice(), "application/zip", "ZIP archive"),
        (b"This is plain text".as_slice(), "text/plain", "Plain text"),
    ];

    for (content, expected_prefix, description) in test_cases {
        let mut temp_file = NamedTempFile::new()?;
        temp_file.write_all(content)?;
        
        let mime_type = detector.mime_detector().detect_mime_type(temp_file.path()).await?;
        assert!(
            mime_type.starts_with(expected_prefix),
            "Failed for {}: expected '{}' but got '{}'",
            description, expected_prefix, mime_type
        );
    }

    println!("✅ All MIME type detection tests passed!");
    Ok(())
}
// Simple test file to verify extractor functionality
use std::fs;
use std::io::Write;
use tempfile::TempDir;

// Copy the extractor module content here for isolated testing
mod extractor {
    use std::path::Path;
    use std::time::Duration;
    use async_trait::async_trait;
    use encoding_rs::WINDOWS_1252;
    use scraper::{Html, Selector};

    pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

    #[async_trait]
    pub trait ContentExtractor: Send + Sync {
        async fn extract(&self, file_path: &Path) -> Result<String>;
        fn supported_extensions(&self) -> &[&str];
        fn requires_external_tool(&self) -> bool;
    }

    pub struct PdfExtractor;

    impl PdfExtractor {
        async fn extract_pdf_text(&self, file_path: &Path) -> Result<String> {
            let file_path = file_path.to_path_buf();
            
            tokio::task::spawn_blocking(move || {
                let bytes = std::fs::read(&file_path)?;
                let text = pdf_extract::extract_text_from_mem(&bytes)?;
                
                let cleaned_text = text
                    .lines()
                    .map(|line| line.trim())
                    .filter(|line| !line.is_empty())
                    .collect::<Vec<_>>()
                    .join("\n");

                Ok(cleaned_text)
            })
            .await?
        }
    }

    #[async_trait]
    impl ContentExtractor for PdfExtractor {
        async fn extract(&self, file_path: &Path) -> Result<String> {
            self.extract_pdf_text(file_path).await
        }

        fn supported_extensions(&self) -> &[&str] {
            &["pdf"]
        }

        fn requires_external_tool(&self) -> bool {
            false
        }
    }

    pub struct NativeTextExtractor;

    impl NativeTextExtractor {
        async fn extract_plain_text(&self, file_path: &Path) -> Result<String> {
            let bytes = tokio::fs::read(file_path).await?;

            if let Ok(content) = std::str::from_utf8(&bytes) {
                return Ok(content.to_string());
            }

            let (content, _encoding, had_errors) = WINDOWS_1252.decode(&bytes);
            if had_errors {
                return Err(format!("Failed to decode file: {}", file_path.display()).into());
            }

            Ok(content.into_owned())
        }
    }

    #[async_trait]
    impl ContentExtractor for NativeTextExtractor {
        async fn extract(&self, file_path: &Path) -> Result<String> {
            self.extract_plain_text(file_path).await
        }

        fn supported_extensions(&self) -> &[&str] {
            &["txt", "md", "rs", "py", "js", "ts", "java", "go", "c", "cpp", "h", "hpp"]
        }

        fn requires_external_tool(&self) -> bool {
            false
        }
    }
}

#[tokio::test]
async fn test_text_extractor() {
    let temp_dir = TempDir::new().unwrap();
    let file_path = temp_dir.path().join("test.txt");
    
    let content = "This is a test file.\n\nWith multiple paragraphs.\n\nAnd some content.";
    fs::write(&file_path, content).unwrap();
    
    let extractor = extractor::NativeTextExtractor;
    let result = extractor.extract(&file_path).await.unwrap();
    
    assert_eq!(result, content);
    println!("✅ Text extraction test passed");
}

#[tokio::test]
async fn test_pdf_extractor_error_handling() {
    let temp_dir = TempDir::new().unwrap();
    let file_path = temp_dir.path().join("invalid.pdf");
    
    // Create an invalid PDF file (just some text)
    fs::write(&file_path, "This is not a valid PDF file").unwrap();
    
    let extractor = extractor::PdfExtractor;
    let result = extractor.extract(&file_path).await;
    
    // Should return an error for invalid PDF
    assert!(result.is_err());
    println!("✅ PDF error handling test passed");
}

#[tokio::main]
async fn main() {
    println!("Running extractor tests...");
    
    test_text_extractor().await;
    test_pdf_extractor_error_handling().await;
    
    println!("✅ All extractor tests passed!");
}
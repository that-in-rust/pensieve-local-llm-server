//! Content extraction from various file formats

use crate::prelude::*;
use crate::errors::ExtractionError;
use async_trait::async_trait;
use encoding_rs::WINDOWS_1252;
use scraper::{Html, Selector};
use std::path::Path;
use std::time::Duration;

/// Trait for content extraction strategies
#[async_trait]
pub trait ContentExtractor: Send + Sync {
    /// Extract text content from a file
    async fn extract(&self, file_path: &Path) -> Result<String>;
    
    /// Get supported file extensions
    fn supported_extensions(&self) -> &[&str];
    
    /// Check if this extractor requires external tools
    fn requires_external_tool(&self) -> bool;
}

/// Native text file extractor for Tier 1 formats
pub struct NativeTextExtractor;

impl NativeTextExtractor {
    /// Extract plain text with encoding detection
    async fn extract_plain_text(&self, file_path: &Path) -> Result<String> {
        let bytes = tokio::fs::read(file_path).await
            .map_err(|e| PensieveError::FileProcessing {
                file_path: file_path.to_path_buf(),
                cause: format!("Failed to read file: {}", e),
            })?;

        // Try UTF-8 first
        if let Ok(content) = std::str::from_utf8(&bytes) {
            return Ok(content.to_string());
        }

        // Fall back to Windows-1252 (Latin-1 compatible)
        let (content, _encoding, had_errors) = WINDOWS_1252.decode(&bytes);
        if had_errors {
            return Err(PensieveError::ContentExtraction(
                ExtractionError::Encoding(format!(
                    "Failed to decode file with UTF-8 or Windows-1252: {}",
                    file_path.display()
                ))
            ));
        }

        Ok(content.into_owned())
    }

    /// Extract and clean JSON content
    async fn extract_json(&self, file_path: &Path) -> Result<String> {
        let content = self.extract_plain_text(file_path).await?;
        
        // Parse JSON to validate and extract string values
        match serde_json::from_str::<serde_json::Value>(&content) {
            Ok(json) => Ok(self.extract_json_strings(&json)),
            Err(_) => {
                // If JSON parsing fails, treat as plain text
                Ok(content)
            }
        }
    }

    /// Extract and clean YAML content
    async fn extract_yaml(&self, file_path: &Path) -> Result<String> {
        let content = self.extract_plain_text(file_path).await?;
        
        // Parse YAML to validate and extract string values
        match serde_yaml::from_str::<serde_yaml::Value>(&content) {
            Ok(yaml) => Ok(self.extract_yaml_strings(&yaml)),
            Err(_) => {
                // If YAML parsing fails, treat as plain text
                Ok(content)
            }
        }
    }

    /// Extract and clean TOML content
    async fn extract_toml(&self, file_path: &Path) -> Result<String> {
        let content = self.extract_plain_text(file_path).await?;
        
        // Parse TOML to validate and extract string values
        match toml::from_str::<toml::Value>(&content) {
            Ok(toml_value) => Ok(self.extract_toml_strings(&toml_value)),
            Err(_) => {
                // If TOML parsing fails, treat as plain text
                Ok(content)
            }
        }
    }

    /// Recursively extract string values from JSON
    fn extract_json_strings(&self, value: &serde_json::Value) -> String {
        let mut strings = Vec::new();
        self.collect_json_strings(value, &mut strings);
        strings.join("\n")
    }

    /// Recursively collect string values from JSON
    fn collect_json_strings(&self, value: &serde_json::Value, strings: &mut Vec<String>) {
        match value {
            serde_json::Value::String(s) => strings.push(s.clone()),
            serde_json::Value::Array(arr) => {
                for item in arr {
                    self.collect_json_strings(item, strings);
                }
            }
            serde_json::Value::Object(obj) => {
                for (key, val) in obj {
                    strings.push(key.clone());
                    self.collect_json_strings(val, strings);
                }
            }
            _ => {} // Skip numbers, booleans, null
        }
    }

    /// Recursively extract string values from YAML
    fn extract_yaml_strings(&self, value: &serde_yaml::Value) -> String {
        let mut strings = Vec::new();
        self.collect_yaml_strings(value, &mut strings);
        strings.join("\n")
    }

    /// Recursively collect string values from YAML
    fn collect_yaml_strings(&self, value: &serde_yaml::Value, strings: &mut Vec<String>) {
        match value {
            serde_yaml::Value::String(s) => strings.push(s.clone()),
            serde_yaml::Value::Sequence(seq) => {
                for item in seq {
                    self.collect_yaml_strings(item, strings);
                }
            }
            serde_yaml::Value::Mapping(map) => {
                for (key, val) in map {
                    if let serde_yaml::Value::String(key_str) = key {
                        strings.push(key_str.clone());
                    }
                    self.collect_yaml_strings(val, strings);
                }
            }
            _ => {} // Skip numbers, booleans, null
        }
    }

    /// Recursively extract string values from TOML
    fn extract_toml_strings(&self, value: &toml::Value) -> String {
        let mut strings = Vec::new();
        self.collect_toml_strings(value, &mut strings);
        strings.join("\n")
    }

    /// Recursively collect string values from TOML
    fn collect_toml_strings(&self, value: &toml::Value, strings: &mut Vec<String>) {
        match value {
            toml::Value::String(s) => strings.push(s.clone()),
            toml::Value::Array(arr) => {
                for item in arr {
                    self.collect_toml_strings(item, strings);
                }
            }
            toml::Value::Table(table) => {
                for (key, val) in table {
                    strings.push(key.clone());
                    self.collect_toml_strings(val, strings);
                }
            }
            _ => {} // Skip numbers, booleans, datetime
        }
    }
}

#[async_trait]
impl ContentExtractor for NativeTextExtractor {
    async fn extract(&self, file_path: &Path) -> Result<String> {
        let extension = file_path
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("")
            .to_lowercase();

        match extension.as_str() {
            // Structured formats that need special parsing
            "json" => self.extract_json(file_path).await,
            "yaml" | "yml" => self.extract_yaml(file_path).await,
            "toml" => self.extract_toml(file_path).await,
            
            // All other text formats - treat as plain text
            _ => self.extract_plain_text(file_path).await,
        }
    }

    fn supported_extensions(&self) -> &[&str] {
        &[
            "txt", "md", "rst", "org", "adoc", "wiki",
            "rs", "py", "js", "ts", "java", "go", "c", "cpp", "h", "hpp",
            "json", "yaml", "yml", "toml", "ini", "cfg", "env",
            "css", "xml", "svg",
            "sh", "bat", "ps1", "dockerfile",
            "csv", "tsv", "log", "sql"
        ]
    }

    fn requires_external_tool(&self) -> bool {
        false
    }
}

/// HTML content extractor with cleaning and optional Markdown conversion
pub struct HtmlExtractor {
    /// Whether to preserve document structure
    pub preserve_structure: bool,
    /// Whether to convert to Markdown
    pub convert_to_markdown: bool,
}

impl HtmlExtractor {
    /// Create new HTML extractor
    pub fn new() -> Self {
        Self {
            preserve_structure: true,
            convert_to_markdown: true,
        }
    }

    /// Configure structure preservation
    pub fn preserve_structure(mut self, preserve: bool) -> Self {
        self.preserve_structure = preserve;
        self
    }

    /// Configure Markdown conversion
    pub fn convert_to_markdown(mut self, convert: bool) -> Self {
        self.convert_to_markdown = convert;
        self
    }

    /// Extract plain text from HTML content
    fn extract_text_from_html(&self, html: &str) -> String {
        let document = Html::parse_document(html);
        
        // Select all text nodes, excluding script and style
        let text_selector = Selector::parse("*:not(script):not(style)").unwrap();
        
        let mut text_parts = Vec::new();
        for element in document.select(&text_selector) {
            for text_node in element.text() {
                let trimmed = text_node.trim();
                if !trimmed.is_empty() {
                    text_parts.push(trimmed.to_string());
                }
            }
        }
        
        text_parts.join(" ")
    }
}

#[async_trait]
impl ContentExtractor for HtmlExtractor {
    async fn extract(&self, file_path: &Path) -> Result<String> {
        // Read the HTML file with encoding detection
        let bytes = tokio::fs::read(file_path).await
            .map_err(|e| PensieveError::FileProcessing {
                file_path: file_path.to_path_buf(),
                cause: format!("Failed to read HTML file: {}", e),
            })?;

        // Try UTF-8 first
        let html_content = if let Ok(content) = std::str::from_utf8(&bytes) {
            content.to_string()
        } else {
            // Fall back to Windows-1252
            let (content, _encoding, had_errors) = WINDOWS_1252.decode(&bytes);
            if had_errors {
                return Err(PensieveError::ContentExtraction(
                    ExtractionError::Encoding(format!(
                        "Failed to decode HTML file: {}",
                        file_path.display()
                    ))
                ));
            }
            content.into_owned()
        };

        // Parse HTML and extract content
        let document = Html::parse_document(&html_content);
        
        // Parse HTML and extract content
        
        // Extract main content
        let main_content = if let Ok(main_selector) = Selector::parse("main, article, .content, #content") {
            document.select(&main_selector).next()
                .map(|element| element.html())
                .unwrap_or_else(|| {
                    // If no main content area found, use body
                    if let Ok(body_selector) = Selector::parse("body") {
                        document.select(&body_selector).next()
                            .map(|element| element.html())
                            .unwrap_or(html_content)
                    } else {
                        html_content
                    }
                })
        } else {
            html_content
        };

        // Convert to text
        let text_content = if self.convert_to_markdown {
            // Convert HTML to Markdown to preserve structure
            html2md::parse_html(&main_content)
        } else {
            // Extract plain text
            self.extract_text_from_html(&main_content)
        };

        Ok(text_content)
    }

    fn supported_extensions(&self) -> &[&str] {
        &["html", "htm", "xhtml"]
    }

    fn requires_external_tool(&self) -> bool {
        false
    }
}

/// Basic PDF text extractor using native Rust crates
pub struct PdfExtractor;

impl PdfExtractor {
    /// Extract text content from PDF using pdf-extract crate
    async fn extract_pdf_text(&self, file_path: &Path) -> Result<String> {
        // Use tokio::task::spawn_blocking for CPU-intensive PDF parsing
        let file_path = file_path.to_path_buf();
        
        tokio::task::spawn_blocking(move || {
            // Read PDF file and extract text
            let bytes = std::fs::read(&file_path)
                .map_err(|e| PensieveError::FileProcessing {
                    file_path: file_path.clone(),
                    cause: format!("Failed to read PDF file: {}", e),
                })?;

            // Extract text using pdf-extract
            let text = pdf_extract::extract_text_from_mem(&bytes)
                .map_err(|e| PensieveError::ContentExtraction(
                    ExtractionError::FormatParsing {
                        format: "PDF".to_string(),
                        cause: format!("PDF parsing failed: {}", e),
                    }
                ))?;

            // Clean up the extracted text
            let cleaned_text = text
                .lines()
                .map(|line| line.trim())
                .filter(|line| !line.is_empty())
                .collect::<Vec<_>>()
                .join("\n");

            Ok(cleaned_text)
        })
        .await
        .map_err(|e| PensieveError::ContentExtraction(
            ExtractionError::FormatParsing {
                format: "PDF".to_string(),
                cause: format!("PDF extraction task failed: {}", e),
            }
        ))?
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

/// Basic DOCX text extractor using ZIP and XML parsing
pub struct DocxExtractor;

#[async_trait]
impl ContentExtractor for DocxExtractor {
    async fn extract(&self, file_path: &Path) -> Result<String> {
        let file = std::fs::File::open(file_path)
            .map_err(|e| PensieveError::FileProcessing {
                file_path: file_path.to_path_buf(),
                cause: format!("Failed to open DOCX file: {}", e),
            })?;

        let mut archive = zip::ZipArchive::new(file)
            .map_err(|e| PensieveError::ContentExtraction(
                ExtractionError::FormatParsing {
                    format: "DOCX".to_string(),
                    cause: format!("Failed to open DOCX archive: {}", e),
                }
            ))?;

        // Extract document.xml which contains the main text content
        let mut document_xml = archive.by_name("word/document.xml")
            .map_err(|e| PensieveError::ContentExtraction(
                ExtractionError::FormatParsing {
                    format: "DOCX".to_string(),
                    cause: format!("Failed to find document.xml: {}", e),
                }
            ))?;

        let mut xml_content = String::new();
        std::io::Read::read_to_string(&mut document_xml, &mut xml_content)
            .map_err(|e| PensieveError::ContentExtraction(
                ExtractionError::FormatParsing {
                    format: "DOCX".to_string(),
                    cause: format!("Failed to read document.xml: {}", e),
                }
            ))?;

        // Parse XML and extract text content
        self.extract_text_from_docx_xml(&xml_content)
    }

    fn supported_extensions(&self) -> &[&str] {
        &["docx"]
    }

    fn requires_external_tool(&self) -> bool {
        false
    }
}

impl DocxExtractor {
    /// Extract text content from DOCX document.xml
    fn extract_text_from_docx_xml(&self, xml_content: &str) -> Result<String> {
        use quick_xml::events::Event;
        use quick_xml::Reader;

        let mut reader = Reader::from_str(xml_content);
        reader.trim_text(true);

        let mut text_parts = Vec::new();
        let mut buf = Vec::new();
        let mut in_text_element = false;

        loop {
            match reader.read_event_into(&mut buf) {
                Ok(Event::Start(ref e)) => {
                    // Look for text elements (w:t in Word XML)
                    if e.name().as_ref() == b"w:t" {
                        in_text_element = true;
                    }
                }
                Ok(Event::Text(e)) => {
                    if in_text_element {
                        if let Ok(text) = e.unescape() {
                            let text_str = text.trim();
                            if !text_str.is_empty() {
                                text_parts.push(text_str.to_string());
                            }
                        }
                    }
                }
                Ok(Event::End(ref e)) => {
                    if e.name().as_ref() == b"w:t" {
                        in_text_element = false;
                    }
                }
                Ok(Event::Eof) => break,
                Err(e) => {
                    return Err(PensieveError::ContentExtraction(
                        ExtractionError::FormatParsing {
                            format: "DOCX XML".to_string(),
                            cause: format!("XML parsing error: {}", e),
                        }
                    ));
                }
                _ => {}
            }
            buf.clear();
        }

        Ok(text_parts.join(" "))
    }
}

/// External tool orchestrator for Tier 2 formats
pub struct ExternalToolExtractor {
    /// Path to the external tool
    pub tool_path: std::path::PathBuf,
    /// Command line arguments template
    pub args_template: String,
    /// Execution timeout
    pub timeout: Duration,
    /// Supported file extensions
    pub extensions: Vec<String>,
}

impl ExternalToolExtractor {
    /// Create new external tool extractor
    pub fn new(
        tool_path: impl AsRef<Path>,
        args_template: String,
        extensions: Vec<String>,
    ) -> Self {
        Self {
            tool_path: tool_path.as_ref().to_path_buf(),
            args_template,
            timeout: Duration::from_secs(120),
            extensions,
        }
    }

    /// Set execution timeout
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Check if tool is available
    pub async fn is_available(&self) -> bool {
        // TODO: Implement tool availability check
        // This will be implemented in a later task
        false
    }
}

#[async_trait]
impl ContentExtractor for ExternalToolExtractor {
    async fn extract(&self, _file_path: &Path) -> Result<String> {
        // TODO: Implement external tool execution
        // This will be implemented in a later task
        Err(PensieveError::ContentExtraction(
            ExtractionError::ToolNotFound {
                tool: self.tool_path.display().to_string(),
            }
        ))
    }

    fn supported_extensions(&self) -> &[&str] {
        // Convert Vec<String> to &[&str] - this is a limitation we'll address later
        &[]
    }

    fn requires_external_tool(&self) -> bool {
        true
    }
}

/// Content processor for paragraph splitting and normalization
pub struct ContentProcessor;

impl ContentProcessor {
    /// Split content into paragraphs by double newlines
    pub fn split_paragraphs(content: &str) -> Vec<String> {
        content
            .split("\n\n")
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty() && s.len() >= 10) // Skip very short paragraphs
            .collect()
    }

    /// Normalize text content while preserving paragraph boundaries
    pub fn normalize_text(content: &str) -> String {
        // Split content by double newlines to preserve paragraph boundaries
        let paragraphs: Vec<String> = content
            .split("\n\n")
            .map(|paragraph| {
                // Normalize each paragraph individually
                let normalized_paragraph = paragraph
                    .trim()
                    .lines()
                    .map(|line| line.trim())
                    .filter(|line| !line.is_empty())
                    .collect::<Vec<_>>()
                    .join(" ");
                
                // Collapse multiple whitespace within the paragraph
                let mut result = String::new();
                let mut prev_was_space = false;
                
                for ch in normalized_paragraph.chars() {
                    if ch.is_whitespace() {
                        if !prev_was_space {
                            result.push(' ');
                            prev_was_space = true;
                        }
                    } else {
                        result.push(ch);
                        prev_was_space = false;
                    }
                }
                
                result.trim().to_string()
            })
            .filter(|paragraph| !paragraph.is_empty())
            .collect();
        
        // Rejoin paragraphs with double newlines
        paragraphs.join("\n\n")
    }

    /// Estimate token count for content (simple approximation)
    pub fn estimate_tokens(content: &str) -> u32 {
        // Simple approximation: ~4 characters per token for English text
        // This is suitable for MVP requirements
        (content.len() as f64 / 4.0).ceil() as u32
    }

    /// Calculate SHA-256 hash for content deduplication
    pub fn calculate_content_hash(content: &str) -> String {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(content.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    /// Count words in content
    pub fn count_words(content: &str) -> u32 {
        content.split_whitespace().count() as u32
    }

    /// Count characters in content
    pub fn count_characters(content: &str) -> u32 {
        content.chars().count() as u32
    }
}

/// Extraction strategy manager
pub struct ExtractionManager {
    /// Available extractors
    extractors: Vec<Box<dyn ContentExtractor>>,
}

impl ExtractionManager {
    /// Create new extraction manager
    pub fn new() -> Self {
        Self {
            extractors: vec![
                Box::new(NativeTextExtractor),
                Box::new(HtmlExtractor::new()),
                Box::new(DocxExtractor),
                Box::new(PdfExtractor),
            ],
        }
    }

    /// Add an extractor
    pub fn add_extractor(&mut self, extractor: Box<dyn ContentExtractor>) {
        self.extractors.push(extractor);
    }

    /// Find appropriate extractor for file
    pub fn find_extractor(&self, file_path: &Path) -> Option<&dyn ContentExtractor> {
        let extension = file_path
            .extension()?
            .to_str()?
            .to_lowercase();

        self.extractors
            .iter()
            .find(|extractor| {
                extractor.supported_extensions()
                    .iter()
                    .any(|ext| ext.to_lowercase() == extension)
            })
            .map(|boxed| boxed.as_ref())
    }

    /// Extract content from file using appropriate extractor
    pub async fn extract_content(&self, file_path: &Path) -> Result<String> {
        let extractor = self.find_extractor(file_path)
            .ok_or_else(|| {
                let extension = file_path
                    .extension()
                    .and_then(|ext| ext.to_str())
                    .unwrap_or("unknown");
                PensieveError::ContentExtraction(
                    ExtractionError::UnsupportedType {
                        extension: extension.to_string(),
                    }
                )
            })?;

        extractor.extract(file_path).await
    }
}

impl Default for ExtractionManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Write;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_content_processor_functions() {
        let content = "First paragraph with some content.\n\nSecond paragraph here.\n\n\n\nThird paragraph after extra newlines.";
        
        // Test paragraph splitting
        let paragraphs = ContentProcessor::split_paragraphs(content);
        assert_eq!(paragraphs.len(), 3);
        assert_eq!(paragraphs[0], "First paragraph with some content.");
        assert_eq!(paragraphs[1], "Second paragraph here.");
        assert_eq!(paragraphs[2], "Third paragraph after extra newlines.");
        
        // Test text normalization
        let messy_text = "  Multiple   spaces   and\n\n\nextra\n\nlines  ";
        let normalized = ContentProcessor::normalize_text(messy_text);
        assert_eq!(normalized, "Multiple spaces and extra lines");
        
        // Test token estimation
        let test_text = "This is a test sentence with exactly eight words.";
        let tokens = ContentProcessor::estimate_tokens(test_text);
        assert!(tokens > 0);
        
        // Test hash calculation
        let hash1 = ContentProcessor::calculate_content_hash("test content");
        let hash2 = ContentProcessor::calculate_content_hash("test content");
        let hash3 = ContentProcessor::calculate_content_hash("different content");
        
        assert_eq!(hash1, hash2); // Same content should have same hash
        assert_ne!(hash1, hash3); // Different content should have different hash
        assert_eq!(hash1.len(), 64); // SHA-256 hash should be 64 hex characters
        
        // Test word and character counting
        let test_text = "Hello world test";
        assert_eq!(ContentProcessor::count_words(test_text), 3);
        assert_eq!(ContentProcessor::count_characters(test_text), 16);
    }

    #[tokio::test]
    async fn test_native_text_extractor() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.txt");
        
        let content = "This is a test file.\n\nWith multiple paragraphs.\n\nAnd some content.";
        fs::write(&file_path, content).unwrap();
        
        let extractor = NativeTextExtractor;
        let result = extractor.extract(&file_path).await.unwrap();
        
        assert_eq!(result, content);
    }

    #[tokio::test]
    async fn test_json_extraction() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.json");
        
        let json_content = r#"{
            "name": "Test Document",
            "description": "This is a test JSON file",
            "items": [
                "First item",
                "Second item"
            ],
            "metadata": {
                "author": "Test Author",
                "version": "1.0"
            }
        }"#;
        
        fs::write(&file_path, json_content).unwrap();
        
        let extractor = NativeTextExtractor;
        let result = extractor.extract(&file_path).await.unwrap();
        
        // Should extract string values
        assert!(result.contains("Test Document"));
        assert!(result.contains("This is a test JSON file"));
        assert!(result.contains("First item"));
        assert!(result.contains("Test Author"));
    }

    #[tokio::test]
    async fn test_yaml_extraction() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.yaml");
        
        let yaml_content = r#"
name: Test Document
description: This is a test YAML file
items:
  - First item
  - Second item
metadata:
  author: Test Author
  version: 1.0
"#;
        
        fs::write(&file_path, yaml_content).unwrap();
        
        let extractor = NativeTextExtractor;
        let result = extractor.extract(&file_path).await.unwrap();
        
        // Should extract string values
        assert!(result.contains("Test Document"));
        assert!(result.contains("This is a test YAML file"));
        assert!(result.contains("First item"));
        assert!(result.contains("Test Author"));
    }

    #[tokio::test]
    async fn test_toml_extraction() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.toml");
        
        let toml_content = r#"
name = "Test Document"
description = "This is a test TOML file"
items = ["First item", "Second item"]

[metadata]
author = "Test Author"
version = "1.0"
"#;
        
        fs::write(&file_path, toml_content).unwrap();
        
        let extractor = NativeTextExtractor;
        let result = extractor.extract(&file_path).await.unwrap();
        
        // Should extract string values
        assert!(result.contains("Test Document"));
        assert!(result.contains("This is a test TOML file"));
        assert!(result.contains("First item"));
        assert!(result.contains("Test Author"));
    }

    #[tokio::test]
    async fn test_html_extraction() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.html");
        
        let html_content = r#"<!DOCTYPE html>
<html>
<head>
    <title>Test Document</title>
    <style>body { font-family: Arial; }</style>
</head>
<body>
    <header>
        <nav>Navigation</nav>
    </header>
    <main>
        <h1>Main Title</h1>
        <p>This is the main content of the document.</p>
        <p>It has multiple paragraphs with useful information.</p>
    </main>
    <script>console.log('test');</script>
    <footer>Footer content</footer>
</body>
</html>"#;
        
        fs::write(&file_path, html_content).unwrap();
        
        let extractor = HtmlExtractor::new();
        let result = extractor.extract(&file_path).await.unwrap();
        
        // Should extract main content and convert to markdown
        assert!(result.contains("Main Title"));
        assert!(result.contains("main content"));
        assert!(result.contains("multiple paragraphs"));
        
        // Should not contain script or style content
        assert!(!result.contains("console.log"));
        assert!(!result.contains("font-family"));
    }

    #[tokio::test]
    async fn test_docx_extraction() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.docx");
        
        // Create a minimal ZIP file structure for DOCX
        let file = std::fs::File::create(&file_path).unwrap();
        let mut zip = zip::ZipWriter::new(file);
        
        // Add document.xml with basic content
        let document_xml = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
    <w:body>
        <w:p>
            <w:r>
                <w:t>This is a test DOCX document.</w:t>
            </w:r>
        </w:p>
        <w:p>
            <w:r>
                <w:t>It contains multiple paragraphs.</w:t>
            </w:r>
        </w:p>
    </w:body>
</w:document>"#;
        
        zip.start_file("word/document.xml", zip::write::FileOptions::default()).unwrap();
        zip.write_all(document_xml.as_bytes()).unwrap();
        zip.finish().unwrap();
        
        let extractor = DocxExtractor;
        let result = extractor.extract(&file_path).await.unwrap();
        
        // Should extract text content from DOCX
        assert!(result.contains("This is a test DOCX document."));
        assert!(result.contains("It contains multiple paragraphs."));
    }

    #[tokio::test]
    async fn test_encoding_detection() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test_utf8.txt");
        
        // Test UTF-8 content with special characters
        let utf8_content = "Hello 世界! This is UTF-8 content with émojis 🚀";
        fs::write(&file_path, utf8_content).unwrap();
        
        let extractor = NativeTextExtractor;
        let result = extractor.extract(&file_path).await.unwrap();
        
        assert_eq!(result, utf8_content);
    }

    #[tokio::test]
    async fn test_pdf_extractor_error_handling() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("invalid.pdf");
        
        // Create an invalid PDF file (just some text)
        fs::write(&file_path, "This is not a valid PDF file").unwrap();
        
        let extractor = PdfExtractor;
        let result = extractor.extract(&file_path).await;
        
        // Should return an error for invalid PDF
        assert!(result.is_err());
        
        // Check that it's the right type of error
        match result.unwrap_err() {
            PensieveError::ContentExtraction(ExtractionError::FormatParsing { format, .. }) => {
                assert_eq!(format, "PDF");
            }
            other => panic!("Expected PDF format parsing error, got: {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_extraction_manager() {
        let temp_dir = TempDir::new().unwrap();
        let manager = ExtractionManager::new();
        
        // Test text file
        let txt_path = temp_dir.path().join("test.txt");
        fs::write(&txt_path, "Test content").unwrap();
        
        let extractor = manager.find_extractor(&txt_path);
        assert!(extractor.is_some());
        assert!(!extractor.unwrap().requires_external_tool());
        
        // Test HTML file
        let html_path = temp_dir.path().join("test.html");
        fs::write(&html_path, "<html><body>Test</body></html>").unwrap();
        
        let extractor = manager.find_extractor(&html_path);
        assert!(extractor.is_some());
        assert!(!extractor.unwrap().requires_external_tool());
        
        // Test unsupported file
        let unsupported_path = temp_dir.path().join("test.xyz");
        fs::write(&unsupported_path, "content").unwrap();
        
        let extractor = manager.find_extractor(&unsupported_path);
        assert!(extractor.is_none());
        
        // Test extraction with unsupported file
        let result = manager.extract_content(&unsupported_path).await;
        assert!(result.is_err());
        
        match result.unwrap_err() {
            PensieveError::ContentExtraction(ExtractionError::UnsupportedType { extension }) => {
                assert_eq!(extension, "xyz");
            }
            other => panic!("Expected unsupported type error, got: {:?}", other),
        }
    }
}
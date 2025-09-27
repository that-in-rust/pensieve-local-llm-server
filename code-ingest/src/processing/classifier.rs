use super::FileType;
use std::collections::HashMap;
use std::path::Path;

/// Configuration for file type classification
#[derive(Debug, Clone)]
pub struct ClassifierConfig {
    /// Custom file type mappings (extension -> FileType)
    pub custom_mappings: HashMap<String, FileType>,
    /// Whether to treat unknown extensions as non-text
    pub unknown_as_non_text: bool,
}

impl Default for ClassifierConfig {
    fn default() -> Self {
        Self {
            custom_mappings: HashMap::new(),
            unknown_as_non_text: true,
        }
    }
}

/// File classifier that categorizes files into three types based on extensions
#[derive(Debug, Clone)]
pub struct FileClassifier {
    config: ClassifierConfig,
    direct_text_extensions: HashMap<String, ()>,
    convertible_extensions: HashMap<String, ()>,
}

impl FileClassifier {
    /// Create a new file classifier with default configuration
    pub fn new() -> Self {
        Self::with_config(ClassifierConfig::default())
    }

    /// Create a new file classifier with custom configuration
    pub fn with_config(config: ClassifierConfig) -> Self {
        let mut classifier = Self {
            config,
            direct_text_extensions: HashMap::new(),
            convertible_extensions: HashMap::new(),
        };

        classifier.initialize_default_mappings();
        classifier
    }

    /// Initialize default file type mappings
    fn initialize_default_mappings(&mut self) {
        // Type 1: Direct text files
        let direct_text_exts = [
            // Programming languages
            "rs", "py", "js", "ts", "java", "c", "cpp", "cc", "cxx", "h", "hpp", "hxx",
            "cs", "php", "rb", "go", "kt", "swift", "scala", "clj", "hs", "ml", "fs",
            "elm", "dart", "r", "m", "mm", "pl", "pm", "sh", "bash", "zsh", "fish",
            "ps1", "bat", "cmd", "vb", "vbs", "asm", "s", "S", "f", "f90", "f95",
            "for", "ftn", "pas", "pp", "inc", "lua", "tcl", "awk", "sed", "vim",
            // Web technologies
            "html", "htm", "xml", "xhtml", "svg", "css", "scss", "sass", "less",
            "jsx", "tsx", "vue", "svelte", "astro", "php", "asp", "aspx", "jsp",
            "erb", "ejs", "hbs", "mustache", "twig", "blade",
            // Configuration and data
            "json", "yaml", "yml", "toml", "ini", "cfg", "conf", "config", "properties",
            "env", "dotenv", "editorconfig", "gitignore", "gitattributes", "dockerignore",
            // Documentation
            "md", "markdown", "txt", "rst", "adoc", "asciidoc", "org", "tex", "latex",
            "rtf", "wiki", "textile",
            // Build and project files
            "makefile", "cmake", "gradle", "sbt", "pom", "build", "proj", "csproj",
            "vcxproj", "pbxproj", "xcodeproj", "package", "lock", "sum", "mod",
            // Database and query files
            "sql", "psql", "mysql", "sqlite", "cypher", "sparql", "graphql", "gql",
            // Logs and data
            "log", "logs", "csv", "tsv", "jsonl", "ndjson", "ldif",
            // Scripts and automation
            "dockerfile", "containerfile", "vagrantfile", "rakefile", "gulpfile",
            "gruntfile", "webpack", "rollup", "vite", "esbuild",
        ];

        for ext in direct_text_exts {
            self.direct_text_extensions.insert(ext.to_lowercase(), ());
        }

        // Type 2: Convertible files
        let convertible_exts = [
            // Documents
            "pdf", "doc", "docx", "odt", "rtf", "pages",
            // Spreadsheets
            "xls", "xlsx", "ods", "numbers", "csv",
            // Presentations
            "ppt", "pptx", "odp", "key",
            // E-books
            "epub", "mobi", "azw", "azw3", "fb2",
            // Archives (can extract text from contents)
            "zip", "tar", "gz", "bz2", "xz", "7z", "rar",
        ];

        for ext in convertible_exts {
            self.convertible_extensions.insert(ext.to_lowercase(), ());
        }
    }

    /// Classify a file based on its path
    pub fn classify_file<P: AsRef<Path>>(&self, file_path: P) -> FileType {
        let path = file_path.as_ref();
        
        // Extract extension
        let extension = match path.extension() {
            Some(ext) => ext.to_string_lossy().to_lowercase(),
            None => {
                // Files without extensions - check if they're known text files
                let filename = path.file_name()
                    .map(|n| n.to_string_lossy().to_lowercase())
                    .unwrap_or_default();
                
                return self.classify_by_filename(&filename);
            }
        };

        self.classify_by_extension(&extension)
    }

    /// Classify by file extension
    pub fn classify_by_extension(&self, extension: &str) -> FileType {
        let ext_lower = extension.to_lowercase();

        // Check custom mappings first
        if let Some(&file_type) = self.config.custom_mappings.get(&ext_lower) {
            return file_type;
        }

        // Check default mappings
        if self.direct_text_extensions.contains_key(&ext_lower) {
            FileType::DirectText
        } else if self.convertible_extensions.contains_key(&ext_lower) {
            FileType::Convertible
        } else if self.config.unknown_as_non_text {
            FileType::NonText
        } else {
            // Conservative approach: treat unknown as direct text for analysis
            FileType::DirectText
        }
    }

    /// Classify files without extensions by filename
    fn classify_by_filename(&self, filename: &str) -> FileType {
        let filename_lower = filename.to_lowercase();

        // Common files without extensions that are text
        let text_files = [
            "makefile", "dockerfile", "containerfile", "vagrantfile", "rakefile",
            "gulpfile", "gruntfile", "readme", "license", "changelog", "authors",
            "contributors", "copying", "install", "news", "todo", "version",
            "manifest", "gemfile", "podfile", "brewfile", "procfile", "aptfile",
        ];

        for text_file in &text_files {
            if filename_lower == *text_file || filename_lower.starts_with(text_file) {
                return FileType::DirectText;
            }
        }

        // Default for unknown files without extensions
        if self.config.unknown_as_non_text {
            FileType::NonText
        } else {
            FileType::DirectText
        }
    }

    /// Add a custom file type mapping
    pub fn add_custom_mapping(&mut self, extension: String, file_type: FileType) {
        self.config.custom_mappings.insert(extension.to_lowercase(), file_type);
    }

    /// Remove a custom file type mapping
    pub fn remove_custom_mapping(&mut self, extension: &str) -> Option<FileType> {
        self.config.custom_mappings.remove(&extension.to_lowercase())
    }

    /// Get all supported direct text extensions
    pub fn get_direct_text_extensions(&self) -> Vec<String> {
        self.direct_text_extensions.keys().cloned().collect()
    }

    /// Get all supported convertible extensions
    pub fn get_convertible_extensions(&self) -> Vec<String> {
        self.convertible_extensions.keys().cloned().collect()
    }

    /// Check if an extension is supported for text processing
    pub fn is_text_processable(&self, extension: &str) -> bool {
        let file_type = self.classify_by_extension(extension);
        matches!(file_type, FileType::DirectText | FileType::Convertible)
    }

    /// Validate file type detection
    pub fn validate_classification<P: AsRef<Path>>(&self, file_path: P) -> Result<FileType, String> {
        let path = file_path.as_ref();
        
        if !path.exists() {
            return Err(format!("File does not exist: {}", path.display()));
        }

        if !path.is_file() {
            return Err(format!("Path is not a file: {}", path.display()));
        }

        Ok(self.classify_file(path))
    }
}

impl Default for FileClassifier {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_direct_text_classification() {
        let classifier = FileClassifier::new();

        // Programming languages
        assert_eq!(classifier.classify_by_extension("rs"), FileType::DirectText);
        assert_eq!(classifier.classify_by_extension("py"), FileType::DirectText);
        assert_eq!(classifier.classify_by_extension("js"), FileType::DirectText);
        assert_eq!(classifier.classify_by_extension("java"), FileType::DirectText);
        assert_eq!(classifier.classify_by_extension("c"), FileType::DirectText);
        assert_eq!(classifier.classify_by_extension("cpp"), FileType::DirectText);

        // Web technologies
        assert_eq!(classifier.classify_by_extension("html"), FileType::DirectText);
        assert_eq!(classifier.classify_by_extension("css"), FileType::DirectText);
        assert_eq!(classifier.classify_by_extension("json"), FileType::DirectText);
        assert_eq!(classifier.classify_by_extension("xml"), FileType::DirectText);

        // Configuration files
        assert_eq!(classifier.classify_by_extension("yaml"), FileType::DirectText);
        assert_eq!(classifier.classify_by_extension("toml"), FileType::DirectText);
        assert_eq!(classifier.classify_by_extension("ini"), FileType::DirectText);

        // Documentation
        assert_eq!(classifier.classify_by_extension("md"), FileType::DirectText);
        assert_eq!(classifier.classify_by_extension("txt"), FileType::DirectText);
        assert_eq!(classifier.classify_by_extension("rst"), FileType::DirectText);
    }

    #[test]
    fn test_convertible_classification() {
        let classifier = FileClassifier::new();

        // Documents
        assert_eq!(classifier.classify_by_extension("pdf"), FileType::Convertible);
        assert_eq!(classifier.classify_by_extension("docx"), FileType::Convertible);
        assert_eq!(classifier.classify_by_extension("odt"), FileType::Convertible);

        // Spreadsheets
        assert_eq!(classifier.classify_by_extension("xlsx"), FileType::Convertible);
        assert_eq!(classifier.classify_by_extension("ods"), FileType::Convertible);

        // Presentations
        assert_eq!(classifier.classify_by_extension("pptx"), FileType::Convertible);
        assert_eq!(classifier.classify_by_extension("odp"), FileType::Convertible);

        // Archives
        assert_eq!(classifier.classify_by_extension("zip"), FileType::Convertible);
        assert_eq!(classifier.classify_by_extension("tar"), FileType::Convertible);
    }

    #[test]
    fn test_non_text_classification() {
        let classifier = FileClassifier::new();

        // Images
        assert_eq!(classifier.classify_by_extension("jpg"), FileType::NonText);
        assert_eq!(classifier.classify_by_extension("png"), FileType::NonText);
        assert_eq!(classifier.classify_by_extension("gif"), FileType::NonText);

        // Videos
        assert_eq!(classifier.classify_by_extension("mp4"), FileType::NonText);
        assert_eq!(classifier.classify_by_extension("avi"), FileType::NonText);

        // Binaries
        assert_eq!(classifier.classify_by_extension("exe"), FileType::NonText);
        assert_eq!(classifier.classify_by_extension("bin"), FileType::NonText);

        // Unknown extensions
        assert_eq!(classifier.classify_by_extension("unknown"), FileType::NonText);
    }

    #[test]
    fn test_case_insensitive_classification() {
        let classifier = FileClassifier::new();

        assert_eq!(classifier.classify_by_extension("RS"), FileType::DirectText);
        assert_eq!(classifier.classify_by_extension("Py"), FileType::DirectText);
        assert_eq!(classifier.classify_by_extension("PDF"), FileType::Convertible);
        assert_eq!(classifier.classify_by_extension("JPG"), FileType::NonText);
    }

    #[test]
    fn test_filename_without_extension() {
        let classifier = FileClassifier::new();

        // Known text files without extensions
        assert_eq!(classifier.classify_file(Path::new("Makefile")), FileType::DirectText);
        assert_eq!(classifier.classify_file(Path::new("Dockerfile")), FileType::DirectText);
        assert_eq!(classifier.classify_file(Path::new("README")), FileType::DirectText);
        assert_eq!(classifier.classify_file(Path::new("LICENSE")), FileType::DirectText);

        // Unknown files without extensions
        assert_eq!(classifier.classify_file(Path::new("unknown_file")), FileType::NonText);
    }

    #[test]
    fn test_custom_mappings() {
        let mut classifier = FileClassifier::new();

        // Add custom mapping
        classifier.add_custom_mapping("custom".to_string(), FileType::DirectText);
        assert_eq!(classifier.classify_by_extension("custom"), FileType::DirectText);

        // Override existing mapping
        classifier.add_custom_mapping("pdf".to_string(), FileType::NonText);
        assert_eq!(classifier.classify_by_extension("pdf"), FileType::NonText);

        // Remove custom mapping
        let removed = classifier.remove_custom_mapping("custom");
        assert_eq!(removed, Some(FileType::DirectText));
        assert_eq!(classifier.classify_by_extension("custom"), FileType::NonText);
    }

    #[test]
    fn test_custom_config() {
        let mut config = ClassifierConfig::default();
        config.unknown_as_non_text = false;
        config.custom_mappings.insert("special".to_string(), FileType::Convertible);

        let classifier = FileClassifier::with_config(config);

        // Custom mapping should work
        assert_eq!(classifier.classify_by_extension("special"), FileType::Convertible);

        // Unknown extensions should be treated as direct text
        assert_eq!(classifier.classify_by_extension("unknown"), FileType::DirectText);
    }

    #[test]
    fn test_file_path_classification() {
        let classifier = FileClassifier::new();

        let test_cases = [
            ("src/main.rs", FileType::DirectText),
            ("docs/README.md", FileType::DirectText),
            ("config/app.toml", FileType::DirectText),
            ("assets/image.jpg", FileType::NonText),
            ("documents/report.pdf", FileType::Convertible),
            ("data/spreadsheet.xlsx", FileType::Convertible),
        ];

        for (path_str, expected_type) in test_cases {
            let path = PathBuf::from(path_str);
            assert_eq!(
                classifier.classify_file(&path),
                expected_type,
                "Failed for path: {}",
                path_str
            );
        }
    }

    #[test]
    fn test_extension_lists() {
        let classifier = FileClassifier::new();

        let direct_text_exts = classifier.get_direct_text_extensions();
        let convertible_exts = classifier.get_convertible_extensions();

        // Should contain expected extensions
        assert!(direct_text_exts.contains(&"rs".to_string()));
        assert!(direct_text_exts.contains(&"py".to_string()));
        assert!(direct_text_exts.contains(&"md".to_string()));

        assert!(convertible_exts.contains(&"pdf".to_string()));
        assert!(convertible_exts.contains(&"docx".to_string()));
        assert!(convertible_exts.contains(&"xlsx".to_string()));

        // Lists should not overlap
        for ext in &direct_text_exts {
            assert!(!convertible_exts.contains(ext), "Extension {} appears in both lists", ext);
        }
    }

    #[test]
    fn test_text_processable_check() {
        let classifier = FileClassifier::new();

        assert!(classifier.is_text_processable("rs"));
        assert!(classifier.is_text_processable("pdf"));
        assert!(!classifier.is_text_processable("jpg"));
        assert!(!classifier.is_text_processable("unknown"));
    }

    #[test]
    fn test_comprehensive_extension_coverage() {
        let classifier = FileClassifier::new();

        // Test a comprehensive set of common file extensions
        let test_cases = [
            // Programming languages - should all be DirectText
            ("rs", FileType::DirectText),
            ("py", FileType::DirectText),
            ("js", FileType::DirectText),
            ("ts", FileType::DirectText),
            ("java", FileType::DirectText),
            ("c", FileType::DirectText),
            ("cpp", FileType::DirectText),
            ("h", FileType::DirectText),
            ("cs", FileType::DirectText),
            ("php", FileType::DirectText),
            ("rb", FileType::DirectText),
            ("go", FileType::DirectText),
            ("swift", FileType::DirectText),
            ("kt", FileType::DirectText),
            
            // Web files - should all be DirectText
            ("html", FileType::DirectText),
            ("css", FileType::DirectText),
            ("scss", FileType::DirectText),
            ("jsx", FileType::DirectText),
            ("vue", FileType::DirectText),
            
            // Config files - should all be DirectText
            ("json", FileType::DirectText),
            ("yaml", FileType::DirectText),
            ("toml", FileType::DirectText),
            ("ini", FileType::DirectText),
            ("xml", FileType::DirectText),
            
            // Documents - should be Convertible
            ("pdf", FileType::Convertible),
            ("docx", FileType::Convertible),
            ("xlsx", FileType::Convertible),
            ("pptx", FileType::Convertible),
            
            // Media files - should be NonText
            ("jpg", FileType::NonText),
            ("png", FileType::NonText),
            ("mp4", FileType::NonText),
            ("mp3", FileType::NonText),
            
            // Binary files - should be NonText
            ("exe", FileType::NonText),
            ("dll", FileType::NonText),
            ("so", FileType::NonText),
        ];

        for (extension, expected_type) in test_cases {
            let actual_type = classifier.classify_by_extension(extension);
            assert_eq!(
                actual_type, expected_type,
                "Extension '{}' classified as {:?}, expected {:?}",
                extension, actual_type, expected_type
            );
        }
    }
}
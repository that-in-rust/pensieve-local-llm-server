//! Demonstration of the file type detection system

use pensieve::scanner::{FileTypeDetector, FileClassification};
use std::path::Path;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let detector = FileTypeDetector::new();

    println!("🔍 Pensieve File Type Detection Demo");
    println!("=====================================\n");

    // Test various file extensions
    let test_files = vec![
        // Tier 1 files
        ("src/main.rs", "Rust source code"),
        ("config.json", "JSON configuration"),
        ("README.md", "Markdown documentation"),
        ("script.py", "Python script"),
        ("index.html", "HTML document"),
        ("styles.css", "CSS stylesheet"),
        ("data.csv", "CSV data file"),
        
        // Tier 2 files
        ("document.pdf", "PDF document"),
        ("report.docx", "Word document"),
        ("spreadsheet.xlsx", "Excel spreadsheet"),
        ("presentation.pptx", "PowerPoint presentation"),
        ("book.epub", "EPUB e-book"),
        
        // Binary files
        ("photo.jpg", "JPEG image"),
        ("video.mp4", "MP4 video"),
        ("music.mp3", "MP3 audio"),
        ("archive.zip", "ZIP archive"),
        ("program.exe", "Windows executable"),
        ("library.dll", "Dynamic library"),
    ];

    for (filename, description) in &test_files {
        let path = Path::new(filename);
        
        match detector.detect_type(path).await {
            Ok(classification) => {
                let (tier, action) = match classification {
                    FileClassification::Tier1Native => ("Tier 1", "✅ Process natively"),
                    FileClassification::Tier2External => ("Tier 2", "🔧 Use external tool"),
                    FileClassification::Binary => ("Binary", "❌ Exclude from processing"),
                };
                
                let should_exclude = detector.should_exclude(path).await;
                let status = if should_exclude { "EXCLUDED" } else { "INCLUDED" };
                
                println!("{:<20} | {:<25} | {:<8} | {:<22} | {}",
                    filename, description, tier, action, status);
            }
            Err(e) => {
                println!("{:<20} | {:<25} | ERROR: {}", filename, description, e);
            }
        }
    }

    println!("\n📊 File Type Statistics:");
    println!("========================");
    
    let mut tier1_count = 0;
    let mut tier2_count = 0;
    let mut binary_count = 0;
    
    for (filename, _) in &test_files {
        let path = Path::new(filename);
        if let Ok(classification) = detector.detect_type(path).await {
            match classification {
                FileClassification::Tier1Native => tier1_count += 1,
                FileClassification::Tier2External => tier2_count += 1,
                FileClassification::Binary => binary_count += 1,
            }
        }
    }
    
    println!("Tier 1 (Native):     {} files", tier1_count);
    println!("Tier 2 (External):   {} files", tier2_count);
    println!("Binary (Excluded):   {} files", binary_count);
    println!("Total:               {} files", test_files.len());
    
    let processing_ratio = (tier1_count + tier2_count) as f64 / test_files.len() as f64 * 100.0;
    println!("Processing ratio:    {:.1}%", processing_ratio);

    println!("\n🎯 Key Features Demonstrated:");
    println!("==============================");
    println!("✅ Extension-based classification");
    println!("✅ MIME type detection with magic numbers");
    println!("✅ Binary file detection and exclusion");
    println!("✅ Comprehensive format support");
    println!("✅ Tier-based processing strategy");

    Ok(())
}
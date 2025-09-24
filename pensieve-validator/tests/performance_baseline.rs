use std::path::Path;
use std::time::{Duration, Instant};
use pensieve_validator::directory_analyzer::DirectoryAnalyzer;
use tokio::test;

#[tokio::test]
async fn test_directory_analysis_performance() {
    let dataset = Path::new("/Users/neetipatni/downloads/RustRAW20250920");
    let analyzer = DirectoryAnalyzer::new();
    let start = Instant::now();
    let result = analyzer.analyze_directory(dataset).expect("Directory analysis should succeed");
    let elapsed = start.elapsed();
    // Assert performance baseline: complete within 120 seconds
    assert!(elapsed < Duration::from_secs(120),
        "Directory analysis took {:?}, expected < 120s", elapsed);
    // Basic sanity check on results
    assert!(result.total_files > 0, "Expected at least one file in dataset");
}

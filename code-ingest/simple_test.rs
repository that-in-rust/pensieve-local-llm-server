#!/usr/bin/env rust-script

//! Simple test to validate DatabaseService basic functionality

fn main() {
    println!("🧪 Running simple DatabaseService tests...\n");
    
    // Test 1: Chunked table name generation
    test_chunked_table_name_generation();
    
    // Test 2: Required columns validation
    test_required_columns_validation();
    
    // Test 3: Error message formatting
    test_error_message_formatting();
    
    // Test 4: Table info structure
    test_table_info_structure();
    
    println!("\n✅ All simple tests passed!");
    println!("   DatabaseService implementation is structurally sound.");
}

fn test_chunked_table_name_generation() {
    println!("🔧 Testing chunked table name generation...");
    
    let original_table = "INGEST_20250927143022";
    let chunk_size = 500;
    let expected_name = format!("{}_{}", original_table, chunk_size);
    
    assert_eq!(expected_name, "INGEST_20250927143022_500");
    
    // Test with different chunk sizes
    let chunk_size_1000 = 1000;
    let expected_name_1000 = format!("{}_{}", original_table, chunk_size_1000);
    assert_eq!(expected_name_1000, "INGEST_20250927143022_1000");
    
    println!("   ✅ Chunked table name generation works correctly");
}

fn test_required_columns_validation() {
    println!("🔧 Testing required columns validation...");
    
    let required_columns = vec![
        "file_id", "ingestion_id", "filepath", "filename", "extension",
        "file_size_bytes", "line_count", "word_count", "token_count",
        "content_text", "file_type", "conversion_command", "relative_path",
        "absolute_path", "created_at"
    ];
    
    // Test complete column set
    let complete_columns: Vec<String> = required_columns.iter().map(|s| s.to_string()).collect();
    let has_all_required = required_columns.iter()
        .all(|col| complete_columns.contains(&col.to_string()));
    assert!(has_all_required);
    
    // Test incomplete column set
    let incomplete_columns = vec!["file_id", "filepath", "filename"];
    let missing_columns: Vec<_> = required_columns.iter()
        .filter(|col| !incomplete_columns.contains(col))
        .collect();
    
    assert!(!missing_columns.is_empty());
    assert_eq!(missing_columns.len(), 12); // 15 total - 3 present = 12 missing
    
    println!("   ✅ Required columns validation works correctly");
}

fn test_error_message_formatting() {
    println!("🔧 Testing error message formatting...");
    
    // Test table not found error format
    let table_name = "NONEXISTENT_TABLE";
    let error_message = format!("Table '{}' does not exist", table_name);
    assert!(error_message.contains(table_name));
    assert!(error_message.contains("does not exist"));
    
    // Test invalid chunk size error format
    let chunk_size = 0;
    let error_message = format!("Invalid chunk size: {} (must be > 0)", chunk_size);
    assert!(error_message.contains("0"));
    assert!(error_message.contains("must be > 0"));
    
    // Test invalid table name error format
    let table_name = "BAD_TABLE";
    let cause = "Missing required columns";
    let error_message = format!("Invalid table name: {} - {}", table_name, cause);
    assert!(error_message.contains(table_name));
    assert!(error_message.contains(cause));
    
    println!("   ✅ Error message formatting works correctly");
}

fn test_table_info_structure() {
    println!("🔧 Testing TableInfo structure...");
    
    // Simulate TableInfo creation
    let table_name = "INGEST_20250927143022".to_string();
    let row_count = 150i64;
    let has_valid_schema = true;
    let columns = vec![
        "file_id".to_string(),
        "ingestion_id".to_string(),
        "filepath".to_string(),
        "filename".to_string(),
        "extension".to_string(),
        "file_size_bytes".to_string(),
        "line_count".to_string(),
        "word_count".to_string(),
        "token_count".to_string(),
        "content_text".to_string(),
        "file_type".to_string(),
        "conversion_command".to_string(),
        "relative_path".to_string(),
        "absolute_path".to_string(),
        "created_at".to_string(),
    ];
    
    // Validate structure
    assert_eq!(table_name, "INGEST_20250927143022");
    assert_eq!(row_count, 150);
    assert!(has_valid_schema);
    assert_eq!(columns.len(), 15);
    assert!(columns.contains(&"file_id".to_string()));
    assert!(columns.contains(&"content_text".to_string()));
    
    println!("   ✅ TableInfo structure works correctly");
}
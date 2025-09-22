use pensieve::database::Database;
use pensieve::types::{Paragraph, ParagraphId, ParagraphSource, FileId, ProcessingError};
use chrono::Utc;
use uuid::Uuid;
use tempfile::NamedTempFile;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a temporary database
    let temp_file = NamedTempFile::new()?;
    let db_path = temp_file.path();
    let db = Database::new(db_path).await?;
    db.initialize_schema().await?;

    // Test paragraph insertion
    let paragraph = Paragraph {
        id: ParagraphId::new(),
        content_hash: "test_hash_123".to_string(),
        content: "This is a test paragraph content.".to_string(),
        estimated_tokens: 10,
        word_count: 7,
        char_count: 33,
        created_at: Utc::now(),
    };

    println!("Testing insert_paragraph...");
    db.insert_paragraph(&paragraph).await?;
    println!("✓ Paragraph inserted successfully");

    // Test paragraph retrieval
    println!("Testing get_paragraph_by_hash...");
    let retrieved = db.get_paragraph_by_hash("test_hash_123").await?;
    match retrieved {
        Some(p) => {
            println!("✓ Paragraph retrieved successfully");
            println!("  Content: {}", p.content);
            println!("  Tokens: {}", p.estimated_tokens);
        }
        None => println!("✗ Paragraph not found"),
    }

    // Test paragraph source insertion
    let source = ParagraphSource {
        paragraph_id: paragraph.id,
        file_id: FileId::new(),
        paragraph_index: 0,
        byte_offset_start: 0,
        byte_offset_end: 33,
    };

    println!("Testing insert_paragraph_source...");
    db.insert_paragraph_source(&source).await?;
    println!("✓ Paragraph source inserted successfully");

    // Test error insertion
    let error = ProcessingError {
        id: Uuid::new_v4(),
        file_id: Some(FileId::new()),
        error_type: "TestError".to_string(),
        error_message: "This is a test error".to_string(),
        stack_trace: Some("test stack trace".to_string()),
        occurred_at: Utc::now(),
    };

    println!("Testing insert_error...");
    db.insert_error(&error).await?;
    println!("✓ Error inserted successfully");

    // Test batch operations
    let paragraphs = vec![
        Paragraph {
            id: ParagraphId::new(),
            content_hash: "batch_hash_1".to_string(),
            content: "Batch paragraph 1".to_string(),
            estimated_tokens: 5,
            word_count: 3,
            char_count: 17,
            created_at: Utc::now(),
        },
        Paragraph {
            id: ParagraphId::new(),
            content_hash: "batch_hash_2".to_string(),
            content: "Batch paragraph 2".to_string(),
            estimated_tokens: 5,
            word_count: 3,
            char_count: 17,
            created_at: Utc::now(),
        },
    ];

    println!("Testing insert_paragraphs_batch...");
    db.insert_paragraphs_batch(&paragraphs).await?;
    println!("✓ Batch paragraphs inserted successfully");

    println!("\nAll paragraph database methods working correctly! ✓");
    Ok(())
}
use code_ingest::processing::chunking::{ChunkingEngine, ChunkingConfig};

fn main() {
    let engine = ChunkingEngine::with_chunk_size(10);
    let content = (1..=25)
        .map(|i| format!("line {}", i))
        .collect::<Vec<_>>()
        .join("\n");
    
    let result = engine.chunk_content(
        &content,
        "test_file".to_string(),
        "test.txt".to_string(),
        "test.txt".to_string(),
        Some("txt".to_string()),
    ).unwrap();

    println!("Chunking result:");
    println!("Total lines: {}", result.total_lines);
    println!("Was chunked: {}", result.was_chunked);
    println!("Number of chunks: {}", result.chunks.len());
    
    for chunk in &result.chunks {
        println!("Chunk {}: lines {}-{} ({} lines)", 
                 chunk.metadata.chunk_number,
                 chunk.metadata.start_line,
                 chunk.metadata.end_line,
                 chunk.metadata.line_count);
    }
    
    // Validate chunks
    engine.validate_chunks(&result.chunks).unwrap();
    println!("Chunk validation passed!");
}
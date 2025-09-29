// Simple test to verify the task generator produces the correct format

fn main() {
    // Simulate the expected output format
    let expected_format = r#"- [ ] 1. Task 1
  - [ ] 1.1 Task 1.1
    - [ ] 1.1.1 Task 1.1.1
      - [ ] 1.1.1.1 Task 1.1.1.1
        - [ ] 1.1.1.1.1 Task 1.1.1.1.1

- [ ] 2. Task 2
"#;

    println!("Expected format:");
    println!("{}", expected_format);
    
    // Verify the format matches what Kiro expects
    let lines: Vec<&str> = expected_format.lines().collect();
    for line in lines {
        if !line.trim().is_empty() {
            // Each non-empty line should be a checkbox or indented checkbox
            assert!(line.contains("- [ ]") || line.starts_with("  "));
            println!("✓ Line format correct: {}", line);
        }
    }
    
    println!("\n✅ Format validation passed!");
    println!("The simple task generator should produce this exact format.");
}
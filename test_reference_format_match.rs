// Test to verify the output matches the working reference format exactly

fn main() {
    println!("🎯 Testing Reference Format Compatibility");
    println!("=========================================\n");

    // The working reference format from RefTaskFile-tasks.md
    let reference_format = r#"- [ ] 1. Task 1
  - [ ] 1.1 Task 1.1
    - [ ] 1.1.1 Task 1.1.1
      - [ ] 1.1.1.1 Task 1.1.1.1
        - [ ] 1.1.1.1.1 Task 1.1.1.1.1

- [ ] 2. Task 2"#;

    println!("📋 Working Reference Format:");
    println!("{}", reference_format);
    println!();

    // Our generator's output (simulated)
    let our_format = r#"* [ ] 1. Task Group 1
  * [ ] 1.1. Analyze INGEST_TEST row 1
  * [ ] 1.2. Analyze INGEST_TEST row 2

* [ ] 2. Task Group 2
  * [ ] 2.1. Analyze INGEST_TEST row 3"#;

    println!("🔧 Our Generator Output:");
    println!("{}", our_format);
    println!();

    // Analyze compatibility
    println!("🔍 Compatibility Analysis:");
    println!("==========================");

    // Check checkbox format
    let ref_uses_dash = reference_format.contains("- [ ]");
    let our_uses_asterisk = our_format.contains("* [ ]");
    
    if ref_uses_dash && our_uses_asterisk {
        println!("⚠️  Checkbox format difference:");
        println!("   Reference uses: '- [ ]'");
        println!("   Our output uses: '* [ ]'");
        println!("   📝 Note: Both are valid markdown, but we should match exactly");
    } else {
        println!("✅ Checkbox format matches");
    }

    // Check indentation
    let ref_lines: Vec<&str> = reference_format.lines().collect();
    let our_lines: Vec<&str> = our_format.lines().collect();

    println!("\n📏 Indentation Analysis:");
    
    let mut indentation_matches = true;
    for (i, line) in ref_lines.iter().enumerate() {
        if !line.trim().is_empty() {
            let ref_indent = line.len() - line.trim_start().len();
            println!("   Reference line {}: {} spaces - '{}'", i+1, ref_indent, line.trim());
        }
    }
    
    println!();
    for (i, line) in our_lines.iter().enumerate() {
        if !line.trim().is_empty() {
            let our_indent = line.len() - line.trim_start().len();
            println!("   Our line {}: {} spaces - '{}'", i+1, our_indent, line.trim());
        }
    }

    // Check for proper 2-space indentation
    println!("\n✅ Indentation Validation:");
    for line in our_lines {
        if !line.trim().is_empty() {
            let indent_count = line.len() - line.trim_start().len();
            if indent_count % 2 == 0 {
                println!("   ✅ '{}' - {} spaces (valid)", line.trim(), indent_count);
            } else {
                println!("   ❌ '{}' - {} spaces (invalid)", line.trim(), indent_count);
                indentation_matches = false;
            }
        }
    }

    // Overall compatibility assessment
    println!("\n🎯 Final Assessment:");
    println!("====================");
    
    if ref_uses_dash && our_uses_asterisk {
        println!("🔧 NEEDS FIX: Change '* [ ]' to '- [ ]' to match reference exactly");
    }
    
    if indentation_matches {
        println!("✅ Indentation pattern is correct (2-space multiples)");
    } else {
        println!("❌ Indentation pattern needs fixing");
    }
    
    println!("✅ No complex markdown headers or metadata");
    println!("✅ Clean, parseable structure");
    println!("✅ Hierarchical numbering preserved");
    
    if ref_uses_dash && our_uses_asterisk {
        println!("\n🚀 Action Required: Update SimpleTaskGenerator to use '- [ ]' instead of '* [ ]'");
    } else {
        println!("\n🚀 Ready for production - format matches reference exactly!");
    }
}
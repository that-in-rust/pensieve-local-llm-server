---
inclusion: always
---

# Rust Code Conventions

## The Essence

**Write Rust code that compiles on the first try by using the type system to prevent bugs.**

## Core Rules

### Ownership & Borrowing
- Each value has one owner
- Either one `&mut T` OR many `&T` references
- Values dropped when owner goes out of scope

### API Design
- **Accept**: `&str`, `&[T]` (borrowed)
- **Store**: `String`, `Vec<T>` (owned)  
- **Return**: `String`, `Vec<T>` (owned)

### Error Handling
- **Libraries**: `thiserror` for structured errors
- **Applications**: `anyhow` for context
- **Propagation**: `?` operator

### Type Safety
- **Newtype pattern**: `UserId(Uuid)` prevents confusion
- **Make invalid states unrepresentable**: Use enums
- **Parse, don't validate**: Validation in constructors

## Essential Patterns

### Smart Pointers
- `Box<T>`: Heap allocation
- `Rc<T>` / `Arc<T>`: Shared ownership (single/multi-threaded)
- `RefCell<T>` / `Mutex<T>`: Interior mutability

### Async Patterns
```rust
// Structured concurrency
use tokio::task::JoinSet;

// Timeout all external calls
tokio::time::timeout(Duration::from_secs(30), operation()).await??

// Offload blocking work
tokio::task::spawn_blocking(|| cpu_intensive_work()).await?
```

### Database with SQLx
```rust
// Compile-time SQL validation
sqlx::query_as!(User, "SELECT * FROM users WHERE id = $1", id)
    .fetch_optional(&pool)
    .await?
```

### Testing
```rust
#[tokio::test]
async fn test_performance_contract() {
    let start = Instant::now();
    let result = operation().await.unwrap();
    assert!(start.elapsed() < Duration::from_millis(100));
}
```

## Common Patterns

### Error Handling
```rust
// Library errors
#[derive(Error, Debug)]
pub enum MyError {
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),
    #[error("Validation failed: {field}")]
    Validation { field: String },
}

// Application errors
pub async fn operation() -> anyhow::Result<()> {
    risky_call()
        .await
        .with_context(|| "Operation failed")?;
    Ok(())
}
```

### Resource Management
```rust
// RAII with Drop
pub struct Guard {
    resource: Option<Resource>,
}

impl Drop for Guard {
    fn drop(&mut self) {
        if let Some(r) = self.resource.take() {
            r.cleanup();
        }
    }
}
```

## Development Tools

### Code Search & Analysis
- **Use `ast-grep` instead of `grep`**: Syntax-aware searching for Rust code
- **Pattern matching**: `ast-grep --pattern 'fn $NAME($$$) { $$$ }'` finds all functions
- **Structural search**: Find complex patterns like error handling, async usage
- **Refactoring**: Safe code transformations with AST understanding

```bash
# Find all unwrap() calls (potential panics)
ast-grep --pattern '$_.unwrap()'

# Find functions without error handling
ast-grep --pattern 'fn $NAME($$$) -> $TYPE { $$$ }' --not-pattern 'Result'

# Find async functions
ast-grep --pattern 'async fn $NAME($$$) { $$$ }'
```

## Anti-Patterns

❌ **Never panic in production**: Use `Result` instead of `unwrap()`  
❌ **Don't ignore errors**: Handle with `?` or explicit match  
❌ **Avoid unnecessary cloning**: Use references when possible  
❌ **Don't mix threading types**: `Rc` is not `Send`, use `Arc`  
❌ **Don't use grep for code search**: Use `ast-grep` for syntax-aware analysis
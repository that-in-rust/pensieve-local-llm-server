# pensieve-validator

A comprehensive validation framework for the Pensieve CLI tool and other command-line applications.

## Real-World Dataset Validation

This framework can validate Pensieve against large real-world datasets such as the RustRAW20250920 corpus.

### Quick Start

```bash
# Ensure you have the Pensieve binary built:
# cargo build --release --bin pensieve

# Validate the real dataset (requires explicit confirmation)
pensieve-validator validate \
  --directory /Users/neetipatni/downloads/RustRAW20250920 \
  --confirm \
  --output-dir ./validation_reports \
  --pensieve-binary ../target/release/pensieve
```

### Running the Integration Test

After adding dependencies, you can run the end-to-end integration test:

```bash
cargo test --test real_world_dataset
```

The test will invoke the `validate` command on the real dataset and assert a successful exit.

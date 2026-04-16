# nemotron-asr-rs

Rust bindings for [nemotron-asr.cpp](https://github.com/your-org/nemotron-asr.cpp) - high-performance streaming automatic speech recognition (ASR) with NVIDIA's Nemotron model.

## Project Structure

This workspace contains two crates:

### `nemotron-asr-sys`

Low-level FFI bindings to the C API of nemotron-asr.cpp. Generated using `bindgen` and includes:

- Raw C function bindings
- Type definitions matching the C structs
- GGML backend query functions

**You probably don't want to use this directly.** Use the safe `nemotron-asr` wrapper instead.

### `nemotron-asr`

Safe, idiomatic Rust wrapper providing streaming ASR interface.

See [nemotron-asr/README.md](nemotron-asr/README.md) for usage examples.

## Quick Start

```bash
# Build everything
cargo build --release

# Run tests
cargo test

# Try the example (requires model and audio)
cargo run --example transcribe_stream -- model.gguf audio.pcm

# List available backends
cargo run --example transcribe_stream
```

## Requirements

The `nemotron-asr.cpp` library must be built before using these bindings. The build script expects to find:

- `../nemotron-asr.cpp/libnemotron_asr.a` - Main library
- `../nemotron-asr.cpp/ggml/build/src/libggml.a` - GGML core
- `../nemotron-asr.cpp/ggml/build/src/libggml-base.a` - GGML base
- Header files in `../nemotron-asr.cpp/src/` and `../nemotron-asr.cpp/ggml/include/`

Build nemotron-asr.cpp with:

```bash
cd ../nemotron-asr.cpp
make clean
make GGML_BUILD=ggml/build-static
```

## Features

- **Streaming ASR** with adjustable latency (80ms to 1120ms)
- **Backend Selection** - any GGML supported backend, such as Vulkan, CUDA, Metal, or CPU.

## Example Usage

```rust
use nemotron_asr::{Context, CacheConfig, LatencyMode};

// Initialize model (e.g. load weights)
let mut ctx = Context::new("model.gguf", Some("Vulkan"))?;

// Configure for low-latency streaming
let config = CacheConfig::with_latency(LatencyMode::PureCausal);

// Create stream (multiple streams can be created for a single context)
let mut stream = ctx.create_stream(Some(&config))?;

// Process audio chunks
for chunk in audio_chunks {
    let text = stream.process(&chunk);
    print!("{}", text);
}

// Get final transcription
let final_text = stream.finalize();
println!("{}", final_text);
```

## License

MIT

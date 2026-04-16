# nemotron-asr

Safe, idiomatic Rust bindings for [nemotron-asr.cpp](https://github.com/your-org/nemotron-asr.cpp) - a high-performance streaming automatic speech recognition (ASR) engine based on NVIDIA's Nemotron model.

## Features

- **Safe Rust API** - Memory-safe abstractions over the C++ library
- **Streaming ASR** - Low-latency real-time transcription with configurable lookahead
- **Multiple Backends** - Support for CPU, Vulkan, CUDA, and Metal backends via GGML
- **Flexible Configuration** - Adjustable latency modes and cache configurations
- **Easy to Use** - Intuitive API with proper error handling

## Example

```rust
use nemotron_asr::{NemoContext, CacheConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load model with auto-selected backend
    let mut ctx = NemoContext::new("model.gguf", None)?;

    // Create streaming context with default config
    let config = CacheConfig::default();
    let mut stream = ctx.create_stream(Some(&config))?;

    // Process audio chunks (16-bit PCM, 16kHz, mono)
    let audio: Vec<i16> = /* your audio data */;
    let text = stream.process(&audio)?;
    println!("{}", text);

    // Finalize to get remaining transcription
    let final_text = stream.finalize()?;
    println!("{}", final_text);

    Ok(())
}
```

## Command-line Example

The crate includes a `transcribe_stream` example that mimics the functionality of the C++ version:

```bash
# Transcribe from file
cargo run --example transcribe_stream model.gguf audio.pcm

# With custom latency mode (0=pure causal, 1=ultra low, 6=low, 13=default)
cargo run --example transcribe_stream model.gguf audio.pcm 80 0

# Select specific backend
cargo run --example transcribe_stream model.gguf audio.pcm 80 0 --backend Vulkan

# Stream from stdin (pipe from ffmpeg)
ffmpeg -i audio.mp3 -f s16le -ar 16000 -ac 1 - | \
  cargo run --example transcribe_stream model.gguf -

# Real-time microphone transcription
arecord -f S16_LE -r 16000 -c 1 - | \
  cargo run --example transcribe_stream model.gguf - 80 0
```

## Latency Modes

The library supports several latency/quality tradeoffs:

| Mode | Right Context | Latency | Use Case |
|------|--------------|---------|----------|
| Pure Causal | 0 | 80ms | Ultra-low latency real-time |
| Ultra Low | 1 | 160ms | Very responsive interactive |
| Low | 6 | 560ms | Balanced latency/quality |
| Default | 13 | 1120ms | Best quality offline |

## Backend Support

The library automatically discovers available backends at runtime:

```rust
use nemotron_asr::{load_backends, list_backends};

load_backends();
for backend in list_backends() {
    println!("Available: {}", backend.name()?);
}
```

Supported backends (via GGML):
- **Vulkan** - Cross-platform GPU acceleration
- **CUDA** - NVIDIA GPU acceleration
- **Metal** - Apple Silicon GPU acceleration
- **CPU** - Optimized CPU variants (AVX2, AVX512, etc.)

## Requirements

- Rust 2021 edition or later
- nemotron-asr.cpp library (built and linked via nemotron-asr-sys)
- Model weights in GGUF format

## Architecture

This crate provides safe, idiomatic Rust wrappers around the raw FFI bindings in `nemotron-asr-sys`. The main types are:

- `NemoContext` - Model context with automatic resource management
- `StreamContext` - Streaming transcription session
- `CacheConfig` - Configuration for streaming parameters
- `BackendDevice` - Information about available compute backends

All types implement proper `Drop` to ensure resources are freed, and are `Send` for safe use in multi-threaded contexts.

## License

MIT

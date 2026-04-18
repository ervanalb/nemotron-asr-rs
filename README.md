# nemotron-asr-rs

[![Crates.io](https://img.shields.io/crates/v/nemotron-asr.svg)](https://crates.io/crates/nemotron-asr)
[![Documentation](https://docs.rs/nemotron-asr/badge.svg)](https://docs.rs/nemotron-asr)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Rust bindings for [nemotron-asr.cpp](https://github.com/m1el/nemotron-asr.cpp) - high-performance streaming automatic speech recognition (ASR) with NVIDIA's Nemotron model, featuring:

- **Streaming ASR** with adjustable latency (80ms to 1120ms)
- **Backend Selection** - any GGML supported backend, such as Vulkan, CUDA, Metal, or CPU.
- **Low dependency** - and able to link fully statically (certain backends may require additional system libraries)

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
nemotron-asr = "0.1"
```

## Project Structure

This workspace contains two crates:

`nemotron-asr-sys` is low-level FFI bindings to the C API of nemotron-asr.cpp.
You probably don't want to use this directly.
Use the safe `nemotron-asr` wrapper instead.

`nemotron-asr` provides safe, idiomatic Rust wrapper providing streaming ASR interface.

## Quick Start

```bash
# Show help and list available backends
cargo run --example transcribe_stream -- --help

# Transcribe a WAV file
cargo run --example transcribe_stream -- --wav audio.wav

# Transcribe from system microphone
cargo run --example transcribe_stream -- --microphone

# Use the 8-bit quantized model
cargo run --example transcribe_stream -- --model nemotron-speech-streaming-0.6B-v0.1.Q8_0.gguf --microphone

# Use a specific backend and latency settings
cargo run --example transcribe_stream -- --microphone --backend Vulkan0 --right-context 0

# Change the compile-time configuration
GGML_VULKAN=ON cargo run --example transcribe_stream --features ggml_backend_dl -- #<runtime options>#
```

## Example Usage

```rust
use nemotron_asr::{Context, CacheConfig, LatencyMode};

// Initialize model (e.g. load weights)
let mut ctx = Context::new("model.gguf", Some("CPU"))?;

// Configure 560ms latency
let config = CacheConfig::with_latency(LatencyMode::Low);

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

## Building

For your convenience, the source code for `nemotron-asr.cpp` is included.
By default, `build.rs` will compile it with the given configuration options specified through environment variables.
This requires `cmake` and a compiler.
If you want to skip this and link your own library, disable the `vendored` feature.

## Configuration options

The default settings will produce a mostly static build that supports CPU execution.
The system must provide OpenMP and the C++ standard library.

You should consider using the `ggml_backend_dl` feature for use cases beyond local prototypes.
Enabling this feature will build a folder of dylibs, one for each enabled backend.
Compatible backends are enumerated and linked at runtime
while the main program linkage remains static.
This adds some complexity in that you will have to bundle and distribute these dylibs with your application,
but it allows flexibility if you're unsure what hardware or libraries may be present on a target system.

Here are some other ways to configure the build:

* `CXXSTDLIB_LINKAGE=[dylib]|static|none` Sets how the C++ standard library should be linked in.
* `CXXSTDLIB=[stdc++]|something else` Sets the library name of the C++ standard library
* `GGML_CPU=OFF|[ON]` Whether to build the CPU backend.
* `GGML_OPENMP=OFF|[ON]` Whether to build with OpenMP for the CPU backend. Requires a system library for OpenMP.
* `GGML_CUDA=[OFF]|ON` Whether to build the CUDA backend. Requires CUDA runtime and libraries.
* `GGML_VULKAN=[OFF]|ON` Whether to build the Vulkan backend. Requires a Vulkan system library.
* `GGML_METAL=[OFF]|ON` Whether to build the Metal backend (macOS only). Requires Metal framework.
* `GGML_SYCL=[OFF]|ON` Whether to build the SYCL backend. Requires SYCL implementation like Intel oneAPI.
* `GGML_OPENCL=[OFF]|ON` Whether to build the OpenCL backend. Requires OpenCL library.
* `GGML_CANN=[OFF]|ON` Whether to build the CANN backend. Requires Ascend CANN libraries.

## License

MIT

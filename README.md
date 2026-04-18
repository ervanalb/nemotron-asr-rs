# nemotron-asr-rs

Rust bindings for [nemotron-asr.cpp](https://github.com/your-org/nemotron-asr.cpp) - high-performance streaming automatic speech recognition (ASR) with NVIDIA's Nemotron model, featuring:

- **Streaming ASR** with adjustable latency (80ms to 1120ms)
- **Backend Selection** - any GGML supported backend, such as Vulkan, CUDA, Metal, or CPU.
- **Low system library count** - able to link fully statically (certain backends may require additional system libraries)

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
# List available backends
cargo run --example transcribe_stream

# Try the example (requires model and audio)
cargo run --example transcribe_stream -- model.gguf audio.pcm

# Run the example with a different configuration
cargo run --example transcribe_stream --features ggml_backend_dl
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
* `GGML_OPENMP=OFF|[ON]` Whether to build with OpenMP for the CPU backend. Requires a system library for OpenMP.

## Example Usage

```rust
use nemotron_asr::{Context, CacheConfig, LatencyMode};

// Initialize model (e.g. load weights)
let mut ctx = Context::new("model.gguf", Some("CPU"))?;

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

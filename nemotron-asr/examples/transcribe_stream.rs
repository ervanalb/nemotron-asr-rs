use nemotron_asr::{backend_count, get_backend, load_backends, CacheConfig, Context};
use std::env;
use std::fs::File;
use std::io::{self, stdin, Read};
use std::time::Instant;

fn print_usage(prog: &str) {
    eprintln!("Usage: {} <model.gguf> <audio.pcm|-|--stdin> [chunk_ms] [right_context] [--backend <name>]", prog);
    eprintln!();
    eprintln!("  model.gguf      - GGUF model file");
    eprintln!("  audio.pcm       - Audio file (PCM i16le 16kHz mono)");
    eprintln!("  - or --stdin    - Read audio from stdin");
    eprintln!("  chunk_ms        - Chunk size in milliseconds (default: 80)");
    eprintln!("  right_context   - Attention right context (0, 1, 6, or 13, default: 0)");
    eprintln!("  --backend <name> - Select backend (default: auto-select first available)");
    eprintln!();

    // Load backends to discover what's available
    load_backends();
    let dev_count = backend_count();

    eprintln!("Available backends:");
    if dev_count == 0 {
        eprintln!("  (none available)");
    } else {
        for i in 0..dev_count {
            if let Some(dev) = get_backend(i) {
                eprintln!("  {}{}", dev.name(), if i == 0 { " (default)" } else { "" });
            }
        }
    }

    eprintln!();
    eprintln!("Streaming modes:");
    eprintln!("  right_context=0  - Pure causal, 80ms latency");
    eprintln!("  right_context=1  - 160ms latency");
    eprintln!("  right_context=6  - 560ms latency");
    eprintln!("  right_context=13 - 1120ms latency (best quality)");
    eprintln!();
    eprintln!("Examples:");
    eprintln!("  {} weights/model.gguf audio.pcm 80 0", prog);
    eprintln!(
        "  {} weights/model.gguf audio.pcm 80 0 --backend Vulkan",
        prog
    );
    eprintln!(
        "  ffmpeg -i audio.mp3 -f s16le -ar 16000 -ac 1 - | {} weights/model.gguf -",
        prog
    );
    eprintln!(
        "  arecord -f S16_LE -r 16000 -c 1 - | {} weights/model.gguf - 80 0 --backend CPU",
        prog
    );
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 {
        print_usage(&args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];
    let audio_path = &args[2];
    let mut chunk_ms = 80;
    let mut right_context = 0;
    let mut backend_name: Option<String> = None;
    let read_from_stdin = audio_path == "-" || audio_path == "--stdin";

    // Parse arguments
    let mut i = 3;
    let mut positional_arg = 0;

    while i < args.len() {
        if args[i] == "--backend" {
            if i + 1 < args.len() {
                backend_name = Some(args[i + 1].clone());
                i += 2;
            } else {
                eprintln!("Error: --backend requires a backend name");
                std::process::exit(1);
            }
        } else {
            if positional_arg == 0 {
                chunk_ms = args[i].parse().unwrap_or(80);
                if chunk_ms < 10 {
                    eprintln!("Error: chunk_ms must be >= 10 (got {})", chunk_ms);
                    std::process::exit(1);
                }
                positional_arg += 1;
            } else if positional_arg == 1 {
                right_context = args[i].parse().unwrap_or(0);
                if ![0, 1, 6, 13].contains(&right_context) {
                    eprintln!(
                        "Warning: non-standard right_context={} (use 0, 1, 6, or 13)",
                        right_context
                    );
                }
                positional_arg += 1;
            }
            i += 1;
        }
    }

    // Calculate chunk size in samples
    let chunk_samples = chunk_ms * 16; // 16 samples per ms at 16kHz

    eprintln!("Configuration:");
    eprintln!("  Model:          {}", model_path);
    eprintln!(
        "  Audio:          {}",
        if read_from_stdin { "stdin" } else { audio_path }
    );
    eprintln!(
        "  Chunk size:     {} ms ({} samples)",
        chunk_ms, chunk_samples
    );
    eprintln!(
        "  Right context:  {} (latency: {} ms)",
        right_context,
        80 + right_context * 80
    );
    eprintln!();

    eprintln!("Loading model from {}...", model_path);
    let mut ctx = Context::new(model_path, backend_name.as_deref())?;

    eprintln!(
        "Model loaded successfully (backend: {})",
        ctx.backend_name()
    );

    // Initialize cache-aware streaming context
    let mut cache_cfg = CacheConfig::default();
    cache_cfg.set_right_context(right_context);

    let computed_chunk_samples = cache_cfg.chunk_samples();

    let mut stream_ctx = ctx.create_stream(Some(&cache_cfg))?;

    // Open input source (stdin or file)
    let mut input: Box<dyn Read> = if read_from_stdin {
        eprintln!("Reading audio from stdin...\n");
        Box::new(stdin())
    } else {
        eprintln!("Streaming from file...\n");
        Box::new(File::open(audio_path)?)
    };

    let start_time = Instant::now();
    let mut total_samples_processed = 0usize;
    let mut buffer = vec![0i16; computed_chunk_samples as usize];

    loop {
        // Read chunk_samples worth of i16 samples
        let bytes_to_read = (computed_chunk_samples as usize) * std::mem::size_of::<i16>();
        let mut byte_buffer = vec![0u8; bytes_to_read];

        let bytes_read = input.read(&mut byte_buffer)?;

        if bytes_read == 0 {
            break;
        }

        // Convert bytes to i16 samples
        let n_samples = bytes_read / std::mem::size_of::<i16>();
        for i in 0..n_samples {
            let byte_idx = i * 2;
            if byte_idx + 1 < byte_buffer.len() {
                buffer[i] = i16::from_le_bytes([byte_buffer[byte_idx], byte_buffer[byte_idx + 1]]);
            }
        }

        total_samples_processed += n_samples;

        let new_text = stream_ctx.process(&buffer[..n_samples]);

        if !new_text.is_empty() {
            print!("{}", new_text);
            io::Write::flush(&mut io::stdout())?;
        }

        if bytes_read < bytes_to_read {
            break;
        }
    }

    // Finalize to flush any remaining audio
    let final_text = stream_ctx.finalize();
    if !final_text.is_empty() {
        print!("{}", final_text);
        io::Write::flush(&mut io::stdout())?;
    }
    println!();

    let processing_time = start_time.elapsed();
    let processing_time_sec = processing_time.as_secs_f64();
    let total_duration_sec = total_samples_processed as f64 / 16000.0;

    eprintln!("\n=== Complete ===");
    eprintln!("\nStatistics:");
    eprintln!("  Audio duration:      {:.2} sec", total_duration_sec);
    eprintln!("  Processing time:     {:.2} sec", processing_time_sec);
    if total_duration_sec > 0.0 {
        eprintln!(
            "  Real-time factor:    {:.3}x",
            processing_time_sec / total_duration_sec
        );
    }

    Ok(())
}

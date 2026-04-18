use clap::{Arg, ArgAction, Command};
use indicatif::{ProgressBar, ProgressStyle};
use nemotron_asr::{backend_count, get_backend, CacheConfig, Context};
use std::fs::{self, File};
use std::io;
use std::path::{Path, PathBuf};
use std::time::Instant;

struct Args {
    model: PathBuf,
    wav: Option<PathBuf>,
    microphone: bool,
    chunk_ms: i32,
    right_context: i32,
    backend: Option<String>,
    download_url: String,
}

fn init() {
    #[cfg(feature = "ggml_backend_dl")]
    {
        // In an actual application, you wouldn't use this compile-time environment variable directly.
        // Instead, in your build.rs, you would copy all the backend dylibs
        // from the location specified in this environment variable
        // to a resource folder in your application's OUT_DIR.
        // Then, you would bundle that resource folder with your application,
        // and use its (relative) path in the load_backends_from_path() call.

        let backend_dir = env!("DEP_NEMOTRON_ASR_GGML_BACKEND_DIR");
        eprintln!("Loading plugins from {backend_dir}");
        nemotron_asr::load_backends_from_path(backend_dir);
    }
}

fn download_model(url: &str, dest_path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("\nModel not found locally. Downloading from {}", url);

    // Create parent directory if it doesn't exist
    if let Some(parent) = dest_path.parent() {
        fs::create_dir_all(parent)?;
    }

    let mut response = reqwest::blocking::get(url)?;
    let total_size = response.content_length().unwrap_or(0);

    let pb = ProgressBar::new(total_size);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{msg}\n{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}, {eta})")?
            .progress_chars("#>-"),
    );
    pb.set_message(format!("Downloading to {}", dest_path.display()));

    let mut file = File::create(dest_path)?;
    std::io::copy(&mut response, &mut pb.wrap_write(&mut file))?;

    Ok(())
}

fn ensure_model(
    model_path: &Path,
    download_url: &str,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    if model_path.exists() {
        return Ok(model_path.to_path_buf());
    }

    // Construct download URL
    let filename = model_path
        .file_name()
        .ok_or("Invalid model path")?
        .to_str()
        .ok_or("Invalid UTF-8 in filename")?;

    let full_url = if download_url.ends_with('/') {
        format!("{}{}", download_url, filename)
    } else {
        format!("{}/{}", download_url, filename)
    };

    download_model(&full_url, model_path)?;
    Ok(model_path.to_path_buf())
}

fn validate_wav_format(
    reader: &hound::WavReader<std::io::BufReader<File>>,
) -> Result<(), Box<dyn std::error::Error>> {
    let spec = reader.spec();

    if spec.channels != 1 {
        return Err(format!(
            "WAV file must be mono (1 channel), got {} channels",
            spec.channels
        )
        .into());
    }

    if spec.sample_rate != 16000 {
        return Err(format!(
            "WAV file must be 16kHz sample rate, got {} Hz",
            spec.sample_rate
        )
        .into());
    }

    if spec.bits_per_sample != 16 {
        return Err(format!("WAV file must be 16-bit, got {} bits", spec.bits_per_sample).into());
    }

    if spec.sample_format != hound::SampleFormat::Int {
        return Err("WAV file must use integer sample format".into());
    }

    Ok(())
}

fn process_wav_file(
    reader: &mut hound::WavReader<std::io::BufReader<File>>,
    stream_ctx: &mut nemotron_asr::Stream,
    computed_chunk_samples: i32,
) -> Result<(usize, f64), Box<dyn std::error::Error>> {
    let start_time = Instant::now();
    let mut total_samples_processed = 0usize;

    let samples: Result<Vec<i16>, _> = reader.samples::<i16>().collect();
    let samples = samples?;

    for chunk in samples.chunks(computed_chunk_samples as usize) {
        total_samples_processed += chunk.len();

        let new_text = stream_ctx.process(chunk);

        if !new_text.is_empty() {
            print!("{}", new_text);
            io::Write::flush(&mut io::stdout())?;
        }
    }

    // Finalize to flush any remaining audio
    let final_text = stream_ctx.finalize();
    if !final_text.is_empty() {
        print!("{}", final_text);
        io::Write::flush(&mut io::stdout())?;
    }
    println!();

    let processing_time = start_time.elapsed().as_secs_f64();
    Ok((total_samples_processed, processing_time))
}

fn process_microphone(
    stream_ctx: nemotron_asr::Stream,
    computed_chunk_samples: i32,
) -> Result<(usize, f64), Box<dyn std::error::Error>> {
    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
    use std::sync::{Arc, Mutex};

    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or("No input device available")?;

    eprintln!("Using microphone: {}", device.name()?);

    let config = cpal::StreamConfig {
        channels: 1,
        sample_rate: cpal::SampleRate(16000),
        buffer_size: cpal::BufferSize::Fixed(computed_chunk_samples as u32),
    };

    let start_time = Instant::now();
    let total_samples = Arc::new(Mutex::new(0usize));
    let total_samples_clone = total_samples.clone();

    let stream_ctx_arc = Arc::new(Mutex::new(stream_ctx));
    let stream_ctx_clone = stream_ctx_arc.clone();

    let stream = device.build_input_stream(
        &config,
        move |data: &[i16], _: &cpal::InputCallbackInfo| {
            let mut ctx = stream_ctx_clone.lock().unwrap();
            let new_text = ctx.process(data);

            if !new_text.is_empty() {
                print!("{}", new_text);
                io::Write::flush(&mut io::stdout()).ok();
            }

            *total_samples_clone.lock().unwrap() += data.len();
        },
        move |err| {
            eprintln!("Stream error: {}", err);
        },
        None,
    )?;

    stream.play()?;

    eprintln!("Recording from microphone... Press Ctrl+C to stop.\n");

    // Wait for Ctrl+C
    let running = Arc::new(Mutex::new(true));
    let r = running.clone();

    ctrlc::set_handler(move || {
        *r.lock().unwrap() = false;
    })?;

    while *running.lock().unwrap() {
        std::thread::sleep(std::time::Duration::from_millis(100));
    }

    drop(stream);

    // Finalize
    let final_text = stream_ctx_arc.lock().unwrap().finalize();
    if !final_text.is_empty() {
        print!("{}", final_text);
        io::Write::flush(&mut io::stdout())?;
    }
    println!();

    let processing_time = start_time.elapsed().as_secs_f64();
    let total = *total_samples.lock().unwrap();

    Ok((total, processing_time))
}

fn get_backends_help() -> String {
    let dev_count = backend_count();
    let mut help = String::from("\n\x1b[1m\x1b[4mAvailable backends:\x1b[0m\n");

    if dev_count == 0 {
        help.push_str("  (none available)\n");
    } else {
        for i in 0..dev_count {
            if let Some(dev) = get_backend(i) {
                help.push_str(&format!(
                    "  \x1b[1m{}\x1b[0m{}\n",
                    dev.name(),
                    if i == 0 { " (default)" } else { "" }
                ));
            }
        }
    }

    help
}

fn build_cli() -> Command {
    Command::new("transcribe_stream")
        .about("Stream ASR transcription using Nemotron models")
        .arg(
            Arg::new("model")
                .short('m')
                .long("model")
                .value_name("MODEL")
                .help("Path to the GGUF model file (will auto-download if not found)")
                .default_value("weights/nemotron-speech-streaming-0.6B-v0.1.f32.gguf")
        )
        .arg(
            Arg::new("wav")
                .short('w')
                .long("wav")
                .value_name("WAV")
                .help("WAV file to transcribe")
                .conflicts_with("microphone")
        )
        .arg(
            Arg::new("microphone")
                .short('i')
                .long("microphone")
                .help("Transcribe from system microphone")
                .action(ArgAction::SetTrue)
                .conflicts_with("wav")
        )
        .arg(
            Arg::new("chunk_ms")
                .short('c')
                .long("chunk-ms")
                .value_name("CHUNK_MS")
                .help("Chunk size in milliseconds")
                .default_value("80")
        )
        .arg(
            Arg::new("right_context")
                .short('r')
                .long("right-context")
                .value_name("RIGHT_CONTEXT")
                .help("Attention right context (0, 1, 6, or 13)\n  0 = Pure causal, 80ms latency\n  1 = 160ms latency\n  6 = 560ms latency\n  13 = 1120ms latency (best quality)")
                .default_value("6")
        )
        .arg(
            Arg::new("backend")
                .short('b')
                .long("backend")
                .value_name("BACKEND")
                .help("Select backend (default: auto-select first available)")
        )
        .arg(
            Arg::new("download_url")
                .long("download-url")
                .value_name("DOWNLOAD_URL")
                .help("Base URL for downloading models")
                .default_value("https://huggingface.co/m1el/nemotron-speech-streaming-0.6B-gguf/resolve/main/")
        )
        .after_help(get_backends_help())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    init();

    let matches = build_cli().get_matches();

    let args = Args {
        model: PathBuf::from(matches.get_one::<String>("model").unwrap()),
        wav: matches.get_one::<String>("wav").map(PathBuf::from),
        microphone: matches.get_flag("microphone"),
        chunk_ms: matches.get_one::<String>("chunk_ms").unwrap().parse()?,
        right_context: matches
            .get_one::<String>("right_context")
            .unwrap()
            .parse()?,
        backend: matches.get_one::<String>("backend").cloned(),
        download_url: matches
            .get_one::<String>("download_url")
            .unwrap()
            .to_string(),
    };

    // Validate arguments
    if args.chunk_ms < 10 {
        eprintln!("Error: chunk_ms must be >= 10 (got {})", args.chunk_ms);
        std::process::exit(1);
    }

    if ![0, 1, 6, 13].contains(&args.right_context) {
        eprintln!(
            "Warning: non-standard right_context={} (use 0, 1, 6, or 13)",
            args.right_context
        );
    }

    if !args.microphone && args.wav.is_none() {
        eprintln!("Error: Either --wav <file> or --microphone must be specified");
        std::process::exit(1);
    }

    // Ensure model exists (download if necessary)
    let model_path = ensure_model(&args.model, &args.download_url)?;

    eprintln!("Configuration:");
    eprintln!("  Model:          {}", model_path.display());
    if let Some(ref wav) = args.wav {
        eprintln!("  Audio:          {}", wav.display());
    } else {
        eprintln!("  Audio:          microphone");
    }
    eprintln!(
        "  Chunk size:     {} ms ({} samples)",
        args.chunk_ms,
        args.chunk_ms * 16
    );
    eprintln!(
        "  Right context:  {} (latency: {} ms)",
        args.right_context,
        80 + args.right_context * 80
    );

    eprintln!();

    eprintln!("Loading model from {}...", model_path.display());
    let mut ctx = Context::new(
        model_path.to_str().ok_or("Invalid UTF-8 in model path")?,
        args.backend.as_deref(),
    )?;

    eprintln!(
        "Model loaded successfully (backend: {})\n",
        ctx.backend_name()
    );

    // Initialize cache-aware streaming context
    let mut cache_cfg = CacheConfig::default();
    cache_cfg.set_right_context(args.right_context);

    let computed_chunk_samples = cache_cfg.chunk_samples();

    // Process audio based on source
    let (total_samples_processed, processing_time_sec) = if let Some(wav_path) = args.wav {
        let mut stream_ctx = ctx.create_stream(Some(&cache_cfg))?;
        let mut reader = hound::WavReader::open(&wav_path)?;
        validate_wav_format(&reader)?;

        eprintln!("Streaming from WAV file...\n");
        process_wav_file(&mut reader, &mut stream_ctx, computed_chunk_samples)?
    } else {
        let stream_ctx = ctx.create_stream(Some(&cache_cfg))?;
        process_microphone(stream_ctx, computed_chunk_samples)?
    };

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

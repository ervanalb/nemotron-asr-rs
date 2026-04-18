use std::env;

fn main() {
    // Propagate the ggml_backend_dir from nemotron-asr-sys to dependent crates
    if let Ok(backend_dir) = env::var("DEP_NEMOTRON_ASR_SYS_GGML_BACKEND_DIR") {
        // Expose to dependent crates via DEP_ variable
        println!("cargo:ggml_backend_dir={}", backend_dir);

        // Also expose to this crate's own code (including examples) via compile-time env var
        println!(
            "cargo:rustc-env=DEP_NEMOTRON_ASR_GGML_BACKEND_DIR={}",
            backend_dir
        );
    }
}

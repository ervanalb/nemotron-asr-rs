use std::env;
use std::path::PathBuf;

fn main() {
    // Get the path to the nemotron-asr.cpp directory
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let nemotron_dir = PathBuf::from(&manifest_dir)
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("nemotron-asr.cpp");

    let wrapper_path = PathBuf::from(&manifest_dir).join("wrapper.h");

    // Tell cargo to look for the library in the nemotron-asr.cpp directory
    println!("cargo:rustc-link-search=native={}", nemotron_dir.display());

    // Also add GGML library paths
    let ggml_lib_path = nemotron_dir.join("ggml/build/src");
    println!("cargo:rustc-link-search=native={}", ggml_lib_path.display());

    // Link to the static libraries (order matters for static linking)
    println!("cargo:rustc-link-lib=static=nemotron_asr");
    println!("cargo:rustc-link-lib=static=ggml");
    println!("cargo:rustc-link-lib=static=ggml-base");

    // Link to required system libraries
    // These may vary depending on the backend and platform
    println!("cargo:rustc-link-lib=stdc++");

    // Rerun if the library or headers change
    println!(
        "cargo:rerun-if-changed={}",
        nemotron_dir.join("libnemotron_asr.a").display()
    );
    println!("cargo:rerun-if-changed={}", wrapper_path.display());

    // Generate bindings using bindgen
    let bindings = bindgen::Builder::default()
        // Input header
        .header(wrapper_path.to_str().unwrap())
        // Allowlist Nemo API
        .allowlist_function("c_nemo_.*")
        .allowlist_function("nemo_cache_config_.*")
        .allowlist_type("nemo_.*")
        // Allowlist GGML backend API
        .allowlist_function("ggml_backend_load_all")
        .allowlist_function("ggml_backend_dev_count")
        .allowlist_function("ggml_backend_dev_get")
        .allowlist_function("ggml_backend_dev_name")
        .allowlist_type("ggml_backend_dev_t")
        // Rustify enums
        .rustified_enum("nemo_latency_mode")
        .generate()
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}

use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let manifest_path = PathBuf::from(&manifest_dir);

    // Path to nemotron-asr.cpp (now a submodule)
    let library_dir = manifest_path.parent().unwrap().join("nemotron-asr.cpp");

    #[cfg(feature = "vendored")]
    {
        // Build the vendored library
        build_vendored(&library_dir);
        link_vendored(&library_dir);
    }

    #[cfg(not(feature = "vendored"))]
    {
        // Link against system library
        link_system();
    }

    #[cfg(feature = "bindgen")]
    {
        // Generate bindings
        generate_bindings(&manifest_path);
    }
}

#[cfg(feature = "vendored")]
fn build_vendored(library_dir: &PathBuf) {
    let ggml_dir = library_dir.join("ggml");
    let build_dir = ggml_dir.join("build");

    // Clean previous builds - delete build directory if it exists
    if build_dir.exists() {
        std::fs::remove_dir_all(&build_dir).expect("Failed to remove build directory");
    }

    // Create fresh build directory
    std::fs::create_dir_all(&build_dir).expect("Failed to create build directory");

    // Run make clean in the library directory
    let clean_status = Command::new("make")
        .current_dir(&library_dir)
        .arg("clean")
        .status()
        .expect("Failed to run make");

    if !clean_status.success() {
        panic!("Make clean failed");
    }

    // Run cmake to configure GGML
    let cmake_status = Command::new("cmake")
        .current_dir(&build_dir)
        .arg("..")
        .arg("-DBUILD_SHARED_LIBS=OFF")
        .arg("-DGGML_OPENMP=OFF")
        //.arg("-DCMAKE_CXX_FLAGS=-static-libstdc++")
        .status()
        .expect("Failed to run cmake");

    if !cmake_status.success() {
        panic!("GGML configuration failed");
    }

    // Build GGML
    let num_jobs = num_cpus::get().to_string();

    let build_status = Command::new("cmake")
        .current_dir(&build_dir)
        .arg("--build")
        .arg(".")
        .arg("-j")
        .arg(&num_jobs)
        .status()
        .expect("Failed to run cmake");

    if !build_status.success() {
        panic!("GGML build failed");
    }

    // Build nemotron-asr using make (build only the static library)
    let make_status = Command::new("make")
        .current_dir(&library_dir)
        .arg("libnemotron_asr.a")
        //.env("LDFLAGS", "-static-libstdc++")
        .status()
        .expect("Failed to run make");

    if !make_status.success() {
        panic!("nemotron-asr build failed");
    }

    // Tell cargo to rerun if source files change
    println!(
        "cargo:rerun-if-changed={}",
        library_dir.join("src").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        library_dir.join("Makefile").display()
    );
}

#[cfg(feature = "vendored")]
fn link_vendored(nemotron_dir: &PathBuf) {
    // Tell cargo to look for the library in the nemotron-asr.cpp directory
    println!("cargo:rustc-link-search=native={}", nemotron_dir.display());

    // Link nemotron_asr library
    println!("cargo:rustc-link-lib=static=nemotron_asr");

    // Add GGML library path
    let ggml_lib_path = nemotron_dir.join("ggml/build/src");
    println!("cargo:rustc-link-search=native={}", ggml_lib_path.display());

    // Find and link all .a files in the GGML build directory
    if let Ok(entries) = fs::read_dir(&ggml_lib_path) {
        let mut lib_names: Vec<String> = Vec::new();

        for entry in entries.flatten() {
            if let Ok(file_name) = entry.file_name().into_string() {
                if file_name.starts_with("lib") && file_name.ends_with(".a") {
                    // Extract library name: libfoo.a -> foo
                    let lib_name = file_name
                        .strip_prefix("lib")
                        .unwrap()
                        .strip_suffix(".a")
                        .unwrap()
                        .to_string();
                    lib_names.push(lib_name);
                }
            }
        }

        // Sort for consistent linking order
        lib_names.sort();

        // Link each library
        for lib_name in lib_names {
            println!("cargo:rustc-link-lib=static={}", lib_name);
        }
    }

    // Link C++ standard library based on environment variables
    println!("cargo:rerun-if-env-changed=CXXSTDLIB_LINKAGE");
    println!("cargo:rerun-if-env-changed=CXXSTDLIB");
    let cxxstdlib_linkage = env::var("CXXSTDLIB_LINKAGE").unwrap_or_else(|_| "dylib".to_string());
    let cxxstdlib = env::var("CXXSTDLIB").unwrap_or_else(|_| "stdc++".to_string());

    match cxxstdlib_linkage.as_str() {
        "static" => {
            println!("cargo:rustc-link-lib=static={}", cxxstdlib);
        }
        "dylib" => {
            println!("cargo:rustc-link-lib={}", cxxstdlib);
        }
        "none" => {
            // Don't link C++ standard library
        }
        _ => {
            panic!(
                "Invalid CXXSTDLIB_LINKAGE value: {}. Must be 'static', 'dylib', or 'none'",
                cxxstdlib_linkage
            );
        }
    }
}

#[cfg(not(feature = "vendored"))]
fn link_system() {
    // Link against system-installed library
    println!("cargo:rustc-link-lib=nemotron_asr");
}

#[cfg(feature = "bindgen")]
fn generate_bindings(manifest_path: &PathBuf) {
    let wrapper_path = manifest_path.join("wrapper.h");

    println!("cargo:rerun-if-changed={}", wrapper_path.display());

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

    // Write the bindings to src/bindings.rs
    let bindings_path = manifest_path.join("src").join("bindings.rs");
    bindings
        .write_to_file(bindings_path)
        .expect("Couldn't write bindings!");
}

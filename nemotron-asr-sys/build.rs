use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

/// Get a CMake bool value from an environment variable
/// Returns the default if not set, validates that the value is either ON or OFF
fn get_cmake_bool(var_name: &str, default: bool) -> bool {
    println!("cargo:rerun-if-env-changed={}", var_name);

    let value =
        env::var(var_name).unwrap_or_else(|_| if default { "ON" } else { "OFF" }.to_string());
    match value.as_str() {
        "ON" => true,
        "OFF" => false,
        _ => panic!(
            "Invalid value for {}: '{}'. Must be 'ON' or 'OFF'",
            var_name, value
        ),
    }
}

/// Format a CMake boolean argument
fn cmake_bool_arg(name: &str, value: bool) -> String {
    format!("-D{}={}", name, if value { "ON" } else { "OFF" })
}

fn main() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let manifest_path = PathBuf::from(&manifest_dir);

    // Path to nemotron-asr.cpp source (submodule in this directory)
    let source_library_dir = manifest_path.join("nemotron-asr.cpp");

    #[cfg(feature = "vendored")]
    {
        // Copy source to OUT_DIR and build there to avoid modifying source tree
        let out_dir = env::var("OUT_DIR").unwrap();
        let build_library_dir = PathBuf::from(&out_dir).join("nemotron-asr.cpp");

        // Remove any existing copy
        if build_library_dir.exists() {
            fs::remove_dir_all(&build_library_dir).expect("Failed to remove old build directory");
        }

        // Copy source directory to OUT_DIR
        copy_dir_recursive(&source_library_dir, &build_library_dir)
            .expect("Failed to copy source directory to OUT_DIR");

        // Tell cargo to rerun if source files change
        println!(
            "cargo:rerun-if-changed={}",
            source_library_dir.join("src").display()
        );
        println!(
            "cargo:rerun-if-changed={}",
            source_library_dir.join("Makefile").display()
        );
        println!(
            "cargo:rerun-if-changed={}",
            source_library_dir.join("ggml").display()
        );

        // Build and link the vendored library from OUT_DIR
        build_and_link_vendored(&build_library_dir);
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

/// Recursively copy a directory
fn copy_dir_recursive(src: &PathBuf, dst: &PathBuf) -> std::io::Result<()> {
    fs::create_dir_all(dst)?;

    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let src_path = entry.path();
        let dst_path = dst.join(entry.file_name());

        if src_path.is_dir() {
            // Skip .git directories
            if entry.file_name() == ".git" {
                continue;
            }
            copy_dir_recursive(&src_path, &dst_path)?;
        } else {
            fs::copy(&src_path, &dst_path)?;
        }
    }

    Ok(())
}

#[cfg(feature = "vendored")]
fn build_and_link_vendored(library_dir: &PathBuf) {
    let ggml_dir = library_dir.join("ggml");
    let build_dir = ggml_dir.join("build");

    // Create build directory
    std::fs::create_dir_all(&build_dir).expect("Failed to create build directory");

    // Run cmake to configure GGML
    let backend_dl = cfg!(feature = "ggml_backend_dl");
    let openmp = get_cmake_bool("GGML_OPENMP", true);

    // Backend configuration
    let cpu = get_cmake_bool("GGML_CPU", true);
    let cuda = get_cmake_bool("GGML_CUDA", false);
    let vulkan = get_cmake_bool("GGML_VULKAN", false);
    let metal = get_cmake_bool("GGML_METAL", false);
    let sycl = get_cmake_bool("GGML_SYCL", false);
    let opencl = get_cmake_bool("GGML_OPENCL", false);
    let cann = get_cmake_bool("GGML_CANN", false);

    let mut cmake_cmd = Command::new("cmake");
    cmake_cmd
        .current_dir(&build_dir)
        .arg("..")
        .arg("-DBUILD_SHARED_LIBS=OFF")
        .arg(cmake_bool_arg("GGML_BACKEND_DL", backend_dl))
        .arg(cmake_bool_arg("GGML_OPENMP", openmp))
        .arg(cmake_bool_arg("GGML_CPU", cpu))
        .arg(cmake_bool_arg("GGML_CUDA", cuda))
        .arg(cmake_bool_arg("GGML_VULKAN", vulkan))
        .arg(cmake_bool_arg("GGML_METAL", metal))
        .arg(cmake_bool_arg("GGML_SYCL", sycl))
        .arg(cmake_bool_arg("GGML_OPENCL", opencl))
        .arg(cmake_bool_arg("GGML_CANN", cann));

    // When using backend_dl, also set GGML_NATIVE=OFF and GGML_CPU_ALL_VARIANTS=ON
    if backend_dl {
        cmake_cmd
            .arg("-DGGML_NATIVE=OFF")
            .arg("-DGGML_CPU_ALL_VARIANTS=ON");
    }

    let cmake_status = cmake_cmd.status().expect("Failed to run cmake");

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

    // Expose GGML backend plugins directory when using backend_dl
    if backend_dl {
        let bin_dir = ggml_dir.join("build/bin");

        // Expose the backends directory path to dependent crates
        println!("cargo:ggml_backend_dir={}", bin_dir.display());
    }

    // Build nemotron-asr using make (build only the static library)
    let make_status = Command::new("make")
        .current_dir(&library_dir)
        .arg("libnemotron_asr.a")
        .status()
        .expect("Failed to run make");

    if !make_status.success() {
        panic!("nemotron-asr build failed");
    }

    // === Linking ===

    // Tell cargo to look for the library in the nemotron-asr.cpp directory
    println!("cargo:rustc-link-search=native={}", library_dir.display());

    // Link nemotron_asr library
    println!("cargo:rustc-link-lib=static=nemotron_asr");

    // Find and link all .a files in the GGML build src directory (recursively)
    let ggml_lib_path = library_dir.join("ggml/build/src");
    let mut lib_names: Vec<String> = Vec::new();
    let mut search_dirs: std::collections::HashSet<PathBuf> = std::collections::HashSet::new();

    fn find_static_libs(
        dir: &PathBuf,
        lib_names: &mut Vec<String>,
        search_dirs: &mut std::collections::HashSet<PathBuf>,
    ) {
        if let Ok(entries) = fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    find_static_libs(&path, lib_names, search_dirs);
                } else if let Some(file_name) = path.file_name().and_then(|n| n.to_str()) {
                    if file_name.starts_with("lib") && file_name.ends_with(".a") {
                        // Extract library name: libfoo.a -> foo
                        let lib_name = file_name
                            .strip_prefix("lib")
                            .unwrap()
                            .strip_suffix(".a")
                            .unwrap()
                            .to_string();
                        lib_names.push(lib_name);
                        // Add the parent directory to search paths
                        if let Some(parent) = path.parent() {
                            search_dirs.insert(parent.to_path_buf());
                        }
                    }
                }
            }
        }
    }

    find_static_libs(&ggml_lib_path, &mut lib_names, &mut search_dirs);

    // Add all directories containing .a files to the search path
    for dir in &search_dirs {
        println!("cargo:rustc-link-search=native={}", dir.display());
    }

    // Sort for consistent linking order
    lib_names.sort();

    // Link each library
    for lib_name in lib_names {
        println!("cargo:rustc-link-lib=static={}", lib_name);
    }

    // Link system libraries for backends when not using backend_dl
    if !backend_dl {
        if openmp {
            println!("cargo:rustc-link-lib=gomp");
        }
        if cuda {
            println!("cargo:rustc-link-lib=cuda");
            println!("cargo:rustc-link-lib=cudart");
        }
        if vulkan {
            println!("cargo:rustc-link-lib=vulkan");
        }
        if metal {
            // Metal framework is macOS-specific
            println!("cargo:rustc-link-lib=framework=Metal");
            println!("cargo:rustc-link-lib=framework=Foundation");
        }
        if sycl {
            // SYCL library linking depends on implementation (Intel oneAPI, etc.)
            println!("cargo:rustc-link-lib=sycl");
        }
        if opencl {
            println!("cargo:rustc-link-lib=OpenCL");
        }
        if cann {
            println!("cargo:rustc-link-lib=ascendcl");
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
        .allowlist_function("ggml_backend_load_all_from_path");

    let bindings = bindings
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

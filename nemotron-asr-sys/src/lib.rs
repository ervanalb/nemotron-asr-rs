//! Low-level FFI bindings for nemotron-asr.cpp
//!
//! This crate provides raw, unsafe bindings to the C API of nemotron-asr.cpp.
//! For a safe, idiomatic Rust API, use the `nemotron-asr` crate instead.
//!
//! # Usage
//!
//! The bindings are automatically generated from the C header files using bindgen.
//! All functions are unsafe and require careful handling of pointers and memory management.
//!
//! # Example
//!
//! ```no_run
//! use nemotron_asr_sys::*;
//! use std::ffi::CString;
//! use std::ptr;
//!
//! unsafe {
//!     // Initialize model with default backend
//!     let model_path = CString::new("/path/to/model.gguf").unwrap();
//!     let ctx = c_nemo_init_with_backend(model_path.as_ptr(), ptr::null());
//!
//!     if !ctx.is_null() {
//!         // Create streaming context with default config
//!         let stream_ctx = c_nemo_stream_init(ctx, ptr::null());
//!
//!         // Process audio...
//!
//!         // Cleanup
//!         c_nemo_stream_free(stream_ctx);
//!         c_nemo_free(ctx);
//!     }
//! }
//! ```

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

// Include the auto-generated bindings
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_config_default() {
        unsafe {
            let config = nemo_cache_config_default();
            assert_eq!(config.sample_rate, 16000);
            assert_eq!(config.n_mels, 128);
        }
    }

    #[test]
    fn test_latency_mode_values() {
        assert_eq!(nemo_latency_mode::NEMO_LATENCY_PURE_CAUSAL as i32, 0);
        assert_eq!(nemo_latency_mode::NEMO_LATENCY_ULTRA_LOW as i32, 1);
        assert_eq!(nemo_latency_mode::NEMO_LATENCY_LOW as i32, 6);
        assert_eq!(nemo_latency_mode::NEMO_LATENCY_DEFAULT as i32, 13);
    }
}

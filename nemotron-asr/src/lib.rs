use nemotron_asr_sys as ffi;
use std::ffi::{CStr, CString};
use std::path::Path;
use std::ptr;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum InitializationError {
    #[error("Failed to initialize model")]
    InitializationFailed,
}

#[derive(Error, Debug)]
pub enum StreamInitializationError {
    #[error("Failed to create streaming context")]
    StreamInitializationFailed,
}

/// Latency mode for streaming ASR
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum LatencyMode {
    /// Pure causal, 80ms latency
    PureCausal,
    /// 160ms latency
    UltraLow,
    /// 560ms latency
    Low,
    /// 1.12s latency (best quality)
    Default,
}

impl From<LatencyMode> for ffi::nemo_latency_mode {
    fn from(mode: LatencyMode) -> Self {
        match mode {
            LatencyMode::PureCausal => ffi::nemo_latency_mode::NEMO_LATENCY_PURE_CAUSAL,
            LatencyMode::UltraLow => ffi::nemo_latency_mode::NEMO_LATENCY_ULTRA_LOW,
            LatencyMode::Low => ffi::nemo_latency_mode::NEMO_LATENCY_LOW,
            LatencyMode::Default => ffi::nemo_latency_mode::NEMO_LATENCY_DEFAULT,
        }
    }
}

/// Cache configuration for streaming
#[derive(Debug, Clone)]
pub struct CacheConfig {
    inner: ffi::nemo_cache_config,
}

impl CacheConfig {
    /// Create default cache configuration
    pub fn default() -> Self {
        Self {
            inner: unsafe { ffi::nemo_cache_config_default() },
        }
    }

    /// Create cache configuration with specific latency mode
    pub fn with_latency(mode: LatencyMode) -> Self {
        Self {
            inner: unsafe { ffi::nemo_cache_config_with_latency(mode.into()) },
        }
    }

    /// Set right context (lookahead frames)
    pub fn set_right_context(&mut self, context: i32) -> &mut Self {
        self.inner.att_right_context = context;
        self
    }

    /// Get chunk size in mel frames
    pub fn chunk_mel_frames(&self) -> usize {
        unsafe { ffi::nemo_cache_config_get_chunk_mel_frames(&self.inner) }
    }

    /// Get chunk size in audio samples
    pub fn chunk_samples(&self) -> i32 {
        unsafe { ffi::nemo_cache_config_get_chunk_samples(&self.inner) }
    }

    /// Get latency in milliseconds
    pub fn latency_ms(&self) -> i32 {
        unsafe { ffi::nemo_cache_config_get_latency_ms(&self.inner) }
    }
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self::default()
    }
}

/// Backend device information
#[derive(Debug)]
pub struct BackendDevice {
    ptr: ffi::ggml_backend_dev_t,
}

impl BackendDevice {
    /// Get device name
    pub fn name(&self) -> &str {
        unsafe {
            let name_ptr = ffi::ggml_backend_dev_name(self.ptr);
            if name_ptr.is_null() {
                panic!("ggml_backend_dev_name returned NULL");
            }
            CStr::from_ptr(name_ptr)
                .to_str()
                .expect("ggml_backend_dev_name returned invalid UTF-8")
        }
    }
}

/// Load all available GGML backends from the specified path
#[cfg(feature = "ggml_backend_dl")]
pub fn load_backends_from_path(path: impl AsRef<Path>) {
    let path_str = path.as_ref().to_str().expect("path must be valid UTF-8");
    let path_c = CString::new(path_str).expect("path must not contain null bytes");
    unsafe {
        ffi::ggml_backend_load_all_from_path(path_c.as_ptr());
    }
}

/// Get number of available backend devices
pub fn backend_count() -> usize {
    unsafe { ffi::ggml_backend_dev_count() }
}

/// Get backend device by index
pub fn get_backend(index: usize) -> Option<BackendDevice> {
    unsafe {
        let ptr = ffi::ggml_backend_dev_get(index);
        if ptr.is_null() {
            None
        } else {
            Some(BackendDevice { ptr })
        }
    }
}

/// List all available backends
pub fn list_backends() -> Vec<BackendDevice> {
    let count = backend_count();
    (0..count).filter_map(get_backend).collect()
}

/// Main model context
pub struct Context {
    ptr: *mut ffi::nemo_context_ffi,
}

impl Context {
    /// Initialize model from GGUF file with optional backend selection
    ///
    /// # Arguments
    /// * `model_path` - Path to the GGUF model file
    /// * `backend` - Optional backend name (e.g., "CPU", "Vulkan"). None for auto-select.
    pub fn new(
        model_path: impl AsRef<Path>,
        backend: Option<&str>,
    ) -> Result<Self, InitializationError> {
        let path_str = model_path
            .as_ref()
            .to_str()
            .expect("model path must be valid UTF-8");
        let model_path_c = CString::new(path_str).expect("model path must not contain null bytes");
        let backend_c = backend.map(|s| CString::new(s).unwrap());

        let ptr = unsafe {
            ffi::c_nemo_init_with_backend(
                model_path_c.as_ptr(),
                backend_c.as_ref().map_or(ptr::null(), |s| s.as_ptr()),
            )
        };

        if ptr.is_null() {
            Err(InitializationError::InitializationFailed)
        } else {
            Ok(Self { ptr })
        }
    }

    /// Get the name of the backend being used
    pub fn backend_name(&self) -> &str {
        unsafe {
            let name_ptr = ffi::c_nemo_get_backend_name(self.ptr);
            if name_ptr.is_null() {
                panic!("c_nemo_get_backend_name returned NULL");
            }
            CStr::from_ptr(name_ptr)
                .to_str()
                .expect("c_nemo_get_backend_name returned invalid UTF-8")
        }
    }

    /// Create a streaming context
    pub fn create_stream(
        &mut self,
        config: Option<&CacheConfig>,
    ) -> Result<Stream, StreamInitializationError> {
        let config_ptr = config.map_or(ptr::null(), |c| &c.inner);

        let ptr = unsafe { ffi::c_nemo_stream_init(self.ptr, config_ptr) };

        if ptr.is_null() {
            Err(StreamInitializationError::StreamInitializationFailed)
        } else {
            Ok(Stream { ptr })
        }
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe {
            ffi::c_nemo_free(self.ptr);
        }
    }
}

unsafe impl Send for Context {}

/// Streaming transcription context
pub struct Stream {
    ptr: *mut ffi::nemo_stream_context_ffi,
}

impl Stream {
    /// Process audio chunk incrementally
    ///
    /// # Arguments
    /// * `audio` - PCM audio samples (16-bit signed, 16kHz, mono)
    ///
    /// Returns new transcription text (may be empty if no new tokens)
    pub fn process(&mut self, audio: &[i16]) -> String {
        let text_ptr = unsafe {
            ffi::c_nemo_stream_process_incremental(self.ptr, audio.as_ptr(), audio.len() as i32)
        };

        if text_ptr.is_null() {
            return String::new();
        }

        unsafe {
            let text = CStr::from_ptr(text_ptr).to_string_lossy().to_string();
            ffi::c_nemo_free_string(text_ptr);
            text
        }
    }

    /// Finalize streaming and flush remaining audio
    ///
    /// Returns final transcription text
    pub fn finalize(&mut self) -> String {
        let text_ptr = unsafe { ffi::c_nemo_stream_finalize(self.ptr) };

        if text_ptr.is_null() {
            return String::new();
        }

        unsafe {
            let text = CStr::from_ptr(text_ptr).to_string_lossy().to_string();
            ffi::c_nemo_free_string(text_ptr);
            text
        }
    }

    /// Get full accumulated transcript
    pub fn get_transcript(&self) -> String {
        let text_ptr = unsafe { ffi::c_nemo_stream_get_transcript(self.ptr) };

        if text_ptr.is_null() {
            return String::new();
        }

        unsafe {
            let text = CStr::from_ptr(text_ptr).to_string_lossy().to_string();
            ffi::c_nemo_free_string(text_ptr);
            text
        }
    }

    /// Reset streaming state (clear caches and transcript)
    pub fn reset(&mut self) {
        unsafe {
            ffi::c_nemo_stream_reset(self.ptr);
        }
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        unsafe {
            ffi::c_nemo_stream_free(self.ptr);
        }
    }
}

unsafe impl Send for Stream {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_config() {
        let config = CacheConfig::default();
        assert!(config.chunk_samples() > 0);
        assert!(config.latency_ms() > 0);
    }

    #[test]
    fn test_backend_list() {
        let count = backend_count();
        println!("Available backends: {}", count);

        for i in 0..count {
            if let Some(backend) = get_backend(i) {
                println!("  Backend {}: {}", i, backend.name());
            }
        }
    }
}

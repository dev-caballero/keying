//! ML background removal and closed-form alpha matting as a standalone Rust
//! library.
//!
//! - `bg`: inference engine for the `isnet-general-use` ONNX model via `ort`
//!   (onnxruntime); a faithful port of the rembg pipeline. Minimal API:
//!   `engine()` (one session cached per process) → `remove_background(model,
//!   img, smooth)`.
//! - `matting`: closed-form alpha matting (port of pymatting), used when
//!   `smooth=true` and usable on its own.
//!
//! Testable and reusable from an app, a CLI, or a web service. The model is
//! located through `bg::resolve_model_path()`
//! (`ISNET_MODEL_PATH` → the crate's `resources/models` → `~/.rembg`);
//! packaged consumers can resolve their own resources and
//! inject the path via `ISNET_MODEL_PATH` before that fallback.

pub mod bg;
pub mod matting;
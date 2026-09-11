//! Background-removal engine: `isnet-general-use` ONNX model via `ort`, as a
//! standalone Rust library.
//!
//! Faithful port of the rembg pipeline (sessions/dis_general_use.py + base.py):
//! - preprocess: RGB → exact resize to 1024×1024 (LANCZOS) → /255 →
//!   (x − 0.5) / 1.0 (DisSession's normalize: mean 0.5, std 1.0),
//!   CHW layout with batch 1.
//! - postprocess: output → channel 0 → min-max (like DisSession.predict) →
//!   ×255 → L mask → LANCZOS resize back to the original size.
//! - compositing: the image's alpha = the mask (rembg's naive_cutout: the
//!   binary output without smoothing is identical to rembg's).
//! - `smooth` = rembg's alpha_matting: closed-form matting via
//!   `matting::cutout` (1:1 port of pymatting, parity validated < 3/255).
//!
//! One ORT session per process. KeyingEngine's lock covers the whole
//! inference — in ort the session needs `&mut` (not thread-safe for
//! concurrent inference), so the lock serializes inference (like _BG_LOCK in
//! server.py).

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use image::{DynamicImage, ImageBuffer, Luma};
use ort::{inputs, session::Session, value::Tensor};
use parking_lot::Mutex;

use crate::matting;

const INPUT_SIZE: u32 = 1024;
const MEAN: f32 = 0.5; // matches dis_general_use.py
const STD: f32 = 1.0;

/// One engine per process: `engine()` returns the cached session.
static ENGINE: OnceLock<KeyingEngine> = OnceLock::new();

pub fn engine() -> &'static KeyingEngine {
    ENGINE.get_or_init(KeyingEngine::new)
}

/// Background-removal engine: cached ORT session + one-inference-at-a-time lock.
pub struct KeyingEngine(Mutex<Option<Session>>);

impl KeyingEngine {
    pub fn new() -> Self {
        Self(Mutex::new(None))
    }

    /// Removes the background: returns the RGBA image with alpha = the model
    /// mask. Holds the lock for the entire inference (session not concurrent).
    pub fn remove_background(
        &self,
        model: &Path,
        img: &DynamicImage,
        smooth: bool,
    ) -> Result<DynamicImage, String> {
        let mut guard = self.0.lock();
        if guard.is_none() {
            *guard = Some(load_session(model)?);
        }
        let session = guard.as_mut().expect("session just loaded");
        let mask = predict(session, img)?;
        let mut rgba = img.to_rgba8();

        // alpha = mask; with smooth, closed-form matting (port of pymatting —
        // see matting.rs) with parity to rembg's alpha_matting.
        if smooth {
            let rgb = DynamicImage::ImageRgba8(rgba.clone()).to_rgb8();
            return Ok(DynamicImage::ImageRgba8(matting::cutout(&rgb, mask.as_raw())));
        }
        for (px, m) in rgba.pixels_mut().zip(mask.pixels()) {
            px[3] = m[0];
        }
        Ok(DynamicImage::ImageRgba8(rgba))
    }
}

fn load_session(model: &Path) -> Result<Session, String> {
    Session::builder()
        .map_err(|e| e.to_string())?
        .commit_from_file(model)
        .map_err(|e| e.to_string())
}

/// DisSession.predict: L mask (0-255) at the original size.
fn predict(
    session: &mut Session,
    img: &DynamicImage,
) -> Result<ImageBuffer<Luma<u8>, Vec<u8>>, String> {
    let (w, h) = (img.width(), img.height());
    let resized = img
        .resize_exact(INPUT_SIZE, INPUT_SIZE, image::imageops::FilterType::Lanczos3)
        .to_rgb8();

    // CHW: 3 channel-first planes, normalized values.
    let n = (INPUT_SIZE * INPUT_SIZE) as usize;
    let mut data = vec![0f32; 3 * n];
    let mut i = 0usize;
    for px in resized.pixels() {
        data[i] = norm(px[0]);
        data[i + n] = norm(px[1]);
        data[i + 2 * n] = norm(px[2]);
        i += 1;
    }
    let arr = ndarray::Array4::from_shape_vec((1, 3, INPUT_SIZE as usize, INPUT_SIZE as usize), data)
        .map_err(|e| e.to_string())?;
    let value = Tensor::from_array(arr).map_err(|e| e.to_string())?;

    let outputs = session
        .run(inputs![value])
        .map_err(|e| e.to_string())?;
    let (shape, raw) = outputs[0]
        .try_extract_tensor::<f32>()
        .map_err(|e| e.to_string())?;

    // Output channel 0 (pred[:,0] in rembg); isnet's output is
    // (1, 1, H, W), so the flat data is already the mask.
    let raw = &raw[..n.min(raw.len())];
    debug_assert_eq!(shape.num_elements(), n.min(raw.len()));

    // min-max normalization (same as DisSession.predict).
    let mut mi = f32::INFINITY;
    let mut ma = f32::NEG_INFINITY;
    for &v in raw {
        mi = mi.min(v);
        ma = ma.max(v);
    }
    let range = (ma - mi).max(1e-6);

    let mut mask = vec![0u8; n];
    for (i, &v) in raw.iter().enumerate() {
        mask[i] = ((v - mi) / range * 255.0).clamp(0.0, 255.0) as u8;
    }
    let mask_img = ImageBuffer::from_raw(INPUT_SIZE, INPUT_SIZE, mask)
        .ok_or("invalid mask size")?;

    Ok(DynamicImage::ImageLuma8(mask_img)
        .resize_exact(w, h, image::imageops::FilterType::Lanczos3)
        .to_luma8())
}

fn norm(v: u8) -> f32 {
    (v as f32 / 255.0 - MEAN) / STD
}

/// Model resolution: ISNET_MODEL_PATH → the crate's resources/models →
/// ~/.rembg (deprecated Python runtime). A packaged consumer resolves its
/// own resources and injects ISNET_MODEL_PATH before this fallback.
pub fn resolve_model_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("ISNET_MODEL_PATH") {
        let b = PathBuf::from(p);
        if !b.to_string_lossy().is_empty() && b.is_file() {
            return Some(b);
        }
    }
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let resources = manifest.join("resources/models/isnet-general-use.onnx");
    if resources.is_file() {
        return Some(resources);
    }
    let home = std::env::var("HOME").ok()?;
    let rembg_default =
        Path::new(&home).join(".rembg/models/isnet-general-use/isnet-general-use.onnx");
    rembg_default.is_file().then_some(rembg_default)
}
//! Engine tests: with the model, an image with a green background and a red
//! subject comes out RGBA with a transparent background.
//!
//! Require the isnet-general-use model: ISNET_MODEL_PATH, resources/models
//! (scripts/fetch-model.sh) or ~/.rembg. Without the model the test skips —
//! a fresh clone must not break the suite.

use image::{DynamicImage, Rgba, RgbaImage};
use keying::bg::{engine, resolve_model_path};

/// Test image: a red square centered on a green background.
fn test_image(w: u32, h: u32) -> RgbaImage {
    let mut img = RgbaImage::from_pixel(w, h, Rgba([0, 180, 0, 255]));
    let (x0, x1) = (w / 4, 3 * w / 4);
    let (y0, y1) = (h / 4, 3 * h / 4);
    for y in y0..y1 {
        for x in x0..x1 {
            img.put_pixel(x, y, Rgba([200, 10, 10, 255]));
        }
    }
    img
}

#[test]
fn remove_background_returns_rgba_with_alpha() {
    let model = resolve_model_path();
    if model.is_none() {
        println!("no isnet-general-use model — skipping (scripts/fetch-model.sh)");
        return;
    }

    let img = DynamicImage::ImageRgba8(test_image(320, 240));
    let out = engine()
        .remove_background(model.as_deref().expect("model present"), &img, false)
        .expect("background removal ok");

    assert_eq!((out.width(), out.height()), (320, 240));

    let rgba = out.to_rgba8();
    let corner = rgba.get_pixel(5, 5)[3];
    let center = rgba.get_pixel(160, 120)[3];
    assert!(corner < 64, "background not transparent (alpha={corner})");
    assert!(center > 192, "subject not opaque (alpha={center})");
}
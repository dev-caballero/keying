# keying

ML background removal ([isnet-general-use](https://github.com/danielgatis/rembg) via ONNX Runtime) and closed-form alpha matting as a standalone Rust library — usable from any app, CLI, or web service.

```rust
use image::open;
use keying::bg::{engine, resolve_model_path};

let model = resolve_model_path()
    .ok_or("isnet-general-use model not found — run scripts/fetch-model.sh")?;
let img = open("input.png")?;
let out = engine().remove_background(&model, &img, false)?;
out.save("output.png")?;
```

## Features

- rembg-compatible background removal with the `isnet-general-use` ONNX model, executed through [`ort`](https://github.com/pykeio/ort) (onnxruntime).
- Optional smooth edges: closed-form alpha matting (1:1 port of [pymatting](https://github.com/pymatting/pymatting)) with foreground color decontamination — pass `smooth = true`.
- Any image format supported by the [`image`](https://github.com/image-rs/image) crate (PNG, JPEG, …).
- One cached ONNX session per process; safe for concurrent use (inference is serialized internally).
- Pure Rust, no Python runtime.
- Binary output (no smoothing) is pixel-identical to rembg's `naive_cutout`.

## Requirements

- Rust 1.77.2 or newer (edition 2021).
- The `isnet-general-use` ONNX model (~176 MB) — see [Model setup](#model-setup).

## Model setup

`bg::resolve_model_path()` looks for the model in this order:

1. `$ISNET_MODEL_PATH` — a path to the `.onnx` file.
2. `<crate>/resources/models/isnet-general-use.onnx` — where the bundled script downloads it.
3. `~/.rembg/models/isnet-general-use/isnet-general-use.onnx` — the legacy rembg location.

Download it with the bundled script:

```bash
./scripts/fetch-model.sh
```

or manually from the [rembg releases](https://github.com/danielgatis/rembg/releases/download/v0.0.0/isnet-general-use.onnx). The model is MIT-licensed.

Bundled applications should resolve the model from their own resources and set `ISNET_MODEL_PATH` before calling the API.

## Usage

Add the dependency:

```toml
[dependencies]
keying = "0.1"
```

The examples below also use the `image` crate directly (`cargo add image`).

### Basic background removal

```rust
use image::open;
use keying::bg::{engine, resolve_model_path};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = resolve_model_path()
        .ok_or("isnet-general-use model not found — run scripts/fetch-model.sh")?;
    let img = open("input.png")?;

    let out = engine().remove_background(&model, &img, false)?;
    out.save("output.png")?;
    Ok(())
}
```

`KeyingEngine::remove_background(model, img, smooth)` takes:

| Argument | Description |
|---|---|
| `model` | Path to the `isnet-general-use.onnx` file. |
| `img` | Any `DynamicImage` (from the `image` crate). |
| `smooth` | `true` applies closed-form alpha matting for anti-aliased edges. |

It returns an `Rgba8` image (`DynamicImage::ImageRgba8`) whose alpha channel is the predicted mask, or a `String` error. The ONNX session is loaded on first call to `engine()` and cached for the rest of the process.

### Smooth edges

```rust
let out = engine().remove_background(&model, &img, true)?;
```

`smooth = true` additionally solves a closed-form matting problem (sparse linear system + foreground color decontamination) to recover per-pixel alpha at object edges. It is significantly slower than the plain mask; it matches rembg's "alpha matting (smooth edges)" option.

### Custom model path

```rust
use std::path::Path;

let out = engine().remove_background(Path::new("/opt/models/isnet-general-use.onnx"), &img, false)?;
```

### Matting only

`keying::matting` is a standalone closed-form alpha matting implementation, usable without the ONNX model:

```rust
use keying::matting::cutout;

// rgb: RgbImage, mask: raw mask bytes (0-255 per pixel, same size as rgb)
let rgba: image::RgbaImage = cutout(&rgb, &mask);
```

## API

| Item | Description |
|---|---|
| `bg::engine()` | Returns the process-wide cached engine (one ORT session). |
| `bg::KeyingEngine::remove_background()` | Removes the background; returns the RGBA image. |
| `bg::resolve_model_path()` | Locates the model: `ISNET_MODEL_PATH` → crate `resources/models` → `~/.rembg`. |
| `matting::cutout()` | Closed-form alpha matting cutout from an RGB image and a mask. |

## How it works

- **`bg`** replicates rembg's pipeline exactly: RGB → resize to 1024×1024 (LANCZOS) → normalize (mean 0.5, std 1.0) → CHW tensor → ONNX inference → output channel 0 → min-max → ×255 → L mask → resize back to original size → alpha channel.
- **`matting`** is a 1:1 port of pymatting: eroded trimap → closed-form matting Laplacian (Levin 2007) → conjugate gradient (Jacobi preconditioner — same solution, fewer dependencies than ichol) → foreground decontamination. Parity with pymatting validated at < 3/255 per channel.

## Testing

```bash
cargo test
```

The matting unit tests run without any model. The engine integration test (in `tests/engine.rs`) requires the model — it skips when none is found, so a fresh clone never breaks the suite.

## License

MIT — see [LICENSE](LICENSE).
//! 1:1 port of pymatting's closed-form alpha matting, exactly as rembg uses
//! it in `alpha_matting_cutout` (the "Smooth edges" option of the deprecated
//! pipeline):
//!
//!   1. trimap: mask > 240 and < 10, eroded with a 10×10 window
//!      (border_value=1 for the background, as in rembg).
//!   2. closed-form matting laplacian (Levin 2007): 3×3 windows,
//!      ε = 1e-7 (pymatting's cf_laplacian).
//!   3. alpha = CG on L_U α = −R·m (pymatting uses ichol as preconditioner;
//!      here Jacobi — the solution is the same, only the speed changes),
//!      rtol = 1e-7, maxiter = 10000 (pymatting defaults).
//!   4. foreground color decontamination (pymatting's
//!      estimate_foreground_ml: per-pixel multiscale relaxation), which
//!      removes background-color fringe at the edges.
//!
//! Reference: pymatting (alpha/estimate_alpha_cf.py,
//! laplacian/cf_laplacian.py, foreground/estimate_foreground_ml.py) and
//! rembg/bg.py (alpha_matting_cutout).

use image::{RgbImage, Rgba, RgbaImage};

const EPS: f64 = 1e-7; // cf_laplacian
const FG_THRESHOLD: u8 = 240;
const BG_THRESHOLD: u8 = 10;
const ERODE_SIZE: usize = 10;
const CG_RTOL: f64 = 1e-7;
const CG_MAXITER: usize = 10000;

/// 3×3 windows (r=1): pairs within [−2, 2]² (25 slots per row).
const R: i64 = 1;
const EXTENT: usize = (2 * R + 1) as usize; // 3
const PAD: usize = (4 * R + 1) as usize; // 5, neighbor offsets
const AREA: f64 = (EXTENT * EXTENT) as f64; // 9

const FG_ML_REG: f64 = 1e-5;
const FG_ML_GRAD_W: f64 = 1.0;
const FG_ML_SMALL_SIZE: usize = 32;
const FG_ML_SMALL_ITERS: usize = 10;
const FG_ML_BIG_ITERS: usize = 2;

/// Full cutout (decontaminated foreground + alpha) with smoothing, a replica
/// of rembg's `alpha_matting_cutout`. `mask` is the model mask (0-255).
pub fn cutout(rgb: &RgbImage, mask: &[u8]) -> RgbaImage {
    let w = rgb.width() as usize;
    let h = rgb.height() as usize;
    debug_assert_eq!(mask.len(), w * h);

    let alpha = estimate_alpha(rgb, mask);
    let (fg, _bg) = estimate_foreground_ml(rgb, &alpha);

    let mut out = RgbaImage::new(w as u32, h as u32);
    for (i, px) in out.pixels_mut().enumerate() {
        *px = Rgba([
            (fg[i][0] * 255.0).clamp(0.0, 255.0) as u8,
            (fg[i][1] * 255.0).clamp(0.0, 255.0) as u8,
            (fg[i][2] * 255.0).clamp(0.0, 255.0) as u8,
            (alpha[i] * 255.0).clamp(0.0, 255.0) as u8,
        ]);
    }
    out
}

/// Steps 1-3: eroded trimap + laplacian + CG → alpha [0,1] per pixel.
fn estimate_alpha(rgb: &RgbImage, mask: &[u8]) -> Vec<f32> {
    let w = rgb.width() as usize;
    let h = rgb.height() as usize;
    let n = w * h;

    let is_fg_raw: Vec<bool> = mask.iter().map(|&m| m > FG_THRESHOLD).collect();
    let is_bg_raw: Vec<bool> = mask.iter().map(|&m| m < BG_THRESHOLD).collect();
    // scipy's binary_erosion(border_value): the background erodes with border=1.
    let is_fg = erode(&is_fg_raw, w, h, false);
    let is_bg = erode(&is_bg_raw, w, h, true);

    let mut trimap = vec![128.0 / 255.0; n]; // unknown (trimap 128/255)
    let mut is_known = vec![false; n];
    for i in 0..n {
        if is_fg[i] {
            trimap[i] = 1.0;
            is_known[i] = true;
        } else if is_bg[i] {
            trimap[i] = 0.0;
            is_known[i] = true;
        }
    }

    let img: Vec<f32> = rgb
        .as_raw()
        .iter()
        .map(|&v| v as f32 / 255.0)
        .collect();

    // L in row-sparse format: row i → vec![(col, val)].
    let laplacian: Vec<Vec<(usize, f64)>> = cf_laplacian(&img, w, h, &is_known);

    // Global indices → local ones.
    let mut u_to_local = vec![usize::MAX; n];
    let mut u_index: Vec<usize> = Vec::new();
    let mut k_index: Vec<usize> = Vec::new();
    for i in 0..n {
        if is_known[i] {
            k_index.push(i);
        } else {
            u_to_local[i] = u_index.len();
            u_index.push(i);
        }
    }
    let nu = u_index.len();

    // L_U (unknown×unknown) and b = −R·m (m = known fg).
    let mut lu_rows: Vec<Vec<(usize, f64)>> = vec![Vec::new(); nu];
    let mut b = vec![0f64; nu];
    for (li, &gi) in u_index.iter().enumerate() {
        for &(gj, val) in &laplacian[gi] {
            if is_known[gj] {
                if is_fg[gj] {
                    b[li] -= val; // −L[i,j]·1 (m=1 for fg, 0 for bg)
                }
            } else {
                lu_rows[li].push((u_to_local[gj], val));
            }
        }
    }

    let x = cg_jacobi(&lu_rows, &b);

    let mut alpha = vec![0f32; n];
    for i in 0..n {
        alpha[i] = if is_known[i] {
            trimap[i]
        } else {
            x[u_to_local[i]].clamp(0.0, 1.0) as f32
        };
    }
    alpha
}

/// scipy.ndimage.binary_erosion with a ones((10,10)) structure: origin (5,5),
/// window [y−5, y+4]×[x−5, x+4]; `border` fills outside the image.
fn erode(input: &[bool], w: usize, h: usize, border: bool) -> Vec<bool> {
    let origin = (ERODE_SIZE / 2) as i64; // 5 for a 10×10 window
    let mut out = vec![false; input.len()];
    for y in 0..h {
        for x in 0..w {
            let mut all = true;
            'win: for dy in -origin..origin {
                for dx in -origin..origin {
                    let (ny, nx) = (y as i64 + dy, x as i64 + dx);
                    let v = if ny < 0 || nx < 0 || ny >= h as i64 || nx >= w as i64 {
                        border
                    } else {
                        input[(ny as usize) * w + nx as usize]
                    };
                    if !v {
                        all = false;
                        break 'win;
                    }
                }
            }
            out[y * w + x] = all;
        }
    }
    out
}

/// pymatting's cf_laplacian (row i → vec![(col, val)]).
fn cf_laplacian(img: &[f32], w: usize, h: usize, is_known: &[bool]) -> Vec<Vec<(usize, f64)>> {
    let n = w * h;
    // Neighbors: slot k = dy*PAD + dx with dy,dx ∈ [−2, 2]; PAD=5.
    let mut slots: Vec<[usize; PAD * PAD]> = vec![[usize::MAX; PAD * PAD]; n];
    for i in 0..n {
        let x = (i % w) as i64;
        let y = (i / w) as i64;
        for (dy, dx) in offsets() {
            let (nx, ny) = (x + dx, y + dy);
            if nx >= 0 && ny >= 0 && (nx as usize) < w && (ny as usize) < h {
                let k = (dy + 2) as usize * PAD + (dx + 2) as usize;
                slots[i][k] = (ny as usize) * w + nx as usize;
            }
        }
    }

    // Dense accumulator (n, PAD, PAD) like pymatting.
    let mut values: Vec<f64> = vec![0.0; n * PAD * PAD];

    let mut c = [[[0f64; 3]; EXTENT]; EXTENT];
    for y in R..h as i64 - R {
        for x in R..w as i64 - R {
            // Skip fully known windows.
            let mut all_known = true;
            for dy in 0..EXTENT {
                for dx in 0..EXTENT {
                    let pi = ((y + dy as i64 - R) as usize) * w + (x + dx as i64 - R) as usize;
                    if !is_known[pi] {
                        all_known = false;
                        break;
                    }
                }
            }
            if all_known {
                continue;
            }

            // Channel-wise centered colors.
            let mut sums = [0f64; 3];
            for dy in 0..EXTENT {
                for dx in 0..EXTENT {
                    let pi = ((y + dy as i64 - R) as usize) * w + (x + dx as i64 - R) as usize;
                    for ch in 0..3 {
                        sums[ch] += img[pi * 3 + ch] as f64;
                    }
                }
            }
            for dy in 0..EXTENT {
                for dx in 0..EXTENT {
                    let pi = ((y + dy as i64 - R) as usize) * w + (x + dx as i64 - R) as usize;
                    for ch in 0..3 {
                        c[dy][dx][ch] = img[pi * 3 + ch] as f64 - sums[ch] / AREA;
                    }
                }
            }

            // Regularized covariance and inverse (explicit 3×3 formula).
            let mut a = [[0f64; 3]; 3];
            for dy in 0..EXTENT {
                for dx in 0..EXTENT {
                    for c1 in 0..3 {
                        for c2 in 0..3 {
                            a[c1][c2] += c[dy][dx][c1] * c[dy][dx][c2];
                        }
                    }
                }
            }
            for c in 0..3 {
                for d in 0..3 {
                    a[c][d] = a[c][d] / AREA + if c == d { EPS } else { 0.0 };
                }
            }
            let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
                - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
                + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
            // Watch the sign: cf_laplacian.py's formulas use a negated
            // determinant (det_py = −standard_det) with negated cofactor
            // numerators → its inverse is the true inverse. With the
            // standard det, negate inv.
            let inv = -1.0 / det;
            // Symmetric 3×3 inverse with the 6 unique entries (exact formulas
            // from cf_laplacian.py).
            let m00 = (a[1][2] * a[1][2] - a[1][1] * a[2][2]) * inv;
            let m01 = (a[0][1] * a[2][2] - a[0][2] * a[1][2]) * inv;
            let m02 = (a[0][2] * a[1][1] - a[0][1] * a[1][2]) * inv;
            let m11 = (a[0][2] * a[0][2] - a[0][0] * a[2][2]) * inv;
            let m12 = (a[0][0] * a[1][2] - a[0][1] * a[0][2]) * inv;
            let m22 = (a[0][1] * a[0][1] - a[0][0] * a[1][1]) * inv;

            // pairs ((yi,xi),(yj,xj)) within the 3×3 window.
            for dyi in 0..EXTENT {
                for dxi in 0..EXTENT {
                    let s = c[dyi][dxi][0];
                    let t = c[dyi][dxi][1];
                    let u = c[dyi][dxi][2];
                    let c0 = m00 * s + m01 * t + m02 * u;
                    let c1 = m01 * s + m11 * t + m12 * u;
                    let c2 = m02 * s + m12 * t + m22 * u;

                    for dyj in 0..EXTENT {
                        for dxj in 0..EXTENT {
                            let (xi, yi) = (x + dxi as i64 - R, y + dyi as i64 - R);
                            let (xj, yj) = (x + dxj as i64 - R, y + dyj as i64 - R);
                            let pi = yi as usize * w + xi as usize;
                            let pj = yj as usize * w + xj as usize;

                            let temp = c0 * c[dyj][dxj][0]
                                + c1 * c[dyj][dxj][1]
                                + c2 * c[dyj][dxj][2];
                            let value = if pi == pj { 1.0 } else { 0.0 } - (1.0 + temp) / AREA;

                            let offset_dy = (yj - yi + 2 * R) as usize;
                            let offset_dx = (xj - xi + 2 * R) as usize;
                            values[pi * PAD * PAD + offset_dy * PAD + offset_dx] += value;
                        }
                    }
                }
            }
        }
    }

    // Per-row CSR (values may accumulate to 0; kept as-is).
    let mut rows: Vec<Vec<(usize, f64)>> = Vec::with_capacity(n);
    for i in 0..n {
        let mut row: Vec<(usize, f64)> = Vec::with_capacity(PAD * PAD);
        for k in 0..PAD * PAD {
            let col = slots[i][k];
            if col != usize::MAX {
                row.push((col, values[i * PAD * PAD + k]));
            }
        }
        rows.push(row);
    }
    rows
}

fn offsets() -> impl Iterator<Item = (i64, i64)> {
    (-2i64..=2).flat_map(move |dy| (-2i64..=2).map(move |dx| (dy, dx)))
}

/// CG with a Jacobi preconditioner (pymatting uses ichol; the solution is
/// the same, only the iteration count changes). rtol = 1e-7, maxiter 10k.
fn cg_jacobi(rows: &[Vec<(usize, f64)>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut x = vec![0f64; n];
    let mut r = b.to_vec();
    let mut diag = vec![0f64; n];
    for (i, row) in rows.iter().enumerate() {
        for &(j, v) in row {
            if i == j {
                diag[i] = v;
            }
        }
    }
    let norm_b = r.iter().map(|v| v * v).sum::<f64>().sqrt();
    if norm_b == 0.0 {
        return x;
    }
    let mut p = vec![0f64; n];
    let mut rho = 0f64;
    for i in 0..n {
        let z = if diag[i].abs() > 1e-14 { r[i] / diag[i] } else { r[i] };
        p[i] = z;
        rho += r[i] * z;
    }
    for _ in 0..CG_MAXITER {
        let mut ap = vec![0f64; n];
        for (i, row) in rows.iter().enumerate() {
            let mut s = 0f64;
            for &(j, v) in row {
                s += v * p[j];
            }
            ap[i] = s;
        }
        let denom = p.iter().zip(&ap).map(|(p, a)| p * a).sum::<f64>();
        if denom.abs() < 1e-300 {
            break;
        }
        let alpha = rho / denom;
        let mut rn = vec![0f64; n];
        for i in 0..n {
            x[i] += alpha * p[i];
            rn[i] = r[i] - alpha * ap[i];
        }
        let norm_r = rn.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm_r <= CG_RTOL * norm_b {
            return x;
        }
        let mut rho_new = 0f64;
        for i in 0..n {
            let z = if diag[i].abs() > 1e-14 { rn[i] / diag[i] } else { rn[i] };
            rho_new += rn[i] * z;
        }
        let beta = rho_new / rho;
        for i in 0..n {
            p[i] = z_p(&rn, &p, &diag, beta, i);
        }
        r.copy_from_slice(&rn);
        rho = rho_new;
    }
    x
}

#[inline]
fn z_p(rn: &[f64], p: &[f64], diag: &[f64], beta: f64, i: usize) -> f64 {
    let z = if diag[i].abs() > 1e-14 { rn[i] / diag[i] } else { rn[i] };
    z + beta * p[i]
}

/// pymatting's estimate_foreground_ml: multiscale relaxation.
fn estimate_foreground_ml(rgb: &RgbImage, alpha: &[f32]) -> (Vec<[f32; 3]>, Vec<[f32; 3]>) {
    let w = rgb.width() as usize;
    let h = rgb.height() as usize;
    let depth = 3;
    let img: Vec<[f32; 3]> = rgb
        .as_raw()
        .chunks_exact(3)
        .map(|c| [c[0] as f32 / 255.0, c[1] as f32 / 255.0, c[2] as f32 / 255.0])
        .collect();

    // Global means of fg (alpha>0.9) and bg (alpha<0.1).
    let mut f_mean = [0f32; 3];
    let mut b_mean = [0f32; 3];
    let (mut f_count, mut b_count) = (0f32, 0f32);
    for i in 0..w * h {
        if alpha[i] > 0.9 {
            for c in 0..depth {
                f_mean[c] += img[i][c];
            }
            f_count += 1.0;
        }
        if alpha[i] < 0.1 {
            for c in 0..depth {
                b_mean[c] += img[i][c];
            }
            b_count += 1.0;
        }
    }
    let inv_f = 1.0 / (f_count + 1e-5);
    let inv_b = 1.0 / (b_count + 1e-5);
    for c in 0..depth {
        f_mean[c] *= inv_f;
        b_mean[c] *= inv_b;
    }

    let max_dim = w.max(h);
    let n_levels = (max_dim as f64).log2().ceil() as usize;
    let mut f_prev: Vec<[f32; 3]> = vec![f_mean];
    let mut b_prev: Vec<[f32; 3]> = vec![b_mean];
    let (mut h_prev, mut w_prev) = (1usize, 1usize);

    for level in 0..=n_levels {
        let scale = level as f64 / n_levels as f64;
        let cw = (w as f64).powf(scale).round() as usize;
        let ch = (h as f64).powf(scale).round() as usize;
        if cw == 0 || ch == 0 {
            continue;
        }
        let (cw, ch) = (cw.max(1), ch.max(1));
        let s_img = resize_nearest_3(&img, w, h, cw, ch);
        let s_alpha = resize_nearest_1(alpha, w, h, cw, ch);
        let mut f = resize_nearest_3(&f_prev, w_prev, h_prev, cw, ch);
        let mut b = resize_nearest_3(&b_prev, w_prev, h_prev, cw, ch);

        let n_iter = if cw <= FG_ML_SMALL_SIZE && ch <= FG_ML_SMALL_SIZE {
            FG_ML_SMALL_ITERS
        } else {
            FG_ML_BIG_ITERS
        };

        let dx = [-1i64, 1, 0, 0];
        let dy = [0i64, 0, -1, 1];
        for _ in 0..n_iter {
            for y in 0..ch {
                for x in 0..cw {
                    let i = y * cw + x;
                    let a0 = s_alpha[i] as f64;
                    let a1 = 1.0 - a0;
                    let mut a00 = a0 * a0;
                    let a01 = a0 * a1;
                    let mut a11 = a1 * a1;
                    let mut b0 = [
                        a0 * s_img[i][0] as f64,
                        a0 * s_img[i][1] as f64,
                        a0 * s_img[i][2] as f64,
                    ];
                    let mut b1 = [
                        a1 * s_img[i][0] as f64,
                        a1 * s_img[i][1] as f64,
                        a1 * s_img[i][2] as f64,
                    ];
                    for d in 0..4 {
                        let x2 = (x as i64 + dx[d]).clamp(0, cw as i64 - 1) as usize;
                        let y2 = (y as i64 + dy[d]).clamp(0, ch as i64 - 1) as usize;
                        let gradient = (a0 - s_alpha[y2 * cw + x2] as f64).abs();
                        let da = FG_ML_REG + FG_ML_GRAD_W * gradient;
                        a00 += da;
                        a11 += da;
                        for c in 0..3 {
                            b0[c] += da * f[y2 * cw + x2][c] as f64;
                            b1[c] += da * b[y2 * cw + x2][c] as f64;
                        }
                    }
                    let det = a00 * a11 - a01 * a01;
                    let inv = 1.0 / det;
                    let (b00, b01, b11) = (inv * a11, inv * -a01, inv * a00);
                    for c in 0..3 {
                        let fc = (b00 * b0[c] + b01 * b1[c]).clamp(0.0, 1.0) as f32;
                        let bc = (b01 * b0[c] + b11 * b1[c]).clamp(0.0, 1.0) as f32;
                        f[i][c] = fc;
                        b[i][c] = bc;
                    }
                }
            }
        }

        f_prev = f;
        b_prev = b;
        h_prev = ch;
        w_prev = cw;
    }

    // Ensure full final resolution (the last level is already, through
    // rounding, but take the max in case it misses).
    (f_prev, b_prev)
}

/// Multi-channel nearest-neighbor resize (x_src = x_dst·w_src//w_dst).
fn resize_nearest_3(src: &[[f32; 3]], w_src: usize, h_src: usize, w_dst: usize, h_dst: usize) -> Vec<[f32; 3]> {
    let mut dst = vec![[0f32; 3]; w_dst * h_dst];
    for y in 0..h_dst {
        let ys = (y * h_src / h_dst).min(h_src - 1);
        for x in 0..w_dst {
            let xs = (x * w_src / w_dst).min(w_src - 1);
            dst[y * w_dst + x] = src[ys * w_src + xs];
        }
    }
    dst
}

/// Single-channel nearest-neighbor resize (alpha).
fn resize_nearest_1(src: &[f32], w_src: usize, h_src: usize, w_dst: usize, h_dst: usize) -> Vec<f32> {
    let mut dst = vec![0f32; w_dst * h_dst];
    for y in 0..h_dst {
        let ys = (y * h_src / h_dst).min(h_src - 1);
        for x in 0..w_dst {
            let xs = (x * w_src / w_dst).min(w_src - 1);
            dst[y * w_dst + x] = src[ys * w_src + xs];
        }
    }
    dst
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, RgbImage};

    #[test]
    fn cutout_keeps_confident_regions_and_softens_band() {
        let (w, h) = (48u32, 48u32);
        let mut rgb = RgbImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let c = if x < 30 { [200u8, 10, 10] } else { [10, 10, 200] };
                rgb.put_pixel(x, y, Rgb(c));
            }
        }
        // Mask: opaque interior, transparent exterior, 128 band (24..30).
        let mut mask = vec![0u8; (w * h) as usize];
        for y in 0..h {
            for x in 0..w {
                mask[(y * w + x) as usize] = if x < 24 {
                    255
                } else if x > 30 {
                    0
                } else {
                    128
                };
            }
        }
        let out = cutout(&rgb, &mask);
        let alpha = |x: u32, y: u32| out.get_pixel(x, y)[3];
        // Interior/exterior pinned (eroded trimap).
        assert_eq!(alpha(10, 24), 255, "interior opaque");
        assert_eq!(alpha(40, 24), 0, "exterior transparent");
        // The uncertainty band yields smooth, in-between values (not hard 128).
        let band: Vec<u8> = (26..30).map(|x| alpha(x, 24)).collect();
        assert!(
            band.iter().any(|&v| v > 0 && v < 255),
            "band not smoothed: {band:?}"
        );
    }
}
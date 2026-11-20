//! Dependency-free 8-bit RGB PNG writer and tiny raster canvas, so the
//! shared timeline can be inspected on machines without an SVG renderer.

use std::fs;
use std::io;
use std::path::Path;

fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &b in data {
        crc ^= b as u32;
        for _ in 0..8 {
            crc = if crc & 1 != 0 { (crc >> 1) ^ 0xEDB8_8320 } else { crc >> 1 };
        }
    }
    !crc
}

fn adler32(data: &[u8]) -> u32 {
    let (mut a, mut b) = (1u32, 0u32);
    for &x in data {
        a = (a + x as u32) % 65521;
        b = (b + a) % 65521;
    }
    (b << 16) | a
}

fn chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    out.extend_from_slice(kind);
    out.extend_from_slice(data);
    let mut crc_in = Vec::with_capacity(4 + data.len());
    crc_in.extend_from_slice(kind);
    crc_in.extend_from_slice(data);
    out.extend_from_slice(&crc32(&crc_in).to_be_bytes());
}

pub fn write_png(path: &Path, width: u32, height: u32, rgb: &[u8]) -> io::Result<()> {
    assert_eq!(rgb.len(), (width * height * 3) as usize);
    // Raw scanlines, filter byte 0 per row.
    let stride = (width * 3) as usize;
    let mut raw = Vec::with_capacity((stride + 1) * height as usize);
    for row in 0..height as usize {
        raw.push(0);
        raw.extend_from_slice(&rgb[row * stride..(row + 1) * stride]);
    }
    // zlib stream of stored (uncompressed) deflate blocks.
    let mut z = vec![0x78u8, 0x01];
    let mut pos = 0;
    while pos < raw.len() {
        let n = (raw.len() - pos).min(65535);
        let last = pos + n == raw.len();
        z.push(if last { 1 } else { 0 });
        z.extend_from_slice(&(n as u16).to_le_bytes());
        z.extend_from_slice(&(!(n as u16)).to_le_bytes());
        z.extend_from_slice(&raw[pos..pos + n]);
        pos += n;
    }
    z.extend_from_slice(&adler32(&raw).to_be_bytes());

    let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&width.to_be_bytes());
    ihdr.extend_from_slice(&height.to_be_bytes());
    ihdr.extend_from_slice(&[8, 2, 0, 0, 0]); // 8-bit RGB
    chunk(&mut png, b"IHDR", &ihdr);
    chunk(&mut png, b"IDAT", &z);
    chunk(&mut png, b"IEND", &[]);
    fs::write(path, png)
}

pub type Color = (u8, u8, u8);

pub struct Canvas {
    pub w: usize,
    pub h: usize,
    px: Vec<u8>,
}

impl Canvas {
    pub fn new(w: usize, h: usize, bg: Color) -> Self {
        let px = vec![0u8; w * h * 3];
        let mut c = Canvas { w, h, px };
        c.fill(bg);
        c
    }

    pub fn fill(&mut self, c: Color) {
        for p in self.px.chunks_exact_mut(3) {
            p.copy_from_slice(&[c.0, c.1, c.2]);
        }
    }

    pub fn set(&mut self, x: i32, y: i32, c: Color) {
        if x >= 0 && y >= 0 && (x as usize) < self.w && (y as usize) < self.h {
            let i = (y as usize * self.w + x as usize) * 3;
            self.px[i..i + 3].copy_from_slice(&[c.0, c.1, c.2]);
        }
    }

    pub fn blend(&mut self, x: i32, y: i32, c: Color, alpha: f32) {
        if x >= 0 && y >= 0 && (x as usize) < self.w && (y as usize) < self.h {
            let i = (y as usize * self.w + x as usize) * 3;
            for (k, ch) in [c.0, c.1, c.2].iter().enumerate() {
                let old = self.px[i + k] as f32;
                self.px[i + k] = (old * (1.0 - alpha) + *ch as f32 * alpha) as u8;
            }
        }
    }

    pub fn rect(&mut self, x0: i32, y0: i32, x1: i32, y1: i32, c: Color) {
        for y in y0..y1 {
            for x in x0..x1 {
                self.set(x, y, c);
            }
        }
    }

    pub fn rect_blend(&mut self, x0: i32, y0: i32, x1: i32, y1: i32, c: Color, a: f32) {
        for y in y0..y1 {
            for x in x0..x1 {
                self.blend(x, y, c, a);
            }
        }
    }

    pub fn line(&mut self, x0: f32, y0: f32, x1: f32, y1: f32, c: Color) {
        let steps = ((x1 - x0).abs().max((y1 - y0).abs()) as i32).max(1);
        for i in 0..=steps {
            let t = i as f32 / steps as f32;
            self.set(
                (x0 + (x1 - x0) * t).round() as i32,
                (y0 + (y1 - y0) * t).round() as i32,
                c,
            );
        }
    }

    pub fn polyline(&mut self, pts: &[(f32, f32)], c: Color) {
        for w in pts.windows(2) {
            self.line(w[0].0, w[0].1, w[1].0, w[1].1, c);
        }
    }

    pub fn write(&self, path: &Path) -> io::Result<()> {
        write_png(path, self.w as u32, self.h as u32, &self.px)
    }
}

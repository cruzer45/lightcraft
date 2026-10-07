//! Canon CRX codec (the compressed sensor data inside CR3 files).
//!
//! **Provenance — reference implementation, not clean-room.** This module was written from
//! `CR3-FORMAT.md` ("Canon CR3 / CRX Raw Format — Decoder Guide", Maurice Rogers, 2026-10-06, MIT), a
//! pseudocode-level guide whose author derived it from a decoder that used LibRaw's `crx.cpp` as its reference.
//! LightCraft's clean-room rule (CLAUDE.md) does not allow LibRaw-derived work in mainline: this code lives on the
//! `feat/cr3-crx` branch as a reference only and must not be merged without the maintainers' explicit approval of
//! that provenance. The container layout (ISO-BMFF, `CMP1`, `IAD1`) also follows the public `lclevy/canon_cr3`
//! format notes and was checked black-box on EOS R6 Mark III files.
//!
//! Scope (what the guide marks verified on real files): `CMP1` version `0x0200`, one tile, 1–3 wavelet levels with a
//! per-tile QP map, extended `FF11`/`FF12`/`FF13` headers, encoding types 0 and 3, progressive (LL) and
//! non-progressive (other subbands) band modes. Lossless planes, the rounded progressive mode, legacy `FF03` C-RAW
//! quantisation and multi-tile images return [`RawError::Unsupported`].
//!
//! Pipeline: per plane, decode the QP map → step tables; entropy-decode each subband (adaptive Rice + runs);
//! dequantise; inverse LeGall 5/3 wavelet. Then the four half-resolution planes become a Bayer mosaic.

use crate::{RawError, Result};
use rayon::prelude::*;

/// `CMP1` image header (the payload after the box header).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Cmp1 {
    pub version: u16,
    pub width: usize,
    pub height: usize,
    pub tile_width: usize,
    pub tile_height: usize,
    pub bits: u32,
    pub planes: usize,
    pub cfa: u8,
    pub encoding: u8,
    pub levels: usize,
    pub header_size: usize,
}

fn be16(b: &[u8], at: usize) -> Option<u16> {
    b.get(at..at.checked_add(2)?).map(|s| u16::from_be_bytes([s[0], s[1]]))
}
fn be32(b: &[u8], at: usize) -> Option<u32> {
    b.get(at..at.checked_add(4)?).map(|s| u32::from_be_bytes([s[0], s[1], s[2], s[3]]))
}
fn corrupt(what: &str) -> RawError {
    RawError::Corrupt(format!("CR3: {what}"))
}

impl Cmp1 {
    pub fn parse(p: &[u8]) -> Result<Cmp1> {
        let u = |at| be32(p, at).map(|v| v as usize).ok_or_else(|| corrupt("short CMP1"));
        let b = |at: usize| p.get(at).copied().ok_or_else(|| corrupt("short CMP1"));
        let c = Cmp1 {
            version: be16(p, 4).ok_or_else(|| corrupt("short CMP1"))?,
            width: u(8)?,
            height: u(12)?,
            tile_width: u(16)?,
            tile_height: u(20)?,
            bits: b(24)? as u32,
            planes: (b(25)? >> 4) as usize,
            cfa: b(25)? & 15,
            encoding: b(26)? >> 4,
            levels: (b(26)? & 15) as usize,
            header_size: u(28)?,
        };
        if c.width == 0 || c.height == 0 || c.tile_width == 0 || c.tile_height == 0 || !(8..=16).contains(&c.bits) {
            return Err(corrupt("implausible CMP1 geometry"));
        }
        Ok(c)
    }
    fn plane_dims(&self) -> (usize, usize) {
        if self.planes == 4 { (self.tile_width / 2, self.tile_height / 2) } else { (self.tile_width, self.tile_height) }
    }
}

/// One subband's entropy payload (absolute file range) and quantiser.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Subband {
    offset: usize,
    data_size: usize,
    q_base: u32,
    q_mult: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Plane {
    supports_partial: bool,
    rounded_bits: u8,
    subbands: Vec<Subband>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Tile {
    qp: Option<(usize, usize)>,
    planes: Vec<Plane>,
}

/// Parse the nested tile / plane / subband headers at the start of the raw sample (`sample` = absolute offset).
fn parse_headers(file: &[u8], sample: usize, c: &Cmp1) -> Result<Vec<Tile>> {
    let hdr_end = sample.checked_add(c.header_size).filter(|&e| e <= file.len()).ok_or_else(|| corrupt("CRX header outside file"))?;
    let hdr = file.get(sample..hdr_end).ok_or_else(|| corrupt("CRX header outside file"))?;
    let n_sub = 3 * c.levels + 1;
    let n_tiles = c.width.div_ceil(c.tile_width).saturating_mul(c.height.div_ceil(c.tile_height));
    if n_tiles == 0 || n_tiles > 64 || c.planes == 0 || c.planes > 4 {
        return Err(corrupt("implausible tile / plane count"));
    }
    let mut at = 0usize;
    let mut tile_off = hdr_end;
    let mut tiles = Vec::new();
    for t in 0..n_tiles {
        let (marker, size) = (be16(hdr, at).ok_or_else(|| corrupt("truncated tile header"))?, be16(hdr, at + 2).unwrap_or(0));
        if !matches!(marker, 0xff01 | 0xff11) || !matches!(size, 8 | 16) || (marker == 0xff01 && size != 8) {
            return Err(corrupt("bad tile header"));
        }
        let tile_size = be32(hdr, at + 4).ok_or_else(|| corrupt("truncated tile header"))? as usize;
        if be16(hdr, at + 8) != Some(t as u16) {
            return Err(corrupt("tile index out of order"));
        }
        let (mut qp_size, mut extra) = (0usize, 0usize);
        let has_qp = size == 16;
        if has_qp {
            qp_size = be32(hdr, at + 12).ok_or_else(|| corrupt("truncated tile header"))? as usize;
            extra = be16(hdr, at + 16).ok_or_else(|| corrupt("truncated tile header"))? as usize;
        }
        at += 4 + size as usize;
        let mut plane_off = tile_off.checked_add(qp_size).and_then(|v| v.checked_add(extra)).ok_or_else(|| corrupt("tile offset overflow"))?;
        let mut planes = Vec::new();
        for p in 0..c.planes {
            if be16(hdr, at)
                != Some(match marker {
                    0xff01 => 0xff02,
                    _ => 0xff12,
                })
                || be16(hdr, at + 2) != Some(8)
            {
                return Err(corrupt("bad plane header"));
            }
            let plane_size = be32(hdr, at + 4).ok_or_else(|| corrupt("truncated plane header"))? as usize;
            let flags = hdr.get(at + 8).copied().ok_or_else(|| corrupt("truncated plane header"))?;
            if (flags >> 4) as usize != p {
                return Err(corrupt("plane index out of order"));
            }
            at += 12;
            let mut sub_off = plane_off;
            let mut subbands = Vec::new();
            for s in 0..n_sub {
                let m = be16(hdr, at).ok_or_else(|| corrupt("truncated subband header"))?;
                let sb = match m {
                    0xff13 => {
                        let stored = be32(hdr, at + 4).ok_or_else(|| corrupt("truncated subband header"))? as usize;
                        let idx = be16(hdr, at + 8).unwrap_or(0xffff);
                        if (idx >> 12) as usize != s || idx & 0x0fff != 0 {
                            return Err(corrupt("subband index out of order"));
                        }
                        let mult = be16(hdr, at + 10).unwrap_or(0) as u32;
                        let base = be32(hdr, at + 12).unwrap_or(0);
                        let extra = be16(hdr, at + 16).unwrap_or(0) as usize;
                        at += 20;
                        (
                            stored,
                            Subband {
                                offset: sub_off,
                                data_size: stored.checked_sub(extra).ok_or_else(|| corrupt("subband sizes"))?,
                                q_base: base,
                                q_mult: mult,
                            },
                        )
                    }
                    0xff03 => return Err(RawError::Unsupported("CR3 with legacy (FF03) CRX subbands".into())),
                    _ => return Err(corrupt("bad subband header")),
                };
                let (stored, sb) = sb;
                if sb.offset.checked_add(sb.data_size).is_none_or(|e| e > file.len()) {
                    return Err(corrupt("subband data outside file"));
                }
                subbands.push(sb);
                sub_off = sub_off.checked_add(stored).ok_or_else(|| corrupt("subband offset overflow"))?;
            }
            planes.push(Plane { supports_partial: flags & 8 != 0, rounded_bits: (flags >> 1) & 3, subbands });
            plane_off = plane_off.checked_add(plane_size).ok_or_else(|| corrupt("plane offset overflow"))?;
        }
        let qp = has_qp.then_some((tile_off, qp_size));
        if let Some((o, l)) = qp
            && o.checked_add(l).is_none_or(|e| e > file.len())
        {
            return Err(corrupt("QP data outside file"));
        }
        tiles.push(Tile { qp, planes });
        tile_off = tile_off.checked_add(tile_size).ok_or_else(|| corrupt("tile offset overflow"))?;
    }
    Ok(tiles)
}

// ---- bitstream primitives (guide §7)

struct Bits<'a> {
    data: &'a [u8],
    pos: usize,
    cache: u64,
    n: u32,
}

impl<'a> Bits<'a> {
    fn new(data: &'a [u8]) -> Self {
        Bits { data, pos: 0, cache: 0, n: 0 }
    }
    fn refill(&mut self) {
        while self.n <= 56 {
            let byte = self.data.get(self.pos).copied().unwrap_or(0);
            self.cache |= (byte as u64) << (56 - self.n);
            self.pos += 1;
            self.n += 8;
        }
    }
    fn check(&self) -> Result<()> {
        // reading a few zero bytes past the payload is tolerated (padding); far past it is corruption
        if self.pos > self.data.len() + 16 { Err(corrupt("CRX bitstream overrun")) } else { Ok(()) }
    }
    fn bits(&mut self, k: u32) -> Result<u32> {
        if k == 0 {
            return Ok(0);
        }
        self.refill();
        let v = (self.cache >> (64 - k)) as u32;
        self.cache <<= k;
        self.n -= k;
        self.check()?;
        Ok(v)
    }
    /// Zeros up to the next `1` (consumed).
    fn zeros(&mut self) -> Result<u32> {
        let mut q = 0u32;
        loop {
            self.refill();
            let lz = self.cache.leading_zeros().min(self.n);
            if lz < self.n {
                q += lz;
                self.cache <<= lz + 1;
                self.n -= lz + 1;
                return Ok(q);
            }
            q += self.n;
            self.cache = 0;
            self.n = 0;
            self.check()?;
            if q > 1_000_000 {
                return Err(corrupt("CRX unary run too long"));
            }
        }
    }
    fn rice(&mut self, k: u32) -> Result<u32> {
        let q = self.zeros()?;
        if q >= 41 {
            return self.bits(21);
        }
        Ok((q << k) | self.bits(k)?)
    }
}

fn signed(c: u32) -> i32 {
    ((c >> 1) as i32) ^ -((c & 1) as i32)
}

fn adapt_k(k: u32, code: u32, max: u32) -> u32 {
    let mut n = k as i32;
    if code < ((1u32 << k) >> 1) {
        n -= 1;
    }
    if (code >> k) > 2 {
        n += 1;
    }
    if (code >> k) > 5 {
        n += 1;
    }
    let n = n.max(0) as u32;
    if max != 0 { n.min(max) } else { n.min(31) }
}

const JS: [u32; 32] =
    [1, 1, 1, 1, 2, 2, 2, 2, 4, 4, 4, 4, 8, 8, 8, 8, 16, 16, 32, 32, 64, 64, 128, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768];
const J: [u32; 32] = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 9, 10, 11, 12, 13, 14, 15];

/// Entropy state of one band: Rice parameter and run-length state.
struct Coder<'a> {
    bits: Bits<'a>,
    k: u32,
    s: usize,
}

impl Coder<'_> {
    fn run(&mut self, remaining: usize) -> Result<usize> {
        if remaining == 0 || self.bits.bits(1)? == 0 {
            return Ok(0);
        }
        let mut n = 1usize;
        while self.bits.bits(1)? == 1 {
            n += JS[self.s] as usize;
            if n > remaining {
                return Ok(remaining);
            }
            if self.s < 31 {
                self.s += 1;
            }
            if n == remaining {
                return Ok(n);
            }
        }
        if n < remaining {
            n += self.bits.bits(J[self.s])? as usize;
            if self.s > 0 {
                self.s -= 1;
            }
        }
        if n > remaining {
            return Err(corrupt("CRX run past the end of a line"));
        }
        Ok(n)
    }
}

fn median(left: i32, top: i32, top_left: i32) -> i32 {
    let d = top.wrapping_sub(top_left);
    let idx = if (top_left < left) ^ (d < 0) { 2 } else { 0 } + if (left < top) ^ (d < 0) { 1 } else { 0 };
    match idx {
        0 | 1 => left.wrapping_add(d),
        2 => left,
        _ => top,
    }
}

// ---- band decoding (guide §8)

/// Decode a `width × height` band. `progressive`: the plane's `supportsPartial` (LL only).
fn decode_band(data: &[u8], width: usize, height: usize, progressive: bool) -> Result<Vec<i32>> {
    let mut out = vec![0i32; width.checked_mul(height).ok_or_else(|| corrupt("band size"))?];
    if width == 0 || height == 0 {
        return Ok(out);
    }
    let mut c = Coder { bits: Bits::new(data), k: 0, s: 0 };
    let mut prev = vec![0i32; width + 2];
    let mut cur = vec![0i32; width + 2];
    let mut k_hist = vec![0u32; width + 2];
    for y in 0..height {
        if progressive {
            if y == 0 { progressive_top(&mut c, &mut cur, width)? } else { progressive_line(&mut c, &prev, &mut cur, width)? }
        } else if y == 0 {
            np_top(&mut c, &mut prev, &mut cur, &mut k_hist, width)?
        } else {
            np_line(&mut c, &prev, &mut cur, &mut k_hist, width)?
        }
        if let (Some(dst), Some(src)) = (out.get_mut(y * width..(y + 1) * width), cur.get(1..=width)) {
            dst.copy_from_slice(src);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    Ok(out)
}

fn progressive_top(c: &mut Coder<'_>, cur: &mut [i32], width: usize) -> Result<()> {
    cur[0] = 0;
    let (mut x, mut rem) = (0usize, width);
    while rem > 1 {
        if cur[x] != 0 {
            cur[x + 1] = cur[x];
        } else {
            let run = c.run(rem)?;
            rem -= run;
            for _ in 0..run {
                cur[x + 1] = cur[x];
                x += 1;
            }
            if rem == 0 {
                break;
            }
            cur[x + 1] = 0;
        }
        let code = c.bits.rice(c.k)?;
        cur[x + 1] = cur[x + 1].wrapping_add(signed(code));
        c.k = adapt_k(c.k, code, 15);
        x += 1;
        rem -= 1;
    }
    if rem == 1 {
        cur[x + 1] = cur[x];
        let code = c.bits.rice(c.k)?;
        cur[x + 1] = cur[x + 1].wrapping_add(signed(code));
        c.k = adapt_k(c.k, code, 15);
        x += 1;
    }
    cur[x + 1] = cur[x].wrapping_add(1);
    Ok(())
}

fn progressive_line(c: &mut Coder<'_>, prev: &[i32], cur: &mut [i32], width: usize) -> Result<()> {
    cur[0] = prev[1];
    let (mut x, mut rem) = (0usize, width);
    let symbol = |c: &mut Coder<'_>, cur: &mut [i32], x: usize, med: bool, not_eol: bool| -> Result<()> {
        let p = if med { median(cur[x], prev[x + 1], prev[x]) } else { prev[x + 1] };
        let code = c.bits.rice(c.k)?;
        cur[x + 1] = p.wrapping_add(signed(code));
        let est = if not_eol {
            ((code as u64 + ((prev[x + 2] as i64 - prev[x + 1] as i64).unsigned_abs() << 1)) >> 1).min(u32::MAX as u64) as u32
        } else {
            code
        };
        c.k = adapt_k(c.k, est, 15);
        Ok(())
    };
    while rem > 1 {
        if cur[x] != prev[x + 1] || cur[x] != prev[x + 2] {
            symbol(c, cur, x, true, true)?;
            x += 1;
            rem -= 1;
        } else {
            let run = c.run(rem)?;
            rem -= run;
            for _ in 0..run {
                cur[x + 1] = cur[x];
                x += 1;
            }
            if rem > 0 {
                symbol(c, cur, x, false, rem > 1)?;
                x += 1;
                rem -= 1;
            }
        }
    }
    if rem == 1 {
        symbol(c, cur, x, true, false)?;
        x += 1;
    }
    cur[x + 1] = cur[x].wrapping_add(1);
    Ok(())
}

fn adapt_np(k: u32, code: u32, k_next: u32) -> u32 {
    let k = adapt_k(k, code, 0);
    if k_next as i64 - k as i64 <= 1 { k.min(15) } else { k + 1 }
}

fn np_top(c: &mut Coder<'_>, prev: &mut [i32], cur: &mut [i32], k_hist: &mut [u32], width: usize) -> Result<()> {
    prev[0] = 0;
    cur[0] = 0;
    let (mut x, mut rem) = (0usize, width);
    while rem > 1 {
        if cur[x] != 0 {
            let code = c.bits.rice(c.k)?;
            cur[x + 1] = signed(code);
            c.k = adapt_k(c.k, code, 15);
        } else {
            let run = c.run(rem)?;
            rem -= run;
            for _ in 0..run {
                k_hist[x] = 0;
                cur[x + 1] = 0;
                x += 1;
            }
            if rem == 0 {
                break;
            }
            let code = c.bits.rice(c.k)?;
            cur[x + 1] = signed(code.saturating_add(1));
            c.k = adapt_k(c.k, code, 15);
        }
        k_hist[x] = c.k;
        x += 1;
        rem -= 1;
    }
    if rem == 1 {
        let code = c.bits.rice(c.k)?;
        cur[x + 1] = signed(code);
        c.k = adapt_k(c.k, code, 15);
        k_hist[x] = c.k;
        x += 1;
    }
    cur[x + 1] = 0;
    Ok(())
}

fn np_line(c: &mut Coder<'_>, prev: &[i32], cur: &mut [i32], k_hist: &mut [u32], width: usize) -> Result<()> {
    let mut i = 0usize;
    while i + 1 < width {
        if (prev[i + 2] | prev[i + 1] | cur[i]) != 0 {
            let code = c.bits.rice(c.k)?;
            cur[i + 1] = signed(code);
            c.k = adapt_np(c.k, code, k_hist[i + 1]);
        } else {
            let run = c.run(width - i)?;
            if run > 0 {
                cur[i + 1..=i + run].fill(0);
                k_hist[i..i + run].fill(0);
                i += run;
            }
            if i + 1 >= width {
                if i + 1 == width {
                    let code = c.bits.rice(c.k)?;
                    cur[i + 1] = signed(code.saturating_add(1));
                    c.k = adapt_k(c.k, code, 15);
                    k_hist[i] = c.k;
                }
                i += 1;
                continue;
            }
            let code = c.bits.rice(c.k)?;
            cur[i + 1] = signed(code.saturating_add(1));
            c.k = adapt_np(c.k, code, k_hist[i + 1]);
        }
        k_hist[i] = c.k;
        i += 1;
    }
    if i + 1 == width {
        let code = c.bits.rice(c.k)?;
        cur[i + 1] = signed(code);
        c.k = adapt_k(c.k, code, 15);
        k_hist[i] = c.k;
    }
    cur[width + 1] = 0;
    Ok(())
}

// ---- quantisation (guide §9)

fn decode_qp(data: &[u8], plane_w: usize, plane_h: usize) -> Result<(Vec<i32>, usize, usize)> {
    let (cols, rows) = (plane_w.div_ceil(8), plane_h.div_ceil(2));
    let mut b = Bits::new(data);
    let mut k = 0u32;
    let mut prev = vec![0i32; cols + 2];
    let mut cur = vec![0i32; cols + 2];
    let mut qp = vec![0i32; cols * rows];
    for y in 0..rows {
        cur[0] = if y == 0 { 0 } else { prev[1] };
        let mut dh = if y == 0 { 0 } else { prev[1].wrapping_sub(prev[0]) };
        for x in 0..cols {
            let p = if y == 0 {
                cur[x]
            } else {
                let (left, top, dv) = (cur[x], prev[x + 1], prev[x].wrapping_sub(cur[x]));
                let idx = if (dv < 0) ^ (dh < 0) { 2 } else { 0 } + if (left < top) ^ (dh < 0) { 1 } else { 0 };
                match idx {
                    0 | 1 => left.wrapping_add(dh),
                    2 => left,
                    _ => top,
                }
            };
            let q = b.zeros()?;
            let code = if q >= 23 { b.bits(8)? } else { (q << k) | b.bits(k)? };
            cur[x + 1] = p.wrapping_add(signed(code));
            if y != 0 && x + 1 < cols {
                dh = prev[x + 2].wrapping_sub(prev[x + 1]);
                k = adapt_k(k, ((code as u64 + 2 * dh.unsigned_abs() as u64) >> 1).min(u32::MAX as u64) as u32, 7);
            } else {
                k = adapt_k(k, code, 7);
            }
            qp[y * cols + x] = cur[x + 1].saturating_add(4);
        }
        cur[cols + 1] = cur[cols].wrapping_add(1);
        std::mem::swap(&mut prev, &mut cur);
    }
    Ok((qp, cols, rows))
}

fn q_step(q: i32) -> u32 {
    const BASE: [u32; 6] = [40, 45, 51, 57, 64, 72];
    let (d, m) = (q / 6, (q % 6).max(0) as usize);
    let base = BASE[m.min(5)];
    if d >= 6 { base.checked_shl((d - 6).min(31) as u32).unwrap_or(u32::MAX) } else { base >> (6 - d).min(31) as u32 }
}

/// Step table for `level` (rows of the level's subbands × QP columns).
fn step_table(qp: &[i32], cols: usize, rows: usize, plane_h: usize, level: usize) -> (Vec<u32>, usize) {
    let n_rows = plane_h.div_ceil(1 << level).max(1);
    let agg = 1usize << (level - 1);
    let mut t = vec![0u32; n_rows * cols];
    for y in 0..n_rows {
        for x in 0..cols {
            let sum: i64 = (0..agg).map(|j| qp.get((y * agg + j).min(rows.saturating_sub(1)) * cols + x).copied().unwrap_or(0) as i64).sum();
            t[y * cols + x] = q_step((sum / agg as i64) as i32);
        }
    }
    (t, n_rows)
}

fn dequantize(coeffs: &mut [i32], w: usize, sb: &Subband, level: usize, table: &(Vec<u32>, usize), cols: usize) {
    let (steps, rows) = table;
    let shift = 3 - level.min(3);
    for (y, line) in coeffs.chunks_mut(w.max(1)).enumerate() {
        let ty = y.min(rows.saturating_sub(1));
        for (x, v) in line.iter_mut().enumerate() {
            let base = steps.get(ty * cols + (x >> shift).min(cols.saturating_sub(1))).copied().unwrap_or(1) as u64;
            let quant = (sb.q_base as u64 + ((base * sb.q_mult as u64) >> 3)).clamp(1, 0x168000) as i64;
            *v = (*v as i64 * quant).clamp(i32::MIN as i64, i32::MAX as i64) as i32;
        }
    }
}

// ---- inverse wavelet (guide §10)

fn inverse53(low: &[i32], high: &[i32], out: &mut [i32]) {
    let (e, o) = (low.len(), high.len());
    if e == 0 {
        return;
    }
    let h = |i: isize| if o == 0 { 0 } else { high[i.clamp(0, o as isize - 1) as usize] };
    let mut even = vec![0i32; e];
    for (i, ev) in even.iter_mut().enumerate() {
        let (dl, dr) = (h(i as isize - 1), h(i as isize));
        *ev = low[i].wrapping_sub(dl.wrapping_add(dr).wrapping_add(2) >> 2);
    }
    for i in 0..e {
        if let Some(slot) = out.get_mut(2 * i) {
            *slot = even[i];
        }
    }
    for i in 0..o {
        let v = high[i].wrapping_add(even[i].wrapping_add(even[(i + 1).min(e - 1)]) >> 1);
        if let Some(slot) = out.get_mut(2 * i + 1) {
            *slot = v;
        }
    }
}

/// Band sizes for a plane: `[(w, h)]` in storage order (LL, then HL/LH/HH from the coarsest level).
fn geometry(w: usize, h: usize, levels: usize) -> Vec<(usize, usize, usize)> {
    let mut l = vec![(w, h)];
    for k in 1..=levels {
        let (pw, ph) = l[k - 1];
        l.push((pw.div_ceil(2), ph.div_ceil(2)));
    }
    let mut out = vec![(l[levels].0, l[levels].1, levels)];
    for k in (1..=levels).rev() {
        let (pw, ph) = l[k - 1];
        let (lw, lh) = l[k];
        out.push((pw / 2, lh, k));
        out.push((lw, ph / 2, k));
        out.push((pw / 2, ph / 2, k));
    }
    out
}

fn inverse_wavelet(mut bands: Vec<Vec<i32>>, geo: &[(usize, usize, usize)], levels: usize) -> Result<Vec<i32>> {
    let mut ll = std::mem::take(bands.first_mut().ok_or_else(|| corrupt("no LL band"))?);
    let (mut lw, mut lh) = (geo[0].0, geo[0].1);
    for step in 0..levels {
        let b = 1 + 3 * step;
        let (hl, lhb, hh) = (std::mem::take(&mut bands[b]), std::mem::take(&mut bands[b + 1]), std::mem::take(&mut bands[b + 2]));
        let (hlw, _) = (geo[b].0, geo[b].1);
        let lhh = geo[b + 1].1;
        let (w, h) = (lw + hlw, lh + lhh);
        // horizontal: rows of [LL|HL] and [LH|HH]
        let mut low_rows = vec![0i32; w * lh];
        for y in 0..lh {
            inverse53(&ll[y * lw..(y + 1) * lw], &hl[y * hlw..(y + 1) * hlw], &mut low_rows[y * w..(y + 1) * w]);
        }
        let mut high_rows = vec![0i32; w * lhh];
        for y in 0..lhh {
            inverse53(&lhb[y * lw..(y + 1) * lw], &hh[y * hlw..(y + 1) * hlw], &mut high_rows[y * w..(y + 1) * w]);
        }
        // vertical, column by column
        let mut next = vec![0i32; w * h];
        let (mut lo, mut hi, mut col) = (vec![0i32; lh], vec![0i32; lhh], vec![0i32; h]);
        for x in 0..w {
            for y in 0..lh {
                lo[y] = low_rows[y * w + x];
            }
            for y in 0..lhh {
                hi[y] = high_rows[y * w + x];
            }
            inverse53(&lo, &hi, &mut col);
            for y in 0..h {
                next[y * w + x] = col[y];
            }
        }
        ll = next;
        (lw, lh) = (w, h);
    }
    Ok(ll)
}

/// Decode one plane of the (single) tile.
fn decode_plane(file: &[u8], tile: &Tile, plane: &Plane, c: &Cmp1) -> Result<Vec<i32>> {
    let (pw, ph) = c.plane_dims();
    if plane.rounded_bits != 0 {
        return Err(RawError::Unsupported("CR3 with rounded CRX bands".into()));
    }
    let (qo, ql) = tile.qp.ok_or_else(|| RawError::Unsupported("CR3 without a QP map (legacy C-RAW)".into()))?;
    let (qp, cols, rows) = decode_qp(file.get(qo..qo + ql).ok_or_else(|| corrupt("QP data"))?, pw, ph)?;
    let tables: Vec<(Vec<u32>, usize)> = (1..=c.levels).map(|l| step_table(&qp, cols, rows, ph, l)).collect();
    let geo = geometry(pw, ph, c.levels);
    if geo.len() != plane.subbands.len() {
        return Err(corrupt("subband count does not match the wavelet levels"));
    }
    let mut bands = Vec::with_capacity(geo.len());
    for (i, (sb, &(w, h, level))) in plane.subbands.iter().zip(&geo).enumerate() {
        let data = file.get(sb.offset..sb.offset + sb.data_size).ok_or_else(|| corrupt("subband data"))?;
        let mut coeffs = decode_band(data, w, h, i == 0 && plane.supports_partial)?;
        dequantize(&mut coeffs, w, sb, level, &tables[level - 1], cols);
        bands.push(coeffs);
    }
    inverse_wavelet(bands, &geo, c.levels)
}

/// Decode the raw sample at `sample` (absolute offset) into a `width × height` Bayer mosaic.
pub(crate) fn decode(file: &[u8], sample: usize, c: &Cmp1) -> Result<Vec<u16>> {
    if c.version != 0x0200 {
        return Err(RawError::Unsupported(format!("CR3 CRX version {:#06x}", c.version)));
    }
    if c.levels == 0 {
        return Err(RawError::Unsupported("lossless CR3 (CRX without wavelet levels)".into()));
    }
    if c.levels > 3 || c.planes != 4 || !matches!(c.encoding, 0 | 3) {
        return Err(RawError::Unsupported(format!("CR3 CRX variant (levels {}, planes {}, encoding {})", c.levels, c.planes, c.encoding)));
    }
    if c.tile_width != c.width || c.tile_height != c.height {
        return Err(RawError::Unsupported("multi-tile CR3".into()));
    }
    let total = c.width.checked_mul(c.height).filter(|&n| n <= crate::MAX_SAMPLES).ok_or(RawError::Limit("image too large"))?;
    let tiles = parse_headers(file, sample, c)?;
    let tile = tiles.first().ok_or_else(|| corrupt("no tile"))?;
    let planes: Vec<Vec<i32>> = tile.planes.par_iter().map(|p| decode_plane(file, tile, p, c)).collect::<Result<_>>()?;
    let (pw, ph) = c.plane_dims();
    if planes.iter().any(|p| p.len() != pw * ph) {
        return Err(corrupt("plane size mismatch"));
    }
    let max = ((1u32 << c.bits) - 1) as i64;
    let median = 1i64 << (c.bits - 1);
    let clamp = |v: i64| v.clamp(0, max) as u16;
    let mut out = vec![0u16; total];
    // CFA placement: which plane (0..3 = R, G1, G2, B) goes to (0,0), (1,0), (0,1), (1,1)
    let place: [usize; 4] = match c.cfa {
        0 => [0, 1, 2, 3],
        1 => [1, 0, 3, 2],
        2 => [2, 3, 0, 1],
        _ => [3, 2, 1, 0],
    };
    out.par_chunks_mut(c.width * 2).enumerate().for_each(|(py, rows)| {
        for px in 0..pw {
            let i = py * pw + px;
            let (p0, p1, p2, p3) = (planes[0][i] as i64, planes[1][i] as i64, planes[2][i] as i64, planes[3][i] as i64);
            let rggb = if c.encoding == 3 {
                let mut gr = (median << 10) + (p0 << 10) - 168 * p1 - 585 * p3;
                gr = gr.signum() * (((gr.abs() + 512) >> 9) & !1);
                [
                    clamp(((median << 10) + (p0 << 10) + 1510 * p3 + 512) >> 10),
                    clamp((p2 + gr + 1) >> 1),
                    clamp((gr - p2 + 1) >> 1),
                    clamp(((median << 10) + (p0 << 10) + 1927 * p1 + 512) >> 10),
                ]
            } else {
                [clamp(median + p0), clamp(median + p1), clamp(median + p2), clamp(median + p3)]
            };
            for (slot, &pl) in place.iter().enumerate() {
                let (dx, dy) = (slot & 1, slot >> 1);
                if let Some(v) = rows.get_mut(dy * c.width + 2 * px + dx) {
                    *v = rggb[pl];
                }
            }
        }
        let _ = ph;
    });
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signed_mapping_and_k_adaptation() {
        assert_eq!([0, 1, 2, 3, 4].map(signed), [0, -1, 1, -2, 2]);
        assert_eq!(adapt_k(0, 0, 15), 0);
        assert_eq!(adapt_k(2, 0, 15), 1); // small code lowers k
        assert_eq!(adapt_k(2, 13, 15), 3); // code >> k = 3 > 2
        assert_eq!(adapt_k(2, 30, 15), 4); // code >> k = 7 > 5
        assert_eq!(adapt_k(15, 1 << 20, 15), 15);
    }

    #[test]
    fn median_predictor_cases() {
        // flat neighbourhood → left + gradient
        assert_eq!(median(10, 10, 10), 10);
        // edge cases mirror LOCO-I: min/max of left and top
        assert_eq!(median(10, 20, 25), 10);
        assert_eq!(median(10, 20, 5), 20);
        assert_eq!(median(10, 20, 15), 15);
    }

    #[test]
    fn inverse_53_reconstructs_a_forward_transform() {
        // forward LeGall 5/3 with the same boundary replication, then the inverse must return the input
        let x: Vec<i32> = vec![3, 7, 1, -4, 9, 12, 0, 5, 8];
        let n = x.len();
        let (e, o) = (n.div_ceil(2), n / 2);
        let xe = |i: isize| x[(i.clamp(0, n as isize - 1)) as usize];
        let high: Vec<i32> = (0..o).map(|i| x[2 * i + 1] - ((x[2 * i] + xe(2 * i as isize + 2)) >> 1)).collect();
        let h = |i: isize| high[i.clamp(0, o as isize - 1) as usize];
        let low: Vec<i32> = (0..e).map(|i| x[2 * i] + ((h(i as isize - 1) + h(i as isize) + 2) >> 2)).collect();
        let mut out = vec![0; n];
        inverse53(&low, &high, &mut out);
        assert_eq!(out, x);
    }

    #[test]
    fn geometry_of_three_levels() {
        let g = geometry(3572, 2380, 3);
        assert_eq!(g.len(), 10);
        assert_eq!(g[0], (447, 298, 3));
        assert_eq!(g[1], (446, 298, 3)); // HL3: floor(893/2)
        assert_eq!(g[9], (1786, 1190, 1)); // HH1
    }

    #[test]
    fn hostile_streams_error_instead_of_panicking() {
        for data in [&[][..], &[0u8; 3][..], &[0xff; 64][..], &[0x01, 0x80, 0x00, 0xff, 0x7f][..]] {
            for prog in [true, false] {
                let _ = decode_band(data, 17, 5, prog);
            }
            let _ = decode_qp(data, 40, 6);
        }
        assert!(Cmp1::parse(&[0; 10]).is_err());
    }
}

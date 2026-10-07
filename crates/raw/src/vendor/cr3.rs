//! Canon CR3: container (via [`lightcraft_meta::cr3`]), image area, white balance and the CRX decoder
//! ([`super::crx`] — a reference implementation, see its provenance note).
//!
//! - Raw track: the largest `CRAW` track with a `CMP1` header (the full-size raw; the other is a reduced copy).
//! - `IAD1` (in `CDI1`, layout per the public `lclevy/canon_cr3` notes, checked on EOS R6 Mark III files): the
//!   recommended crop and the left optical-black columns, which give the black level.
//! - White balance: the maker note (`CMT3`) `ColorBalance` (0x4001), probed like CR2.

use super::{black_from_columns, cr2, crx, white_from_data};
use crate::{BlackLevel, Cfa, ColorData, Mode, OpcodeLists, RawData, RawError, RawFormat, RawImage, Rect, Result};
use lightcraft_geom::Orientation;
use lightcraft_meta::cr3::{Cr3TrackKind, parse_cr3};
use lightcraft_tiff::{Tiff, tags as t};

/// The `IAD1` rectangles (inclusive pixel coordinates in the file, converted to rects).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ImageArea {
    crop: Rect,
    left_black: Rect,
}

fn image_area(p: &[u8], w: usize, h: usize) -> Option<ImageArea> {
    let v = |i: usize| p.get(2 * i..2 * i + 2).map(|s| u16::from_be_bytes([s[0], s[1]]) as usize);
    let rect = |l: usize, t: usize, r: usize, b: usize| (r >= l && b >= t).then(|| Rect::new(l, t, r - l + 1, b - t + 1).clipped(w, h));
    // 0..3: zero, zero, width, height; 4..7: flags; 8..11 crop; 12..15 left optical black
    let crop = rect(v(8)?, v(9)?, v(10)?, v(11)?)?;
    let left_black = rect(v(12)?, v(13)?, v(14)?, v(15)?)?;
    (crop.width > 0 && crop.height > 0).then_some(ImageArea { crop, left_black })
}

pub(crate) fn decode(bytes: &[u8], mode: Mode) -> Result<RawImage> {
    let c = parse_cr3(bytes).ok_or(RawError::NotRaw)?;
    let (cmp1, iad1, sample) = c
        .tracks
        .iter()
        .filter_map(|tr| match tr.kind {
            Cr3TrackKind::Raw { width, height, cmp1: Some(cmp1), iad1 } => Some((width as usize * height as usize, cmp1, iad1, tr.data?)),
            _ => None,
        })
        .max_by_key(|t| t.0)
        .map(|(_, cmp1, iad1, data)| (cmp1, iad1, data))
        .ok_or_else(|| RawError::Unsupported("CR3 without a raw track we can decode".into()))?;
    let cmp1 = crx::Cmp1::parse(bytes.get(cmp1.0..cmp1.0 + cmp1.1).ok_or_else(|| RawError::Corrupt("CR3: CMP1 outside file".into()))?)?;
    let (width, height) = (cmp1.width, cmp1.height);
    if width.checked_mul(height).is_none_or(|n| n > crate::MAX_SAMPLES) {
        return Err(RawError::Limit("image too large"));
    }
    let data = match mode {
        Mode::Full => crx::decode(bytes, sample.0, &cmp1)?,
        Mode::Header => {
            // fail at import, not at first render, for variants we can't decode
            if cmp1.version != 0x0200 || cmp1.levels == 0 || cmp1.tile_width != width || cmp1.tile_height != height {
                return Err(RawError::Unsupported("CR3 CRX variant not decoded yet".into()));
            }
            Vec::new()
        }
    };

    let area = iad1.and_then(|(o, l)| bytes.get(o..o + l)).and_then(|p| image_area(p, width, height));
    let full = Rect::new(0, 0, width, height);
    let crop = area.map(|a| a.crop).unwrap_or(full);
    let black = match area {
        Some(a) if mode == Mode::Full && a.left_black.width > 8 => {
            // inner columns of the optical-black strip, over the crop's rows (anchored at the active area)
            let cols = a.left_black.x + 4..a.left_black.x + a.left_black.width - 4;
            black_from_columns(&data, width, cols, crop.y..crop.y + crop.height, full)
        }
        _ => BlackLevel::uniform(0.0),
    };
    let white = if mode == Mode::Full { white_from_data(&data, cmp1.bits) } else { ((1u32 << cmp1.bits) - 1) as f32 };
    let cfa = Cfa::bayer_static(match cmp1.cfa {
        0 => "RGGB",
        1 => "GRBG",
        2 => "GBRG",
        _ => "BGGR",
    });

    let ifd0 = c.cmt[0].and_then(|b| Tiff::parse(b).ok());
    let orientation = Orientation::from_exif(ifd0.as_ref().and_then(|t| t.ifds.first()).and_then(|i| i.u16(t::ORIENTATION)).unwrap_or(1));
    let wb = c.cmt[2]
        .and_then(|b| Tiff::parse(b).ok())
        .and_then(|t| t.ifds.into_iter().next())
        .and_then(|mn| mn.u64s(cr2::COLOR_BALANCE))
        .and_then(|v| cr2::wb_from_color_balance(&v));
    let mut metadata = lightcraft_meta::cr3::merged_exif(&c).map(|e| lightcraft_meta::read_exif(&e)).unwrap_or_default();
    metadata.width = Some(crop.width as u32);
    metadata.height = Some(crop.height as u32);
    let img = RawImage {
        format: RawFormat::Cr3,
        width,
        height,
        cpp: 1,
        data: RawData::U16(data),
        cfa: Some(cfa),
        bits: cmp1.bits,
        black,
        white: vec![white],
        // like CR2: the recommended crop is the active area, the default crop all of it
        active_area: crop,
        crop: Rect::new(0, 0, crop.width, crop.height),
        orientation,
        color: ColorData::default(),
        wb_multipliers: wb,
        linearized: false,
        opcodes: OpcodeLists::default(),
        metadata,
    };
    img.validate_for(mode)?;
    Ok(img)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_area_from_iad1() {
        // EOS R6 Mark III values
        let vals: [u16; 16] = [0, 0, 7144, 4760, 1, 2, 1, 0, 172, 108, 7131, 4747, 0, 0, 159, 4751];
        let p: Vec<u8> = vals.iter().flat_map(|v| v.to_be_bytes()).collect();
        let a = image_area(&p, 7144, 4760).unwrap();
        assert_eq!(a.crop, Rect::new(172, 108, 6960, 4640));
        assert_eq!(a.left_black, Rect::new(0, 0, 160, 4752).clipped(7144, 4760));
        assert!(image_area(&p[..10], 7144, 4760).is_none());
    }
}

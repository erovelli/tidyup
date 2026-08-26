//! Small UTF-8-safe string helpers shared across the pipeline.
//!
//! Byte-indexed slices (`&s[..n]`) panic when `n` falls in the middle of a
//! multi-byte codepoint. Extracted document text and filenames are arbitrary
//! UTF-8 (accented Latin, CJK, emoji), so every fixed-size truncation must land
//! on a char boundary.

/// The longest prefix of `s` that is at most `max_bytes` long and ends on a
/// char boundary. Never panics; returns `s` unchanged when it already fits.
///
/// Semantically "the first ~`max_bytes` bytes", rounded *down* to a boundary —
/// a drop-in, panic-safe replacement for `&s[..s.len().min(max_bytes)]`.
#[must_use]
pub fn char_prefix(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    let end = s
        .char_indices()
        .map(|(index, _)| index)
        .take_while(|index| *index <= max_bytes)
        .last()
        .unwrap_or(0);
    s.get(..end).unwrap_or_default()
}

/// Find the first standalone year in the supported `2000..=2039` range.
#[must_use]
pub fn find_year(s: &str) -> Option<i32> {
    let bytes = s.as_bytes();
    for (index, window) in bytes.windows(4).enumerate() {
        if let [b'2', b'0', third, fourth] = window {
            if !third.is_ascii_digit() || !fourth.is_ascii_digit() {
                continue;
            }
            let before_ok = index
                .checked_sub(1)
                .and_then(|before| bytes.get(before))
                .is_none_or(|byte| !byte.is_ascii_digit());
            let after_ok = index
                .checked_add(4)
                .and_then(|after| bytes.get(after))
                .is_none_or(|byte| !byte.is_ascii_digit());
            let year = std::str::from_utf8(window).ok()?.parse::<i32>().ok()?;
            if before_ok && after_ok && (2000..=2039).contains(&year) {
                return Some(year);
            }
        }
    }
    None
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn returns_whole_string_when_shorter() {
        assert_eq!(char_prefix("abc", 10), "abc");
        assert_eq!(char_prefix("", 10), "");
    }

    #[test]
    fn ascii_truncates_exactly() {
        assert_eq!(char_prefix("abcdef", 3), "abc");
    }

    #[test]
    fn never_splits_a_multibyte_codepoint() {
        // "é" is 2 bytes (0xC3 0xA9); "𐍈" is 4 bytes. Cutting at a byte that
        // lands mid-codepoint must round down, not panic.
        let s = "aé𐍈bc"; // bytes: 'a'(1) 'é'(2) '𐍈'(4) 'b'(1) 'c'(1) = 9 bytes
        for n in 0..=s.len() {
            let p = char_prefix(s, n); // must not panic for any cut point
            assert!(s.starts_with(p));
            assert!(p.len() <= n);
        }
        // A cut inside "é" (byte 2) rounds down to just "a".
        assert_eq!(char_prefix(s, 2), "a");
        // A cut inside "𐍈" (bytes 4..6) rounds down to "aé".
        assert_eq!(char_prefix(s, 5), "aé");
    }

    #[test]
    fn full_length_is_identity() {
        let s = "aé𐍈";
        assert_eq!(char_prefix(s, s.len()), s);
    }
}

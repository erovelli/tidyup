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
    let mut end = max_bytes;
    // `is_char_boundary(0)` is always true, so this terminates.
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
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

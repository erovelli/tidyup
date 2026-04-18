//! Small, dependency-free helpers: cosine similarity, L2 normalization, year
//! extraction from filenames and content.

/// Cosine similarity of two vectors. Returns `0.0` if either vector has zero
/// norm. Works correctly on non-normalized inputs.
#[must_use]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

/// L2-normalize `v` in place. No-op if the vector has zero norm.
pub fn l2_normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Extract a 4-digit year (2000–2039) from a filename or text body.
///
/// Checks the filename first (stronger signal — users put dates in filenames
/// more consistently than prose), then falls back to the first ~1 000 chars
/// of body text. Window size is bounded to keep this cheap on large files.
#[must_use]
pub fn extract_year(text: &str, filename: &str) -> Option<i32> {
    find_year_in(filename).or_else(|| {
        let window = &text[..text.len().min(1000)];
        find_year_in(window)
    })
}

/// Scan `s` for the first standalone 4-digit year in `2000..=2039`. The
/// match must be bounded on both sides by non-digit characters or string
/// edges, so `file12024.pdf` and `file20249.pdf` both return `None`.
fn find_year_in(s: &str) -> Option<i32> {
    let bytes = s.as_bytes();
    let len = bytes.len();
    if len < 4 {
        return None;
    }
    for i in 0..=len - 4 {
        if bytes[i] == b'2'
            && bytes[i + 1] == b'0'
            && bytes[i + 2].is_ascii_digit()
            && bytes[i + 3].is_ascii_digit()
        {
            let before_ok = i == 0 || !bytes[i - 1].is_ascii_digit();
            let after_ok = i + 4 >= len || !bytes[i + 4].is_ascii_digit();
            if before_ok && after_ok {
                if let Ok(year) = s[i..i + 4].parse::<i32>() {
                    if (2000..=2039).contains(&year) {
                        return Some(year);
                    }
                }
            }
        }
    }
    None
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn cosine_identical_vectors() {
        let v = vec![1.0, 2.0, 3.0];
        let s = cosine_similarity(&v, &v);
        assert!((s - 1.0).abs() < 1e-5);
    }

    #[test]
    fn cosine_orthogonal_vectors() {
        let s = cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]);
        assert!(s.abs() < 1e-5);
    }

    #[test]
    fn cosine_zero_vector() {
        assert_eq!(cosine_similarity(&[1.0, 2.0], &[0.0, 0.0]), 0.0);
    }

    #[test]
    fn cosine_opposite_vectors() {
        let s = cosine_similarity(&[1.0, 0.0], &[-1.0, 0.0]);
        assert!((s + 1.0).abs() < 1e-5);
    }

    #[test]
    fn l2_normalize_unit_length() {
        let mut v = vec![3.0, 4.0];
        l2_normalize(&mut v);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn l2_normalize_zero_vector_noop() {
        let mut v = vec![0.0, 0.0, 0.0];
        l2_normalize(&mut v);
        assert_eq!(v, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn extract_year_from_filename() {
        assert_eq!(extract_year("", "2024_tax_return.pdf"), Some(2024));
        assert_eq!(extract_year("", "IMG_4269.HEIC"), None);
        assert_eq!(extract_year("", "report-2023-q4.pdf"), Some(2023));
    }

    #[test]
    fn extract_year_from_content() {
        assert_eq!(
            extract_year("Form 1040 Tax Year 2024", "scan.pdf"),
            Some(2024),
        );
        assert_eq!(extract_year("no year here", "scan.pdf"), None);
    }

    #[test]
    fn extract_year_boundaries() {
        assert_eq!(extract_year("", "file12024.pdf"), None);
        assert_eq!(extract_year("", "file20249.pdf"), None);
        assert_eq!(extract_year("", "2025report.pdf"), Some(2025));
    }

    #[test]
    fn extract_year_range() {
        assert_eq!(find_year_in("1999"), None);
        assert_eq!(find_year_in("2000"), Some(2000));
        assert_eq!(find_year_in("2039"), Some(2039));
        assert_eq!(find_year_in("2040"), None);
    }

    #[test]
    fn extract_year_prefers_filename_over_content() {
        assert_eq!(
            extract_year("mentions 2015 in body", "2024_return.pdf"),
            Some(2024),
        );
    }
}

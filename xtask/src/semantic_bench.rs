//! Warm-path semantic image latency measurement.

use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use tidyup_core::inference::ImageEmbeddingBackend;
use tidyup_embeddings_ort::SigLipEmbeddings;

#[allow(clippy::redundant_pub_crate)]
pub(super) fn run(image: &Path, iterations: usize, fail_over_ms: u64) -> Result<()> {
    if iterations == 0 {
        bail!("--iterations must be greater than zero");
    }
    tidyup_embeddings_ort::verify_siglip_model().context("verify SigLIP model bundle")?;

    let load_started = Instant::now();
    let backend = SigLipEmbeddings::load_default().context("load SigLIP")?;
    let model_load = load_started.elapsed();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("build benchmark runtime")?;
    let concepts = runtime
        .block_on(tidyup_pipeline::semantic::prepare_visual_concepts(&backend))
        .context("prepare visual concept bank")?;

    let warm_bytes = std::fs::read(image)
        .with_context(|| format!("read benchmark image {}", image.display()))?;
    runtime
        .block_on(backend.embed_image(&warm_bytes, "application/octet-stream"))
        .context("warm SigLIP image tower")?;

    let mut samples = Vec::with_capacity(iterations);
    let mut last_ranked = Vec::new();
    let mut last_rename = tidyup_pipeline::naming::RenameProposal::Keep;
    let mut last_embedding = Vec::new();
    for _ in 0..iterations {
        let started = Instant::now();
        let bytes = std::fs::read(image)
            .with_context(|| format!("read benchmark image {}", image.display()))?;
        let _content_hash = blake3::hash(&bytes);
        let embedding = runtime
            .block_on(backend.embed_image(&bytes, "application/octet-stream"))
            .context("embed benchmark image")?;
        let ranked = tidyup_pipeline::semantic::rank_concepts(&embedding, &concepts);
        let naming_concepts = ranked.get(..2).unwrap_or(ranked.as_slice());
        let rename = tidyup_pipeline::naming::propose_grounded_rename(image, naming_concepts);
        std::hint::black_box(&ranked);
        std::hint::black_box(&rename);
        last_ranked = ranked;
        last_rename = rename;
        last_embedding = embedding;
        samples.push(started.elapsed());
    }
    samples.sort_unstable();
    let p50 = percentile(&samples, 50);
    let p95 = percentile(&samples, 95);
    let worst = samples.last().copied().unwrap_or_default();

    println!("semantic image benchmark");
    println!("  model load (cold): {}", format_duration(model_load));
    println!("  warm iterations:   {iterations}");
    println!("  p50:               {}", format_duration(p50));
    println!("  p95:               {}", format_duration(p95));
    println!("  worst:             {}", format_duration(worst));
    println!("  gate:              {fail_over_ms} ms");
    let top_concepts = last_ranked
        .iter()
        .take(5)
        .map(|concept| format!("{}:{:.3}", concept.label, concept.score))
        .collect::<Vec<_>>()
        .join(", ");
    println!("  top concepts:      {top_concepts}");
    println!("  rename candidate:  {last_rename:?}");
    let mut raw_concepts = concepts
        .iter()
        .map(|concept| {
            (
                concept.label.as_str(),
                tidyup_embeddings_ort::cosine_similarity(&last_embedding, &concept.embedding),
            )
        })
        .collect::<Vec<_>>();
    raw_concepts.sort_by(|left, right| {
        right
            .1
            .partial_cmp(&left.1)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let raw_top = raw_concepts
        .iter()
        .take(5)
        .map(|(label, score)| format!("{label}:{score:.4}"))
        .collect::<Vec<_>>()
        .join(", ");
    println!("  raw cosine top:    {raw_top}");

    if p95 > Duration::from_millis(fail_over_ms) {
        bail!(
            "warm semantic image p95 ({}) exceeds {} ms",
            format_duration(p95),
            fail_over_ms
        );
    }
    Ok(())
}

fn percentile(samples: &[Duration], percentile: usize) -> Duration {
    if samples.is_empty() {
        return Duration::default();
    }
    let rank = samples
        .len()
        .saturating_mul(percentile)
        .div_ceil(100)
        .saturating_sub(1)
        .min(samples.len().saturating_sub(1));
    samples.get(rank).copied().unwrap_or_default()
}

fn format_duration(duration: Duration) -> String {
    format!("{:.1} ms", duration.as_secs_f64() * 1_000.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_uses_nearest_rank() {
        let samples = (1_u64..=20).map(Duration::from_millis).collect::<Vec<_>>();
        assert_eq!(percentile(&samples, 50), Duration::from_millis(10));
        assert_eq!(percentile(&samples, 95), Duration::from_millis(19));
    }
}

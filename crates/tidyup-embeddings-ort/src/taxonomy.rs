//! Default hierarchical taxonomy for scan-mode classification.
//!
//! Each [`TaxonomyEntry`] is a leaf in the target folder tree with a rich
//! description tuned for cosine-similarity matching against file content.
//! Entries marked `temporal` append a year subdirectory when a year is found
//! in the file.
//!
//! # Cache
//!
//! Taxonomy embeddings are cached to disk to avoid the ~1 s cost of embedding
//! all ~70 descriptions on every startup. Invalidation is by BLAKE3 hash of
//! model id + all descriptions — any change in either (new model, tweaked
//! copy, added entry) forces a rebuild.

use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// A single leaf in the default taxonomy tree.
#[derive(Debug, Clone, Copy)]
pub struct TaxonomyEntry {
    /// Target folder path (e.g. `"Finance/Taxes/"`). Must end with `'/'`.
    pub path: &'static str,
    /// Rich description used to compute the category embedding.
    pub description: &'static str,
    /// Whether to append a year subdirectory (`"Finance/Taxes/2024/"`) when
    /// a year can be extracted from the content or filename.
    pub temporal: bool,
}

/// On-disk taxonomy embedding cache.
///
/// Cache is valid iff (`model_id`, `blake3_hash(descriptions)`) match. Entry
/// count is stored for diagnostics but not used for validation — the hash
/// already captures it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaxonomyCache {
    /// Model identifier at the time embeddings were computed.
    pub model_id: String,
    /// Number of taxonomy entries (informational).
    pub entry_count: usize,
    /// BLAKE3 hash (hex) of concatenated entry descriptions, NUL-separated.
    pub descriptions_hash: String,
    /// One embedding per entry, in the same order.
    pub embeddings: Vec<Vec<f32>>,
}

impl TaxonomyCache {
    /// Hash the taxonomy descriptions with BLAKE3.
    #[must_use]
    pub fn compute_hash(entries: &[TaxonomyEntry]) -> String {
        let mut hasher = blake3::Hasher::new();
        for entry in entries {
            hasher.update(entry.description.as_bytes());
            hasher.update(b"\0");
        }
        hasher.finalize().to_hex().to_string()
    }

    /// Returns `true` if this cache is valid for the given model + taxonomy.
    #[must_use]
    pub fn is_valid(&self, model_id: &str, entries: &[TaxonomyEntry]) -> bool {
        self.model_id == model_id && self.descriptions_hash == Self::compute_hash(entries)
    }

    /// Load a cache from disk. Returns `None` if the file is missing,
    /// unreadable, or the JSON shape doesn't parse — cache misses are
    /// non-fatal.
    #[must_use]
    pub fn load(path: &Path) -> Option<Self> {
        let data = std::fs::read_to_string(path).ok()?;
        serde_json::from_str(&data).ok()
    }

    /// Write a cache to disk. Creates the parent directory if missing.
    ///
    /// # Errors
    /// Propagates filesystem and JSON serialization errors.
    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create cache dir {}", parent.display()))?;
        }
        let data = serde_json::to_string(self).context("serialize taxonomy cache")?;
        std::fs::write(path, data)
            .with_context(|| format!("write taxonomy cache to {}", path.display()))?;
        Ok(())
    }
}

/// Default hierarchical taxonomy. Each entry is a leaf in the folder tree.
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn default_taxonomy() -> Vec<TaxonomyEntry> {
    vec![
        // ── Finance ──────────────────────────────────────────────────────
        TaxonomyEntry {
            path: "Finance/Taxes/",
            description: "tax return W-2 W2 1099 1040 income tax IRS federal state tax refund withholding deduction schedule K-1 estimated tax",
            temporal: true,
        },
        TaxonomyEntry {
            path: "Finance/Invoices/",
            description: "invoice bill payment due amount owed billing statement vendor purchase order accounts payable",
            temporal: true,
        },
        TaxonomyEntry {
            path: "Finance/Banking/",
            description: "bank statement account summary transactions deposits withdrawals checking savings balance wire transfer routing number",
            temporal: true,
        },
        TaxonomyEntry {
            path: "Finance/Insurance/",
            description: "insurance policy premium coverage deductible claim auto home health life liability umbrella renters",
            temporal: true,
        },
        TaxonomyEntry {
            path: "Finance/Receipts/",
            description: "receipt purchase transaction store payment confirmation order total items bought retail",
            temporal: true,
        },
        TaxonomyEntry {
            path: "Finance/Investments/",
            description: "investment portfolio brokerage stock bond dividend 401k IRA mutual fund capital gains securities trading",
            temporal: true,
        },
        TaxonomyEntry {
            path: "Finance/Payroll/",
            description: "pay stub payslip paycheck earnings statement salary wages deductions net pay gross pay direct deposit",
            temporal: true,
        },
        TaxonomyEntry {
            path: "Finance/Donations/",
            description: "donation receipt charitable contribution 501c3 tax-deductible nonprofit giving philanthropy tithe",
            temporal: true,
        },
        TaxonomyEntry {
            path: "Finance/Budget/",
            description: "budget expense tracker spending plan monthly budget financial plan cash flow savings goal debt repayment",
            temporal: true,
        },
        TaxonomyEntry {
            path: "Finance/Credit/",
            description: "credit report credit score FICO credit card statement annual percentage rate APR credit limit balance transfer",
            temporal: true,
        },
        TaxonomyEntry {
            path: "Finance/Loans/",
            description: "loan agreement promissory note amortization schedule interest rate monthly payment principal balance payoff student loan personal loan",
            temporal: true,
        },
        // ── Legal ────────────────────────────────────────────────────────
        TaxonomyEntry {
            path: "Legal/Contracts/",
            description: "contract agreement terms conditions signatures parties obligations NDA non-disclosure employment service",
            temporal: true,
        },
        TaxonomyEntry {
            path: "Legal/Leases/",
            description: "lease rental agreement tenant landlord property rent security deposit residential commercial sublease",
            temporal: true,
        },
        TaxonomyEntry {
            path: "Legal/Correspondence/",
            description: "legal letter notice subpoena court attorney summons filing motion petition affidavit",
            temporal: true,
        },
        TaxonomyEntry {
            path: "Legal/Identity/",
            description: "passport driver license birth certificate social security card identification visa green card immigration naturalization",
            temporal: false,
        },
        TaxonomyEntry {
            path: "Legal/Estate/",
            description: "last will testament power of attorney trust agreement estate plan beneficiary probate executor inheritance",
            temporal: false,
        },
        TaxonomyEntry {
            path: "Legal/Permits/",
            description: "permit license certification professional license building permit business license occupational license regulatory approval",
            temporal: true,
        },
        // ── Medical ──────────────────────────────────────────────────────
        TaxonomyEntry {
            path: "Medical/Records/",
            description: "medical records prescription lab results diagnosis health doctor patient hospital clinical treatment referral physician",
            temporal: true,
        },
        TaxonomyEntry {
            path: "Medical/Billing/",
            description: "medical bill explanation of benefits EOB copay health insurance claim hospital charges deductible out of pocket",
            temporal: true,
        },
        TaxonomyEntry {
            path: "Medical/Vaccinations/",
            description: "vaccination immunization vaccine card vaccine record booster shot flu shot COVID vaccine antibody test",
            temporal: false,
        },
        TaxonomyEntry {
            path: "Medical/Dental/",
            description: "dental records dentist cleaning x-ray crown filling root canal orthodontics braces oral surgery",
            temporal: true,
        },
        TaxonomyEntry {
            path: "Medical/Vision/",
            description: "eye exam optometrist ophthalmologist prescription glasses contacts lens vision test retina",
            temporal: true,
        },
        TaxonomyEntry {
            path: "Medical/Mental Health/",
            description: "therapy counseling psychiatrist psychologist mental health treatment session notes anxiety depression ADHD",
            temporal: true,
        },
        // ── School / Academic ────────────────────────────────────────────
        TaxonomyEntry {
            path: "School/Assignments/",
            description: "homework assignment coursework project rubric syllabus class submission grading problem set lab report",
            temporal: false,
        },
        TaxonomyEntry {
            path: "School/Transcripts/",
            description: "transcript grades GPA academic record university college degree diploma semester enrollment verification",
            temporal: false,
        },
        TaxonomyEntry {
            path: "School/Notes/",
            description: "lecture notes study notes class notes review summary outline study guide chapter notes",
            temporal: false,
        },
        TaxonomyEntry {
            path: "School/Research/",
            description: "research paper thesis dissertation journal article bibliography citation abstract methodology peer review",
            temporal: false,
        },
        TaxonomyEntry {
            path: "School/Financial Aid/",
            description: "financial aid FAFSA scholarship tuition student loan bursar award letter grant work-study",
            temporal: true,
        },
        TaxonomyEntry {
            path: "School/Certificates/",
            description: "certificate completion certification course certificate training certificate professional development continuing education credential",
            temporal: false,
        },
        // ── Work ─────────────────────────────────────────────────────────
        TaxonomyEntry {
            path: "Work/Reports/",
            description: "business report quarterly annual performance review analysis metrics KPI earnings status update",
            temporal: true,
        },
        TaxonomyEntry {
            path: "Work/Presentations/",
            description: "presentation slides powerpoint keynote deck pitch proposal meeting agenda demo",
            temporal: false,
        },
        TaxonomyEntry {
            path: "Work/Correspondence/",
            description: "business email memo letter correspondence professional communication office internal external",
            temporal: true,
        },
        TaxonomyEntry {
            path: "Work/Career/",
            description: "resume curriculum vitae CV cover letter job application LinkedIn profile references recommendation letter portfolio",
            temporal: false,
        },
        TaxonomyEntry {
            path: "Work/Meetings/",
            description: "meeting minutes meeting notes action items standup agenda retrospective sprint planning sync",
            temporal: true,
        },
        TaxonomyEntry {
            path: "Work/Employment/",
            description: "offer letter employment agreement onboarding employee handbook termination notice severance benefits enrollment non-compete",
            temporal: true,
        },
        TaxonomyEntry {
            path: "Work/Projects/",
            description: "project plan timeline gantt chart milestone deliverable scope requirements specification roadmap backlog sprint",
            temporal: false,
        },
        // ── Travel ───────────────────────────────────────────────────────
        TaxonomyEntry {
            path: "Travel/Flights/",
            description: "boarding pass flight confirmation airline ticket departure arrival gate terminal seat assignment e-ticket",
            temporal: true,
        },
        TaxonomyEntry {
            path: "Travel/Hotels/",
            description: "hotel reservation booking confirmation check-in checkout room accommodation lodging Airbnb VRBO hostel",
            temporal: true,
        },
        TaxonomyEntry {
            path: "Travel/Itineraries/",
            description: "travel itinerary trip plan vacation schedule tour excursion cruise road trip sightseeing attraction",
            temporal: true,
        },
        TaxonomyEntry {
            path: "Travel/Tickets/",
            description: "event ticket admission pass entry museum concert theater show festival exhibition park ticket valid on access time",
            temporal: true,
        },
        TaxonomyEntry {
            path: "Travel/Visas/",
            description: "visa application travel visa work permit entry permit immigration customs declaration border crossing ESTA",
            temporal: true,
        },
        TaxonomyEntry {
            path: "Travel/Car Rental/",
            description: "car rental vehicle rental reservation pickup return rental agreement collision damage waiver",
            temporal: true,
        },
        // ── Real Estate / Property ───────────────────────────────────────
        TaxonomyEntry {
            path: "Real Estate/",
            description: "mortgage closing disclosure title deed property tax appraisal homeowners association HOA settlement escrow home inspection survey",
            temporal: true,
        },
        // ── Automotive ───────────────────────────────────────────────────
        TaxonomyEntry {
            path: "Automotive/",
            description: "vehicle registration VIN car title DMV emissions test auto repair maintenance record oil change tire rotation inspection service",
            temporal: true,
        },
        // ── Pets ─────────────────────────────────────────────────────────
        TaxonomyEntry {
            path: "Pets/",
            description: "veterinary vet record pet vaccination rabies microchip pet insurance neuter spay animal health pet license adoption",
            temporal: true,
        },
        // ── Documents ────────────────────────────────────────────────────
        TaxonomyEntry {
            path: "Documents/Personal/",
            description: "personal document letter form application certificate registration membership personal correspondence",
            temporal: false,
        },
        TaxonomyEntry {
            path: "Documents/Manuals/",
            description: "manual instruction guide handbook reference documentation user guide tutorial how-to owner manual quick start",
            temporal: false,
        },
        TaxonomyEntry {
            path: "Documents/Templates/",
            description: "template form blank document fillable boilerplate standard format reusable letterhead",
            temporal: false,
        },
        // ── Warranties ───────────────────────────────────────────────────
        TaxonomyEntry {
            path: "Warranties/",
            description: "warranty guarantee product registration serial number proof of purchase extended warranty service plan protection plan",
            temporal: true,
        },
        // ── Recipes ──────────────────────────────────────────────────────
        TaxonomyEntry {
            path: "Recipes/",
            description: "recipe ingredients tablespoon teaspoon cups preheat bake serves prep time cook time meal cooking food nutrition",
            temporal: false,
        },
        // ── Health & Fitness ─────────────────────────────────────────────
        TaxonomyEntry {
            path: "Health & Fitness/",
            description: "workout exercise training plan meal plan calories macros body weight fitness routine gym running cycling yoga stretching",
            temporal: false,
        },
        // ── Family / Children ────────────────────────────────────────────
        TaxonomyEntry {
            path: "Family/Children/",
            description: "report card parent teacher conference permission slip field trip daycare pediatrician child custody school enrollment immunization",
            temporal: true,
        },
        // ── Government ───────────────────────────────────────────────────
        TaxonomyEntry {
            path: "Government/",
            description: "voter registration jury duty selective service government benefit social security statement citizenship military service records",
            temporal: true,
        },
        // ── Media ────────────────────────────────────────────────────────
        TaxonomyEntry {
            path: "Photos/",
            description: "photograph picture image photo camera portrait landscape selfie vacation family event wedding",
            temporal: true,
        },
        TaxonomyEntry {
            path: "Screenshots/",
            description: "screenshot screen capture screen grab desktop browser window UI interface error message notification",
            temporal: false,
        },
        TaxonomyEntry {
            path: "Videos/",
            description: "video recording clip footage movie film camera handheld vlog home video personal recording",
            temporal: true,
        },
        TaxonomyEntry {
            path: "Videos/Recordings/",
            description: "screen recording tutorial presentation walkthrough demo screencast webinar how-to capture lecture recording",
            temporal: false,
        },
        TaxonomyEntry {
            path: "Music/",
            description: "music song track album artist band singer genre pop rock hip-hop classical jazz electronic country",
            temporal: false,
        },
        TaxonomyEntry {
            path: "Audio/Podcasts/",
            description: "podcast episode show interview discussion talk radio host guest series subscribe",
            temporal: false,
        },
        TaxonomyEntry {
            path: "Audio/Memos/",
            description: "voice memo recording note dictation phone call voicemail audio message voice note",
            temporal: false,
        },
        TaxonomyEntry {
            path: "Audio/Audiobooks/",
            description: "audiobook book narration narrator chapter listening spoken word Audible librivox",
            temporal: false,
        },
        // ── Technical ────────────────────────────────────────────────────
        TaxonomyEntry {
            path: "Code/",
            description: "source code programming script function class module software development repository algorithm implementation",
            temporal: false,
        },
        TaxonomyEntry {
            path: "Code/Config/",
            description: "configuration file settings environment variables YAML TOML JSON INI dotfile properties",
            temporal: false,
        },
        TaxonomyEntry {
            path: "Spreadsheets/",
            description: "spreadsheet excel worksheet data table CSV calculations formulas rows columns pivot chart Google Sheets",
            temporal: false,
        },
        TaxonomyEntry {
            path: "Databases/",
            description: "database SQL dump export backup migration schema table query sqlite postgres mysql mongo",
            temporal: false,
        },
        // ── Books / Reading ──────────────────────────────────────────────
        TaxonomyEntry {
            path: "Books/",
            description: "ebook book publication novel textbook reading literature epub kindle PDF chapter fiction nonfiction",
            temporal: false,
        },
        // ── Creative ─────────────────────────────────────────────────────
        TaxonomyEntry {
            path: "Creative/Design/",
            description: "design graphic illustration mockup wireframe UI UX logo icon vector Photoshop Illustrator Figma Sketch",
            temporal: false,
        },
        TaxonomyEntry {
            path: "Creative/Writing/",
            description: "creative writing story poem essay draft fiction narrative blog post article manuscript journal entry",
            temporal: false,
        },
        TaxonomyEntry {
            path: "Creative/Music Production/",
            description: "music production beat DAW MIDI synthesizer mix master sample loop Ableton FL Studio Logic Pro GarageBand",
            temporal: false,
        },
        TaxonomyEntry {
            path: "Creative/Video Projects/",
            description: "video editing project timeline Premiere Final Cut DaVinci Resolve raw footage B-roll color grading export render",
            temporal: false,
        },
        // ── Software / Downloads ─────────────────────────────────────────
        TaxonomyEntry {
            path: "Software/Installers/",
            description: "installer setup application download DMG EXE MSI package DEB RPM AppImage software installation program",
            temporal: false,
        },
        TaxonomyEntry {
            path: "Software/Disk Images/",
            description: "disk image ISO IMG VMDK VDI QCOW2 bootable virtual machine backup clone system image",
            temporal: false,
        },
        // ── Archives ─────────────────────────────────────────────────────
        TaxonomyEntry {
            path: "Archives/",
            description: "compressed archive ZIP TAR GZ BZ2 7Z RAR extraction bundle package backup archive file",
            temporal: false,
        },
        // ── 3D / CAD ─────────────────────────────────────────────────────
        TaxonomyEntry {
            path: "3D Models/",
            description: "3D model STL OBJ FBX GLTF Blender CAD rendering mesh polygon sculpture print prototype",
            temporal: false,
        },
        // ── Fonts ────────────────────────────────────────────────────────
        TaxonomyEntry {
            path: "Fonts/",
            description: "font typeface TTF OTF WOFF typography lettering sans-serif serif monospace display handwriting",
            temporal: false,
        },
        // ── Maps / GIS ───────────────────────────────────────────────────
        TaxonomyEntry {
            path: "Maps/",
            description: "map geospatial GIS shapefile KML GPX coordinates latitude longitude geographic terrain satellite topographic",
            temporal: false,
        },
        // ── Presentations (standalone) ───────────────────────────────────
        TaxonomyEntry {
            path: "Presentations/",
            description: "presentation slides powerpoint keynote deck Google Slides pitch demo conference talk workshop seminar",
            temporal: false,
        },
        // ── Catch-all ────────────────────────────────────────────────────
        TaxonomyEntry {
            path: "Misc/",
            description: "miscellaneous other uncategorized general random assorted file unknown",
            temporal: false,
        },
    ]
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn taxonomy_has_entries() {
        assert!(default_taxonomy().len() >= 50);
    }

    #[test]
    fn taxonomy_paths_end_with_slash() {
        for entry in default_taxonomy() {
            assert!(
                entry.path.ends_with('/'),
                "path missing trailing slash: {}",
                entry.path,
            );
        }
    }

    #[test]
    fn taxonomy_descriptions_nonempty() {
        for entry in default_taxonomy() {
            assert!(
                !entry.description.trim().is_empty(),
                "empty description for {}",
                entry.path,
            );
        }
    }

    #[test]
    fn cache_hash_is_deterministic() {
        let entries = default_taxonomy();
        let h1 = TaxonomyCache::compute_hash(&entries);
        let h2 = TaxonomyCache::compute_hash(&entries);
        assert_eq!(h1, h2);
        assert_eq!(h1.len(), 64); // blake3 hex
    }

    #[test]
    fn cache_hash_differs_on_description_change() {
        let a = vec![TaxonomyEntry {
            path: "X/",
            description: "alpha",
            temporal: false,
        }];
        let b = vec![TaxonomyEntry {
            path: "X/",
            description: "beta",
            temporal: false,
        }];
        assert_ne!(
            TaxonomyCache::compute_hash(&a),
            TaxonomyCache::compute_hash(&b)
        );
    }

    #[test]
    fn cache_roundtrip_via_disk() {
        let entries = default_taxonomy();
        let hash = TaxonomyCache::compute_hash(&entries);
        let cache = TaxonomyCache {
            model_id: "test-model".into(),
            entry_count: entries.len(),
            descriptions_hash: hash,
            embeddings: vec![vec![0.1, 0.2, 0.3]; entries.len()],
        };

        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("cache.json");
        cache.save(&path).unwrap();

        let loaded = TaxonomyCache::load(&path).unwrap();
        assert!(loaded.is_valid("test-model", &entries));
        assert!(!loaded.is_valid("other-model", &entries));
    }

    #[test]
    fn cache_load_returns_none_on_missing() {
        assert!(TaxonomyCache::load(Path::new("/definitely/missing.json")).is_none());
    }
}

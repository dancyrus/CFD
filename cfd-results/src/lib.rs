//! Results get committed, not reported in chat (CLAUDE.md).
//!
//! Every acceptance-ladder rung and every benchmark calls into this crate to
//! write its measured numbers to `docs/results/<suite>-<machine>.json` —
//! from inside the test, BEFORE its own asserts, so emission cannot be
//! skipped and a failing run still leaves its numbers behind (with
//! `pass: false`). The schema is documented in `docs/results/README.md`;
//! extend it there, never invent a second format.
//!
//! Mechanics: tests run as parallel threads across several test binaries, so
//! each record is first appended to a line-oriented staging file under
//! `target/` (guarded by a lock file), and the canonical JSON document is
//! rebuilt from staging on every append. Staging lines are tagged with the
//! commit; a new commit starts a fresh document, so one file is always one
//! coherent snapshot of one commit on one machine.
//!
//! The machine label comes from the HARDWARE (CPU brand + core count), never
//! from a flag or hostname, so the same machine always writes the same file
//! and results are diffable across time.

#![forbid(unsafe_code)]

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// A test row: `expected` is the pass criterion as a number when it is a
/// simple threshold or a string when it is a band ("0.35-0.70", ">= 1.90").
#[derive(Debug, Clone)]
pub struct TestResult {
    pub id: String,
    pub name: String,
    pub expected: Value,
    pub actual: Value,
    pub units: String,
    pub pass: bool,
}

/// A benchmark row.
#[derive(Debug, Clone)]
pub struct Benchmark {
    pub case: String,
    pub setting: String,
    pub cells: u64,
    pub steps_per_sec: f64,
    pub seconds_to_steady: f64,
}

/// Number or string, so thresholds and bands both fit.
#[derive(Debug, Clone)]
pub enum Value {
    Num(f64),
    Str(String),
}

impl From<f64> for Value {
    fn from(v: f64) -> Self { Value::Num(v) }
}
impl From<&str> for Value {
    fn from(v: &str) -> Self { Value::Str(v.to_string()) }
}
impl From<u64> for Value {
    fn from(v: u64) -> Self { Value::Num(v as f64) }
}

impl Value {
    fn to_json(&self) -> String {
        match self {
            Value::Num(v) => json_num(*v),
            Value::Str(s) => json_str(s),
        }
    }
}

/// Record one test row into `docs/results/<suite>-<machine>.json`.
pub fn record_test(suite: &str, t: TestResult) {
    let json = format!(
        "{{\"id\":{},\"name\":{},\"expected\":{},\"actual\":{},\"units\":{},\"pass\":{}}}",
        json_str(&t.id),
        json_str(&t.name),
        t.expected.to_json(),
        t.actual.to_json(),
        json_str(&t.units),
        t.pass
    );
    record(suite, "test", &t.id, &json);
}

/// Record one benchmark row.
pub fn record_benchmark(suite: &str, b: Benchmark) {
    // The dedup key must never contain the staging-line separator '|', or the
    // rebuilt document inherits a stray prefix and stops being JSON.
    let key = format!("{}::{}", b.case, b.setting).replace('|', "/");
    let json = format!(
        "{{\"case\":{},\"setting\":{},\"cells\":{},\"steps_per_sec\":{},\"seconds_to_steady\":{}}}",
        json_str(&b.case),
        json_str(&b.setting),
        b.cells,
        json_num(b.steps_per_sec),
        json_num(b.seconds_to_steady)
    );
    record(suite, "bench", &key, &json);
}

/// Record a free-text note (anything that needs a human).
pub fn record_note(suite: &str, key: &str, note: &str) {
    record(suite, "note", key, &json_str(note));
}

// ---------------------------------------------------------------------------

fn record(suite: &str, kind: &str, key: &str, json: &str) {
    // Never panic the calling test over result-file plumbing: report loudly
    // to stderr instead. The asserts that follow the record call are the
    // test's actual verdict.
    if let Err(e) = try_record(suite, kind, key, json) {
        eprintln!("cfd-results: could not record {suite}/{kind}/{key}: {e}");
    }
}

fn try_record(suite: &str, kind: &str, key: &str, json: &str) -> std::io::Result<()> {
    assert!(!key.contains('\n') && !json.contains('\n'), "records are line-oriented");
    assert!(!key.contains('|'), "'|' is the staging-line separator; keys must not contain it");
    let root = repo_root();
    let machine = machine_label();
    let commit = git_commit(&root);
    let staging_dir = root.join("target").join("results-staging");
    fs::create_dir_all(&staging_dir)?;
    let staging = staging_dir.join(format!("{suite}-{machine}.records"));
    let out_dir = root.join("docs").join("results");
    fs::create_dir_all(&out_dir)?;
    let out = out_dir.join(format!("{suite}-{machine}.json"));

    let _lock = FileLock::acquire(&staging_dir.join(format!("{suite}-{machine}.lock")))?;

    // Append the new record, keeping only lines from the current commit — a
    // result file is one coherent snapshot of one commit.
    let mut lines: Vec<String> = match fs::read_to_string(&staging) {
        Ok(s) => s
            .lines()
            .filter(|l| l.starts_with(&format!("{commit}|")))
            .map(|l| l.to_string())
            .collect(),
        Err(_) => Vec::new(),
    };
    let tag = format!("{commit}|{kind}|{key}|");
    lines.retain(|l| !l.starts_with(&tag)); // rerun of the same test replaces its row
    lines.push(format!("{tag}{json}"));
    fs::write(&staging, lines.join("\n") + "\n")?;

    // Rebuild the canonical document.
    let (mut tests, mut benches, mut notes) = (Vec::new(), Vec::new(), Vec::new());
    for l in &lines {
        let mut it = l.splitn(4, '|');
        let (_c, k, _key, body) = (
            it.next().unwrap_or(""),
            it.next().unwrap_or(""),
            it.next().unwrap_or(""),
            it.next().unwrap_or(""),
        );
        match k {
            "test" => tests.push(body.to_string()),
            "bench" => benches.push(body.to_string()),
            "note" => notes.push(body.to_string()),
            _ => {}
        }
    }
    let doc = format!
        ("{{\n  \"commit\": {},\n  \"timestamp\": {},\n  \"machine\": {},\n  \"tests\": [\n    {}\n  ],\n  \"benchmarks\": [\n    {}\n  ],\n  \"notes\": [\n    {}\n  ]\n}}\n",
        json_str(&commit),
        json_str(&iso8601_utc_now()),
        json_str(&machine),
        tests.join(",\n    "),
        benches.join(",\n    "),
        notes.join(",\n    "));
    // Empty arrays without a dangling blank element.
    let doc = doc
        .replace("[\n    \n  ]", "[]");
    fs::write(&out, doc)
}

/// Workspace root: this crate's manifest dir is `<root>/cfd-results`.
fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf()
}

fn git_commit(root: &Path) -> String {
    Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(root)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".into())
}

/// Hardware-derived machine label: CPU brand + logical core count, slugged.
/// "Apple M1" on 8 cores -> "apple-m1-8c"; an AMD EPYC 7B13 with 16 threads
/// -> "amd-epyc-7b13-16c". Never a flag, never a hostname.
pub fn machine_label() -> String {
    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0);
    let brand = cpu_brand().unwrap_or_else(|| std::env::consts::ARCH.to_string());
    let mut slug = String::new();
    for tok in brand.split_whitespace() {
        let t: String = tok
            .to_ascii_lowercase()
            .replace("(r)", "")
            .replace("(tm)", "")
            .replace("(c)", "")
            .chars()
            .filter(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '.')
            .collect();
        // Drop marketing filler and frequency tokens; keep the identity.
        if t.is_empty()
            || t == "processor"
            || t == "cpu"
            || t == "with"
            || t == "@"
            || t.ends_with("ghz")
            || t.ends_with("-core")
        {
            continue;
        }
        if !slug.is_empty() {
            slug.push('-');
        }
        slug.push_str(t.trim_matches('-'));
    }
    if slug.is_empty() {
        slug = std::env::consts::ARCH.to_string();
    }
    format!("{slug}-{cores}c")
}

fn cpu_brand() -> Option<String> {
    #[cfg(target_os = "linux")]
    {
        let info = fs::read_to_string("/proc/cpuinfo").ok()?;
        for line in info.lines() {
            if let Some(rest) = line.strip_prefix("model name") {
                return Some(rest.trim_start_matches([' ', '\t', ':']).trim().to_string());
            }
        }
        None
    }
    #[cfg(target_os = "macos")]
    {
        Command::new("sysctl")
            .args(["-n", "machdep.cpu.brand_string"])
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        None
    }
}

// ---- tiny formatting helpers ----------------------------------------------

fn json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn json_num(v: f64) -> String {
    if !v.is_finite() {
        return "null".into(); // JSON has no NaN/inf
    }
    if v == v.trunc() && v.abs() < 1e15 {
        format!("{}", v as i64)
    } else {
        format!("{v}")
    }
}

/// ISO 8601 UTC from the system clock, no external crates (Hinnant's
/// civil-from-days algorithm).
fn iso8601_utc_now() -> String {
    let secs = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();
    let (days, rem) = (secs / 86_400, secs % 86_400);
    let (h, m, s) = (rem / 3600, (rem % 3600) / 60, rem % 60);
    let z = days as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let mo = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if mo <= 2 { y + 1 } else { y };
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{m:02}:{s:02}Z")
}

/// Lock file via create_new; stale locks (crashed writer) expire after 30 s.
struct FileLock {
    path: PathBuf,
}

impl FileLock {
    fn acquire(path: &Path) -> std::io::Result<FileLock> {
        let deadline = SystemTime::now() + Duration::from_secs(20);
        loop {
            match fs::OpenOptions::new().write(true).create_new(true).open(path) {
                Ok(mut f) => {
                    let _ = writeln!(f, "{}", std::process::id());
                    return Ok(FileLock { path: path.to_path_buf() });
                }
                Err(_) => {
                    if let Ok(meta) = fs::metadata(path) {
                        if let Ok(age) = meta.modified().and_then(|t| {
                            SystemTime::now().duration_since(t).map_err(std::io::Error::other)
                        }) {
                            if age > Duration::from_secs(30) {
                                let _ = fs::remove_file(path);
                                continue;
                            }
                        }
                    }
                    if SystemTime::now() > deadline {
                        return Err(std::io::Error::other("results lock timeout"));
                    }
                    std::thread::sleep(Duration::from_millis(25));
                }
            }
        }
    }
}

impl Drop for FileLock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn machine_label_is_a_stable_slug() {
        let a = machine_label();
        let b = machine_label();
        assert_eq!(a, b, "label must be deterministic");
        assert!(a.ends_with('c'), "ends with core count: {a}");
        assert!(
            a.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '.'),
            "slug chars only: {a}"
        );
        assert!(!a.contains("processor") && !a.contains("ghz"), "{a}");
    }

    #[test]
    fn json_helpers_escape_and_format() {
        assert_eq!(json_str("a\"b\\c\n"), "\"a\\\"b\\\\c\\n\"");
        assert_eq!(json_num(2.0), "2");
        assert_eq!(json_num(2.5), "2.5");
        assert_eq!(json_num(f64::NAN), "null");
    }

    #[test]
    fn timestamp_is_iso8601() {
        let t = iso8601_utc_now();
        assert_eq!(t.len(), 20, "{t}");
        assert!(t.ends_with('Z') && t.contains('T'), "{t}");
        assert!(t.starts_with("20"), "{t}");
    }

    /// End-to-end: two records land in one coherent document; rerunning a
    /// test replaces its row instead of duplicating it.
    #[test]
    fn records_build_one_document() {
        let suite = "selftest";
        record_test(suite, TestResult {
            id: "X1".into(),
            name: "self test".into(),
            expected: 1.0.into(),
            actual: 0.5.into(),
            units: "widgets".into(),
            pass: true,
        });
        record_benchmark(suite, Benchmark {
            case: "demo".into(),
            setting: "graded | long".into(), // '|' must not corrupt the document
            cells: 42,
            steps_per_sec: 10.5,
            seconds_to_steady: 3.25,
        });
        record_note(suite, "n1", "a note");
        record_test(suite, TestResult {
            id: "X1".into(),
            name: "self test".into(),
            expected: 1.0.into(),
            actual: 0.75.into(),
            units: "widgets".into(),
            pass: true,
        });
        let path = repo_root()
            .join("docs/results")
            .join(format!("{suite}-{}.json", machine_label()));
        let doc = fs::read_to_string(&path).unwrap();
        // Structural sanity: balanced braces/brackets and no stray staging
        // separators outside strings (a leaked '|' prefix broke this once).
        assert_eq!(doc.matches('{').count(), doc.matches('}').count(), "{doc}");
        assert!(!doc.contains(")|{"), "staging prefix leaked into the document: {doc}");
        assert_eq!(doc.matches("\"X1\"").count(), 1, "rerun must replace, not append");
        assert!(doc.contains("\"actual\":0.75"), "{doc}");
        assert!(doc.contains("\"steps_per_sec\":10.5"), "{doc}");
        assert!(doc.contains("a note"), "{doc}");
        assert!(doc.contains("\"machine\""), "{doc}");
        // Clean up the self-test artifacts: only real suites get committed.
        let _ = fs::remove_file(&path);
        let _ = fs::remove_file(
            repo_root()
                .join("target/results-staging")
                .join(format!("{suite}-{}.records", machine_label())),
        );
    }
}

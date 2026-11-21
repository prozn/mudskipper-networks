//! Fits `EVAL_SCALE` to a shuffled bulletformat file.
//!
//! bullet's target is `wdl · result + (1 − wdl) · sigmoid(score / EVAL_SCALE)`. This
//! finds the K at which `sigmoid(score / K)` best predicts the game results, under the
//! same squared-error loss the trainer uses. At that K the two halves of the target
//! agree, so the WDL blend does not stretch or shrink the network's output scale.
//!
//! ```text
//! cargo run --release --example fit-eval-scale -- data/pilot.shuffled.bin [max-abs-score]
//! ```
//!
//! Each 32-byte record is bulletformat's `ChessBoard`: the side to move's score is an
//! `i16` at byte 24, and its result a `u8` at byte 26 (0 loss, 1 draw, 2 win). Positions
//! whose |score| exceeds `max-abs-score`, if given, are left out.

use std::{
    env,
    fs::File,
    io::{BufReader, Read},
};

/// Bytes per bulletformat `ChessBoard`.
const RECORD: usize = 32;

/// The range of K searched.
const K_RANGE: (f64, f64) = (10.0, 2000.0);

/// The scales the loss is printed at for comparison, besides the fitted ones.
const COMPARE: [f64; 6] = [100.0, 150.0, 200.0, 240.0, 300.0, 400.0];

/// Positions counted by score and result: `counts[score + 32768][result]`.
struct Histogram {
    counts: Vec<[u32; 3]>,
}

impl Histogram {
    fn new() -> Self {
        Self {
            counts: vec![[0; 3]; 1 << 16],
        }
    }

    fn add(&mut self, score: i16, result: u8) {
        let index = usize::from(score.cast_unsigned() ^ 0x8000);
        self.counts[index][usize::from(result)] += 1;
    }

    /// Each score with its counts of losses, draws and wins.
    fn bins(&self) -> impl Iterator<Item = (f64, &[u32; 3])> {
        (i16::MIN..=i16::MAX).map(f64::from).zip(&self.counts)
    }

    /// Positions in total, and the fraction that were losses, draws and wins.
    fn split(&self) -> (f64, [f64; 3]) {
        let mut totals = [0.0; 3];
        for (_, counts) in self.bins() {
            for (total, &count) in totals.iter_mut().zip(counts) {
                *total += f64::from(count);
            }
        }
        let n: f64 = totals.iter().sum();
        (n, totals.map(|total| total / n))
    }

    /// The mean of `loss(prediction, result)` with prediction `sigmoid(score / k)`.
    fn mean_loss(&self, k: f64, loss: impl Fn(f64, f64) -> f64) -> f64 {
        let (mut sum, mut n) = (0.0, 0.0);
        for (score, counts) in self.bins() {
            let prediction = sigmoid(score / k);
            for (result, &count) in [0.0, 0.5, 1.0].into_iter().zip(counts) {
                sum += f64::from(count) * loss(prediction, result);
                n += f64::from(count);
            }
        }
        sum / n
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// bullet's loss: `output.sigmoid().squared_error(target)`.
fn squared_error(prediction: f64, result: f64) -> f64 {
    (prediction - result).powi(2)
}

fn log_loss(prediction: f64, result: f64) -> f64 {
    let p = prediction.clamp(1e-12, 1.0 - 1e-12);
    -(result * p.ln() + (1.0 - result) * (1.0 - p).ln())
}

/// The K in `K_RANGE` minimising `loss`, by golden-section search.
fn minimise(loss: impl Fn(f64) -> f64) -> f64 {
    let ratio = (5f64.sqrt() - 1.0) / 2.0;
    let (mut lo, mut hi) = K_RANGE;
    while hi - lo > 0.01 {
        let a = hi - ratio * (hi - lo);
        let b = lo + ratio * (hi - lo);
        if loss(a) < loss(b) {
            hi = b;
        } else {
            lo = a;
        }
    }
    f64::midpoint(lo, hi)
}

/// Reads every record of `path` into a histogram, skipping scores beyond `cap`.
/// Returns the histogram and the number of records skipped.
fn read(path: &str, cap: Option<u16>) -> (Histogram, u64) {
    let file = File::open(path).unwrap_or_else(|e| panic!("cannot open {path}: {e}"));
    let mut reader = BufReader::with_capacity(1 << 24, file);
    let mut histogram = Histogram::new();
    let mut skipped = 0;
    let mut buffer = vec![0; RECORD * 65_536];
    loop {
        let mut filled = 0;
        while filled < buffer.len() {
            match reader.read(&mut buffer[filled..]) {
                Ok(0) => break,
                Ok(n) => filled += n,
                Err(e) => panic!("cannot read {path}: {e}"),
            }
        }
        assert!(
            filled.is_multiple_of(RECORD),
            "{path} is not a whole number of {RECORD}-byte records"
        );
        for record in buffer[..filled].chunks_exact(RECORD) {
            let score = i16::from_le_bytes([record[24], record[25]]);
            let result = record[26];
            assert!(result <= 2, "{path} has a record with result {result}");
            if cap.is_some_and(|cap| score.unsigned_abs() > cap) {
                skipped += 1;
            } else {
                histogram.add(score, result);
            }
        }
        if filled < buffer.len() {
            return (histogram, skipped);
        }
    }
}

fn main() {
    let mut args = env::args().skip(1);
    let path = args
        .next()
        .expect("usage: fit-eval-scale <file.bin> [max-abs-score]");
    let cap = args
        .next()
        .map(|cap| cap.parse().expect("max-abs-score is a number up to 32767"));

    let (histogram, skipped) = read(&path, cap);
    let (n, [loss, draw, win]) = histogram.split();
    println!("{n} positions used, {skipped} beyond the score cap");
    println!(
        "side to move: win {:.1}%  draw {:.1}%  loss {:.1}%",
        100.0 * win,
        100.0 * draw,
        100.0 * loss
    );

    let fitted = minimise(|k| histogram.mean_loss(k, squared_error));
    let fitted_log = minimise(|k| histogram.mean_loss(k, log_loss));
    println!("EVAL_SCALE fitted to bullet's loss (squared error): {fitted:.1}");
    println!("EVAL_SCALE fitted to log loss, for comparison:      {fitted_log:.1}");
    println!("squared error by scale:");
    for k in COMPARE.into_iter().chain([fitted.round()]) {
        println!("  {k:>6.0}  {:.6}", histogram.mean_loss(k, squared_error));
    }
}

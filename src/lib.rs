//! Trains mudskipper's networks with bullet.
//!
//! Every network has one binary in `src/bin/`, named after its release, holding its
//! [`Config`]. The architecture, quantisation and loss are shared and follow
//! bullet's `examples/progression/1_simple.rs`: `(768 → N)x2 → 1`, perspective, `SCReLU`.

use std::{fs, path::Path};

use bullet::{
    game::inputs::Chess768,
    nn::optimiser::AdamW,
    trainer::{
        save::SavedFormat,
        schedule::{TrainingSchedule, TrainingSteps, lr, wdl},
        settings::LocalSettings,
    },
    value::{ValueTrainerBuilder, loader::DirectSequentialDataLoader},
};

/// Positions per batch, as in `1_simple.rs`.
const BATCH_SIZE: usize = 16_384;

/// Maps the output to centipawns: the network learns `cp / EVAL_SCALE`, and mudskipper
/// multiplies by the same value (its `SCALE`). Changing it means changing the engine.
/// Fitted to the pilot data with `examples/fit-eval-scale.rs`; see the README.
const EVAL_SCALE: f32 = 143.0;

/// Inputs per perspective: 2 colours × 6 piece types × 64 squares.
const INPUTS: usize = 768;

/// Starting learning rate of the cosine decay, as in `1_simple.rs`.
const INITIAL_LR: f32 = 0.001;

/// Learning rate the cosine decay reaches at the last superbatch, as in `1_simple.rs`.
const FINAL_LR: f32 = 0.001 * 0.3 * 0.3 * 0.3 * 0.3 * 0.3;

/// Superbatches between checkpoints, as in `1_simple.rs`. The last one is always saved.
const SAVE_RATE: usize = 10;

/// Data loader threads and queued batches, as in `1_simple.rs`. The loader only reads
/// a pre-shuffled file, so these rarely limit throughput.
const LOADER_THREADS: usize = 2;
const BATCH_QUEUE_SIZE: usize = 32;

/// Quantisation of the feature transformer, mudskipper's `QA`.
const QA: i16 = 255;

/// Quantisation of the output weights, mudskipper's `QB`.
const QB: i16 = 64;

/// Where bullet writes `<net_id>-<superbatch>/` checkpoints.
const OUTPUT_DIRECTORY: &str = "checkpoints";

/// Positions evaluated after training, as a sanity check on sign and scale.
const CHECK_FENS: [&str; 3] = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    // White a queen up, with White then Black to move. Scores are relative to the side to
    // move, so these two should have opposite signs.
    "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1",
];

/// One network's training settings: everything that differs between rows of `HISTORY.md`.
#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    /// Release name, `<generation>-<hidden>[-label]`. Also bullet's `net_id`, so the
    /// final network lands in `checkpoints/<name>-<superbatches>/quantised.bin`.
    pub name: &'static str,
    /// Hidden layer size N. mudskipper requires a multiple of 32.
    pub hidden: usize,
    /// Shuffled bulletformat file, relative to the crate root.
    pub dataset: &'static str,
    /// Number of superbatches, which is also where the cosine decay ends.
    pub superbatches: usize,
    /// Batches per superbatch.
    pub batches_per_superbatch: usize,
    /// Weight of the game result in the target; the rest is the search score.
    pub wdl: f32,
}

impl Config {
    /// Trains the network, then checks the final `quantised.bin` and prints a few
    /// evaluations in centipawns.
    ///
    /// # Panics
    ///
    /// Panics if `hidden` is not a multiple of 32, if the dataset is missing, or if the
    /// final `quantised.bin` is missing or the wrong size (bullet skips writing it when
    /// quantisation overflows).
    pub fn train(&self) {
        assert!(
            self.hidden.is_multiple_of(32),
            "hidden size {} is not a multiple of 32",
            self.hidden
        );
        assert!(
            Path::new(self.dataset).is_file(),
            "dataset {} not found; see the README's data steps",
            self.dataset
        );

        let hidden = self.hidden;
        let mut trainer = ValueTrainerBuilder::default()
            .dual_perspective()
            .optimiser(AdamW)
            .inputs(Chess768)
            .save_format(&[
                SavedFormat::id("l0w").round().quantise::<i16>(QA),
                SavedFormat::id("l0b").round().quantise::<i16>(QA),
                SavedFormat::id("l1w").round().quantise::<i16>(QB),
                SavedFormat::id("l1b").round().quantise::<i16>(QA * QB),
            ])
            .loss_fn(|output, target| output.sigmoid().squared_error(target))
            .build(|builder, stm_inputs, ntm_inputs| {
                let l0 = builder.new_affine("l0", INPUTS, hidden);
                let l1 = builder.new_affine("l1", 2 * hidden, 1);

                let stm_hidden = l0.forward(stm_inputs).screlu();
                let ntm_hidden = l0.forward(ntm_inputs).screlu();
                l1.forward(stm_hidden.concat(ntm_hidden))
            });

        let schedule = TrainingSchedule {
            net_id: self.name.to_string(),
            eval_scale: EVAL_SCALE,
            steps: TrainingSteps {
                batch_size: BATCH_SIZE,
                batches_per_superbatch: self.batches_per_superbatch,
                start_superbatch: 1,
                end_superbatch: self.superbatches,
            },
            wdl_scheduler: wdl::ConstantWDL { value: self.wdl },
            lr_scheduler: lr::CosineDecayLR {
                initial_lr: INITIAL_LR,
                final_lr: FINAL_LR,
                final_superbatch: self.superbatches,
            },
            save_rate: SAVE_RATE,
        };

        let settings = LocalSettings {
            threads: LOADER_THREADS,
            test_set: None,
            output_directory: OUTPUT_DIRECTORY,
            batch_queue_size: BATCH_QUEUE_SIZE,
        };

        let dataloader = DirectSequentialDataLoader::new(&[self.dataset]);

        trainer.run(&schedule, &settings, &dataloader);

        let quantised = format!(
            "{OUTPUT_DIRECTORY}/{}-{}/quantised.bin",
            self.name, self.superbatches
        );
        let len = fs::metadata(&quantised)
            .map_or_else(|e| panic!("cannot read {quantised}: {e}"), |m| m.len());
        assert_eq!(len, quantised_len(hidden), "{quantised} has the wrong size");
        println!("{quantised}: {len} bytes");

        for fen in CHECK_FENS {
            let cp = trainer.eval(fen) * EVAL_SCALE;
            println!("{cp:>8.0} cp  {fen}");
        }
    }
}

/// Size in bytes of the `quantised.bin` bullet writes for hidden size `hidden`: the
/// `l0w`, `l0b`, `l1w` and `l1b` values as `i16`, padded to a multiple of 64 bytes.
#[must_use]
pub const fn quantised_len(hidden: usize) -> u64 {
    let values = INPUTS * hidden + hidden + 2 * hidden + 1;
    // `usize` to `u64` never truncates on the 32- and 64-bit targets bullet supports.
    (values * 2).next_multiple_of(64) as u64
}

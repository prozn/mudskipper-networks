//! Generation 1: 100M positions, about 40 passes.

use training::Config;

/// The blend of the pilot run that won its SPRT. Unset until the pilot is decided (#81).
const PILOT_WINNER_WDL: Option<f32> = None;

fn main() {
    let wdl = PILOT_WINNER_WDL.expect("set PILOT_WINNER_WDL to the winning pilot's blend first");
    Config {
        name: "gen1-256",
        hidden: 256,
        dataset: "data/gen1.shuffled.bin",
        superbatches: 40,
        batches_per_superbatch: 6104,
        wdl,
    }
    .train();
}

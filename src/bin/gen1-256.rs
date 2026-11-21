//! Generation 1: 100M positions, about 40 passes. WDL blend 0.40, from the pilot that
//! won its SPRT: pilot-128-wdl40 gained +394.88 against `PeSTO`, pilot-128-wdl75 +213.66.

use training::Config;

fn main() {
    Config {
        name: "gen1-256",
        hidden: 256,
        dataset: "data/gen1.shuffled.bin",
        superbatches: 40,
        batches_per_superbatch: 6104,
        wdl: 0.40,
    }
    .train();
}

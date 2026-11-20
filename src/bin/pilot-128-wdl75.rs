//! Pilot network, WDL blend 0.75: 15–20M positions, about one pass per superbatch.

use training::Config;

fn main() {
    Config {
        name: "pilot-128-wdl75",
        hidden: 128,
        dataset: "data/pilot.shuffled.bin",
        superbatches: 20,
        batches_per_superbatch: 1024,
        wdl: 0.75,
    }
    .train();
}

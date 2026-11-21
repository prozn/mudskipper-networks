# mudskipper-networks

Training for the [mudskipper](https://github.com/prozn/mudskipper) chess engine's networks, and the networks themselves, published as release assets.

Each network is a `(768 → N)x2 → 1` perspective network with SCReLU, trained with [bullet](https://github.com/jw1912/bullet) pinned to `5d70177680437b7c0ddf958bd4722d95afa78b0c`. `HISTORY.md` records where every network came from.

## Layout

| Path | What it is |
|---|---|
| `src/lib.rs` | The shared architecture, quantisation and schedule, from bullet's `examples/progression/1_simple.rs` |
| `src/bin/<name>.rs` | One config per network, named after its release |
| `examples/fit-eval-scale.rs` | Fits `EVAL_SCALE` to a shuffled data file |
| `filter.toml` | The position filter for `splat` |
| `HISTORY.md` | One row per network |

## Releases

A network's release is named `<generation>-<hidden>[-label]`, for example `pilot-128-wdl75` or `gen1-256`. The name is also the binary's name and bullet's `net_id`. The release asset is the final `quantised.bin` bullet writes, at `checkpoints/<name>-<superbatches>/quantised.bin`. mudskipper names the network it builds with in its `network.txt`.

## Build

Training needs an NVIDIA GPU and the `cuda` feature. Without a GPU feature bullet uses a mock device and cannot train.

1. Install the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit), as recent a version as possible.
2. Set `CUDA_PATH` to the install directory, the one that contains `bin`, `lib` and `include`. bullet's build script panics without it.
3. Build with `cargo build --release --features cuda`.

`cargo build` without the feature still compiles the configs, which is what CI checks.

The data tools need no GPU. Install them from the same bullet commit:

```text
cargo install --locked --git https://github.com/jw1912/bullet --rev 5d70177680437b7c0ddf958bd4722d95afa78b0c bullet-utils
```

## Data

mudskipper's `datagen` command (built with `--features datagen`) writes one viriformat file per thread:

```text
mudskipper datagen <threads> <games-or-0> out/
```

Turn those files into a shuffled bulletformat file. Every step streams or caps its memory, so 100M+ positions fit in 32 GB of RAM. The example is for Generation 1; name the files after the generation being built, so the pilot's output is `data/pilot.shuffled.bin`, the path its configs read:

```text
bullet-utils viribinpack interleave out/*.vf -o data/gen1.vf
bullet-utils viribinpack splat data/gen1.vf data/gen1.bin filter.toml
bullet-utils validate --input data/gen1.bin
bullet-utils shuffle -i data/gen1.bin -o data/gen1.shuffled.bin -m 8192
```

- **`interleave`** streams, so its memory use is constant. It mixes the files a game at a time, and needs at least two of them: with a single file, use it as `data/gen1.vf` directly. On Windows, list the files by hand: `cmd` and PowerShell do not expand `out/*.vf`.
- **`splat`** keeps only the positions that pass `filter.toml`. Watch for two things:
  - **A wrong filter path silently disables filtering.** If the file cannot be read or parsed, `splat` falls back to no filter at all and says nothing. Run it from this repo's root so `filter.toml` resolves.
  - **The "Positions" count it prints ignores the filter.** It counts every move. The number of positions kept is the size of `data/gen1.bin` divided by 32, and it should be well below the printed count.
- **`validate`** checks king squares and pieces, and prints the win/draw/loss split. Read the split before training.
- **`shuffle` is required.** The trainer reads the file front to back and never shuffles, and `splat` writes positions in game order. With `-m 8192` peak RAM stays at or under 8 GiB: smaller files are shuffled in memory, and larger ones in parts that are merged. Free disk space must be twice the size of the `.bin`. The output must not be the input, and `<output>.tmp/` must not already exist.

**Later generations.** Shuffle each generation's `.bin` on its own as above, then interleave the shuffled files. This `interleave` is the bulletformat one, not `viribinpack interleave`:

```text
bullet-utils interleave data/gen1.shuffled.bin data/gen2.shuffled.bin -o data/gen2-all.shuffled.bin
```

## Fit the eval scale

`EVAL_SCALE` in `src/lib.rs` maps the network's output to centipawns. It should be the K at which `sigmoid(score / K)` best predicts the game results in the training data. At that K the score and result halves of the target agree, and the WDL blend leaves the output scale alone. With a K that is too large, a high WDL blend inflates every evaluation. Fit it on the shuffled file before training:

```text
cargo run --release --example fit-eval-scale -- data/pilot.shuffled.bin
```

It reads the score and result of every position and prints the win/draw/loss split. Then it prints the K that minimises bullet's squared-error loss, the log-loss K for comparison, and the loss at a few scales. It takes under a second for 20M positions. An optional second argument leaves out positions whose |score| exceeds it, which shows whether extreme scores move the fit. On the pilot data the fitted K stayed at 143 with any cap.

Use the squared-error K, rounded, as `EVAL_SCALE`. mudskipper's `SCALE` must then be set to the same value, so a change here is also a change to the engine. Refit when the data changes a lot, for example when a new generation's search scores the positions.

## Train

Check that the shuffled file is at the path the network's config names (`data/` is git-ignored), then run its binary:

```text
cargo run --release --features cuda --bin pilot-128-wdl75
```

A checkpoint lands in `checkpoints/<name>-<superbatch>/` every 10 superbatches and at the end, and `log.txt` in each one records the loss. bullet has no validation loss, so before an SPRT check that:

- `validate`'s win/draw/loss split looks sensible.
- The loss in `log.txt` falls and flattens as the learning rate decays.
- The scale is right. After training, the binary checks the size of `quantised.bin` and prints its evaluation of a few positions in centipawns. The start position should be near zero, and a queen up should be several hundred centipawns.

Then publish the release, add the network's row to `HISTORY.md`, and fill in its SPRT result once the SPRT finishes.

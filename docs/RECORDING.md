# Recording `docs/demo.gif`

**This is a reconstruction, not the original procedure.** The procedure that
produced `docs/demo.gif` was never written down. What follows separates what is
*measured* from the committed artifact, what is *inferred* from surrounding
evidence, and what is simply *unknown* — so the next person regenerating the GIF
knows exactly how much they are guessing.

Nothing here should be treated as "how it was done" until someone who has the
original setup confirms or replaces it.

## Status

The GIF is **out of date**. It was recorded before the stop-gate's audit sink was
wired up, so it shows 9 events; `loopward-demo` now emits 10 — the extra line is

```text
  [gate] phase 'verify' [auto] -> approve (auto-approved)
```

which appears between `review: ok via strategy 'structured'` and `verify: start`.
No other line changed.

The README caption that used to assert the GIF showed real `loopward-demo`
output was **deliberately withdrawn**: the claim stopped being true once the
demo changed, and a dated promise to regenerate ages badly in a README. The
caption there now reads only "Illustrative" and points here. This file is the
single source of record for that debt — restore a caption asserting the GIF
matches real output only once the GIF has actually been re-recorded.

## Measured from the artifact

Read directly out of the GIF byte stream (header, logical screen descriptor, and
a walk of the block chain), so these are re-derivable from the committed file:

| Property | Value |
| --- | --- |
| Format | `GIF89a` |
| Size on disk | 599,538 bytes |
| Dimensions | 1533 × 646 px |
| Frames | 34 |
| Total duration | 13.14 s (sum of the graphic-control delays) |
| Global colour table | present, 256 colours |
| Application block | `NETSCAPE2.0` (the loop extension) |
| Comment block | **none** — no encoder signed the file |

The average of ~2.6 frames per second is the *result* of the pipeline, not an
input to it: it is consistent both with a low-fps capture and with frame
deduplication of a higher-fps one. It does not identify which.

## Inferred pipeline

> Inference from two pieces of in-repo evidence. Not confirmed.

1. A screen capture of a terminal running `loopward-demo`, saved as `demo.mp4`.
2. `ffmpeg` `palettegen` → `palette.png`, then `paletteuse` → `docs/demo.gif`.

The evidence:

- [`.gitignore`](../.gitignore) excludes `demo.mp4` and `palette.png` under the
  comment `# demo recording intermediates (docs/demo.gif IS committed)`. Those
  two filenames are the canonical intermediates of an ffmpeg two-pass palette
  workflow.
- [`loopward/demo.py`](../loopward/demo.py) documents the `--demo-delay` flag as
  `0.0 = no delay (fast for CI). Try 0.4 for screen recording.`, and
  `console()` defaults to `0.4` — "paced for screen recording". So the capture
  was of a real terminal in real time, not a headless render.
- The 256-colour global table and the `NETSCAPE2.0` loop block are what
  `paletteuse` emits by default. They are consistent with the inference but do
  not prove it: several encoders produce the same shape.
- 1533 × 646 is not a standard terminal grid at any common font size. It reads
  as a hand-cropped window capture, which is why the geometry cannot be
  recovered by calculation.

## Known: the content side

This part *is* documented, in code, and is deterministic:

```bash
loopward-demo          # console entry point, delay 0.4, paced for recording
python demo/run_demo.py # same demo, delay 0.0, fast for CI
```

The demo runs on the offline `fake` provider with a scripted reviewer
(`loopward/demo.py`), so the console output is byte-for-byte reproducible on any
machine. Only the *presentation* — font, colours, window — is unrecoverable.

**What the script is scripted to do: fail.** `scripted_reviewer` returns
unparseable prose for the first strategy whatever the prompt says — it routes on
`task` and never reads the prompt. So the three failed attempts and the
class-jump in the GIF are authored inputs, not a model that refused. That is
deliberate, and it is what makes the recording reproducible; but a reader
watching the GIF is watching the mechanism fire on a fixture, and nothing in the
frames says so. The same applies to `verify: confirmed 2, rejected 0`: the fake
confirms everything by construction, so the GIF has never shown — and on this
script cannot show — a finding being rejected.

Re-recording will not change any of that on its own. Replacing the scripted
failure with a recorded transcript of a real model is a separate decision, and
an open one.

## Unknown — what still needs documenting

Whoever regenerates the GIF should record these, because none of them can be
recovered from the artifact:

- **Screen recorder** used to produce `demo.mp4`. Nothing in the repo names it.
- **Terminal emulator** and its window chrome (the capture appears cropped).
- **Font family and size.**
- **Colour theme** — foreground, background, and the palette the ANSI-free
  output was rendered against.
- **Capture geometry** — how 1533 × 646 was arrived at.
- **Capture frame rate**, and whether frames were deduplicated afterwards.
- **`palettegen` flags**, in particular `stats_mode` (`full` vs `diff`).
- **`paletteuse` flags**, in particular the `dither` mode.

Until those are filled in, a regenerated GIF will differ visibly from the one in
`README.md` even if the terminal output it shows is identical.

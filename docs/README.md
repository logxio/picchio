# Replay a local AI run

Open `index.html` directly. Four sanitized recorded runs are preloaded in `recorded_runs.js`. D3 v7.9.0 and d3-sankey 0.12.3 live in `vendor/`, so the page makes no network requests and needs no build step.

One screen holds three figures, all drawn from the receipts in the browser:

A. The four runs side by side: GPU power and busy samples over time, Metal layers, prefill and decode speed, first token latency, peak RSS against the 4 GiB budget, context nonces at the engine, the five boundary probes and the three independent checks.

B. The process tree flowing into the Seatbelt policy: allowed loopback and workspace writes, the five denied probes per run, then the agent claim splitting into workspace tests, frozen tests and the SQLite total.

C. Fourteen local AI tools and this workbench across six audit points, graded from official docs and repositories read on 18 November 2026. Hover a glyph for its meaning and a tool name for the evidence note.

Click a run header or any flow to highlight that run across all three figures. Open receipt imports a local receipt and adds it as a new column. 2 Min Tour opens the 98 second recorded walkthrough across 28 moving shots. A warm ambient AAC track accompanies the video at about minus 18 LUFS, with a 1.2 second opening fade and a 2.2 second closing fade. Clicking 2 Min Tour plays with sound. Opening the tour link starts muted playback with the native sound control available. Its video clock selects the same run on the live canvas. Opening index.html#tour starts playback. Chapter buttons seek the video and canvas together. The closing sequence shows the tool matrix and three palettes.

`app.js` writes receipt strings as SVG text nodes. Imported model text is never injected as HTML. Save receipt creates a local JSON Blob for the highlighted run.

From the repository root, run the command shown in the bottom bar to verify all four captured receipts and exercise live host boundary and frozen SQLite checks. Python 3.11 or newer is sufficient. See `../runner/README.md` for model execution.

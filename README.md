# picchio

Picchio is Italian for woodpecker. Woodpeckers find hollow wood by knocking
on it and listening. This is a single Python file that knocks on your local
llama.cpp setup and listens for the two most common hollow spots: tok/s
numbers that do not mean what you think they mean, and a GPU that quietly
did nothing while the CPU did all the work.

```
python3 picchio.py /path/to/model.gguf
```

No pip, no dependencies, no config. If you have python3 and llama.cpp on
your machine, you already have everything it needs. It runs your model
twice with a fixed prompt, parses the engine's own logs, and prints a
verdict block you can paste into an issue or a comment.

## Why I wrote this

Last week I had proof that my app was slowing local models down by a
factor of three. Bare llama.cpp gave me 36 tok/s. The same model inside
the app gave 11.5. Same machine, same day, case closed.

Then I reran both sides properly: same binary, same parameters, a 32 cell
matrix across CPU and GPU, cold and warm. The 36 never reproduced. Not in
one cell. The number I had built a theory on was a rate from a different
lane, most likely prefill or a wall clock reading from some other run,
remembered as if it were generation speed. I never wrote down which lane
it came from, so it got to mean whatever my theory needed it to mean.

The real slowdown was somewhere else entirely. On some runs the engine
put every layer on the CPU without saying anything at the level you
normally look at. Generation speed barely moved, which is what makes this
failure mode invisible. Time to first token on a long prompt is what
explodes: about 5 seconds on the GPU became about 50 on the CPU for a
2.5k token prompt, measured on the same machine during that
investigation.

So the app was not 3x slower. My benchmark was lying, and separately, the
GPU was sometimes not working at all. Two different bugs, both mine, both
invisible in a single tok/s number. picchio is that week of debugging
folded into one file you can run in a minute.

## What it prints

Real output from the machine this was built on, unedited
([examples/healthy-metal.txt](examples/healthy-metal.txt)):

```
picchio v0.1.0 .................................. 2026-07-11 03:00
machine   Apple M5, 32 GB ram, macOS 26.5.1
engine    llama.cpp build 9430 (d48a56eff), 4 of 10 cpu threads
model     Qwen3.5-9B-Q4_K_M.gguf, 8.95 B params, 5.28 GiB on disk
gpu       ENGAGED: 33/33 layers on GPU (Metal: Apple M5)

                   prefill          decode       wallclock
  pass 1       591.9 tok/s      19.6 tok/s      12.5 tok/s
  pass 2       596.0 tok/s      20.9 tok/s      14.7 tok/s

where pass 1 went (10.1 s wall)
  load weights    1.7 s  #####.......................   17%
  prefill         1.3 s  ####........................   13%
  decode          6.5 s  ##################..........   64%
  engine misc     0.6 s  ##..........................    6%

VERDICT: HEALTHY
  The GPU did the work. Quote the decode number (20.9 tok/s)
  when you compare setups. 596 tok/s is real too, but that is
  prefill: prompt reading speed, not generation speed.
==================================================================
```

## The three numbers

Every tok/s figure belongs to one of three lanes, and picchio never adds
them together or averages them.

Prefill is how fast the model reads your prompt. Decode is how fast it
writes the answer. Wallclock is generated tokens divided by everything,
load and warmup included, which is what your stopwatch and your gut
measure. On the machine above these are 596, 21 and 15 in the same run.
On the CPU run below they are 25, 11 and 3. A single unlabeled number
spanning a 30x range is not a measurement, it is a rumor.

When a screenshot shows a Mac doing 500 tok/s, that is almost always
prefill. When llama-bench prints tg128, that is decode. When an app feels
slow before the first word appears, that is cold load plus prefill, and
no decode number will explain it.

## The hollow spot: silent CPU fallback

Same machine, same model, same file, forced to CPU
([examples/cpu-fallback.txt](examples/cpu-fallback.txt)):

```
gpu       NOT ENGAGED: 0/33 layers on GPU

                   prefill          decode       wallclock
  pass 1        23.0 tok/s      10.8 tok/s       2.7 tok/s
  pass 2        24.7 tok/s      10.7 tok/s       2.8 tok/s

where pass 1 went (47.9 s wall)
  load weights    2.1 s  #...........................    4%
  prefill        33.1 s  ###################.........   69%
  decode         11.8 s  #######.....................   25%
  engine misc     0.9 s  #...........................    2%

VERDICT: SILENT CPU FALLBACK
  The engine loaded, answered, and never used the GPU: 0 of 33
  layers offloaded, no GPU device initialized. Decode looks
  almost normal (10.7 tok/s), which is why nobody notices.
  Prefill gives it away: at 25 tok/s, a 2500 token prompt sits
  101 s before the first word appears. Check -ngl and your build
  flags.
==================================================================
```

Look at what moved and what did not. Decode dropped 2x, from 21 to 11.
In a chat you might shrug at that. Prefill dropped 24x, from 596 to 25,
and the first word of a long prompt now takes minutes. picchio calls
this from two directions at once: the engine's own layer placement log
(0/33 offloaded) and the prefill signature. You can reproduce this
verdict on any Apple Silicon machine with:

```
python3 picchio.py model.gguf -- --device none -ngl 0
```

Anything after the bare `--` goes straight to the engine binary.

## The number you saw somewhere

The third thing picchio does is interrogate a number for you. Someone
posts a tok/s figure, or you remember one, and you want to know what it
probably was:

```
python3 picchio.py model.gguf --explain 36
```

```
YOUR NUMBER: 36.0 tok/s -> MATCHES NOTHING MEASURED HERE
  36.0 tok/s is not within 30% of anything measured here
  (closest: decode, off by 2.0x; measured: prefill 567.9, decode
  18.4, wallclock 13.1 tok/s). Before trusting that number, ask
  which of the three rates it was, and on what hardware, quant,
  and context length.
```

That 36 is the exact number from the story above, asked against the
machine it supposedly came from. After a diagnostic run picchio caches
the rates, so later you can call `--explain` alone without rerunning.

## Is this not just llama-bench?

llama-bench is good and you should use it. It answers a different
question. It tells you how fast this machine can run this model: separate
pp and tg rates, steady state, warmup on by default. picchio tells you
what actually happened on a real run and why it felt the way it felt.

Concretely, measured on this machine, same model, same day:

| tool, config              | prompt side   | generation side | notes                     |
|---------------------------|---------------|-----------------|---------------------------|
| llama-bench, default      | pp256: 610.13 | tg64: 20.87     | backend column: BLAS,MTL  |
| llama-bench, -ngl 0 (CPU) | pp128: 30.66  | tg32: 13.25     | backend column: BLAS,MTL  |

Both rows report the same backend, because that column describes what
the binary was compiled with, not where your tokens were computed. The
20x prompt side collapse is the only visible trace of the CPU run, and
you can only read it if you already know the healthy baseline. There is
also no load time, no cold and warm split, and no interpretation; that
last part is fair, a benchmark is not supposed to have opinions.

picchio exists for the layer under the numbers: was the GPU engaged
(with the engine's own placement log as evidence), where did the first
ten seconds go, and which lane does a given number belong to. As for
`ollama ps`, it shows where the weights sit for a loaded model, which is
placement but not speed, and I have not wrapped ollama at all in v0.1;
picchio drives the llama.cpp binaries directly.

## Measured on this machine

Apple M5, 32 GB, macOS 26.5.1, llama.cpp build 9430, Qwen3.5-9B Q4_K_M,
4 of 10 cpu threads, roughly 730 prompt tokens and 128 generated tokens
per pass. Ranges are min to max across the recorded runs in
[examples/](examples/). Every number here came out of a real run on
2026-07-11; there are no projections in this table.

| config              | prefill tok/s | decode tok/s | wallclock tok/s |
|---------------------|---------------|--------------|-----------------|
| Metal, 33/33 layers | 559.5 - 605.0 | 18.4 - 21.0  | 12.5 - 14.7     |
| CPU, 0/33 layers    | 23.0 - 27.2   | 9.9 - 10.8   | 2.7 - 3.0       |

Load time for the 5.28 GiB file: 3.3 s the first time it was ever read,
1.7 s after a cache flush, 0.4 s when the weights were still in the disk
cache. picchio prints a note when your pass 1 was not a true cold start,
because a cached load will flatter your first token time.

## Verdicts from other machines

This tool has been run on exactly one computer. That is the weakest
thing about it, and you can fix it in two minutes: run picchio, open an
issue with the title `verdict: <chip> <model>`, and paste the block.
Numbers land here only after someone measured them.

| chip     | ram   | model             | prefill | decode | wallclock | verdict | source                |
|----------|-------|-------------------|---------|--------|-----------|---------|-----------------------|
| Apple M5 | 32 GB | Qwen3.5-9B Q4_K_M | 596.0   | 20.9   | 14.7      | HEALTHY | examples/ (this repo) |

## What it does not do yet

The tested path is Apple Silicon plus llama.cpp, sample size one
machine. Linux parsing (CUDA and Vulkan log lines, /proc hardware info)
is written but has not touched real hardware yet; if you run it there, I
want the verdict block either way. It wraps llama-completion and
llama-cli, so llama-server, ollama, MLX and LM Studio are out of frame
for now. Old llama.cpp builds are handled with a flag fallback ladder
and the engine's log format has been stable for a long time, but very
old builds may only get partial evidence, and picchio will say so rather
than guess. Both passes run back to back, so pass 1 is only a true cold
start if the model was not recently loaded; when the load times give
that away, the verdict says so.

Exit codes, for scripting: 0 healthy or no evidence, 2 could not run,
3 partial offload, 4 silent CPU fallback.

## License

MIT.

<!-- TODO: footer product link pending publisher identity decision -->

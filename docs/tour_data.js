window.WORKBENCH_TOUR = {
  "duration": 105,
  "chapters": [
    {
      "start": 0,
      "selection": null,
      "label": "Overview",
      "palette": 0,
      "focus": "all"
    },
    {
      "start": 13,
      "selection": 0,
      "label": "GPU 86.1 tok/s",
      "palette": 0,
      "focus": "all"
    },
    {
      "start": 25,
      "selection": 1,
      "label": "CPU 35.3 tok/s",
      "palette": 0,
      "focus": "all"
    },
    {
      "start": 35,
      "selection": 2,
      "label": "Seatbelt 5/5 EPERM",
      "palette": 0,
      "focus": "all"
    },
    {
      "start": 61,
      "selection": 3,
      "label": "SQLite 13 vs 42",
      "palette": 0,
      "focus": "all"
    },
    {
      "start": 88,
      "selection": null,
      "label": "14 tool matrix",
      "palette": 0,
      "focus": "matrix"
    },
    {
      "start": 94,
      "selection": null,
      "label": "Export & Inspect AI",
      "palette": 0,
      "focus": "matrix"
    }
  ],
  "captions": [
    {
      "start": 0,
      "end": 3.008,
      "text": "Everyone tells communities to run small AI models locally for privacy."
    },
    {
      "start": 3.008,
      "end": 9.023,
      "text": "But today, turning on GPU speed in local tools silently disables sandboxes, and small models can pass checks by deleting the rules."
    },
    {
      "start": 9.023,
      "end": 13.125,
      "text": "We built Picchio so you can see and lock what your local AI actually did."
    },
    {
      "start": 13.125,
      "end": 24.682,
      "text": "Here on an Apple M5 laptop, a 1.5B local model fixes a community ledger inside a 1.19 GiB memory footprint at 86.1 tokens per second on Metal GPU."
    },
    {
      "start": 24.682,
      "end": 35.0,
      "text": "Column 2 catches a silent CPU fallback and middle context loss on the exact same task, where first token latency jumps 7.6x to 3,620 milliseconds."
    },
    {
      "start": 35,
      "end": 52.679,
      "text": "In Run 3, while Metal GPU inference runs at full speed, a tool child process tries five escapes: reading private files outside the folder, overwriting frozen tests, and opening an external network socket."
    },
    {
      "start": 52.679,
      "end": 61.25,
      "text": "Our inherited OS Seatbelt wall blocks all five at the kernel level with zero bytes leaked."
    },
    {
      "start": 61.25,
      "end": 73.828,
      "text": "In Run 4, the local agent claims success and its workspace tests turn green because it deleted an assertion and injected a skip."
    },
    {
      "start": 73.828,
      "end": 87.5,
      "text": "Picchio replays frozen original tests outside the agent's reach and checks SQLite directly, catching that the ledger is still stuck at 13 instead of 42."
    },
    {
      "start": 87.5,
      "end": 95.221,
      "text": "Everything exports to Mozilla LocalScore and Inspect AI in a 42 KB MIT licensed receipt."
    },
    {
      "start": 95.221,
      "end": 105.0,
      "text": "With Mozilla's support, we will bring verifiable local AI boundaries to every community maintainer on macOS, Linux, and Windows."
    }
  ]
};

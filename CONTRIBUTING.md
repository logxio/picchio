# Contributing to Picchio

A run Picchio gets wrong is useful. So is a run on hardware missing from the [measured examples](examples/).

## Report a run

From a clone, reproduce with a GGUF file, an Ollama tag or a running llama-server URL:

```sh
python3 picchio.py MODEL
python3 picchio.py --version
```

Paste the complete result into the [result report](https://github.com/logxio/picchio/issues/new?template=verdict-report.md). If Picchio disagrees with the engine or your OS, use the [misdiagnosis report](https://github.com/logxio/picchio/issues/new?template=misdiagnosis-report.md). Include the command, whether you ran v1.0.0 or current `main`, what you expected, what happened, the engine build or Ollama version, OS, GPU, model and quantization. An engine log, `ollama ps`, `nvidia-smi` or another repeatable reading helps show the disagreement. Include the exit code if Picchio stopped with an error.

For a case the result cannot explain, `python3 picchio.py MODEL --keep-logs logs` saves the engine output. Check every file before attaching it. Remove usernames, hostnames, local paths, IP addresses, access tokens, private prompts and other personal data; keep the relevant timing, settings and GPU placement lines. Never post an unreviewed log bundle.

## Send a fix

Open an issue first when the expected result is unclear. For a focused change, fork the repo, make a branch and open a pull request against `main`. Link the report or describe the exact input and output that led to the change. Add a small reproducible case when behavior changes, and say what you ran to check the fix.

For behavior changes, run the source selftest before opening the PR:

```sh
python3 picchio.py --selftest
```

For source changes, rebuild the single-file release artifact and test it too:

```sh
python3 scripts/build_zipapp.py
python3 public/picchio.pyz --selftest
```

For docs-only changes, check the links and run `git diff --check`. The [selftest workflow](https://github.com/logxio/picchio/actions/workflows/selftest.yml) runs the source and single-file selftests on macOS, Linux and Windows with Python 3.9 and 3.13 for every pull request. Review any new example or test data for the same private details before committing it.

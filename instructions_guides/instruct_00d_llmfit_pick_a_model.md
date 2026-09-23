# Guide 00d: `llmfit`, which models fit your machine?

A model that fits in memory runs; one that does not either fails or crawls. `llmfit`
reads your CPU, RAM, GPU and VRAM and scores hundreds of open models on **fit**,
**speed**, **quality** and **context**. It is a single command-line program written in
Rust and published on PyPI, so `uvx` runs it without installing anything into the
project. Source: [GitHub](https://github.com/AlexsJones/llmfit); website and
documentation: [llmfit.axjns.dev](https://llmfit.axjns.dev/).

```bash
uvx llmfit doctor        # what llmfit detected about your hardware
uvx llmfit recommend     # the models it recommends for this machine
uvx llmfit fit           # every model in its catalog, ranked by fit
uvx llmfit info "qwen3:1.7b"   # one model in detail
```

`uvx` downloads the tool into a cache the first time (a few seconds) and reuses it
later. Without arguments, `uvx llmfit` opens an interactive browser; press `q` to leave.

## Reading the output

- **Fit**: does the model's memory need (weights plus context) fit your RAM or VRAM?
- **Speed**: an estimate of tokens per second on your hardware.
- **Quality**: a rough quality score for the model size.
- **Context**: how long a prompt the model can hold.

The recommendation is about *hardware*. Lecture 1b adds the other half: the business
requirements (task, languages, license, context, privacy) that decide which of the
models that fit you should actually use.

## Choosing your tier

Every notebook starts with a configuration cell:

```python
TIER = "cpu"      # 8-16 GB RAM, no GPU (default)
# TIER = "gpu"    # >= 8 GB VRAM: bigger, better answers
```

Leave `"cpu"` unless `llmfit recommend` lists `qwen3:8b` as fitting comfortably on a
GPU.

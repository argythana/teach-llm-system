# Lecture 01: Your laptop as an LLM server, choosing a model, prompts as code

Two sessions. Session 1: `lec_01a` and `lec_01b`. Session 2: `lec_01c` and `lec_01d`.

## Learning Goals

### Required

- Set up the course environment with `uv sync` and run the notebooks in VS Code with
  that environment as the kernel; explain why `pip install` is not used in a `uv`
  project. *(guide `01a_git_uv`)* <!-- G1 -->
- Run a local model through Ollama from Python, read tokens in, tokens out and tokens
  per second from the response, and tell whether the model runs on CPU or GPU. *(lec_01a
  §2)* <!-- G2 -->
- Explain what a token is, count tokens with a tokenizer, and say why Greek text costs
  more tokens than English. *(lec_01a §3)* <!-- G3 -->
- Explain the context window and temperature and choose their values for a given use.
  *(lec_01a §4-5)* <!-- G4 -->
- Write a requirements table (task, languages, license, context, latency, hardware,
  privacy, maintenance) before choosing a model, and check candidates against it.
  *(lec_01b §1, §3, §6)* <!-- G5 -->
- Search the Hugging Face Hub from Python, read a model card's license and language
  metadata, and read the quantization variants and sizes of a GGUF repository. *(lec_01b
  §2-4)* <!-- G6 -->
- Use `llmfit` to check hardware fit and Ollama to pull a model from the Hub; write a
  short decision record. *(lec_01b §5-8)* <!-- G7 -->
- Write a prompt as a versioned Python function, use the system, user and assistant
  roles, and resend history for a multi-turn conversation. *(lec_01c §1-2)* <!-- G8 -->
- Use few-shot examples and JSON-schema structured output to get answers a program can
  parse, load them into a DataFrame, and explain what a schema enforces (the shape) and
  what it does not (the content, or rules only the prompt can carry). *(lec_01c §3-5)*
      <!-- G9 -->
- Build a LangChain chain (prompt template, chat model, output parser), read the prompt
  it renders, run it with `invoke` and `batch`, and use structured output inside it;
  explain what the framework adds over the plain client, what it costs, and when the
  plain client is enough, and why this course chose LangChain. *(lec_01d §1-3, §5-6)*
      <!-- G10 -->

### Optional / Career track

- Explain the problem `transformers` solves (one interface that turns a model's
  published files into a working network), run a small model in-process with it, and
  explain what Ollama does for you (weights, chat template, memory). *(lec_01e)*
  <!-- O1 -->
- Define structured output with a pydantic model, and swap the local model for a hosted
  OpenAI-compatible endpoint without changing the chain. *(lec_01f)* <!-- O2 -->

## Files

### Required notebooks

- `lec_01a_first_call_tokens_context.ipynb`: Ollama as a local model server and its
  Python client, local versus hosted, the first call and its reply, tokens, context
  window, temperature.
- `lec_01b_choosing_a_model_requirements_hf_hub.ipynb`: requirements table, Hub API,
  model cards, licenses and open weights, quantization, llmfit, comparing candidates,
  decision record.
- `lec_01c_prompts_roles_structured_output.ipynb`: prompts as functions, roles,
  few-shot, JSON-schema output and its limits, DataFrames, caching by model and prompt
  version, streaming.
- `lec_01d_langchain_chains.ipynb`: why a framework, LangChain parts and chains,
  rendered prompts, structured output in a chain, the one-line model swap, what
  LangChain adds and what it costs.

### Optional / Further reading

- `lec_01e_run_a_model_with_transformers.ipynb`: what `transformers` is and solves, and
  the model without the server. Career-track value: understanding what inference engines
  do, and what "running a model" costs.
- `lec_01f_pydantic_and_hosted_inference.ipynb`: pydantic schemas and validation, the
  OpenAI-compatible API, and the one-line swap to a hosted model, weighed against a
  local one. Career-track value: how production systems mix local and hosted models.
- Guides in `infra_tools/` (`01a_git_uv` to `01d_env_hugging_face`): git and uv, llmfit,
  Ollama (the server, its address and its API), the `.env` settings file and a Hugging
  Face token.

## Practice

`practice_exercises/lec_01_exercises.ipynb` (solutions in
`lec_01_exercises_solutions.ipynb`). Exercises are tagged with the goal they practise,
e.g. `[G5]`; `[O1]` marks stretch exercises.

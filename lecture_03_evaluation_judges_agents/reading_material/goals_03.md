# Lecture 03: Evaluation, LLM judges and agents

Two sessions. Session 5: `lec_03a` and `lec_03b`. Session 6: `lec_03c` and `lec_03d`.

## Learning Goals

### Required

- Explain why one good answer proves nothing, and turn an evaluation set into MLflow
  records with `inputs` and `expectations`. *(lec_03a §1, §3)* <!-- G1 -->
- Read a trace: its spans, their types, inputs and outputs, and the token usage, in the
  MLflow UI and from code; trace your own function with `@mlflow.trace`. *(lec_03a §2)*
  <!-- G2 -->
- Write deterministic scorers with `@scorer`, run `mlflow.genai.evaluate`, and read the
  aggregate metrics against the per-question rows. *(lec_03a §4-6)* <!-- G3 -->
- Name what a keyword scorer cannot read, build an LLM judge with `make_judge`, and read
  its verdict and rationale. *(lec_03b §1-2)* <!-- G4 -->
- Compare two variants of the RAG system on the same questions and scorers, on quality
  and on cost, in the MLflow UI and from code. *(lec_03b §3)* <!-- G5 -->
- Measure how far to trust an LLM judge: agreement with reference labels, a confusion
  matrix, repeat stability, its window, and a known bias. *(lec_03c)* <!-- G6 -->
- Write a tool with `@tool`, run the tool loop by hand (the model asks, your code runs),
  and validate a tool's arguments in code. *(lec_03d §1-4)* <!-- G7 -->
- Run an agent with `create_agent`, read its tool spans in the trace, explain when
  thinking is worth its cost, keep tools safe, and decide from evidence whether an agent
  is worth its calls. *(lec_03d §5-8)* <!-- G8 -->

### Optional / Career track

- Build an agentic RAG graph with LangGraph: rewrite the question, retrieve, grade the
  search, answer, with a hard limit on the loop; compare it with the plain chain and
  record the decision. *(lec_03e)* <!-- O1 -->
- Optimize a prompt with DSPy on a dev split, evaluate it on held-out questions, and
  explain why the metric decides what the optimizer keeps. *(lec_03f)* <!-- O2 -->

## Files

### Required notebooks

- `lec_03a_tracing_and_deterministic_scorers.ipynb`: tracing with MLflow, an evaluation
  set as MLflow records, code-based scorers, one evaluation run, aggregates against
  rows.
- `lec_03b_llm_judge_comparing_variants.ipynb`: what keywords cannot read, an LLM judge
  with `make_judge`, two retrieval variants compared on quality and cost.
- `lec_03c_judging_the_judge.ipynb`: an LLM judge against reference labels, its context
  window, and a known bias.
- `lec_03d_tools_and_agents.ipynb`: tools, the tool loop by hand, argument checks, a
  small model's decision, `create_agent`, safe tools, when an agent is worth it.

### Optional / Further reading

- `lec_03e_agentic_rag_langgraph.ipynb`: a LangGraph graph that rewrites, retrieves,
  grades and answers within a loop limit, compared with the plain chain. Career-track
  value: building your own agent graph, and measuring whether it pays.
- `lec_03f_dspy_prompt_optimization.ipynb`: DSPy prompt optimization. Career-track
  value: tuning prompts by measurement instead of by hand, and its pitfalls.
- Guide in `infra_tools/` (`03b_dspy_optional_group`): installing the optional
  `career-track` group, and why a plain `uv sync` removes it.

## Terms

Every term this lecture defines, with a link to its definition, is in the course
glossary, [`glossary.md`](../../glossary.md).

## Practice

`practice_exercises/lec_03_exercises.ipynb` (solutions in
`lec_03_exercises_solutions.ipynb`). Exercises are tagged with the goal they practise,
e.g. `[G2]`; `[O1]` marks optional exercises on the career-track notebook `lec_03e`. The
MLflow tracking server is started with guide `infra_tools/03a_mlflow_server`.

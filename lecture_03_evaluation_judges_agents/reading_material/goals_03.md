# Lecture 03: Evaluation, LLM judges and agents

Two sessions. Session 5: `lec_03a` and `lec_03b`. Session 6: `lec_03c` and `lec_03d`.

## Learning Goals

### Required

- Explain why one good answer proves nothing, and turn an evaluation set into MLflow
  records with `inputs` and `expectations`. *(lec_03a §1-2)* <!-- G1 -->
- Write deterministic scorers with `@scorer`, run `mlflow.genai.evaluate`, and read the
  aggregate metrics against the per-question rows; name what a keyword scorer cannot
  see. *(lec_03a §3-6)* <!-- G2 -->
- Build an LLM judge with `make_judge`, read its verdict and rationale, and check that
  its prompt fits the context window. *(lec_03b §1-2, §4)* <!-- G3 -->
- Compare two variants of the RAG system on the same questions and scorers, on quality
  and on cost, in the MLflow UI and from code. *(lec_03b §3)* <!-- G4 -->
- Write a tool with `@tool`, run the tool loop by hand (the model asks, your code runs),
  and validate a tool's arguments in code. *(lec_03c §1-4)* <!-- G5 -->
- Run an agent with `create_agent`, read its tool spans in the trace, explain when
  thinking is worth its cost, and keep tools safe. *(lec_03c §5-7)* <!-- G6 -->
- Build an agentic RAG graph with LangGraph: rewrite the question, retrieve, grade the
  search, answer, with a hard limit on the loop; trace one run step by step. *(lec_03d
  §1-6)* <!-- G7 -->
- Decide from measurements whether an agentic loop is worth its extra tokens and time,
  and record the decision. *(lec_03d §7-8)* <!-- G8 -->

### Optional / Career track

- Measure how far to trust an LLM judge: agreement with reference labels, a confusion
  matrix, repeat stability, and a known bias. *(lec_03e)* <!-- O1 -->
- Optimize a prompt with DSPy on a dev split, evaluate it on held-out questions, and
  explain why the metric decides what the optimizer keeps. *(lec_03f)* <!-- O2 -->

## Files

### Required notebooks

- `lec_03a_eval_set_deterministic_scorers.ipynb`: an evaluation set as MLflow records,
  code-based scorers, one evaluation run, aggregates against rows, the limits of
  keywords.
- `lec_03b_llm_judge_comparing_variants.ipynb`: an LLM judge with `make_judge`, two
  retrieval variants compared on quality and cost, the judge's context window.
- `lec_03c_tools_and_agents.ipynb`: tools, the tool loop by hand, argument checks, a
  small model's decision, `create_agent`, safe tools.
- `lec_03d_agentic_rag_langgraph.ipynb`: a LangGraph graph that rewrites, retrieves,
  grades and answers within a loop limit, compared with the plain chain, and a decision
  record.

### Optional / Further reading

- `lec_03e_judging_the_judge.ipynb`: an LLM judge against reference labels. Career-track
  value: knowing when an automated grade can replace a person, and when it cannot.
- `lec_03f_dspy_prompt_optimization.ipynb`: DSPy prompt optimization. Career-track
  value: tuning prompts by measurement instead of by hand, and its pitfalls.
- Guide in `infra_tools/` (`03a_dspy_optional_group`): installing the optional
  `career-track` group, and why a plain `uv sync` removes it.

## Practice

`practice_exercises/lec_03_exercises.ipynb` (solutions in
`lec_03_exercises_solutions.ipynb`). Exercises are tagged with the goal they practise,
e.g. `[G2]`.

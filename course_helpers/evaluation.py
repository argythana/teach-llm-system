"""Evaluation helpers, promoted from lec_03a and lec_03b.

Functions only, in the order an evaluation is built: records -> deterministic scorers
-> per-question table (lec_03a) -> LLM judge -> comparison of variants (lec_03b)
-> the judge checked against hand labels (lec_03c).
Every code-based scorer is a plain function wrapped by MLflow's ``@scorer``
decorator, so ``mlflow.genai.evaluate`` can call it on each answer; called directly, it
scores one answer, which is how lec_03a tests a scorer on hand-written text.
"""

import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from mlflow.entities import SpanType, Trace
from mlflow.genai.scorers import scorer

# Phrases a RAG answer uses to say "the notes do not contain this" (lec_03a section 4),
# matched in lower case after "n't" is spelled out as " not".
REFUSAL_PHRASES = [
    "not contain",
    "not mention",
    "not cover",
    "no information",
    "not explicitly mentioned",
    "cannot determine",
    "cannot answer",
]

# Words too common to say anything about a fact (lec_03a section 4).
STOPWORDS = frozenset(
    "a an and are as at be by can do does for from has have how in into is it its "
    "of on or that the their them they this to use uses was when which with you your".split()
)


def build_eval_records_from(eval_set: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Turn evaluation-set rows into MLflow records: ``inputs`` and ``expectations``.

    ``inputs`` holds the keyword arguments of the ``predict_fn`` (here ``question``);
    ``expectations`` holds what the scorers compare with: the expected source file
    and the expected facts.
    """
    return [
        {
            "inputs": {"question": row["question"]},
            "expectations": {
                "expected_source": row["expected_source"],
                **row["expectations"],
            },
        }
        for row in eval_set
    ]


def retrieved_sources(trace: Trace) -> list[str]:
    """Source files of the chunks in the trace's first RETRIEVER span, in rank order."""
    spans = trace.search_spans(span_type=SpanType.RETRIEVER)
    if not spans:
        return []
    return [doc["metadata"]["source"] for doc in spans[0].outputs or []]


def is_refusal(text: str) -> bool:
    """True if ``text`` contains one of the ``REFUSAL_PHRASES``."""
    text = text.lower().replace("n't", " not")
    return any(phrase in text for phrase in REFUSAL_PHRASES)


def extract_keywords_from(fact: str) -> list[str]:
    """The words of ``fact`` that carry meaning: lower case, stopwords dropped."""
    words = re.findall(r"[a-z0-9_]+", fact.lower())
    return [w for w in words if len(w) > 1 and w not in STOPWORDS]


def keyword_coverage(answer: str, facts: Sequence[str]) -> float:
    """Share of ``facts`` with at least half of their keywords somewhere in ``answer``.

    A crude first attempt: it matches substrings, not meaning (lec_03b section 1).
    """
    if not facts:
        return 0.0
    text = answer.lower()
    found = 0
    for fact in facts:
        keywords = extract_keywords_from(fact)
        if keywords and sum(k in text for k in keywords) >= len(keywords) / 2:
            found += 1
    return found / len(facts)


@scorer
def cites_expected_source(
    outputs: str, expectations: dict[str, Any] | None
) -> bool | None:
    """Does the answer name the expected file (its name without ``.md``)?

    Abstains (``None``) when the record has no expected file, e.g. an out-of-scope
    question.
    """
    if not expectations or not expectations.get("expected_source"):
        return None
    return Path(expectations["expected_source"]).stem in outputs


@scorer
def retrieval_hit(trace: Trace, expectations: dict[str, Any] | None) -> bool | None:
    """Was a chunk of the expected file among the chunks this answer was given?

    Abstains (``None``) when the record has no expected file.
    """
    if not expectations or not expectations.get("expected_source"):
        return None
    return expectations["expected_source"] in retrieved_sources(trace)


@scorer(aggregations=["mean", "max"])
def answer_tokens(trace: Trace) -> int:
    """Number of tokens the model generated for this answer (from the trace)."""
    usage = trace.info.token_usage or {}
    return usage.get("output_tokens", 0)


@scorer(aggregations=["mean"])
def prompt_tokens(trace: Trace) -> int:
    """Number of tokens the model read for this answer: the prompt with its chunks.

    The input side of ``answer_tokens``, from the same trace. A hosted model bills
    these tokens too; more retrieved chunks make a longer prompt (lec_03b).
    """
    usage = trace.info.token_usage or {}
    return usage.get("input_tokens", 0)


@scorer
def refused(outputs: str) -> bool:
    """Does the answer say that the notes do not contain the answer?"""
    return is_refusal(outputs)


@scorer
def fact_keywords_found(
    outputs: str, expectations: dict[str, Any] | None
) -> float | None:
    """Share of the expected facts whose keywords appear in the answer (crude).

    Abstains (``None``) when the record has no expected facts.
    """
    if not expectations or not expectations.get("expected_facts"):
        return None
    return keyword_coverage(outputs, expectations["expected_facts"])


# The lec_03a suite, ready for mlflow.genai.evaluate(..., scorers=DETERMINISTIC_SCORERS).
DETERMINISTIC_SCORERS = [
    cites_expected_source,
    retrieval_hit,
    answer_tokens,
    refused,
    fact_keywords_found,
]


def per_question_table(
    results: Any, eval_set: Iterable[dict[str, Any]], scorers: Sequence[Any]
) -> pd.DataFrame:
    """One row per question of an evaluation result, one column per scorer.

    ``results`` is what ``mlflow.genai.evaluate`` returns; ``scorers`` the scorers it
    ran (each has a ``.name``). Rows are matched to ``eval_set`` by question text and
    indexed by its ``id``, sorted.
    """
    ids = {row["question"]: row["id"] for row in eval_set}
    frame = results.result_df
    # "request" is a dict after a predict_fn, the plain question when re-scoring traces
    questions = [r["question"] if isinstance(r, dict) else r for r in frame["request"]]
    table = pd.DataFrame(
        {s.name: frame[f"{s.name}/value"].to_numpy() for s in scorers},
        index=[ids[question] for question in questions],
    )
    return table.rename_axis("id").sort_index()


# --- LLM judges and comparing variants (promoted from lec_03b) ---------------------

# The rubric of the lec_03b judge. The double braces are MLflow's placeholders, filled
# from each record: the question, the answer and the expectations.
FACTS_RUBRIC = (
    "Question: {{ inputs }}\n\n"
    "Answer: {{ outputs }}\n\n"
    "Expected facts: {{ expectations }}\n\n"
    "Reply yes if the answer states every expected fact, in any wording, "
    "and contradicts none of them. "
    "Reply no if any expected fact is missing or the answer states its opposite."
)

# Where MLflow records how many tokens a judge read for one verdict.
JUDGE_INPUT_TOKENS = "mlflow.assessment.judgeInputTokens"


def make_facts_judge(
    model: str,
    ollama_host: str = "http://localhost:11434",
    name: str = "states_facts",
) -> Any:
    """An LLM judge on a local Ollama model: does the answer state the expected facts?

    Temperature 0 and thinking off (``reasoning_effort: "none"`` on Ollama's
    OpenAI-compatible endpoint). ``base_url`` sends the requests to ``ollama_host``,
    which MLflow's built-in judges cannot do. The judge logs the mean of its yes/no
    verdicts (the share of "yes") as a metric, which ``make_judge`` does not do by
    default.
    """
    from typing import Literal

    from mlflow.genai.judges import make_judge

    judge = make_judge(
        name=name,
        instructions=FACTS_RUBRIC,
        model=f"ollama:/{model}",
        feedback_value_type=Literal["yes", "no"],
        inference_params={"temperature": 0, "reasoning_effort": "none"},
        base_url=f"{ollama_host}/v1/chat/completions",
    )
    judge.aggregations = ["mean"]
    return judge


def repeat_judgement(
    judge: Any, record: dict[str, Any], outputs: str, times: int = 5
) -> list[str]:
    """Ask ``judge`` the same judgement ``times`` times; return the verdicts as text."""
    return [
        str(
            judge(
                inputs=record["inputs"],
                outputs=outputs,
                expectations=record["expectations"],
            ).value
        )
        for _ in range(times)
    ]


def compare_variants(results_by_variant: dict[str, Any]) -> pd.DataFrame:
    """One column per variant: its evaluation metrics and mean seconds per answer.

    ``results_by_variant`` maps a variant's name to what ``mlflow.genai.evaluate``
    returned for it. Rows are sorted, because the order of ``results.metrics`` varies
    between runs.
    """
    columns = {}
    for name, results in results_by_variant.items():
        column = dict(results.metrics)
        # execution_duration is in milliseconds
        column["seconds_per_answer/mean"] = (
            results.result_df["execution_duration"].mean() / 1000
        )
        columns[name] = column
    return pd.DataFrame(columns).sort_index()


def judge_input_tokens(results: Any, judge_name: str) -> list[int]:
    """Tokens the judge ``judge_name`` read for each verdict of an evaluation result."""
    tokens = []
    for assessments in results.result_df["assessments"]:
        for assessment in assessments:
            metadata = assessment.get("metadata") or {}
            if (
                assessment.get("assessment_name") == judge_name
                and JUDGE_INPUT_TOKENS in metadata
            ):
                tokens.append(int(metadata[JUDGE_INPUT_TOKENS]))
    return tokens


def model_with_window(client: Any, model: str, num_ctx: int) -> str:
    """Create (or update) a copy of an Ollama ``model`` with a ``num_ctx`` window.

    Same weights, nothing downloaded: Ollama stores only the new setting. Returns the
    copy's name, e.g. ``qwen3:1.7b-ctx8k``. A judge on Ollama's OpenAI-compatible
    endpoint cannot ask for a window per request; it can use this copy (lec_03c).
    """
    name = f"{model}-ctx{num_ctx // 1024}k"
    client.create(model=name, from_=model, parameters={"num_ctx": num_ctx})
    return name


# --- Checking the judge against hand labels (promoted from lec_03c) ----------------


def run_judge_on(
    judge: Any, labelled: pd.DataFrame, records_by_id: dict[str, dict[str, Any]]
) -> list[str]:
    """One verdict of ``judge`` per row of ``labelled``, as text, in row order.

    ``labelled`` has the columns ``question_id`` and ``answer`` (as in
    ``corpus/eval/judge_labels.jsonl``); ``records_by_id`` maps a question id to its
    MLflow record, whose question and expectations the judge reads.
    """
    verdicts = []
    for question_id, answer in zip(labelled["question_id"], labelled["answer"]):
        record = records_by_id[question_id]
        feedback = judge(
            inputs=record["inputs"],
            outputs=answer,
            expectations=record["expectations"],
        )
        verdicts.append(str(feedback.value))
    return verdicts


def measure_agreement_between(
    labels: Sequence[str], verdicts: Sequence[str]
) -> pd.Series:
    """How a judge's ``"yes"``/``"no"`` verdicts compare with hand labels.

    - ``agreement``: share of answers where verdict and label are the same;
    - ``false_yes_rate``: share of the answers labelled ``"no"`` that the judge
      passed (how lenient it is);
    - ``false_no_rate``: share of the answers labelled ``"yes"`` that the judge did
      not pass (how strict it is).
    """
    labels = pd.Series(list(labels))
    verdicts = pd.Series(list(verdicts))
    return pd.Series(
        {
            "agreement": (labels == verdicts).mean(),
            "false_yes_rate": (verdicts[labels == "no"] == "yes").mean(),
            "false_no_rate": (verdicts[labels == "yes"] != "yes").mean(),
        }
    )


def confusion_table(labels: Sequence[str], verdicts: Sequence[str]) -> pd.DataFrame:
    """Confusion matrix of verdicts against labels, ``"yes"`` first, with readable names.

    Rows are the hand labels, columns the judge's verdicts: the lower-left cell
    counts false "yes", the upper-right false "no". Verdicts other than ``"yes"``
    and ``"no"`` are left out, as scikit-learn does.
    """
    from sklearn.metrics import confusion_matrix

    values = ["yes", "no"]
    return pd.DataFrame(
        confusion_matrix(labels, verdicts, labels=values),
        index=[f"label {v}" for v in values],
        columns=[f"judge {v}" for v in values],
    )

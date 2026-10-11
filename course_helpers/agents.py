"""Tools and agents, promoted from lec_03c and lec_03d.

Functions only, in the order the notebooks build them: the course's tools -> running
the tool calls a model asks for -> reading the messages an agent exchanged (lec_03c)
-> the agentic RAG graph, its step table and its cost scorers (lec_03d).

A tool returns an ``{"error": ...}`` dict instead of raising: an exception would stop
the program, a returned error is read by the model, which can correct its call or tell
the user (lec_03c section 4).
"""

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool, tool
from mlflow.entities import SpanType, Trace
from mlflow.genai.scorers import scorer

from course_helpers.rag import DEFAULT_RAG_PROMPT, format_context_from

# The Python course has lectures 1 to 13 (corpus/uoa_py_course/lecture_01 ... _13).
FIRST_LECTURE, LAST_LECTURE = 1, 13


def find_notebooks_of(documents: Iterable[Document], lecture: int) -> list[str]:
    """File names of the lecture notebooks (``lec_...``) of one lecture, sorted.

    Exercises and solutions are left out. A lecture that does not exist gives an
    empty list: check the number before calling (lec_03c section 4).
    """
    return sorted(
        Path(d.metadata["source"]).name
        for d in documents
        if d.metadata["lecture"] == lecture
        and d.metadata["kind"] == "lecture"
        and Path(d.metadata["source"]).name.startswith("lec_")
    )


def make_course_tools(
    documents: Sequence[Document],
    retriever: Runnable[str, list[Document]] | None = None,
) -> list[BaseTool]:
    """Return the course's tools: ``count_mentions``, ``list_notebooks``, ``search_notes``.

    ``documents`` are the notes from ``course_helpers.corpus.load_corpus``; ``retriever``
    (for example ``store.as_retriever(...)``, lec_02d) adds ``search_notes``. Every tool
    only reads the notes: no files written, no network, no shell (lec_03c section 7).
    """

    @tool
    def count_mentions(term: str) -> dict:
        """Count the files of the course notes that contain an exact term.

        Use it for questions such as "how many files mention X?". The term is matched
        exactly, capital letters included.
        """
        if not term.strip():
            return {"error": "The term is empty; give the word to count."}
        files = [d for d in documents if term in d.page_content]
        occurrences = sum(d.page_content.count(term) for d in files)
        return {"term": term, "files": len(files), "occurrences": occurrences}

    @tool
    def list_notebooks(lecture: int) -> dict:
        """List the lecture notebooks of one lecture (1 to 13) of the Python course.

        Exercises and solutions are not included.
        """
        if not FIRST_LECTURE <= lecture <= LAST_LECTURE:
            return {
                "error": f"There is no lecture {lecture}; "
                f"the lectures are {FIRST_LECTURE} to {LAST_LECTURE}."
            }
        names = find_notebooks_of(documents, lecture)
        return {"lecture": lecture, "count": len(names), "notebooks": names}

    tools: list[BaseTool] = [count_mentions, list_notebooks]
    if retriever is not None:

        @tool
        def search_notes(query: str) -> str:
            """Search the course notes and return the most relevant passages.

            Use it for questions about what the notes explain or how to do something.
            """
            return format_context_from(retriever.invoke(query))

        tools.append(search_notes)
    return tools


def run_tool_calls(reply: AIMessage, tools: Iterable[BaseTool]) -> list[ToolMessage]:
    """Run every tool call in ``reply``; return one ToolMessage per call.

    Never raises on a bad call: an unknown tool name or arguments that fail the type
    hints come back as an error message the model can read, as LangChain's agent does.
    """
    by_name = {t.name: t for t in tools}
    results: list[ToolMessage] = []
    for call in reply.tool_calls:
        if call["name"] not in by_name:
            content = (
                f"Error: there is no tool {call['name']!r}; "
                f"the tools are {sorted(by_name)}."
            )
        else:
            try:
                output = by_name[call["name"]].invoke(call["args"])
                content = output if isinstance(output, str) else json.dumps(output)
            except Exception as error:  # pydantic's ValidationError, among others
                # Three lines: pydantic's fourth is a link to its docs.
                first_lines = str(error).splitlines()[:3]
                details = " ".join(line.strip() for line in first_lines)
                content = f"Error: {type(error).__name__}: {details}"
        results.append(ToolMessage(content, tool_call_id=call["id"], name=call["name"]))
    return results


def message_table(
    messages: Iterable[BaseMessage], width: int = 70
) -> list[dict[str, Any]]:
    """One row per message: who wrote it, the tool calls it asks for, its text.

    Pass the result to ``pandas.DataFrame`` to read an agent's conversation step by step.
    """
    rows = []
    for message in messages:
        calls = getattr(message, "tool_calls", None) or []
        rows.append(
            {
                "role": message.type,  # human, ai, tool
                "tool calls": "; ".join(f"{c['name']}({c['args']})" for c in calls),
                "content": " ".join(str(message.content).split())[:width],
            }
        )
    return rows


# --- Agentic RAG with LangGraph (promoted from lec_03d) ----------------------------

# The model rewrites the student's question into a search query in the notes' words.
REWRITE_PROMPT = (
    "Rewrite the question below as a short search query for the notes of a Python course, "
    "with the technical terms the notes would use. Reply with the query only."
    "\n\nQuestion: {question}"
)

# After a search graded "no", the model writes a different query.
RETRY_PROMPT = (
    "The search query below did not find the answer to the question. Write another short "
    "search query for the same question, in other words. "
    "Reply with the query only.\n\nQuestion: {question}\n\nQuery that failed: {query}"
)

# The grade asks about the topic, not the answer: in lec_03d's example runs a stricter
# wording said "no" to chunks that did answer, and every wrong "no" costs a search.
GRADE_PROMPT = (
    "Notes:\n{context}\n\nQuestion: {question}\n\n"
    "Is the question's topic explained in the notes above? Reply yes or no."
)
GRADE_SCHEMA = {
    "title": "Grade",
    "type": "object",
    "properties": {"topic_explained": {"type": "string", "enum": ["yes", "no"]}},
    "required": ["topic_explained"],
}


def make_agentic_rag(
    retriever: Runnable[str, list[Document]],
    llm: BaseChatModel,
    max_searches: int = 2,
    answer_prompt: str | None = None,
) -> Any:
    """Build lec_03d's graph: rewrite -> retrieve -> grade -> answer, or search again.

    The model writes the query and grades the chunks; the search and the limit are
    code. After ``max_searches`` searches the graph answers with what it found, and the
    answer prompt (``DEFAULT_RAG_PROMPT`` unless given) says so when the notes are
    silent. Returns the compiled graph: ``graph.invoke({"question": ...})["answer"]``.
    """
    from typing import TypedDict

    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langgraph.graph import END, START, StateGraph

    rag_state = TypedDict(
        "RagState",
        {
            "question": str,
            "query": str,
            "documents": list,
            "searches": int,
            "grade": str,
            "answer": str,
        },
    )
    rewrite_chain = (
        ChatPromptTemplate.from_template(REWRITE_PROMPT) | llm | StrOutputParser()
    )
    retry_chain = (
        ChatPromptTemplate.from_template(RETRY_PROMPT) | llm | StrOutputParser()
    )
    grade_chain = ChatPromptTemplate.from_template(
        GRADE_PROMPT
    ) | llm.with_structured_output(GRADE_SCHEMA, method="json_schema")
    answer_chain = (
        ChatPromptTemplate.from_template(answer_prompt or DEFAULT_RAG_PROMPT)
        | llm
        | StrOutputParser()
    )

    def rewrite(state: dict[str, Any]) -> dict[str, Any]:
        if state.get("searches", 0) == 0:
            query = rewrite_chain.invoke({"question": state["question"]})
        else:
            query = retry_chain.invoke(
                {"question": state["question"], "query": state["query"]}
            )
        return {"query": query.strip()}

    def retrieve(state: dict[str, Any]) -> dict[str, Any]:
        return {
            "documents": retriever.invoke(state["query"]),
            "searches": state.get("searches", 0) + 1,
        }

    def grade(state: dict[str, Any]) -> dict[str, Any]:
        verdict = grade_chain.invoke(
            {
                "context": format_context_from(state["documents"]),
                "question": state["question"],
            }
        )
        return {"grade": verdict["topic_explained"]}

    def answer(state: dict[str, Any]) -> dict[str, Any]:
        text = answer_chain.invoke(
            {
                "context": format_context_from(state["documents"]),
                "question": state["question"],
            }
        )
        return {"answer": text}

    def route_after_grade(state: dict[str, Any]) -> str:
        if state["grade"] == "yes" or state["searches"] >= max_searches:
            return "answer"
        return "rewrite"

    builder = StateGraph(rag_state)
    builder.add_node("rewrite", rewrite)
    builder.add_node("retrieve", retrieve)
    builder.add_node("grade", grade)
    builder.add_node("answer", answer)
    builder.add_edge(START, "rewrite")
    builder.add_edge("rewrite", "retrieve")
    builder.add_edge("retrieve", "grade")
    builder.add_conditional_edges("grade", route_after_grade, ["answer", "rewrite"])
    builder.add_edge("answer", END)
    return builder.compile()


def step_table(
    updates: Iterable[dict[str, Any]], width: int = 50
) -> list[dict[str, Any]]:
    """One row per node that ran: its name and the state keys it changed.

    ``updates`` are what ``graph.stream(..., stream_mode="updates")`` yields; chunks
    are shown as the file names they come from. Pass the result to ``pandas.DataFrame``.
    """
    rows = []
    for update in updates:
        for node, changes in update.items():
            row = {"node": node}
            for key, value in changes.items():
                if key == "documents":
                    value = ", ".join(
                        Path(d.metadata["source"]).stem[:10] for d in value
                    )
                row[key] = " ".join(str(value).split())[:width]
            rows.append(row)
    return rows


@scorer
def searches(trace: Trace) -> int:
    """How many searches (RETRIEVER spans) the system made for this answer."""
    return len(trace.search_spans(span_type=SpanType.RETRIEVER))


@scorer
def total_tokens(trace: Trace) -> int:
    """Tokens read and written by all the model calls of this answer."""
    usage = trace.info.token_usage or {}
    return usage.get("total_tokens", 0)

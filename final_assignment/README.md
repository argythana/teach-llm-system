# Final assignment

**Status: Optional / career-track.**

The multiple-choice exam is the only mandatory part of this course. This assignment is
for students who want a portfolio piece that shows they can build, evaluate and publish
an LLM system: it does not count towards the grade.

You build a question-answering assistant over scientific papers in your own field, run
it on a free Kaggle GPU, evaluate it with MLflow and publish it as a public Kaggle
notebook. A reader sees questions, answers that cite their papers, and the evaluation
that says how far to trust them.

Papers on your dissertation topic are a good choice: the assistant then helps you with
your own literature review. Work alone: one student, one corpus, one notebook.

## Why Kaggle, and what changes

The course runs every model on your own computer, for privacy. Your papers here are
already published, so nothing private leaves your computer, and
[Kaggle](https://www.kaggle.com/) gives you a free GPU on its servers: larger models
than a laptop can run, at no cost.

Everything else is the course's own stack: Ollama, PostgreSQL with pgvector, LangChain
and MLflow. Record this privacy trade-off in your decision record.

Never put private material in the notebook: drafts of your dissertation, interview data,
or anything you may not publish.

## Before you start

- **A Kaggle account, phone-verified**: Kaggle turns on a notebook's GPU and internet
  access only after you verify a phone number. The GPU has a weekly quota, shown in your
  account settings.
- **The optional notebooks** the assignment builds on: `lec_01f` section 1 (pydantic),
  `lec_02e` (pgvector), `lec_02f` (a corpus at scale), and `lec_03e` (LangGraph) or
  `lec_03f` (DSPy).
- **The worked example**: an assistant over 23 ACL Anthology papers on evaluating LLM
  agents and judges. Read the
  [public version on Kaggle](https://www.kaggle.com/code/thanasisargyriou/paper-assistant-worked-example),
  with every output, and its
  [dataset](https://www.kaggle.com/datasets/thanasisargyriou/acl-2025-agent-eval-papers);
  the source is [`example/paper_assistant.ipynb`](example/paper_assistant.ipynb). Copy
  it, replace the corpus, and adapt it.

## What you build

Seven parts, all required.

### 1. A corpus you may republish

Choose 20 to 50 papers in your field. Your notebook shows passages of them to anyone on
the internet, so each paper needs a license that allows republishing: CC BY, CC BY-SA or
CC0.

- **[ACL Anthology](https://aclanthology.org/)**: every paper since 2016 is CC BY 4.0
  ([copyright FAQ](https://aclanthology.org/faq/copyright)).
- **[arXiv](https://arxiv.org/)**: the author chooses the license of each paper
  ([license page](https://info.arxiv.org/help/license/)). The default arXiv license lets
  arXiv distribute the paper, not you: skip those papers.
- **Open-access journals**: check the license printed on each article.

Upload them as a public Kaggle dataset, with `papers.csv`: one row per paper with its
id, title, authors, year, link and license. The notebook stops if a license is not on
the allowed list.

Put the PDFs in a `pdfs/` folder. Kaggle's dataset page previews the first file of the
list, and a PDF there makes some browsers download it on every visit. Fill in the
dataset's description, cover image and file descriptions too: Kaggle shows the result as
a usability score.

Describe every column of your CSV files as well, on the dataset page: select the file,
click the pencil next to *About this file* and fill in the box under each column. The
Kaggle command-line tool accepts column descriptions but does not save them.

### 2. An index in pgvector

Split the papers into chunks, embed them with Ollama and store them in PostgreSQL with
pgvector, as in `lec_02e`. Kaggle has no Docker: install PostgreSQL from its apt
repository and connect without a password, as the worked example does.

### 3. Answers that cite their papers

The answer is a pydantic schema (`lec_01f`): the text and the ids of the papers it used.
A reader sees where each answer comes from, and a scorer can check the citation.

### 4. One career-track technique

Choose one:

- **An agentic graph** with LangGraph (`lec_03e`): search again in other words, and
  refuse when no search finds the answer. The worked example does this.
- **An optimized prompt** with DSPy (`lec_03f`): optimize the answer prompt on part of
  your evaluation set and compare it with your hand-written prompt on the rest.

### 5. An evaluation

With MLflow, as in lecture 3, stored in a file `mlflow.db` in `/kaggle/working`:

- **Evaluation set**: at least 15 questions you wrote, each with the facts a correct
  answer must state and the paper they come from, plus at least 2 questions that no
  paper answers.
- **Scorers**: at least one deterministic scorer and one LLM judge built with
  `make_judge`, on a larger model than the ones it grades.
- **Judge check**: at least 10 fixed answers labelled by hand, at least a third of them
  `no`, with the judge's agreement and false-yes rate, as in `lec_03c`.
- **Two variants**: two versions of the system on the same questions, compared on
  quality and on time. For example two models, two values of `k`, or the graph against
  the plain chain.

### 6. A decision record

A section of the notebook that chooses the model, requirements first, as in
[`lec_01b`](../lecture_01_ollama_models_prompts_langchain/reading_material/lec_01b_choosing_a_model_requirements_hf_hub.ipynb)
section 8: the task, the languages, the context your chunks need, latency, hardware,
license, privacy, and the candidates you compared. It cites the numbers of your saved
version.

### 7. Publish and announce

- **The notebook**: a public Kaggle notebook whose saved version (*Save & Run All*)
  shows every output, with `mlflow.db` among its outputs.
- **The dataset**: public, so that anyone can copy the notebook and run it.
- **LinkedIn**: a post that links the notebook.
- **eClass**: a post on the course's eClass page with the same link.

## Acceptance criteria

The assignment is accepted when all of these hold:

- The notebook and its dataset open in a private browser window, without logging in.
- The saved version ran from top to bottom on Kaggle and shows its outputs.
- Every paper has a row in `papers.csv` with CC BY, CC BY-SA or CC0.
- No access key or password appears in the notebook or the dataset.
- Answers cite papers by id, and a scorer checks the citations.
- The evaluation has at least 15 questions plus 2 unanswerable ones, a deterministic
  scorer, an LLM judge with its check against at least 10 labels, and two variants
  compared.
- The decision record starts from the requirements and cites the evaluation.
- The LinkedIn and eClass posts link the notebook.

## Optional: a live app

A notebook shows the answers of one run; a web app answers a visitor's own question. The
[`example/optional_live_app/`](example/optional_live_app/) folder has a prototype: the
same assistant with a small model, PostgreSQL and a Gradio page in one Docker container.

Hosting it is not free: a Docker Space on Hugging Face needs a PRO account (USD 9 a
month at the time of writing). Treat it as an extra for after the assignment, not a part
of it.

## Common pitfalls

- **A notebook without outputs.** Only a version saved with *Save & Run All* keeps its
  outputs; a quick save of an unrun notebook publishes empty cells.
- **A queue for the GPU.** At busy times Kaggle puts a GPU session in a queue. Start
  early; a deadline evening is a busy time.
- **A license you cannot use.** A paper you can read for free is not necessarily one you
  may republish; check the license, not the price.
- **Two embedding models.** An index built with one embedding model and questions
  embedded with another retrieve the wrong chunks without any error.
- **A judge that reads too much.** The judge sees the whole `expectations` of a record:
  keep only the facts there, and let the other scorers look up the expected paper by
  question.
- **A question with two right answers.** If your facts and a good answer disagree, check
  the paper before blaming the model: the question may be ambiguous.

## Credit

There is no grade. When the acceptance criteria hold, the instructor endorses the
relevant skills on your LinkedIn profile: RAG, LLM evaluation, MLflow and LangChain.

**Deadline:** to be announced on eClass.

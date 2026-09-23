# Guide 00c: create a Hugging Face token

The **Hugging Face Hub** is where open models live (you used it in the Python course to
deploy a Gradio app to a Space). Lecture 1b searches it through its Python API to choose
a model. Public information works without logging in, but anonymous requests are
rate-limited and some models (for example Meta's Llama family) require you to accept a
license first, which needs an account. A **read token** solves both.

The Hub is documented at
[huggingface.co/docs/hub](https://huggingface.co/docs/hub/index). The Python library
that queries it, `huggingface_hub`, is installed by `uv sync`:
[source on GitHub](https://github.com/huggingface/huggingface_hub),
[documentation](https://huggingface.co/docs/huggingface_hub/index).

## 1. Create the token

1. Log in at <https://huggingface.co> (create an account if you deleted yours).
1. Open <https://huggingface.co/settings/tokens>, click **Create new token**.
1. Choose the **Read** role, name it `teach-llm-system`, create it, and copy it once.

## 2. Store it in `.env`

In the repository folder, copy `.env.example` to `.env` and paste the token:

```text
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx
```

`.env` is listed in `.gitignore` and the pre-commit hook `gitleaks` blocks commits that
contain a token. The notebooks load it with `python-dotenv`:

```python
from dotenv import load_dotenv

load_dotenv()  # reads .env into environment variables
```

The `huggingface_hub` library then picks up `HF_TOKEN` automatically. `python-dotenv` is
the small package that reads `.env` files:
[source on GitHub](https://github.com/theskumar/python-dotenv).

## 3. If a token ever leaks

Open the tokens page, click **Invalidate and refresh** (or delete it) and create a new
one. Tokens are credentials, like passwords: never paste one into a notebook cell, a
chat, or a screenshot.

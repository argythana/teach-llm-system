# About this corpus

`uoa_py_course/` is a Markdown export of the teaching notebooks, scripts, goals and readings
of [argythana/uoa_py_course](https://github.com/argythana/uoa_py_course) (Python for Data
Science, University of Athens) by Thanasis Argyriou, the author of this course. The commit
exported and the list of files are in `manifest.json`; regenerate with
`uv run python tools/export_corpus.py --clone`.

Notebook outputs, images and data files are not included; only the text students read and
the code they ran. Each file starts with an HTML comment naming its source path.

The export is republished here under
[CC BY 4.0](../LICENSE-CC-BY-4.0.txt) as course material. Do not edit these files by
hand: they are a dataset, and the manifest checks their hashes.

`eval/qa_eval_set.jsonl` holds the evaluation questions written for this course.

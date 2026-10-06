"""Browser illustrations for guide 02a: the MLflow UI at http://127.0.0.1:5010.

The layout copies MLflow 3.16's GenAI view: the home page with Recent Experiments, and
an experiment's Traces page. The trace rows are questions the course notebooks ask.
"""

CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
html { zoom: 2; overflow: hidden; }
body { width: 1000px; overflow: hidden; position: relative;
       font-family: "Noto Sans", "Ubuntu", "DejaVu Sans", sans-serif; font-size: 12.5px;
       color: #1f272d; background: #dfe6f0; }
.mono { font-family: "DejaVu Sans Mono", monospace; }
.win { position: absolute; left: 20px; right: 20px; top: 18px; bottom: 18px;
       background: #fff; border-radius: 9px; border: 1px solid #b9c2d0;
       box-shadow: 0 8px 26px rgba(0,0,0,.25); display: flex; flex-direction: column;
       overflow: hidden; }
.tabsbar { height: 34px; background: #e8ebf0; display: flex; align-items: flex-end;
           padding: 0 10px; gap: 6px; flex: none; }
.btab { background: #fff; border-radius: 8px 8px 0 0; height: 28px; padding: 0 14px;
        display: flex; align-items: center; gap: 8px; width: 220px; font-size: 12px; }
.btab .logo { width: 14px; height: 14px; border-radius: 3px; background: #43c9ed; }
.btab .x { margin-left: auto; color: #666; }
.ctl { margin-left: auto; align-self: center; display: flex; gap: 22px; color: #444;
       font-size: 14px; padding-right: 4px; }
.navbar { height: 40px; display: flex; align-items: center; gap: 16px; padding: 0 14px;
          border-bottom: 1px solid #e3e6ea; flex: none; color: #555; font-size: 15px; }
.url { flex: 1; height: 28px; border-radius: 14px; background: #eef1f5; display: flex;
       align-items: center; padding: 0 14px; font-size: 13px; color: #1f272d; }
.app { flex: 1; display: flex; min-height: 0; background: #f4f6f8; }
.nav { width: 190px; flex: none; padding: 12px 10px; font-size: 12.5px; }
.brand { font-size: 20px; font-weight: 700; color: #0194e2; letter-spacing: -.5px;
         padding-left: 12px; }
.brand b { color: #1f272d; }
.ver { font-size: 10.5px; color: #6b7480; padding: 0 0 10px 12px; }
.mode { display: flex; border: 1px solid #c9d0d8; border-radius: 14px; margin: 0 2px 10px;
        font-size: 11.5px; overflow: hidden; }
.mode span { flex: 1; text-align: center; padding: 4px 0; color: #6b7480; font-size: 10.5px; }
.mode span.on { color: #1f272d; box-shadow: inset 0 0 0 1.5px #e65b7a; border-radius: 14px; }
.item { padding: 6px 10px; border-radius: 5px; display: flex; gap: 9px; align-items: center; }
.item.on { background: #d6e6f5; font-weight: 600; }
.item i { width: 13px; height: 13px; border: 1.5px solid #5f6b77; border-radius: 3px; flex: none; }
.group { font-size: 11px; color: #6b7480; padding: 8px 10px 3px; }
.expbox { border: 1px solid #c9d0d8; border-radius: 5px; padding: 6px 10px; margin-bottom: 8px;
          background: #fff; }
.page { flex: 1; margin: 8px 8px 8px 0; background: #fff; border-radius: 8px;
        border: 1px solid #e3e6ea; padding: 16px 18px; min-width: 0; }
.page h1 { font-size: 18px; font-weight: 700; margin-bottom: 14px; }
.page h2 { font-size: 15px; font-weight: 700; margin: 4px 0 10px; }
.crumbs { font-size: 11.5px; color: #2272b4; margin-bottom: 4px; }
.cards { display: flex; gap: 10px; margin-bottom: 18px; }
.card { flex: 1; border: 1px solid #e3e6ea; border-radius: 6px; padding: 9px 11px; }
.card b { display: block; margin-bottom: 3px; }
.card span { color: #5f6b77; font-size: 11.5px; }
table { width: 100%; border-collapse: collapse; border: 1px solid #e3e6ea; font-size: 12px; }
th { text-align: left; font-weight: 600; padding: 8px 10px; border-bottom: 1px solid #e3e6ea; }
td { padding: 8px 10px; border-bottom: 1px solid #eef0f2; white-space: nowrap;
     overflow: hidden; max-width: 290px; text-overflow: ellipsis; }
td.link { color: #2272b4; }
.tools { display: flex; gap: 8px; margin-bottom: 12px; }
.tools span { border: 1px solid #c9d0d8; border-radius: 4px; padding: 4px 10px; color: #5f6b77; }
.tools span.search { flex: 1; }
.circled { position: relative; display: inline-block; }
.ring { position: absolute; left: -14px; right: -14px; top: -5px; bottom: -5px;
        border: 4px solid #12a150; border-radius: 50%; z-index: 10;
        box-shadow: 0 0 0 1.5px rgba(255,255,255,.9), inset 0 0 0 1.5px rgba(255,255,255,.9); }
.badge { position: absolute; right: -48px; top: calc(50% - 14px); width: 28px; height: 28px;
         border-radius: 50%; background: #12a150; color: #fff; font-weight: 700;
         font-size: 15px; display: flex; align-items: center; justify-content: center;
         z-index: 11; border: 2px solid #fff; }
.note { color: #0f7d3e; font-weight: 600; font-size: 11.5px; }
"""


def circled(inner: str, number: str = "", style: str = "") -> str:
    badge = f'<span class="badge">{number}</span>' if number else ""
    return f'<span class="circled" style="{style}">{inner}<span class="ring"></span>{badge}</span>'


def browser(url: str, nav: str, page: str, height: int) -> tuple[str, int]:
    return (
        f"""<body style="height:{height}px"><div class="win">
  <div class="tabsbar"><div class="btab"><div class="logo"></div>MLflow<span class="x">✕</span></div>
    <div class="ctl"><span>—</span><span>☐</span><span>✕</span></div></div>
  <div class="navbar"><span>←</span><span>→</span><span>⟳</span>
    <div class="url">{url}</div></div>
  <div class="app"><div class="nav">
    <div class="brand">ml<b>flow</b></div><div class="ver">3.16.0</div>
    <div class="mode"><span class="on">GenAI</span><span>Model training</span></div>
    {nav}</div>
    <div class="page">{page}</div></div>
</div></body>""",
        height,
    )


def item(text: str, on: bool = False) -> str:
    return f'<div class="item{" on" if on else ""}"><i></i>{text}</div>'


HOME_NAV = "".join(
    (item("Home", on=True), item("Experiments"), item("Prompts"), item("AI Gateway"))
)

HOME_PAGE = f"""<h1>Welcome to MLflow</h1>
<div class="cards">
  <div class="card"><b>Tracing</b><span>Capture and debug LLM interactions</span></div>
  <div class="card"><b>Evaluation</b><span>Measure and compare LLM quality</span></div>
  <div class="card"><b>Prompts</b><span>Version control and manage prompts</span></div>
</div>
<h2>Recent Experiments</h2>
<table>
  <tr><th>Name</th><th>Time created</th><th>Last modified</th><th>Description</th></tr>
  <tr><td class="link">{circled("llm-course-02-rag", "2", "margin-left:12px")}</td>
      <td>10/06/2026, 16:52:10</td><td>10/06/2026, 16:58:41</td><td>-</td></tr>
  <tr><td class="link">Default</td><td>10/06/2026, 16:44:24</td>
      <td>10/06/2026, 16:44:24</td><td>-</td></tr>
</table>"""

HOME = browser(
    circled('<span class="mono">127.0.0.1:5010</span>', "1", "margin-left:8px"),
    HOME_NAV,
    HOME_PAGE,
    480,
)

TRACES_NAV = f"""<div class="expbox">← llm-course-02-rag</div>
{item("Overview")}
<div class="group">Observability</div>
{circled(item("Traces", on=True), "1", "display:block;margin:4px 40px 4px 14px")}
{item("Sessions")}
<div class="group">Evaluation</div>
{item("Judges")}{item("Datasets")}{item("Evaluation runs")}
"""

ROWS = [
    (
        "2 minutes ago",
        '"Which lecture covers data leakage?"',
        '"The note that mentions data leakage is from **source: lecture_12/…',
    ),
    (
        "2 minutes ago",
        '{"question": "How do I automatically try every combination…',
        '[{"source": "lecture_13/lec_13_exercises.md", …',
    ),
    (
        "3 minutes ago",
        '{"question": "What does GridSearchCV do?", "k": 3}',
        '"The `GridSearchCV` is a method in scikit-learn that…',
    ),
]

TRACE_ROWS = "".join(
    f"<tr><td>{time}</td><td>{inp}</td><td>{out}</td></tr>"
    for time, inp, out in ROWS[1:]
)

TRACES_PAGE = f"""<div class="crumbs">Experiments › llm-course-02-rag ›</div>
<h1>Traces</h1>
<div class="tools"><span class="search">Search traces by id, input, or output</span>
  <span>Filter</span><span>Columns</span></div>
<table>
  <tr><th>Time</th><th>Input</th><th>Response</th></tr>
  <tr><td>{ROWS[0][0]}</td><td style="overflow:visible">{circled(ROWS[0][1], "2", "margin-left:12px")}</td><td>{ROWS[0][2]}</td></tr>
  {TRACE_ROWS}
</table>
<p class="note" style="margin-top:14px">click a row to open its trace: every step, its input and output, and its time</p>"""

TRACES = browser(
    '<span class="mono">127.0.0.1:5010/#/experiments/1/traces</span>',
    TRACES_NAV,
    TRACES_PAGE,
    520,
)

PAGES = {
    "02a_browser_mlflow_home": HOME,
    "02a_browser_mlflow_traces": TRACES,
}

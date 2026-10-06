"""VS Code illustrations for guide 02a: open a terminal, start the MLflow server in it.

The terminal output copies a real `mlflow server` start (MLflow 3.16, uvicorn) on its
first run, with example timestamps; the paths are those of the other Windows pictures.
"""

CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
html { zoom: 2; overflow: hidden; }
body { width: 1000px; overflow: hidden; position: relative;
       font-family: "Noto Sans", "Ubuntu", "DejaVu Sans", sans-serif; font-size: 13px;
       color: #3b3b3b; background: #dfe6f0; }
.mono { font-family: "DejaVu Sans Mono", monospace; }
.win { position: absolute; left: 20px; right: 20px; top: 18px; bottom: 18px;
       background: #fff; border-radius: 9px; border: 1px solid #b9c2d0;
       box-shadow: 0 8px 26px rgba(0,0,0,.25); display: flex; flex-direction: column; }
.title { height: 34px; background: #f8f8f8; border-bottom: 1px solid #e5e5e5;
         border-radius: 9px 9px 0 0; display: flex; align-items: center; padding: 0 12px;
         gap: 15px; font-size: 12.5px; flex: none; position: relative; }
.title .cmd { position: absolute; left: 62%; transform: translateX(-50%); width: 240px;
              height: 22px; border: 1px solid #d4d4d4; border-radius: 6px; background: #fff;
              text-align: center; line-height: 20px; color: #616161; font-size: 12px; }
.title .ctl { margin-left: auto; display: flex; gap: 22px; color: #444; font-size: 14px; }
.menu-on { background: #e1e1e1; border-radius: 4px; padding: 2px 6px; margin: 0 -6px; }
.main { flex: 1; display: flex; min-height: 0; }
.activity { width: 44px; background: #f8f8f8; border-right: 1px solid #e5e5e5;
            display: flex; flex-direction: column; align-items: center; gap: 18px;
            padding-top: 14px; flex: none; }
.activity i { width: 20px; height: 20px; border: 2px solid #8a8a8a; border-radius: 4px; }
.activity i.on { border-color: #005fb8; }
.side { width: 236px; background: #f8f8f8; border-right: 1px solid #e5e5e5; flex: none;
        padding: 8px 0; font-size: 12.5px; }
.side .head { font-size: 11px; letter-spacing: .4px; padding: 2px 16px 8px; color: #555; }
.side .root { font-weight: 700; font-size: 11px; padding: 3px 10px; }
.tree { padding: 3px 0; white-space: nowrap; overflow: hidden; }
.editor { flex: 1; display: flex; flex-direction: column; min-width: 0; }
.tabs { height: 34px; background: #f8f8f8; border-bottom: 1px solid #e5e5e5; flex: none;
        display: flex; }
.tab { background: #fff; border-top: 2px solid #005fb8; border-right: 1px solid #e5e5e5;
       padding: 0 12px; display: flex; align-items: center; gap: 8px; font-size: 12.5px; }
.nb { width: 13px; height: 15px; border: 1.5px solid #e37933; border-radius: 2px; flex: none; }
.cells { padding: 16px 22px 0 40px; }
.cells h1 { font-size: 20px; font-weight: 600; color: #222; margin-bottom: 8px; }
.cells p { margin-bottom: 12px; }
.panel { flex: none; border-top: 1px solid #e5e5e5; display: flex; flex-direction: column; }
.ptabs { height: 30px; display: flex; align-items: center; gap: 20px; padding: 0 14px;
         font-size: 11px; letter-spacing: .3px; color: #6f6f6f; flex: none; }
.ptabs .on { color: #3b3b3b; border-bottom: 1px solid #005fb8; padding-bottom: 3px; }
.ptabs .shell { margin-left: auto; letter-spacing: 0; font-size: 12px; color: #3b3b3b; }
.term { padding: 8px 14px 10px; font-size: 11px; line-height: 2.05; white-space: pre;
        color: #1f1f1f; overflow: hidden; }
.dim { color: #6f6f6f; }
.cur { display: inline-block; width: 7px; height: 13px; background: #1f1f1f;
       vertical-align: -2px; }
.status { height: 22px; background: #f8f8f8; border-top: 1px solid #e5e5e5; flex: none;
          border-radius: 0 0 9px 9px; display: flex; align-items: center; padding: 0 12px;
          gap: 16px; font-size: 11.5px; color: #555; }
.dropdown { position: absolute; width: 330px; background: #f8f8f8; border: 1px solid #cecece;
            border-radius: 7px; box-shadow: 0 4px 18px rgba(0,0,0,.28); padding: 5px;
            z-index: 5; font-size: 12.5px; }
.dropdown div { display: flex; padding: 4px 12px; border-radius: 4px; }
.dropdown div.on { background: #0060c0; color: #fff; }
.dropdown div span { margin-left: auto; color: #6f6f6f; font-size: 12px; }
.dropdown div.on span { color: #fff; }
.dropdown hr { border: none; border-top: 1px solid #dcdcdc; margin: 4px 8px; }
.circled { position: relative; display: inline-block; }
.ring { position: absolute; left: -14px; right: -14px; top: -5px; bottom: -5px;
        border: 4px solid #12a150; border-radius: 50%; z-index: 10;
        box-shadow: 0 0 0 1.5px rgba(255,255,255,.9), inset 0 0 0 1.5px rgba(255,255,255,.9); }
.badge { position: absolute; right: -48px; top: calc(50% - 14px); width: 28px; height: 28px;
         border-radius: 50%; background: #12a150; color: #fff; font-weight: 700;
         font-size: 15px; display: flex; align-items: center; justify-content: center;
         z-index: 11; border: 2px solid #fff; font-family: "Noto Sans", sans-serif; }
.note { color: #0f7d3e; font-weight: 600; font-size: 11.5px; margin-left: 58px;
        font-family: "Noto Sans", sans-serif; }
"""

FOLDER = r"C:\Users\you\courses\teach-llm-system"


def circled(inner: str, number: str = "", style: str = "") -> str:
    badge = f'<span class="badge">{number}</span>' if number else ""
    return f'<span class="circled" style="{style}">{inner}<span class="ring"></span>{badge}</span>'


def tree(indent: int, text: str) -> str:
    return f'<div class="tree" style="padding-left:{12 + 12 * indent}px">{text}</div>'


def window(menu: str, side: str, editor: str, panel: str, height: int) -> str:
    return f"""<body style="height:{height}px"><div class="win">
  <div class="title">{menu}<div class="cmd">teach-llm-system</div>
    <div class="ctl"><span>—</span><span>☐</span><span>✕</span></div></div>
  <div class="main">
    <div class="activity"><i class="on"></i><i></i><i></i><i></i><i></i></div>
    {side}
    <div class="editor">
      <div class="tabs"><div class="tab"><div class="nb"></div>lec_02b_hand_built_rag_traced.ipynb
        <span>✕</span></div></div>
      {editor}
      <div style="flex:1"></div>
      {panel}
    </div>
  </div>
  <div class="status"><span>⑂ main</span><span>⊗ 0 ⚠ 0</span></div>
</div>"""


MENU_ITEMS = ["File", "Edit", "Selection", "View", "Go", "Run", "Terminal", "Help"]


def menu(open_item: str = "") -> str:
    items = []
    for item in MENU_ITEMS:
        if item == open_item:
            items.append(
                circled(
                    f'<span class="menu-on">{item}</span>', "1", "margin:0 44px 0 4px"
                )
            )
        else:
            items.append(f"<span>{item}</span>")
    return "".join(items)


SIDE = f"""<div class="side">
  <div class="head">EXPLORER</div>
  <div class="root">⌄ TEACH-LLM-SYSTEM</div>
  {tree(0, "›&nbsp; .venv")}
  {tree(0, "›&nbsp; corpus")}
  {tree(0, "›&nbsp; data")}
  {tree(0, "›&nbsp; lecture_01_ollama_models_pro…")}
  {tree(0, "›&nbsp; lecture_02_embeddings_rag_ve…")}
  {tree(0, "›&nbsp; llm_course")}
  {tree(0, "›&nbsp; mlflow_server")}
  {tree(0, "&nbsp;&nbsp; .env")}
  {tree(0, "&nbsp;&nbsp; pyproject.toml")}
  {tree(0, "&nbsp;&nbsp; README.md")}
</div>"""

EDITOR = """<div class="cells">
  <h1>Lecture 02b: A hand-built RAG, traced</h1>
  <p><b>Status: Mandatory reading.</b></p>
</div>"""

DROPDOWN = f"""<div class="dropdown" style="left:318px;top:46px">
  {circled('<div class="on" style="width:318px">New Terminal<span>Ctrl+Shift+`</span></div>', "2", "margin:0;display:block")}
  <div>Split Terminal<span>Ctrl+Shift+5</span></div>
  <div>New Terminal Window<span>Ctrl+Shift+Alt+`</span></div>
  <hr>
  <div>Run Task...</div>
  <div>Run Build Task...<span>Ctrl+Shift+B</span></div>
  <div>Run Active File</div>
  <div>Run Selected Text</div>
  <hr>
  <div>Configure Tasks...</div>
</div>"""

NEW_TERMINAL = (
    window(menu("Terminal"), SIDE, EDITOR, "", 470) + DROPDOWN + "</body>",
    470,
)

SERVER_LINES = [
    f"PS {FOLDER}&gt; " + circled("cd mlflow_server", "1", "margin-left:14px"),
    f"PS {FOLDER}\\mlflow_server&gt; "
    + circled(
        "uv run mlflow server --host 127.0.0.1 --port 5010", "2", "margin-left:14px"
    ),
    "Backend store URI not provided. Using sqlite:///mlflow.db",
    "Registry store URI not provided. Using backend store URI.",
    '<span class="dim">2026/10/06 16:44:24 INFO mlflow.store.db.utils: Creating initial MLflow database tables...</span>',
    '<span class="dim">2026/10/06 16:44:24 INFO mlflow.store.db.utils: Updating database tables</span>',
    '<span class="dim">[MLflow] Security middleware enabled with default settings (localhost-only). To allow connections</span>',
    '<span class="dim">from other hosts, use --host 0.0.0.0 and configure --allowed-hosts and --cors-allowed-origins.</span>',
    "2026/10/06 16:44:26 INFO:     "
    + circled("Uvicorn running on http://127.0.0.1:5010 (Press CTRL+C to quit)", "3"),
    "2026/10/06 16:44:26 INFO:     Started parent process [8124]",
    "2026/10/06 16:44:28 INFO:     Application startup complete."
    + '<span class="note">the server keeps running here: leave this terminal open</span>',
    '<span class="cur"></span>',
]

SERVER_PANEL = f"""<div class="panel">
  <div class="ptabs"><span>PROBLEMS</span><span>OUTPUT</span><span>DEBUG CONSOLE</span>
    <span class="on">TERMINAL</span><span>PORTS</span><span class="shell">⌄ powershell</span></div>
  <div class="term mono">{"<br>".join(SERVER_LINES)}</div>
</div>"""

MLFLOW_SERVER = (
    window(menu(), "", EDITOR, SERVER_PANEL, 520) + "</body>",
    520,
)

PAGES = {
    "02a_vscode_new_terminal": NEW_TERMINAL,
    "02a_vscode_mlflow_server": MLFLOW_SERVER,
}

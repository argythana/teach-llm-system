"""VS Code illustration for guide 01a: selecting the course kernel for a notebook."""

CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
html { zoom: 2; overflow: hidden; }
body { width: 1000px; height: 520px; overflow: hidden; position: relative;
       font-family: "Noto Sans", "Ubuntu", "DejaVu Sans", sans-serif; font-size: 13px;
       color: #3b3b3b; background: #dfe6f0; }
.mono { font-family: "DejaVu Sans Mono", monospace; }
.win { position: absolute; left: 20px; right: 20px; top: 18px; bottom: 18px;
       background: #fff; border-radius: 9px; border: 1px solid #b9c2d0;
       box-shadow: 0 8px 26px rgba(0,0,0,.25); overflow: hidden; display: flex;
       flex-direction: column; }
.title { height: 34px; background: #f8f8f8; border-bottom: 1px solid #e5e5e5;
         display: flex; align-items: center; padding: 0 12px; gap: 14px; font-size: 12.5px;
         flex: none; position: relative; }
.title .cmd { position: absolute; left: 50%; transform: translateX(-50%); width: 330px;
              height: 22px; border: 1px solid #d4d4d4; border-radius: 6px; background: #fff;
              text-align: center; line-height: 20px; color: #616161; font-size: 12px; }
.title .ctl { margin-left: auto; display: flex; gap: 22px; color: #444; font-size: 14px; }
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
.tree { padding: 3px 0 3px 0; white-space: nowrap; overflow: hidden; }
.tree.on { background: #e4e6f1; }
.editor { flex: 1; display: flex; flex-direction: column; min-width: 0; }
.tabs { height: 34px; background: #f8f8f8; border-bottom: 1px solid #e5e5e5; flex: none;
        display: flex; }
.tab { background: #fff; border-top: 2px solid #005fb8; border-right: 1px solid #e5e5e5;
       padding: 0 12px; display: flex; align-items: center; gap: 8px; font-size: 12.5px; }
.nb { width: 13px; height: 15px; border: 1.5px solid #e37933; border-radius: 2px; flex: none; }
.bar { height: 36px; display: flex; align-items: center; gap: 16px; padding: 0 26px 0 14px;
       border-bottom: 1px solid #ececec; font-size: 12.5px; flex: none; }
.kernel { margin-left: auto; display: flex; align-items: center; gap: 7px; }
.kernel i { width: 13px; height: 13px; border: 1.5px solid #3b3b3b; border-radius: 3px; }
.cells { padding: 16px 22px 0 40px; }
.cells h1 { font-size: 21px; font-weight: 600; color: #222; margin-bottom: 8px; }
.cells p { margin-bottom: 14px; }
.code { border: 1px solid #dcdcdc; border-radius: 4px; background: #f6f6f6;
        padding: 9px 12px; font-size: 12px; line-height: 1.6; position: relative; }
.code:before { content: "▷"; position: absolute; left: -24px; top: 8px; color: #666; }
.kw { color: #af00db; }
.status { height: 22px; background: #f8f8f8; border-top: 1px solid #e5e5e5; flex: none;
          display: flex; align-items: center; padding: 0 12px; gap: 16px; font-size: 11.5px;
          color: #555; }
.pick { position: absolute; left: 50%; transform: translateX(-50%); top: 22px; width: 500px;
        background: #f8f8f8; border: 1px solid #cecece; border-radius: 7px;
        box-shadow: 0 4px 18px rgba(0,0,0,.28); padding: 6px; z-index: 5; }
.pick .ttl { display: flex; align-items: center; justify-content: center; height: 24px;
             font-size: 12.5px; position: relative; }
.pick .ttl span { position: absolute; left: 8px; font-size: 15px; color: #555; }
.pick .in { height: 26px; border: 1px solid #005fb8; border-radius: 4px; background: #fff;
            margin: 4px 0 6px; padding: 0 8px; line-height: 24px; color: #8a8a8a;
            font-size: 12.5px; }
.item { display: flex; align-items: baseline; gap: 10px; padding: 5px 10px;
        border-radius: 4px; font-size: 12.5px; }
.item.on { background: #e8e8e8; }
.item small { color: #6f6f6f; font-size: 11.5px; }
.item em { margin-left: auto; font-style: normal; color: #005fb8; font-size: 11.5px; }
.ring { position: absolute; border: 5px solid #12a150; border-radius: 50%;
        box-shadow: 0 0 0 1.5px rgba(255,255,255,.9), inset 0 0 0 1.5px rgba(255,255,255,.9);
        z-index: 10; }
.badge { position: absolute; width: 28px; height: 28px; border-radius: 50%;
         background: #12a150; color: #fff; font-weight: 700; font-size: 15px;
         display: flex; align-items: center; justify-content: center; z-index: 11;
         border: 2px solid #fff; }
"""


def tree(indent: int, text: str, on: bool = False) -> str:
    return f'<div class="tree{" on" if on else ""}" style="padding-left:{12 + 12 * indent}px">{text}</div>'


def ring(left: int, top: int, width: int, height: int) -> str:
    return f'<div class="ring" style="left:{left}px;top:{top}px;width:{width}px;height:{height}px"></div>'


def badge(left: int, top: int, text: str) -> str:
    return f'<div class="badge" style="left:{left}px;top:{top}px">{text}</div>'


SIDE = "".join(
    (
        tree(0, "›&nbsp; .venv"),
        tree(0, "›&nbsp; corpus"),
        tree(0, "⌄&nbsp; lecture_01_ollama_models_pro…"),
        tree(1, "›&nbsp; infra_tools"),
        tree(1, "›&nbsp; practice_exercises"),
        tree(1, "⌄&nbsp; reading_material"),
        tree(2, "&nbsp;&nbsp; goals_01.md"),
        tree(2, "&nbsp;&nbsp; lec_01a_first_call_to…", on=True),
        tree(2, "&nbsp;&nbsp; lec_01b_choosing_a_mo…"),
        tree(2, "&nbsp;&nbsp; lec_01c_prompts_roles_…"),
        tree(0, "›&nbsp; lecture_02_embeddings_rag_ve…"),
        tree(0, "›&nbsp; llm_course"),
        tree(0, "&nbsp;&nbsp; pyproject.toml"),
        tree(0, "&nbsp;&nbsp; README.md"),
        tree(0, "&nbsp;&nbsp; uv.lock"),
    )
)

BODY = f"""
<body>
  <div class="win">
    <div class="title"><span>File</span><span>Edit</span><span>Selection</span><span>···</span>
      <div class="cmd">teach-llm-system</div>
      <div class="ctl"><span>—</span><span>☐</span><span>✕</span></div>
    </div>
    <div class="main">
      <div class="activity"><i class="on"></i><i></i><i></i><i></i><i></i></div>
      <div class="side">
        <div class="head">EXPLORER</div>
        <div class="root">⌄ TEACH-LLM-SYSTEM</div>
        {SIDE}
      </div>
      <div class="editor">
        <div class="tabs"><div class="tab"><div class="nb"></div>lec_01a_first_call_tokens_context.ipynb
          <span>✕</span></div></div>
        <div class="bar"><span>＋ Code</span><span>＋ Markdown</span><span>▷ Run All</span>
          <span>☰ Outline</span><span>···</span>
          <div class="kernel"><i></i>Select Kernel</div></div>
        <div class="cells">
          <h1>Lecture 01a: Talking to a local model</h1>
          <p><b>Status: Mandatory reading.</b></p>
          <p>In the Python course you trained models yourself: <span class="mono">fit</span>,
            <span class="mono">predict</span>, a metric. A <b>large language model (LLM)</b> is
            different: someone else trained it, and your job is to <i>run</i> it and
            <i>talk</i> to it.</p>
          <div class="code mono"><span class="kw">import</span> ollama<br>
            <span class="kw">import</span> pandas <span class="kw">as</span> pd<br>
            <span class="kw">from</span> transformers <span class="kw">import</span> AutoTokenizer</div>
        </div>
        <div style="flex:1"></div>
      </div>
    </div>
    <div class="status"><span>⑂ main</span><span>⊗ 0 ⚠ 0</span>
      <span style="margin-left:auto">Cell 1 of 35</span></div>
  </div>
  <div class="pick">
    <div class="ttl"><span>←</span>Select a Python Environment</div>
    <div class="in">Type to choose a Python environment</div>
    <div class="item on">★ teach-llm-system (3.12.7)
      <small class="mono">.venv\\Scripts\\python.exe</small><em>Recommended</em></div>
    <div class="item">Python 3.13.5
      <small class="mono">C:\\Python313\\python.exe</small><em>Global Env</em></div>
    <div class="item">Python 3.11.9
      <small class="mono">C:\\Python311\\python.exe</small><em>Global Env</em></div>
  </div>
  {ring(824, 84, 150, 42)}{badge(808, 70, "1")}
  {ring(242, 82, 400, 44)}{badge(228, 68, "2")}
</body>"""

PAGES = {"01a_vscode_select_kernel": (BODY, 520)}

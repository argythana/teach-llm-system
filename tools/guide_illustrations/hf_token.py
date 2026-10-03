"""Illustrations for guide 01d: create a Hugging Face read token and put it in .env."""

CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
html { zoom: 2; overflow: hidden; }
body { width: 1000px; overflow: hidden; position: relative; background: #dfe6f0;
       font-family: "Noto Sans", "Ubuntu", "DejaVu Sans", sans-serif; font-size: 13px;
       color: #1f2937; }
.mono { font-family: "DejaVu Sans Mono", monospace; }
.frame { position: absolute; left: 20px; right: 20px; top: 18px; bottom: 18px;
         background: #fff; border-radius: 9px; border: 1px solid #b9c2d0;
         box-shadow: 0 8px 26px rgba(0,0,0,.25); }
.circled { position: relative; display: inline-block; }
.ring { position: absolute; left: -20px; right: -20px; top: -11px; bottom: -11px;
        border: 5px solid #12a150; border-radius: 50%; z-index: 10; pointer-events: none;
        box-shadow: 0 0 0 1.5px rgba(255,255,255,.9), inset 0 0 0 1.5px rgba(255,255,255,.9); }
.badge.top { left: calc(50% - 14px); top: -36px; }
.badge.right { left: auto; right: -52px; }
.badge { position: absolute; left: -52px; top: calc(50% - 14px); width: 28px; height: 28px;
         border-radius: 50%; background: #12a150; color: #fff; font-weight: 700;
         font-size: 15px; display: flex; align-items: center; justify-content: center;
         z-index: 11; border: 2px solid #fff; font-family: "Noto Sans", sans-serif; }
.note { color: #0b7a3b; font-weight: 600; font-size: 12px;
        font-family: "Noto Sans", sans-serif; }

/* browser + Hugging Face page */
.chrome { height: 34px; background: #eef1f5; border-bottom: 1px solid #d5dae2;
          border-radius: 9px 9px 0 0; display: flex; align-items: center; padding: 0 12px;
          gap: 10px; }
.dots { display: flex; gap: 6px; }
.dots i { width: 10px; height: 10px; border-radius: 50%; background: #c3c9d3; }
.url { flex: 1; height: 22px; background: #fff; border: 1px solid #d5dae2;
       border-radius: 11px; padding: 0 12px; line-height: 20px; font-size: 12px;
       color: #4b5563; }
.hfnav { height: 44px; border-bottom: 1px solid #eceff3; display: flex; align-items: center;
         padding: 0 18px; gap: 18px; color: #374151; }
.hflogo { display: flex; align-items: center; gap: 7px; font-weight: 700; color: #111; }
.hflogo i { width: 20px; height: 20px; border-radius: 50%; background: #ffd21e;
            border: 2px solid #ff9d00; }
.hfsearch { width: 230px; height: 26px; border: 1px solid #e0e4ea; border-radius: 7px;
            padding: 0 10px; line-height: 24px; color: #9ca3af; font-size: 12px; }
.avatar { width: 24px; height: 24px; border-radius: 50%; background: #9aa6bd; }
.hfbody { display: flex; gap: 34px; padding: 18px 22px; }
.menu { width: 210px; border: 1px solid #eceff3; border-radius: 9px; flex: none;
        background: #fafbfc; }
.who { display: flex; align-items: center; gap: 10px; padding: 12px 14px;
       border-bottom: 1px solid #eceff3; }
.who .avatar { width: 34px; height: 34px; }
.who b { display: block; font-size: 13.5px; }
.who span { font-size: 11.5px; background: #eef1f5; border-radius: 4px; padding: 0 4px; }
.menu div.it { padding: 8px 14px; border-bottom: 1px solid #eceff3; color: #4b5563; }
.menu div.it.on { font-weight: 700; color: #111; background: #f1f3f6; }
.menu div.it:last-child { border-bottom: 0; }
.hfmain { flex: 1; }
.hfmain h1 { font-size: 19px; margin-bottom: 14px; color: #111; }
.hfmain h2 { font-size: 14.5px; color: #111; }
.hfmain p { color: #4b5563; line-height: 1.55; margin: 8px 0 14px; max-width: 520px; }
.rowhead { display: flex; align-items: center; justify-content: space-between;
           padding-right: 40px; }
.btn { display: inline-block; border: 1px solid #d5dae2; border-radius: 7px;
       padding: 6px 12px; background: #fff; color: #111; font-size: 13px;
       box-shadow: 0 1px 1px rgba(0,0,0,.06); }
.btn.dark { background: #111827; color: #fff; border-color: #111827; }
.thead { display: grid; grid-template-columns: 1.2fr 1fr 1fr 1fr 1fr; color: #6b7280;
         font-size: 12px; border-bottom: 1px solid #eceff3; padding: 8px 0; margin-right: 40px; }
.empty { color: #9ca3af; padding: 14px 0; font-size: 12.5px; }
.seg { display: inline-flex; border: 1px solid #d5dae2; border-radius: 8px; padding: 3px;
       gap: 3px; background: #f3f4f6; }
.seg .opt { padding: 5px 16px; border-radius: 6px; color: #4b5563; display: inline-block; }
.seg .opt.on { background: #fff; color: #111; font-weight: 600;
               box-shadow: 0 1px 2px rgba(0,0,0,.15); }
.lab { font-size: 12.5px; color: #374151; margin: 14px 0 6px; font-weight: 600; }
.field { width: 330px; height: 32px; border: 1px solid #d5dae2; border-radius: 7px;
         padding: 0 10px; line-height: 30px; display: inline-block; }
.shade { position: absolute; left: 0; right: 0; top: 34px; bottom: 0;
         background: rgba(17,24,39,.45); border-radius: 0 0 9px 9px; }
.modal { position: absolute; left: 50%; top: 74px; transform: translateX(-50%);
         width: 500px; background: #fff; border-radius: 10px; padding: 20px 22px;
         box-shadow: 0 10px 30px rgba(0,0,0,.35); }
.modal h2 { font-size: 16px; margin-bottom: 8px; color: #111; }
.modal p { color: #4b5563; line-height: 1.5; margin-bottom: 12px; }
.tokbox { border: 1px solid #d5dae2; border-radius: 7px; padding: 8px 10px;
          background: #f9fafb; font-size: 12.5px; margin-bottom: 14px; }
.modal .acts { display: flex; gap: 46px; align-items: center; padding-left: 8px; }

/* PowerShell */
.psbar { height: 32px; background: #1f1f1f; color: #e5e5e5; border-radius: 9px 9px 0 0;
         display: flex; align-items: center; padding: 0 12px; gap: 9px; font-size: 12.5px; }
.psbar .ico { width: 16px; height: 16px; background: #012456; border: 1px solid #5b7bb0;
              border-radius: 3px; }
.psbar .ctl { margin-left: auto; display: flex; gap: 22px; font-size: 13px; }
.psbody { position: absolute; left: 0; right: 0; top: 32px; bottom: 0; background: #012456;
          color: #eeedf0; border-radius: 0 0 9px 9px; padding: 16px 18px; font-size: 13px;
          line-height: 2.5; }
.psbody .cmd { color: #f9f1a5; }
.cur { display: inline-block; width: 8px; height: 15px; background: #eeedf0;
       vertical-align: -2px; }

/* VS Code */
.vs { display: flex; flex-direction: column; }
.vtitle { height: 32px; background: #f8f8f8; border-bottom: 1px solid #e5e5e5;
          border-radius: 9px 9px 0 0; display: flex; align-items: center; padding: 0 12px;
          gap: 14px; font-size: 12.5px; color: #3b3b3b; flex: none; position: relative; }
.vtitle .cmdc { position: absolute; left: 50%; transform: translateX(-50%); width: 300px;
                height: 22px; border: 1px solid #d4d4d4; border-radius: 6px; background: #fff;
                text-align: center; line-height: 20px; color: #616161; font-size: 12px; }
.vtitle .ctl { margin-left: auto; display: flex; gap: 22px; font-size: 14px; }
.vmain { flex: 1; display: flex; min-height: 0; }
.vact { width: 44px; background: #f8f8f8; border-right: 1px solid #e5e5e5; display: flex;
        flex-direction: column; align-items: center; gap: 18px; padding-top: 14px; flex: none;
        border-radius: 0 0 0 9px; }
.vact i { width: 20px; height: 20px; border: 2px solid #8a8a8a; border-radius: 4px; }
.vact i.on { border-color: #005fb8; }
.vside { width: 236px; background: #f8f8f8; border-right: 1px solid #e5e5e5; flex: none;
         padding: 8px 0; font-size: 12.5px; color: #3b3b3b; }
.vside .head { font-size: 11px; letter-spacing: .4px; padding: 2px 16px 8px; color: #555; }
.vside .root { font-weight: 700; font-size: 11px; padding: 3px 10px; }
.tree { padding: 3px 0 3px 12px; white-space: nowrap; }
.tree.on { background: #e4e6f1; }
.ved { flex: 1; display: flex; flex-direction: column; min-width: 0; }
.vtabs { height: 34px; background: #f8f8f8; border-bottom: 1px solid #e5e5e5; flex: none;
         display: flex; }
.vtab { background: #fff; border-top: 2px solid #005fb8; border-right: 1px solid #e5e5e5;
        padding: 0 14px; display: flex; align-items: center; gap: 10px; font-size: 12.5px; }
.code { padding: 14px 0; font-size: 12.5px; line-height: 3.1; color: #3b3b3b; }
.ln { display: flex; white-space: nowrap; }
.ln > b { width: 46px; text-align: right; padding-right: 16px; color: #9a9a9a;
          font-weight: 400; flex: none; }
.cm { color: #008000; }
.key { color: #0451a5; }
"""


def circled(inner: str, number: str = "", style: str = "", pos: str = "") -> str:
    badge = f'<span class="badge {pos}">{number}</span>' if number else ""
    return f'<span class="circled" style="{style}">{inner}<span class="ring"></span>{badge}</span>'


def browser(url: str, content: str, overlay: str = "") -> str:
    return f"""
<div class="frame">
  <div class="chrome"><div class="dots"><i></i><i></i><i></i></div><div class="url">{url}</div></div>
  <div class="hfnav"><div class="hflogo"><i></i>Hugging Face</div>
    <div class="hfsearch">Search models, datasets, users...</div>
    <span style="margin-left:auto">Models</span><span>Datasets</span><span>Spaces</span>
    <span>Docs</span><span>Pricing</span><div class="avatar"></div></div>
  <div class="hfbody">
    <div class="menu">
      <div class="who"><div class="avatar"></div><div><b>Your Name</b><span class="mono">your-username</span></div></div>
      <div class="it">Profile</div><div class="it">Account</div><div class="it">Authentication</div>
      <div class="it">Billing</div><div class="it on">Access Tokens</div>
    </div>
    <div class="hfmain">{content}</div>
  </div>
  {overlay}
</div>"""


def tokens_list(circle: bool) -> str:
    button = '<span class="btn">+ Create new token</span>'
    return f"""
<h1>Access Tokens</h1>
<div class="rowhead"><h2>User Access Tokens</h2>
  {circled(button) if circle else button}</div>
<p>Access tokens authenticate your identity to the Hugging Face Hub and allow applications
  to perform actions based on token permissions.</p>
<div class="thead"><span>Name</span><span>Value</span><span>Last Refreshed Date</span>
  <span>Last Used Date</span><span>Permissions</span></div>
<div class="empty">You have no access tokens yet.</div>
"""


CREATE_FORM = f"""
<h1>Create new Access Token</h1>
<div class="lab" style="margin-top:0">Token type</div>
<div class="seg" style="margin-top:26px"><span class="opt">Fine-grained</span>{circled('<span class="opt on">Read</span>', "1", "margin:0 22px", "top")}<span class="opt">Write</span></div>
<p style="margin:10px 0 0">This token has read-only access to all your and your orgs resources.</p>
<div class="lab">Token name</div>
{circled('<span class="field mono">teach-llm-system</span>', "2", "margin:6px 0 0 56px")}
<div style="margin-top:26px;padding-left:56px">{circled('<span class="btn dark">Create token</span>', "3")}</div>
"""

COPY_MODAL = f"""
<div class="shade"></div>
<div class="modal">
  <h2>Save your Access Token</h2>
  <p>Save your token value somewhere safe. You will not be able to see it again after you
    close this window.</p>
  <div class="tokbox mono">hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx</div>
  <div class="acts">{circled('<span class="btn">⧉ Copy</span>')}<span class="btn dark">Done</span></div>
</div>
"""

POWERSHELL = f"""
<div class="frame" style="background:#012456;border-color:#333">
  <div class="psbar"><div class="ico"></div>Windows PowerShell
    <div class="ctl"><span>—</span><span>☐</span><span>✕</span></div></div>
  <div class="psbody mono">
    <div>PS C:\\Users\\you\\courses\\teach-llm-system&gt;
      {circled('<span class="cmd">Copy-Item</span> .env.example .env', style="margin-left:22px")}</div>
    <div>PS C:\\Users\\you\\courses\\teach-llm-system&gt; <span class="cur"></span></div>
  </div>
</div>"""


def tree(indent: int, text: str, on: bool = False) -> str:
    return f'<div class="tree{" on" if on else ""}" style="padding-left:{12 + 12 * indent}px">{text}</div>'


VS_SIDE = "".join(
    (
        tree(0, "›&nbsp; .venv"),
        tree(0, "›&nbsp; lecture_01_ollama_models_pro…"),
        tree(0, "›&nbsp; lecture_02_embeddings_rag_ve…"),
        tree(0, "›&nbsp; llm_course"),
        f'<div class="tree on" style="padding:6px 0 6px 36px;margin:12px 0">{circled(".env", "1", "padding-right:60px", "right")}</div>',
        tree(0, "&nbsp;&nbsp;&nbsp; .env.example"),
        tree(0, "&nbsp;&nbsp;&nbsp; .gitignore"),
        tree(0, "&nbsp;&nbsp;&nbsp; pyproject.toml"),
        tree(0, "&nbsp;&nbsp;&nbsp; README.md"),
        tree(0, "&nbsp;&nbsp;&nbsp; uv.lock"),
    )
)

TOKEN_LINE = circled(
    '<span class="key">HF_TOKEN</span>=hf_xxxxxxxxxxxxxxxxxxxxxxxx',
    "2",
    "margin-left:10px",
    "right",
)

VSCODE = f"""
<div class="frame vs">
  <div class="vtitle"><span>File</span><span>Edit</span><span>Selection</span><span>···</span>
    <div class="cmdc">teach-llm-system</div>
    <div class="ctl"><span>—</span><span>☐</span><span>✕</span></div></div>
  <div class="vmain">
    <div class="vact"><i class="on"></i><i></i><i></i><i></i></div>
    <div class="vside"><div class="head">EXPLORER</div>
      <div class="root">⌄ TEACH-LLM-SYSTEM</div>{VS_SIDE}</div>
    <div class="ved">
      <div class="vtabs"><div class="vtab">⚙ .env <span>✕</span></div></div>
      <div class="code mono">
        <div class="ln"><b>1</b><span class="cm"># Copy to `.env` (gitignored) and fill in the values.</span></div>
        <div class="ln"><b>2</b><span class="cm"># Every notebook's configuration cell loads this file.</span></div>
        <div class="ln"><b>3</b>{TOKEN_LINE}
          <span class="note" style="margin-left:70px">← example only: paste your own token here</span></div>
        <div class="ln"><b>4</b><span><span class="key">OLLAMA_HOST</span>=http://localhost:11434</span></div>
        <div class="ln"><b>5</b><span><span class="key">MLFLOW_TRACKING_URI</span>=http://127.0.0.1:5010</span></div>
      </div>
    </div>
  </div>
</div>"""


def page(content: str, height: int) -> tuple[str, int]:
    return f'<body style="height:{height}px">{content}</body>', height


TOKENS_URL = "huggingface.co/settings/tokens"

PAGES = {
    "01d_hf_tokens_page": page(browser(TOKENS_URL, tokens_list(True)), 400),
    "01d_hf_create_read_token": page(browser(f"{TOKENS_URL}/new", CREATE_FORM), 440),
    "01d_hf_copy_token": page(browser(TOKENS_URL, tokens_list(False), COPY_MODAL), 400),
    "01d_powershell_copy_env": page(POWERSHELL, 190),
    "01d_vscode_env_token": page(VSCODE, 400),
}

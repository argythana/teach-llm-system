"""Windows illustrations for guide 01a: the Start menu search and File Explorer."""

CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
html { zoom: 2; overflow: hidden; }
body { overflow: hidden; }
body { font-family: "Noto Sans", "Ubuntu", "DejaVu Sans", sans-serif; font-size: 14px;
       color: #1b1b1b; position: relative; }
.desktop { background: linear-gradient(135deg, #7fb2e6 0%, #3d7fd0 45%, #1f4f9f 100%); }
.taskbar { position: absolute; left: 0; right: 0; bottom: 0; height: 48px;
           background: #e9eef6; border-top: 1px solid #cfd6e2; display: flex;
           align-items: center; justify-content: center; gap: 10px; }
.tb-icon { width: 30px; height: 30px; border-radius: 6px; }
.win { display: grid; grid-template-columns: 1fr 1fr; grid-template-rows: 1fr 1fr;
       gap: 2px; padding: 5px; }
.win i { background: #0a6cd6; border-radius: 1px; }
.tb-search { width: 190px; height: 32px; border-radius: 16px; background: #fff;
             border: 1px solid #c9d0dc; display: flex; align-items: center;
             padding: 0 12px; color: #1b1b1b; gap: 8px; }
.lens { width: 12px; height: 12px; border: 2px solid #444; border-radius: 50%;
        position: relative; flex: none; }
.lens:after { content: ""; position: absolute; width: 6px; height: 2px; background: #444;
              right: -6px; bottom: -3px; transform: rotate(45deg); }
.panel { position: absolute; left: 50%; transform: translateX(-50%); top: 22px;
         width: 780px; height: 560px; background: #f3f5f9; border-radius: 10px;
         border: 1px solid #c8cfdb; box-shadow: 0 10px 34px rgba(0,0,0,.35);
         padding: 18px 20px; }
.searchbox { height: 40px; border-radius: 20px; background: #fff;
             border: 1px solid #c9d0dc; border-bottom: 2px solid #0a6cd6; display: flex;
             align-items: center; padding: 0 16px; gap: 10px; font-size: 15px; }
.caret { width: 1px; height: 18px; background: #111; margin-left: -8px; }
.tabs { display: flex; gap: 20px; margin: 14px 4px 12px; font-size: 13px; color: #333; }
.tabs b { border-bottom: 3px solid #0a6cd6; padding-bottom: 4px; }
.cols { display: grid; grid-template-columns: 330px 1fr; gap: 14px; height: 440px; }
.label { font-size: 12.5px; font-weight: 600; margin: 8px 6px 6px; }
.row { display: flex; align-items: center; gap: 12px; padding: 7px 10px;
       border-radius: 6px; font-size: 13.5px; }
.row.best { background: #dde6f3; padding: 10px; }
.row small { display: block; color: #555; font-size: 12px; }
.chev { margin-left: auto; color: #666; }
.ps { background: #012456; border-radius: 5px; color: #fff; font-family: "DejaVu Sans Mono",
      monospace; font-weight: 700; display: flex; align-items: center;
      justify-content: center; flex: none; }
.ps.s { width: 22px; height: 22px; font-size: 10px; }
.ps.m { width: 34px; height: 34px; font-size: 14px; }
.ps.l { width: 72px; height: 72px; font-size: 28px; border-radius: 10px; }
.ico { width: 22px; height: 22px; border-radius: 4px; flex: none; }
.right { background: #fff; border-radius: 8px; border: 1px solid #dfe3ea;
         padding: 26px 18px 10px; }
.hero { display: flex; flex-direction: column; align-items: center; gap: 8px;
        padding-bottom: 18px; border-bottom: 1px solid #e1e4ea; margin-bottom: 10px; }
.hero h2 { font-size: 17px; font-weight: 500; }
.hero span { color: #555; font-size: 13px; }
.act { display: flex; align-items: center; gap: 12px; padding: 8px 10px; font-size: 13.5px; }
.glyph { width: 18px; height: 18px; border: 1.6px solid #0a6cd6; border-radius: 3px;
         flex: none; position: relative; }
.glyph.shield { border-radius: 3px 3px 9px 9px; }
.ring { position: absolute; border: 5px solid #12a150; border-radius: 50%;
        box-shadow: 0 0 0 1.5px rgba(255,255,255,.9), inset 0 0 0 1.5px rgba(255,255,255,.9);
        z-index: 10; }
.badge { position: absolute; width: 30px; height: 30px; border-radius: 50%;
         background: #12a150; color: #fff; font-weight: 700; font-size: 16px;
         display: flex; align-items: center; justify-content: center; z-index: 11;
         border: 2px solid #fff; }

/* File Explorer */
.explorer-bg { background: #dfe6f0; }
.exp { position: absolute; left: 24px; right: 24px; top: 20px; bottom: 20px;
       background: #fff; border-radius: 9px; border: 1px solid #b9c2d0;
       box-shadow: 0 8px 26px rgba(0,0,0,.25); overflow: hidden; }
.titlebar { height: 40px; background: #e6edf7; display: flex; align-items: flex-end;
            padding-left: 10px; position: relative; }
.tab { background: #f7f9fc; border-radius: 8px 8px 0 0; height: 32px; width: 230px;
       display: flex; align-items: center; gap: 9px; padding: 0 12px; font-size: 13px; }
.tab .x { margin-left: auto; color: #555; }
.ctl { position: absolute; right: 0; top: 0; height: 40px; display: flex; }
.ctl span { width: 46px; text-align: center; line-height: 38px; color: #333; font-size: 15px; }
.navrow { height: 48px; background: #f7f9fc; display: flex; align-items: center;
          gap: 16px; padding: 0 14px; border-bottom: 1px solid #e3e7ee; }
.arrow { color: #555; font-size: 18px; width: 18px; text-align: center; }
.addr { flex: 1; height: 32px; background: #fff; border: 1px solid #c9d0dc;
        border-bottom: 2px solid #0a6cd6; border-radius: 5px; display: flex;
        align-items: center; padding: 0 12px; font-size: 14px; gap: 9px; }
.addr .sel { font-family: "DejaVu Sans Mono", monospace; font-size: 14px; }
.find { width: 210px; height: 32px; background: #fff; border: 1px solid #c9d0dc;
        border-radius: 5px; display: flex; align-items: center; padding: 0 12px;
        color: #666; font-size: 13px; justify-content: space-between; }
.tools { height: 44px; display: flex; align-items: center; gap: 22px; padding: 0 18px;
         border-bottom: 1px solid #e3e7ee; font-size: 13px; color: #333; }
.tools .new { font-weight: 600; }
.body { display: grid; grid-template-columns: 200px 1fr; height: 100%; }
.side { border-right: 1px solid #e3e7ee; padding: 10px 8px; }
.side .row { padding: 6px 10px; font-size: 13px; }
.side .row.on { background: #dde6f3; }
.files { padding: 8px 18px; }
.fhead { display: grid; grid-template-columns: 1fr 190px 130px; color: #555;
         font-size: 12.5px; padding: 6px 10px; border-bottom: 1px solid #eceff4; }
.frow { display: grid; grid-template-columns: 1fr 190px 130px; align-items: center;
        padding: 7px 10px; font-size: 13px; }
.frow div:first-child { display: flex; align-items: center; gap: 10px; }
.frow span { color: #555; }
.folder { width: 22px; height: 17px; background: #f4c542; border-radius: 3px;
          position: relative; flex: none; }
.folder:before { content: ""; position: absolute; left: 0; top: -4px; width: 10px;
                 height: 5px; background: #e0ac1f; border-radius: 3px 3px 0 0; }
"""

PS = '<div class="ps {size}">&gt;_</div>'
WIN = '<div class="tb-icon win"><i></i><i></i><i></i><i></i></div>'


def start_menu(overlay: str) -> str:
    """The Start menu search panel with "powershell" typed."""
    apps = "".join(
        f'<div class="row">{PS.format(size="s")}<div>{name}</div><span class="chev">›</span></div>'
        for name in (
            "Windows PowerShell ISE",
            "Windows PowerShell (x86)",
            "Windows PowerShell ISE (x86)",
        )
    )
    actions = "".join(
        f'<div class="act"><div class="glyph {kind}"></div>{text}</div>'
        for kind, text in (
            ("", "Open"),
            ("shield", "Run as administrator"),
            ("shield", "Run ISE as administrator"),
            ("", "Windows PowerShell ISE"),
        )
    )
    return f"""
<body class="desktop" style="width:1000px;height:660px">
  <div class="panel">
    <div class="searchbox"><div class="lens"></div><span>powershell</span><div class="caret"></div></div>
    <div class="tabs"><b>All</b><span>Apps</span><span>Documents</span><span>Web</span>
      <span>Settings</span><span>Folders</span><span>Photos</span></div>
    <div class="cols">
      <div>
        <div class="label">Best match</div>
        <div class="row best">{PS.format(size="m")}<div>Windows PowerShell<small>App</small></div></div>
        <div class="label" style="margin-top:14px">Apps</div>
        {apps}
        <div class="label" style="margin-top:14px">Search the web</div>
        <div class="row"><div class="lens"></div><div>powershell
          <span style="color:#666">- See more search results</span></div><span class="chev">›</span></div>
      </div>
      <div class="right">
        <div class="hero">{PS.format(size="l")}<h2>Windows PowerShell</h2><span>App</span></div>
        {actions}
      </div>
    </div>
  </div>
  <div class="taskbar">{WIN}
    <div class="tb-search"><div class="lens"></div>powershell</div>
    <div class="tb-icon" style="background:#f4c542"></div>
    <div class="tb-icon" style="background:#2a9d8f"></div>
    <div class="tb-icon" style="background:#5b6abf"></div>
  </div>
  {overlay}
</body>"""


def explorer(overlay: str) -> str:
    """File Explorer with `powershell` typed in the address bar."""
    side = "".join(
        f'<div class="row{" on" if on else ""}"><div class="ico" style="background:{colour}"></div>{name}</div>'
        for name, colour, on in (
            ("Home", "#5b8def", False),
            ("Desktop", "#3aa0d8", False),
            ("Downloads", "#2a9d8f", False),
            ("Documents", "#7a8aa6", False),
            ("courses", "#f4c542", True),
            ("Pictures", "#5b6abf", False),
            ("This PC", "#3d7fd0", False),
        )
    )
    rows = "".join(
        f'<div class="frow"><div><div class="folder"></div>{name}</div><span>{date}</span><span>File folder</span></div>'
        for name, date in (
            ("python_course", "12/03/2026 10:41"),
            ("statistics", "04/05/2026 18:02"),
            ("thesis", "21/06/2026 09:15"),
        )
    )
    return f"""
<body class="explorer-bg" style="width:1000px;height:440px">
  <div class="exp">
    <div class="titlebar">
      <div class="tab"><div class="folder"></div>courses<span class="x">✕</span></div>
      <div class="ctl"><span>—</span><span>☐</span><span>✕</span></div>
    </div>
    <div class="navrow">
      <span class="arrow">←</span><span class="arrow">→</span><span class="arrow">↑</span>
      <span class="arrow">⟳</span>
      <div class="addr"><div class="folder"></div><span class="sel">powershell</span><div class="caret" style="margin-left:-7px"></div></div>
      <div class="find">Search courses<div class="lens"></div></div>
    </div>
    <div class="tools"><span class="new">⊕ New</span><span>Cut</span><span>Copy</span>
      <span>Paste</span><span>Rename</span><span>Share</span><span>Delete</span>
      <span style="margin-left:auto">Sort</span><span>View</span><span>···</span></div>
    <div class="body">
      <div class="side">{side}</div>
      <div class="files">
        <div class="fhead"><span>Name</span><span>Date modified</span><span>Type</span></div>
        {rows}
      </div>
    </div>
  </div>
  {overlay}
</body>"""


def ring(left: int, top: int, width: int, height: int) -> str:
    return f'<div class="ring" style="left:{left}px;top:{top}px;width:{width}px;height:{height}px"></div>'


def badge(left: int, top: int, text: str) -> str:
    return f'<div class="badge" style="left:{left}px;top:{top}px">{text}</div>'


PAGES = {
    "01a_windows_search_powershell": (
        start_menu(
            ring(126, 30, 170, 62)
            + badge(110, 46, "1")
            + ring(96, 156, 400, 78)
            + badge(82, 180, "2")
        ),
        660,
    ),
    "01a_windows_run_as_administrator": (start_menu(ring(468, 347, 260, 46)), 660),
    "01a_windows_explorer_powershell": (explorer(ring(166, 55, 200, 60)), 440),
}

"""Docker Desktop illustration for guide 02b: the app is open and its engine is running."""

CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
html { zoom: 2; overflow: hidden; }
body { width: 1000px; overflow: hidden; position: relative;
       font-family: "Noto Sans", "Ubuntu", "DejaVu Sans", sans-serif; font-size: 12.5px;
       color: #1f2a37; background: #dfe6f0; }
.win { position: absolute; left: 20px; right: 20px; top: 18px; bottom: 18px;
       background: #fff; border-radius: 9px; border: 1px solid #b9c2d0;
       box-shadow: 0 8px 26px rgba(0,0,0,.25); display: flex; flex-direction: column;
       overflow: hidden; }
.top { height: 40px; background: #0b214a; color: #fff; display: flex; align-items: center;
       padding: 0 14px; gap: 18px; flex: none; }
.top .brand { font-weight: 700; font-size: 14px; display: flex; align-items: center; gap: 8px; }
.top .whale { width: 20px; height: 13px; background: #2496ed; border-radius: 3px 3px 8px 8px; }
.top .search { flex: 1; max-width: 360px; margin: 0 auto; height: 24px; border-radius: 5px;
               background: #1f3a66; color: #b8c4d8; padding: 0 10px; line-height: 24px;
               font-size: 12px; }
.top .ctl { display: flex; gap: 22px; font-size: 14px; }
.main { flex: 1; display: flex; min-height: 0; }
.nav { width: 200px; flex: none; border-right: 1px solid #e3e7ee; padding: 10px 8px; }
.item { padding: 6px 10px; border-radius: 5px; display: flex; gap: 10px; align-items: center; }
.item.on { background: #e5effd; color: #1d63ed; font-weight: 600; }
.item i { width: 14px; height: 14px; border: 1.5px solid #5f6b7a; border-radius: 3px; flex: none; }
.item.on i { border-color: #1d63ed; }
.page { flex: 1; padding: 18px 24px; min-width: 0; }
.page h1 { font-size: 20px; font-weight: 700; margin-bottom: 4px; }
.page .sub { color: #5f6b7a; margin-bottom: 26px; }
.empty { border: 1px dashed #c9d2df; border-radius: 8px; height: 170px; display: flex;
         flex-direction: column; align-items: center; justify-content: center; gap: 6px;
         color: #5f6b7a; }
.empty b { color: #1f2a37; font-size: 14px; }
.foot { height: 40px; border-top: 1px solid #e3e7ee; display: flex; align-items: center;
        padding: 0 14px; gap: 18px; font-size: 11.5px; color: #5f6b7a; flex: none; }
.engine { display: inline-flex; align-items: center; gap: 7px; color: #1f2a37; font-weight: 600;
          background: #e3f5ea; border-radius: 4px; padding: 2px 8px; }
.engine i { width: 9px; height: 9px; border-radius: 50%; background: #1a9e55; }
.circled { position: relative; display: inline-block; }
.ring { position: absolute; left: -14px; right: -14px; top: -6px; bottom: -6px;
        border: 4px solid #12a150; border-radius: 50%; z-index: 10;
        box-shadow: 0 0 0 1.5px rgba(255,255,255,.9), inset 0 0 0 1.5px rgba(255,255,255,.9); }
.badge { position: absolute; right: -48px; top: calc(50% - 14px); width: 28px; height: 28px;
         border-radius: 50%; background: #12a150; color: #fff; font-weight: 700;
         font-size: 15px; display: flex; align-items: center; justify-content: center;
         z-index: 11; border: 2px solid #fff; }
.note { color: #0f7d3e; font-weight: 600; margin-left: 40px; }
"""


def item(text: str, on: bool = False) -> str:
    return f'<div class="item{" on" if on else ""}"><i></i>{text}</div>'


NAV = "".join(
    item(text, on=text == "Containers")
    for text in (
        "Ask Gordon",
        "Containers",
        "Images",
        "Volumes",
        "Builds",
        "Models",
        "Docker Hub",
        "Docker Scout",
        "Extensions",
    )
)

BODY = f"""<body style="height:440px"><div class="win">
  <div class="top"><div class="brand"><div class="whale"></div>docker desktop</div>
    <div class="search">Search</div>
    <div class="ctl"><span>—</span><span>☐</span><span>✕</span></div></div>
  <div class="main">
    <div class="nav">{NAV}</div>
    <div class="page"><h1>Containers</h1>
      <div class="sub">View all your running containers and applications.</div>
      <div class="empty"><b>Your running containers show up here</b>
        <span>A container is an isolated environment for your code</span></div>
    </div>
  </div>
  <div class="foot">
    <span class="circled" style="margin-left:14px"><span class="engine"><i></i>Engine running</span>
      <span class="ring"></span><span class="badge">1</span></span>
    <span class="note">the docker commands work only while this says running</span>
    <span style="margin-left:auto">RAM 1.62 GB · CPU 0.20%</span></div>
</div></body>"""

PAGES = {"02b_docker_desktop_engine_running": (BODY, 440)}

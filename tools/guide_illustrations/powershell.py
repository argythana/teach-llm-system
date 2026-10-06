"""PowerShell illustrations for guides 01b (llmfit), 01c (Ollama) and 02b (Docker) on Windows.

Each page is a Windows PowerShell window: the command the student types, the output
it prints, and green circles on what to type and what to read. The outputs copy the
real ones of llmfit 1.1.16, uv, Ollama's install.ps1, `ollama run --verbose` and
Docker Compose v2, with example hardware and timings.
"""

CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
html { zoom: 2; overflow: hidden; }
body { width: 1000px; overflow: hidden; position: relative; background: #dfe6f0;
       font-family: "Noto Sans", "Ubuntu", "DejaVu Sans", sans-serif; font-size: 13px; }
.mono { font-family: "DejaVu Sans Mono", monospace; }
.frame { position: absolute; left: 20px; right: 20px; top: 18px; bottom: 18px;
         background: #012456; border-radius: 9px; border: 1px solid #333;
         box-shadow: 0 8px 26px rgba(0,0,0,.25); }
.psbar { height: 32px; background: #1f1f1f; color: #e5e5e5; border-radius: 9px 9px 0 0;
         display: flex; align-items: center; padding: 0 12px; gap: 9px; font-size: 12.5px; }
.psbar .ico { width: 16px; height: 16px; background: #012456; border: 1px solid #5b7bb0;
              border-radius: 3px; }
.psbar .ctl { margin-left: auto; display: flex; gap: 22px; font-size: 13px; }
.psbody { position: absolute; left: 0; right: 0; top: 32px; bottom: 0;
          color: #eeedf0; padding: 14px 18px; font-size: 12.5px; line-height: 1.85;
          white-space: pre; }
.cmd { color: #f9f1a5; }
.str { color: #61d6d6; }
.cur { display: inline-block; width: 8px; height: 15px; background: #eeedf0;
       vertical-align: -2px; }
.circled { position: relative; display: inline-block; }
.ring { position: absolute; left: -14px; right: -14px; top: -5px; bottom: -5px;
        border: 4px solid #12a150; border-radius: 50%; z-index: 10; pointer-events: none;
        box-shadow: 0 0 0 1.5px rgba(255,255,255,.9), inset 0 0 0 1.5px rgba(255,255,255,.9); }
.badge { position: absolute; right: -46px; top: calc(50% - 13px); width: 26px; height: 26px;
         border-radius: 50%; background: #12a150; color: #fff; font-weight: 700;
         font-size: 14px; display: flex; align-items: center; justify-content: center;
         z-index: 11; border: 2px solid #fff; font-family: "Noto Sans", sans-serif; }
.small { font-size: 9.6px; }
.note { color: #7ee2a8; font-weight: 600; font-size: 12px; margin-left: 58px;
        font-family: "Noto Sans", sans-serif; }
"""

PROMPT = r"PS C:\Users\you&gt; "
CURSOR = f'{PROMPT}<span class="cur"></span>'


def circled(inner: str, number: str = "", style: str = "") -> str:
    badge = f'<span class="badge">{number}</span>' if number else ""
    return f'<span class="circled" style="{style}">{inner}<span class="ring"></span>{badge}</span>'


def typed(command: str, number: str) -> str:
    """A circled command after the prompt, moved right so the ring clears the prompt."""
    return PROMPT + circled(command, number, "margin-left:14px")


def window(lines: list[str], height: int) -> tuple[str, int]:
    body = "\n".join(lines)
    return (
        f"""<body style="height:{height}px"><div class="frame">
  <div class="psbar"><div class="ico"></div>Windows PowerShell
    <div class="ctl"><span>—</span><span>☐</span><span>✕</span></div></div>
  <div class="psbody mono">{body}</div>
</div></body>""",
        height,
    )


INSTALL_LLMFIT = window(
    [
        typed('<span class="cmd">uv</span> tool install llmfit', "1"),
        "Resolved 1 package in 412ms",
        "Prepared 1 package in 2.31s",
        "Installed 1 package in 18ms",
        " + llmfit==1.1.16",
        circled("Installed 1 executable: llmfit", "2"),
        CURSOR,
    ],
    270,
)

LLMFIT_PLAN = window(
    [
        PROMPT
        + '<span class="cmd">llmfit</span> plan <span class="str">"Qwen/Qwen3-8B"</span>'
        " --quant Q4_K_M --context 8192",
        "",
        "=== System Specifications ===",
        "CPU: 12th Gen Intel(R) Core(TM) i7-12700H (14 cores)",
        "Total RAM: 15.69 GB",
        "Available RAM: 9.84 GB",
        "RAM Bandwidth: ~68 GB/s (measured)",
        "Backend: CUDA",
        "GPU 1: NVIDIA GeForce RTX 4060 Laptop GPU (8.00 GB VRAM,"
        + circled("7.12 GB free", "1", "margin-left:22px")
        + '<span style="margin-left:50px">, CUDA)</span>',
        "",
        "=== Hardware Planning Estimate ===",
        "Model: Qwen/Qwen3-8B",
        "Provider: Alibaba",
        "Context: 8192",
        "Quantization: Q4_K_M",
        "Disk (est): 4.75 GB (weights only)",
        "KV cache: fp16",
        "Note: Estimate-based output using current llmfit fit/speed heuristics; "
        "not an exact benchmark.",
        "",
        "Minimum Hardware:",
        "  "
        + circled("VRAM: 6.4 GB", "2")
        + '<span class="note">6.4 GB ≤ 7.12 GB free: the gpu tier</span>',
        "  RAM: 8.0 GB",
        "  CPU Cores: 4",
        "",
    ],
    640,
)

INSTALL_OLLAMA = window(
    [
        typed(
            '<span class="cmd">irm</span> https://ollama.com/install.ps1 | '
            '<span class="cmd">iex</span>',
            "1",
        ),
        "&gt;&gt;&gt; Downloading Ollama for Windows...",
        "######################################## 100.0%",
        "&gt;&gt;&gt; Installing Ollama...",
        "&gt;&gt;&gt; Install complete. Run 'ollama' from the command line.",
        PROMPT + '<span class="cmd">ollama</span> --version',
        circled("ollama version is 0.34.0", "2"),
        CURSOR,
    ],
    290,
)

OLLAMA_RUN = window(
    [
        PROMPT
        + '<span class="cmd">ollama</span> run qwen3:1.7b --verbose --think=false '
        '<span class="str">"Say hello in one sentence."</span>',
        "Hello! How can I assist you today?",
        "",
        "total duration:       3.104227711s",
        "load duration:        2.598315904s",
        "prompt eval count:    22 token(s)",
        "prompt eval duration: 199.402ms",
        "prompt eval rate:     110.33 tokens/s",
        "eval count:           10 token(s)",
        "eval duration:        282.507ms",
        circled("eval rate:            35.40 tokens/s")
        + '<span class="note">your speed: the course needs 10 or more</span>',
        CURSOR,
    ],
    380,
)

COURSE = r"PS C:\Users\you\courses\teach-llm-system&gt; "

COMPOSE_UP = window(
    [
        COURSE
        + circled(
            '<span class="cmd">docker</span> compose up -d', "1", "margin-left:14px"
        ),
        "[+] Running 3/3",
        " ✔ Network teach-llm-system_default          Created    0.1s",
        ' ✔ Volume "teach-llm-system_pgvector-data"   Created    0.0s',
        " ✔ Container llm-course-pgvector             Started    0.7s",
        COURSE
        + circled(
            '<span class="cmd">docker</span> compose ps', "2", "margin-left:14px"
        ),
        '<span class="small">NAME                  IMAGE                    COMMAND                  SERVICE    CREATED          STATUS                       PORTS</span>',
        '<span class="small">llm-course-pgvector   pgvector/pgvector:pg17   "docker-entrypoint.s…"   pgvector   12 seconds ago   </span>'
        + circled(
            '<span class="small">Up 11 seconds (healthy)</span>', "3", "margin-left:2px"
        )
        + '<span class="small" style="margin-left:50px">0.0.0.0:5433-&gt;5432/tcp</span>',
        CURSOR.replace(PROMPT, COURSE),
    ],
    340,
)

PAGES = {
    "01b_powershell_install_llmfit": INSTALL_LLMFIT,
    "01b_powershell_llmfit_plan": LLMFIT_PLAN,
    "01c_powershell_install_ollama": INSTALL_OLLAMA,
    "01c_powershell_ollama_run": OLLAMA_RUN,
    "02b_powershell_compose_up": COMPOSE_UP,
}

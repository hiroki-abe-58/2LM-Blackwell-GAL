"""環境診断. 学習を始める前に、ここを全部通してから先へ進む.

Windows + NVIDIA の地雷は「import は通るのに実際のカーネルで落ちる」型が多い。
だから import できたかではなく、GPU で実際に計算を1回走らせて確認する。

    python check_env.py                # 全項目を診断する
    python check_env.py --no-spill     # VRAM 超過の試験だけ省く
    python check_env.py --fail-demo    # わざと失敗させた表示を見る
    python check_env.py --color always # ファイルに落としても色を残す (記事の画像用)

最後に「あなたのカードでは micro_bs いくつ / grad_accum いくつ」を出す。
RTX 5090 を持っていない読者がここで離脱しないための項目である。
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import runtime as rt  # noqa: E402

# sm_120 に対して cu126 以下の wheel を使うと、import と is_available() は
# 通るのに最初の実カーネル起動で落ちる。cu128 を下限として弾く。
MIN_CUDA_TAG = 128

# 公開スペックから拾った bf16 テンソルコアの概数 (FP32 累積・dense)。
# 厳密な理論値ではなく「桁がおかしくないか」を見るための参考値である。
# 実測がこれを超えることもあるので、超えたら異常ではなく参考値のほうを疑う。
THEORETICAL_BF16_TFLOPS = {
    "RTX 5090": 209.5,
    "RTX 5080": 112.0,
    "RTX 4090": 165.2,
    "RTX 4080": 97.5,
    "RTX 3090": 71.0,
}

OK, WARN, NG, INFO = "OK", "注意", "NG", "情報"

RESET = "\033[0m"
COLORS = {OK: "\033[32m", WARN: "\033[33m", NG: "\033[31m", INFO: "\033[36m"}
_use_color = False


def setup_color(mode: str) -> None:
    """色を使うかを決め、必要なら Windows コンソールの ANSI を有効化する.

    既定の auto は、リダイレクト先がファイルなら色を付けない。
    記事用の画像を作るときだけ always にする (tools/render_terminal.py が色を解釈する)。
    """
    global _use_color
    _use_color = mode == "always" or (mode == "auto" and sys.stdout.isatty())
    if not _use_color or os.name != "nt":
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode_flags = ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode_flags)):
            kernel32.SetConsoleMode(handle, mode_flags.value | 0x0004)
    except Exception:
        pass


class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []

    def add(self, status: str, item: str, detail: str) -> None:
        self.rows.append((status, item, detail))
        mark = {OK: "[ OK ]", WARN: "[注意]", NG: "[ NG ]", INFO: "[情報]"}[status]
        if _use_color:
            mark = f"{COLORS[status]}{mark}{RESET}"
        print(f"{mark} {item}")
        for line in str(detail).splitlines():
            print(f"       {line}")

    @property
    def failed(self) -> list[tuple[str, str, str]]:
        return [r for r in self.rows if r[0] == NG]

    @property
    def warned(self) -> list[tuple[str, str, str]]:
        return [r for r in self.rows if r[0] == WARN]


def rule(title: str = "") -> None:
    print("-" * 74)
    if title:
        print(title)
        print("-" * 74)


def parse_cuda_tag(version: str) -> int | None:
    """'2.13.0+cu130' から 130 を取り出す. cpu 版や素の版なら None."""
    if "+cu" not in version:
        return None
    tag = version.split("+cu", 1)[1]
    digits = "".join(c for c in tag if c.isdigit())
    return int(digits) if digits else None


def theoretical_tflops(name: str) -> float | None:
    for key, value in THEORETICAL_BF16_TFLOPS.items():
        if key.replace("RTX ", "") in name.replace("RTX ", ""):
            return value
    return None


# ---------------------------------------------------------------- 1 / 2 / 12
def check_python(report: Report) -> None:
    bits = 64 if sys.maxsize > 2**32 else 32
    detail = (
        f"{platform.python_version()} / {bits} bit / {sys.executable}\n"
        f"{platform.platform()}"
    )
    if bits != 64:
        report.add(NG, "Python", detail + "\n32bit では PyTorch が動きません。")
    elif sys.version_info < (3, 10):  # noqa: UP036  読者が古い Python で来る前提の判定
        report.add(NG, "Python", detail + "\n3.10 以上が必要です。")
    else:
        report.add(OK, "Python のバージョンとビット数", detail)


def check_encoding(report: Report) -> None:
    stdout_enc = (sys.stdout.encoding or "").lower()
    lines = [
        f"標準出力の encoding : {sys.stdout.encoding}",
        f"既定のファイル encoding : {__import__('locale').getpreferredencoding(False)}",
        f"PYTHONUTF8 = {os.environ.get('PYTHONUTF8', '(未設定)')}",
        f"PYTHONIOENCODING = {os.environ.get('PYTHONIOENCODING', '(未設定)')}",
    ]
    if "utf" not in stdout_enc:
        lines.append(
            "cp932 のままだと、学習ログに日本語サンプルを出した瞬間に"
            " UnicodeEncodeError で落ちます。"
        )
        lines.append('対処: $env:PYTHONUTF8="1"; $env:PYTHONIOENCODING="utf-8"')
        report.add(NG, "標準出力の文字コード", "\n".join(lines))
    else:
        report.add(OK, "標準出力の文字コード", "\n".join(lines))

    long_paths = read_long_paths_enabled()
    if long_paths == 1:
        report.add(OK, "Windows の長いパス", "LongPathsEnabled = 1")
    else:
        report.add(
            WARN,
            "Windows の長いパス",
            f"LongPathsEnabled = {long_paths}\n"
            "torch.compile のキャッシュパスが 260 文字制限に当たると"
            " FileNotFoundError になります。\n"
            "TORCHINDUCTOR_CACHE_DIR を短いパス (例 E:\\ti_cache) にしていれば実害は出ません。\n"
            "根本対処には管理者権限で次を実行します:\n"
            "  reg add HKLM\\SYSTEM\\CurrentControlSet\\Control\\FileSystem"
            " /v LongPathsEnabled /t REG_DWORD /d 1 /f",
        )


def read_long_paths_enabled() -> int:
    if os.name != "nt":
        return -1
    try:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\FileSystem"
        ) as key:
            return int(winreg.QueryValueEx(key, "LongPathsEnabled")[0])
    except Exception:
        return -1


def check_torch_wheel(report: Report, fail_demo: bool) -> object | None:
    try:
        import torch
    except Exception as exc:
        report.add(NG, "PyTorch の import", f"{type(exc).__name__}: {exc}")
        return None

    version = "2.5.1+cu121" if fail_demo else torch.__version__
    tag = parse_cuda_tag(version)
    detail = f"torch {version} / ビルド時 CUDA {torch.version.cuda}"
    if tag is None:
        report.add(
            NG,
            "PyTorch の wheel",
            detail
            + "\n+cuXXX が付いていません。CPU 版が入っています。\n"
            "対処: pip install torch --index-url"
            " https://download.pytorch.org/whl/cu130\n"
            "--index-url を忘れると CPU 版が入ります。",
        )
    elif tag < MIN_CUDA_TAG:
        report.add(
            NG,
            "PyTorch の wheel",
            detail
            + f"\ncu{tag} は sm_120 (Blackwell) 非対応です。"
            "import は通り is_available() も True を返しますが、\n"
            "最初の実カーネル起動で"
            " 'no kernel image is available for execution on the device' で落ちます。\n"
            f"対処: cu{MIN_CUDA_TAG} 以上の wheel を入れ直してください。",
        )
    else:
        report.add(OK, "PyTorch の wheel (+cu タグ)", detail)
    return torch


# ------------------------------------------------------------------ 3 / 4 / 5
def check_device(report: Report, torch) -> bool:
    if not torch.cuda.is_available():
        report.add(
            NG,
            "torch.cuda.is_available()",
            "False です。ドライバか wheel を疑ってください。\n"
            "nvidia-smi が通るかを先に確認します。",
        )
        return False
    summary = rt.device_summary()
    report.add(
        OK,
        "GPU の認識",
        f"{summary.name}\n"
        f"compute capability : {summary.capability[0]}.{summary.capability[1]}"
        f" (sm_{summary.capability[0]}{summary.capability[1]})\n"
        f"世代               : {summary.generation}\n"
        f"VRAM               : {summary.total_vram_gb:.1f} GB\n"
        f"ドライバ           : {summary.driver_version}",
    )

    capability = summary.capability
    if capability >= (12, 0):
        verdict = "学習: 可 / データ生成: 32B クラスまで可"
    elif capability >= (8, 9):
        verdict = "学習: 可 / データ生成: VRAM 次第"
    elif capability >= (8, 0):
        verdict = "学習: 可 (bf16 が使える) / データ生成: 8B クラスまで"
    else:
        verdict = "未検証。bf16 が使えるかを実測して判断してください"
    report.add(INFO, "世代からの判定", verdict)
    return True


def check_real_kernel(report: Report, torch) -> bool:
    """import が通るだけでは不十分. 実際に GPU でカーネルを起動させる."""
    try:
        a = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
        value = float((a @ a).float().abs().mean())
        torch.cuda.synchronize()
    except Exception as exc:
        report.add(
            NG,
            "GPU で実際に行列積を回す",
            f"{type(exc).__name__}: {exc}\n"
            "no kernel image が出た場合、wheel の CUDA ビルドが"
            " このカードの compute capability に対応していません。",
        )
        return False
    report.add(
        OK,
        "GPU で実際に行列積を回す",
        f"bf16 512x512 の平均絶対値 {value:.3f} (ここが通れば wheel は正しい)",
    )
    return True


# ---------------------------------------------------------------------- 6
def check_triton(report: Report, torch) -> None:
    try:
        import triton

        triton_version = triton.__version__
    except Exception as exc:
        report.add(
            NG,
            "triton の import",
            f"{type(exc).__name__}: {exc}\n"
            "Windows には公式 Triton の wheel がありません。\n"
            '対処: pip install -U "triton-windows<3.8"'
            " (公式 triton が入っているなら先に uninstall)",
        )
        return

    def tiny(x):
        return torch.nn.functional.gelu(x) * torch.sigmoid(x)

    try:
        compiled = torch.compile(tiny, dynamic=False)
        x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
        started = time.perf_counter()
        compiled(x)
        torch.cuda.synchronize()
        first = time.perf_counter() - started
    except Exception as exc:
        report.add(
            NG,
            "torch.compile",
            f"triton {triton_version} は import できましたがコンパイルで落ちました。\n"
            f"{type(exc).__name__}: {str(exc)[:400]}\n"
            "ptxas が sm_120 で tensormap.replace を拒否している場合、"
            "Triton が誤って sm_120a を狙っています。\n"
            "自分のコードを疑う前に triton-windows の版を疑ってください。",
        )
        return
    report.add(
        OK,
        "triton と torch.compile",
        f"triton {triton_version} / 初回コンパイル {first:.2f} 秒\n"
        f"キャッシュ: {os.environ.get('TORCHINDUCTOR_CACHE_DIR', '(既定)')}",
    )


def check_sdpa(report: Report, torch) -> None:
    """SDPA のどの backend が実際に使えるかを1つずつ試す.

    Windows の wheel には FlashAttention が入っていないことがある。
    「flash backend を使う」という前提が崩れるので、必ず実測で確認する。
    """
    from torch.nn.attention import SDPBackend, sdpa_kernel

    q = torch.randn(4, 6, 256, 64, device="cuda", dtype=torch.bfloat16)
    backends = [
        ("flash", SDPBackend.FLASH_ATTENTION),
        ("cudnn", SDPBackend.CUDNN_ATTENTION),
        ("mem_efficient", SDPBackend.EFFICIENT_ATTENTION),
        ("math", SDPBackend.MATH),
    ]
    usable, lines = [], []
    for name, backend in backends:
        try:
            with sdpa_kernel(backend):
                torch.nn.functional.scaled_dot_product_attention(q, q, q, is_causal=True)
            torch.cuda.synchronize()
            usable.append(name)
            lines.append(f"{name:<14} 使える")
        except Exception as exc:
            lines.append(f"{name:<14} 使えない ({type(exc).__name__})")
    lines.append(f"使える backend: {', '.join(usable) or 'なし'}")
    if "flash" not in usable:
        lines.append(
            "flash が無い環境です。Windows の wheel は"
            " 'Torch was not compiled with flash attention' になることがあります。"
        )
        lines.append("cudnn か mem_efficient が使えれば学習は問題なく回ります。")
    status = OK if usable else NG
    report.add(status, "SDPA の backend", "\n".join(lines))


# ---------------------------------------------------------------------- 8
def check_matmul_tflops(report: Report, torch) -> None:
    size = 8192
    torch.set_float32_matmul_precision("high")
    a = torch.randn(size, size, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(size, size, device="cuda", dtype=torch.bfloat16)
    for _ in range(3):  # ウォームアップ. 含めて測ると数字が壊れる。
        a @ b
    torch.cuda.synchronize()
    iters = 20
    started = time.perf_counter()
    for _ in range(iters):
        a @ b
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    flops = 2 * size**3 * iters
    measured = flops / elapsed / 1e12

    name = torch.cuda.get_device_name(0)
    reference = theoretical_tflops(name)
    lines = [f"bf16 {size}^3 の実効 {measured:.1f} TFLOPS (ウォームアップ後)"]
    if reference:
        ratio = measured / reference * 100
        lines.append(f"参考値 {reference:.1f} TFLOPS との比 {ratio:.0f}%")
        if ratio >= 100:
            lines.append(
                "参考値を超えています。参考値は公開スペックから拾った概数なので、"
                "ブーストクロックや cuBLAS の実装次第で実測が上回ります。異常ではありません。"
            )
            status = OK
        elif ratio >= 55:
            lines.append("大きい行列積で 55% 以上なら正常です。")
            status = OK
        elif ratio >= 35:
            lines.append("やや低めです。電力制限か温度を確認してください。")
            status = WARN
        else:
            lines.append(
                "30% 台まで落ちています。何かがおかしいので"
                " 電力・温度・他プロセスを確認してください。"
            )
            status = WARN
    else:
        lines.append("このカードの参考値は表に無いので比は出しません。")
        status = INFO
    del a, b
    torch.cuda.empty_cache()
    report.add(status, "実効 TFLOPS と理論値比", "\n".join(lines))


def measured_train_time() -> str:
    """自分の runs/loss.csv から所要時間を読む.

    数値をソースに書き込むと、あとで学習し直したときに嘘になる。
    走らせていなければ「未測定」と正直に出す。
    """
    csv_path = Path(__file__).resolve().parent / "runs" / "loss.csv"
    if not csv_path.exists():
        return "未測定 (src/train.py をまだ走らせていません)"
    seconds = 0.0
    steps = 0
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("elapsed_sec"):
                seconds = max(seconds, float(row["elapsed_sec"]))
                steps = max(steps, int(row["step"]))
    if steps == 0:
        return "未測定 (runs/loss.csv が空です)"
    return f"{seconds:.1f} 秒 ({steps:,} ステップの実測)"


# ---------------------------------------------------------------------- 7
def check_batch_plan(report: Report, torch) -> None:
    total = torch.cuda.get_device_properties(0).total_memory / rt.BYTES_PER_GB
    plan = rt.plan_batch(total)
    report.add(
        INFO,
        "推奨バッチ",
        f"あなたのカードでは {plan}\n"
        f"3,600 ステップの所要時間: {measured_train_time()}\n"
        "グローバルバッチのトークン数は固定です。"
        "VRAM に合わせてバッチだけ変えると実効学習率が変わり、別のモデルになります。",
    )


# ---------------------------------------------------------------------- 9
def _fill_and_measure(torch, nbytes: int, baseline: float | None) -> tuple[float, float]:
    """確保して実際に書き込み、書き込み帯域と共有GPUメモリの増分を返す.

    torch.empty() は要求を受け付けるだけで物理ページを割り当てない。
    書き込んで初めてこぼれるので、確保だけの検査では静かな低速化を捕まえられない。
    """
    # uint8 で fill_ すると1バイトずつのカーネルになって帯域が出ない。
    # 絶対値が誤解を招くので 8 バイト幅で書き込む。
    blob = torch.empty(nbytes // 8, dtype=torch.int64, device="cuda")
    torch.cuda.synchronize()
    started = time.perf_counter()
    blob.fill_(1)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    shared = rt.shared_memory_gb()
    delta = 0.0 if shared is None or baseline is None else shared - baseline
    gbps = blob.numel() * 8 / rt.BYTES_PER_GB / elapsed
    del blob
    torch.cuda.empty_cache()
    return gbps, delta


def check_sysmem_fallback(report: Report, torch, enabled: bool) -> None:
    if not enabled:
        report.add(INFO, "VRAM 超過時の挙動", "--no-spill が指定されたので省略しました。")
        return
    if not rt.shared_memory_available():
        report.add(
            WARN,
            "共有GPUメモリの取得",
            f"パフォーマンスカウンタを読めません: {rt.shared_memory_error()}\n"
            "こぼれても検出できないので、代わりに"
            " set_per_process_memory_fraction で蓋をしてください。",
        )
    # 先に CUDA コンテキストを張っておく。張る前は自プロセスがカウンタの
    # インスタンスとして現れないので、基準値が 0 になってしまう。
    torch.zeros(1, device="cuda")
    torch.cuda.synchronize()

    total = torch.cuda.get_device_properties(0).total_memory
    baseline = rt.shared_memory_gb()
    lines = [
        f"基準の共有GPUメモリ: {baseline:.3f} GB" if baseline is not None
        else "基準の共有GPUメモリ: 取得できず"
    ]
    spilled = None
    try:
        # VRAM に収まる側を先に測って、比較の基準にする。
        fit_gbps, fit_shared = _fill_and_measure(torch, int(total * 0.25), baseline)
        lines.append(
            f"VRAM に収まる {total * 0.25 / rt.BYTES_PER_GB:.1f} GB へ書き込み: "
            f"{fit_gbps:.0f} GB/s / 共有GPUメモリ増分 {fit_shared:+.3f} GB"
        )
        # 確保だけでは何も起きない。実際に書き込まないとこぼれない。
        over_gbps, over_shared = _fill_and_measure(torch, int(total * 1.10), baseline)
        lines.append(
            f"VRAM を超える {total * 1.10 / rt.BYTES_PER_GB:.1f} GB へ書き込み: "
            f"{over_gbps:.0f} GB/s / 共有GPUメモリ増分 {over_shared:+.3f} GB"
        )
        lines.append("OOM 例外は出ませんでした。")
        spilled = over_shared > 0.05
        if spilled:
            ratio = fit_gbps / over_gbps if over_gbps > 0 else float("inf")
            lines.append(
                f"書き込み帯域が {ratio:.1f} 倍遅くなりました。"
                "これが「静かに遅くなる」の正体です。"
            )
            lines.append(
                "確保した時点では共有GPUメモリは増えません。"
                "実際に触った瞬間に移るので、確保だけを試す検査では見つかりません。"
            )
    except torch.cuda.OutOfMemoryError:
        lines.append(
            "OutOfMemoryError が出ました。これが望ましい状態です"
            " (Prefer No Sysmem Fallback 相当)。"
        )
        spilled = False
    except RuntimeError as exc:
        spilled = True
        lines.append(f"RuntimeError: {str(exc)[:200]}")
    finally:
        torch.cuda.empty_cache()

    if spilled:
        lines.append("")
        lines.append("対策を3つ全部やってください。")
        lines.append(
            "1. NVIDIA コントロールパネル → 3D設定の管理 →"
            " CUDA - Sysmem Fallback Policy を Prefer No Sysmem Fallback"
        )
        lines.append("2. runtime.configure(0.85) で PyTorch 側から上限を切る")
        lines.append("3. MemoryGuard で毎バッチ共有GPUメモリを見て、0 を超えたら止める")
        lines.append(
            "上の帯域は単純な書き込みでの比です。実際の学習や推論では"
            "細かい転送が何度も起きるので、差はこれより大きくなります。"
        )
        report.add(WARN, "VRAM 超過時の挙動", "\n".join(lines))
    else:
        report.add(OK, "VRAM 超過時の挙動", "\n".join(lines))


# --------------------------------------------------------------------- 10
def check_vram_holders(report: Report) -> None:
    holders = rt.vram_holders()
    mine = os.getpid()
    others = [h for h in holders if h[0] != mine]
    if not others:
        report.add(OK, "VRAM を掴んでいるプロセス", "他にありません。")
        return
    lines = [f"pid {pid} / {name} / {mib} MiB" for pid, name, mib in others]
    lines.append("")
    lines.append("死んだプロセスが掴んでいる場合、タスクマネージャでは見えません。")
    lines.append("taskkill /PID <pid> /F で落としてください。")
    report.add(WARN, "VRAM を掴んでいるプロセス", "\n".join(lines))


# --------------------------------------------------------------------- 11
def check_bitsandbytes(report: Report, torch) -> None:
    try:
        import bitsandbytes  # noqa: F401
    except Exception as exc:
        report.add(
            WARN,
            "bitsandbytes",
            f"import できません: {type(exc).__name__}: {exc}\n"
            "その3 のデータ生成で 4bit 量子化に落ちる可能性があるので、\n"
            "ここで先に入れておくと計画が崩れません:"
            " pip install bitsandbytes",
        )
        return
    try:
        from bitsandbytes.nn import Linear4bit

        layer = Linear4bit(256, 256, compute_dtype=torch.bfloat16).cuda()
        out = layer(torch.randn(4, 256, device="cuda", dtype=torch.bfloat16))
        torch.cuda.synchronize()
        report.add(
            OK,
            "bitsandbytes の 4bit",
            f"{bitsandbytes.__version__} / Linear4bit の出力 {tuple(out.shape)}",
        )
    except Exception as exc:
        report.add(
            WARN,
            "bitsandbytes の 4bit",
            f"import は通りましたが 4bit の実行で落ちました。\n"
            f"{type(exc).__name__}: {str(exc)[:300]}",
        )


# --------------------------------------------------------------------- 13
def check_hf_cache(report: Report) -> None:
    hf_home = os.environ.get("HF_HOME")
    path = Path(hf_home) if hf_home else Path.home() / ".cache" / "huggingface"
    lines = [f"HF_HOME = {hf_home or '(未設定 → ' + str(path) + ')'}"]
    try:
        usage = shutil.disk_usage(path if path.exists() else path.anchor)
        free_gb = usage.free / rt.BYTES_PER_GB
        lines.append(f"空き容量 {free_gb:.0f} GB")
    except Exception as exc:
        free_gb = None
        lines.append(f"空き容量を取れません: {exc}")
    inductor = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    lines.append(f"TORCHINDUCTOR_CACHE_DIR = {inductor or '(未設定)'}")
    if hf_home is None:
        lines.append(
            "未設定だと C: を数十GB 食います (32B の INT4 だけで 18GB 前後)。"
        )
        report.add(WARN, "HF キャッシュ", "\n".join(lines))
    elif free_gb is not None and free_gb < 60:
        lines.append("その3 で 32B モデルを落とすには 60 GB 程度の余裕が欲しいところです。")
        report.add(WARN, "HF キャッシュ", "\n".join(lines))
    else:
        report.add(OK, "HF キャッシュ", "\n".join(lines))


def check_nvidia_smi(report: Report) -> None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        report.add(OK, "nvidia-smi", out.stdout.strip())
    except FileNotFoundError:
        report.add(NG, "nvidia-smi", "見つかりません。ドライバが入っていません。")
    except Exception as exc:
        report.add(NG, "nvidia-smi", f"{type(exc).__name__}: {exc}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--no-spill",
        action="store_true",
        help="VRAM を意図的に超過させる試験を省く",
    )
    ap.add_argument(
        "--fail-demo",
        action="store_true",
        help="wheel の判定をわざと失敗させて表示を確認する",
    )
    ap.add_argument(
        "--color",
        choices=("auto", "always", "never"),
        default="auto",
        help="判定マークに色を付けるか (既定 auto: 端末のときだけ付ける)",
    )
    args = ap.parse_args()
    setup_color(args.color)

    print("=" * 74)
    print("  環境診断 — Windows ネイティブ + NVIDIA")
    print("=" * 74)

    report = Report()
    rule("1. 土台")
    check_python(report)
    check_encoding(report)
    check_nvidia_smi(report)

    rule("2. PyTorch と GPU")
    torch = check_torch_wheel(report, args.fail_demo)
    device_ok = False
    if torch is not None:
        device_ok = check_device(report, torch)
    if torch is not None and device_ok and check_real_kernel(report, torch):
        rule("3. 高速化の道具")
        check_triton(report, torch)
        check_sdpa(report, torch)
        check_matmul_tflops(report, torch)

        rule("4. メモリ")
        check_vram_holders(report)
        check_sysmem_fallback(report, torch, not args.no_spill)
        check_batch_plan(report, torch)

        rule("5. その3 の準備")
        check_bitsandbytes(report, torch)
    rule("6. 保存先")
    check_hf_cache(report)

    rule("結果")
    print(f"OK {len([r for r in report.rows if r[0] == OK])} 件 / "
          f"注意 {len(report.warned)} 件 / NG {len(report.failed)} 件")
    if report.failed:
        print()
        print("NG が残っています。ここを直すまで学習に進まないでください。")
        for _status, item, _detail in report.failed:
            print(f"  - {item}")
        return 1
    if report.warned:
        print()
        print("注意 の項目は動きますが、記事の数値が壊れる原因になります。")
        for _status, item, _detail in report.warned:
            print(f"  - {item}")
    print()
    print("ここまで通ったら学習に進めます。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

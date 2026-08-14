"""実行環境の監視と、走らせる前の見積もり (Windows + NVIDIA 版).

Mac 版 (MLX) の data/gal/runtime.py に対応する。3層の構造は変えていない。

    事前   preflight()   走らせる前に「収まらない」と分かる
    走行中 MemoryGuard   落ちる前に自分で止まる
    復旧   呼び出し側    1バッチごとに追記保存する

見る値だけが Windows 用に変わる。Mac は VRAM を使い切ると OS ごと落ちたが、
Windows は落ちない。代わりにシステムRAM へこぼれて10倍前後遅くなり、
エラーも警告も出ない。落ちないので気付けないほうが厄介である。

こぼれたかどうかは nvidia-smi では見えない。専用VRAM しか報告しないからである。
Windows のパフォーマンスカウンタ "GPU Process Memory / Shared Usage" を
直接読む必要がある。それを shared_memory_gb() で行う。
"""

from __future__ import annotations

import ctypes
import math
import os
import shutil
import subprocess
import time
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path

# 学習の数値精度は bf16 に固定する。全世代 (sm_80 以降) で使え、
# fp16 と違って損失スケーリングが要らないので実行ごとのぶれが出ない。
GLOBAL_BATCH_TOKENS = 64 * 256  # Mac 版と同じ 16,384 トークン
BYTES_PER_GB = 1024**3

_GENERATIONS = {
    (6, 1): "Pascal (GTX 10)",
    (7, 0): "Volta",
    (7, 5): "Turing (RTX 20)",
    (8, 0): "Ampere (A100)",
    (8, 6): "Ampere (RTX 30)",
    (8, 9): "Ada (RTX 40)",
    (9, 0): "Hopper (H100)",
    (10, 0): "Blackwell (datacenter)",
    (12, 0): "Blackwell (RTX 50)",
}


def generation_name(capability: tuple[int, int]) -> str:
    return _GENERATIONS.get(capability, f"未検証 (sm_{capability[0]}{capability[1]})")


# --------------------------------------------------------------------------
# 共有GPUメモリ (Windows のパフォーマンスカウンタ経由)
# --------------------------------------------------------------------------
# nvidia-smi は専用VRAM しか見せない。ドライバがシステムRAM へこぼした量は
# "GPU Process Memory" カウンタの Shared Usage にしか出ない。
# PowerShell の Get-Counter でも取れるがプロセス起動に1秒以上かかるので、
# バッチごとの監視には使えない。PDH API をハンドル保持で叩く。

PDH_FMT_LARGE = 0x00000400
_ERROR_SUCCESS = 0
_PDH_MORE_DATA = 0x800007D2
_PDH_NO_DATA = 0x800007D5
_PDH_INVALID_DATA = 0xC0000BC6
_PDH_CALC_NEGATIVE_DENOMINATOR = 0x800007D8


class _CounterValue(ctypes.Structure):
    # x64 では union が8バイト境界に揃うので、CStatus の後に4バイトの詰め物が入る。
    _fields_ = [
        ("CStatus", wintypes.DWORD),
        ("_pad", wintypes.DWORD),
        ("largeValue", ctypes.c_longlong),
    ]


class _CounterItem(ctypes.Structure):
    _fields_ = [
        ("szName", wintypes.LPWSTR),
        ("FmtValue", _CounterValue),
    ]


class SharedMemoryProbe:
    """共有GPUメモリの使用量を繰り返し読むための、開いたままのカウンタ.

    使えない環境 (カウンタ未登録など) では available が False になる。
    その場合は監視を諦めるのではなく、専用VRAM のピーク監視だけで進める。
    """

    COUNTER_PATH = r"\GPU Process Memory(*)\Shared Usage"

    def __init__(self, pid: int | None = None):
        self.pid = os.getpid() if pid is None else pid
        self._prefix = f"pid_{self.pid}_"
        self._query = None
        self._counter = None
        self._pdh = None
        self.error: str | None = None
        self._open()

    def _open(self) -> None:
        if os.name != "nt":
            self.error = "Windows 以外では使えない"
            return
        try:
            pdh = ctypes.WinDLL("pdh.dll")
        except OSError as exc:  # pragma: no cover - Windows なら必ずある
            self.error = f"pdh.dll を読めない: {exc}"
            return
        # PDH_STATUS は符号なし. restype を宣言しないと ctypes が signed int と
        # みなすので、PDH_MORE_DATA (0x800007D2) が負の値になって比較が外れる。
        # 「カウンタは読めているのに常に None が返る」という形で出るので厄介である。
        for name in (
            "PdhOpenQueryW",
            "PdhAddEnglishCounterW",
            "PdhCollectQueryData",
            "PdhGetFormattedCounterArrayW",
            "PdhCloseQuery",
        ):
            getattr(pdh, name).restype = ctypes.c_ulong

        query = ctypes.c_void_p()
        status = pdh.PdhOpenQueryW(None, 0, ctypes.byref(query))
        if status != _ERROR_SUCCESS:
            self.error = f"PdhOpenQueryW が 0x{status:08X}"
            return
        counter = ctypes.c_void_p()
        status = pdh.PdhAddEnglishCounterW(
            query, self.COUNTER_PATH, 0, ctypes.byref(counter)
        )
        if status != _ERROR_SUCCESS:
            pdh.PdhCloseQuery(query)
            self.error = (
                f"PdhAddEnglishCounterW が 0x{status:08X}"
                " (GPU Process Memory カウンタが無い)"
            )
            return
        self._pdh, self._query, self._counter = pdh, query, counter
        # 1回目の収集は基準点を作るだけなので値が入らないことがある。
        pdh.PdhCollectQueryData(query)

    @property
    def available(self) -> bool:
        return self._counter is not None

    def read_gb(self) -> float | None:
        """自プロセスの共有GPUメモリ使用量 (GB). 取れなければ None.

        GPU がまだ使われていないプロセスは、そもそもインスタンスとして
        現れない。その場合は「取れない」ではなく 0 である。
        """
        if not self.available:
            return None
        pdh = self._pdh
        status = pdh.PdhCollectQueryData(self._query)
        if status != _ERROR_SUCCESS:
            return None
        size = wintypes.DWORD(0)
        count = wintypes.DWORD(0)
        status = pdh.PdhGetFormattedCounterArrayW(
            self._counter, PDH_FMT_LARGE, ctypes.byref(size), ctypes.byref(count), None
        )
        if status in (_PDH_NO_DATA, _ERROR_SUCCESS):
            return 0.0
        if status != _PDH_MORE_DATA:
            return None
        buffer = ctypes.create_string_buffer(size.value)
        status = pdh.PdhGetFormattedCounterArrayW(
            self._counter,
            PDH_FMT_LARGE,
            ctypes.byref(size),
            ctypes.byref(count),
            buffer,
        )
        if status != _ERROR_SUCCESS:
            return None
        items = ctypes.cast(
            buffer, ctypes.POINTER(_CounterItem * count.value)
        ).contents
        total = 0
        for item in items:
            name = item.szName or ""
            if not name.startswith(self._prefix):
                continue
            if item.FmtValue.CStatus != _ERROR_SUCCESS:
                continue
            total += max(item.FmtValue.largeValue, 0)
        return total / BYTES_PER_GB

    def close(self) -> None:
        if self._query is not None and self._pdh is not None:
            self._pdh.PdhCloseQuery(self._query)
        self._query = self._counter = None


_PROBE: SharedMemoryProbe | None = None


def shared_memory_gb() -> float | None:
    """共有GPUメモリの現在値 (GB). 0 でなければシステムRAM へこぼれている."""
    global _PROBE
    if _PROBE is None:
        _PROBE = SharedMemoryProbe()
    return _PROBE.read_gb()


def shared_memory_available() -> bool:
    global _PROBE
    if _PROBE is None:
        _PROBE = SharedMemoryProbe()
    return _PROBE.available


def shared_memory_error() -> str | None:
    global _PROBE
    if _PROBE is None:
        _PROBE = SharedMemoryProbe()
    return _PROBE.error


# --------------------------------------------------------------------------
# デバイス情報
# --------------------------------------------------------------------------
@dataclass
class DeviceSummary:
    name: str
    capability: tuple[int, int]
    generation: str
    total_vram_gb: float
    driver_version: str
    torch_version: str
    cuda_build: str | None

    def __str__(self) -> str:
        return (
            f"{self.name} / sm_{self.capability[0]}{self.capability[1]}"
            f" / {self.generation} / VRAM {self.total_vram_gb:.1f} GB"
            f" / driver {self.driver_version}"
            f" / torch {self.torch_version} (cuda {self.cuda_build})"
        )


def driver_version() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        )
        return out.stdout.strip().splitlines()[0].strip()
    except Exception:
        return "不明"


def device_summary(index: int = 0) -> DeviceSummary:
    import torch

    props = torch.cuda.get_device_properties(index)
    return DeviceSummary(
        name=props.name,
        capability=(props.major, props.minor),
        generation=generation_name((props.major, props.minor)),
        total_vram_gb=props.total_memory / BYTES_PER_GB,
        driver_version=driver_version(),
        torch_version=torch.__version__,
        cuda_build=torch.version.cuda,
    )


def configure(fraction: float = 0.85, index: int = 0) -> None:
    """PyTorch が確保できる量に上限を切る.

    Sysmem Fallback Policy の設定を無視してこぼれる報告があるので、
    ドライバ設定に頼らず PyTorch 側からも蓋をしておく。
    こうすると「静かに10倍遅くなる」が「OOM 例外で落ちる」に変わる。
    """
    import torch

    torch.cuda.set_per_process_memory_fraction(fraction, index)


def free_vram_gb(index: int = 0) -> float:
    import torch

    free, _total = torch.cuda.mem_get_info(index)
    return free / BYTES_PER_GB


# --------------------------------------------------------------------------
# 事前見積もり
# --------------------------------------------------------------------------
def preflight(required_gb: float, index: int = 0, headroom_gb: float = 2.0) -> None:
    """必要量が VRAM に収まるかを確認し、収まらなければ走らせない."""
    free = free_vram_gb(index)
    if required_gb + headroom_gb > free:
        raise MemoryBudgetError(
            f"必要量 {required_gb:.1f} GB + 予備 {headroom_gb:.1f} GB が"
            f" 空き VRAM {free:.1f} GB に収まりません。"
            "バッチサイズを下げるか、VRAM を掴んでいるプロセスを終了してください。"
        )


class MemoryBudgetError(RuntimeError):
    """自分で止めるための例外. ドライバに任せると静かに遅くなるだけで済んでしまう."""


def kv_bytes_per_token(config: dict) -> int:
    """1トークンあたりの KVキャッシュ量 (バイト).

        2 (K と V) x 層数 x KVヘッド数 x ヘッド次元 x 2バイト (fp16)

    バッチサイズを決めるのはパラメータ数ではなく KVヘッド数である。
    GQA の有無で、パラメータ数が多いモデルのほうが軽くなる逆転が起きる。
    """
    layers = config.get("num_hidden_layers") or config["n_layer"]
    heads = config.get("num_attention_heads") or config["n_head"]
    kv_heads = config.get("num_key_value_heads", heads)
    head_dim = config.get("head_dim") or (
        (config.get("hidden_size") or config["n_embd"]) // heads
    )
    return 2 * layers * kv_heads * head_dim * 2


def kv_report(
    config: dict,
    weight_gb: float,
    total_vram_gb: float,
    max_tokens: int = 2048,
    batches: tuple[int, ...] = (1, 4, 8, 16, 24, 32, 48, 64),
    reserve_gb: float = 2.0,
) -> str:
    """バッチサイズごとの必要量を表で出す.

    モデルをダウンロードする前に叩けること。config.json だけあれば計算できる。
    Mac 版はこれを後回しにして 17GB を2回落としている。
    """
    per_token = kv_bytes_per_token(config)
    budget = total_vram_gb - weight_gb - reserve_gb
    lines = [
        f"1トークンあたり KV : {per_token / 1024:.0f} KB",
        f"重み (推定)        : {weight_gb:.1f} GB",
        f"予備               : {reserve_gb:.1f} GB",
        f"KV に使える領域    : {budget:.1f} GB (VRAM {total_vram_gb:.1f} GB)",
        f"最大トークン数     : {max_tokens}",
        "",
        f"{'バッチ':>6} {'KV 必要量':>12}  判定",
    ]
    for batch in batches:
        need = per_token * max_tokens * batch / BYTES_PER_GB
        verdict = "収まる" if need <= budget else "収まらない"
        lines.append(f"{batch:>6} {need:>10.2f} GB  {verdict}")
    lines.append("")
    lines.append(
        "「動いた」を基準にしないこと。共有GPUメモリが 0 のまま動いた最大値を採る。"
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# VRAM 適応バッチ (世代ではなく容量で決まる)
# --------------------------------------------------------------------------
@dataclass
class BatchPlan:
    micro_bs: int
    grad_accum: int
    global_batch_tokens: int
    reason: str

    def __str__(self) -> str:
        return (
            f"micro_bs {self.micro_bs} / grad_accum {self.grad_accum}"
            f" (グローバルバッチ {self.global_batch_tokens:,} トークン) — {self.reason}"
        )


def plan_batch(
    total_vram_gb: float,
    ctx_len: int = 256,
    global_batch_tokens: int = GLOBAL_BATCH_TOKENS,
    bytes_per_token: int = 24_000,
) -> BatchPlan:
    """VRAM 容量から micro_bs を決め、grad_accum で補う.

    グローバルバッチのトークン数は固定する。VRAM に合わせてバッチサイズだけ
    変えると実効学習率が変わり、別のモデルが出来上がってしまう。

    bytes_per_token は 11.5〜13.8M / ctx 256 の実測から置いた概算値。
    小さいモデルなのでほとんどのカードで満量が通る想定である。
    """
    max_tokens_at_once = global_batch_tokens
    usable_gb = max(total_vram_gb - 2.0, 1.0)
    affordable_tokens = int(usable_gb * BYTES_PER_GB / bytes_per_token)
    affordable_tokens = min(affordable_tokens, max_tokens_at_once)

    micro_bs = max(affordable_tokens // ctx_len, 1)
    # グローバルバッチを割り切れるように2の冪へ丸める。
    full_bs = global_batch_tokens // ctx_len
    micro_bs = min(2 ** int(math.log2(micro_bs)), full_bs)
    grad_accum = max(full_bs // micro_bs, 1)

    if micro_bs == full_bs:
        reason = "満量が VRAM に収まるので Mac 版と同じ1回で回せる"
    else:
        reason = f"VRAM {total_vram_gb:.0f} GB なので {grad_accum} 回に分けて累積する"
    return BatchPlan(micro_bs, grad_accum, global_batch_tokens, reason)


# --------------------------------------------------------------------------
# 走行中の監視
# --------------------------------------------------------------------------
class MemoryGuard:
    """バッチごとにピークと共有GPUメモリを確認し、超えたら自作例外で止める.

    「遅くなってから気付く」のではなく「こぼれた瞬間に止める」のが要点。

    毎ステップ呼ぶ前提で作っている。VRAM のピークは torch から取るだけなので
    ほぼ無料だが、共有GPUメモリはパフォーマンスカウンタへの問い合わせで
    数ミリ秒かかる。1ステップ 14 ミリ秒の学習では無視できないので、
    共有側だけ shared_interval_sec で間引く。
    """

    def __init__(
        self,
        total_vram_gb: float,
        limit_fraction: float = 0.85,
        shared_tolerance_gb: float = 0.05,
        index: int = 0,
        raise_on_spill: bool = True,
        shared_interval_sec: float = 1.0,
    ):
        self.index = index
        self.limit_gb = total_vram_gb * limit_fraction
        self.shared_tolerance_gb = shared_tolerance_gb
        self.raise_on_spill = raise_on_spill
        self.shared_interval_sec = shared_interval_sec
        self.peak_gb = 0.0
        self.peak_shared_gb = 0.0
        self.spilled = False
        self._last_shared_at = 0.0
        # CUDA コンテキストを張るだけで数MBの共有分が出る環境がある。
        # 絶対値ではなく開始時からの増分で判定する。
        self.baseline_shared_gb = shared_memory_gb() or 0.0

    def check(self, label: str = "", force: bool = False) -> None:
        import torch

        allocated = torch.cuda.max_memory_allocated(self.index) / BYTES_PER_GB
        self.peak_gb = max(self.peak_gb, allocated)
        now = time.time()
        due = force or (now - self._last_shared_at >= self.shared_interval_sec)
        shared = shared_memory_gb() if due else None
        if shared is not None:
            self._last_shared_at = now
            spill = shared - self.baseline_shared_gb
            self.peak_shared_gb = max(self.peak_shared_gb, spill)
            if spill > self.shared_tolerance_gb:
                self.spilled = True
                message = (
                    f"共有GPUメモリが {spill:.2f} GB 増えました{self._where(label)}。"
                    "システムRAM へこぼれています。"
                    "このまま続けるとエラーも警告も出ないまま10倍前後遅くなります。"
                    "バッチサイズを1つ前の値に戻してください。"
                )
                if self.raise_on_spill:
                    raise MemoryBudgetError(message)
                print(f"[警告] {message}", flush=True)
        if allocated > self.limit_gb:
            raise MemoryBudgetError(
                f"確保量が {allocated:.1f} GB で上限 {self.limit_gb:.1f} GB を超えました"
                f"{self._where(label)}。"
            )

    @staticmethod
    def _where(label: str) -> str:
        return f" ({label})" if label else ""

    def report(self) -> str:
        shared = (
            f"+{self.peak_shared_gb:.2f} GB"
            if shared_memory_available()
            else "取得不可"
        )
        return (
            f"専用VRAM ピーク {self.peak_gb:.2f} GB /"
            f" 共有GPUメモリ の増分ピーク {shared}"
        )


def power_report(index: int = 0) -> dict[str, float | None]:
    """温度・クロック・消費電力を1回読む.

    熱によるクロック低下は静かに進むので、学習中に定期的に記録しておく。
    ベンチと通しの実測が乖離する原因の切り分けに使う。
    """
    fields = (
        "temperature.gpu",
        "clocks.sm",
        "power.draw",
        "utilization.gpu",
        "memory.used",
    )
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={','.join(fields)}",
                "--format=csv,noheader,nounits",
                f"--id={index}",
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        )
        values = [v.strip() for v in out.stdout.strip().splitlines()[0].split(",")]
        result: dict[str, float | None] = {}
        for key, value in zip(fields, values, strict=False):
            try:
                result[key] = float(value)
            except ValueError:
                result[key] = None
        return result
    except Exception:
        return dict.fromkeys(fields)


def vram_holders() -> list[tuple[int, str, int]]:
    """VRAM を掴んでいるプロセス (pid, 名前, MiB).

    死んだプロセスが VRAM を掴んでいてもタスクマネージャでは見えない。
    """
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        )
    except Exception:
        return []
    holders = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            holders.append((int(parts[0]), parts[1], int(parts[2])))
        except ValueError:
            continue
    return holders


def release(index: int = 0) -> None:
    """終了処理で必ず呼ぶ. 掴んだままだと次の実行が OOM になる."""
    try:
        import torch

        torch.cuda.synchronize(index)
        torch.cuda.empty_cache()
    except Exception:
        pass


def replace_dir(src: Path, dst: Path, attempts: int = 20, wait: float = 0.1) -> None:
    """ディレクトリ src を dst に置き換える. Windows の遅延削除に耐える.

    素直に書くと次のようになるが、Windows ではときどき落ちる。

        if dst.exists():
            shutil.rmtree(dst)
        os.replace(src, dst)

    rmtree が返ってきても、Windows はディレクトリを「削除待ち」の印を付けた
    だけで、開いている handle が全部閉じるまで実体を残す。その隙間に
    os.replace を呼ぶと PermissionError (WinError 5) になる。
    ウイルス対策や検索インデックスが後ろで開いていると起きやすく、
    学習を20回続けて回すと数回に1回踏む。posix では起きない。

    そこで削除を待たずに済ませる。先に旧を別名へ退避 (rename は同期的に完了する)、
    新を本名へ移し、退避した旧をあとで消す。本名が空いている状態で移すので
    削除待ちと競合しない。それでも駄目なら少し待って何度か試す。
    """
    src, dst = Path(src), Path(dst)
    stale = dst.with_name(dst.name + ".stale")
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            if dst.exists():
                if stale.exists():
                    shutil.rmtree(stale, ignore_errors=True)
                os.replace(dst, stale)
            os.replace(src, dst)
            shutil.rmtree(stale, ignore_errors=True)
            return
        except PermissionError as exc:  # noqa: PERF203
            last = exc
            time.sleep(wait * (attempt + 1))
    raise OSError(
        f"{dst} を置き換えられませんでした ({attempts} 回試行)。"
        "エクスプローラや同期ソフトがこのディレクトリを開いていませんか。"
    ) from last

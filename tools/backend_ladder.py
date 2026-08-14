"""推論バックエンドのはしごを、8B で先に検証する.

32B の 18GB を落とす前に、6GB の 8B で「そのバックエンドが Windows で動くか」
だけを確かめる。Mac 版は 17GB を2回ダウンロードしている。

    python tools/backend_ladder.py --rung awq
    python tools/backend_ladder.py --rung nf4
    python tools/backend_ladder.py --rung gguf --target C:/path/model.gguf

見るのは3つ。
  1. 読み込めるか (カーネルの wheel が Windows に無いのはここで出る)
  2. <think> が混入しないか (enable_thinking=False が効いているか)
  3. 共有GPUメモリが 0 のままか (増えていたら静かに10倍遅くなっている)

結果は runs/backend_ladder.json に1行ずつ追記する。落ちた場合も追記する。
落ちた記録のほうが記事では価値がある。
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data" / "gal"))

from backends import Sampling, load_backend  # noqa: E402

import runtime  # noqa: E402

OUT_JSON = ROOT / "runs" / "backend_ladder.json"

# 8B で先に試す。32B と同じ構造 (GQA・chat template) なので、
# 動くかどうかの判定はそのまま持ち上がる。
DEFAULT_TARGETS = {
    "awq": "tokyotech-llm/Qwen3-Swallow-8B-RL-v0.2-AWQ-INT4",
    "nf4": "tokyotech-llm/Qwen3-Swallow-8B-RL-v0.2",
    "none": "tokyotech-llm/Qwen3-Swallow-8B-RL-v0.2",
}

# 検証用の短い会話。ギャル生成と同じ形 (system + user) にしてある。
PROBE = [
    [
        {"role": "system", "content": "あなたは日本語のセリフを書くプロの脚本家です。"},
        {
            "role": "user",
            "content": "「ラーメン」について、友達同士のLINEのやりとりを3通り作ってください。"
            "1行に1往復、全角の縦棒 ｜ で区切ってください。",
        },
    ],
    [
        {"role": "system", "content": "あなたは日本語の語彙に詳しい編集者です。"},
        {
            "role": "user",
            "content": "「アルバイト」に関係する具体的な話題を5個あげてください。1行に1つだけ。",
        },
    ],
]


def append_result(record: dict) -> None:
    records = []
    if OUT_JSON.exists():
        records = json.loads(OUT_JSON.read_text(encoding="utf-8"))
    records.append(record)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", choices=("awq", "nf4", "none", "gguf"), required=True)
    ap.add_argument("--target", default=None, help="リポジトリID か GGUF のパス")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument(
        "--template-repo",
        default=None,
        help="GGUF のとき、チャットテンプレートをこのリポジトリから借りる"
        " (enable_thinking=False を渡すため)",
    )
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    target = args.target or DEFAULT_TARGETS.get(args.rung)
    if target is None:
        raise SystemExit("--target が必要です")

    conversations = (PROBE * ((args.batch // len(PROBE)) + 1))[: args.batch]
    sampling = Sampling(seed=args.seed)

    print("=" * 74)
    print(f"はしご {args.rung}: {target}")
    print(f"サンプリング: {sampling.describe()}  (non-thinking 側の公式推奨値)")
    print("=" * 74)

    print(runtime.device_summary())
    holders = runtime.vram_holders()
    if holders:
        print("警告: VRAM を掴んでいるプロセスがあります")
        for pid, name, mib in holders:
            print(f"  pid {pid} {name} {mib} MiB")
    baseline_shared = runtime.shared_memory_gb()
    print(f"共有GPUメモリ (開始時): {baseline_shared}")
    print()

    record: dict = {
        "at": datetime.now().isoformat(timespec="seconds"),
        "rung": args.rung,
        "target": target,
        "batch": args.batch,
        "max_new_tokens": args.max_new_tokens,
        "sampling": sampling.describe(),
    }

    backend = None
    try:
        extra = {"template_repo": args.template_repo} if args.rung == "gguf" else {}
        backend = load_backend(args.rung, target, sampling=sampling, **extra)
        print(f"読み込めた: {backend.name}")
        print(f"  {backend.detail}")
        record["backend"] = backend.name
        record["detail"] = backend.detail

        # 基準はモデルを読み込んだ後に取り直す。VRAM に余裕があっても
        # 読み込みだけで 0.08 GB ほど共有側に出るのが Windows の常態で、
        # 読み込み前を基準にすると常に「こぼれた」と誤判定する。
        after_load_shared = runtime.shared_memory_gb()
        if after_load_shared is not None and baseline_shared is not None:
            record["load_shared_gb"] = round(after_load_shared - baseline_shared, 3)
            print(f"  読み込みで増えた共有GPUメモリ: +{record['load_shared_gb']:.2f} GB (これを基準にする)")

        # 1回目はカーネルの初期化を含むので、2回測って2回目を採る
        for attempt in (1, 2):
            result = backend.chat_batch(conversations, args.max_new_tokens)
            shared = runtime.shared_memory_gb()
            spill = None
            if shared is not None and after_load_shared is not None:
                spill = shared - after_load_shared
            print(
                f"  {attempt}回目: {result.seconds:.1f}秒 /"
                f" {result.completion_tokens} トークン /"
                f" {result.tokens_per_sec:.1f} tok/s /"
                f" VRAM ピーク {result.peak_vram_gb:.1f} GB /"
                f" 共有 {'取得不可' if spill is None else f'+{spill:.2f} GB'}"
            )
        record["seconds"] = round(result.seconds, 2)
        record["completion_tokens"] = result.completion_tokens
        record["tokens_per_sec"] = round(result.tokens_per_sec, 1)
        record["peak_vram_gb"] = round(result.peak_vram_gb, 2)
        record["shared_spill_gb"] = None if spill is None else round(spill, 3)
        record["notes"] = result.notes

        has_think = any("<think>" in t or "</think>" in t for t in result.texts)
        record["think_leaked"] = has_think
        print(f"  <think> の混入: {'あり (対策が効いていない)' if has_think else 'なし'}")
        print(f"  こぼれ: {'あり' if (spill or 0) > 0.05 else 'なし'}")
        print("\n--- 生成された文 (先頭2件) ---")
        for text in result.texts[:2]:
            print(text.strip()[:400])
            print("-" * 40)
        record["sample"] = [t.strip()[:400] for t in result.texts[:2]]
        record["ok"] = True
    except Exception as exc:
        record["ok"] = False
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc()[-2000:]
        print(f"\n落ちた: {type(exc).__name__}: {exc}")
        print(traceback.format_exc())
    finally:
        if backend is not None:
            backend.close()
        runtime.release()
        append_result(record)
        print(f"\n記録: {OUT_JSON}")


if __name__ == "__main__":
    main()

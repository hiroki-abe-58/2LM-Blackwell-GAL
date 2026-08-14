"""モデルを1バイトも落とさずに、量子化版の在庫と KV の必要量を調べる.

Mac 版は 17GB を2回ダウンロードした。config.json だけ取れば
1トークンあたりの KV は計算できるので、先に計算してから落とす。

    python tools/model_survey.py
    python tools/model_survey.py --save

在庫の確認は Hugging Face のファイル一覧だけを見る。AWQ / GGUF の
「あるはず」を信じないこと。tokyotech-llm は GPTQ 版を性能劣化のため
非公開にした前例がある（DATAGEN.md 3.1）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from runtime import kv_bytes_per_token, kv_report  # noqa: E402

OUT_JSON = ROOT / "runs" / "model_survey.json"

# 素の重み（bf16）と、その量子化版として在庫を確認したい候補。
CANDIDATES = (
    "tokyotech-llm/Qwen3-Swallow-32B-RL-v0.2",
    "tokyotech-llm/Qwen3-Swallow-32B-RL-v0.2-AWQ-INT4",
    "tokyotech-llm/Qwen3-Swallow-30B-A3B-RL-v0.2",
    "tokyotech-llm/Qwen3-Swallow-30B-A3B-RL-v0.2-AWQ-INT4",
    "tokyotech-llm/Qwen3-Swallow-8B-RL-v0.2",
    "tokyotech-llm/Qwen3-Swallow-8B-RL-v0.2-AWQ-INT4",
    "Qwen/Qwen2.5-32B-Instruct-AWQ",
    "Qwen/Qwen3-8B-AWQ",
)

VRAM_GB = 32.0


def survey(repo_id: str) -> dict:
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    row: dict = {"repo_id": repo_id}
    try:
        info = api.model_info(repo_id, files_metadata=True)
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
        return row

    siblings = info.siblings or []
    row["files"] = len(siblings)
    weights = [s for s in siblings if s.rfilename.endswith((".safetensors", ".gguf", ".bin"))]
    total = sum(s.size or 0 for s in weights)
    row["weight_gb"] = round(total / 1024**3, 2) if total else None
    row["formats"] = sorted({Path(s.rfilename).suffix for s in weights})
    row["license"] = (info.card_data or {}).get("license") if info.card_data else None

    try:
        path = hf_hub_download(repo_id, "config.json")
        config = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        row["error"] = f"config.json: {type(exc).__name__}: {exc}"
        return row

    row["arch"] = (config.get("architectures") or ["?"])[0]
    row["layers"] = config.get("num_hidden_layers")
    row["q_heads"] = config.get("num_attention_heads")
    row["kv_heads"] = config.get("num_key_value_heads")
    row["head_dim"] = config.get("head_dim") or (
        (config.get("hidden_size") or 0) // (config.get("num_attention_heads") or 1)
    )
    row["kv_kb_per_token"] = round(kv_bytes_per_token(config) / 1024, 1)
    quant = config.get("quantization_config")
    if quant:
        row["quant"] = f"{quant.get('quant_method')} {quant.get('bits', '')}bit".strip()
    else:
        row["quant"] = config.get("torch_dtype") or "なし"
    # MoE かどうか。デコードの速さに直結する
    if config.get("num_experts") or config.get("n_routed_experts"):
        row["experts"] = config.get("num_experts") or config.get("n_routed_experts")
        row["experts_used"] = config.get("num_experts_per_tok")
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=2048)
    args = ap.parse_args()

    rows = [survey(r) for r in CANDIDATES]

    print("=" * 100)
    print("量子化版の在庫と KV（ダウンロードはしていない。config.json だけ取得）")
    print("=" * 100)
    header = (
        f"{'モデル':<48} {'量子化':<12} {'重み':>8} {'層':>4} {'Q':>4} {'KV':>4} {'KV/tok':>8}"
    )
    print(header)
    print("-" * 100)
    for row in rows:
        name = row["repo_id"].split("/")[-1]
        if "error" in row and "layers" not in row:
            print(f"{name:<48} {'取得できず':<12}  {row['error'][:40]}")
            continue
        weight = f"{row['weight_gb']:.1f}GB" if row.get("weight_gb") else "?"
        print(
            f"{name:<48} {str(row.get('quant')):<12} {weight:>8}"
            f" {row.get('layers', '?'):>4} {row.get('q_heads', '?'):>4}"
            f" {row.get('kv_heads', '?'):>4} {row['kv_kb_per_token']:>7.0f}KB"
        )
        if row.get("experts"):
            print(f"{'':<48} MoE: {row['experts']} エキスパートのうち {row['experts_used']} を使う")

    print()
    print("パラメータ数ではなく KVヘッド数でバッチが決まる。")
    print("8B のほうが 30B-A3B より KV が大きいなら、それが Mac 版 その3 の教訓の再演である。")

    usable = [r for r in rows if r.get("kv_kb_per_token") and r.get("weight_gb")]
    for row in usable:
        print()
        print("=" * 74)
        print(f"{row['repo_id']}  ({row['quant']})")
        print("=" * 74)
        config = {
            "num_hidden_layers": row["layers"],
            "num_attention_heads": row["q_heads"],
            "num_key_value_heads": row["kv_heads"],
            "head_dim": row["head_dim"],
        }
        print(kv_report(config, row["weight_gb"], VRAM_GB, max_tokens=args.max_tokens))

    if args.save:
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        OUT_JSON.write_text(
            json.dumps(
                {"vram_gb": VRAM_GB, "max_tokens": args.max_tokens, "models": rows},
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print(f"\n保存: {OUT_JSON}")


if __name__ == "__main__":
    main()

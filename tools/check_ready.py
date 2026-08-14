"""境界の総当たりを始める前に、必要なものが揃っているか確かめる.

途中まで走ってから「基準モデルが無い」で落ちると、20回の学習のうち
何回ぶんかを捨てることになる。先に全部確認する。
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

NEEDED = (
    ("data/corpus.txt", "公開データのコーパス (2LM から流用)"),
    ("data/raw/gal_line.jsonl", "検査を通ったギャル語の会話"),
    ("data/tokenizer/tokenizer.model", "語彙8,000のトークナイザ (作り直さない)"),
    ("checkpoints/final", "追加学習の出発点となる事前学習済みモデル"),
    ("eval/questions.jsonl", "評価に使う固定設問"),
    ("src/train.py", "学習"),
    ("tools/reply_metrics.py", "口調と崩れの計測"),
    ("tools/boundary_sweep.py", "総当たり"),
)


def main() -> int:
    missing = 0
    for rel, why in NEEDED:
        path = ROOT / rel
        if path.is_dir():
            files = sorted(f.name for f in path.iterdir())
            size = sum(f.stat().st_size for f in path.iterdir() if f.is_file())
            print(f"OK  {rel:<32} {size / 1e6:>7.1f} MB  {', '.join(files)}")
        elif path.exists():
            print(f"OK  {rel:<32} {path.stat().st_size / 1e6:>7.1f} MB  {why}")
        else:
            print(f"NG  {rel:<32} {'':>7}     {why}")
            missing += 1

    gal = ROOT / "data" / "raw" / "gal_line.jsonl"
    if gal.exists():
        count = sum(1 for line in gal.read_text(encoding="utf-8").splitlines() if line.strip())
        print()
        print(f"ギャル語の会話: {count:,} 件")
        for need in (1000, 1500, 2000, 3000, 5000):
            mark = "足りる" if count >= need else "足りない"
            print(f"  {need:>5,} 件の条件: {mark}")

    if missing:
        print(f"\n{missing} 件足りません")
        return 1
    print("\nすべて揃っています")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

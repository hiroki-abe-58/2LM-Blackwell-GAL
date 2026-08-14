"""同じ問いかけに、追加学習の前と後で何を返すかを並べて出す.

種を揃えるので、違いは重みの違いだけになる。記事とスライドの
「効果が一目で分かる図」がこれになる。

    python tools/compare_style.py
    python tools/compare_style.py --limit 8 > runs/style.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from reply_metrics import generate_replies, load_prompts, style_scores  # noqa: E402
from src.generate import load_bundle  # noqa: E402


def replies_for(ckpt: str, prompts: list[str], seed: int, max_new_tokens: int) -> list[str]:
    model, tokenizer = load_bundle(ckpt)
    texts, _ = generate_replies(
        model, tokenizer, prompts, seed, max_new_tokens,
        repeats=1, temperature=0.8, top_k=40, repetition_penalty=1.15,
    )
    del model
    torch.cuda.empty_cache()
    return texts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", default="checkpoints/final")
    ap.add_argument("--after", default="checkpoints/gal")
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=120)
    args = ap.parse_args()

    prompts = load_prompts(args.limit)
    before = replies_for(args.before, prompts, args.seed, args.max_new_tokens)
    after = replies_for(args.after, prompts, args.seed, args.max_new_tokens)

    print("=" * 74)
    print(f"  同じ問いかけ・同じ種 ({args.seed}) で、追加学習の前と後")
    print("=" * 74)
    for prompt, b, a in zip(prompts, before, after, strict=True):
        print()
        print(f"  問い  {prompt}")
        print(f"  前    {b.strip()}")
        print(f"  後    {a.strip()}")

    print()
    print("-" * 74)
    for label, texts in (("前", before), ("後", after)):
        scores = style_scores(texts)
        print(
            f"  {label}  ギャル度 {scores['gal_rate']:.2f}"
            f" / 一人称「うち」 {scores['first_person_rate']:.2f}"
            f" / 敬語 {scores['polite_rate']:.2f}"
            f" / 平均 {scores['avg_len']:.0f}文字"
        )


if __name__ == "__main__":
    main()

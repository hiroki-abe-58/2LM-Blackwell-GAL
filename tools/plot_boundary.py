"""境界の実測結果を図にする.

3段に分ける。上から順に「口調が入ったか」「文として終われたか」
「繰り返しに落ちたか」。横軸は会話数、線の色が混合比。

崩れの指標は3つ出す。1つでは足りないことが実測で分かった。

  距離 (excess_bits) 文字の並びが壊れる崩れを捉える。ただし混合比が違うと
                     水準ごと変わるので、同じ混合比の中でだけ比べる
  打ち切り           長く書き続けて止まれない崩れを捉える
  繰り返し           同じ言い回しから抜け出せない崩れを捉える

    python tools/plot_boundary.py --out ../docs/images/03-boundary.png
    python tools/plot_boundary.py --json runs/lr.json --x lr \
        --out ../docs/images/03-lr.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from plotting import SERIES_COLORS, dark_figure

ROOT = Path(__file__).resolve().parents[1]

PANELS = (
    ("gal_rate", "口調 — ギャル度", "一人称・語尾で判定", (-0.05, 1.05)),
    ("excess_bits", "崩れ — 学習データ自身からの距離", "bits/char の超過 (0 が学習データと同等)", None),
    ("loop_rate", "崩れ — 繰り返した率", "抜け出せなかった割合", None),
)

X_LABELS = {
    "count": "学習に使った会話数",
    "epochs": "コーパスの周回数",
    "lr": "学習率",
}
FIXED_NOTE = {
    "count": "周回数 8 固定",
    "epochs": "会話 2,000 件 固定",
    "lr": "会話 2,000 件 / 24 周 固定",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(ROOT / "runs" / "boundary.json"))
    ap.add_argument("--out", default=str(ROOT / "runs" / "boundary.png"))
    ap.add_argument("--x", choices=tuple(X_LABELS), default="count")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    data = json.loads(Path(args.json).read_text(encoding="utf-8"))
    runs = [r for r in data["runs"] if r.get("gal_rate") is not None]
    mixes = sorted({r["mix"] for r in runs}, reverse=True)

    fig, axes = dark_figure(len(PANELS), figsize=(8.4, 9.0))
    for index, mix in enumerate(mixes):
        series = sorted((r for r in runs if r["mix"] == mix), key=lambda r: r[args.x])
        xs = [r[args.x] for r in series]
        color = SERIES_COLORS[index % len(SERIES_COLORS)]
        for ax, (field, _, _, _) in zip(axes, PANELS, strict=True):
            ax.plot(xs, [r.get(field) for r in series], marker="o", color=color,
                    label=f"ギャル {mix}%")

    for ax, (field, title, note, ylim) in zip(axes, PANELS, strict=True):
        ax.set_ylabel(note, fontsize=9)
        ax.set_title(title, fontsize=10, loc="left")
        if ylim:
            ax.set_ylim(*ylim)
        if field == "excess_bits":
            # 0 は「学習データのギャル語と同じ予測しやすさ」。上に出たら崩れている。
            ax.axhline(0.0, color="#ffffff", alpha=0.35, linestyle="--", linewidth=1)
        ax.legend(loc="best", fontsize=8, framealpha=0.2)
        if args.x in ("epochs", "lr"):
            ax.set_xscale("log")
    axes[-1].set_xlabel(X_LABELS[args.x])

    fig.suptitle(
        args.title or f"崩れる境界の実測 — 横軸 {X_LABELS[args.x]} ({FIXED_NOTE[args.x]})",
        fontsize=11,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, facecolor=fig.get_facecolor())
    print(f"書き出し: {out} ({len(runs)} 条件)")


if __name__ == "__main__":
    main()

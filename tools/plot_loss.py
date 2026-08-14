"""runs/loss.csv から学習曲線の画像を作る (記事用).

    python tools/plot_loss.py
    python tools/plot_loss.py --csv runs/loss.csv --out docs/images/loss-curve.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.loss_log import read_curve  # noqa: E402
from tools.plotting import SERIES_COLORS, dark_axes  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="runs/loss.csv")
    ap.add_argument("--out", default="docs/images/loss-curve.png")
    ap.add_argument("--title", default="2LM-Blackwell training curve (RTX 5090)")
    args = ap.parse_args()

    curve = read_curve(args.csv)
    train_x = [s for s, _ in curve.train]
    train_y = [v for _, v in curve.train]
    val_x = [s for s, _ in curve.val]
    val_y = [v for _, v in curve.val]

    fig, ax = dark_axes()
    ax.plot(train_x, train_y, color=SERIES_COLORS[0], lw=1.6, label="train loss")
    ax.plot(val_x, val_y, color=SERIES_COLORS[1], lw=1.8, marker="o", ms=3, label="val loss")
    ax.set_xlabel("step")
    # サブワードなので分母はトークン。文字レベル版の nats/char と混ぜてはいけない。
    ax.set_ylabel("cross entropy (nats / token)")
    ax.set_title(args.title, loc="left", fontsize=11)
    ax.legend(frameon=False)
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=fig.get_facecolor())
    print(f"保存: {out}  (train {len(train_x)}点 / val {len(val_x)}点)")


if __name__ == "__main__":
    main()

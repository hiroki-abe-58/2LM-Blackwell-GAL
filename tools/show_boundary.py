"""runs/boundary.json を表にして出す.

総当たりの途中でも読める。走らせながら傾向を見るために使う。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(ROOT / "runs" / "boundary.json"))
    ap.add_argument("--examples", action="store_true", help="返答の見本も出す")
    args = ap.parse_args()

    data = json.loads(Path(args.json).read_text(encoding="utf-8"))
    runs = data["runs"]
    print(f"周回数 {data['epochs']} / lr {data['lr']} / {len(runs)} 条件")
    print()
    # 日本語は幅2で表示されるが f-string の桁揃えは1と数える。
    # 見出しは実際の表示幅で手作りする。
    print("  会話  ギャル比       lr    周    歩  ギャル度   敬語   打切   繰返     距離")
    print("-" * 66)
    for r in runs:
        print(
            f"{r['count']:>6,} {r['mix']:>8}% {r.get('lr', 0):>8.0e} "
            f"{r.get('epochs', 0):>4g} {r.get('steps', 0):>5} "
            f"{r.get('gal_rate', float('nan')):>8} {r.get('polite_rate', 0):>6} "
            f"{r.get('cut_rate', float('nan')):>6} {r.get('loop_rate', float('nan')):>6} "
            f"{r.get('excess_bits', float('nan')):>8}"
        )

    if args.examples:
        for r in runs:
            print()
            print(f"--- 会話 {r['count']:,} / ギャル {r['mix']}% ---")
            for text in r.get("examples", []):
                print(f"  {text}")


if __name__ == "__main__":
    main()

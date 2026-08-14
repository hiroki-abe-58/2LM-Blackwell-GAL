"""総当たりで残したチェックポイントを、学習し直さずに再採点する.

指標を足したり直したりするたびに 20 回学習し直すのは無駄なので、
採点だけを分けてある。runs/sweep/ck_* が残っている限り何度でも回せる。

    python tools/rescore_boundary.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
WORK = ROOT / "runs" / "sweep"
RESULT = ROOT / "runs" / "boundary.json"
BASE_CKPT = ROOT / "checkpoints" / "final"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(RESULT))
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    path = Path(args.json)
    data = json.loads(path.read_text(encoding="utf-8"))
    started = time.time()

    for record in data["runs"]:
        key = record["key"]
        ckpt = WORK / f"ck_{key}"
        if not ckpt.exists():
            print(f"[飛ばす] {key}: {ckpt} が無い")
            continue

        metrics_path = WORK / f"metrics_{key}.json"
        log = WORK / f"metrics_{key}.log"
        with log.open("w", encoding="utf-8", newline="\n") as f:
            proc = subprocess.run(
                [
                    PY, "tools/reply_metrics.py",
                    "--ckpt", str(ckpt),
                    "--reference", str(BASE_CKPT),
                    "--repeats", str(args.repeats),
                    "--json", str(metrics_path),
                ],
                stdout=f, stderr=subprocess.STDOUT, cwd=ROOT, text=True,
            )
        if proc.returncode != 0:
            tail = "\n".join(log.read_text(encoding="utf-8").splitlines()[-20:])
            raise SystemExit(f"失敗: {key}\n--- {log} の末尾 ---\n{tail}")

        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        record.update({k: v for k, v in metrics.items() if k != "replies"})
        record["examples"] = metrics["replies"][:4]
        print(
            f"{key:>10}  ギャル度 {record.get('gal_rate')} / 打ち切り {record.get('cut_rate')}"
            f" / 繰り返し {record.get('loop_rate')} / 崩れ {record.get('excess_bits')}"
        )

    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"\n更新: {path} ({round(time.time() - started, 1)} 秒)")


if __name__ == "__main__":
    main()

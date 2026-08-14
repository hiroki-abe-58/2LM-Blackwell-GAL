"""日本語が崩れなくなる境界を、会話数 × 混合比の総当たりで実測する.

Mac 版で分かっていたのは2点だけだった。

  855件で混ぜないと崩れた / 2,610件では崩れなかった
  ギャルの文字数比率が 85% 以下だと口調が元に戻る

その間のどこに境界があるかは測っていない。ここで埋める。

    会話数   1,000 / 1,500 / 2,000 / 3,000 / 5,000
    混合比   ギャルの文字数比率 0 / 50 / 85 / 100 %

**周回数を揃えるのが肝心**である。ステップ数を固定すると、少ないデータほど
何周も回ることになり「データ量の効果」と「丸暗記の効果」が混ざる。
ここでは混合後のコーパスに対する周回数を固定し、ステップ数を逆算する。

    python tools/boundary_sweep.py                       # 総当たり (20通り)
    python tools/boundary_sweep.py --counts 2000 --mixes 100
    python tools/boundary_sweep.py --resume              # 済みは飛ばす
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from mix_corpus import load_part  # noqa: E402

PY = sys.executable
GAL_JSONL = ROOT / "data" / "raw" / "gal_line.jsonl"
PUBLIC_CORPUS = ROOT / "data" / "corpus.txt"
BASE_CKPT = ROOT / "checkpoints" / "final"
WORK = ROOT / "runs" / "sweep"
RESULT = ROOT / "runs" / "boundary.json"

COUNTS = (1000, 1500, 2000, 3000, 5000)
MIXES = (0, 50, 85, 100)


def run(cmd: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8", newline="\n") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT, text=True)
    if proc.returncode != 0:
        tail = "\n".join(log.read_text(encoding="utf-8").splitlines()[-25:])
        raise SystemExit(f"失敗: {' '.join(cmd)}\n--- {log} の末尾 ---\n{tail}")


def subsample(count: int, seed: int) -> Path:
    """会話を count 件だけ抜き出して、prepare.py にかけられる形で置く."""
    pairs = [
        line
        for line in GAL_JSONL.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(pairs) < count:
        raise SystemExit(
            f"{GAL_JSONL} は {len(pairs)} 件しかありません ({count} 件必要)。"
            "先に data/gal/generate.py で増やしてください。"
        )
    picked = random.Random(seed).sample(pairs, count)
    raw_dir = WORK / f"raw_{count}"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "gal_line.jsonl").write_text(
        "\n".join(picked) + "\n", encoding="utf-8", newline="\n"
    )
    return raw_dir


def gal_corpus(count: int, seed: int) -> Path:
    """会話 count 件からギャル語コーパスを作る (公開データは混ぜない)."""
    out = WORK / f"corpus_gal_{count}.txt"
    if out.exists():
        return out
    raw_dir = subsample(count, seed)
    # --min-char-freq 1: 事前学習済みの語彙には byte fallback があるので
    # 低頻度文字を捨てる必要がない。数千件で 10 回未満を捨てると何も残らない。
    run(
        [
            PY,
            "data/prepare.py",
            "--no-hf",
            "--raw-dir",
            str(raw_dir),
            "--out",
            str(out),
            "--min-char-freq",
            "1",
        ],
        WORK / f"prepare_{count}.log",
    )
    return out


def mixed_corpus(count: int, gal_percent: int, seed: int) -> tuple[Path, dict]:
    """ギャルの文字数比率が gal_percent になるようコーパスを混ぜる."""
    out = WORK / f"corpus_{count}_{gal_percent}.txt"
    gal_path = gal_corpus(count, seed)
    gal_lines = [ln for ln in gal_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    gal_chars = sum(len(ln) for ln in gal_lines)
    public_chars = sum(
        len(ln) for ln in PUBLIC_CORPUS.read_text(encoding="utf-8").splitlines() if ln.strip()
    )

    rng = random.Random(seed)
    if gal_percent >= 100:
        merged = list(gal_lines)
        public_ratio = 0.0
    else:
        share = gal_percent / 100
        if share == 0:
            # 対照実験。ギャル語を1件も入れず、規模だけ揃える。
            public_ratio = gal_chars / public_chars
            merged = []
        else:
            public_ratio = gal_chars * (1 - share) / (share * public_chars)
            merged = list(gal_lines)
        _, public_lines = load_part(f"{PUBLIC_CORPUS}:{public_ratio}", rng)
        merged += public_lines

    rng.shuffle(merged)
    out.write_text("\n".join(merged) + "\n", encoding="utf-8", newline="\n")
    total = sum(len(ln) for ln in merged)
    info = {
        "gal_chars": gal_chars,
        "public_ratio": round(public_ratio, 6),
        "total_chars": total,
        "gal_share": round(gal_chars / total, 4) if gal_percent else 0.0,
        "lines": len(merged),
    }
    return out, info


def token_count(corpus: Path, cache_dir: Path) -> int:
    """トークン化してステップ数の逆算に使う.

    ここで作った cache は train.py がそのまま使い回す (二重にトークン化しない)。
    """
    from src.tokenizer import load_tokenizer
    from src.train import build_dataset

    tokens, _ = build_dataset(corpus, cache_dir, 1, 8000, load_tokenizer(BASE_CKPT))
    return int(tokens.size)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", type=int, nargs="+", default=list(COUNTS))
    ap.add_argument("--mixes", type=int, nargs="+", default=list(MIXES))
    ap.add_argument(
        "--epochs",
        type=float,
        nargs="+",
        default=[8.0],
        help="混合後コーパスの周回数。複数渡すと周回数も軸にする",
    )
    ap.add_argument("--out", default=str(RESULT), help="記録の置き場")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument(
        "--lr",
        type=float,
        nargs="+",
        default=[1e-4],
        help="学習率。複数渡すと学習率も軸にする",
    )
    ap.add_argument("--warmup", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="済みの組は飛ばす")
    ap.add_argument("--dry-run", action="store_true", help="コーパスを作って規模だけ出す")
    args = ap.parse_args()

    if not GAL_JSONL.exists():
        raise SystemExit(f"{GAL_JSONL} がありません。先にデータ生成を回してください。")
    if not PUBLIC_CORPUS.exists():
        raise SystemExit(f"{PUBLIC_CORPUS} がありません。data/prepare.py を回してください。")

    WORK.mkdir(parents=True, exist_ok=True)
    result_path = Path(args.out)
    # 値が1つしかない軸は鍵に入れない。1軸だけ動かした記録と読み合わせられるようにする。
    axes = {"epochs": args.epochs, "lr": args.lr}
    varying = [name for name, values in axes.items() if len(values) > 1]
    done: dict[str, dict] = {}
    if args.resume and result_path.exists():
        done = {r["key"]: r for r in json.loads(result_path.read_text(encoding="utf-8"))["runs"]}

    results: list[dict] = list(done.values())
    started = time.time()
    grid = [
        (c, m, e, lr)
        for c in args.counts
        for m in args.mixes
        for e in args.epochs
        for lr in args.lr
    ]
    for count, mix, epochs, lr in grid:
        parts = {"epochs": f"{epochs:g}", "lr": f"{lr:g}"}
        key = "_".join([f"{count}", f"{mix}", *(parts[name] for name in varying)])
        label = f"会話 {count} 件 / ギャル比率 {mix}% / {epochs:g} 周 / lr {lr:g}"
        if key in done:
            print(f"[飛ばす] {label}")
            continue

        print("=" * 74)
        print(f"  {label}")
        print("=" * 74)
        corpus, info = mixed_corpus(count, mix, args.seed)
        # cache はコーパスに対応する。周回数を変えても同じものを使い回す。
        cache = WORK / f"cache_{count}_{mix}"
        tokens = token_count(corpus, cache)
        per_step = args.batch_size * args.block_size
        steps = max(args.warmup + 8, round(epochs * tokens / per_step))
        print(
            f"  {info['lines']:,} 行 / {info['total_chars']:,} 文字 / {tokens:,} トークン"
            f" / ギャル {info['gal_share']:.1%}"
        )
        print(f"  {epochs:g} 周 = {steps} ステップ")
        base = {
            "key": key, "count": count, "mix": mix,
            "epochs": epochs, "lr": lr, "steps": steps,
        }
        if args.dry_run:
            results.append({**base, **info})
            continue

        ckpt = WORK / f"ck_{key}"
        t0 = time.time()
        run(
            [
                PY, "src/train.py",
                "--init-from", str(BASE_CKPT),
                "--corpus", str(corpus),
                "--cache-dir", str(cache),
                "--out", str(ckpt),
                "--resume-dir", str(WORK / f"last_{key}"),
                "--log", str(WORK / f"loss_{key}.csv"),
                "--lr", str(lr),
                "--warmup", str(args.warmup),
                "--steps", str(steps),
                "--batch-size", str(args.batch_size),
                "--eval-interval", str(steps),
                "--no-samples",
            ],
            WORK / f"train_{key}.log",
        )
        train_seconds = round(time.time() - t0, 1)

        metrics_path = WORK / f"metrics_{key}.json"
        run(
            [
                PY, "tools/reply_metrics.py",
                "--ckpt", str(ckpt),
                "--reference", str(BASE_CKPT),
                "--json", str(metrics_path),
            ],
            WORK / f"metrics_{key}.log",
        )
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

        record = {
            **base,
            "tokens": tokens,
            "train_seconds": train_seconds,
            **info,
            **{k: v for k, v in metrics.items() if k != "replies"},
            "examples": metrics["replies"][:4],
        }
        results.append(record)
        print(
            f"  ギャル度 {record.get('gal_rate')} / 打ち切り {record.get('cut_rate')}"
            f" / 繰り返し {record.get('loop_rate')} / {train_seconds}秒"
        )

        results.sort(key=lambda r: (r["count"], r["mix"], r.get("epochs", 0), r.get("lr", 0)))
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps(
                {"epochs": args.epochs, "lr": args.lr, "varying": varying, "runs": results},
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )

    print()
    print(
        f"{'会話':>6} {'ギャル比率':>10} {'周':>5} {'lr':>8} {'歩':>6} "
        f"{'ギャル度':>8} {'打切':>6} {'繰返':>6}"
    )
    for r in results:
        print(
            f"{r['count']:>6,} {r['mix']:>9}% {r.get('epochs', 0):>5g} {r.get('lr', 0):>8.0e} "
            f"{r.get('steps', 0):>6} "
            f"{r.get('gal_rate', float('nan')):>8} {r.get('cut_rate', float('nan')):>6} "
            f"{r.get('loop_rate', float('nan')):>6}"
        )
    print(f"\n合計 {round(time.time() - started, 1)} 秒 / 記録: {result_path}")


if __name__ == "__main__":
    main()

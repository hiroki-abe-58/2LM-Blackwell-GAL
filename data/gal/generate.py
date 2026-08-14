"""ローカルLLMに架空の「ギャルのLINE」会話を書かせてコーパスを作る.

生成には Apache-2.0 のローカルモデルだけを使う。出力の利用に条件が付かないので、
作ったデータセットも、そこから学習した重みも配布できる。
着手前に tools/license_check.py を通すこと (LICENSING の5項目)。

4段階に分かれている。

  calibrate : バッチサイズを決める。「共有GPUメモリが 0 のまま動いた最大値」を採る
  topics    : ジャンルだけ与えて、具体的な話題をモデルに列挙させる
  pairs     : 話題 x 用件 x 機嫌 の組み合わせごとに往復を書かせ、生のまま追記する
  build     : 生の出力を検査してふるいにかけ、学習用の1本にまとめる

日本語の中身をこちらで書かないのは意図的で、話題の語彙まで人間が用意すると
そこが多様性の上限になる。ジャンルという骨組みだけ決めて、肉は全部モデルに付けさせる。

pairs は1バッチごとに追記していく。途中で止まっても、もう一度同じコマンドを
叩けば済んだ組み合わせを飛ばして続きから走る。Windows では「落ちずに10倍遅くなる」
という止まり方をするので、人間が途中で止められる形にしておくことがより重要になる。

使い方:
    python data/gal/generate.py --stage calibrate
    python data/gal/generate.py --stage topics
    python data/gal/generate.py --stage pairs --target 12000 --batch-size 24
    python data/gal/generate.py --stage build
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from itertools import product
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from backends import Sampling, load_backend  # noqa: E402
from validate import filter_pairs, report  # noqa: E402

import runtime  # noqa: E402

TOPICS_PATH = HERE / "topics.txt"
# 検査前の生の出力。data/raw/ に置くと prepare.py が未検査のまま拾ってしまうので、
# 意図的にこちら側へ置いている。
RAW_PATH = HERE / "raw.jsonl"
DEFAULT_OUT = ROOT / "data" / "raw" / "gal_line.jsonl"
META_OUT = ROOT / "data" / "raw" / "gal_line.meta.json"

# AutoAWQ 製の AWQ を選ぶこと。tokyotech-llm の AWQ-INT4 は gptqmodel 製で、
# Windows では読み込めてしまうのに重みが壊れる (backends.check_awq_checkpoint)。
DEFAULT_MODEL = "Qwen/Qwen2.5-32B-Instruct-AWQ"
DEFAULT_QUANT = "awq"

# 文体は「仕様」として渡す。例文を書いて渡すと、モデルはそれを言い換えるだけになり、
# 手本の数だけしか語彙が広がらない。禁止事項だけ具体的に、中身は指定しない。
STYLE = """あなたは日本語のセリフを書くプロの脚本家です。
架空のキャラクター「ギャル」がLINEで返信する場面のセリフを書きます。
実在の人物とは関係のない、完全な創作です。

このキャラクターの話し方:
- 一人称は「うち」
- 敬語を使わない。友達に送るくだけた口調
- 返信は短い。40文字を超えない
- 相手を否定しない。明るくてノリがいい
- 知ったかぶりをせず、わからないことは正直にわからないと言う
- ときどきボケる。まじめに答えすぎない

絶対に守る決まり:
- 絵文字と顔文字を使わない
- 使ってよい記号は 、。！？〜… と w だけ
- 英単語を書かない。カタカナで書く
- 関西弁などの方言にしない。標準語のくだけた話し方で"""

# ジャンルだけを決める。ここから先の具体的な話題はモデルに出させる。
CATEGORIES = (
    "食べ物と飲み物", "学校と勉強", "アルバイト", "恋愛", "友達づきあい",
    "お金", "天気と季節", "旅行とおでかけ", "美容とファッション", "体調と健康",
    "音楽と映画", "ゲームとアニメ", "スマホとインターネット", "家族", "家事と生活",
    "仕事と将来", "スポーツと運動", "動物とペット", "科学と技術", "世の中のできごと",
)

# ユーザー側が何をしてくるか。会話の型が偏らないようにする。
INTENTS = (
    "質問する", "感想を言う", "愚痴をこぼす", "誘う", "報告する",
    "相談する", "頼みごとをする", "あいさつする",
)

# 返す側の機嫌。同じ話題でも返しの温度が変わる。
# Mac 版はこれを均等に振ったので、空腹と眠気が全体の40%を占めた。
# 13.8M のモデルは「今どの機嫌か」を条件として切り分けられず、平均的な人格として出る。
MOODS = ("機嫌がいい", "眠くてだるい", "テンションが高い", "ちょっと呆れている", "腹をすかせている")

_TOPIC_LINE_RE = re.compile(r"^\s*(?:[-*・]|\d+[.、)）])?\s*(.+?)\s*$")

# 1行に1往復を「｜」で区切って書かせる。本文に出ない全角記号を選んでいる。
SEPARATOR = "｜"
# モデルは指示しても番号・箇条書き・話者名・かぎかっこを付けてくる。
# 何度指示を書き直しても一定の割合で混ざるので、諦めて機械的に剥がす。
_LEAD_NOISE_RE = re.compile(r"^\s*(?:[-*・]|\d+\s*[.、)）:：])?\s*")
_ROLE_RE = re.compile(r"^\s*(?:ユーザー|相手|あなた|ギャル|返信|assistant|user)\s*[:：]\s*", re.I)
_QUOTE_RE = re.compile(r"^[「『\"'（(]+|[」』\"')）]+$")
_DIGITS_ONLY_RE = re.compile(r"^\d+$")


def _clean_side(text: str) -> str:
    """1発言ぶんから、番号・話者名・かぎかっこを剥がす."""
    text = _LEAD_NOISE_RE.sub("", text.strip())
    text = _ROLE_RE.sub("", text)
    for _ in range(2):  # 「「〜」」のように重なっていることがある
        text = _QUOTE_RE.sub("", text.strip())
    return text.strip()


def parse_pairs(text: str) -> list[tuple[str, str]]:
    pairs = []
    for line in text.splitlines():
        # 区切りが2つ以上ある行は、1行に2往復を詰めたか末尾に余分を付けたか判別できない。
        # 最初の2つを採ると往復の対応がずれるので、行ごと捨てる。
        fields = line.split(SEPARATOR)
        if len(fields) != 2:
            continue
        user, assistant = (_clean_side(s) for s in fields)
        # 番号だけの行を往復として拾ってしまう事故があった。数字だけの側は捨てる。
        if not user or not assistant:
            continue
        if _DIGITS_ONLY_RE.match(user) or _DIGITS_ONLY_RE.match(assistant):
            continue
        pairs.append((user, assistant))
    return pairs


def topic_prompt(category: str, per_category: int) -> list[dict]:
    return [
        {"role": "system", "content": "あなたは日本語の語彙に詳しい編集者です。"},
        {
            "role": "user",
            "content": (
                f"「{category}」に関係する具体的な話題を{per_category}個あげてください。\n"
                "条件: 1行に1つだけ。名詞か短い言い回しで。番号や記号は付けない。説明も書かない。"
            ),
        },
    ]


def pair_prompt(topic: str, intent: str, mood: str, per_request: int) -> list[dict]:
    return [
        {"role": "system", "content": STYLE},
        {
            "role": "user",
            "content": (
                f"話題「{topic}」について、独立したLINEのやりとりを{per_request}通り作ってください。\n"
                f"相手は{intent}。ギャルは{mood}という設定です。\n"
                "\n"
                "書き方:\n"
                f"- 1行に1往復。ぜんぶで{per_request}行\n"
                "- 相手の発言とギャルの返信を、全角の縦棒 ｜ ひとつで区切る\n"
                "- 各行は独立した別のやりとり。前の行の続きにしない\n"
                "- 相手の発言はギャル語にしない。ふつうの話し方で\n"
                "\n"
                "書かないこと:\n"
                "- 行番号、箇条書きの記号\n"
                "- 「ユーザー」「ギャル」などの話者名\n"
                "- かぎかっこ、前置き、説明"
            ),
        },
    ]


# --- calibrate: バッチサイズを決める ----------------------------------------


def calibrate(backend, per_request: int, start: int, limit: int, guard) -> int:
    """「動いた」ではなく「共有GPUメモリが 0 のまま動いた」最大値を探す.

    Windows は VRAM を使い切ってもエラーを出さない。システムRAM へこぼして
    実行を続け、10倍前後遅くなる。だから「動いた」を基準にすると、
    気付かないまま13時間かかる設定を選んでしまう。
    """
    print("バッチサイズを決める (共有GPUメモリが 0 のまま動いた最大値を採る)")
    print(f"{'バッチ':>6} {'秒':>7} {'件/分':>8} {'VRAM':>8} {'共有':>9}  判定")
    best = 0
    batch = start
    while batch <= limit:
        prompts = [
            pair_prompt(f"検証用の話題{i}", INTENTS[i % len(INTENTS)], MOODS[i % len(MOODS)], per_request)
            for i in range(batch)
        ]
        try:
            result = backend.chat_batch(prompts, per_request * 40)
        except Exception as exc:
            print(f"{batch:>6} {'-':>7} {'-':>8} {'-':>8} {'-':>9}  落ちた: {type(exc).__name__}")
            break
        shared = guard.peak_shared_gb
        guard.check(label=f"batch {batch}", force=True)
        rate = batch / result.seconds * 60
        spilled = shared > guard.shared_tolerance_gb
        verdict = "こぼれた" if spilled else "こぼれなし"
        print(
            f"{batch:>6} {result.seconds:>7.1f} {rate:>8.1f}"
            f" {result.peak_vram_gb:>7.1f}G {shared:>8.2f}G  {verdict}"
        )
        if spilled:
            break
        best = batch
        batch *= 2
    if best == 0:
        raise SystemExit("バッチ1すら通りませんでした。モデルを小さくしてください。")
    print(f"\n確定: バッチ {best}")
    return best


# --- topics: 話題出し -------------------------------------------------------


def ask_topics(backend, per_category: int, batch_size: int, guard) -> list[str]:
    prompts = [topic_prompt(category, per_category) for category in CATEGORIES]
    topics: list[str] = []
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start : start + batch_size]
        result = backend.chat_batch(chunk, per_category * 16)
        for text in result.texts:
            for line in text.splitlines():
                match = _TOPIC_LINE_RE.match(line)
                if not match:
                    continue
                topic = match.group(1)
                # 話題として短すぎ・長すぎるものと、見出し行を落とす
                if 2 <= len(topic) <= 14 and "：" not in topic and ":" not in topic:
                    topics.append(topic)
        guard.check(label="topics", force=True)
        done = min(start + batch_size, len(prompts))
        print(f"  {done}/{len(prompts)} ジャンル  累計 {len(topics)} 件", flush=True)

    unique = sorted(set(topics))
    print(f"  重複を除いて {len(unique)} 話題")
    return unique


# --- pairs: 会話生成 --------------------------------------------------------


def load_done_combos() -> set[tuple[str, str, str]]:
    """すでに生成し終えた組み合わせを、生の出力から読み直す."""
    if not RAW_PATH.exists():
        return set()
    done = set()
    for line in RAW_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            # 落ちた瞬間に書きかけの行が残ることがある。読めない行は捨てる。
            continue
        done.add((obj["topic"], obj["intent"], obj["mood"]))
    return done


def count_raw() -> int:
    if not RAW_PATH.exists():
        return 0
    return sum(1 for line in RAW_PATH.read_text(encoding="utf-8").splitlines() if line.strip())


def ask_pairs(
    backend, topics: list[str], target: int, per_request: int,
    batch_size: int, rng: random.Random, guard,
) -> dict:
    done = load_done_combos()
    combos = [c for c in product(topics, INTENTS, MOODS) if c not in done]
    rng.shuffle(combos)
    print(f"  組み合わせ {len(combos)} 件が未処理 (済み {len(done)} 件)")

    collected = count_raw()
    already = collected
    print(f"  生の出力はすでに {collected} 件ある")
    started = time.time()
    think_leaks = 0
    tokens = 0

    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(combos), batch_size):
        if collected >= target:
            print("  目標に達したので打ち切ります")
            break
        chunk = combos[start : start + batch_size]
        prompts = [pair_prompt(t, i, m, per_request) for t, i, m in chunk]
        result = backend.chat_batch(prompts, per_request * 40)
        tokens += result.completion_tokens

        # バッチが終わった時点で必ず書き出す。ここで落ちても失うのは1バッチだけ。
        # newline="\n" を明示する。Windows の既定は \r\n で、\r が混ざると
        # サブワードの分割が変わってしまう。
        with RAW_PATH.open("a", encoding="utf-8", newline="\n") as f:
            for (topic, intent, mood), text in zip(chunk, result.texts, strict=True):
                if "<think>" in text or "</think>" in text:
                    think_leaks += 1
                for user, assistant in parse_pairs(text):
                    record = {
                        "topic": topic, "intent": intent, "mood": mood,
                        "user": user, "assistant": assistant,
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    collected += 1

        guard.check(label=f"batch {start // batch_size}", force=True)
        elapsed = time.time() - started
        print(
            f"  {min(start + batch_size, len(combos))}/{len(combos)} 組  "
            f"生 {collected} 件  {elapsed / 60:.1f}分  "
            f"{(collected - already) / elapsed * 60:.0f} 件/分  "
            f"VRAM {guard.peak_gb:.1f}GB / 共有 +{guard.peak_shared_gb:.2f}GB",
            flush=True,
        )

    elapsed = time.time() - started
    return {
        "raw_pairs": collected,
        "new_pairs": collected - already,
        "minutes": round(elapsed / 60, 2),
        "pairs_per_min": round((collected - already) / elapsed * 60, 1) if elapsed else 0,
        "completion_tokens": tokens,
        "tokens_per_sec": round(tokens / elapsed, 1) if elapsed else 0,
        "think_leaks": think_leaks,
        "batch_size": batch_size,
        "peak_vram_gb": round(guard.peak_gb, 2),
        "peak_shared_gb": round(guard.peak_shared_gb, 3),
    }


# --- build: 検査してまとめる -------------------------------------------------


def build(out_path: Path, rng: random.Random, preview: int, extra: dict | None = None) -> None:
    if not RAW_PATH.exists():
        raise SystemExit(f"{RAW_PATH} がありません。先に --stage pairs を回してください。")

    raw: list[tuple[str, str]] = []
    topics_seen: dict[str, int] = {}
    for line in RAW_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        raw.append((obj["user"], obj["assistant"]))
        topics_seen[obj["topic"]] = topics_seen.get(obj["topic"], 0) + 1
    print(f"生のまま {len(raw)} 件 / 話題 {len(topics_seen)} 種類\n")

    print("検査")
    pairs, rejections = filter_pairs(raw)
    rejections.show(len(pairs))
    print()
    stats = report(pairs)

    if preview:
        print("\n見本")
        for user, assistant in pairs[:preview]:
            print(f"  user      : {user}")
            print(f"  assistant : {assistant}")
        return

    rng.shuffle(pairs)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for user, assistant in pairs:
            f.write(json.dumps({"user": user, "assistant": assistant}, ensure_ascii=False) + "\n")
    print(f"\n書き出し: {out_path}")

    meta = {
        "raw_pairs": len(raw),
        "kept": len(pairs),
        "reject_rate": round(rejections.rate(len(pairs)), 4),
        "rejections": dict(rejections.counts),
        "topics": len(topics_seen),
        **stats,
    }
    if extra:
        meta.update(extra)
    META_OUT.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"記録: {META_OUT}")


# --- 入り口 -----------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--quant", default=DEFAULT_QUANT, choices=("awq", "nf4", "none", "gguf"))
    ap.add_argument(
        "--stage", choices=("calibrate", "topics", "pairs", "build", "all"), default="all"
    )
    ap.add_argument("--target", type=int, default=100000, help="生の出力を何件集めたら止めるか")
    ap.add_argument("--topics-per-category", type=int, default=30)
    ap.add_argument("--pairs-per-request", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=8, help="同時に走らせる生成数")
    ap.add_argument("--calibrate-limit", type=int, default=64)
    ap.add_argument("--memory-fraction", type=float, default=0.90)
    ap.add_argument("--temp", type=float, default=0.7, help="non-thinking 側の推奨値")
    ap.add_argument("--top-p", type=float, default=0.8, help="non-thinking 側の推奨値")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--rng-seed", type=int, default=0)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--preview", type=int, default=0, help="書き出さずに見本を出す件数")
    args = ap.parse_args()

    rng = random.Random(args.rng_seed)

    # build だけならモデルを読まない。検査のやり直しが速い。
    if args.stage == "build":
        build(Path(args.out), rng, args.preview)
        return

    print("環境の確認")
    print(f"  {runtime.device_summary()}")
    holders = runtime.vram_holders()
    if holders:
        print("  警告: VRAM を掴んでいるプロセスがあります")
        for pid, name, mib in holders:
            print(f"    pid {pid} {name} {mib} MiB")
    shared = runtime.shared_memory_gb()
    if shared is None:
        print(f"  共有GPUメモリ: 取得できない ({runtime.shared_memory_error()})")
    else:
        print(f"  共有GPUメモリ: {shared:.2f} GB")
    print()

    sampling = Sampling(
        temperature=args.temp, top_p=args.top_p, top_k=args.top_k, seed=args.rng_seed
    )
    print(f"モデルを読み込みます: {args.model} ({args.quant})")
    print(f"  サンプリング: {sampling.describe()}")
    backend = load_backend(
        args.quant, args.model, sampling=sampling, max_memory_fraction=args.memory_fraction
    )
    print(f"  {backend.detail}")
    # こぼれたら例外で止める。遅くなってから気付くのでは13時間を失う。
    guard = runtime.MemoryGuard(
        runtime.device_summary().total_vram_gb, limit_fraction=args.memory_fraction
    )
    print()

    try:
        batch_size = args.batch_size
        if args.stage == "calibrate":
            calibrate(backend, args.pairs_per_request, args.batch_size, args.calibrate_limit, guard)
            return

        if args.stage in ("topics", "all"):
            print("1段目: 話題出し")
            topics = ask_topics(backend, args.topics_per_category, batch_size, guard)
            TOPICS_PATH.write_text(
                "\n".join(topics) + "\n", encoding="utf-8", newline="\n"
            )
            print(f"  書き出し: {TOPICS_PATH}\n")
            if args.stage == "topics":
                return
        else:
            topics = [
                t for t in TOPICS_PATH.read_text(encoding="utf-8").splitlines() if t.strip()
            ]
            print(f"話題を読み込みました: {len(topics)} 件\n")

        print("2段目: 会話生成")
        stats = ask_pairs(
            backend, topics, args.target, args.pairs_per_request, batch_size, rng, guard
        )
        stats["model"] = args.model
        stats["quant"] = args.quant
        stats["sampling"] = sampling.describe()
        print(f"\n  {guard.report()}")

        if args.stage == "all":
            print("\n3段目: 検査してまとめる")
            build(Path(args.out), rng, args.preview, extra=stats)
        else:
            (ROOT / "runs" / "datagen_stats.json").write_text(
                json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
                newline="\n",
            )
    finally:
        backend.close()
        runtime.release()


if __name__ == "__main__":
    main()

"""追加学習したモデルの返答を、口調と崩れの2軸で数値にする.

Mac 版はここを目で見て判断していた。「崩れた」「崩れなかった」の境界を
測るには、目視では足りない。2つに分けて数える。

  口調   ギャルらしさ。一人称・語尾・敬語の有無で見る (規則で判定できる)
  崩れ   日本語として成り立っているか

崩れの測り方が問題になる。「あ、そっかりでしょw」のような文は、文字の並びは
自然なのに意味が通らない。規則では捕まらない。

そこで **事前学習済みモデル (追加学習する前の 2LM) を物差しに使う**。
きれいな日本語で学習したモデルにとって、崩れた文は「ありそうにない」ので
1文字あたりのビット数が上がる。

ただしギャル語そのものも 2LM から見れば「ありそうにない」ので、口調を変えた
だけでビット数は上がる。そこで **学習データ自身のギャル語の値を基準線** に置く。
基準線からどれだけ離れたかが崩れの量になる。

    python tools/reply_metrics.py --ckpt checkpoints/gal
    python tools/reply_metrics.py --ckpt checkpoints/gal --json runs/gal_metrics.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.generate import build_chat_prompt, generate_stream, load_bundle  # noqa: E402

GAL_JSONL = ROOT / "data" / "raw" / "gal_line.jsonl"
QUESTIONS = ROOT / "eval" / "questions.jsonl"

# 口調の判定。ギャル側に寄っているか、元の機械翻訳調に戻っているかを見る。
_FIRST_PERSON = re.compile(r"うち")
_CASUAL_TAIL = re.compile(r"(じゃん|だよね|だよ|かな|よね|でしょ|っしょ|ね〜|〜|w|ww|マジ|めっちゃ|ヤバ)")
_POLITE = re.compile(r"(です|ます|ました|ません|でしょう|ください|ございます)")

# 元の 2LM が持っていた癖。公開データ側の文体に戻ったかどうかの目印。
_ASSISTANT_ISH = re.compile(r"(オープンアシスタント|私は|申し訳|お手伝い|いかがでしょうか)")

# 同じ文字が4つ以上続く。「そうそうそうそう」のような手前で止まれない状態。
_RUN_RE = re.compile(r"(.)\1{3,}")


def loop_score(text: str, width: int = 4) -> bool:
    """同じ言い回しを繰り返して抜け出せなくなっているか.

    口調とは無関係に測れる。ギャル語でも標準語でも、同じ4文字が
    3回以上出てくる返答は文として成立していない。
    """
    body = text.strip()
    if _RUN_RE.search(body):
        return True
    if len(body) < width * 3:
        return False
    seen: dict[str, int] = {}
    for i in range(len(body) - width + 1):
        gram = body[i : i + width]
        seen[gram] = seen.get(gram, 0) + 1
        if seen[gram] >= 3:
            return True
    return False


@torch.no_grad()
def reply_bits_per_char(model, tokenizer, replies: list[str]) -> float:
    """返答そのものを、渡したモデルで採点して 1文字あたりのビット数にする.

    eval/run.py の bits_per_char は 256トークンの窓で切るので、
    数十文字の返答には使えない。こちらは1本ずつ、返答の部分だけを採点する。
    """
    device = next(model.parameters()).device
    total_bits, total_chars = 0.0, 0
    for reply in replies:
        text = reply.strip()
        if len(text) < 4:
            continue
        # 学習時と同じ並びで包む。文脈が違うと採点の水準が変わる。
        prefix = tokenizer.encode("<|assistant|>")
        body = tokenizer.encode(text + "<|end|>")
        ids = (prefix + body)[: model.cfg.block_size]
        if len(ids) < len(prefix) + 2:
            continue
        arr = torch.tensor([ids], dtype=torch.long, device=device)
        logits = model(arr[:, :-1])
        # 採点するのは返答の部分だけ。マーカーの予測しやすさを混ぜない。
        start = len(prefix) - 1
        loss = F.cross_entropy(
            logits[0, start:].float(),
            arr[0, start + 1 :],
            reduction="sum",
        )
        total_bits += float(loss) / math.log(2)
        total_chars += len(text)
    return total_bits / total_chars if total_chars else float("nan")


def breakdown_scores(replies: list[str], truncated: list[bool]) -> dict[str, float]:
    """口調に依存しない崩れの指標.

    excess_bits は「学習データ自身との距離」なので、口調を変えただけでも動く。
    こちらは口調と無関係に、文として終われているかだけを見る。

      cut_rate  : <|end|> を出せずに上限まで書き続けた割合
      loop_rate : 同じ言い回しを繰り返して抜け出せなかった割合
    """
    if not replies:
        return {}
    return {
        "cut_rate": round(sum(truncated) / len(replies), 3),
        "loop_rate": round(sum(loop_score(r) for r in replies) / len(replies), 3),
    }


def style_scores(replies: list[str]) -> dict[str, float]:
    usable = [r.strip() for r in replies if r.strip()]
    if not usable:
        return {}
    return {
        "gal_rate": round(
            sum(bool(_FIRST_PERSON.search(r) or _CASUAL_TAIL.search(r)) for r in usable)
            / len(usable),
            3,
        ),
        "first_person_rate": round(
            sum(bool(_FIRST_PERSON.search(r)) for r in usable) / len(usable), 3
        ),
        "polite_rate": round(sum(bool(_POLITE.search(r)) for r in usable) / len(usable), 3),
        "assistant_ish_rate": round(
            sum(bool(_ASSISTANT_ISH.search(r)) for r in usable) / len(usable), 3
        ),
        "avg_len": round(sum(len(r) for r in usable) / len(usable), 1),
    }


def load_prompts(limit: int | None = None) -> list[str]:
    """評価に使う問いかけ。公開データ由来の固定設問をそのまま使う."""
    prompts = []
    for line in QUESTIONS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        q = json.loads(line)
        prompts.append(q["turns"][0])
    return prompts[:limit] if limit else prompts


def generate_replies(
    model,
    tokenizer,
    prompts: list[str],
    seed: int,
    max_new_tokens: int,
    repeats: int = 1,
    **sampling,
) -> tuple[list[str], list[bool]]:
    """設問ごとに repeats 本ずつ生成する.

    設問は20問しかない。1本ずつだと割合の刻みが 0.05 になり、
    条件どうしの差が見分けられない。シードを振って標本を増やす。
    種は (設問, 回) から決めるので、何度走らせても同じ結果になる。

    返り値の2つ目は「上限まで書き続けて打ち切られたか」。
    停止マーカーを出せない状態は、口調と関係のない崩れとして数える。
    """
    replies, truncated = [], []
    for round_index in range(repeats):
        for index, prompt in enumerate(prompts):
            step = seed + round_index * 10_000 + index
            torch.manual_seed(step)
            torch.cuda.manual_seed_all(step)
            ids = build_chat_prompt(tokenizer, [], prompt, model.cfg.block_size)
            out = list(
                generate_stream(
                    model,
                    ids,
                    stop_ids=(tokenizer.end_id, tokenizer.user_id),
                    max_new_tokens=max_new_tokens,
                    **sampling,
                )
            )
            replies.append(tokenizer.decode(out))
            truncated.append(len(out) >= max_new_tokens)
    return replies, truncated


def gal_baseline(reference_model, tokenizer, limit: int = 200) -> float:
    """学習データのギャル語自身を、事前学習済みモデルで採点した値.

    これが崩れの基準線になる。ここより悪ければ、口調のぶんでは説明できない。
    """
    if not GAL_JSONL.exists():
        return float("nan")
    replies = []
    for line in GAL_JSONL.read_text(encoding="utf-8").splitlines()[:limit]:
        if line.strip():
            replies.append(json.loads(line)["assistant"])
    return reply_bits_per_char(reference_model, tokenizer, replies)


def measure(
    ckpt: str,
    reference: str = "checkpoints/final",
    seed: int = 777,
    temperature: float = 0.8,
    top_k: int = 40,
    repetition_penalty: float = 1.15,
    max_new_tokens: int = 120,
    limit: int | None = None,
    repeats: int = 5,
) -> dict:
    model, tokenizer = load_bundle(ckpt)
    prompts = load_prompts(limit)
    replies, truncated = generate_replies(
        model,
        tokenizer,
        prompts,
        seed,
        max_new_tokens,
        repeats=repeats,
        temperature=temperature,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
    )
    del model
    torch.cuda.empty_cache()

    reference_model, reference_tokenizer = load_bundle(reference)
    fluency = reply_bits_per_char(reference_model, reference_tokenizer, replies)
    baseline = gal_baseline(reference_model, reference_tokenizer)
    del reference_model
    torch.cuda.empty_cache()

    result = {
        "ckpt": ckpt,
        "reference": reference,
        "samples": len(replies),
        "fluency_bits": round(fluency, 3),
        "gal_data_bits": round(baseline, 3),
        "excess_bits": round(fluency - baseline, 3),
        **breakdown_scores(replies, truncated),
        **style_scores(replies),
        "replies": replies,
    }
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/gal")
    ap.add_argument("--reference", default="checkpoints/final")
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--limit", type=int, default=None, help="設問数を絞る")
    ap.add_argument("--repeats", type=int, default=5, help="設問ごとに何本生成するか")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    result = measure(
        args.ckpt,
        args.reference,
        seed=args.seed,
        limit=args.limit,
        repeats=args.repeats,
    )
    print("=" * 70)
    print(f"  {args.ckpt}  (標本 {result['samples']} 本)")
    print("=" * 70)
    print(f"  ギャル度       : {result.get('gal_rate')}")
    print(f"  一人称「うち」 : {result.get('first_person_rate')}")
    print(f"  敬語の割合     : {result.get('polite_rate')}")
    print(f"  元の文体の癖   : {result.get('assistant_ish_rate')}")
    print(f"  平均返答長     : {result.get('avg_len')}")
    print(f"  打ち切られた率 : {result.get('cut_rate')} (停止マーカーを出せなかった)")
    print(f"  繰り返した率   : {result.get('loop_rate')}")
    print(f"  返答の bits/char (事前学習モデルで採点): {result['fluency_bits']}")
    print(f"  学習データ自身の値 (基準線)            : {result['gal_data_bits']}")
    print(f"  基準線からの超過 = 崩れの量            : {result['excess_bits']}")
    print()
    for reply in result["replies"][:8]:
        print(f"  -> {reply}")

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print(f"\n保存: {out}")


if __name__ == "__main__":
    main()

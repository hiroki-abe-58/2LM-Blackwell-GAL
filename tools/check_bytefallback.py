"""byte_fallback のストリーミングで文字化けが起きないかを確かめる.

サブワード語彙 8,000 では絵文字や珍しい漢字が語彙に無く、SentencePiece の
byte_fallback が 1 バイト = 1 トークンに分解する。ストリーミングで
「トークンが来たそばから decode する」実装だと、この途中バイトが
U+FFFD (置換文字) として画面に出てしまう。

src.generate.decode_incrementally がそれを防げているかを、
語彙に無い文字を含む文字列で往復させて確認する。

    python tools/check_bytefallback.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.generate import decode_incrementally  # noqa: E402
from src.tokenizer import load_tokenizer  # noqa: E402

SAMPLES = [
    "こんにちは",  # 語彙内。素直に流れるはず
    "🐱が3匹",  # 絵文字は byte_fallback で4トークンに割れる
    "𠮟る",  # サロゲートペアの漢字
    "café ☕ で ¥1,200",  # ラテン拡張と通貨記号の混在
]


def main() -> int:
    tokenizer = load_tokenizer(Path("checkpoints/final"))
    print(f"tokenizer: {type(tokenizer).__name__} / vocab {tokenizer.vocab_size}")
    print()

    failed = 0
    for text in SAMPLES:
        ids = tokenizer.encode(text)
        pieces = list(decode_incrementally(tokenizer, iter(ids)))
        joined = "".join(pieces)

        # 1トークンずつ素朴に復号した場合（壊れる方の実装）と並べる
        naive = "".join(tokenizer.decode([i]) for i in ids)

        ok = joined == text and "\ufffd" not in joined
        failed += 0 if ok else 1
        mark = "OK  " if ok else "NG  "
        print(f"{mark}{text!r}")
        print(f"      トークン数 {len(ids)} / 分割 {len(pieces)} 回で描画")
        print(f"      バッファあり: {joined!r}")
        print(f"      バッファなし: {naive!r}")
        if "\ufffd" in naive:
            print("      → バッファなしだと U+FFFD が出る（この対策が必要な理由）")
        print()

    if failed:
        print(f"NG: {failed} 件で復元できなかった")
        return 1
    print("すべて元の文字列に復元できた（U+FFFD なし）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""ターミナルの出力 (ANSI付き) を画像にする (記事用).

実際に走らせた CLI の出力をそのまま流し込んで、記事に貼れる画像にする。
日本語は全角として2セル分進めることで、等幅レイアウトを崩さない。

    python src/chat_cli.py < demo.txt > cli.txt
    python tools/render_terminal.py cli.txt --out docs/images/cli-chat.png

Mac 版からの変更点はフォントの探索だけ。Menlo / ヒラギノは Windows に無いので、
Cascadia Mono と BIZ UDゴシック (どちらも Windows 10 以降に標準で入る) を使う。
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_DIR = Path("C:/Windows/Fonts")
# 等幅。Cascadia Mono は Windows Terminal の既定フォントなので見た目が本物に近い。
MONO_CANDIDATES = ("CascadiaMono.ttf", "consola.ttf", "msgothic.ttc")
# 全角。BIZ UDゴシックは全角が固定ピッチなので、ターミナルの桁が揃う。
JP_CANDIDATES = ("BIZ-UDGothicR.ttc", "YuGothR.ttc", "meiryo.ttc", "msgothic.ttc")

ANSI_RE = re.compile(r"\033\[([0-9;]*)m")
PALETTE = {
    0: (226, 232, 240),
    2: (128, 138, 158),   # dim
    31: (255, 122, 138),
    32: (126, 231, 160),
    33: (255, 214, 122),
    36: (125, 211, 252),
}


def find_font(candidates: tuple[str, ...]) -> str:
    for name in candidates:
        path = FONT_DIR / name
        if path.exists():
            return str(path)
    raise FileNotFoundError(f"フォントが見つかりません: {candidates}")


def parse_ansi(text: str) -> list[list[tuple[str, tuple[int, int, int]]]]:
    """ANSIを解釈して、行ごとの (文字, 色) 列にする."""
    lines: list[list[tuple[str, tuple[int, int, int]]]] = [[]]
    color = PALETTE[0]
    pos = 0
    for match in ANSI_RE.finditer(text):
        for ch in text[pos : match.start()]:
            if ch == "\n":
                lines.append([])
            elif ch != "\r":
                lines[-1].append((ch, color))
        codes = [int(c) for c in match.group(1).split(";") if c != ""] or [0]
        for code in codes:
            color = PALETTE.get(code, color if code not in (0,) else PALETTE[0])
        pos = match.end()
    for ch in text[pos:]:
        if ch == "\n":
            lines.append([])
        elif ch != "\r":
            lines[-1].append((ch, color))
    return lines


Cell = tuple[str, tuple[int, int, int]]


def is_wide(ch: str) -> bool:
    return unicodedata.east_asian_width(ch) in ("W", "F", "A") and ord(ch) > 0x2000


def wrap(lines: list[list[Cell]], cols: int) -> list[list[Cell]]:
    """ターミナルと同じように、桁数を超えた分を次の行へ折り返す."""
    wrapped: list[list[Cell]] = []
    for line in lines:
        current: list[Cell] = []
        width = 0
        for cell in line:
            w = 2 if is_wide(cell[0]) else 1
            if width + w > cols:
                wrapped.append(current)
                current, width = [], 0
            current.append(cell)
            width += w
        wrapped.append(current)
    return wrapped


def gradient_background(width: int, height: int) -> Image.Image:
    top = np.array([16, 23, 54], dtype=np.float32)
    bottom = np.array([21, 15, 44], dtype=np.float32)
    ramp = np.linspace(0, 1, height, dtype=np.float32)[:, None, None]
    data = top[None, None, :] * (1 - ramp) + bottom[None, None, :] * ramp
    return Image.fromarray(np.repeat(data.astype(np.uint8), width, axis=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--out", default="docs/images/cli-chat.png")
    # 区切りは / にしてある。日本語フォントは円記号として \ を描くので、
    # Windows 風に src\chat_cli.py と書くと「src¥chat_cli.py」に見える。
    ap.add_argument("--title", default="2LM-Blackwell — python src/chat_cli.py")
    ap.add_argument("--font-size", type=int, default=15)
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--cols", type=int, default=92)
    ap.add_argument(
        "--lines",
        default=None,
        help="切り出す行範囲。1始まりで両端を含む (例 24:52)。長いログの一部だけ画像にする用",
    )
    args = ap.parse_args()

    s = args.scale
    size = args.font_size * s
    mono_path = find_font(MONO_CANDIDATES)
    jp_path = find_font(JP_CANDIDATES)
    mono = ImageFont.truetype(mono_path, size)
    cell_w = mono.getlength("M")
    # 全角は2セル幅。フォントのemを2セルに合わせておかないと字間が空いて見える。
    jp = ImageFont.truetype(jp_path, int(cell_w * 2 * 0.95))
    baseline_offset = mono.getmetrics()[0]

    line_h = size * 1.62
    pad = 22 * s
    bar_h = 34 * s
    margin = 26 * s

    # PowerShell の Out-File -Encoding utf8 は BOM を付ける (PS 5.1)。
    # utf-8 で読むと先頭に U+FEFF が残り、1行目の頭に見えない文字が入る。
    lines = parse_ansi(Path(args.input).read_text(encoding="utf-8-sig"))
    while lines and not lines[-1]:
        lines.pop()
    if args.lines:
        # 色は行をまたいで続くので、切り出しは ANSI を解釈したあとで行う。
        # 先に文字列を切ると、途中で色指定が失われて全部白くなる。
        first, last = (int(v) for v in args.lines.split(":"))
        lines = lines[first - 1 : last]
    lines = wrap(lines, args.cols)

    win_w = int(cell_w * args.cols + pad * 2)
    win_h = int(bar_h + pad * 2 + line_h * len(lines))
    img = gradient_background(win_w + margin * 2, win_h + margin * 2)
    draw = ImageDraw.Draw(img, "RGBA")

    radius = 14 * s
    draw.rounded_rectangle(
        [margin, margin, margin + win_w, margin + win_h],
        radius=radius,
        fill=(10, 14, 30, 235),
        outline=(255, 255, 255, 46),
        width=s,
    )
    draw.rounded_rectangle(
        [margin, margin, margin + win_w, margin + bar_h + radius],
        radius=radius,
        fill=(255, 255, 255, 14),
    )
    # Windows のウィンドウらしく、右上に最小化/最大化/閉じるを線で描く。
    icon_y = margin + bar_h / 2
    icon_right = margin + win_w - pad
    box = 4.5 * s
    for i, kind in enumerate(("close", "max", "min")):
        cx = icon_right - i * 18 * s
        if kind == "min":
            draw.line([cx - box, icon_y, cx + box, icon_y], fill=(168, 178, 198), width=s)
        elif kind == "max":
            draw.rectangle(
                [cx - box, icon_y - box, cx + box, icon_y + box],
                outline=(168, 178, 198),
                width=s,
            )
        else:
            draw.line([cx - box, icon_y - box, cx + box, icon_y + box],
                      fill=(226, 140, 150), width=s)
            draw.line([cx - box, icon_y + box, cx + box, icon_y - box],
                      fill=(226, 140, 150), width=s)

    title_font = ImageFont.truetype(jp_path, int(size * 0.82))
    draw.text(
        (margin + win_w / 2, margin + bar_h / 2),
        args.title,
        font=title_font,
        fill=(168, 178, 198),
        anchor="mm",
    )

    y = margin + bar_h + pad
    for line in lines:
        x = margin + pad
        for ch, color in line:
            if ch == " ":
                x += cell_w
                continue
            wide = is_wide(ch)
            font = jp if ord(ch) > 0x2000 else mono
            # ベースラインを揃えないと、和欧混在の行で文字が上下にずれる
            draw.text((x, y + baseline_offset), ch, font=font, fill=color, anchor="ls")
            x += cell_w * (2 if wide else 1)
        y += line_h

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    print(f"保存: {out} ({img.width}x{img.height}, {len(lines)}行)")


if __name__ == "__main__":
    main()

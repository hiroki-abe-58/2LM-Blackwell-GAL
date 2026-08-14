"""同値確認. 移植が正しいかを安い順に確かめる.

    python tools/equivalence.py --mac-ckpt E:/ref/2LM-MLX/checkpoints/final

いきなり 3,600 ステップ回して val loss を比べるのは最悪の進め方である。
ずれたときに、アーキテクチャのバグか初期化の違いか学習率スケジュールのずれか、
切り分けられない。安い順に段階を分けてやる。

    段階0  Mac のチェックポイントが strict=True で読めるか (キー名と形)
    段階1  順伝播の logits が独立実装と一致するか
    段階1b Mac の重みで実際に日本語が出るか
    段階1c トークナイザが Mac と同じIDを返すか (今回はこれが土台)
    段階2  初期化直後の loss が ln(vocab_size) の近傍にあるか
    段階3  通しで回して val loss を比べる (src/train.py 側)

段階1c は前作 (1LM) には無かった段。今回は語彙を SentencePiece で持つので、
同じ文章が同じIDに割れることが比較の前提になる。ここがずれると
loss の水準そのものが変わり、val loss を並べても意味がなくなる。

段階1 について。Mac 版は MLX で書かれていて Windows では動かないので、
MLX の logits を直接得ることはできない。代わりに MLX 版のソース仕様から
NumPy で参照実装を独立に書き、それを基準にする。PyTorch の実装とコードを
共有していないので、配線 (行列の向き・LayerNorm の位置・weight tying の順序・
GELU の種類) が違えば必ず差が出る。

浮動小数の丸めと配線の違いを混ぜないため、2つの比較を出す。

    float64 同士   : 配線が同じなら 1e-10 台になる。ここが本命
    float32 との差 : fp32 の丸め幅そのもの。PORTING.md の合格ライン < 1e-4
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from src.model import GPTConfig, MiniGPT  # noqa: E402
from src.tokenizer import (  # noqa: E402
    ASSISTANT,
    END,
    USER,
    Tokenizer,
    load_tokenizer,
)

_ERF = np.vectorize(math.erf, otypes=[np.float64])


# --------------------------------------------------------------------------
# NumPy による独立した参照実装 (MLX 版 src/model.py の仕様から書き起こしたもの)
# --------------------------------------------------------------------------
def _layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + 1e-5) * weight + bias


def _gelu_erf(x: np.ndarray) -> np.ndarray:
    """MLX の nn.gelu は erf 版. tanh 近似ではない."""
    return 0.5 * x * (1.0 + _ERF(x / math.sqrt(2.0)))


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - x.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def reference_forward(
    state: dict[str, np.ndarray], cfg: GPTConfig, idx: np.ndarray
) -> np.ndarray:
    """PyTorch を一切使わない順伝播. dropout は推論なので効かせない."""
    B, T = idx.shape
    n_head, head_dim = cfg.n_head, cfg.n_embd // cfg.n_head
    scale = head_dim**-0.5

    tok = state["tok_emb.weight"]
    x = tok[idx] + state["pos_emb.weight"][:T][None, :, :]

    causal = np.triu(np.full((T, T), -np.inf), k=1)

    for layer in range(cfg.n_layer):
        p = f"blocks.{layer}."
        h = _layer_norm(x, state[p + "ln1.weight"], state[p + "ln1.bias"])

        # MLX の nn.Linear は [out_features, in_features] で持つので、
        # PyTorch と同じく x @ W.T になる。転置は要らない。
        qkv = h @ state[p + "attn.qkv.weight"].T  # bias なし
        q, k, v = np.split(qkv, 3, axis=-1)
        q = q.reshape(B, T, n_head, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, n_head, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, n_head, head_dim).transpose(0, 2, 1, 3)

        att = _softmax(q @ k.transpose(0, 1, 3, 2) * scale + causal)
        out = (att @ v).transpose(0, 2, 1, 3).reshape(B, T, cfg.n_embd)
        x = x + out @ state[p + "attn.proj.weight"].T  # bias なし

        h = _layer_norm(x, state[p + "ln2.weight"], state[p + "ln2.bias"])
        h = _gelu_erf(h @ state[p + "mlp.fc.weight"].T + state[p + "mlp.fc.bias"])
        x = x + h @ state[p + "mlp.proj.weight"].T + state[p + "mlp.proj.bias"]

    x = _layer_norm(x, state["ln_f.weight"], state["ln_f.bias"])
    return x @ tok.T  # weight tying は最終 LayerNorm の後


# --------------------------------------------------------------------------
EXPECTED_SHAPES = {
    "tok_emb.weight": lambda c: (c.vocab_size, c.n_embd),
    "pos_emb.weight": lambda c: (c.block_size, c.n_embd),
    "blocks.0.attn.qkv.weight": lambda c: (3 * c.n_embd, c.n_embd),
    "blocks.0.attn.proj.weight": lambda c: (c.n_embd, c.n_embd),
    "blocks.0.mlp.fc.bias": lambda c: (4 * c.n_embd,),
}


def stage0_load(mac_ckpt: Path) -> tuple[MiniGPT, GPTConfig, dict[str, np.ndarray]]:
    from safetensors.numpy import load_file

    print("[段階0] Mac のチェックポイントを strict=True で読む")
    cfg = GPTConfig.load(mac_ckpt / "config.json")
    print(f"  config      : {cfg}")
    model = MiniGPT.from_pretrained(mac_ckpt)
    model.eval()  # dropout を切る。忘れると一致しない。
    print(f"  テンソル数  : {len(model.state_dict())} 本")
    print(f"  パラメータ  : {model.n_params / 1e6:.2f} M "
          f"(非埋め込み {model.n_params_non_embedding / 1e6:.2f} M)")

    raw = load_file(str(mac_ckpt / "model.safetensors"))
    ok = True
    for key, shape_of in EXPECTED_SHAPES.items():
        want, got = shape_of(cfg), tuple(raw[key].shape)
        mark = "一致" if want == got else "ずれ"
        if want != got:
            ok = False
        print(f"  {key:<28} {str(got):<16} 期待 {str(want):<16} {mark}")
    print(f"  -> {'合格' if ok else '不合格'}: 代表テンソルの形")
    print()
    return model, cfg, {k: v.astype(np.float64) for k, v in raw.items()}


def stage1_logits(
    model: MiniGPT, cfg: GPTConfig, state64: dict[str, np.ndarray], batch: int = 2
) -> bool:
    print("[段階1] 順伝播の logits を独立実装と比べる")
    T = cfg.block_size
    rng = np.random.default_rng(1234)
    idx = rng.integers(0, cfg.vocab_size, size=(batch, T), dtype=np.int64)

    ref = reference_forward(state64, cfg, idx)

    idx_t = torch.from_numpy(idx)
    with torch.no_grad():
        logits32 = model(idx_t).numpy().astype(np.float64)
        logits64 = model.double()(idx_t).numpy()
    model.float()

    diff64 = float(np.abs(logits64 - ref).max())
    diff32 = float(np.abs(logits32 - ref).max())
    scale = float(np.abs(ref).max())

    print(f"  入力            : {idx.shape} のランダムなトークン列 (seed 1234)")
    print(f"  logits の値域   : ±{scale:.3f}")
    print(f"  float64 同士    : max|diff| = {diff64:.3e}  (配線が同じなら 1e-10 台)")
    print(f"  float32 との差  : max|diff| = {diff32:.3e}  (合格ライン < 1e-4)")

    wiring_ok = diff64 < 1e-9
    fp32_ok = diff32 < 1e-4
    print(f"  -> {'合格' if wiring_ok else '不合格'}: 配線 (float64)")
    print(f"  -> {'合格' if fp32_ok else '不合格'}: PORTING.md の 1e-4 基準 (float32)")
    print()
    return wiring_ok and fp32_ok


def stage1b_generate(
    model: MiniGPT, tokenizer: Tokenizer, prompts: tuple[str, ...]
) -> None:
    """Mac の重みで実際に日本語が出るかを見る.

    配線が違えば数値だけでなく出力も壊れる。数値の一致とは別方向の確認になる。
    貪欲デコードなので乱数に依存せず、誰が実行しても同じ文字列が出る。
    """
    print("[段階1b] Mac の重みで貪欲デコードする (temperature 0 なので毎回同じ)")
    model.eval()
    for prompt in prompts:
        ids = tokenizer.encode(f"{USER}{prompt}{ASSISTANT}")
        out: list[int] = []
        for _ in range(60):
            context = torch.tensor([ids[-model.cfg.block_size :]], dtype=torch.long)
            with torch.no_grad():
                next_id = int(model(context)[0, -1].argmax())
            if next_id in (tokenizer.end_id, tokenizer.user_id):
                break
            ids.append(next_id)
            out.append(next_id)
        print(f"  {prompt} -> {tokenizer.decode(out)}")
    print()


def stage1c_tokenizer(tokenizer: Tokenizer, corpus: Path) -> tuple[bool, dict]:
    """流用したトークナイザが期待どおりに動くかを確かめる.

    サブワードで比較するときの前提は3つある。

      1. マーカー (<|user|> など) が1トークンに固定されている
      2. encode -> decode で文章が元に戻る (正規化で「？」が半角に化けない)
      3. byte_fallback が効いていて未知の文字でも UNK にならない

    3番目は語彙に無い絵文字で試す。1文字が複数トークンに割れて、
    それをまとめて復号すると元の文字に戻るのが正しい挙動である。
    """
    print("[段階1c] トークナイザを確かめる (今回の比較はここが土台)")
    print(f"  語彙数        : {tokenizer.vocab_size}")

    marker_ok = True
    for name, text in (("user", USER), ("assistant", ASSISTANT), ("end", END)):
        ids = tokenizer.encode(text)
        mark = "1トークン" if len(ids) == 1 else f"{len(ids)}トークンに分解"
        if len(ids) != 1:
            marker_ok = False
        print(f"  {text:<15} -> id {ids} ({mark}) / {name}_id")

    samples = ("こんにちは。今日は良い天気ですね？", "おすすめの本は？ 3冊お願いします！")
    if corpus.exists():
        # 実データのほうが正規化の事故を拾いやすいので、先頭行も混ぜる。
        with open(corpus, encoding="utf-8") as f:
            samples = (*samples, f.readline().rstrip("\n"))

    roundtrip_ok = True
    total_chars = total_tokens = 0
    for text in samples:
        ids = tokenizer.encode(text)
        back = tokenizer.decode(ids, skip_special=False)
        total_chars += len(text)
        total_tokens += len(ids)
        if back != text:
            roundtrip_ok = False
            print(f"  復元できない  : {text[:30]}... -> {back[:30]}...")
    print(f"  往復          : {len(samples)} 本 / "
          f"{'すべて元に戻った' if roundtrip_ok else '戻らないものがある'}")
    print(f"  圧縮率        : {total_chars / max(total_tokens, 1):.3f} 文字/トークン (標本)")

    emoji = "🐱"
    emoji_ids = tokenizer.encode(emoji)
    emoji_ok = tokenizer.decode(emoji_ids, skip_special=False) == emoji
    print(f"  byte_fallback : {emoji} -> {len(emoji_ids)} トークン -> "
          f"{'復元できた' if emoji_ok else '復元できない (UNK に落ちている)'}")

    ok = marker_ok and roundtrip_ok and emoji_ok
    print(f"  -> {'合格' if ok else '不合格'}: マーカー・往復・byte_fallback")
    print()
    return ok, {
        "vocab_size": tokenizer.vocab_size,
        "markers_single_token": marker_ok,
        "roundtrip": roundtrip_ok,
        "byte_fallback": emoji_ok,
        "emoji_tokens": len(emoji_ids),
    }


def _init_loss(cfg: GPTConfig, seed: int, embed_std: float | None = None) -> tuple[float, float]:
    """初期化直後の loss と logits の標準偏差を測る."""
    torch.manual_seed(seed)
    model = MiniGPT(cfg)
    if embed_std is not None:  # わざと壊した初期化を試すため
        torch.nn.init.normal_(model.tok_emb.weight, mean=0.0, std=embed_std)
        torch.nn.init.normal_(model.pos_emb.weight, mean=0.0, std=embed_std)
    model.eval()
    rng = np.random.default_rng(seed)
    size = (8, cfg.block_size)
    idx = torch.from_numpy(rng.integers(0, cfg.vocab_size, size=size, dtype=np.int64))
    target = torch.from_numpy(rng.integers(0, cfg.vocab_size, size=size, dtype=np.int64))
    with torch.no_grad():
        return float(model.loss(idx, target)), float(model(idx).std())


def stage2_step0_loss(cfg: GPTConfig, seed: int = 1234, trials: int = 3) -> bool:
    """初期化直後の loss を、理論値から導いた予測と比べる.

    PORTING.md は「ln(vocab_size) の近傍 (±0.3 程度)」としているが、
    実測すると系統的に +0.5 ずれる。これは移植のバグではない。

    ln(V) は「logits が全て等しい」ときの値である。ところがこのモデルは
    weight tying をしていて、埋め込みの標準偏差が 1/sqrt(n_embd) なので、
    初期 logits の分散がちょうど 1 になる。

        Var(logits) = n_embd x Var(h) x Var(W) = 384 x 1 x (1/384) = 1

    logits が N(0, s^2) に散っているとき、交差エントロピーの期待値は

        E[loss] = ln(V) + ln E[exp(z)] = ln(V) + s^2 / 2

    となる。s = 1 なので +0.5 である。だから正しく移植できていても
    step 0 の loss は ln(8000) = 8.987 ではなく 9.49 前後になる。
    ln(V) との差だけを見て「初期化が違う」と判断すると、
    正しいコードを何時間も疑うことになる。

    本当の判別は桁である。埋め込みを PyTorch の既定 N(0,1) にすると
    標準偏差が sqrt(n_embd) = 19.6 倍になり、loss は3桁に跳ねる。
    """
    ln_v = math.log(cfg.vocab_size)
    print("[段階2] 初期化直後の loss を理論値と比べる")
    print(f"  ln({cfg.vocab_size}) = {ln_v:.4f}")
    print("  weight tying + 埋め込み std = 1/sqrt(n_embd) なので")
    print("  初期 logits の分散が 1 になり、期待値は ln(V) + std^2/2 になる。")
    print()
    print(f"  {'seed':>6} {'logits std':>11} {'loss':>8} {'予測':>9} {'差':>8}")

    worst = 0.0
    for trial in range(trials):
        loss, std = _init_loss(cfg, seed + trial)
        predicted = ln_v + std * std / 2
        worst = max(worst, abs(loss - predicted))
        print(
            f"  {seed + trial:>6} {std:>11.4f} {loss:>8.4f} {predicted:>9.4f}"
            f" {loss - predicted:>+8.4f}"
        )

    broken_loss, broken_std = _init_loss(cfg, seed, embed_std=1.0)
    print()
    print(f"  ln(V) からの差   : {loss - ln_v:+.4f} (資料の ±0.3 には収まらないが正常)")
    print(f"  壊した初期化の例 : 埋め込みを N(0,1) にすると logits std {broken_std:.1f}"
          f" / loss {broken_loss:.1f}")
    print(f"                     標準偏差が sqrt({cfg.n_embd}) = {cfg.n_embd**0.5:.1f} 倍になるため")

    ok = worst <= 0.1 and abs(loss - ln_v) < 1.0
    print(f"  -> {'合格' if ok else '不合格'}: 予測との最大の差 {worst:.4f} (許容 ±0.1)")
    print()
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mac-ckpt",
        default="E:/ref/2LM-MLX/checkpoints/final",
        help="Mac 版リポジトリの checkpoints/final",
    )
    ap.add_argument("--corpus", default="data/corpus.txt")
    ap.add_argument("--out", default="runs/equivalence.json")
    args = ap.parse_args()

    mac_ckpt = Path(args.mac_ckpt)
    if not (mac_ckpt / "model.safetensors").exists():
        raise SystemExit(
            f"{mac_ckpt} に model.safetensors がありません。\n"
            "git clone https://github.com/hiroki-abe-58/2LM-MLX.git で取得してください。"
        )

    print("=" * 74)
    print("  同値確認 — Mac 版 (MLX) からの移植が正しいか")
    print("=" * 74)
    torch.set_grad_enabled(False)

    model, cfg, state64 = stage0_load(mac_ckpt)
    stage1_ok = stage1_logits(model, cfg, state64)
    tokenizer = load_tokenizer(mac_ckpt)
    stage1b_generate(model, tokenizer, ("こんにちは", "おすすめの本を教えてください"))
    stage1c_ok, tokenizer_report = stage1c_tokenizer(tokenizer, Path(args.corpus))
    stage2_ok = stage2_step0_loss(cfg)

    print("-" * 74)
    passed = stage1_ok and stage1c_ok and stage2_ok
    print(f"段階1  (logits)     : {'合格' if stage1_ok else '不合格'}")
    print(f"段階1c (トークナイザ): {'合格' if stage1c_ok else '不合格'}")
    print(f"段階2  (step 0 loss): {'合格' if stage2_ok else '不合格'}")
    print()
    if passed:
        print("アーキテクチャの移植は終わりです。")
        print("以降のずれは学習側の問題に限定できます。段階3 に進んでください。")
        print("  python src/train.py --equivalence-run")
    else:
        print("先に進まないでください。PORTING.md 第3章と第4章を読み直します。")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "stage1_logits": stage1_ok,
                "stage1c_tokenizer": tokenizer_report,
                "stage2_step0_loss": stage2_ok,
                "vocab_size": cfg.vocab_size,
                "ln_vocab_size": math.log(cfg.vocab_size),
                "step0_loss_note": (
                    "初期 logits の分散が 1 になるため、期待値は ln(V) + 0.5 である。"
                    "ln(V) との差 +0.5 は移植のバグではない。"
                ),
                "n_params": model.n_params,
                "n_params_non_embedding": model.n_params_non_embedding,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
        newline="\n",
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

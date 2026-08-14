"""学習ループ (PyTorch 版).

やることは3行で書ける:
  1. コーパスから連続した文字列をランダムに切り出す
  2. 「1文字ずらした列」を正解として次文字予測の誤差を計算する
  3. 誤差が小さくなる方向にパラメータを動かす

これを数千回繰り返すだけで、モデルは日本語の並び方と会話の書式を覚える。

使い方:
    python src/train.py --equivalence-run    # Mac 版と同じ条件で答え合わせ
    python src/train.py --tokens 60_000_000  # トークン予算で切る
    python src/train.py --resume             # 中断したところから再開
    python src/train.py --vocab-size 0       # 文字レベルで学習する (対照実験用)
    python src/train.py --init-from checkpoints/final   # 追加学習

前作 (1LM-Blackwell) との違いは、トークナイザを差し替えられることだけ。
--tokenizer-model に既存の SentencePiece モデルを渡すと、それをそのまま使う。
語彙を作り直すとIDの対応が変わり、事前学習した重みが全部無意味になるので、
追加学習と同値確認では必ず既存のものを渡す。

Mac 版から意図的に変えた点が4つある。

  時間ではなくトークンで切る
    --minutes は「30分の予算」を意味しない。速いGPUなら同じ30分で3倍学習し、
    遅いGPUなら3分の1しか学習しない。別のモデルが出来上がってしまう。
    比較できる量はトークン数なので、そちらを予算にする。

  グローバルバッチのトークン数を固定する
    VRAM が足りないカードでは micro_bs を下げて grad_accum で回数を増やす。
    バッチだけ小さくすると実効学習率が変わり、これも別のモデルになる。
    どのカードでも 64 x 256 = 16,384 トークンで1回更新する。

  中断して再開できる
    --resume で optimizer の状態・ステップ数・データ取り出しの乱数まで戻す。
    保存は os.replace() で原子的に行う。Windows では書き込み中のディレクトリを
    ウイルス対策ソフトが開いていて上書きに失敗することがある。

  停止点と学習率スケジュールの長さを分ける
    Mac 版は --steps 4300 で起動し、--minutes 35 の時間打ち切りが先に効いて
    3,600 ステップで止まっている。つまり学習率は 4,300 ステップ分の
    cosine を途中まで辿った形になっている。同じ形にしないと再現できないので、
    --steps (止める場所) と --schedule-steps (cosine の分母) を別に持つ。
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import runtime as rt  # noqa: E402
from src.generate import chat_stream  # noqa: E402
from src.model import GPTConfig, MiniGPT  # noqa: E402
from src.optim import MlxAdamW  # noqa: E402
from src.tokenizer import (  # noqa: E402
    CharTokenizer,
    SubwordTokenizer,
    Tokenizer,
    load_tokenizer,
)

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PROMPTS = ("こんにちは", "おすすめの本を教えてください")

# Mac 版 (BASELINES.md 3.1 の「最終」列) の実測値。比較のために表示する。
# 1LM と違って時間打ち切りは効いておらず、3,600 ステップを走り切っている。
MAC_BASELINE = {
    "steps": 3600,
    "seconds": 1620.2,
    "final_train_loss": 3.4205,
    "best_val_loss": 3.6029,
    "best_val_step": 3600,
    "divergence_step": 0,  # val が train を追い越さないまま終わった
    "params_m": 13.81,
    "vocab_size": 8000,
}


def build_dataset(
    corpus: Path,
    cache_dir: Path,
    min_char_freq: int,
    vocab_size: int,
    pretrained: Tokenizer | None = None,
) -> tuple[np.ndarray, Tokenizer]:
    """コーパスをトークンID列 (uint16) に変換してキャッシュする.

    語彙が65536未満なら uint16 で足りる。949万文字を語彙8,000で割ると
    約470万トークン、9.4MB で収まるのでメモリに丸ごと載る。

    vocab_size が 0 なら文字レベル、正の値なら SentencePiece を学習する。
    pretrained を渡した場合は学習済みのトークナイザをそのまま使う。
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    tokens_path = cache_dir / "tokens.npy"
    stamp_path = cache_dir / "stamp.json"
    stamp = {
        "corpus": str(corpus),
        "mtime": corpus.stat().st_mtime,
        "min_char_freq": min_char_freq,
        "vocab_size": vocab_size,
        "pretrained": None
        if pretrained is None
        else f"{type(pretrained).__name__}:{pretrained.vocab_size}",
    }

    cached = tokens_path.exists() and stamp_path.exists()
    if cached and json.loads(stamp_path.read_text(encoding="utf-8")) == stamp:
        return np.load(tokens_path), load_tokenizer(cache_dir)

    text = corpus.read_text(encoding="utf-8")
    if "\r" in text:
        raise SystemExit(
            "コーパスに \\r が混入しています。data/prepare.py の書き出しに"
            ' newline="\\n" を付けてから再生成してください。\n'
            "文字レベルなら \\r が1文字として語彙に入り、サブワードでも"
            "\\r を含むトークンができるので、Mac 版と一致しなくなります。"
        )
    if pretrained is not None:
        tokenizer = pretrained
    elif vocab_size:
        tokenizer = SubwordTokenizer.train(text, vocab_size=vocab_size, model_dir=cache_dir)
    else:
        tokenizer = CharTokenizer.train(text, min_freq=min_char_freq)
    if tokenizer.vocab_size >= 2**16:
        raise SystemExit("語彙が65536を超えました。--vocab-size を下げてください。")
    tokens = np.array(tokenizer.encode(text), dtype=np.uint16)
    np.save(tokens_path, tokens)
    tokenizer.save(cache_dir)
    stamp_path.write_text(json.dumps(stamp), encoding="utf-8", newline="\n")
    return tokens, tokenizer


def lr_at(step: int, lr: float, warmup: int, schedule_steps: int, min_ratio: float) -> float:
    """MLX の join_schedules(linear_schedule, cosine_decay) をそのまま写したもの.

    step は0起点の更新回数。最初の更新では lr x 0.02 から始まる。
    """
    if step < warmup:
        start = lr * 0.02
        return start + step * (lr - start) / warmup
    decay_steps = max(schedule_steps - warmup, 1)
    s = min(step - warmup, decay_steps)
    end = lr * min_ratio
    return end + 0.5 * (1.0 + math.cos(math.pi * s / decay_steps)) * (lr - end)


def save_bundle(path: Path, model: MiniGPT, tokenizer: Tokenizer) -> None:
    """重み・設定・トークナイザを1つのディレクトリにまとめる.

    save_pretrained はディレクトリごと置き換えるので、トークナイザは後に書く。
    順序を逆にすると tokenizer.model が消えて、読み込み側が落ちる。
    """
    model.save_pretrained(path)
    tokenizer.save(path)


def save_resume_state(
    path: Path,
    model: MiniGPT,
    optimizer: torch.optim.Optimizer,
    step: int,
    rng: np.random.Generator,
    best_val: float,
    elapsed: float,
    tokens_done: int,
) -> None:
    """再開に必要なものを原子的に保存する.

    重みだけでは戻れない。AdamW の1次2次モーメントと、データを切り出す
    乱数の状態まで戻さないと、再開したところで別の学習になる。
    """
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "numpy_rng": rng.bit_generator.state,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "best_val": best_val,
            "elapsed": elapsed,
            "tokens_done": tokens_done,
        },
        tmp / "state.pt",
    )
    rt.replace_dir(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(ROOT / "data" / "corpus.txt"))
    ap.add_argument("--out", default=str(ROOT / "checkpoints" / "final"))
    ap.add_argument("--resume-dir", default=str(ROOT / "checkpoints" / "last"))
    ap.add_argument("--log", default=str(ROOT / "runs" / "loss.csv"))
    ap.add_argument(
        "--cache-dir",
        default=str(ROOT / "data" / "cache"),
        help="トークン化した結果の置き場。別のコーパスや語彙を使うときは分ける",
    )
    ap.add_argument(
        "--init-from",
        default="",
        help="このチェックポイントから続けて学習する (追加学習)。"
        "構造と語彙は保存されているものに合わせ、--n-layer などは無視する",
    )

    ap.add_argument("--steps", type=int, default=3600, help="ここで止める")
    ap.add_argument(
        "--schedule-steps",
        type=int,
        default=None,
        help="cosine の分母。省略すると --steps と同じ",
    )
    ap.add_argument(
        "--tokens",
        type=int,
        default=None,
        help="トークン予算。指定するとステップ数より優先する",
    )
    ap.add_argument("--batch-size", type=int, default=64, help="グローバルバッチ (系列数)")
    ap.add_argument(
        "--micro-bs",
        type=int,
        default=None,
        help="1回で GPU に載せる系列数。省略すると VRAM から自動で決める",
    )
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument("--n-layer", type=int, default=6)
    ap.add_argument("--n-head", type=int, default=6)
    ap.add_argument("--n-embd", type=int, default=384)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-interval", type=int, default=250)
    ap.add_argument("--eval-batches", type=int, default=20)
    ap.add_argument("--log-interval", type=int, default=50)
    ap.add_argument("--min-char-freq", type=int, default=1)
    ap.add_argument(
        "--vocab-size",
        type=int,
        default=8000,
        help="SentencePiece の語彙数。0 なら文字レベル (前作 1LM と同じ方式)",
    )
    ap.add_argument(
        "--tokenizer-model",
        default=str(ROOT / "data" / "tokenizer" / "tokenizer.model"),
        help="既存の SentencePiece モデルを使う。空文字にすると学習し直す。"
        "既定は Mac 版 (2LM-MLX) の語彙8,000をそのまま置いたもの",
    )
    ap.add_argument("--seed", type=int, default=1234)

    ap.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    ap.add_argument(
        "--bias-correction",
        action="store_true",
        help="Adam のバイアス補正を有効にする (torch.optim.AdamW と同じ式)。"
        "MLX は既定で無効なので、Mac 版と比べるときは付けない",
    )
    ap.add_argument("--compile", dest="compile", action="store_true", default=True)
    ap.add_argument("--no-compile", dest="compile", action="store_false")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument(
        "--minutes",
        type=float,
        default=0.0,
        help="安全弁。0 なら無効。時間で切ると比較できなくなるので既定は無効",
    )
    ap.add_argument(
        "--equivalence-run",
        action="store_true",
        help="Mac 版と同じ条件に固定する (BASELINES.md 1章と 2.1)",
    )
    ap.add_argument("--no-samples", action="store_true", help="学習中の試し生成を省く")
    args = ap.parse_args()

    if args.equivalence_run:
        # 前作 (1LM) は --minutes の時間打ち切りが先に効いて 3,600 で止まったため
        # cosine の分母が 4,300 だった。こちらは 3,600 を走り切っているので
        # (ログの最終 lr が下限 3e-5 まで落ちている) 分母も 3,600 でよい。
        args.steps, args.schedule_steps = 3600, 3600
        args.tokens = None
        args.batch_size, args.block_size = 64, 256
        args.lr, args.min_lr_ratio, args.warmup = 3e-4, 0.1, 200
        args.weight_decay, args.grad_clip, args.dropout = 0.1, 1.0, 0.1
        args.eval_interval, args.eval_batches, args.seed = 250, 20, 1234
        args.minutes = 0.0
        args.vocab_size = MAC_BASELINE["vocab_size"]
        args.bias_correction = False  # MLX の既定に合わせる
        if not Path(args.tokenizer_model).exists():
            raise SystemExit(
                f"{args.tokenizer_model} がありません。\n"
                "同値確認では Mac 版の tokenizer.model をそのまま使います。"
                "語彙を学習し直すとIDの対応が変わり、val loss を比べられません。"
            )

    schedule_steps = args.schedule_steps or args.steps
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise SystemExit("GPU が見つかりません。python check_env.py で診断してください。")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    rng = np.random.default_rng(args.seed)
    # fp32 を指定したときに TF32 (仮数10bit) へ落とされては fp32 の意味がない。
    # bf16 のときは fp32 行列積がほぼ出てこないので TF32 を許して速さを取る。
    torch.set_float32_matmul_precision(
        "highest" if args.precision == "fp32" else "high"
    )
    rt.configure(0.85)

    # 追加学習では語彙も構造も、事前学習したものに合わせないといけない。
    base = Path(args.init_from) if args.init_from else None
    if base is not None:
        pretrained = load_tokenizer(base)
    elif args.vocab_size and args.tokenizer_model:
        pretrained = SubwordTokenizer(args.tokenizer_model)
    else:
        pretrained = None

    tokens, tokenizer = build_dataset(
        Path(args.corpus),
        Path(args.cache_dir),
        args.min_char_freq,
        args.vocab_size,
        pretrained,
    )
    n_val = min(len(tokens) // 100, 200_000)
    train_data, val_data = tokens[:-n_val], tokens[-n_val:]

    if base is not None:
        cfg = GPTConfig.load(base / "config.json")
        cfg.dropout = args.dropout
        if cfg.vocab_size != tokenizer.vocab_size:
            raise SystemExit(
                f"語彙が合いません (チェックポイント {cfg.vocab_size} / "
                f"トークナイザ {tokenizer.vocab_size})。"
            )
    else:
        cfg = GPTConfig(
            vocab_size=tokenizer.vocab_size,
            block_size=args.block_size,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            dropout=args.dropout,
        )
    model = (
        MiniGPT.from_pretrained(base, device=device)
        if base is not None
        else MiniGPT(cfg).to(device)
    )
    # from_pretrained は保存時の dropout で作るので、上書き分を反映させる。
    if base is not None:
        model.cfg.dropout = cfg.dropout
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = cfg.dropout
        # 文脈長は位置埋め込みの行数そのものなので、こちらも合わせる。
        args.block_size = cfg.block_size

    # micro_bs を決める。満量が載るなら Mac 版と同じ1回で回る。
    if args.micro_bs is None:
        summary = rt.device_summary()
        plan = rt.plan_batch(
            summary.total_vram_gb,
            ctx_len=args.block_size,
            global_batch_tokens=args.batch_size * args.block_size,
        )
        micro_bs, grad_accum = plan.micro_bs, plan.grad_accum
        plan_note = plan.reason
    else:
        micro_bs = args.micro_bs
        if args.batch_size % micro_bs != 0:
            raise SystemExit(
                f"--batch-size {args.batch_size} は --micro-bs {micro_bs} で"
                "割り切れません。グローバルバッチが変わってしまいます。"
            )
        grad_accum = args.batch_size // micro_bs
        plan_note = "手動指定"

    global_batch_tokens = args.batch_size * args.block_size
    if args.tokens is not None:
        args.steps = max(args.tokens // global_batch_tokens, 1)
        schedule_steps = args.schedule_steps or args.steps

    autocast_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32
    use_autocast = args.precision == "bf16"

    summary = rt.device_summary()
    kind = "サブワード (SentencePiece)" if isinstance(tokenizer, SubwordTokenizer) else "文字レベル"
    if pretrained is not None:
        kind += " / 既存の語彙を流用"
    print("=" * 70)
    print(f"  {summary}")
    if base is not None:
        print(f"  続きから学習  : {base}")
    print(f"  トークナイザ  : {kind}")
    print(f"  語彙数        : {cfg.vocab_size}")
    print(f"  学習トークン  : {len(train_data):,} (検証 {len(val_data):,})")
    meta_path = Path(args.corpus).with_suffix(".meta.json")
    if meta_path.exists():
        chars = json.loads(meta_path.read_text(encoding="utf-8"))["chars"]
        print(f"  コーパス      : {chars:,} 文字 / 1トークン {chars / len(tokens):.3f} 文字")
    print(f"  パラメータ数  : {model.n_params / 1e6:.2f} M "
          f"(非埋め込み {model.n_params_non_embedding / 1e6:.2f} M)")
    print(f"  1ステップ     : {args.batch_size} x {args.block_size} = "
          f"{global_batch_tokens:,} トークン (固定)")
    print(f"  内訳          : micro_bs {micro_bs} x grad_accum {grad_accum} — {plan_note}")
    print(f"  予算          : {args.steps} ステップ = "
          f"{args.steps * global_batch_tokens / 1e6:.1f}M トークン")
    print(f"  学習率        : {args.lr} / warmup {args.warmup} / "
          f"cosine の分母 {schedule_steps} / 下限 {args.lr * args.min_lr_ratio:.1e}")
    print(f"  精度          : {args.precision}"
          f"{' (autocast + fp32 マスタ重み)' if use_autocast else ''}")
    print(f"  optimizer     : MlxAdamW / バイアス補正 "
          f"{'あり (torch.optim.AdamW と同じ式)' if args.bias_correction else 'なし (MLX の既定)'}")
    print(f"  torch.compile : {'あり' if args.compile else 'なし'}")
    print("=" * 70)

    # 既定は MLX 互換 (バイアス補正なし)。torch.optim.AdamW と同じ引数でも
    # 更新式が違うので、Mac 版の val loss と比べるならこちらを使う。
    # 詳細は src/optim.py の説明を読むこと。
    optimizer = MlxAdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=args.weight_decay,
        bias_correction=args.bias_correction,
    )

    start_step, best_val, resumed_elapsed, tokens_done = 0, math.inf, 0.0, 0
    resume_dir = Path(args.resume_dir)
    if args.resume:
        state_path = resume_dir / "state.pt"
        if not state_path.exists():
            raise SystemExit(f"{state_path} がありません。--resume を外してください。")
        state = torch.load(state_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_step = state["step"]
        rng.bit_generator.state = state["numpy_rng"]
        # map_location=device で読むと乱数状態まで GPU に載ってしまい、
        # set_rng_state が「RNG state must be a torch.ByteTensor」で落ちる。
        # 乱数状態は CPU の ByteTensor でなければならない。
        torch.set_rng_state(state["torch_rng"].cpu())
        if state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_rng"]])
        best_val = state["best_val"]
        resumed_elapsed = state["elapsed"]
        tokens_done = state["tokens_done"]
        print(f"  再開: step {start_step} / 経過 {resumed_elapsed / 60:.1f} 分 / "
              f"最良 val {best_val:.4f}")
        print("=" * 70)

    # torch.compile(model) の戻り値から .loss を呼んではいけない。
    # OptimizedModule は未知の属性を元のモジュールへ転送するので、
    # model.loss の中の self(idx) が素の forward に行き、コンパイルが効かない。
    # 例外も警告も出ず、ただ速くならないだけなので気付けない。
    # 損失計算そのものをコンパイルする。
    train_loss_fn = (
        torch.compile(model.loss, dynamic=False) if args.compile else model.loss
    )

    def get_batch(data: np.ndarray, generator: np.random.Generator):
        """グローバルバッチぶんの切り出し位置を一度に決める.

        micro_bs の値に関係なく同じ位置を使うので、grad_accum が違っても
        まったく同じデータで学習する。カードが違っても結果が揃う。
        """
        ix = generator.integers(
            0, len(data) - args.block_size - 1, size=args.batch_size
        )
        x = np.stack([data[i : i + args.block_size] for i in ix]).astype(np.int64)
        y = np.stack([data[i + 1 : i + 1 + args.block_size] for i in ix]).astype(np.int64)
        return torch.from_numpy(x), torch.from_numpy(y)

    def evaluate() -> float:
        """毎回同じ検証バッチで測る. 精度は fp32 に固定する.

        検証は比較のための数値なので、autocast の丸めを混ぜない。
        Mac 版は全体が fp32 だったので、そちらに揃えるほうが素直に比べられる。
        """
        model.eval()
        eval_rng = np.random.default_rng(0)
        total = 0.0
        with torch.no_grad():
            for _ in range(args.eval_batches):
                x, y = get_batch(val_data, eval_rng)
                batch_total = 0.0
                for start in range(0, args.batch_size, micro_bs):
                    xb = x[start : start + micro_bs].to(device, non_blocking=True)
                    yb = y[start : start + micro_bs].to(device, non_blocking=True)
                    batch_total += float(model.loss(xb, yb))
                total += batch_total / grad_accum
        model.train()
        return total / args.eval_batches

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not args.resume or not log_path.exists():
        log_path.write_text(
            "step,elapsed_sec,lr,train_loss,val_loss,tok_per_sec,"
            "peak_vram_gb,shared_gb,gpu_temp_c,power_w\n",
            encoding="utf-8",
            newline="\n",
        )

    guard = rt.MemoryGuard(summary.total_vram_gb, raise_on_spill=False)
    model.train()
    window: list[float] = []
    started = time.time()
    steady_started, steady_tokens = None, 0
    # 検証と試し生成の時間はスループットの分母から抜く。
    # 混ぜると「学習が遅くなっていく」ように見えて数値が壊れる。
    overhead = 0.0
    steady_overhead = 0.0
    stop_reason = "ステップ数に到達"
    step = start_step
    # window は log_interval ごとに空にするので、最後のまとめでは使えない。
    # 直近の表示値をそのまま覚えておく。
    last_train_loss: float | None = None

    try:
        for step in range(start_step + 1, args.steps + 1):
            lr = lr_at(step - 1, args.lr, args.warmup, schedule_steps, args.min_lr_ratio)
            for group in optimizer.param_groups:
                group["lr"] = lr

            x, y = get_batch(train_data, rng)
            optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            for start in range(0, args.batch_size, micro_bs):
                xb = x[start : start + micro_bs].to(device, non_blocking=True)
                yb = y[start : start + micro_bs].to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=autocast_dtype, enabled=use_autocast):
                    loss = train_loss_fn(xb, yb)
                # 累積した勾配の合計がグローバルバッチの平均になるように割る。
                (loss / grad_accum).backward()
                # detach しないと「requires_grad=True のテンソルをスカラーにした」と
                # 警告が出る。表示のための値なので勾配は要らない。
                step_loss += float(loss.detach()) / grad_accum

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            window.append(step_loss)
            tokens_done += global_batch_tokens
            elapsed = resumed_elapsed + time.time() - started
            # 毎ステップ見る。共有GPUメモリ側は MemoryGuard が時間で間引くので、
            # ここで呼んでもスループットは落ちない。
            guard.check(f"step {step}")

            # 最初の数ステップは torch.compile のコンパイル時間を含むので、
            # 定常スループットの計算からは外す。ここを混ぜると数値が壊れる。
            if steady_started is None and step - start_step > 10:
                steady_started, steady_tokens = time.time(), tokens_done
                steady_overhead = overhead

            if step % args.log_interval == 0:
                train_loss = last_train_loss = sum(window) / len(window)
                window.clear()
                if steady_started is not None:
                    span = time.time() - steady_started - (overhead - steady_overhead)
                    tps = (tokens_done - steady_tokens) / max(span, 1e-9)
                else:
                    tps = tokens_done / max(elapsed, 1e-9)
                power = rt.power_report()
                print(
                    f"step {step:5d}/{args.steps} | loss {train_loss:.4f} | "
                    f"lr {lr:.2e} | {tps / 1e3:.1f}k tok/s | "
                    f"{elapsed / 60:.1f}分 | "
                    f"VRAM {guard.peak_gb:.1f}GB | "
                    f"{power['temperature.gpu'] or 0:.0f}C "
                    f"{power['power.draw'] or 0:.0f}W",
                    flush=True,
                )
                with log_path.open("a", encoding="utf-8", newline="\n") as f:
                    f.write(
                        f"{step},{elapsed:.1f},{lr:.6f},{train_loss:.4f},,"
                        f"{tps:.0f},{guard.peak_gb:.2f},{guard.peak_shared_gb:.2f},"
                        f"{power['temperature.gpu'] or ''},{power['power.draw'] or ''}\n"
                    )

            if step % args.eval_interval == 0 or step == args.steps:
                overhead_started = time.time()
                val_loss = evaluate()
                marker = ""
                if val_loss < best_val:
                    best_val = val_loss
                    save_bundle(Path(args.out), model, tokenizer)
                    marker = "  <- 保存"
                print(
                    f"  [検証] step {step} val_loss {val_loss:.4f} "
                    f"(最良 {best_val:.4f}){marker}",
                    flush=True,
                )
                with log_path.open("a", encoding="utf-8", newline="\n") as f:
                    f.write(f"{step},{elapsed:.1f},,,{val_loss:.4f},,,,,\n")
                save_resume_state(
                    resume_dir, model, optimizer, step, rng, best_val, elapsed, tokens_done
                )

                if not args.no_samples:
                    model.eval()
                    for prompt in SAMPLE_PROMPTS:
                        reply = "".join(
                            chat_stream(
                                model, tokenizer, [], prompt,
                                max_new_tokens=60, temperature=0.8, device=device,
                            )
                        )
                        print(f"  [試し] {prompt} -> {reply}", flush=True)
                    model.train()
                overhead += time.time() - overhead_started

            if args.minutes > 0 and elapsed > args.minutes * 60:
                stop_reason = (
                    f"時間予算 {args.minutes} 分に到達 "
                    "(時間で切ると他のカードと比較できなくなる)"
                )
                break
    except KeyboardInterrupt:
        stop_reason = "Ctrl+C で中断"
        elapsed = resumed_elapsed + time.time() - started
        save_resume_state(
            resume_dir, model, optimizer, step, rng, best_val, elapsed, tokens_done
        )
        print(f"\n  step {step} まで保存しました。--resume で再開できます。")
    except rt.MemoryBudgetError as exc:
        stop_reason = f"メモリ監視で停止: {exc}"
        elapsed = resumed_elapsed + time.time() - started
        save_resume_state(
            resume_dir, model, optimizer, step, rng, best_val, elapsed, tokens_done
        )

    total = resumed_elapsed + time.time() - started
    if guard.raise_on_spill:
        guard.raise_on_spill = False  # 最後の1回で例外を出しても得がない
    guard.check("終了時", force=True)
    print("=" * 70)
    print(f"終了: {stop_reason}")
    print(f"  経過          : {total:.1f} 秒 ({total / 60:.1f} 分)")
    print(f"  消費トークン  : {tokens_done / 1e6:.1f} M")
    if window:
        last_train_loss = sum(window) / len(window)
    if last_train_loss is not None:
        # 移動平均。1ステップ分の値は揺れるので、記事に書くならこちらを使う。
        print(f"  train loss    : {last_train_loss:.4f} "
              f"(直近 {args.log_interval} ステップの平均)")
    print(f"  最良 val loss : {best_val:.4f}")
    print(f"  {guard.report()}")
    print(f"  チェックポイント: {args.out}")
    if args.equivalence_run:
        mac = MAC_BASELINE
        print()
        print("  Mac 版 (M1 Max) との比較")
        print(f"    所要時間     : {total:.1f} 秒 vs {mac['seconds']:.1f} 秒 "
              f"({mac['seconds'] / max(total, 1e-9):.2f} 倍)")
        if last_train_loss is not None:
            print(f"    train loss   : {last_train_loss:.4f} vs "
                  f"{mac['final_train_loss']:.4f} "
                  f"(差 {last_train_loss - mac['final_train_loss']:+.4f})")
        print(f"    最良 val loss: {best_val:.4f} vs {mac['best_val_loss']:.4f} "
              f"(差 {best_val - mac['best_val_loss']:+.4f} / 許容 ±0.05)")
        verdict = "合格" if abs(best_val - mac["best_val_loss"]) <= 0.05 else "不合格"
        print(f"    判定         : {verdict}")
    print("=" * 70)
    rt.release()


if __name__ == "__main__":
    main()

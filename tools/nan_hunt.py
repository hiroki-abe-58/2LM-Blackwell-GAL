"""どの層で NaN が生まれたのかを、前向き計算1回で突き止める.

「生成が device-side assert で落ちる」だけでは原因が分からない。
torch.multinomial は確率の合計が 0 か NaN のときに落ちるので、
どこかで NaN が発生している。それが量子化カーネルなのか、
モデルの別の場所なのかで、打つ手が変わる。

各サブモジュールの出力に fork を掛けて、最初に NaN を出した層を報告する。

    python tools/nan_hunt.py --model tokyotech-llm/Qwen3-Swallow-8B-RL-v0.2-AWQ-INT4
    python tools/nan_hunt.py --model ... --dtype bfloat16
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data" / "gal"))

import torch  # noqa: E402
from backends import _patch_autoawq_imports  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="tokyotech-llm/Qwen3-Swallow-8B-RL-v0.2-AWQ-INT4")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--prompt", default="日本の首都は")
    ap.add_argument("--limit", type=int, default=12, help="報告する層の数")
    args = ap.parse_args()

    _patch_autoawq_imports()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, args.dtype)
    print(f"{args.model} / dtype={args.dtype}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, device_map="cuda:0"
    )
    model.eval()
    print(f"config の dtype  : {model.config.dtype if hasattr(model.config, 'dtype') else '不明'}")
    print(f"実際の重み dtype : {next(model.parameters()).dtype}")
    quant = getattr(model.config, "quantization_config", None)
    print(f"量子化設定       : {quant}")

    bad: list[tuple[str, str]] = []
    magnitudes: dict[str, tuple[float, float]] = {}
    captured: tuple[str, object, object] | None = None

    def watch(name: str):
        def hook(module, inputs, output):
            nonlocal captured
            tensor = output[0] if isinstance(output, tuple) else output
            if not torch.is_tensor(tensor):
                return
            if torch.isnan(tensor).any():
                bad.append((name, "NaN"))
            elif torch.isinf(tensor).any():
                bad.append((name, "Inf"))
            else:
                return
            # 壊れた層は、入力の大きさも一緒に残す。fp16 の上限 65,504 に
            # 対してどれくらいの計算をしていたのかが分かる。
            if inputs and torch.is_tensor(inputs[0]):
                finite = inputs[0][torch.isfinite(inputs[0])]
                magnitudes[name] = (
                    float(finite.abs().max()) if finite.numel() else float("nan"),
                    float(inputs[0].shape[-1]),
                )
                # 最初に壊れた層だけ、入力ごと控えて後で分解する
                if captured is None and hasattr(module, "qweight"):
                    captured = (name, module, inputs[0].detach().clone())

        return hook

    handles = [
        module.register_forward_hook(watch(name))
        for name, module in model.named_modules()
        if name
    ]

    ids = tokenizer(args.prompt, return_tensors="pt").to("cuda:0")
    with torch.inference_mode():
        out = model(**ids)
    for handle in handles:
        handle.remove()

    logits = out.logits
    print()
    print(f"logits: NaN {int(torch.isnan(logits).sum())} / Inf {int(torch.isinf(logits).sum())}")
    if not bad:
        print("NaN も Inf も出ていない。原因は前向き計算の外にある。")
        top = logits[0, -1].float().topk(5)
        print("次トークン候補:", [tokenizer.decode([i]) for i in top.indices.tolist()])
        return 0

    print(f"最初に壊れた層 (先頭 {args.limit} 件):")
    for name, kind in bad[: args.limit]:
        extra = ""
        if name in magnitudes:
            peak, width = magnitudes[name]
            extra = f"  入力の最大 {peak:.0f} / 内積の長さ {width:.0f} (fp16 の上限 65,504)"
        print(f"  {kind:>4}  {name}{extra}")

    if captured:
        name, module, x = captured
        print(f"\n{name} を3通りで計算して突き合わせる (入力 {tuple(x.shape)})")
        for label, value in inspect_paths(module, x).items():
            print(f"  {label:<28} {value}")
    return 1


def inspect_paths(module, x) -> dict[str, str]:
    """壊れた AWQ 層を、経路ごとに計算して比べる.

    どの経路が壊れているのかを分けないと、直し方が決まらない。
      GEMM カーネル   トークン数が 1,024 未満のときに使われる方 (fp16 累積)
      復元 + matmul   1,024 以上のときに使われる方 (cuBLAS が fp32 累積)
      fp32 の参照     復元した重みを fp32 に上げて計算した値
    """
    from awq.modules.triton.gemm import awq_dequantize_triton
    from awq.utils.packing_utils import dequantize_gemm

    results: dict[str, str] = {}

    def describe(tensor) -> str:
        inf = int(torch.isinf(tensor).sum())
        nan = int(torch.isnan(tensor).sum())
        finite = tensor[torch.isfinite(tensor)]
        peak = float(finite.abs().max()) if finite.numel() else float("nan")
        return f"Inf {inf} / NaN {nan} / 有限値の最大 {peak:.1f}"

    flat = x.reshape(-1, x.shape[-1]).half()
    with torch.no_grad():
        from awq.modules.linear import gemm as gemm_module

        try:
            out = gemm_module.awq_gemm_triton(
                flat, module.qweight, module.scales, module.qzeros, split_k_iters=8
            )
            results["GEMM カーネル"] = describe(out)
        except Exception as exc:
            results["GEMM カーネル"] = f"落ちた: {type(exc).__name__}: {exc}"

        weight = awq_dequantize_triton(module.qweight, module.scales, module.qzeros)
        results["復元 + matmul (fp16)"] = describe(torch.matmul(flat, weight))
        results["復元 + matmul (fp32)"] = describe(
            torch.matmul(flat.float(), weight.float())
        )
        reference = dequantize_gemm(
            module.qweight, module.qzeros, module.scales, module.w_bit, module.group_size
        )
        results["復元した重みの最大"] = f"{float(reference.abs().max()):.3f}"
    return results


if __name__ == "__main__":
    raise SystemExit(main())

"""AWQ の Triton 経路が sm_120 で正しい値を返すかを、1層だけで確かめる.

はしご1段目 (transformers + AWQ) は、モデルの読み込みまでは通るのに
生成の1トークン目で CUDA の device-side assert で落ちた。落ちた場所は
torch.multinomial で、これは確率の合計が 0 か NaN のときに出るものである。

つまり原因は「AWQ が読めない」ではなく「読めたが計算結果が壊れている」。
どちらなのかを切り分けないと、記事に書ける結論にならない。

AutoAWQ の WQLinear_GEMM は awq_ext (CUDA カーネル) が無いと Triton 実装に
落ちる。Windows には awq_ext の wheel が無いので必ず Triton 経路になる。
その Triton 経路を、行列1本ぶんだけ切り出して素の PyTorch と突き合わせる。

    python tools/awq_triton_probe.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data" / "gal"))

import torch  # noqa: E402
from backends import _patch_autoawq_imports  # noqa: E402


def main() -> int:
    _patch_autoawq_imports()
    from awq.modules.linear.gemm import (
        TRITON_AVAILABLE,
        WQLinear_GEMM,
        awq_ext,
    )
    from awq.utils.packing_utils import dequantize_gemm

    print(f"awq_ext (CUDA カーネル): {'ある' if awq_ext else 'ない'}")
    print(f"Triton 経路            : {'使える' if TRITON_AVAILABLE else '使えない'}")
    print("triton                 : ", end="")
    import triton

    print(triton.__version__)
    props = torch.cuda.get_device_properties(0)
    print(f"GPU                    : {props.name} / sm_{props.major}{props.minor}")
    print()

    def build_layer(in_features: int, out_features: int, group_size: int = 128, w_bit: int = 4):
        """素の線形層を AWQ 形式に量子化して返す (参照用の復元重みも返す)."""
        linear = torch.nn.Linear(in_features, out_features, bias=False).half().cuda()
        # AWQ の scales / zeros は [入力をgroup_sizeで割った数, 出力次元] で持つ。
        # 4bit の格納域は 0〜15 なので、ゼロ点 8 を中心に ±7 を割り当てる。
        groups = in_features // group_size
        blocks = linear.weight.data.reshape(out_features, groups, group_size)
        scales = (blocks.abs().amax(dim=2) / 7.0).clamp(min=1e-4).t().contiguous().half()
        zeros = torch.full_like(scales, 8.0)
        layer = WQLinear_GEMM.from_linear(
            linear, w_bit, group_size, scales=scales, zeros=zeros
        ).cuda()
        weight = dequantize_gemm(layer.qweight, layer.qzeros, layer.scales, w_bit, group_size)
        return layer, weight

    def compare(label: str, layer, weight, tokens: int) -> bool:
        x = torch.randn(1, tokens, layer.in_features, dtype=torch.float16, device="cuda")
        reference = torch.matmul(x.float(), weight.float())
        got = layer(x)
        broken = int(torch.isnan(got).sum()) + int(torch.isinf(got).sum())
        if broken:
            kind = "NaN" if torch.isnan(got).any() else "Inf"
            print(f"  {label}: {kind} が {broken} / {got.numel()} 要素  -> 壊れている")
            return False
        diff = (got.float() - reference).abs().max().item()
        scale = reference.abs().max().item()
        ok = diff <= max(0.02 * scale, 1e-2)
        print(
            f"  {label}: max|diff| {diff:.3f}"
            f" (参照の最大値 {scale:.1f}) -> {'一致' if ok else '不一致'}"
        )
        return ok

    torch.manual_seed(0)
    failures = 0

    # 長さ 1024 が分かれ目。AutoAWQ は x.shape[0]*x.shape[1] >= 1024 なら
    # 重みを復元してから matmul、それ未満なら GEMM カーネルを直接叩く。
    # 生成ではプリフィルが前者、デコードの1トークンずつが後者にあたる。
    print("小さい層 (512 -> 256)。教科書どおりの大きさなら両方通る")
    layer, weight = build_layer(512, 256)
    failures += not compare("デコード相当 (GEMM カーネル)", layer, weight, 8)
    failures += not compare("プリフィル相当 (復元+matmul)", layer, weight, 2048)

    # 実寸。Qwen3-Swallow-8B の mlp.up_proj は 4096 -> 12288。
    # 4,096 個の積を fp16 で足し込むと fp16 の上限 65,504 を超える。
    print("\n実寸の層 (4096 -> 12288)。8B の mlp.up_proj と同じ大きさ")
    layer, weight = build_layer(4096, 12288)
    decode_ok = compare("デコード相当 (GEMM カーネル)", layer, weight, 8)
    prefill_ok = compare("プリフィル相当 (復元+matmul)", layer, weight, 2048)
    failures += (not decode_ok) + (not prefill_ok)

    print()
    if decode_ok and prefill_ok:
        print("Triton 経路は実寸でも正しい値を返している。")
        return 0
    if prefill_ok and not decode_ok:
        print(
            "復元+matmul は正しいが、GEMM カーネルだけが壊れている。\n"
            "累積を fp16 でやっているのが原因なので、少ないトークン側も\n"
            "復元+matmul に寄せれば使える (backends._patch_awq_fp16_accumulation)。"
        )
        return 1
    print("Triton 経路は使えない。AWQ (はしご1段目) はここで塞がっている。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

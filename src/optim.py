"""MLX 互換の AdamW.

移植で最後まで残った 0.07 のずれの原因がここだった。
**PyTorch の AdamW と MLX の AdamW は、既定のままでは別のアルゴリズムである。**

MLX の Adam は論文のとおりバイアス補正を省いた式を既定にしている。

    mlx.optimizers.Adam(learning_rate, betas=[0.9, 0.999], eps=1e-8,
                        bias_correction=False)   # <- 既定が False

Mac 版のコードは次のように呼んでいて、bias_correction を渡していない。

    optim.AdamW(learning_rate=schedule, weight_decay=args.weight_decay)

つまりバイアス補正なしで学習されている。一方 PyTorch の AdamW は常に
バイアス補正を行い、切る引数が無い。同じ「AdamW」という名前で同じ
ハイパーパラメータを渡しても、更新式が違うので違うモデルができる。

式で並べるとこうなる。どちらも weight decay は p <- p (1 - lr λ) の形で
先に効かせる (decoupled) ので、そこは同じである。

    共通       m <- β1 m + (1-β1) g
               v <- β2 v + (1-β2) g^2

    MLX 既定   p <- p - lr · m / (sqrt(v) + ε)
    PyTorch    p <- p - lr/(1-β1^t) · m / (sqrt(v)/sqrt(1-β2^t) + ε)

差が出るのは m と v がまだ 0 に近い最初のうちである。
1ステップ目で比べると、補正なしの更新幅は補正ありの約3倍になる。
β2 = 0.999 の補正係数が 1 に落ち着くまで数千ステップかかるので、
3,600ステップの学習では最後まで影響が残る。

どちらが良いという話ではない。バイアス補正ありのほうが今は標準だが、
Mac 版の数値と比べたいなら Mac 版と同じ式で回すしかない。
そのために自分で書く。
"""

from __future__ import annotations

import math

import torch


class MlxAdamW(torch.optim.Optimizer):
    """mlx.optimizers.AdamW と同じ更新式.

    bias_correction=True にすると PyTorch の AdamW と同じ式になるので、
    同じ実装のまま両方を比べられる。
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        bias_correction: bool = False,
    ):
        if lr < 0.0:
            raise ValueError(f"lr が負です: {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"betas が範囲外です: {betas}")
        super().__init__(
            params,
            {
                "lr": lr,
                "betas": betas,
                "eps": eps,
                "weight_decay": weight_decay,
                "bias_correction": bias_correction,
            },
        )

    @torch.no_grad()
    def step(self, closure=None):  # noqa: D102
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            bias_correction = group["bias_correction"]

            params, grads, moments, velocities = [], [], [], []
            for param in group["params"]:
                if param.grad is None:
                    continue
                if param.grad.is_sparse:
                    raise RuntimeError("疎な勾配には対応していません")
                state = self.state[param]
                if not state:
                    state["m"] = torch.zeros_like(param)
                    state["v"] = torch.zeros_like(param)
                params.append(param)
                grads.append(param.grad)
                moments.append(state["m"])
                velocities.append(state["v"])

            if not params:
                continue

            # MLX の step は全パラメータで共有する1つのカウンタで、
            # update() 1回につき1つ進む。グループ単位で持てば同じになる。
            group["step"] = group.get("step", 0) + 1
            t = group["step"]

            # decoupled weight decay。Adam の更新より先に効かせる。
            if weight_decay != 0.0:
                torch._foreach_mul_(params, 1.0 - lr * weight_decay)

            torch._foreach_mul_(moments, beta1)
            torch._foreach_add_(moments, grads, alpha=1.0 - beta1)
            torch._foreach_mul_(velocities, beta2)
            torch._foreach_addcmul_(velocities, grads, grads, value=1.0 - beta2)

            denominator = torch._foreach_sqrt(velocities)
            if bias_correction:
                torch._foreach_mul_(denominator, 1.0 / math.sqrt(1.0 - beta2**t))
                torch._foreach_add_(denominator, eps)
                step_size = lr / (1.0 - beta1**t)
            else:
                torch._foreach_add_(denominator, eps)
                step_size = lr

            updates = torch._foreach_div(moments, denominator)
            torch._foreach_add_(params, updates, alpha=-step_size)

        return loss

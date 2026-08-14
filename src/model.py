"""ミニGPT本体 (PyTorch 実装).

Mac 版 (MLX) の src/model.py を移植したもの。構造は1つも変えていない。
やっていることは1つだけ: これまでの文字列から「次の1文字」の確率分布を出す。
会話が成立するのは、この予測を1文字ずつ繰り返しているだけである。

移植で必ず踏む箇所を先に潰してある。「だいたい同じ形に書く」では同値にならない。

  1. 埋め込みの初期化。 MLX の nn.Embedding は N(0, 1/sqrt(dims))、
     PyTorch は N(0, 1)。標準偏差が約19.6倍違う。weight tying をしているので
     初期 logits もそのまま19.6倍になり、step 0 の loss が理論値 ln(V) では
     なく数十になる。明示的に上書きする。
  2. GELU。 MLX の既定は erf 版。approximate="tanh" を付けてはいけない。
  3. bias の非対称。 attention の qkv と proj は bias なし、MLP は bias あり。
     意図的な設計ではなく MLX の nn.Linear の既定が True なだけだが、
     同値確認のあいだは同じく非対称に実装する。
  4. 位置表現は学習可能な絶対位置埋め込み。 RoPE ではない。
  5. weight tying は最終 LayerNorm の後。

nanoGPT からコピーしても合わない。nanoGPT は埋め込みも Linear も N(0, 0.02) で、
さらに残差の投影層にスケール初期化を入れている。Mac 版はどちらもやっていない。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 256  # 一度に見られる文脈の長さ (文字数)
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(asdict(self), indent=2), encoding="utf-8", newline="\n"
        )

    @classmethod
    def load(cls, path: str | Path) -> GPTConfig:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {f: data[f] for f in cls.__dataclass_fields__ if f in data}
        return cls(**known)


class CausalSelfAttention(nn.Module):
    """因果マスク付きの自己注意.

    「未来の文字を見てはいけない」という制約 (causal mask) が言語モデルの心臓部。
    ここを外すとカンニングになり、学習損失は下がるのに生成は破綻する。

    アテンション重みへの dropout は無い (MLX 版が渡していないため)。
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        # (B, T, C) -> (B, n_head, T, head_dim)
        shape = (B, T, self.n_head, self.head_dim)
        q = q.reshape(shape).permute(0, 2, 1, 3)
        k = k.reshape(shape).permute(0, 2, 1, 3)
        v = v.reshape(shape).permute(0, 2, 1, 3)
        # scale の既定は MLX も PyTorch も 1/sqrt(head_dim) なので渡さない。
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.permute(0, 2, 1, 3).reshape(B, T, C)
        return self.drop(self.proj(out))


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # approximate を指定しない = erf 版。MLX の nn.gelu と一致する。
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    """Pre-LN + 残差接続. この形が深くしても学習が壊れにくい."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)  # eps は MLX も PyTorch も 1e-5
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class MiniGPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        # 出力層は埋め込み行列を転用する (weight tying)。
        # 語彙2000×次元384ぶんのパラメータを節約でき、小さいモデルでは効きが良い。
        self._init_embeddings()

    def _init_embeddings(self) -> None:
        """MLX の nn.Embedding の既定 N(0, 1/sqrt(dims)) に合わせる.

        PyTorch の既定は N(0, 1) で、標準偏差が sqrt(384) = 19.6 倍大きい。
        ここを直さないと step 0 の loss が ln(vocab_size) ではなく数十になる。
        nn.Linear の既定は両者一致するので触らない
        (MLX は U(-1/sqrt(fan_in), +1/sqrt(fan_in))、PyTorch の
         kaiming_uniform_(a=sqrt(5)) が同じ式になる)。
        """
        std = self.cfg.n_embd**-0.5
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=std)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=std)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        _, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        # tok_emb.as_linear(...) と同じ。bias なしの線形変換。
        return F.linear(self.ln_f(x), self.tok_emb.weight)

    def loss(self, idx: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = self(idx)
        return F.cross_entropy(
            logits.reshape(-1, self.cfg.vocab_size).float(), targets.reshape(-1)
        )

    @property
    def n_params(self) -> int:
        """埋め込みを含むパラメータ数.

        Chinchilla の N は非埋め込みパラメータなので、サイズ設計にはこの値を
        そのまま入れないこと。非埋め込みぶんは n_params_non_embedding を使う。
        """
        return sum(p.numel() for p in self.parameters())

    @property
    def n_params_non_embedding(self) -> int:
        return self.n_params - self.tok_emb.weight.numel() - self.pos_emb.weight.numel()

    @classmethod
    def from_pretrained(
        cls, ckpt_dir: str | Path, device: str | torch.device = "cpu"
    ) -> MiniGPT:
        """Mac 版が保存した checkpoints/final をそのまま読む.

        MLX の nn.Linear は重みを [out_features, in_features] で持つので、
        PyTorch の nn.Linear.weight と並びが同じである。転置は要らない。
        strict=True で通らないならモジュール名がずれている。
        """
        from safetensors.torch import load_file

        ckpt = Path(ckpt_dir)
        model = cls(GPTConfig.load(ckpt / "config.json"))
        state = load_file(str(ckpt / "model.safetensors"))
        model.load_state_dict(state, strict=True)
        return model.to(device)

    def save_pretrained(self, out_dir: str | Path) -> None:
        """一時ディレクトリに書いてから置き換える.

        Windows ではエディタ・同期ソフト・ウイルス対策がファイルを開いていると
        上書きに失敗する。Mac では成功するので移植時に見落とす。
        置き換え自体も一筋縄ではいかない。runtime.replace_dir を見ること。
        """
        import shutil
        import sys

        from safetensors.torch import save_file

        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from runtime import replace_dir

        out = Path(out_dir)
        tmp = out.with_name(out.name + ".tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        state = {k: v.detach().cpu().contiguous() for k, v in self.state_dict().items()}
        save_file(state, str(tmp / "model.safetensors"))
        self.cfg.save(tmp / "config.json")
        replace_dir(tmp, out)

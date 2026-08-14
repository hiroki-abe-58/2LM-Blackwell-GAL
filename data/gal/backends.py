"""データ生成に使う推論バックエンド (Windows ネイティブ).

vLLM は Windows ネイティブに対応していない。使えるものを上から順に試す。

    1. transformers + AWQ-INT4     量子化済みをそのまま読める。本命
    2. llama-cpp-python + GGUF     ネイティブ実績あり
    3. transformers + bitsandbytes 素の重みを落として NF4 に潰す。確実だが重い

どのはしごに乗ったかは記事の主産物なので、生成ログに必ず残す。
バックエンドが違っても呼び出し側が変わらないように、`chat_batch()` だけを
共通の入口にしてある。

thinking モードについて。Qwen3 系は既定で <think> ブロックを出す。
`enable_thinking=False` をチャットテンプレートに渡さないと、思考過程が
会話文として混入する。/no_think を本文に書いても止まらない。
そしてサンプリング条件は thinking の有無で公式推奨値が違う (DATAGEN 2.2)。
non-thinking 側の 0.7 / 0.8 / 20 を既定にしてある。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class Sampling:
    """non-thinking モードの公式推奨値 (temperature 0.7 / top_p 0.8 / top_k 20).

    greedy は使わない。公式が「性能劣化と無限反復を招く」と明記している。
    """

    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    repetition_penalty: float = 1.0
    seed: int = 1234

    def describe(self) -> str:
        return (
            f"temp {self.temperature} / top_p {self.top_p} / top_k {self.top_k}"
            f" / rep_pen {self.repetition_penalty} / seed {self.seed}"
        )


@dataclass
class BatchResult:
    texts: list[str]
    seconds: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    peak_vram_gb: float = 0.0
    shared_gb: float | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def tokens_per_sec(self) -> float:
        return self.completion_tokens / self.seconds if self.seconds else 0.0


class Backend(Protocol):
    name: str
    detail: str

    def chat_batch(
        self, conversations: list[list[dict]], max_new_tokens: int
    ) -> BatchResult: ...

    def close(self) -> None: ...


def _patch_autoawq_imports() -> str | None:
    """AutoAWQ が transformers 4.51.3 で開発を止めた影響を1つだけ埋める.

    transformers の AWQ 統合は `awq` パッケージから WQLinear_GEMM を import する。
    その import が `awq/__init__.py` を通るので、AWQ の「量子化する側」のコードまで
    読み込まれる。そこが `transformers.activations.PytorchGELUTanh` を参照しているが、
    このクラスは 4.55 で `GELUTanh` に改名されて消えている。

    使うのは推論だけで、量子化する側は1度も呼ばない。名前だけ戻せば読める。
    版を 4.51.3 に落とすより副作用が小さいので、こちらを選んだ。
    """
    from transformers import activations

    if hasattr(activations, "PytorchGELUTanh"):
        return None
    replacement = getattr(activations, "GELUTanh", None)
    if replacement is None:
        return None
    activations.PytorchGELUTanh = replacement
    return "transformers.activations.PytorchGELUTanh を GELUTanh で補った"


def check_awq_checkpoint(repo_id: str) -> str | None:
    """AutoAWQ で読めない AWQ かどうかを、読み込む前に config.json で弾く.

    同じ `quant_method: awq` でも、作った道具が違うと中身の並びが違う。
    tokyotech-llm の AWQ-INT4 は gptqmodel 製で `desc_act: true` が付いている。
    これは入力チャネルを並べ替えてから量子化する方式で、AutoAWQ の GEMM 形式は
    その並べ替え表 (g_idx) を持たない。

    ここが Windows で一番危ないところで、AutoAWQ は**エラーを出さずに読み込む**。
    そして復元した重みが壊れる。実測では重みの最大値が 5,220 (本来 ±1 程度) になり、
    3層目の mlp.up_proj で fp16 の上限 65,504 を超えて Inf、次の層で NaN、
    最後に torch.multinomial の device-side assert で落ちた。
    落ちる場所が本当の原因から3層ぶん離れている。
    """
    import json
    from pathlib import Path

    from huggingface_hub import hf_hub_download

    try:
        config = json.loads(
            Path(hf_hub_download(repo_id, "config.json")).read_text(encoding="utf-8")
        )
    except Exception:
        return None
    quant = config.get("quantization_config") or {}
    if quant.get("quant_method") != "awq":
        return None
    quantizer = (quant.get("meta") or {}).get("quantizer")
    if quant.get("desc_act") or quantizer:
        return (
            f"この AWQ は AutoAWQ では読めない (desc_act={quant.get('desc_act')},"
            f" 作った道具={quantizer})。読み込めてしまうが重みが壊れる。"
            " gptqmodel を使うか、AutoAWQ 製の AWQ を選ぶこと。"
        )
    return None


# --------------------------------------------------------------------------
# 1 と 3: transformers (AWQ / bitsandbytes NF4 / 素の bf16)
# --------------------------------------------------------------------------
class TransformersBackend:
    """transformers で読む。量子化の指定だけが違う3通りをここでまとめる.

    左パディングにするのを忘れないこと。生成は末尾から続けるので、右に詰めると
    パディングの上に書き続けて出力が壊れる。警告は出ない。
    """

    def __init__(
        self,
        repo_id: str,
        quant: str = "awq",
        sampling: Sampling | None = None,
        max_memory_fraction: float = 0.90,
        dtype: str = "float16",
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.repo_id = repo_id
        self.quant = quant
        self.sampling = sampling or Sampling()

        torch.cuda.set_per_process_memory_fraction(max_memory_fraction)
        torch.manual_seed(self.sampling.seed)
        torch.cuda.manual_seed_all(self.sampling.seed)

        self.patches: list[str] = []
        kwargs: dict = {"dtype": getattr(torch, dtype), "device_map": "cuda:0"}
        if quant == "awq":
            incompatible = check_awq_checkpoint(repo_id)
            if incompatible:
                raise RuntimeError(incompatible)
            patched = _patch_autoawq_imports()
            if patched:
                self.patches.append(patched)
        if quant == "nf4":
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        elif quant not in ("awq", "none"):
            raise ValueError(f"知らない量子化: {quant}")

        started = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(repo_id)
        self.model = AutoModelForCausalLM.from_pretrained(repo_id, **kwargs)
        self.model.eval()
        self.load_seconds = time.time() - started

        # 左パディング。ここを間違えると出力が静かに壊れる。
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.name = f"transformers + {quant}"
        weights_gb = self.model.get_memory_footprint() / 1024**3
        self.detail = (
            f"{repo_id} / {quant} / 重み {weights_gb:.1f} GB"
            f" / 読み込み {self.load_seconds:.1f}秒"
        )
        if self.patches:
            self.detail += " / 補った箇所: " + " と ".join(self.patches)
        self.weights_gb = weights_gb

    def _render(self, conversations: list[list[dict]]) -> list[str]:
        return [
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                # これが無いと <think> が出る。True のときは /no_think を
                # 本文に書いても消えない (中身が空になるだけ)。
                enable_thinking=False,
            )
            for messages in conversations
        ]

    def chat_batch(
        self, conversations: list[list[dict]], max_new_tokens: int
    ) -> BatchResult:
        torch = self.torch
        texts = self._render(conversations)
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True).to("cuda:0")
        prompt_len = int(inputs["attention_mask"].sum().item())

        torch.cuda.reset_peak_memory_stats()
        started = time.time()
        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                do_sample=True,
                temperature=self.sampling.temperature,
                top_p=self.sampling.top_p,
                top_k=self.sampling.top_k,
                repetition_penalty=self.sampling.repetition_penalty,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        seconds = time.time() - started

        generated = out[:, inputs["input_ids"].shape[1] :]
        completion = int((generated != self.tokenizer.pad_token_id).sum().item())
        decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        return BatchResult(
            texts=decoded,
            seconds=seconds,
            prompt_tokens=prompt_len,
            completion_tokens=completion,
            peak_vram_gb=torch.cuda.max_memory_allocated() / 1024**3,
        )

    def close(self) -> None:
        del self.model
        self.torch.cuda.synchronize()
        self.torch.cuda.empty_cache()


# --------------------------------------------------------------------------
# 2: llama-cpp-python + GGUF
# --------------------------------------------------------------------------
class LlamaCppBackend:
    """GGUF を llama-cpp-python で読む.

    CUDA を有効にした wheel が要る。素の pip install は CPU だけになるので、
    その場合は「動くが遅い」という別の失敗の仕方をする。ここでも
    「動いた」を基準にしないこと。
    """

    def __init__(
        self,
        model_path: str,
        sampling: Sampling | None = None,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
        template_repo: str | None = None,
    ):
        from llama_cpp import Llama, llama_supports_gpu_offload

        self.sampling = sampling or Sampling()
        self.gpu = bool(llama_supports_gpu_offload())
        started = time.time()
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers if self.gpu else 0,
            seed=self.sampling.seed,
            verbose=False,
        )
        self.load_seconds = time.time() - started

        # GGUF に埋まっているテンプレートを llama.cpp に適用させると、Qwen3 では
        # 思考モードが既定で入り <think> が漏れる。create_chat_completion には
        # enable_thinking を渡す口が無いので、テンプレートだけ HF 側で組んで
        # 素の補完として投げる。transformers 経路と同じ文字列になる。
        self.template = None
        if template_repo:
            from transformers import AutoTokenizer

            self.template = AutoTokenizer.from_pretrained(template_repo)

        self.name = "llama-cpp-python + GGUF"
        offload = "GPU オフロードあり" if self.gpu else "CPU のみ (遅い)"
        applied = (
            f"テンプレートは {template_repo} から (enable_thinking=False)"
            if template_repo
            else "GGUF 内蔵テンプレート"
        )
        self.detail = (
            f"{model_path} / {offload} / 読み込み {self.load_seconds:.1f}秒 / {applied}"
        )

    def chat_batch(
        self, conversations: list[list[dict]], max_new_tokens: int
    ) -> BatchResult:
        # llama.cpp の Python 束縛には真のバッチ API が無い。1本ずつ回す。
        started = time.time()
        texts, completion = [], 0
        for messages in conversations:
            if self.template is None:
                out = self.llm.create_chat_completion(
                    messages=messages,
                    temperature=self.sampling.temperature,
                    top_p=self.sampling.top_p,
                    top_k=self.sampling.top_k,
                    repeat_penalty=self.sampling.repetition_penalty,
                    max_tokens=max_new_tokens,
                )
                texts.append(out["choices"][0]["message"]["content"] or "")
            else:
                prompt = self.template.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                out = self.llm.create_completion(
                    prompt,
                    temperature=self.sampling.temperature,
                    top_p=self.sampling.top_p,
                    top_k=self.sampling.top_k,
                    repeat_penalty=self.sampling.repetition_penalty,
                    max_tokens=max_new_tokens,
                    stop=["<|im_end|>", "<|endoftext|>"],
                )
                texts.append(out["choices"][0]["text"] or "")
            completion += out["usage"]["completion_tokens"]
        notes = ["逐次実行 (バッチAPIが無い)"]
        if self.template is None:
            notes.append("GGUF 内蔵テンプレート。Qwen3 では <think> が入る")
        return BatchResult(
            texts=texts,
            seconds=time.time() - started,
            completion_tokens=completion,
            notes=notes,
        )

    def close(self) -> None:
        self.llm.close()


def load_backend(kind: str, target: str, sampling: Sampling | None = None, **kwargs):
    """kind に応じてバックエンドを組み立てる.

    kind は awq / nf4 / none / gguf のいずれか。target はリポジトリIDか GGUF のパス。
    """
    if kind == "gguf":
        return LlamaCppBackend(target, sampling=sampling, **kwargs)
    return TransformersBackend(target, quant=kind, sampling=sampling, **kwargs)

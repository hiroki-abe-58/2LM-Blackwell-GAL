"""チャットGUI用のAPIサーバ.

    python server.py                       # http://127.0.0.1:8000
    python server.py --open                # ブラウザも開く
    python server.py --ckpt checkpoints/final --port 8000

生成はロックで直列化している。GPU は1枚しかないので、同時に走らせると
VRAM のピークが読めなくなり、共有GPUメモリに溢れて全体が静かに遅くなる。
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import webbrowser
from collections.abc import Iterator
from pathlib import Path

import torch
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.generate import DEFAULTS, chat_stream, load_bundle  # noqa: E402
from src.tokenizer import SubwordTokenizer  # noqa: E402

WEB_DIR = ROOT / "web"

app = FastAPI(title="2LM")
_lock = threading.Lock()
_state: dict = {}


class ChatRequest(BaseModel):
    message: str
    history: list[tuple[str, str]] = Field(default_factory=list)
    temperature: float = DEFAULTS["temperature"]
    top_k: int = DEFAULTS["top_k"]
    repetition_penalty: float = DEFAULTS["repetition_penalty"]
    max_new_tokens: int = DEFAULTS["max_new_tokens"]
    history_turns: int = 2
    seed: int | None = None


@app.get("/api/info")
def info() -> dict:
    model, tokenizer = _state["model"], _state["tokenizer"]
    return {
        "checkpoint": _state["ckpt"],
        "params_m": round(model.n_params / 1e6, 2),
        "tokenizer": "subword" if isinstance(tokenizer, SubwordTokenizer) else "char",
        "vocab_size": tokenizer.vocab_size,
        "block_size": model.cfg.block_size,
        "n_layer": model.cfg.n_layer,
        "n_head": model.cfg.n_head,
        "n_embd": model.cfg.n_embd,
        "defaults": DEFAULTS,
    }


@app.post("/api/chat")
def chat(req: ChatRequest) -> StreamingResponse:
    def stream() -> Iterator[str]:
        model, tokenizer = _state["model"], _state["tokenizer"]
        with _lock:
            if req.seed is not None:
                torch.manual_seed(req.seed)
                torch.cuda.manual_seed_all(req.seed)
            start = time.time()
            n = 0
            for piece in chat_stream(
                model,
                tokenizer,
                req.history[-req.history_turns :] if req.history_turns > 0 else [],
                req.message,
                temperature=req.temperature,
                top_k=req.top_k,
                repetition_penalty=req.repetition_penalty,
                max_new_tokens=req.max_new_tokens,
            ):
                n += len(piece)
                yield "data: " + json.dumps({"t": piece}, ensure_ascii=False) + "\n\n"
            took = time.time() - start
        stats = {"chars": n, "seconds": round(took, 2), "cps": round(n / took, 1) if took else 0}
        yield "data: " + json.dumps({"done": True, "stats": stats}, ensure_ascii=False) + "\n\n"

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/final")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--open", action="store_true", help="既定のブラウザで自動的に開く")
    args = ap.parse_args()

    model, tokenizer = load_bundle(args.ckpt)
    _state.update(model=model, tokenizer=tokenizer, ckpt=args.ckpt)
    device = next(model.parameters()).device
    url = f"http://{args.host}:{args.port}"
    print(f"モデル読み込み完了: {model.n_params/1e6:.2f}M params / {device} -> {url}")

    if args.open:
        # webbrowser はサーバが起動する前に開くので、リロードが1回必要になることがある。
        # uvicorn.run はブロックするため、ここで開くのが一番単純。
        webbrowser.open(url)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

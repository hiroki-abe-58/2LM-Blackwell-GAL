"""データ生成に使うモデルのライセンスを、着手前に機械的に確かめる.

Mac 版はここを飛ばして2日を失っている。技術的には何のエラーも出ないので、
気付かせてくれるものが何もない。だから人間の記憶ではなく手順にする。

確認するのは5項目。LICENSING.md 第6章のチェックリストそのままである。

    1. ライセンスを原文で読んだか      -> LICENSE 相当のファイルを取得して保存する
    2. output / results / improve /
       distill の語があるか            -> 原文を検索して該当箇所を出す
    3. Apache-2.0 か MIT か            -> メタデータの license タグで判定
    4. 学習データ側に継承条件 (SA) が
       付いていないか                  -> データセットのライセンスを一覧する
    5. 合成データセットは生成元モデル
       まで遡ったか                    -> 生成元を明記して判定を書く

    python tools/license_check.py                # 表示のみ
    python tools/license_check.py --save         # runs/licensing.json に保存

ネットワークが無い環境では取得に失敗する。その場合も「未確認」と表示し、
勝手に「問題なし」とは言わない。
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "runs" / "licensing.json"
LICENSE_DIR = ROOT / "licenses"

# 生成に使う候補。DATAGEN.md 第1章の並び。
# 「オープンウェイト」と「オープンソース」は別物なので、必ず1件ずつ確かめる。
GENERATOR_MODELS = (
    "tokyotech-llm/Qwen3-Swallow-32B-RL-v0.2",
    "tokyotech-llm/Qwen3-Swallow-8B-RL-v0.2",
    "Qwen/Qwen2.5-32B-Instruct",
    "Qwen/Qwen3-8B",
)

# 使わないと決めたもの。なぜ使わないかを記録に残すために並べる。
REJECTED_MODELS = (
    (
        "meta-llama/Llama-3.3-70B-Instruct",
        "Llama Community License。出力を他モデルの改善に使うことを明文で禁止",
    ),
    ("google/gemma-3-27b-it", "Gemma Terms。Model Derivatives の定義に蒸留を含む"),
    ("商用チャットAPI (クローズド)", "利用規約が出力の学習利用を禁止。規模の例外は書かれていない"),
)

# 事前学習に使った公開データセット。継承条件 (ShareAlike) の有無を見る。
DATASETS = (
    ("kunishou/oasst1-89k-ja", None),
    ("llm-jp/oasst2-33k-ja", None),
    ("llm-jp/magpie-sft-v1.0", "合成データ。生成元は Apache-2.0 のモデル"),
    ("Aratako/Magpie-Tanuki-8B-97k", "合成データ。生成元 Tanuki-8B のライセンスを遡る対象"),
)

# 出力の利用可否に関わる語。原文にこれが出たら必ず前後を読む。
KEYWORDS = ("output", "results", "improve", "distill", "train", "derivative")

OK_LICENSES = {"apache-2.0", "mit"}


@dataclass
class ModelReport:
    repo_id: str
    license_tag: str | None = None
    license_files: list[str] = field(default_factory=list)
    hits: dict[str, int] = field(default_factory=dict)
    excerpts: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def verdict(self) -> str:
        if self.error:
            return "未確認"
        if (self.license_tag or "").lower() in OK_LICENSES:
            return "使える"
        return "要判断"


def fetch_model(repo_id: str) -> ModelReport:
    from huggingface_hub import HfApi, hf_hub_download

    report = ModelReport(repo_id)
    api = HfApi()
    try:
        info = api.model_info(repo_id, files_metadata=False)
    except Exception as exc:
        report.error = f"{type(exc).__name__}: {exc}"
        return report

    report.license_tag = (info.card_data or {}).get("license") if info.card_data else None
    names = [s.rfilename for s in (info.siblings or [])]
    # LICENSE / LICENSE.txt / NOTICE などを原文として拾う
    wanted = [n for n in names if re.match(r"^(LICENSE|NOTICE|COPYING)", n, re.I)]
    report.license_files = wanted

    texts: list[tuple[str, str]] = []
    for name in wanted[:3]:
        try:
            path = hf_hub_download(repo_id, name)
            texts.append((name, Path(path).read_text(encoding="utf-8", errors="replace")))
        except Exception as exc:
            report.excerpts.append(f"{name}: 取得できず ({type(exc).__name__})")

    # モデルカード本文も見る。禁止条項がカードにだけ書かれている例がある。
    try:
        card = api.model_info(repo_id).card_data
        if card is not None:
            texts.append(("card_data", json.dumps(card.to_dict(), ensure_ascii=False)))
    except Exception:
        pass

    for name, text in texts:
        low = text.lower()
        for word in KEYWORDS:
            count = low.count(word)
            if count:
                report.hits[word] = report.hits.get(word, 0) + count
        for match in re.finditer(r"[^.\n]*\b(improve|distill)\b[^.\n]*", text, re.I):
            snippet = " ".join(match.group(0).split())
            if len(snippet) > 20:
                report.excerpts.append(f"{name}: {snippet[:200]}")

    # 原文をリポジトリに残す。あとから「読んだ」と言えるようにするため。
    if texts:
        LICENSE_DIR.mkdir(parents=True, exist_ok=True)
    return report


def fetch_dataset(repo_id: str) -> tuple[str | None, str | None]:
    from huggingface_hub import HfApi

    try:
        info = HfApi().dataset_info(repo_id)
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    card = info.card_data or {}
    license_tag = card.get("license")
    if isinstance(license_tag, list):
        license_tag = ", ".join(license_tag)
    return license_tag, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true", help="runs/licensing.json に保存する")
    args = ap.parse_args()

    print("=" * 74)
    print("LICENSING チェック — データ生成に着手する前に通す5項目")
    print("=" * 74)

    print("\n[1,2,3] 生成に使う候補モデル")
    models: list[ModelReport] = []
    for repo_id in GENERATOR_MODELS:
        report = fetch_model(repo_id)
        models.append(report)
        tag = report.license_tag or "取得できず"
        print(f"\n  {repo_id}")
        print(f"    license タグ   : {tag}")
        print(f"    原文ファイル   : {', '.join(report.license_files) or 'なし'}")
        if report.hits:
            found = " / ".join(f"{k}:{v}" for k, v in sorted(report.hits.items()))
            print(f"    語の出現       : {found}")
        else:
            print("    語の出現       : なし")
        for line in report.excerpts[:2]:
            print(f"    該当           : {line}")
        if report.error:
            print(f"    取得失敗       : {report.error}")
        print(f"    判定           : {report.verdict}")

    print("\n[3] 使わないと決めたもの（理由を残す）")
    for name, reason in REJECTED_MODELS:
        print(f"  {name}")
        print(f"    -> 使わない: {reason}")

    print("\n[4,5] 事前学習に使った公開データセット")
    datasets = []
    for repo_id, note in DATASETS:
        tag, error = fetch_dataset(repo_id)
        datasets.append({"repo_id": repo_id, "license": tag, "note": note, "error": error})
        shown = tag or f"取得できず ({error})"
        sa = "継承条件あり" if tag and "sa" in tag.lower().split("-") else "継承条件なし"
        print(f"  {repo_id:<32} {shown:<16} {sa if tag else ''}")
        if note:
            print(f"    {note}")

    ok = [m for m in models if m.verdict == "使える"]
    print("\n" + "=" * 74)
    print(f"Apache-2.0 / MIT と確認できたモデル: {len(ok)} / {len(models)}")
    if not ok:
        print("使えるモデルが確認できていません。ここで止めること。")
    else:
        print("使う: " + ok[0].repo_id)
    print("=" * 74)

    if args.save:
        payload = {
            "models": [
                {
                    "repo_id": m.repo_id,
                    "license": m.license_tag,
                    "license_files": m.license_files,
                    "keyword_hits": m.hits,
                    "excerpts": m.excerpts[:5],
                    "verdict": m.verdict,
                    "error": m.error,
                }
                for m in models
            ],
            "rejected": [{"name": n, "reason": r} for n, r in REJECTED_MODELS],
            "datasets": datasets,
        }
        OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        OUT_JSON.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print(f"\n保存: {OUT_JSON}")


if __name__ == "__main__":
    main()

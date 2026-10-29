"""把 AISHELL-3 样本包转成评测清单 (jsonl: wav / text / utt)。

AISHELL-3 标注格式:
  train/label_train-set.txt: uttid|拼音|中文文本%分句$
  test/content.txt:          uttid.wav 字 拼音 字 拼音 ...
"""

import glob
import json
import os
import re

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
SAMPLE = os.path.join(ROOT, "aishell3", "AISHELL-3-Sample")


def clean(text: str) -> str:
    return re.sub(r"[\s%$]", "", text)


def from_label_file(path: str):
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "|" not in line:
            continue
        parts = line.split("|")
        if len(parts) < 3:
            continue
        utt, text = parts[0], clean(parts[-1])
        wav = os.path.join(os.path.dirname(path), utt[:7], utt + ".wav")
        if text and os.path.exists(wav):
            yield {"utt": utt, "wav": wav, "text": text}


def from_content_file(path: str):
    base = os.path.dirname(path)
    for line in open(path, encoding="utf-8"):
        toks = line.strip().split()
        if not toks:
            continue
        name = toks[0]
        # 交替的 字/拼音，取汉字位（非拼音 token）
        chars = [t for t in toks[1:] if not re.fullmatch(r"[a-z]+[1-5]", t)]
        wav = os.path.join(base, name[:7], name)
        if chars and os.path.exists(wav):
            yield {"utt": name[:-4], "wav": wav, "text": "".join(chars)}


def main():
    items = []
    for p in glob.glob(os.path.join(SAMPLE, "train", "label_train-set.txt")):
        items += list(from_label_file(p))
    for p in glob.glob(os.path.join(SAMPLE, "test", "content.txt")):
        items += list(from_content_file(p))
    items.sort(key=lambda x: x["utt"])

    out = os.path.join(ROOT, "manifest_aishell3.jsonl")
    with open(out, "w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
    total_dur = 0.0
    import soundfile as sf

    for it in items:
        total_dur += sf.info(it["wav"]).duration
    print(f"utts={len(items)}  total_audio={total_dur / 60:.1f}min  -> {out}")
    for it in items[:3]:
        print(" ", it["utt"], it["text"])


if __name__ == "__main__":
    main()

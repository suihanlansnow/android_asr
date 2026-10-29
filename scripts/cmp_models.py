"""快速对比：paraformer 流式 / 混合精度 transducer 的 CER + RTF + 首字延迟。

用法:
  python scripts/cmp_models.py --cfg paraformer_i8 --limit 200
"""
import argparse, json, os, statistics, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_cer import load_wav, norm, edit_distance, SAMPLE_RATE, ROOT  # noqa
import sherpa_onnx

MODELS = os.path.join(ROOT, "models")


def build(cfg):
    if cfg.startswith("paraformer"):
        d = os.path.join(MODELS, "sherpa-onnx-streaming-paraformer-bilingual-zh-en")
        s = ".int8.onnx" if cfg.endswith("i8") else ".onnx"
        return sherpa_onnx.OnlineRecognizer.from_paraformer(
            tokens=os.path.join(d, "tokens.txt"),
            encoder=os.path.join(d, "encoder" + s),
            decoder=os.path.join(d, "decoder" + s),
            num_threads=2, sample_rate=SAMPLE_RATE, feature_dim=80,
            decoding_method="greedy_search", enable_endpoint_detection=False)

    if cfg.startswith("zh14m"):
        d = os.path.join(MODELS, "sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23")
        s = ".int8.onnx" if cfg.endswith("i8") else ".onnx"
        stem = "epoch-99-avg-1"
        return sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=os.path.join(d, "tokens.txt"),
            encoder=os.path.join(d, f"encoder-{stem}{s}"),
            decoder=os.path.join(d, f"decoder-{stem}{s}"),
            joiner=os.path.join(d, f"joiner-{stem}{s}"),
            num_threads=2, sample_rate=SAMPLE_RATE, feature_dim=80,
            decoding_method="greedy_search", enable_endpoint_detection=False)

    d = os.path.join(MODELS, "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20")
    stem = "epoch-99-avg-1"
    # 混合精度：encoder int8，decoder/joiner fp32（int8 joiner 曾导致重复循环）
    enc_s = ".int8.onnx" if cfg.startswith("bil_mix") else ".onnx"
    return sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=os.path.join(d, "tokens.txt"),
        encoder=os.path.join(d, f"encoder-{stem}{enc_s}"),
        decoder=os.path.join(d, f"decoder-{stem}.onnx"),
        joiner=os.path.join(d, f"joiner-{stem}.onnx"),
        num_threads=2, sample_rate=SAMPLE_RATE, feature_dim=80,
        decoding_method="greedy_search", enable_endpoint_detection=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--feed-ms", type=int, default=40)
    ap.add_argument("--out", default="")
    ap.add_argument("--manifest", default="")
    a = ap.parse_args()

    man = a.manifest or os.path.join(ROOT, "data", "manifest_aishell3.jsonl")
    items = [json.loads(l) for l in open(man, encoding="utf-8")]
    if a.limit:
        items = items[:a.limit]

    t0 = time.perf_counter()
    rec = build(a.cfg)
    print(f"init {time.perf_counter()-t0:.2f}s", flush=True)

    step = int(SAMPLE_RATE * a.feed_ms / 1000)
    errs = chars = 0
    rtfs, firsts, p95s = [], [], []
    bad = []
    for it in items:
        ref = norm(it["text"])
        s = load_wav(it["wav"])
        st = rec.create_stream()
        dec = 0.0
        aud = 0.0
        first = None
        dts = []
        for pos in range(0, len(s), step):
            b = s[pos:pos + step]
            st.accept_waveform(SAMPLE_RATE, b)
            tt = time.perf_counter()
            while rec.is_ready(st):
                rec.decode_stream(st)
            dt = time.perf_counter() - tt
            dec += dt
            dts.append(dt)
            aud += len(b) / SAMPLE_RATE
            cur = rec.get_result(st).strip()
            if cur and first is None:
                first = aud + dt
        st.input_finished()
        while rec.is_ready(st):
            rec.decode_stream(st)
        hyp = norm(rec.get_result(st).strip())
        e = edit_distance(ref, hyp)
        errs += e
        chars += len(ref)
        rtfs.append(dec / aud)
        p95s.append(sorted(dts)[int(len(dts) * 0.95)])
        if first:
            firsts.append(first * 1000)
        if e / max(1, len(ref)) > 0.5 and len(bad) < 5:
            bad.append((it["utt"], ref, hyp))

    res = {
        "cfg": a.cfg, "n": len(items),
        "cer": round(errs / max(1, chars), 4),
        "rtf": round(statistics.mean(rtfs), 4),
        "chunk_p95_ms": round(statistics.mean(p95s) * 1000, 1),
        "first_text_p50_ms": round(statistics.median(firsts), 1) if firsts else None,
    }
    print(json.dumps(res, ensure_ascii=False))
    for u, r, h in bad:
        print(f"  BAD {u}\n    ref {r}\n    hyp {h}")
    if a.out:
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

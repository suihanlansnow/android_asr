"""流式 ASR 批量评测：CER / RTF / 首字延迟 / 解码耗时分位。

模拟真实麦克风节奏（默认 40ms 一块）喂音频，因此首字延迟与端到端延迟都可直接测量。

用法:
  python scripts/eval_cer.py --int8 1 --threads 2 --feed-ms 40 --json-out data/eval_i1_t2.json
"""

import argparse
import json
import math
import os
import re
import statistics
import time

import numpy as np
import psutil
import soundfile as sf
from scipy.signal import resample_poly

import sherpa_onnx

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(
    ROOT, "models", "sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23"
)
SAMPLE_RATE = 16000
PUNCT = re.compile(r"[，。！？、；：\"'（）《》\s]")


def build_recognizer(num_threads: int, use_int8: bool, model_dir: str, arch: str = "transducer",
                     enc_int8=None, dec_int8=None, join_int8=None):
    """use_int8 为总开关；enc_int8/dec_int8/join_int8 单独指定时覆盖总开关，便于混合精度 A/B。"""
    stem = "epoch-99-avg-1"
    tokens = os.path.join(model_dir, "tokens.txt")

    def pick(comp_int8, name):
        flag = use_int8 if comp_int8 is None else bool(comp_int8)
        s = ".int8.onnx" if flag else ".onnx"
        return os.path.join(model_dir, f"{name}-{stem}{s}")

    enc = pick(enc_int8, "encoder")
    dec = pick(dec_int8, "decoder")
    if arch == "paraformer":
        return sherpa_onnx.OnlineRecognizer.from_paraformer(
            tokens=tokens,
            encoder=enc,
            decoder=dec,
            num_threads=num_threads,
            sample_rate=SAMPLE_RATE,
            feature_dim=80,
            decoding_method="greedy_search",
            enable_endpoint_detection=False,
        )
    return sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=tokens,
        encoder=enc,
        decoder=dec,
        joiner=pick(join_int8, "joiner"),
        num_threads=num_threads,
        sample_rate=SAMPLE_RATE,
        feature_dim=80,
        decoding_method="greedy_search",
        enable_endpoint_detection=False,
    )


def norm(text: str) -> str:
    return PUNCT.sub("", text).lower()


def load_wav(path: str) -> np.ndarray:
    samples, sr = sf.read(path, dtype="float32", always_2d=False)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if sr != SAMPLE_RATE:
        g = math.gcd(int(sr), SAMPLE_RATE)
        samples = resample_poly(samples, SAMPLE_RATE // g, int(sr) // g).astype(np.float32)
    return samples


def edit_distance(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def decode_one(rec, samples: np.ndarray, feed_ms: int) -> dict:
    step = int(SAMPLE_RATE * feed_ms / 1000)
    stream = rec.create_stream()
    decode_times = []
    first_text_at = None
    text = ""
    audio_t = 0.0

    for pos in range(0, len(samples), step):
        block = samples[pos : pos + step]
        stream.accept_waveform(SAMPLE_RATE, block)
        t0 = time.perf_counter()
        while rec.is_ready(stream):
            rec.decode_stream(stream)
        dt = time.perf_counter() - t0
        decode_times.append(dt)
        audio_t += len(block) / SAMPLE_RATE

        cur = rec.get_result(stream).strip()
        if cur and first_text_at is None:
            first_text_at = audio_t + dt
        text = cur

    t0 = time.perf_counter()
    stream.input_finished()
    while rec.is_ready(stream):
        rec.decode_stream(stream)
    tail = time.perf_counter() - t0
    text = rec.get_result(stream).strip()

    dur = len(samples) / SAMPLE_RATE
    dt_sorted = sorted(decode_times)
    p95 = dt_sorted[min(len(dt_sorted) - 1, int(len(dt_sorted) * 0.95))]
    return {
        "text": text,
        "audio_sec": dur,
        "decode_sec": sum(decode_times) + tail,
        "rtf": (sum(decode_times) + tail) / dur,
        "decode_p95_ms": p95 * 1000,
        "decode_max_ms": max(decode_times) * 1000,
        "first_text_ms": (first_text_at * 1000) if first_text_at else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(ROOT, "data", "manifest_aishell3.jsonl"))
    ap.add_argument("--int8", type=int, default=1)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--feed-ms", type=int, default=40)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model-dir", default=MODEL_DIR)
    ap.add_argument("--arch", default="transducer")
    ap.add_argument("--json-out", default="")
    ap.add_argument("--enc-int8", type=int, default=None)
    ap.add_argument("--dec-int8", type=int, default=None)
    ap.add_argument("--join-int8", type=int, default=None)
    args = ap.parse_args()

    items = [json.loads(l) for l in open(args.manifest, encoding="utf-8")]
    if args.limit:
        items = items[: args.limit]

    proc = psutil.Process()
    rss0 = proc.memory_info().rss
    rec = build_recognizer(args.threads, bool(args.int8), args.model_dir, args.arch,
                           enc_int8=args.enc_int8, dec_int8=args.dec_int8, join_int8=args.join_int8)
    rss1 = proc.memory_info().rss

    errs = chars = 0
    sent_cer = []
    rtf_list, p95_list, first_list = [], [], []
    rep_ratios = []
    rep_utts = 0
    t_start = time.perf_counter()
    bad = []

    def repetition_ratio(s: str) -> float:
        """连续相同字符占比：重复循环的直接信号。"""
        if len(s) < 2:
            return 0.0
        same = sum(1 for i in range(1, len(s)) if s[i] == s[i - 1])
        return same / len(s)

    for i, it in enumerate(items):
        ref = norm(it["text"])
        r = decode_one(rec, load_wav(it["wav"]), args.feed_ms)
        hyp = norm(r["text"])
        e = edit_distance(ref, hyp)
        errs += e
        chars += len(ref)
        sent_cer.append(e / max(1, len(ref)))
        rtf_list.append(r["rtf"])
        p95_list.append(r["decode_p95_ms"])
        if r["first_text_ms"]:
            first_list.append(r["first_text_ms"])
        rr = repetition_ratio(hyp)
        rep_ratios.append(rr)
        if rr > 0.05:
            rep_utts += 1
        if e / max(1, len(ref)) > 0.5 and len(bad) < 8:
            bad.append({"utt": it["utt"], "ref": ref, "hyp": hyp})
        if (i + 1) % 100 == 0:
            print(f"  .. {i + 1}/{len(items)}  cer={errs / max(1, chars):.4f}", flush=True)

    wall = time.perf_counter() - t_start
    audio_total = sum(sf.info(it["wav"]).duration for it in items)
    enc_i8 = args.int8 if args.enc_int8 is None else args.enc_int8
    dec_i8 = args.int8 if args.dec_int8 is None else args.dec_int8
    join_i8 = args.int8 if args.join_int8 is None else args.join_int8
    stem = "epoch-99-avg-1"
    size_mb = sum(
        os.path.getsize(os.path.join(args.model_dir, f"{name}-{stem}{'.int8.onnx' if flag else '.onnx'}"))
        for name, flag in (("encoder", enc_i8), ("decoder", dec_i8), ("joiner", join_i8))
    ) / 1024 / 1024

    res = {
        "config": {"int8": bool(args.int8), "threads": args.threads, "feed_ms": args.feed_ms,
                   "enc_int8": args.enc_int8, "dec_int8": args.dec_int8, "join_int8": args.join_int8},
        "n_utts": len(items),
        "audio_min": round(audio_total / 60, 1),
        "cer": round(errs / max(1, chars), 4),
        "cer_sent_mean": round(statistics.mean(sent_cer), 4),
        "rep_ratio_mean": round(statistics.mean(rep_ratios), 4),
        "rep_utts": rep_utts,
        "rtf": round(statistics.mean(rtf_list), 4),
        "rtf_p95": round(sorted(rtf_list)[int(len(rtf_list) * 0.95)], 4),
        "decode_p95_ms": round(statistics.mean(p95_list), 1),
        "first_text_p50_ms": round(statistics.median(first_list), 1),
        "first_text_p95_ms": round(sorted(first_list)[int(len(first_list) * 0.95)], 1),
        "model_size_mb": round(size_mb, 1),
        "model_rss_mb": round((rss1 - rss0) / 1024 / 1024, 1),
        "wall_sec": round(wall, 1),
        "speedup_vs_realtime": round(audio_total / wall, 1),
        "worst_samples": bad,
    }
    print(json.dumps({k: v for k, v in res.items() if k != "worst_samples"}, ensure_ascii=False, indent=2))
    for b in bad[:3]:
        print("  BAD", b["utt"], "\n    ref:", b["ref"], "\n    hyp:", b["hyp"])
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(res, fh, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

"""流式 ASR 基准测试：模拟实时音频输入，采集 RTF / 每块解码耗时 / 首字延迟 / 内存。

用法:
  python scripts/asr_bench.py --wav models/.../test_wavs/0.wav --chunk-ms 320 --threads 2
  python scripts/asr_bench.py --dir models/.../test_wavs --chunk-ms 320 --threads 2 --int8 1
"""

import argparse
import glob
import json
import os
import statistics
import sys
import time

import numpy as np
import psutil
import soundfile as sf

import sherpa_onnx

MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23",
)
SAMPLE_RATE = 16000


def build_recognizer(num_threads: int, use_int8: bool, provider: str):
    suffix = ".int8.onnx" if use_int8 else ".onnx"
    stem = "epoch-99-avg-1"
    return sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=os.path.join(MODEL_DIR, "tokens.txt"),
        encoder=os.path.join(MODEL_DIR, f"encoder-{stem}{suffix}"),
        decoder=os.path.join(MODEL_DIR, f"decoder-{stem}{suffix}"),
        joiner=os.path.join(MODEL_DIR, f"joiner-{stem}{suffix}"),
        num_threads=num_threads,
        sample_rate=SAMPLE_RATE,
        feature_dim=80,
        decoding_method="greedy_search",
        provider=provider,
        enable_endpoint_detection=False,
    )


def load_wav(path: str) -> np.ndarray:
    samples, sr = sf.read(path, dtype="float32", always_2d=False)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if sr != SAMPLE_RATE:
        # 线性重采样，仅用于基准测试
        n = int(round(len(samples) * SAMPLE_RATE / sr))
        samples = np.interp(
            np.linspace(0, len(samples) - 1, n), np.arange(len(samples)), samples
        ).astype(np.float32)
    return samples


def run_one(recognizer, samples: np.ndarray, chunk_ms: int) -> dict:
    """按真实音频到达节奏喂块，记录每块解码耗时与首字延迟。"""
    chunk = int(SAMPLE_RATE * chunk_ms / 1000)
    stream = recognizer.create_stream()

    decode_times = []
    first_text_at = None  # 相对音频开始的时间点(秒)
    text = ""
    pos = 0
    t_audio = 0.0

    while pos < len(samples):
        block = samples[pos : pos + chunk]
        pos += chunk
        stream.accept_waveform(SAMPLE_RATE, block)

        t0 = time.perf_counter()
        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)
        dt = time.perf_counter() - t0
        decode_times.append(dt)

        t_audio += len(block) / SAMPLE_RATE
        cur = recognizer.get_result(stream).strip()
        if cur and first_text_at is None:
            first_text_at = t_audio + dt  # 音频到达终点 + 解码耗时
        if cur:
            text = cur

    tail0 = time.perf_counter()
    stream.input_finished()
    while recognizer.is_ready(stream):
        recognizer.decode_stream(stream)
    tail_dt = time.perf_counter() - tail0
    text = recognizer.get_result(stream).strip()

    audio_dur = len(samples) / SAMPLE_RATE
    total_decode = sum(decode_times) + tail_dt
    sorted_dt = sorted(decode_times)
    p95 = sorted_dt[min(len(sorted_dt) - 1, int(len(sorted_dt) * 0.95))]

    return {
        "audio_sec": round(audio_dur, 2),
        "rtf": round(total_decode / audio_dur, 4),
        "chunk_avg_ms": round(statistics.mean(decode_times) * 1000, 1),
        "chunk_p95_ms": round(p95 * 1000, 1),
        "chunk_max_ms": round(max(decode_times) * 1000, 1),
        # 稳态端到端延迟估计: 一个块的音频积累 + 该块解码耗时
        "e2e_latency_ms": round(chunk_ms + p95 * 1000, 1),
        "first_text_ms": round(first_text_at * 1000, 1) if first_text_at else None,
        "realtime_ok": bool(p95 < chunk_ms / 1000),
        "text": text,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", action="append", default=[])
    ap.add_argument("--dir", default="")
    ap.add_argument("--chunk-ms", type=int, default=320)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--int8", type=int, default=1)
    ap.add_argument("--provider", default="cpu")
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    files = list(args.wav)
    if args.dir:
        files += sorted(glob.glob(os.path.join(args.dir, "*.wav")))
    if not files:
        sys.exit("need --wav or --dir")

    proc = psutil.Process()
    rss0 = proc.memory_info().rss
    t0 = time.perf_counter()
    rec = build_recognizer(args.threads, bool(args.int8), args.provider)
    init_ms = (time.perf_counter() - t0) * 1000
    rss1 = proc.memory_info().rss

    rows = []
    for f in files:
        r = run_one(rec, load_wav(f), args.chunk_ms)
        r["file"] = os.path.basename(f)
        rows.append(r)
        print(
            f"{r['file']:<8} audio={r['audio_sec']:>6.2f}s rtf={r['rtf']:<6} "
            f"chunk_avg={r['chunk_avg_ms']:>5}ms p95={r['chunk_p95_ms']:>5}ms "
            f"e2e≈{r['e2e_latency_ms']:>6}ms first={r['first_text_ms']} "
            f"ok={r['realtime_ok']} | {r['text']}"
        )

    summary = {
        "config": {
            "chunk_ms": args.chunk_ms,
            "threads": args.threads,
            "int8": bool(args.int8),
            "provider": args.provider,
        },
        "init_ms": round(init_ms, 1),
        "mem_rss_mb": round((rss1 - rss0) / 1024 / 1024, 1),
        "files": rows,
    }
    print(
        f"\ninit={init_ms:.0f}ms  model_rss≈{(rss1 - rss0) / 1024 / 1024:.1f}MB  "
        f"avg_rtf={statistics.mean([r['rtf'] for r in rows]):.4f}"
    )
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=2)
        print(f"saved -> {args.json_out}")


if __name__ == "__main__":
    main()

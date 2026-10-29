"""生成带 babble 噪声（多人同时说话）的评测集，模拟餐厅/多人场景。

用法: python scripts/make_noisy.py --snr 10 --n 200
产物: data/noise10/*.wav + data/manifest_noise10.jsonl
"""
import argparse, json, os, random
import numpy as np
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SR = 16000


def rms(x):
    return float(np.sqrt(np.mean(np.square(x)) + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snr", type=float, default=10.0)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--babble", type=int, default=4, help="干扰语音条数")
    a = ap.parse_args()

    items = [json.loads(l) for l in open(
        os.path.join(ROOT, "data", "manifest_aishell3.jsonl"), encoding="utf-8")]
    items = items[: a.n]
    random.seed(0)
    tag = "noise%g" % a.snr
    outdir = os.path.join(ROOT, "data", tag)
    os.makedirs(outdir, exist_ok=True)
    out = []

    for it in items:
        y, sr = sf.read(it["wav"], dtype="float32", always_2d=False)
        if y.ndim > 1:
            y = y.mean(axis=1)
        n = np.zeros_like(y)
        for _ in range(a.babble):
            # 取另外随机一条语音片段当干扰，循环拼接对齐长度
            other = random.choice(items)
            z, _ = sf.read(other["wav"], dtype="float32", always_2d=False)
            if z.ndim > 1:
                z = z.mean(axis=1)
            if len(z) < len(y):
                reps = int(np.ceil(len(y) / len(z)))
                z = np.tile(z, reps)
            off = random.randint(0, max(0, len(z) - len(y)))
            n += z[off:off + len(y)]
        n /= a.babble
        scale = rms(y) / (rms(n) * (10 ** (a.snr / 20.0)) + 1e-9)
        mix = y + n * scale
        mix = np.clip(mix, -1, 1)
        name = it["utt"] + ".wav"
        sf.write(os.path.join(outdir, name), mix, sr)
        out.append({"utt": it["utt"], "wav": os.path.join(outdir, name), "text": it["text"]})

    man = os.path.join(ROOT, "data", "manifest_%s.jsonl" % tag)
    with open(man, "w", encoding="utf-8") as fh:
        for row in out:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("wrote", man, len(out))


if __name__ == "__main__":
    main()

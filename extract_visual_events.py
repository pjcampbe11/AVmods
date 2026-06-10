#!/usr/bin/env python3
"""
extract_visual_events.py — Scan a movie for frames matching descriptive
words (disturbing, graphic, brutally violent, gross, horrifying, etc.)
using OpenAI's CLIP model (zero-shot image/text matching, runs locally),
then export the best frames as high-res JPG + PNG stills sized to fill
the screen (default 1920x1080, letterboxed, lanczos upscale) — ready to
drop into an MP4 slideshow with ffmpeg.

Pipeline:
  1. OpenCV samples one frame every N seconds (small/fast copies).
  2. CLIP scores each frame against your word list vs. neutral prompts.
  3. Frames above threshold are picked with a minimum time gap
     (avoids near-duplicate stills from the same shot).
  4. ffmpeg re-grabs each picked timestamp from the ORIGINAL video at
     full quality, scales/pads to target size, writes .jpg (-q:v 1)
     and lossless .png.
  5. CSV log: timestamp, score, best-matching word.

Requirements:
  - ffmpeg on PATH
  - py -3.10 -m pip install torch torchvision open_clip_torch opencv-python pillow

Usage:
  py -3.10 extract_visual_events.py "movie.mp4" -o stills
  py -3.10 extract_visual_events.py "movie.mp4" --words gory terrifying "blood splatter"
  py -3.10 extract_visual_events.py "movie.mp4" --threshold 0.4 --top 50 --size 3840x2160
"""

import argparse
import csv
import os
import subprocess
import sys

import cv2
import numpy as np
import torch
from PIL import Image

DEFAULT_WORDS = [
    "disturbing", "graphic", "brutally violent", "gross", "disgusting",
    "horrifying", "terrifying", "creepy", "bloody", "gruesome",
]

# Neutral/contrast prompts — CLIP scores are relative, so these anchor
# what "not a match" looks like.
NEGATIVE_PROMPTS = [
    "a calm ordinary scene from a movie",
    "a neutral conversation between people",
    "a peaceful landscape",
    "a boring everyday moment",
    "opening credits or a title card",
]

POS_TEMPLATE = "a {} scene from a movie"


def load_clip(device):
    import open_clip
    print("Loading CLIP model (ViT-B-32, ~340 MB download on first run)...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model = model.to(device).eval()
    return model, preprocess, tokenizer


@torch.no_grad()
def text_features(model, tokenizer, words, device):
    prompts = [POS_TEMPLATE.format(w) for w in words] + NEGATIVE_PROMPTS
    feats = model.encode_text(tokenizer(prompts).to(device))
    feats /= feats.norm(dim=-1, keepdim=True)
    return feats, len(words)


@torch.no_grad()
def scan(video_path, model, preprocess, tfeats, n_pos, words, device,
         interval, batch_size=32):
    """Return list of (time_seconds, score, best_word)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, round(fps * interval))
    duration = total_frames / fps

    results, batch, times = [], [], []

    def flush():
        if not batch:
            return
        imgs = torch.stack(batch).to(device)
        ifeats = model.encode_image(imgs)
        ifeats /= ifeats.norm(dim=-1, keepdim=True)
        probs = (100.0 * ifeats @ tfeats.T).softmax(dim=-1)
        pos = probs[:, :n_pos]
        scores = pos.sum(dim=-1)            # total positive probability
        best = pos.argmax(dim=-1)
        for t, s, b in zip(times, scores.tolist(), best.tolist()):
            results.append((t, s, words[b]))
        batch.clear()
        times.clear()

    idx = 0
    while True:
        if not cap.grab():
            break
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if ok:
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                batch.append(preprocess(img))
                times.append(idx / fps)
                if len(batch) >= batch_size:
                    flush()
                    print(f"  scanned {idx / fps:7.1f}/{duration:.1f} s",
                          end="\r")
        idx += 1
    flush()
    cap.release()
    print()
    return results


def pick(results, threshold, min_gap, top):
    hits = [r for r in results if r[1] >= threshold]
    hits.sort(key=lambda r: r[1], reverse=True)   # best first
    picked = []
    for t, s, w in hits:
        if all(abs(t - pt) >= min_gap for pt, _, _ in picked):
            picked.append((t, s, w))
            if top and len(picked) >= top:
                break
    picked.sort()                                  # chronological order
    return picked


def export(video_path, picked, outdir, size):
    w, h = size
    vf = (f"scale={w}:{h}:force_original_aspect_ratio=decrease:flags=lanczos,"
          f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:black")
    os.makedirs(outdir, exist_ok=True)
    for i, (t, s, word) in enumerate(picked, 1):
        safe = word.replace(" ", "-")
        base = os.path.join(outdir, f"frame_{i:04d}_t{t:08.2f}_{safe}")
        for ext, extra in ((".jpg", ["-q:v", "1"]),
                           (".png", ["-pred", "mixed"])):
            subprocess.run(
                ["ffmpeg", "-y", "-ss", f"{t:.3f}", "-i", video_path,
                 "-frames:v", "1", "-vf", vf, *extra, base + ext],
                check=True, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL)
        print(f"  [{i}/{len(picked)}] t={t:.1f}s  score={s:.2f}  ({word})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("-o", "--outdir", default="stills",
                    help="output folder for images (default: stills)")
    ap.add_argument("--words", nargs="+", default=DEFAULT_WORDS,
                    help="descriptive words/phrases to match")
    ap.add_argument("--interval", type=float, default=1.0,
                    help="seconds between sampled frames (default 1.0)")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="match score 0-1 (lower = more frames, default 0.5)")
    ap.add_argument("--min-gap", type=float, default=5.0,
                    help="min seconds between exported frames (default 5)")
    ap.add_argument("--top", type=int, default=0,
                    help="keep only the N highest-scoring frames (0 = all)")
    ap.add_argument("--size", default="1920x1080",
                    help="output WxH, e.g. 1920x1080 or 3840x2160")
    args = ap.parse_args()

    try:
        size = tuple(int(x) for x in args.size.lower().split("x"))
        assert len(size) == 2
    except Exception:
        sys.exit("--size must look like 1920x1080")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    model, preprocess, tokenizer = load_clip(device)
    tfeats, n_pos = text_features(model, tokenizer, args.words, device)

    print("Scanning frames...")
    results = scan(args.video, model, preprocess, tfeats, n_pos,
                   args.words, device, args.interval)

    log_path = os.path.join(args.outdir, "_scores.csv")
    os.makedirs(args.outdir, exist_ok=True)
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        wtr = csv.writer(f)
        wtr.writerow(["time_seconds", "score", "best_word"])
        wtr.writerows((f"{t:.2f}", f"{s:.3f}", w) for t, s, w in results)

    picked = pick(results, args.threshold, args.min_gap, args.top)
    if not picked:
        sys.exit(f"No frames above threshold {args.threshold} — "
                 f"check {log_path} for the score distribution and lower it.")
    print(f"{len(picked)} frames selected. Exporting full-res stills...")
    export(args.video, picked, args.outdir, size)
    print(f"Done. Images + _scores.csv in: {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()
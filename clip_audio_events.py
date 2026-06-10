#!/usr/bin/env python3
"""
clip_audio_events.py — Detect audio events (screaming, yelling, crying,
whispering, etc.) in a movie file and compile the matching clips into a
single MP4, entirely locally.

Pipeline:
  1. ffmpeg extracts mono 16 kHz audio from the video.
  2. Google's YAMNet model (pretrained on AudioSet, 521 sound classes)
     scores the audio in ~1-second windows.
  3. Windows matching your target classes above a confidence threshold are
     merged into segments (with padding).
  4. ffmpeg cuts each segment and concatenates them into one output MP4.
  5. A CSV log of every detection (timestamp, class, confidence) is saved.

Requirements:
  - ffmpeg on PATH  (https://ffmpeg.org or `winget install ffmpeg`)
  - Python 3.9-3.12:
      pip install tensorflow tensorflow-hub numpy soundfile

Usage:
  python clip_audio_events.py "movie.mp4" -o compilation.mp4
  python clip_audio_events.py "movie.mp4" --threshold 0.2 --pad 2.0
  python clip_audio_events.py "movie.mp4" --keywords scream yell whisper
"""

import argparse
import csv
import os
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf

# Default AudioSet class-name keywords to hunt for (substring match,
# case-insensitive). Run with --list-classes to see all 521 class names.
DEFAULT_KEYWORDS = [
    "scream", "yell", "shout", "bellow", "shriek",
    "crying", "sobbing", "whimper", "wail", "moan",
    "whisper", "groan", "gasp",
]

SAMPLE_RATE = 16000          # YAMNet requires 16 kHz mono
YAMNET_HOP = 0.48            # seconds between YAMNet score frames
BLOCK_SECONDS = 600          # process audio in 10-min blocks (memory safety)


def run(cmd, **kw):
    print("  $", " ".join(str(c) for c in cmd))
    return subprocess.run(cmd, check=True, **kw)


def extract_audio(video_path, wav_path):
    run(["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1",
         "-ar", str(SAMPLE_RATE), "-f", "wav", wav_path],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def load_yamnet():
    import tensorflow_hub as hub
    print("Loading YAMNet model (downloads ~17 MB on first run)...")
    model = hub.load("https://tfhub.dev/google/yamnet/1")
    class_map_path = model.class_map_path().numpy().decode("utf-8")
    with open(class_map_path, newline="", encoding="utf-8") as f:
        class_names = [row["display_name"] for row in csv.DictReader(f)]
    return model, class_names


def detect(wav_path, model, class_names, keywords, threshold):
    """Yield (time_seconds, class_name, confidence) detections."""
    target_idx = {
        i: name for i, name in enumerate(class_names)
        if any(k.lower() in name.lower() for k in keywords)
    }
    if not target_idx:
        sys.exit("No YAMNet classes matched your keywords.")
    print("Target classes:", ", ".join(sorted(set(target_idx.values()))))

    detections = []
    block_len = BLOCK_SECONDS * SAMPLE_RATE
    with sf.SoundFile(wav_path) as f:
        total = f.frames / SAMPLE_RATE
        offset = 0.0
        while True:
            block = f.read(block_len, dtype="float32")
            if len(block) == 0:
                break
            scores, _, _ = model(block)
            scores = scores.numpy()
            for fi, frame in enumerate(scores):
                for ci, name in target_idx.items():
                    conf = float(frame[ci])
                    if conf >= threshold:
                        detections.append((offset + fi * YAMNET_HOP, name, conf))
            offset += len(block) / SAMPLE_RATE
            print(f"  analyzed {min(offset, total):7.1f}/{total:.1f} s "
                  f"({len(detections)} hits so far)")
    return detections


def merge_segments(detections, pad, max_gap, min_len):
    """Collapse per-frame detections into (start, end) clip segments."""
    if not detections:
        return []
    times = sorted(t for t, _, _ in detections)
    segments = []
    start = prev = times[0]
    for t in times[1:]:
        if t - prev <= max_gap:
            prev = t
        else:
            segments.append((start, prev))
            start = prev = t
    segments.append((start, prev))
    out = []
    for s, e in segments:
        s, e = max(0.0, s - pad), e + YAMNET_HOP + pad
        if e - s >= min_len:
            if out and s <= out[-1][1]:          # merge overlaps after padding
                out[-1] = (out[-1][0], max(out[-1][1], e))
            else:
                out.append((s, e))
    return out


def cut_and_concat(video_path, segments, out_path, tmpdir):
    clip_paths = []
    for i, (s, e) in enumerate(segments):
        clip = os.path.join(tmpdir, f"clip_{i:04d}.mp4")
        # Re-encode for frame-accurate cuts and safe concatenation.
        run(["ffmpeg", "-y", "-ss", f"{s:.3f}", "-i", video_path,
             "-t", f"{e - s:.3f}",
             "-c:v", "libx264", "-preset", "fast", "-crf", "20",
             "-c:a", "aac", "-b:a", "160k",
             "-avoid_negative_ts", "make_zero", clip],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        clip_paths.append(clip)
    listfile = os.path.join(tmpdir, "concat.txt")
    with open(listfile, "w", encoding="utf-8") as f:
        for p in clip_paths:
            f.write(f"file '{p}'\n")
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", listfile,
         "-c", "copy", out_path],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", help="input video file (mp4/mkv/etc.)")
    ap.add_argument("-o", "--output", default="compilation.mp4")
    ap.add_argument("--keywords", nargs="+", default=DEFAULT_KEYWORDS,
                    help="class-name substrings to match")
    ap.add_argument("--threshold", type=float, default=0.25,
                    help="confidence threshold 0-1 (lower = more clips)")
    ap.add_argument("--pad", type=float, default=1.5,
                    help="seconds of context before/after each event")
    ap.add_argument("--max-gap", type=float, default=2.0,
                    help="merge detections closer than this many seconds")
    ap.add_argument("--min-len", type=float, default=1.0,
                    help="discard segments shorter than this")
    ap.add_argument("--list-classes", action="store_true",
                    help="print all 521 YAMNet class names and exit")
    args = ap.parse_args()

    model, class_names = load_yamnet()
    if args.list_classes:
        print("\n".join(class_names))
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        wav = os.path.join(tmpdir, "audio.wav")
        print("Extracting audio...")
        extract_audio(args.video, wav)

        print("Scanning for events...")
        detections = detect(wav, model, class_names, args.keywords,
                            args.threshold)

        log_path = os.path.splitext(args.output)[0] + "_detections.csv"
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["time_seconds", "class", "confidence"])
            w.writerows((f"{t:.2f}", c, f"{p:.3f}")
                        for t, c, p in sorted(detections))
        print(f"Detection log: {log_path}")

        segments = merge_segments(detections, args.pad, args.max_gap,
                                  args.min_len)
        if not segments:
            sys.exit("No segments found — try lowering --threshold.")
        total = sum(e - s for s, e in segments)
        print(f"{len(segments)} segments, {total:.1f} s total. Cutting...")

        cut_and_concat(args.video, segments, args.output, tmpdir)
    print(f"Done: {args.output}")


if __name__ == "__main__":
    main()
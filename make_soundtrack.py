#!/usr/bin/env python3
"""
make_soundtrack.py — Generate an AI background music/sound bed whose vibe
matches your keywords, exactly as long as the longest of your input videos.
Runs 100% locally using Meta's MusicGen model.

Pipeline:
  1. ffprobe measures both videos; target duration = the longer one.
  2. Your keywords are woven into a music-generation prompt
     (or pass --prompt to write your own).
  3. MusicGen generates music in 30 s chunks; chunks are equal-power
     crossfaded into a seamless bed, looped if needed to fill the target,
     trimmed to the exact duration, with fade-in/out.
  4. Output: soundtrack.wav (32 kHz) + a ready-to-paste ffmpeg command to
     mix it under each video.

Requirements:
  - ffmpeg/ffprobe on PATH
  - py -3.10 -m pip install transformers scipy soundfile numpy
    (torch already installed; first run downloads MusicGen ~2 GB)

Usage:
  py -3.10 make_soundtrack.py compilation.mp4 stills_reel.mp4
  py -3.10 make_soundtrack.py a.mp4 b.mp4 --keywords horrifying creepy whispering
  py -3.10 make_soundtrack.py a.mp4 b.mp4 --prompt "slow eerie ambient drone, distant screams, horror film score"
  py -3.10 make_soundtrack.py a.mp4 b.mp4 --model facebook/musicgen-medium   # better, slower

Note: CPU generation is slow (~2-5 min per 30 s chunk with musicgen-small).
By default the script generates up to --gen-seconds (90 s) of unique
material and loops it to fill longer targets — seamless and much faster.
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import soundfile as sf

DEFAULT_KEYWORDS = ["disturbing", "horrifying", "creepy", "tense", "eerie"]

PROMPT_TEMPLATE = (
    "cinematic horror film score, {kw} atmosphere, dark ambient drones, "
    "dissonant strings, deep sub bass pulses, sparse unsettling piano, "
    "slow build of dread, high production quality"
)

SR = 32000          # MusicGen output sample rate
CHUNK_S = 30        # max seconds MusicGen generates per pass
XFADE_S = 3.0       # crossfade between chunks / loop seam


def duration_of(path):
    out = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_format", path],
        capture_output=True, text=True, check=True).stdout
    return float(json.loads(out)["format"]["duration"])


def load_musicgen(model_name):
    import torch
    from transformers import AutoProcessor, MusicgenForConditionalGeneration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {model_name} on {device} "
          "(first run downloads ~2 GB)...")
    processor = AutoProcessor.from_pretrained(model_name)
    model = MusicgenForConditionalGeneration.from_pretrained(model_name)
    model = model.to(device).eval()
    return model, processor, device


def generate_chunk(model, processor, device, prompt, seconds, seed):
    import torch
    torch.manual_seed(seed)
    inputs = processor(text=[prompt], padding=True,
                       return_tensors="pt").to(device)
    tokens = min(int(seconds * 50) + 4, 1503)   # MusicGen: ~50 tokens/sec
    with torch.no_grad():
        audio = model.generate(**inputs, do_sample=True,
                               guidance_scale=3.0,
                               max_new_tokens=tokens)
    return audio[0, 0].cpu().numpy().astype(np.float32)


def equal_power_xfade(a, b, fade_samples):
    """Append b to a with an equal-power crossfade."""
    fade_samples = min(fade_samples, len(a), len(b))
    t = np.linspace(0, np.pi / 2, fade_samples, dtype=np.float32)
    a_tail = a[-fade_samples:] * np.cos(t)
    b_head = b[:fade_samples] * np.sin(t)
    return np.concatenate([a[:-fade_samples], a_tail + b_head,
                           b[fade_samples:]])


def build_bed(model, processor, device, prompt, gen_seconds, target_seconds,
              seed):
    xfade = int(XFADE_S * SR)
    # 1) generate unique material (in <=30 s chunks, crossfaded)
    unique = None
    made = 0.0
    want = min(gen_seconds, target_seconds + XFADE_S)
    n = 0
    while made < want:
        n += 1
        secs = min(CHUNK_S, want - made + XFADE_S)
        print(f"  generating chunk {n} ({secs:.0f} s)... "
              "(slow on CPU, be patient)")
        chunk = generate_chunk(model, processor, device, prompt, secs,
                               seed + n)
        unique = chunk if unique is None else \
            equal_power_xfade(unique, chunk, xfade)
        made = len(unique) / SR
    # 2) loop the bed to reach the target duration
    bed = unique
    while len(bed) / SR < target_seconds:
        print("  looping bed to extend duration...")
        bed = equal_power_xfade(bed, unique, xfade)
    # 3) trim exactly, then fade in/out
    bed = bed[: int(target_seconds * SR)]
    fade = int(min(2.0, target_seconds / 4) * SR)
    bed[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)
    bed[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
    # 4) normalize to -1 dBFS-ish
    peak = np.abs(bed).max() or 1.0
    bed = bed / peak * 0.89
    return bed


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="+",
                    help="one or more video files; target = longest")
    ap.add_argument("-o", "--output", default="soundtrack.wav")
    ap.add_argument("--keywords", nargs="+", default=DEFAULT_KEYWORDS,
                    help="vibe words woven into the music prompt")
    ap.add_argument("--prompt", default=None,
                    help="full custom prompt (overrides --keywords)")
    ap.add_argument("--model", default="facebook/musicgen-small",
                    help="facebook/musicgen-small | -medium | -large")
    ap.add_argument("--gen-seconds", type=float, default=90,
                    help="seconds of unique material before looping "
                         "(default 90; higher = less repetition, slower)")
    ap.add_argument("--seed", type=int, default=42,
                    help="change for a different take")
    args = ap.parse_args()

    durations = {v: duration_of(v) for v in args.videos}
    for v, d in durations.items():
        print(f"  {os.path.basename(v)}: {d:.2f} s")
    target = max(durations.values())
    longest = max(durations, key=durations.get)
    print(f"Target duration: {target:.2f} s (from {os.path.basename(longest)})")

    prompt = args.prompt or PROMPT_TEMPLATE.format(kw=", ".join(args.keywords))
    print(f"Prompt: {prompt}")

    model, processor, device = load_musicgen(args.model)
    bed = build_bed(model, processor, device, prompt, args.gen_seconds,
                    target, args.seed)
    sf.write(args.output, bed, SR)
    print(f"\nDone: {args.output} ({len(bed)/SR:.2f} s, 32 kHz)")

    print("\nMix it under a video (music at 35% beneath original audio):")
    print(f'  ffmpeg -i "{longest}" -i "{args.output}" -filter_complex '
          f'"[1:a]volume=0.35[m];[0:a][m]amix=inputs=2:duration=first:'
          f'normalize=0" -c:v copy mixed.mp4')
    print("\nOr REPLACE the video's audio with the music:")
    print(f'  ffmpeg -i "{longest}" -i "{args.output}" -map 0:v -map 1:a '
          f"-c:v copy -c:a aac -shortest replaced.mp4")


if __name__ == "__main__":
    main()
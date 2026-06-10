#!/usr/bin/env python3
"""
stills_to_movie.py — Assemble JPG/PNG stills into an MP4 slideshow, with
optional audio extracted from ANY media file (mp4, mkv, mp3, wav, m4a...)
overlaid on top. Pure ffmpeg + Python stdlib — no extra pip packages.

Features:
  - Sorts images by filename (the extractor's frame_0001_... names keep
    chronological order automatically).
  - If the same still exists as both .jpg and .png, uses one (PNG by
    default, --prefer jpg to flip) — no duplicate slides.
  - Static slides, gentle continuous Ken Burns zoom (--zoom), or
    crossfade transitions (--crossfade 1.0).
  - Audio from any source file: seek into it (--audio-start), set volume,
    loop it if shorter than the video, fade in/out, pad with silence —
    final video length always equals the slideshow length.
  - Standalone audio ripper: --extract-only pulls the audio track out of
    any media file to wav/mp3/m4a and exits.

Usage:
  py -3.10 stills_to_movie.py stills -o reel.mp4
  py -3.10 stills_to_movie.py stills -o reel.mp4 --duration 5 --zoom
  py -3.10 stills_to_movie.py stills -o reel.mp4 --crossfade 1.0
  py -3.10 stills_to_movie.py stills -o reel.mp4 --audio soundtrack.wav
  py -3.10 stills_to_movie.py stills -o reel.mp4 --audio movie.mp4 --audio-start 3600 --loop-audio
  py -3.10 stills_to_movie.py --audio movie.mp4 --extract-only ripped.mp3
"""

import argparse
import os
import subprocess
import sys
import tempfile

IMG_EXTS = {".jpg", ".jpeg", ".png"}
AUDIO_CODECS = {".wav": ["-c:a", "pcm_s16le"],
                ".mp3": ["-c:a", "libmp3lame", "-q:a", "0"],
                ".m4a": ["-c:a", "aac", "-b:a", "256k"],
                ".aac": ["-c:a", "aac", "-b:a", "256k"],
                ".flac": ["-c:a", "flac"]}


def run(cmd):
    print("  $", " ".join(str(c) for c in cmd))
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(f"ffmpeg failed (exit {r.returncode})")


def collect_images(folder, prefer):
    files = [f for f in os.listdir(folder)
             if os.path.splitext(f)[1].lower() in IMG_EXTS]
    by_stem = {}
    for f in sorted(files):
        stem, ext = os.path.splitext(f)
        ext = ext.lower().lstrip(".").replace("jpeg", "jpg")
        cur = by_stem.get(stem)
        if cur is None or ext == prefer:
            by_stem[stem] = f
    imgs = [os.path.abspath(os.path.join(folder, f))
            for _, f in sorted(by_stem.items())]
    if not imgs:
        sys.exit(f"No .jpg/.png images found in: {folder}")
    return imgs


def extract_audio(src, dest, start):
    ext = os.path.splitext(dest)[1].lower()
    codec = AUDIO_CODECS.get(ext)
    if codec is None:
        sys.exit(f"--extract-only: use one of {', '.join(AUDIO_CODECS)}")
    cmd = ["ffmpeg", "-y"]
    if start:
        cmd += ["-ss", str(start)]
    cmd += ["-i", src, "-vn", *codec, dest]
    run(cmd)
    print(f"Done: {dest}")


def audio_args_and_filter(args, video_dur, audio_input_index):
    """Returns (extra_input_args, audio_filter, maps) for the final mux."""
    in_args = []
    if args.loop_audio:
        in_args += ["-stream_loop", "-1"]
    if args.audio_start:
        in_args += ["-ss", str(args.audio_start)]
    in_args += ["-i", args.audio]
    fade = args.audio_fade
    chain = [f"volume={args.audio_volume}"]
    if fade > 0:
        chain.append(f"afade=t=in:st=0:d={fade}")
        chain.append(f"afade=t=out:st={max(0, video_dur - fade):.3f}:d={fade}")
    chain.append("apad")     # silence-pad if audio shorter than video
    afilter = f"[{audio_input_index}:a]{','.join(chain)}[aout]"
    return in_args, afilter


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stills", nargs="?", default=None,
                    help="folder containing jpg/png stills")
    ap.add_argument("-o", "--output", default="stills_movie.mp4")
    ap.add_argument("--duration", type=float, default=4.0,
                    help="seconds per image (default 4)")
    ap.add_argument("--size", default="1920x1080",
                    help="output WxH (default 1920x1080)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--prefer", choices=["png", "jpg"], default="png",
                    help="which file to use when a still exists as both")
    ap.add_argument("--zoom", action="store_true",
                    help="gentle continuous Ken Burns zoom")
    ap.add_argument("--crossfade", type=float, default=0.0,
                    help="crossfade seconds between slides (e.g. 1.0); "
                         "not combinable with --zoom")
    # audio options
    ap.add_argument("--audio", default=None,
                    help="any media file to take audio from (mp4/mkv/mp3/wav/m4a...)")
    ap.add_argument("--audio-start", type=float, default=0.0,
                    help="seek this many seconds into the audio source")
    ap.add_argument("--audio-volume", type=float, default=1.0,
                    help="audio gain, 1.0 = unchanged")
    ap.add_argument("--audio-fade", type=float, default=2.0,
                    help="audio fade in/out seconds (0 = off)")
    ap.add_argument("--loop-audio", action="store_true",
                    help="loop the audio if shorter than the video")
    ap.add_argument("--extract-only", default=None, metavar="OUT.wav",
                    help="just rip audio from --audio to this file and exit")
    args = ap.parse_args()

    if args.extract_only:
        if not args.audio:
            sys.exit("--extract-only requires --audio <sourcefile>")
        extract_audio(args.audio, args.extract_only, args.audio_start)
        return

    if not args.stills:
        sys.exit("Provide a stills folder (or use --extract-only mode).")
    if args.zoom and args.crossfade > 0:
        sys.exit("Use either --zoom or --crossfade, not both.")
    try:
        w, h = (int(x) for x in args.size.lower().split("x"))
    except ValueError:
        sys.exit("--size must look like 1920x1080")

    imgs = collect_images(args.stills, args.prefer)
    n = len(imgs)
    fit = (f"scale={w}:{h}:force_original_aspect_ratio=decrease:flags=lanczos,"
           f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:black,setsar=1")
    print(f"{n} stills -> {args.output} ({args.size}, "
          f"{args.duration}s each)")

    if args.crossfade > 0:
        # Windows caps process command lines at ~32K chars; crossfade mode
        # puts every image on the command line. Estimate and fall back.
        est = sum(len(i) + 45 for i in imgs) + 85 * n + 3000
        if est > 28000:
            print(f"WARNING: {n} stills is too many for --crossfade "
                  "(Windows command-line limit). Falling back to concat "
                  "mode (hard cuts). Use --zoom for motion, or reduce the "
                  "still count to keep crossfades.")
            args.crossfade = 0.0

    if args.crossfade > 0:
        # one input per image, chained xfade transitions
        f = min(args.crossfade, args.duration / 2)
        video_dur = n * args.duration - (n - 1) * f
        cmd = ["ffmpeg", "-y"]
        for img in imgs:
            cmd += ["-loop", "1", "-t", str(args.duration),
                    "-framerate", str(args.fps), "-i", img]
        parts = [f"[{i}:v]{fit}[v{i}]" for i in range(n)]
        prev = "v0"
        for j in range(1, n):
            outl = f"x{j}" if j < n - 1 else "vout"
            offset = j * (args.duration - f)
            parts.append(f"[{prev}][v{j}]xfade=transition=fade:"
                         f"duration={f}:offset={offset:.3f}[{outl}]")
            prev = outl
        if n == 1:
            parts[-1] = f"[0:v]{fit}[vout]"
        fc = ";".join(parts)
        maps = ["-map", "[vout]"]
        audio_idx = n
    else:
        # concat demuxer (fast, any number of images)
        listfile = os.path.join(tempfile.gettempdir(), "stills_concat.txt")
        with open(listfile, "w", encoding="utf-8") as fh:
            for img in imgs:
                fh.write(f"file '{img.replace(os.sep, '/')}'\n")
                fh.write(f"duration {args.duration}\n")
            fh.write(f"file '{imgs[-1].replace(os.sep, '/')}'\n")  # quirk
        video_dur = n * args.duration
        vf = f"{fit},fps={args.fps}"
        if args.zoom:
            vf += (f",zoompan=z='min(zoom+0.0008,1.25)':d=1:"
                   f"s={w}x{h}:fps={args.fps}")
        cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", listfile]
        fc = f"[0:v]{vf}[vout]"
        maps = ["-map", "[vout]"]
        audio_idx = 1

    if args.audio:
        a_in, a_filter = audio_args_and_filter(args, video_dur, audio_idx)
        cmd += a_in
        fc += ";" + a_filter
        maps += ["-map", "[aout]", "-c:a", "aac", "-b:a", "192k"]

    cmd += ["-filter_complex", fc, *maps,
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-pix_fmt", "yuv420p", "-t", f"{video_dur:.3f}", args.output]
    run(cmd)
    print(f"\nDone: {args.output} ({video_dur:.1f} s, {n} slides"
          + (", with audio" if args.audio else ", silent") + ")")


if __name__ == "__main__":
    main()

# Local Video Event Mining Toolkit

Two Python scripts that scan local video files (MP4, MKV, AVI, MOV — anything ffmpeg can read) for moments matching a description, then export the results. Everything runs **100% locally** on your machine — no uploads, no API keys, no cloud, no per-minute fees. Both scripts work fine on CPU; a 2-hour movie scans in minutes.

| Script | Senses | Input | Output |
|---|---|---|---|
| `clip_audio_events.py` | **Hearing** — screams, crying, whispers, gunshots, 521 sound classes (YAMNet/AudioSet) | any video file | one compiled MP4 of matching clips + CSV detection log |
| `extract_visual_events.py` | **Sight** — any phrase you can type: "disturbing", "blood splatter", "foggy forest" (CLIP zero-shot) | any video file | high-res JPG + PNG stills (screen-filling, letterboxed) + CSV score log |

They pair naturally: the audio script finds *when something sounds bad*, the visual script finds *when something looks bad*, and ffmpeg glues the results into new videos.

---

## 1. Setup (Windows)

### 1.1 ffmpeg

Check if you already have it:

```powershell
ffmpeg -version
```

If not, install with any one of these:

```powershell
winget install Gyan.FFmpeg          # recommended
# or
choco install ffmpeg                # if you use Chocolatey
# or
scoop install ffmpeg                # if you use Scoop
```

Manual alternative: download the "full" build from <https://www.gyan.dev/ffmpeg/builds/>, extract, and add the `bin` folder to your PATH. Open a **new** PowerShell window afterward and re-run `ffmpeg -version`.

> Any ffmpeg from ~4.x onward works. Both scripts call `ffmpeg` by name, so it must be on PATH.

### 1.2 Python

You need Python 3.9–3.12 (3.10/3.11 recommended). Check what the Windows launcher sees:

```powershell
py --list
```

If nothing suitable: `winget install Python.Python.3.11`, then open a new window.

> **Tip:** on Windows, always invoke a specific interpreter with `py -3.10` (or `py -3.11`). Plain `pip`/`python` can silently point at a different install — the #1 cause of "module not found" errors right after a successful install.

### 1.3 Python packages

For the **audio** script:

```powershell
py -3.10 -m pip install tensorflow tensorflow-hub numpy soundfile
```

For the **visual** script:

```powershell
py -3.10 -m pip install torch torchvision open_clip_torch opencv-python pillow
```

(If you have an NVIDIA GPU and want faster visual scans, install the CUDA build of PyTorch instead — see <https://pytorch.org/get-started/locally/> — the script auto-detects it.)

### 1.4 One-liner sanity check

```powershell
ffmpeg -version | Select-Object -First 1; py -3.10 --version; py -3.10 -c "import tensorflow, tensorflow_hub, soundfile; print('audio deps OK')"; py -3.10 -c "import torch, open_clip, cv2, PIL; print('visual deps OK')"
```

All four lines printing = ready. (The TensorFlow import takes 10–30 s and prints oneDNN/deprecation warnings — normal.)

### 1.5 First-run model downloads

Each model downloads once, then is cached locally:

- YAMNet (audio): ~17 MB, cached in `%TEMP%\tfhub_modules`
- CLIP ViT-B-32 (visual): ~340 MB, cached in `%USERPROFILE%\.cache`

After the first run, both scripts work fully offline.

---

## 2. `clip_audio_events.py` — sound-event clip compiler

**What it does:** extracts the audio track, slides Google's YAMNet model over it in ~1-second windows, flags every window where a target sound class (matched by keyword against AudioSet's 521 class names) exceeds a confidence threshold, merges nearby hits into padded segments, cuts each segment from the original video, and concatenates them into one MP4. Also writes `<output>_detections.csv` with every hit (timestamp, class, confidence).

### Basic use

```powershell
py -3.10 clip_audio_events.py "movie.mp4" -o compilation.mp4
```

### Options

| Flag | Default | Meaning |
|---|---|---|
| `--keywords` | scream, yell, shout, crying, whisper, wail, moan, groan, gasp, … | substrings matched against YAMNet class names |
| `--threshold` | 0.25 | confidence 0–1; lower = more clips |
| `--pad` | 1.5 | seconds of context kept before/after each event |
| `--max-gap` | 2.0 | detections closer than this merge into one clip |
| `--min-len` | 1.0 | discard segments shorter than this |
| `--list-classes` | — | print all 521 detectable sound classes and exit |

### Tuning workflow

1. Run once with defaults.
2. Open `compilation_detections.csv` — see what was detected, when, and at what confidence.
3. Too much junk → raise `--threshold` to 0.35. Missing quiet moments (whispers score low) → drop to 0.15.
4. Clips feel chopped → raise `--pad` to 3.

### Creative uses

- **Horror jump-scare supercut** — the original use case: `--keywords scream shriek` and you have a trailer-ready scare reel.
- **Find every gunshot or explosion** in an action movie: `--keywords gunshot explosion artillery` (yes, those are real AudioSet classes — run `--list-classes` to browse all 521).
- **Laugh track extractor**: `--keywords laughter giggle chuckle` against a sitcom rip → instant blooper-reel energy.
- **Pet cameo finder**: `--keywords bark meow howl` against home videos → compilation of every moment your animal interrupted.
- **Music-only skim**: `--keywords music singing choir` with `--threshold 0.5` pulls just the musical numbers out of a long recording.
- **Baby-monitor / CCTV triage**: point it at an 8-hour overnight recording with `--keywords crying glass alarm` and review 90 seconds instead of 8 hours.
- **Sports highlights without watching**: `--keywords cheering applause crowd` on a full match recording → the crowd tells you where the goals are.
- **Silence inverse**: run with broad keywords and a low threshold, then use the CSV to find the *gaps* — the quiet stretches — for ambient/atmosphere sampling.

---

## 3. `extract_visual_events.py` — description-matched still extractor

**What it does:** samples one frame per second (configurable), scores each against your descriptive words using CLIP (a model that understands arbitrary English phrases — no fixed class list), suppresses near-duplicates with a minimum time gap, then re-grabs each winning timestamp from the original file at full quality. Each pick is exported as both a max-quality `.jpg` and a lossless `.png`, scaled and letterboxed to exactly your target resolution (default 1920×1080) so they fill the screen cleanly in a video. Writes `_scores.csv` with every sampled frame's score.

### Basic use

```powershell
py -3.10 extract_visual_events.py "movie.mp4" -o stills
```

### Options

| Flag | Default | Meaning |
|---|---|---|
| `--words` | disturbing, graphic, brutally violent, gross, disgusting, horrifying, terrifying, creepy, bloody, gruesome | any words/phrases — quote multi-word phrases |
| `--interval` | 1.0 | seconds between sampled frames (0.5 = finer, slower) |
| `--threshold` | 0.5 | match score 0–1; lower = more frames |
| `--min-gap` | 5.0 | min seconds between exported stills |
| `--top` | 0 (all) | keep only the N best frames |
| `--size` | 1920x1080 | output resolution; `3840x2160` for 4K |
| `-o` | stills | output folder |

Filenames embed order, timestamp, and matched word — `frame_0007_t01234.50_horrifying.jpg` — so they sort chronologically and you can trace every still back to its moment in the film.

### Tuning workflow

Because CLIP scores are *relative* (your words vs. built-in neutral prompts), the right threshold varies by movie. Run once, open `_scores.csv`, sort by score, and see where the real matches separate from the noise. `--top 40` is often easier than threshold-hunting: "just give me the 40 most horrifying frames."

### Creative uses

- **Horror poster hunting** — the original use case: pull the 30 most disturbing frames at 4K (`--top 30 --size 3840x2160`) for thumbnails, posters, or a creepy title sequence.
- **Anything you can phrase, you can find.** CLIP isn't limited to scary: `--words "neon city at night" "rain on a window" "foggy forest"` turns any film into a mood-board generator.
- **Cinematography study**: `--words "wide landscape shot" "extreme close-up of a face" "silhouette against light"` to extract a director's signature compositions.
- **Color-palette scenes**: `--words "scene bathed in red light" "cold blue scene"` — works surprisingly well for grading reference.
- **Family-archive face hunting**: `--words "a child blowing out birthday candles" "people hugging"` against old home videos.
- **Screenshot dataset building**: `--words "car" "dog" "computer screen"` with `--threshold 0.4 --min-gap 2` to mass-harvest labeled stills from footage.
- **Desktop wallpaper farm**: `--top 20 --size 3840x2160` with scenic words against a nature documentary.
- **Pair with the audio script**: run both, then cross-reference the two CSVs — moments that score high in *both* (sounds horrifying AND looks horrifying) are the film's true peaks. The timestamps line up because both are seconds-from-start.

---

## 4. Turning stills into video with ffmpeg

The stills are pre-sized and letterboxed, so assembly is trivial.

**Simple slideshow (4 s per image, 1080p):**

```powershell
ffmpeg -framerate 1/4 -pattern_type glob -i "stills/frame_*.jpg" -c:v libx264 -r 30 -pix_fmt yuv420p stills_reel.mp4
```

If your Windows ffmpeg build lacks glob support, build a concat list instead:

```powershell
Get-ChildItem stills\frame_*.jpg | Sort-Object Name | ForEach-Object { "file '$($_.FullName.Replace('\','/'))'"; "duration 4" } | Set-Content stills\list.txt; ffmpeg -f concat -safe 0 -i stills\list.txt -vf "fps=30,format=yuv420p" -c:v libx264 stills_reel.mp4
```

**Slideshow with slow Ken Burns zoom (cinematic):**

```powershell
ffmpeg -framerate 1/5 -pattern_type glob -i "stills/frame_*.jpg" -vf "zoompan=z='min(zoom+0.0008,1.2)':d=150:s=1920x1080:fps=30,format=yuv420p" -c:v libx264 stills_reel_zoom.mp4
```

**Stills reel + the audio compilation's soundtrack** (full circle — scary images set to the movie's own screams):

```powershell
ffmpeg -i stills_reel.mp4 -i compilation.mp4 -map 0:v -map 1:a -c:v copy -c:a aac -shortest final_nightmare_fuel.mp4
```

**Split a large video into chunks** (general utility, keyframe-aligned, no re-encode):

```powershell
ffmpeg -i movie.mp4 -c copy -map 0 -f segment -segment_time 600 -reset_timestamps 1 chunk_%03d.mp4
```

---

## 5. Troubleshooting

| Symptom | Fix |
|---|---|
| `Fatal error in launcher: Unable to create process` from `pip` | A stale Python's `Scripts` folder is on PATH. Bypass with `py -3.10 -m pip …`, and/or remove dead PATH entries. |
| `ModuleNotFoundError` right after a successful install | Packages went into a different Python. Always install and run with the same `py -3.X`. |
| TensorFlow oneDNN / deprecation warnings | Normal noise. Silence with `$env:TF_ENABLE_ONEDNN_OPTS=0` if it bothers you. |
| `No segments found` / `No frames above threshold` | Lower `--threshold`; check the CSV to see actual score ranges. |
| Audio script slow | It's mostly ffmpeg extraction + CPU inference; expect a few minutes for a 2-hour film. |
| Visual script slow | Raise `--interval` to 2.0 (halves the work), or install CUDA PyTorch if you have an NVIDIA GPU. |
| Out-of-sync clips in the compiled MP4 | The script re-encodes each clip, which normalizes most sync issues. If you still see drift (variable-frame-rate source), remux first: `ffmpeg -i movie.mp4 -c copy -video_track_timescale 90000 fixed.mp4`. |
| ffmpeg `glob` pattern error on Windows | Use the PowerShell concat-list one-liner in section 4. |

---

## 6. How the models work (30-second version)

**YAMNet** (audio) is a small neural net trained on AudioSet, Google's dataset of 2+ million labeled YouTube clips. It hears 0.96-second windows and outputs a confidence for each of 521 sound classes — "Screaming", "Whispering", "Crying, sobbing", "Gunshot, gunfire", "Meow"… You pick classes by keyword; the script does the rest.

**CLIP** (visual) was trained by OpenAI on 400 million image–caption pairs to place images and text in the same mathematical space. That means it can score how well *any English phrase* matches *any image* — no retraining, no fixed list. The script compares each frame against your words and against built-in "boring/neutral scene" anchors, and keeps frames where your words win.

Both are free, open, and run entirely on your machine.

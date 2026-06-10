# AVmods — Local Video Event Mining & Soundtrack Toolkit

Three Python scripts that turn any local video file (MP4, MKV, AVI, MOV — anything ffmpeg reads) into new creative material, **100% locally**: no uploads, no API keys, no cloud fees. Everything runs on CPU (GPU auto-detected and used if present).

| # | Script | Senses / Creates | Output |
|---|---|---|---|
| 1 | `clip_audio_events.py` | **Hearing** — screams, crying, whispers, gunshots… 521 sound classes (YAMNet/AudioSet) | one compiled MP4 of matching clips + detection CSV |
| 2 | `extract_visual_events.py` | **Sight** — any phrase you can type: "disturbing", "blood splatter", "foggy forest" (CLIP zero-shot) | high-res screen-filling JPG + PNG stills + score CSV |
| 3 | `make_soundtrack.py` | **Composing** — AI music matched to your keywords' vibe (Meta MusicGen) | WAV music bed, exact duration of your longest video |

Together they form a pipeline: **mine the sounds → mine the visuals → score the result** — a movie goes in, a fully AI-curated, AI-scored highlight reel comes out.

---

## 1. Setup (Windows)

### 1.1 ffmpeg

```powershell
ffmpeg -version    # already installed? skip ahead
winget install Gyan.FFmpeg          # recommended
# or: choco install ffmpeg   /   scoop install ffmpeg
```

Manual: grab the "full" build from <https://www.gyan.dev/ffmpeg/builds/>, extract, add the `bin` folder to PATH. Open a **new** PowerShell window and re-check. Any ffmpeg ≥ 4.x works; scripts 1 and 3 also use `ffprobe`, which ships in the same build.

### 1.2 Python

Python 3.9–3.12 (3.10/3.11 recommended). Check: `py --list`. If missing: `winget install Python.Python.3.11`.

> **Golden rule on Windows:** always install and run with an explicit interpreter — `py -3.10 -m pip install …` and `py -3.10 script.py`. Plain `pip`/`python` can silently point at a different Python install; that mismatch is the #1 cause of `ModuleNotFoundError` right after a "successful" install.

### 1.3 Python packages

```powershell
# Script 1 (audio events)
py -3.10 -m pip install tensorflow tensorflow-hub numpy soundfile

# Script 2 (visual stills)
py -3.10 -m pip install torch torchvision open_clip_torch opencv-python pillow

# Script 3 (soundtrack) — reuses torch from above
py -3.10 -m pip install transformers scipy
```

NVIDIA GPU? Install the CUDA build of PyTorch (<https://pytorch.org/get-started/locally/>) — scripts 2 and 3 auto-detect it and run many times faster.

**About pip "dependency conflict" errors:** if your Python has other projects installed (langchain, pandas, etc.), pip may print red `ERROR: ... requires numpy<2, but you have numpy 2.x` lines. As long as it ends with `Successfully installed …`, these scripts are fine — the warnings concern those *other* packages. The clean long-term fix is one virtual environment per project:

```powershell
py -3.10 -m venv C:\venvs\avmods
C:\venvs\avmods\Scripts\Activate.ps1
pip install tensorflow tensorflow-hub soundfile torch torchvision open_clip_torch opencv-python pillow transformers scipy numpy
```

### 1.4 One-liner sanity check

```powershell
ffmpeg -version | Select-Object -First 1; py -3.10 --version; py -3.10 -c "import tensorflow, tensorflow_hub, soundfile; print('audio deps OK')"; py -3.10 -c "import torch, open_clip, cv2, PIL; print('visual deps OK')"; py -3.10 -c "import transformers, scipy; print('music deps OK')"
```

Five lines printing = fully ready. (TensorFlow's import takes 10–30 s and prints oneDNN warnings — normal.)

### 1.5 First-run model downloads (then fully offline)

| Model | Used by | Size | Cache location |
|---|---|---|---|
| YAMNet | script 1 | ~17 MB | `%TEMP%\tfhub_modules` |
| CLIP ViT-B-32 | script 2 | ~340 MB | `%USERPROFILE%\.cache` |
| MusicGen small | script 3 | ~2 GB | `%USERPROFILE%\.cache\huggingface` |

---

## 2. `clip_audio_events.py` — sound-event clip compiler

Extracts the audio track, slides YAMNet over it in ~1 s windows, flags windows where a target sound class exceeds the confidence threshold, merges nearby hits into padded segments, cuts them from the original video, and concatenates everything into one MP4. Also writes `<output>_detections.csv` (timestamp, class, confidence).

```powershell
py -3.10 clip_audio_events.py "movie.mp4" -o compilation.mp4
```

| Flag | Default | Meaning |
|---|---|---|
| `--keywords` | scream, yell, shout, crying, whisper, wail, moan, groan, gasp… | substrings matched against YAMNet's 521 class names |
| `--threshold` | 0.25 | confidence 0–1; lower = more clips |
| `--pad` | 1.5 | seconds of context before/after each event |
| `--max-gap` | 2.0 | detections closer than this merge into one clip |
| `--min-len` | 1.0 | discard shorter segments |
| `--list-classes` | — | print all 521 sound classes and exit |

**Tuning:** run once → open the CSV → too much junk? raise threshold to 0.35; missing whispers (they score low)? drop to 0.15; clips feel chopped? `--pad 3`.

### Creative uses

- **Horror jump-scare supercut** — `--keywords scream shriek` → trailer-ready scare reel.
- **Action audit** — `--keywords gunshot explosion artillery` (real AudioSet classes; browse with `--list-classes`).
- **Laugh-track extractor** — `--keywords laughter giggle chuckle` on a sitcom → instant blooper energy.
- **Pet cameo finder** — `--keywords bark meow howl` on home videos.
- **Music-only skim** — `--keywords music singing choir --threshold 0.5` pulls musical numbers from long recordings.
- **CCTV/baby-monitor triage** — `--keywords crying glass alarm` on an 8-hour overnight file → review 90 seconds instead of 8 hours.
- **Sports highlights blind** — `--keywords cheering applause crowd`: the crowd tells you where the goals are.

---

## 3. `extract_visual_events.py` — description-matched still extractor

Samples one frame per second, scores each against your words using CLIP (understands arbitrary English — no fixed class list), suppresses near-duplicates via a minimum time gap, then re-grabs each winner from the original file at full quality. Exports max-quality `.jpg` + lossless `.png`, scaled and letterboxed to exactly your target resolution so they fill the screen in a video. Writes `_scores.csv` for every sampled frame.

```powershell
py -3.10 extract_visual_events.py "movie.mp4" -o stills
```

| Flag | Default | Meaning |
|---|---|---|
| `--words` | disturbing, graphic, brutally violent, gross, disgusting, horrifying, terrifying, creepy, bloody, gruesome | any words/phrases; quote multi-word phrases |
| `--interval` | 1.0 | seconds between sampled frames |
| `--threshold` | 0.5 | match score 0–1; lower = more frames |
| `--min-gap` | 5.0 | min seconds between exported stills |
| `--top` | 0 (all) | keep only the N best frames |
| `--size` | 1920x1080 | output resolution; `3840x2160` for 4K |

Filenames embed order + timestamp + matched word (`frame_0007_t01234.50_horrifying.jpg`) so they sort chronologically and trace back to the exact movie moment.

**Tuning:** CLIP scores are relative, so the right threshold varies per film. Often easier: skip threshold-hunting and use `--top 40` — "just give me the 40 most horrifying frames."

### Creative uses

- **Poster/thumbnail hunting** — `--top 30 --size 3840x2160` for the 30 most disturbing frames at 4K.
- **Mood-board generator** — `--words "neon city at night" "rain on a window" "foggy forest"`: anything you can phrase, you can find.
- **Cinematography study** — `--words "wide landscape shot" "extreme close-up of a face" "silhouette against light"` extracts a director's signature compositions.
- **Color-grading reference** — `--words "scene bathed in red light" "cold blue scene"`.
- **Family-archive search** — `--words "a child blowing out birthday candles" "people hugging"` on old home videos.
- **Dataset harvesting** — `--words car dog "computer screen" --threshold 0.4 --min-gap 2` mass-collects labeled stills.
- **Wallpaper farm** — scenic words + `--top 20 --size 3840x2160` against a nature documentary.

---

## 4. `make_soundtrack.py` — keyword-matched AI music bed

Measures your videos with ffprobe and targets the **longest** duration. Weaves your keywords into a music prompt (or take full control with `--prompt`), generates music with Meta's MusicGen in 30 s chunks, equal-power-crossfades them into a seamless bed, loops to fill the target, trims to the exact second, fades in/out, normalizes, and writes `soundtrack.wav` (32 kHz). Finishes by printing two ready-to-paste ffmpeg mix commands.

```powershell
py -3.10 make_soundtrack.py compilation.mp4 stills_reel.mp4 --keywords disturbing horrifying creepy
```

| Flag | Default | Meaning |
|---|---|---|
| `videos` | — | one or more files; target duration = longest |
| `--keywords` | disturbing, horrifying, creepy, tense, eerie | vibe words woven into the prompt |
| `--prompt` | (template) | full custom prompt, overrides keywords |
| `--model` | facebook/musicgen-small | `-medium`/`-large` = better music, slower, bigger download |
| `--gen-seconds` | 90 | unique material before looping; higher = less repetition |
| `--seed` | 42 | change for a completely different take |
| `-o` | soundtrack.wav | output file |

**Speed reality check:** on CPU each 30 s chunk takes ~2–5 min with `musicgen-small`. That's why the script generates 90 s of unique material and loops it for long targets. GPU makes it near-instant.

### Creative uses

- **Score your own supercut** — the intended pipeline (see section 5).
- **Re-score a scene** — feed any clip with `--prompt "upbeat 80s synthwave"` and watch a horror scene become a music video.
- **Ambient sleep/focus beds** — `--prompt "calm ambient pads, slow, warm, no percussion" --gen-seconds 120` against any long video gives an exact-length bed.
- **Trailer mockups** — `--prompt "epic trailer percussion, braams, rising tension"`.
- **A/B takes** — same command, different `--seed`, pick the best take.

---

## 5. The full pipeline — movie in, scored nightmare reel out

```powershell
# 1. Mine the audio: every scream/cry/whisper becomes a clip compilation
py -3.10 clip_audio_events.py "movie.mp4" -o compilation.mp4

# 2. Mine the visuals: the most horrifying frames as 1080p stills
py -3.10 extract_visual_events.py "movie.mp4" -o stills --top 40

# 3. Stills -> slideshow (4 s per image)
ffmpeg -framerate 1/4 -pattern_type glob -i "stills/frame_*.jpg" -c:v libx264 -r 30 -pix_fmt yuv420p stills_reel.mp4

# 4. Compose a soundtrack as long as the longest video
py -3.10 make_soundtrack.py compilation.mp4 stills_reel.mp4 --keywords disturbing horrifying creepy

# 5a. Mix music UNDER the screams (35% volume)
ffmpeg -i compilation.mp4 -i soundtrack.wav -filter_complex "[1:a]volume=0.35[m];[0:a][m]amix=inputs=2:duration=first:normalize=0" -c:v copy scored_compilation.mp4

# 5b. Or set the silent stills reel TO the music
ffmpeg -i stills_reel.mp4 -i soundtrack.wav -map 0:v -map 1:a -c:v copy -c:a aac -shortest scored_stills.mp4
```

**Bonus ffmpeg recipes:**

```powershell
# Ken Burns slow-zoom slideshow instead of static stills
ffmpeg -framerate 1/5 -pattern_type glob -i "stills/frame_*.jpg" -vf "zoompan=z='min(zoom+0.0008,1.2)':d=150:s=1920x1080:fps=30,format=yuv420p" -c:v libx264 stills_reel_zoom.mp4

# If your ffmpeg build lacks glob support (PowerShell concat-list fallback)
Get-ChildItem stills\frame_*.jpg | Sort-Object Name | ForEach-Object { "file '$($_.FullName.Replace('\','/'))'"; "duration 4" } | Set-Content stills\list.txt; ffmpeg -f concat -safe 0 -i stills\list.txt -vf "fps=30,format=yuv420p" -c:v libx264 stills_reel.mp4

# Split a huge video into 10-min chunks (no re-encode, keyframe-aligned)
ffmpeg -i movie.mp4 -c copy -map 0 -f segment -segment_time 600 -reset_timestamps 1 chunk_%03d.mp4
```

**Cross-referencing trick:** scripts 1 and 2 both log timestamps as seconds-from-start. Moments that score high in *both* CSVs — sounds horrifying AND looks horrifying — are the film's true peaks. Sort both by score, intersect the timestamps (±2 s), and you've found the scenes worth keeping.

---

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| `Fatal error in launcher: Unable to create process` from `pip` | A dead Python's `Scripts` folder is on PATH. Bypass with `py -3.10 -m pip …`; remove stale PATH entries when convenient. |
| `ModuleNotFoundError` right after a successful install | Packages went into a different Python. Install and run with the *same* `py -3.X`. |
| pip prints red `requires numpy<2 … incompatible` errors | Warnings about *other* packages in that Python; these scripts are unaffected. Use a venv per project if those other tools matter. |
| TensorFlow oneDNN / deprecation warnings on import | Normal. Silence with `$env:TF_ENABLE_ONEDNN_OPTS=0`. |
| `No segments found` / `No frames above threshold` | Lower `--threshold`; open the CSV to see the real score distribution. |
| Script 2 or 3 slow | Script 2: raise `--interval` to 2.0. Script 3: keep `musicgen-small`, lower `--gen-seconds`. Or install CUDA PyTorch. |
| MusicGen download fails mid-way | Re-run; Hugging Face resumes partial downloads. |
| Music too repetitive | Raise `--gen-seconds` (e.g. 150) or try a new `--seed`. |
| Music drowns the screams | Lower the `volume=0.35` value in the mix command. |
| ffmpeg `glob` error on Windows | Use the concat-list fallback in section 5. |
| Out-of-sync clips in compiled MP4 | Script re-encodes clips, which fixes most cases. Stubborn VFR source: `ffmpeg -i movie.mp4 -c copy -video_track_timescale 90000 fixed.mp4` first. |

---

## 7. The models in 30 seconds

**YAMNet** (audio in) — small neural net trained on AudioSet, Google's 2M+ labeled YouTube clips. Hears 0.96 s windows, outputs confidence for 521 sound classes ("Screaming", "Whispering", "Gunshot, gunfire", "Meow"…).

**CLIP** (vision in) — trained by OpenAI on 400M image–caption pairs to place images and text in one shared space, so it can score how well *any English phrase* matches *any frame* — no retraining, no fixed list.

**MusicGen** (audio out) — Meta's text-to-music transformer trained on 20k hours of licensed music. Describe a vibe, get a stereo-quality score, ~50 audio tokens per second of music.

All three are free, open, cached locally after first download, and never send your files anywhere.

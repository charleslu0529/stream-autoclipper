"""Phase 1: find candidate highlight windows in a VOD without any AI.

Signals (per 1-second bin):
  * audio  - loudness spikes vs. a rolling baseline (use the mic-only track if you have one)
  * chat   - message rate + funny-emote density (TwitchDownloaderCLI chat JSON)
  * marker - your own stream markers / hotkey bookmarks (optional text file)

Output: output/<vod name>/candidates.csv  (ranked), plus optional preview clips.

Usage (Git Bash):
  python src/signals.py "/d/Recordings/2026-10-20 20-01-33.mkv" \
      --chat "/d/Recordings/chat_2026-10-20.json" \
      --markers "/d/Recordings/markers.txt" \
      --previews 10

Requires ffmpeg/ffprobe on PATH and: pip install numpy pyyaml
"""
import argparse
import csv
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent

DEFAULTS = {
    "mic_track": None,       # audio stream index of the mic-only track (0 = first audio track). None = first track.
    "chat_lag_s": 6,         # chat reacts a few seconds after the moment; shift chat earlier by this much
    "pre_s": 40,             # seconds of lead-in before the peak (context for the AI judge)
    "post_s": 20,            # seconds after the peak
    "min_gap_s": 60,         # don't pick two peaks closer than this
    "top_n": 40,             # max candidates per VOD
    "min_score": 2.5,        # combined score threshold
    "weights": {"audio": 1.0, "chat": 1.5, "marker": 5.0},
    "funny_tokens": [
        "kekw", "kekl", "lul", "lulw", "omegalul", "icant", "pepelaugh", "lmao", "lmfao", "lol",
        "xd", "dead", "clip", "clipit", "💀", "😂", "🤣",
    ],
}

FUNNY_RE_EXTRA = re.compile(r"\b(?:ha){2,}\b|\bclip (?:it|that)\b", re.I)


# ---------------------------------------------------------------- config
def load_config() -> dict:
    cfg = {k: (v.copy() if isinstance(v, (dict, list)) else v) for k, v in DEFAULTS.items()}
    path = ROOT / "config.yaml"
    if path.exists():
        user = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("signals") or {}
        for k, v in user.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
    return cfg


# ---------------------------------------------------------------- audio
def probe_duration(vod: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(vod)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return float(out)


def audio_db_per_second(vod: Path, track, sr: int = 8000) -> np.ndarray:
    """Stream mono PCM out of ffmpeg and return RMS loudness (dBFS) per 1 s bin."""
    stream = f"0:a:{track if track is not None else 0}"
    cmd = ["ffmpeg", "-v", "error", "-i", str(vod), "-map", stream, "-ac", "1", "-ar", str(sr),
           "-f", "s16le", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    out, chunk_bytes = [], sr * 2  # 1 second of int16
    while True:
        buf = proc.stdout.read(chunk_bytes)
        if not buf:
            break
        x = np.frombuffer(buf, dtype=np.int16).astype(np.float32) / 32768.0
        rms = np.sqrt(np.mean(x * x)) if x.size else 0.0
        out.append(20 * np.log10(rms + 1e-6))
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed reading audio stream {stream} (wrong mic_track?)")
    return np.array(out, dtype=np.float32)


# ---------------------------------------------------------------- chat
def load_chat(path: Path, funny_tokens):
    """TwitchDownloaderCLI JSON -> list of (offset_seconds, text, is_funny)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    comments = data.get("comments", data if isinstance(data, list) else [])
    tokens = {t.lower() for t in funny_tokens}
    msgs = []
    for c in comments:
        t = c.get("content_offset_seconds")
        body = (c.get("message") or {}).get("body") if isinstance(c.get("message"), dict) else c.get("message")
        if t is None or not body:
            continue
        words = {w.lower() for w in re.findall(r"\S+", body)}
        funny = bool(words & tokens) or any(e in body for e in tokens if not e.isascii()) or bool(FUNNY_RE_EXTRA.search(body))
        msgs.append((float(t), body, funny))
    return msgs


def chat_series(msgs, n: int, lag: int) -> np.ndarray:
    s = np.zeros(n, dtype=np.float32)
    for t, _, funny in msgs:
        i = int(t) - lag
        if 0 <= i < n:
            s[i] += 1.0 + (2.0 if funny else 0.0)
    return np.convolve(s, np.ones(10), mode="same")  # 10 s rolling sum


# ---------------------------------------------------------------- markers
def load_markers(path: Path):
    """One timestamp per line: seconds, MM:SS or HH:MM:SS (extra text after it is ignored)."""
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\s*(\d+(?::\d{1,2}){0,2}(?:\.\d+)?)", line)
        if m:
            secs = 0.0
            for part in m.group(1).split(":"):
                secs = secs * 60 + float(part)
            out.append(secs)
    return out


# ---------------------------------------------------------------- scoring
def rolling_median(x: np.ndarray, win: int) -> np.ndarray:
    pad = win // 2
    xp = np.pad(x, pad, mode="edge")
    step = 5  # median every 5 s then interpolate -> fast on 6-hour VODs
    idx = np.arange(0, len(x), step)
    med = np.array([np.median(xp[i:i + win]) for i in idx])
    return np.interp(np.arange(len(x)), idx, med)


def audio_z(db: np.ndarray, baseline_win: int = 121) -> np.ndarray:
    """dB above the rolling baseline, in units of typical variation (floored at 2 dB)."""
    resid = db - rolling_median(db, baseline_win)
    scale = max(np.median(np.abs(resid - np.median(resid))) * 1.4826, 2.0)
    return np.clip(resid / scale, 0, None)


def chat_z(counts: np.ndarray, baseline_win: int = 301) -> np.ndarray:
    """Poisson-style z-score of the rolling chat count vs. its local baseline (works for quiet chats too)."""
    base = rolling_median(counts, baseline_win)
    return np.clip((counts - base) / np.sqrt(base + 4.0), 0, None)


def pick_peaks(score, min_gap, top_n, min_score):
    picked = []
    for i in np.argsort(score)[::-1]:
        if score[i] < min_score or len(picked) >= top_n:
            break
        if all(abs(i - p) >= min_gap for p in picked):
            picked.append(int(i))
    return picked


def hms(s: float) -> str:
    s = int(s)
    return f"{s // 3600:d}:{s % 3600 // 60:02d}:{s % 60:02d}"


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("vod", type=Path)
    ap.add_argument("--chat", type=Path, help="TwitchDownloaderCLI chat JSON")
    ap.add_argument("--markers", type=Path, help="text file of timestamps")
    ap.add_argument("--previews", type=int, default=0, help="cut quick preview clips for the top N")
    ap.add_argument("--out", type=Path, default=ROOT / "output")
    args = ap.parse_args()
    cfg = load_config()
    w = cfg["weights"]

    dur = probe_duration(args.vod)
    print(f"VOD {args.vod.name}: {hms(dur)}")

    print("audio: analysing loudness ...")
    db = audio_db_per_second(args.vod, cfg["mic_track"])
    n = len(db)
    a_z = audio_z(db)
    a_z = np.maximum.reduce([np.roll(a_z, k) for k in (-2, -1, 0, 1, 2)])  # widen spikes a little

    c_z, msgs = np.zeros(n), []
    if args.chat:
        msgs = load_chat(args.chat, cfg["funny_tokens"])
        print(f"chat: {len(msgs)} messages, {sum(f for *_, f in msgs)} funny")
        c_z = chat_z(chat_series(msgs, n, int(cfg["chat_lag_s"])))

    marker = np.zeros(n)
    markers = load_markers(args.markers) if args.markers else []
    for m in markers:
        lo, hi = max(0, int(m) - 30), min(n, int(m) + 5)  # marker is usually pressed just after the moment
        marker[lo:hi] = 1.0

    score = (w["audio"] * np.minimum(a_z, 6) + w["chat"] * np.minimum(c_z, 6) + w["marker"] * marker)
    if not args.chat:
        score = score * 1.5  # keep thresholds comparable when audio is the only signal
    peaks = pick_peaks(score, int(cfg["min_gap_s"]), int(cfg["top_n"]), float(cfg["min_score"]))

    out_dir = args.out / args.vod.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for rank, p in enumerate(peaks, 1):
        start, end = max(0, p - cfg["pre_s"]), min(n, p + cfg["post_s"])
        lag = cfg["chat_lag_s"]
        window_msgs = [b for t, b, _ in msgs if start + lag <= t <= end + lag]
        top_words = Counter(wd for b in window_msgs for wd in re.findall(r"\S+", b)).most_common(4)
        rows.append({
            "rank": rank, "peak": hms(p), "start_s": start, "end_s": end,
            "score": round(float(score[p]), 2), "audio_z": round(float(a_z[p]), 2),
            "chat_z": round(float(c_z[p]), 2), "marker": int(marker[p] > 0),
            "chat_msgs": len(window_msgs), "top_chat": " ".join(f"{wd}x{c}" for wd, c in top_words),
        })

    csv_path = out_dir / "candidates.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["rank"])
        wr.writeheader()
        wr.writerows(rows)

    print(f"\n{len(rows)} candidates -> {csv_path}")
    for r in rows[:15]:
        print(f"  #{r['rank']:>2} {r['peak']}  score {r['score']:>5}  audio {r['audio_z']:>4}  "
              f"chat {r['chat_z']:>4}  {'M ' if r['marker'] else '  '}{r['top_chat']}")

    for r in rows[: args.previews]:
        clip = out_dir / f"preview_{r['rank']:02d}_{r['peak'].replace(':', '-')}.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(r["start_s"]), "-t", str(r["end_s"] - r["start_s"]),
                        "-i", str(args.vod), "-map", "0:v:0", "-map", "0:a:0", "-c", "copy", str(clip)])
    if args.previews:
        print(f"previews written to {out_dir}")


if __name__ == "__main__":
    main()

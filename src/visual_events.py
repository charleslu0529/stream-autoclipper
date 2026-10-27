"""Detect game-specific on-screen events in a VOD.

Two detectors, both configured per game in assets/<game>/profile.yaml:

* killfeed  (Valorant) - spots the III / IV / V kill-count badge the game shows next to YOUR kill-feed
            entry and emits 3K / 4K / ACE events. Works with any agent, weapon skin and Riot name.
* templates - matches banner images (e.g. a round-end CLUTCH banner):
            <event>.png, with variants as <event>_2.png, <event>_alt.png ...

Templates must be cut from frames normalised the same way as the matcher sees them; grab one with:
    python src/visual_events.py grab "/d/vods/match.mp4" --at 0:31:14
then crop the banner in any image editor and save it as assets/valorant/<event>.png

Detect only (writes output/<vod>/events.csv):
    python src/visual_events.py detect "/d/vods/match.mp4" --game valorant
"""
import argparse
import csv
import subprocess
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"


def load_profile(game: str) -> tuple[dict, dict]:
    """Return (profile settings, {event: [template arrays]}). Events without templates are skipped."""
    pdir = ASSETS / game
    profile = yaml.safe_load((pdir / "profile.yaml").read_text(encoding="utf-8"))
    templates = {}
    for event in profile.get("events") or {}:
        files = sorted(pdir.glob(f"{event}.png")) + sorted(pdir.glob(f"{event}_*.png"))
        imgs = [cv2.imread(str(f), cv2.IMREAD_GRAYSCALE) for f in files]
        imgs = [i for i in imgs if i is not None]
        if imgs:
            templates[event] = imgs
    return profile, templates


def frames(vod: Path, fps: float, width: int, height: int, start: float = 0, duration: float | None = None,
           crop: tuple[int, int, int, int] | None = None, color: bool = False, hwaccel: str | None = None):
    """Yield (t_seconds, frame) at `fps`, scaled to width x height, optionally cropped to (x0, y0, x1, y1).
    Frames are grayscale, or BGR when color=True. hwaccel (e.g. "auto", "d3d11va", "cuda") offloads
    video decoding, which is the slow part, to the GPU."""
    cmd = ["ffmpeg", "-v", "error"] + (["-hwaccel", hwaccel] if hwaccel else [])
    if start:
        cmd += ["-ss", str(start)]
    cmd += ["-i", str(vod)]
    if duration:
        cmd += ["-t", str(duration)]
    vf = f"fps={fps},scale={width}:{height}"
    if crop:
        x0, y0, x1, y1 = crop
        vf += f",crop={x1 - x0}:{y1 - y0}:{x0}:{y0}"
        width, height = x1 - x0, y1 - y0
    cmd += ["-vf", vf, "-pix_fmt", "bgr24" if color else "gray", "-f", "rawvideo", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    shape = (height, width, 3) if color else (height, width)
    size, i = int(np.prod(shape)), 0
    while True:
        buf = proc.stdout.read(size)
        if len(buf) < size:
            break
        yield start + i / fps, np.frombuffer(buf, dtype=np.uint8).reshape(shape)
        i += 1
    proc.wait()


def region_px(region, w, h):
    x0, y0, x1, y1 = region
    return int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)


# ---------------------------------------------------------------- kill feed (Valorant)
# Valorant puts a teal badge with a roman numeral (III / IV / V) next to YOUR kill-feed entry when
# it's your 3rd / 4th / 5th kill of the round. That's independent of agent, weapon skin and Riot name.
BADGE_EVENTS = {"3k": "badge_3k.png", "4k": "badge_4k.png", "ace": "badge_ace.png"}


def load_badges(game: str) -> dict:
    out = {}
    for event, fname in BADGE_EVENTS.items():
        img = cv2.imread(str(ASSETS / game / fname), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            out[event] = img
    return out


def badges_in_frame(feed_bgr: np.ndarray, badges: dict, threshold: float = 0.78) -> list[tuple[str, float, int, int]]:
    """(event, confidence, x, y) for each kill-count badge visible in a kill-feed crop."""
    gray = cv2.cvtColor(feed_bgr, cv2.COLOR_BGR2GRAY)
    teal = cv2.inRange(cv2.cvtColor(feed_bgr, cv2.COLOR_BGR2HSV), (70, 60, 90), (100, 255, 255))
    found = []
    for event, tpl in badges.items():
        th, tw = tpl.shape
        r = cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED)
        while True:
            _, conf, _, (x, y) = cv2.minMaxLoc(r)
            if conf < threshold:
                break
            r[max(0, y - 8):y + 8, max(0, x - 8):x + 8] = -1
            ring = teal[max(0, y - 5):y + th + 5, max(0, x - 5):x + tw + 5]
            if ring.mean() / 255 >= 0.25:  # numeral must sit on the teal badge, not random white pixels
                found.append((event, round(conf, 3), x, y))
    found.sort(key=lambda f: -f[1])
    keep = []  # III and IV overlap a little: keep the best template per location
    for f in found:
        if all(abs(f[2] - k[2]) > 8 or abs(f[3] - k[3]) > 8 for k in keep):
            keep.append(f)
    return keep


def detect_killfeed(vod: Path, game: str, cfg: dict) -> list[dict]:
    badges = load_badges(game)
    if not badges:
        print(f"[{game}] killfeed: no badge templates - skipping")
        return []
    fps, thr = float(cfg.get("fps", 4)), float(cfg.get("threshold", 0.78))
    scores, cooldown = cfg.get("scores", {}), float(cfg.get("cooldown_s", 8))
    print(f"[{game}] killfeed: scanning for 3K / 4K / ACE badges at {fps} fps ...")
    events, last = [], {}
    for t, feed in frames(vod, fps, 1920, 1080, crop=tuple(cfg["region"]), color=True, hwaccel=cfg.get("hwaccel")):
        for event, conf, _, _ in badges_in_frame(feed, badges, thr):
            if t - last.get(event, -1e9) >= cooldown:
                events.append({"t": round(t, 1), "event": event, "confidence": conf,
                               "score": float(scores.get(event, 3))})
            last[event] = t
    return events


# ---------------------------------------------------------------- detect (both)
def detect(vod: Path, game: str, progress: bool = True) -> list[dict]:
    profile, templates = load_profile(game)
    events = []
    kf = profile.get("killfeed") or {}
    if kf.get("enabled"):
        events += detect_killfeed(vod, game, kf)
    if not templates:
        return suppress_lesser(events, profile)
    fps, w = float(profile.get("fps", 2)), int(profile.get("width", 960))
    h = w * 9 // 16
    cooldown = float(profile.get("cooldown_s", 4))
    print(f"[{game}] matching {', '.join(templates)} at {fps} fps ...")

    last_hit = {}
    for t, frame in frames(vod, fps, w, h):
        if progress and int(t) % 600 == 0 and t == int(t):
            print(f"  {int(t) // 60} min ...")
        for name, imgs in templates.items():
            if t - last_hit.get(name, -1e9) < cooldown:
                continue
            cfg = profile["events"][name]
            x0, y0, x1, y1 = region_px(cfg.get("region", [0, 0, 1, 1]), w, h)
            roi = frame[y0:y1, x0:x1]
            best = 0.0
            for tpl in imgs:
                if tpl.shape[0] > roi.shape[0] or tpl.shape[1] > roi.shape[1]:
                    continue
                best = max(best, float(cv2.matchTemplate(roi, tpl, cv2.TM_CCOEFF_NORMED).max()))
            if best >= float(cfg.get("threshold", 0.75)):
                events.append({"t": round(t, 1), "event": name, "confidence": round(best, 3),
                               "score": float(cfg.get("score", 3))})
                last_hit[name] = t
    return suppress_lesser(events, profile)


def suppress_lesser(events: list[dict], profile: dict) -> list[dict]:
    """If e.g. 3K, 4K and ACE all fire within one round, keep only the biggest (profile 'supersedes')."""
    sup = profile.get("supersedes", {})  # e.g. {ace: [4k, 3k], 4k: [3k]}
    window = float(profile.get("supersede_window_s", 20))
    keep = []
    for e in events:
        dominated = any(o["event"] in sup and e["event"] in sup[o["event"]] and abs(o["t"] - e["t"]) <= window
                        for o in events)
        if not dominated:
            keep.append(e)
    return keep


def event_series(events: list[dict], n: int, before: int = 6, after: int = 3) -> np.ndarray:
    """Spread each event's score over [t-before, t+after] seconds (banners appear just after the play)."""
    s = np.zeros(n)
    for e in events:
        lo, hi = max(0, int(e["t"]) - before), min(n, int(e["t"]) + after)
        s[lo:hi] = np.maximum(s[lo:hi], e["score"])
    return s


def grab(vod: Path, at: str, game: str, out_dir: Path):
    secs = 0.0
    for part in at.split(":"):
        secs = secs * 60 + float(part)
    profile, _ = load_profile(game)
    w = int(profile.get("width", 960))
    out_dir.mkdir(parents=True, exist_ok=True)
    for t, frame in frames(vod, 4, w, w * 9 // 16, start=max(0, secs - 1), duration=3):
        path = out_dir / f"frame_{t:.2f}s.png".replace(":", "-")
        cv2.imwrite(str(path), frame)
    print(f"frames saved to {out_dir} - crop the banner from one and save as assets/{game}/<event>.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("grab", help="save normalised frames around a timestamp for cutting templates")
    g.add_argument("vod", type=Path)
    g.add_argument("--at", required=True, help="timestamp, e.g. 0:31:14")
    g.add_argument("--game", default="valorant")
    d = sub.add_parser("detect", help="find events and write events.csv")
    d.add_argument("vod", type=Path)
    d.add_argument("--game", default="valorant")
    args = ap.parse_args()
    if not args.vod.is_file():
        raise SystemExit(f"VOD file not found: {args.vod}")

    out = ROOT / "output" / args.vod.stem
    if args.cmd == "grab":
        grab(args.vod, args.at, args.game, out / "grab")
        return
    events = detect(args.vod, args.game)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "events.csv").open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["t", "event", "confidence", "score"])
        wr.writeheader()
        wr.writerows(events)
    print(f"{len(events)} events -> {out / 'events.csv'}")
    for e in events:
        s = int(e["t"])
        print(f"  {s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}  {e['event']:<8} {e['confidence']}")


if __name__ == "__main__":
    main()

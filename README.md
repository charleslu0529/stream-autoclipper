# stream-autoclipper

Automatically finds the funny moments in my livestream VODs, cuts them into captioned vertical clips, and (eventually) publishes them to Instagram Reels, YouTube Shorts and TikTok. It runs on **free AI models** routed through a local [OmniRoute](https://github.com/diegosouzapw/OmniRoute) gateway, so switching to a paid model later is a config change, not a code change.

> 🚧 Work in progress. Phases 0–1 are done; see the [roadmap](#roadmap).

<!-- TODO: demo GIF: VOD in → ranked candidates → captioned vertical clip out -->

## How it works

```
OBS recording + Twitch chat log
   │
   ▼
[1] Signal pass (local, no AI)    audio loudness spikes · chat velocity + emote density · manual markers
   │                              → ranked candidate windows
   ▼
[2] Transcribe                    faster-whisper on candidate windows only
   ▼
[3] AI judge (via OmniRoute)      text model scores transcript + chat; vision model checks keyframes
   ▼
[4] Auto-edit (ffmpeg)            trim · 9:16 facecam/gameplay layout · burned-in captions
   ▼
[5] Review UI                     approve / reject / re-trim; decisions feed back into tuning
   ▼
[6] Publish                       per-platform adapters + scheduler (IG Reels, YT Shorts, TikTok)
```

### Design decisions

- **Cheap filtering before AI.** Free-tier models are rate-limited, so the local signal pass narrows a multi-hour VOD down to ~40 candidate windows before any model is called. That keeps free usage viable and paid costs low.
- **Gateway, not SDK lock-in.** The app only calls two OmniRoute *combos* (`clipper-text`, `clipper-vision`), which are fallback chains of models. Reordering the chain or adding a paid model happens in the gateway.
- **Robust statistics.** Loudness is scored against a rolling median baseline (MAD-scaled) and chat against a Poisson-style baseline, so quiet streams and busy streams both produce sensible candidates.
- **Adapters for publishing.** Each platform sits behind the same interface, so one platform's API changes don't break the others.

## Quick start

Requirements: Python 3.11+, [ffmpeg](https://ffmpeg.org/) on `PATH`, OmniRoute running locally.

```bash
pip install -r requirements.txt
cp config.example.yaml config.yaml        # add your OmniRoute key

# 1. check the AI gateway returns valid judge JSON
export OMNIROUTE_KEY="..."
python src/smoke_test.py "/path/to/screenshot.jpg"

# 2. find candidate highlights in a VOD
TwitchDownloaderCLI chatdownload --id <vod_id> -o chat.json
python src/signals.py "/path/to/recording.mkv" --chat chat.json --previews 10
```

Output: `output/<vod>/candidates.csv` plus rough preview clips of the top picks.

### Game events (Valorant)

`src/visual_events.py` scans the kill feed for the **III / IV / V badge** Valorant shows next to *your* entry on your 3rd / 4th / 5th kill of a round, and feeds 3K / 4K / ACE events into the ranking. It doesn't depend on agent, weapon skin or Riot name. Optional banner templates (e.g. CLUTCH) can be added per game in `assets/<game>/`.

```bash
python src/visual_events.py detect "/path/to/vod.mp4" --game valorant        # events only
python src/signals.py "/path/to/vod.mp4" --chat chat.json --game valorant    # full ranking
```

Decoding 1080p60 video is the slow part; `hwaccel: auto` in the profile uses the GPU decoder.

Tip: record your mic on its own OBS audio track and set `signals.mic_track`. Otherwise game audio (gunfire, explosions) shows up as "excitement".

## Tests

```bash
pytest
```

## Roadmap

- [x] Phase 0: OmniRoute gateway + combos, JSON smoke test
- [x] Phase 1: local signal pass (audio, chat, markers) → ranked candidates
- [ ] Phase 2: transcription + AI judge
- [ ] Phase 3: auto-edit (vertical layout, captions)
- [ ] Phase 4: review UI
- [ ] Phase 5: fully automatic on new recordings
- [ ] Phase 6: eval set; free vs paid model comparison
- [ ] Phase 7: auto-publishing (YT Shorts → IG Reels → TikTok)

Full plan: [PLAN.md](PLAN.md)

## License

MIT

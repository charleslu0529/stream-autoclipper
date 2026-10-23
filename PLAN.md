# Stream Auto-Clipper — Plan

Goal: after each stream, automatically find the funny moments, cut them into short clips (horizontal + vertical 9:16 with captions), and drop them into a review queue. All AI calls go through **OmniRoute** so free models do the work now and a paid model can be swapped in later with a config change — no code changes.

Project folder: `D:\Dev\Claude\stream-autoclipper\`

---

## 1. How it works (pipeline)

```
OBS recording (.mkv) + Twitch chat log
        │
        ▼
[1] Ingest ── watch folder picks up new VOD, remuxes to .mp4, extracts audio
        │
        ▼
[2] Cheap signal pass (no AI, all local)
     • audio loudness spikes / laughter (ffmpeg ebur128 or librosa)
     • chat velocity + funny-emote density (KEKW, LUL, OMEGALUL, "LMAO", clip requests)
     • manual markers (OBS hotkey / Twitch stream markers) from your bookmarking pipeline
        │  → list of candidate windows (e.g. 60–90 s each), ranked
        ▼
[3] Transcribe ── faster-whisper locally on GPU, word-level timestamps
        │
        ▼
[4] AI judge (via OmniRoute)
     • text model: transcript + chat for each window → JSON
       { funny_score 0-10, category, title, hook_line, start, end, reason }
     • optional vision model: 4–6 keyframes per window for visual gags
       (deaths, fails, ragdolls) that the transcript misses
        │  → keep score ≥ threshold, de-duplicate overlapping clips
        ▼
[5] Auto-edit (ffmpeg)
     • trim to AI-chosen in/out points (+ small padding)
     • 16:9 clip, and 9:16 vertical: facecam on top, gameplay crop below
     • burned-in captions from whisper words (ASS subtitles, word pop highlight)
     • optional: punch-in zoom on the peak moment, cut dead air
        │
        ▼
[6] Review queue ── small local web page: watch, approve / reject / re-trim, edit title
        │  approvals + rejections saved → used to tune prompts & thresholds
        ▼
[7] Publish ── approved clips → AI writes per-platform caption/hashtags → scheduler queue
                → Instagram Reels / YouTube Shorts / TikTok / others via official APIs (see section 7)
```

Key idea: **steps 2–3 are free and local and do most of the filtering.** The LLM only sees the top ~20–40 candidate windows per stream, which keeps free-tier rate limits and future paid costs small.

---

## 2. OmniRoute setup

1. Install: `npm i -g omniroute` then `omniroute` → gateway on `http://localhost:20128/v1` (OpenAI-compatible). Docker alternative: `docker run -p 20128:20128 -v omniroute-data:/app/data diegosouzapw/omniroute`.
2. In the dashboard, add free providers (it ships with some keyless ones; add your Gemini free key too since you already have one).
3. Create two **combos** (fallback chains) — the app only ever calls these names:
   - `clipper-text` — free reasoning/chat models in fallback order
   - `clipper-vision` — free vision-capable models (Gemini free tier is the most reliable bet; local Ollama + LLaVA as last resort)
4. Check limits with `omniroute quota` after a test run.

App config (`config.yaml`):
```yaml
llm:
  base_url: http://localhost:20128/v1
  api_key: ${OMNIROUTE_KEY}
  text_model: clipper-text      # later: point combo at the paid model
  vision_model: clipper-vision
  max_concurrency: 2            # stay friendly to free-tier rate limits
  retries: 3
```

Switching to paid later = edit the `clipper-text` combo in OmniRoute so the paid model is first (keep free models as fallback). Zero code changes.

Notes:
- Free models vary in JSON reliability → validate every response with pydantic, retry once with "return valid JSON only", then fall to the next model.
- Cache AI responses by (VOD hash, window, prompt version) so re-runs don't burn quota.
- Free providers may log prompts; fine here since stream content is public anyway, but don't send anything private through them.

---

## 3. Tech stack

| Part | Choice |
|---|---|
| Language | Python 3.11+ (orchestration), small front-end for review page |
| Video/audio | ffmpeg (cuts, crops, captions, loudness) |
| Transcription | faster-whisper (large-v3 or medium on GPU) |
| Audio signals | ffmpeg `ebur128` / librosa; optional laughter classifier later |
| Chat | Twitch chat log via TwitchDownloader CLI (or your live chat logger) |
| LLM client | `openai` Python SDK pointed at OmniRoute |
| Validation | pydantic |
| Storage | SQLite (candidates, scores, review decisions) |
| Review UI | Vite + React/vanilla page served locally, reads SQLite through a tiny FastAPI |
| Trigger | folder watcher (watchdog) on the OBS recording folder |

---

## 4. Folder layout

```
D:\Dev\Claude\stream-autoclipper\
  PLAN.md
  config.yaml
  prompts\            judge_text.md, judge_vision.md (versioned)
  src\
    ingest.py         watch folder, remux, extract audio
    signals.py        loudness, chat velocity, markers → candidates
    transcribe.py     faster-whisper
    judge.py          OmniRoute calls, JSON validation, caching
    edit.py           ffmpeg cuts, 9:16 layout, captions
    review_api.py     FastAPI for the review page
    publish\
      scheduler.py    queue, time slots, retries, status tracking
      captions.py     per-platform caption/hashtag writer (via OmniRoute)
      instagram.py    Reels via Instagram Graph API
      youtube.py      Shorts via YouTube Data API
      tiktok.py       TikTok Content Posting API
  secrets\            OAuth tokens (gitignored)
  review-ui\          front-end
  data\               sqlite db, caches
  output\<stream-date>\  raw.mp4, vertical.mp4, captions.ass, meta.json per clip
  eval\               labelled moments for model comparison
```

---

## 5. Build phases

**Phase 0 — Gateway (1 evening)**
OmniRoute installed, two combos made, a Python script gets valid JSON back from both.

**Phase 1 — Find moments, no AI (weekend)**
Ingest + loudness + chat velocity + markers → CSV of candidate timestamps. Spot-check against a stream you remember. This alone is already useful.

**Phase 2 — Transcribe + AI judge (weekend)**
Whisper on candidate windows only, `judge.py` scores them, writes to SQLite. Tune the prompt per genre (FPS rage/fails vs Rocket League whiffs vs JRPG commentary bits) — pass the game name/genre into the prompt.

**Phase 3 — Auto-edit (weekend)**
ffmpeg trims, vertical layout preset per scene layout (facecam position), burned captions. Output folder per stream.

**Phase 4 — Review page (1–2 evenings)**
Grid of clips, approve/reject/re-trim, editable title. Decisions saved.

**Phase 5 — Full automation**
Folder watcher runs the whole chain when OBS finishes recording; desktop notification when clips are ready.

**Phase 6 — Evaluate & switch to paid**
See section 6.

**Phase 7 — Auto-publishing**
See section 7. Start with YouTube Shorts (easiest API), then Instagram Reels, then TikTok.

**Later ideas:** auto thumbnail frame pick, sound-effect/meme stingers, compilation videos ("best of the week").

---

## 6. When to switch from free to paid

Build a small eval set first so the switch is a measured decision, not a guess:
1. From 3–5 past streams, hand-label ~50 windows: funny / not funny (+ ideal in/out points).
2. Run the judge with the free combo and with the candidate paid model.
3. Compare: precision of the top 10 picks, how often you'd approve the clip, JSON failure rate, latency, cost per stream.

Switch if the paid model gives clearly more approvable clips per stream (e.g. +30%) or the free chain keeps hitting rate limits / bad JSON. Keep free models as fallback in the combo either way.

Review-page approvals keep growing this eval set over time.

---

## 7. Auto-publishing to Instagram and other platforms

### 7.1 Flow

```
Review page: Approve ✔
   │
   ▼
captions.py ── OmniRoute text model writes per-platform copy from the clip's title, transcript, game:
   │            IG caption + 5–10 hashtags · YT Shorts title (≤100 chars) + description · TikTok caption
   │            (shown on the review page so you can edit before it goes out)
   ▼
scheduler.py ── puts the clip into a publish queue with a time slot per platform
   │             (e.g. 1 clip/day per platform at your best-performing hours; no 10-clip dumps)
   ▼
platform adapters ── upload → poll processing status → publish → save post ID + URL in SQLite
   │
   ▼
Desktop notification / Discord webhook: "Posted to IG, YT, TikTok" (or the error)
```

One clip → one row per platform in a `posts` table: `clip_id, platform, caption, scheduled_at, status (queued/uploading/processing/published/failed), remote_id, url, error, attempts`. Failed posts retry with backoff; after 3 failures they show up on the review page.

The scheduler can just run every 10 minutes via Windows Task Scheduler (or inside the watcher process) and publish whatever is due.

### 7.2 Instagram Reels (Instagram Graph API)

Requirements:
- Instagram account switched to **Professional (Creator or Business)** — free, in the IG app settings.
- A Meta developer app. Two login options:
  - **Instagram Login** (simplest, no Facebook Page needed): permissions `instagram_business_basic` + `instagram_business_content_publish`. Catch: video must be at a **public URL** Meta can fetch (`video_url`).
  - **Facebook Login for Business** (IG account linked to a Facebook Page): permissions `instagram_basic`, `instagram_content_publish`, `pages_read_engagement`. Bonus: supports **resumable upload of the local file**, so no public hosting needed.
- Since you're posting only to your own account, the app can stay in development mode with you as the tester — no app review needed for personal use.

Publish steps:
1. `POST /<IG_ID>/media` with `media_type=REELS`, `video_url` (or `upload_type=resumable` then upload to `rupload.facebook.com`), `caption`, `share_to_feed=true`.
2. Poll `GET /<container_id>?fields=status_code` about once a minute until `FINISHED` (or `ERROR`).
3. `POST /<IG_ID>/media_publish` with `creation_id=<container_id>`. Containers expire after 24 h.
- Limit: ~100 API posts per rolling 24 h (check `/<IG_ID>/content_publishing_limit`) — way more than you need.
- Long-lived tokens expire after ~60 days → the scheduler refreshes them automatically and warns you if refresh fails.

If using `video_url`: upload the vertical clip to cheap object storage (Cloudflare R2 has a free tier) with a short-lived presigned URL, publish, then delete the object.

Specs to target for the 9:16 export: 1080×1920, H.264 + AAC, 30 fps, under 90 s for the safest Reels reach.

### 7.3 YouTube Shorts (YouTube Data API v3)

- Google Cloud project → enable YouTube Data API → OAuth client (Desktop app) → authorize once, store refresh token.
- `videos.insert` with resumable upload. Shorts = vertical + ≤ 3 min; add `#Shorts` to title/description to be safe.
- Set `status.publishAt` + `privacyStatus=private` to let YouTube handle scheduling itself.
- Upload quota is limited per day (plenty for a few clips/day). Google has changed the rules on unverified apps and quota before — check the current `videos.insert` docs when building, and if uploads come out locked to private, apply for the API audit.
- Easiest platform → build this adapter first to prove the whole publish pipeline.

### 7.4 TikTok (Content Posting API)

- TikTok developer app with Login Kit + Content Posting API.
- **Catch:** until the app passes TikTok's audit, anything it posts is **private only**. Two paths:
  - **Upload to drafts/inbox (recommended for personal use):** API sends the clip to your TikTok inbox; you tap to finish & post from the phone (and can add trending sounds there, which the API can't do). Low friction, no audit needed.
  - **Direct Post:** fully hands-off, but requires passing the audit (they expect a real app with UX that follows their content-sharing guidelines).
- Upload via `FILE_UPLOAD` (send the local file in chunks) — avoids domain verification needed for `PULL_FROM_URL`.
- Poll the post status endpoint with the `publish_id`.

### 7.5 Other platforms (optional)

| Platform | Route | Notes |
|---|---|---|
| Facebook Reels | Graph API on your Page (`/<page_id>/video_reels`) | Reuses the same Meta app as IG |
| X / Twitter | API v2 media upload | Paid API tier needed for meaningful posting volume — likely skip |
| Twitch | No clip upload API | Link the YT/IG clips in Discord / socials instead |
| Discord | Webhook | Free and instant: post every published clip to your server |

### 7.6 Shortcut: use a scheduler instead of writing every adapter

If writing the API adapters gets tedious:
- **Postiz** (open source, self-host with Docker) — schedules to IG, YouTube, TikTok, FB, X and more, and has a public API you can call from `scheduler.py`. You still create the developer apps (same TikTok audit limitation), but it handles OAuth, token refresh and the UI.
- Paid posting APIs (Ayrshare, Blotato, etc.) — one API call posts everywhere and they've already passed the platform audits (incl. TikTok direct post). Monthly fee.

Recommendation: write your own **YouTube + Instagram** adapters (well-documented, free, good practice for your portfolio), use **TikTok inbox upload**, and only reach for Postiz or a paid API if maintenance becomes annoying.

### 7.7 Safety rails

- Auto-publish only clips you approved — never straight from the AI score (at least until the eval numbers in section 6 are very good).
- Optional "auto-approve" mode later: clips scoring ≥ 9 get published after a delay (e.g. 2 h) unless you hit "cancel" on the review page.
- Dry-run mode: generates everything and logs what it *would* post.
- Keep tokens in `secrets\` (gitignored) or Windows Credential Manager; never commit them.
- Per-platform daily cap in config to avoid looking spammy.

```yaml
publish:
  enabled: true
  dry_run: true
  platforms:
    youtube:   { enabled: true,  daily_cap: 2, slots: ["12:00", "19:00"] }
    instagram: { enabled: true,  daily_cap: 1, slots: ["18:00"], method: video_url }  # or resumable
    tiktok:    { enabled: true,  daily_cap: 2, mode: inbox }                             # inbox | direct
  notify:
    discord_webhook: ${DISCORD_WEBHOOK}
```

---

## 8. Risks / gotchas

- **Rate limits on free tiers** → candidate filtering in step 2 + concurrency cap + caching.
- **Humor is context-heavy** → send 30 s of lead-in context, chat reaction, and the game name; don't judge the punchline alone.
- **Loud ≠ funny** (FPS gunfire) → normalize loudness against the game-audio baseline; if possible record mic and game audio on separate OBS tracks so mic spikes are measured on their own (big accuracy win).
- **Vertical crop varies by scene** → define one crop preset per OBS scene layout.
- **Copyrighted music in game/stream audio** can get clips muted on upload — flag clips with music for review.
- **Platform API rules change often** (quotas, audits, permissions) → keep each platform in its own adapter so one breaking doesn't stop the others; re-check docs when something fails.
- **Game publisher policies** — some games restrict monetizing footage; check before enabling monetization on clips.

---

## 9. First steps checklist

- [ ] In OBS, enable multi-track recording: mic on its own track
- [ ] Install OmniRoute, create `clipper-text` and `clipper-vision` combos
- [ ] Install ffmpeg, Python 3.11, faster-whisper (CUDA)
- [ ] Pick one recent funny stream as the test VOD + download its chat log
- [ ] Build Phase 1 and compare its candidate list to what you remember
- [ ] Switch Instagram to a Professional account; create Meta, Google Cloud and TikTok developer apps (approval can take days, so start early)

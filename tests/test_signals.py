import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import signals  # noqa: E402


def test_load_markers_formats(tmp_path):
    f = tmp_path / "markers.txt"
    f.write_text("90\n01:30 grenade\n1:02:03.5 rage quit\nnot a timestamp\n", encoding="utf-8")
    assert signals.load_markers(f) == [90.0, 90.0, 3723.5]


def test_pick_peaks_respects_gap_and_threshold():
    score = np.zeros(300)
    score[[50, 55, 200]] = [9.0, 8.0, 5.0]
    assert signals.pick_peaks(score, min_gap=60, top_n=10, min_score=2.5) == [50, 200]


def test_audio_z_flags_spike_not_steady_noise():
    rng = np.random.default_rng(0)
    db = -40 + rng.normal(0, 1, 600).astype(np.float32)
    db[300:303] = -10
    z = signals.audio_z(db)
    assert z[301] > 5
    assert np.percentile(np.delete(z, range(295, 308)), 99) < 2


def test_chat_burst_scores_higher_than_quiet_chat():
    msgs = [(float(t), "hello", False) for t in range(0, 600, 10)]
    msgs += [(120 + i * 0.3, "KEKW", True) for i in range(30)]
    z = signals.chat_z(signals.chat_series(msgs, 600, lag=6))
    assert z[114:130].max() > 5
    assert z[400:500].max() < 1.5


def test_bigger_game_event_supersedes_smaller():
    import visual_events
    profile = {"supersedes": {"ace": ["4k", "3k"], "4k": ["3k"]}, "supersede_window_s": 20}
    events = [{"t": 100.0, "event": "3k", "score": 3}, {"t": 108.0, "event": "ace", "score": 6},
              {"t": 300.0, "event": "3k", "score": 3}]
    kept = visual_events.suppress_lesser(events, profile)
    assert [(e["event"], e["t"]) for e in kept] == [("ace", 108.0), ("3k", 300.0)]


def test_event_series_spreads_score_before_banner():
    import visual_events
    s = visual_events.event_series([{"t": 50.0, "event": "clutch", "score": 5}], 100)
    assert s[44] == 5 and s[52] == 5 and s[43] == 0 and s[53] == 0


def test_killfeed_badges_on_real_frame():
    import cv2
    import visual_events
    feed = cv2.imread(str(Path(__file__).parent / "fixtures" / "valorant_killfeed_ace.png"))
    found = {e for e, *_ in visual_events.badges_in_frame(feed, visual_events.load_badges("valorant"))}
    assert found == {"4k", "ace"}  # frame from an ACE round: IV and V badges on screen


def test_spectated_teammates_badge_is_ignored():
    import cv2
    import visual_events
    # dead and spectating a teammate who has a 3K: badge present, but it isn't ours
    feed = cv2.imread(str(Path(__file__).parent / "fixtures" / "valorant_killfeed_spectating.png"))
    assert visual_events.badges_in_frame(feed, visual_events.load_badges("valorant")) == []

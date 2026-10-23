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

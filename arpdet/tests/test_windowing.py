import math
from data.windowizer import iter_windows

def test_window_and_bins():
    # fabricate rows with ts at 0.05s steps for 1.5s
    rows = [{"ts": i*0.05} for i in range(30)]  # 0.0 .. 1.45
    W, S, M = 1.0, 0.5, 10
    wins = list(iter_windows(rows, W, S, M))
    # Expect windows starting near 0.0 and 0.5
    assert len(wins) >= 2
    t0, t1, wrows, bins = wins[0]
    assert math.isclose(t1-t0, 1.0, rel_tol=1e-6)
    # bins must be within [0,M-1]
    assert min(bins) >= 0 and max(bins) <= M-1

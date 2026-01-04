import math
from dosdet.data.windowizer import iter_windows

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

def test_tc2_window_segmentation_timestamp_assignment():
    rows = [
        {"ts": 0.00},
        {"ts": 0.10},
        {"ts": 0.24},
        {"ts": 0.25},
        {"ts": 0.49},
        {"ts": 0.50},
        {"ts": 0.74},
        {"ts": 0.75},
        {"ts": 0.99},
        {"ts": 1.00},
    ]
    W, S, M = 1.0, 1.0, 4
    wins = list(iter_windows(rows, W, S, M))
    assert len(wins) >= 2

    t0, t1, win_rows, bins = wins[0]
    assert math.isclose(t0, 0.0, rel_tol=1e-9)
    assert math.isclose(t1, 1.0, rel_tol=1e-9)

    expected_ts = [0.00, 0.10, 0.24, 0.25, 0.49, 0.50, 0.74, 0.75, 0.99]
    ts_in_window = [r["ts"] for r in win_rows]
    assert len(ts_in_window) == len(expected_ts)
    for got, exp in zip(ts_in_window, expected_ts):
        assert math.isclose(got, exp, rel_tol=1e-9, abs_tol=1e-12)
    assert all(t0 <= ts < t1 for ts in ts_in_window)

    expected_bins = [0, 0, 0, 1, 1, 2, 2, 3, 3]
    assert bins == expected_bins

    t0_2, t1_2, win_rows_2, bins_2 = wins[1]
    assert math.isclose(t0_2, 1.0, rel_tol=1e-9)
    assert math.isclose(t1_2, 2.0, rel_tol=1e-9)
    assert [r["ts"] for r in win_rows_2] == [1.0]
    assert bins_2 == [0]

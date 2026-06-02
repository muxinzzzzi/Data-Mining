from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

DAILY_PATH = ROOT / "data" / "SH#880823_daily_clean.csv"
M5_PATH = ROOT / "data" / "SH#880823_5min_clean.csv"
OUTPUT_DIR = ROOT / "outputs"
PREDICTION_DIR = OUTPUT_DIR / "predictions"
METRICS_DIR = OUTPUT_DIR / "metrics"
DIAGNOSTICS_DIR = OUTPUT_DIR / "diagnostics"
PLOT_DIR = OUTPUT_DIR / "plots"

INITIAL_CAPITAL = 100_000.0
COST_RATE = 0.001
TRADE_THRESHOLD = max(0.003, 2 * COST_RATE)
OPPORTUNITY_COST_BUFFER = 0.0015
TEST_START = pd.Timestamp("2025-01-01")
TEST_END = pd.Timestamp("2026-05-06")
VALID_START = pd.Timestamp("2024-07-01")
VALID_END = pd.Timestamp("2024-12-31")
VALIDATION_FOLDS = [
    ("2023H2", pd.Timestamp("2023-07-01"), pd.Timestamp("2023-12-31")),
    ("2024H1", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-06-30")),
    ("2024H2", pd.Timestamp("2024-07-01"), pd.Timestamp("2024-12-31")),
]
HORIZONS = [1, 3, 5, 10, 20]

RETRAIN_EVERY = 20
VALID_RETRAIN_EVERY = 252
RANDOM_STATE = 42
MIN_TRAIN_SAMPLES = 500
MIN_TASK_SAMPLES = 40
MIN_TASK_FEATURE_SELECTION_SAMPLES = 60
FEATURE_TOP_K_CANDIDATES: list[int | None] = [50, 80, 120, None]
TAIL_QUANTILE_MIN_SAMPLES = 252

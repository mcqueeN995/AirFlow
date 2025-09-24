from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import pandas as pd
from datetime import timedelta



@dataclass
class Splitter:
    holdout_days: int = 30
    gap_days: int = 0

    def split(self, df: pd.DataFrame, date_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        max_date = df[date_col].max().normalize()
        threshold = max_date - timedelta(days=self.holdout_days)
        # Exclude a gap before the threshold to reduce temporal leakage
        train_end = threshold - timedelta(days=max(self.gap_days, 0))
        train = df[df[date_col] < train_end]
        test = df[df[date_col] >= threshold]
        return train, test


from __future__ import annotations

import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, Tuple


def build_graph(df: pd.DataFrame,
                start_cluster_col: str = "start_cluster",
                end_cluster_col: str = "end_cluster",
                distance_col: str = "distance_km",
                price_col: str = "auction_price",
                start_dt_col: str = "first_point_date_time_start",
                end_dt_col: str = "last_point_date_time_until",
                weekday_col: str = "weekday",
                hour_col: str = "hour",
                price_trend_days: int = 28,
                top_customers_col: str = "customer_name",
                top_customers_count: int = 3) -> nx.DiGraph:
    df = df.copy()
    # Drop noise
    df = df[(df[start_cluster_col] != -1) & (df[end_cluster_col] != -1) & (df[start_cluster_col] != df[end_cluster_col])]

    # Basic fields
    df['date'] = pd.to_datetime(df[start_dt_col]).dt.date

    # Price trend (rolling median per edge)
    df['_edge'] = list(zip(df[start_cluster_col], df[end_cluster_col]))
    df = df.sort_values(by=[start_cluster_col, end_cluster_col, start_dt_col])
    df['price_rolling_med'] = (
        df.groupby('_edge')[price_col]
          .transform(lambda s: s.rolling(window=price_trend_days, min_periods=1).median())
    )

    # Aggregate by edge
    G = nx.DiGraph()
    grouped = df.groupby([start_cluster_col, end_cluster_col])
    for (u, v), g in grouped:
        # node ensures
        G.add_node(u)
        G.add_node(v)

        # time buckets
        g['travel_time_hours'] = (pd.to_datetime(g[end_dt_col]) - pd.to_datetime(g[start_dt_col])).dt.total_seconds() / 3600.0
        g['travel_time_group'] = np.floor((g['travel_time_hours'] - 1e-5) / 48).astype(int).clip(lower=1)

        weights = {}
        avg_time = {}
        avg_price = {}
        days_frequency = {}
        weekday_frequencies = {}
        hour_hist = {}
        median_distance = {}
        top_customers = {}

        # Period days and weekday support
        period_days = max((g['date'].max() - g['date'].min()).days + 1, 1)
        weekday_counts = g.groupby('date')['date'].first().index.to_series().apply(lambda d: d.weekday()).value_counts().to_dict()
        for d in range(7):
            weekday_counts.setdefault(d, 1)

        for ttg, gg in g.groupby('travel_time_group'):
            weights[ttg] = len(gg)
            avg_time[ttg] = gg['travel_time_hours'].mean()
            avg_price[ttg] = gg[price_col].mean()
            days_frequency[ttg] = gg['date'].nunique() / period_days
            median_distance[ttg] = gg[distance_col].median()
            # weekday probabilities
            wd = {}
            for d in range(7):
                day_count = gg[gg[start_dt_col].dt.weekday == d]['date'].nunique()
                wd[d] = day_count / weekday_counts[d]
            weekday_frequencies[ttg] = wd
            # hour histogram (0..23) with Laplace smoothing to avoid zeros
            counts = gg[start_dt_col].dt.hour.value_counts().reindex(range(24), fill_value=0)
            counts = counts + 1  # Laplace smoothing alpha=1
            probs = (counts / counts.sum()).to_dict()
            hour_hist[ttg] = {h: probs.get(h, 0.0) for h in range(24)}
            # top customers
            if top_customers_col in gg.columns and gg[top_customers_col].notna().any():
                shares = gg[top_customers_col].value_counts(normalize=True).head(top_customers_count)
                top_customers[ttg] = "; ".join([f"{n} - {p:.2f}" for n, p in shares.items()])
            else:
                top_customers[ttg] = ""

        G.add_edge(u, v,
                   weight=weights,
                   avg_time=avg_time,
                   avg_price=avg_price,
                   days_frequency=days_frequency,
                   weekday_frequencies=weekday_frequencies,
                   hour_hist=hour_hist,
                   top_customers=top_customers,
                   median_distance=median_distance)

    return G



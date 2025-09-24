from __future__ import annotations

from typing import Dict, List
import pandas as pd


def summarize_routes(routes_by_key: Dict[str, List]):
    rows = []
    for key, routes in routes_by_key.items():
        for r in routes:
            rows.append({
                'key': key,
                'steps': len(r.edges),
                'rev_total': r.total_price(),
                'km_total': r.total_distance(),
                'rev_per_day': r.revenue_per_day(),
                'rev_per_km': r.revenue_per_km(),
            })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def compare_baseline_vs_improved(baseline_routes: Dict[str, List], improved_routes: Dict[str, List]):
    b = summarize_routes(baseline_routes)
    i = summarize_routes(improved_routes)
    res = {}
    for name, df in [('baseline', b), ('improved', i)]:
        if df.empty:
            res[name] = {
                'orders_with_routes': 0,
                'routes_total': 0,
                'rev_per_day_mean': 0,
                'rev_per_km_mean': 0,
            }
            continue
        res[name] = {
            'orders_with_routes': df['key'].nunique(),
            'routes_total': len(df),
            'rev_per_day_mean': df['rev_per_day'].mean(),
            'rev_per_km_mean': df['rev_per_km'].mean(),
        }
    return res



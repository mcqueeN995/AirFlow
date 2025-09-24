from __future__ import annotations

from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from datetime import timedelta


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


def compute_metrics(test_df: pd.DataFrame,
                    routes_by_key: Dict[str, List],
                    start_cluster_col: str = 'start_cluster',
                    end_cluster_col: str = 'end_cluster',
                    start_dt_col: str = 'first_point_date_time_start',
                    time_tolerance_hours: int = 3) -> Dict[str, float]:
    """Compute requested evaluation metrics.

    - Feasibility: fraction of sampled routes that have non-decreasing times and positive durations
    - Accuracy: share of routes whose steps are supported by at least one test order within time tolerance
    - Coverage: fraction of test start clusters that have at least one route
    - Density: average number of routes per covered start cluster
    - Economics: mean revenue per day and per km over routes
    """
    # Prepare index of test edges by cluster pair and day/hour
    df = test_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[start_dt_col]):
        df[start_dt_col] = pd.to_datetime(df[start_dt_col], errors='coerce')
    df['_edge'] = list(zip(df[start_cluster_col], df[end_cluster_col]))
    df['date'] = df[start_dt_col].dt.date
    df['hour'] = df[start_dt_col].dt.hour

    # Group by edge for quick membership
    by_edge = df.groupby('_edge')

    all_routes = [r for rs in routes_by_key.values() for r in rs]
    if not all_routes:
        return {
            'feasibility': 0.0,
            'accuracy': 0.0,
            'coverage': 0.0,
            'density': 0.0,
            'rev_per_day_mean': 0.0,
            'rev_per_km_mean': 0.0,
        }

    # Feasibility: check times are monotonic and durations positive
    feasible_flags = []
    supported_flags = []
    for r in all_routes:
        times = []
        ok = True
        for (u, v, meta) in r.edges:
            dep = meta.get('departure_time')
            if dep is None:
                ok = False
                break
            times.append(dep)
        for i in range(1, len(times)):
            if times[i] < times[i - 1]:
                ok = False
                break
        feasible_flags.append(1 if ok and r.total_hours() > 0 else 0)

        # Accuracy: for each step, see if test has any matching edge near the planned departure hour
        step_support = []
        for (u, v, meta) in r.edges:
            dep = meta.get('departure_time')
            if dep is None:
                step_support.append(0)
                continue
            group = by_edge.get_group((u, v)) if (u, v) in by_edge.groups else None
            if group is None or group.empty:
                step_support.append(0)
                continue
            low = dep - timedelta(hours=time_tolerance_hours)
            high = dep + timedelta(hours=time_tolerance_hours)
            mask = (group[start_dt_col] >= low) & (group[start_dt_col] <= high)
            step_support.append(1 if mask.any() else 0)
        supported_flags.append(1 if step_support and (sum(step_support) == len(step_support)) else 0)

    # Coverage & density by keys (start clusters used as keys)
    covered_keys = {k for k, rs in routes_by_key.items() if len(rs) > 0}
    total_keys = df[start_cluster_col].nunique() if start_cluster_col in df.columns else 0
    coverage = (len(covered_keys) / total_keys) if total_keys > 0 else 0.0
    routes_per_key = (sum(len(rs) for rs in routes_by_key.values()) / len(covered_keys)) if covered_keys else 0.0

    econ_day = np.mean([r.revenue_per_day() for r in all_routes]) if all_routes else 0.0
    econ_km = np.mean([r.revenue_per_km() for r in all_routes]) if all_routes else 0.0

    return {
        'feasibility': float(np.mean(feasible_flags)) if feasible_flags else 0.0,
        'accuracy': float(np.mean(supported_flags)) if supported_flags else 0.0,
        'coverage': float(coverage),
        'density': float(routes_per_key),
        'rev_per_day_mean': float(econ_day),
        'rev_per_km_mean': float(econ_km),
    }


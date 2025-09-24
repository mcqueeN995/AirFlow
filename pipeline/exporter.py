from __future__ import annotations

import os
import pandas as pd
import networkx as nx
from typing import Dict, List


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def export_graph_edges(G: nx.DiGraph, out_dir: str):
    ensure_dir(out_dir)
    rows = []
    for u, v, data in G.edges(data=True):
        for tg in data['weight'].keys():
            rows.append({
                'start_cluster': u,
                'end_cluster': v,
                'travel_time_group': tg,
                'count': data['weight'][tg],
                'avg_time_hours': data['avg_time'][tg],
                'avg_price': data['avg_price'][tg],
                'routes_per_period_day': data['days_frequency'][tg],
            })
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, 'graph_edges.csv'), index=False)


def export_routes(routes_by_order: Dict[str, List], out_dir: str):
    ensure_dir(out_dir)
    rows = []
    for order_id, routes in routes_by_order.items():
        for ridx, r in enumerate(routes, 1):
            rows.append({
                'order_id': order_id,
                'route_index': ridx,
                'steps': len(r.edges),
                'revenue_total': r.total_price(),
                'distance_total_km': r.total_distance(),
                'revenue_per_day': r.revenue_per_day(),
                'revenue_per_km': r.revenue_per_km(),
            })
    cols = ['order_id', 'route_index', 'steps', 'revenue_total', 'distance_total_km', 'revenue_per_day', 'revenue_per_km']
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(os.path.join(out_dir, 'circular_routes.csv'), index=False)


def export_orders_with_clusters(df, out_dir: str):
    ensure_dir(out_dir)
    cols = [
        'order_number', 'first_point_location_city', 'last_point_location_city',
        'first_point_location_lat', 'first_point_location_lon',
        'last_point_location_lat', 'last_point_location_lon',
        'first_point_date_time_start', 'last_point_date_time_until',
        'distance_km', 'auction_price', 'start_cluster', 'end_cluster'
    ]
    existing = [c for c in cols if c in df.columns]
    df[existing].to_csv(os.path.join(out_dir, 'orders_with_clusters.csv'), index=False)



import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output')
REPORT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output', 'report')

os.makedirs(REPORT_DIR, exist_ok=True)

orders_path_csv = os.path.join(OUTPUT_DIR, 'orders_with_clusters.csv')
edges_path_csv = os.path.join(OUTPUT_DIR, 'graph_edges.csv')
routes_path_csv = os.path.join(OUTPUT_DIR, 'circular_routes.csv')

orders = pd.read_csv(orders_path_csv) if os.path.exists(orders_path_csv) else pd.DataFrame()
edges = pd.read_csv(edges_path_csv) if os.path.exists(edges_path_csv) else pd.DataFrame()
routes = pd.read_csv(routes_path_csv) if os.path.exists(routes_path_csv) else pd.DataFrame()

# 1) Summary of graph
if not edges.empty:
    fig = px.histogram(edges, x='avg_time_hours', nbins=30, title='Distribution: Avg travel time (hours)')
    fig.write_html(os.path.join(REPORT_DIR, 'avg_time_distribution.html'))
    try:
        fig.write_image(os.path.join(REPORT_DIR, 'avg_time_distribution.png'))
    except Exception:
        pass

    fig = px.histogram(edges, x='avg_price', nbins=30, title='Distribution: Avg price per edge')
    fig.write_html(os.path.join(REPORT_DIR, 'avg_price_distribution.html'))
    try:
        fig.write_image(os.path.join(REPORT_DIR, 'avg_price_distribution.png'))
    except Exception:
        pass

# 2) Routes KPIs
if not routes.empty:
    routes['steps'] = routes['steps'].fillna(0).astype(int)
    kpi = {
        'routes_total': len(routes),
        'avg_steps': routes['steps'].mean() if 'steps' in routes.columns else 0,
        'avg_rev_per_day': routes['revenue_per_day'].mean() if 'revenue_per_day' in routes.columns else 0,
        'avg_rev_per_km': routes['revenue_per_km'].mean() if 'revenue_per_km' in routes.columns else 0,
    }
    kpi_df = pd.DataFrame([kpi])
    kpi_df.to_csv(os.path.join(REPORT_DIR, 'kpi_summary.csv'), index=False)

    fig = px.histogram(routes, x='revenue_per_km', nbins=40, title='Revenue per KM distribution')
    fig.write_html(os.path.join(REPORT_DIR, 'rev_per_km_distribution.html'))
    try:
        fig.write_image(os.path.join(REPORT_DIR, 'rev_per_km_distribution.png'))
    except Exception:
        pass

    fig = px.histogram(routes, x='revenue_per_day', nbins=40, title='Revenue per Day distribution')
    fig.write_html(os.path.join(REPORT_DIR, 'rev_per_day_distribution.html'))
    try:
        fig.write_image(os.path.join(REPORT_DIR, 'rev_per_day_distribution.png'))
    except Exception:
        pass

print(f"Report generated in: {REPORT_DIR}")

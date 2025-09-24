from __future__ import annotations

import os
import yaml
import typer
from rich import print
from datetime import datetime
import pandas as pd

from pipeline.config import PipelineConfig, ClusteringConfig, SplitConfig, GraphConfig, RoutingConfig, PathsConfig
from pipeline.data import clean_and_prepare
from pipeline.split import Splitter
from pipeline.clustering import cluster_coordinates
from pipeline.graph import build_graph
from pipeline.routing import find_routes, find_routes_baseline
from pipeline.exporter import export_graph_edges, export_routes, ensure_dir, export_orders_with_clusters
from pipeline.eval import compare_baseline_vs_improved, compute_metrics


app = typer.Typer(help="Circular routes discovery pipeline")


def load_config(cfg_path: str | None) -> PipelineConfig:
    if cfg_path is None or not os.path.exists(cfg_path):
        return PipelineConfig()
    with open(cfg_path, 'r') as f:
        data = yaml.safe_load(f) or {}
    cfg = PipelineConfig(
        clustering=ClusteringConfig(**data.get('clustering', {})),
        split=SplitConfig(**data.get('split', {})),
        graph=GraphConfig(**data.get('graph', {})),
        routing=RoutingConfig(**data.get('routing', {})),
        paths=PathsConfig(**data.get('paths', {})),
    )
    return cfg


@app.command()
def run(cfg: str = typer.Option(None, help="Path to config.yaml")):
    config = load_config(cfg)

    print("[bold green]1/6 Load & prepare data[/bold green]")
    df = clean_and_prepare(config.paths.input_csv)

    print("[bold green]2/6 Time split[/bold green]")
    splitter = Splitter(holdout_days=config.split.holdout_days, gap_days=config.split.gap_days)
    train, test = splitter.split(df, 'first_point_date_time_start')
    print(f"Train: {len(train)} rows, Test: {len(test)} rows")

    print("[bold green]3/6 Clustering[/bold green]")
    # start clusters
    train_start, uniq_s, lab_s = cluster_coordinates(
        train,
        lat_col='first_point_location_lat',
        lon_col='first_point_location_lon',
        method=config.clustering.method,
        radius_km=config.clustering.radius_km,
        min_cluster_size=config.clustering.min_cluster_size,
        min_samples=config.clustering.min_samples,
        diameter_guard_km=config.clustering.diameter_guard_km
    )
    # end clusters (reuse params)
    train_end, uniq_e, lab_e = cluster_coordinates(
        train,
        lat_col='last_point_location_lat',
        lon_col='last_point_location_lon',
        method=config.clustering.method,
        radius_km=config.clustering.radius_km,
        min_cluster_size=config.clustering.min_cluster_size,
        min_samples=config.clustering.min_samples,
        diameter_guard_km=config.clustering.diameter_guard_km
    )
    # attach to train/test by matching indices
    train = train.copy()
    train['start_cluster'] = train_start['cluster']
    train['end_cluster'] = train_end['cluster']

    # Assign clusters for test by nearest train cluster centroids to avoid re-fitting on test
    from pipeline.clustering import assign_to_nearest_train_clusters
    test = test.copy()
    test['start_cluster'] = assign_to_nearest_train_clusters(
        test,
        train_unique_points=uniq_s,
        train_labels=lab_s,
        lat_col='first_point_location_lat',
        lon_col='first_point_location_lon'
    )
    test['end_cluster'] = assign_to_nearest_train_clusters(
        test,
        train_unique_points=uniq_e,
        train_labels=lab_e,
        lat_col='last_point_location_lat',
        lon_col='last_point_location_lon'
    )

    print("[bold green]4/6 Build demand graph[/bold green]")
    G = build_graph(train,
                    start_cluster_col='start_cluster', end_cluster_col='end_cluster',
                    distance_col='distance_km', price_col='auction_price',
                    start_dt_col='first_point_date_time_start', end_dt_col='last_point_date_time_until',
                    weekday_col='weekday', hour_col='hour',
                    price_trend_days=config.graph.price_trend_days,
                    top_customers_col='customer_name', top_customers_count=config.graph.top_customers_count)
    ensure_dir(config.paths.output_dir)
    export_graph_edges(G, config.paths.output_dir)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    print("[bold green]5/6 Search circular routes on test[/bold green]")
    # For demo, choose top-N start clusters by degree
    starts = sorted(G.nodes(), key=lambda n: G.out_degree(n), reverse=True)[:20]
    routes_by_order = {}
    baseline_by_order = {}
    for s in starts:
        # pick a reasonable start time: today at 08:00
        start_time = datetime.now().replace(hour=8, minute=0, second=0, microsecond=0)
        routes = find_routes(
            G, s, start_time,
            max_duration_days=config.routing.max_duration_days,
            max_routes=config.routing.max_routes,
            stopover_hours=config.routing.stopover_hours,
            unloading_hours=config.routing.unloading_hours,
            min_weekday_freq=config.routing.min_edge_weekday_freq,
            radius=config.clustering.radius_km,
            min_hourly_prob=config.routing.min_hourly_prob,
            revenue_per_day_min=config.routing.revenue_per_day_min,
            revenue_per_km_min=config.routing.revenue_per_km_min
        )
        if routes:
            routes_by_order[str(s)] = routes
        base = find_routes_baseline(
            G, s, start_time,
            max_duration_days=config.routing.max_duration_days,
            max_routes=config.routing.max_routes,
            stopover_hours=config.routing.stopover_hours,
            unloading_hours=config.routing.unloading_hours,
            min_days_freq=0.1,
            radius=config.clustering.radius_km
        )
        if base:
            baseline_by_order[str(s)] = base

    print("[bold green]6/6 Export results[/bold green]")
    export_routes(routes_by_order, config.paths.output_dir)
    export_orders_with_clusters(pd.concat([train, test], axis=0, ignore_index=True), config.paths.output_dir)
    cmp = compare_baseline_vs_improved(baseline_by_order, routes_by_order)
    print(cmp)
    # Compute extended metrics on improved routes
    metrics = compute_metrics(test, routes_by_order,
                              start_cluster_col='start_cluster', end_cluster_col='end_cluster',
                              start_dt_col='first_point_date_time_start',
                              time_tolerance_hours=3)
    print({'metrics': metrics})
    # Save metrics
    pd.DataFrame([metrics]).to_csv(os.path.join(config.paths.output_dir, 'metrics.csv'), index=False)
    print("Done.")


if __name__ == "__main__":
    app()



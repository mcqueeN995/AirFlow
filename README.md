## Circular Routes Optimization (FTL)

Reproducible pipeline to discover feasible and profitable circular routes from historical FTL orders.

### Quickstart

1) Install deps
```
pip install -r AirFlow/requirements.txt
```

2) Place `orders_source.csv` in `AirFlow/` (same columns as notebooks).

3) Run pipeline with default config
```
python AirFlow/run_pipeline.py run --cfg AirFlow/config.yaml
```

Outputs are written to `output/`: `graph_edges.csv`, `circular_routes.csv`.

### Features
- Time-based train/test split (holdout N days)
- Location clustering (HDBSCAN/DBSCAN or fixed-radius) with diameter guard
- Demand graph with weekday/hour distributions and price rolling median trends
- Probabilistic DFS circular route search with constraints
- Cost model: revenue per day and per km, exportable results



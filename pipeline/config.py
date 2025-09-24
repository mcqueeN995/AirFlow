from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class ClusteringConfig:
    method: str = "hdbscan"  # options: hdbscan, dbscan, fixed_radius
    radius_km: float = 75.0
    min_cluster_size: int = 5
    min_samples: Optional[int] = None
    diameter_guard_km: float = 75.0


@dataclass
class SplitConfig:
    holdout_days: int = 30
    # Exclude the last `gap_days` before the holdout from training to avoid leakage
    gap_days: int = 0


@dataclass
class GraphConfig:
    include_weekday_hour: bool = True
    price_trend_days: int = 28  # rolling window for trend
    top_customers_count: int = 3


@dataclass
class RoutingConfig:
    max_duration_days: int = 5
    max_routes: int = 5
    stopover_hours: int = 8
    unloading_hours: int = 6
    min_edge_weekday_freq: float = 0.4
    revenue_per_day_min: float = 27000
    revenue_per_km_min: float = 55
    # Minimum probability of a load being available at a given hour (0..1)
    min_hourly_prob: float = 0.02


@dataclass
class PathsConfig:
    input_csv: str = "orders_source.csv"
    output_dir: str = "output"


@dataclass
class PipelineConfig:
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    # Evaluation config to control sampling and time tolerances
    @dataclass
    class EvalConfig:
        max_test_orders: int = 1000
        time_tolerance_hours: int = 3

    evaluation: 'PipelineConfig.EvalConfig' = field(default_factory=lambda: PipelineConfig.EvalConfig())



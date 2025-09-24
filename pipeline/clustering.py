from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Literal, Optional
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import hdbscan
import math


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return 6371.0 * c


def ensure_diameter_guard(unique_points: np.ndarray, labels: np.ndarray, max_diameter_km: float) -> np.ndarray:
    if len(unique_points) == 0:
        return labels
    new_labels = labels.copy()
    next_label = (new_labels[new_labels >= 0].max() + 1) if np.any(new_labels >= 0) else 0
    for label in np.unique(labels):
        if label == -1:
            continue
        idx = np.where(labels == label)[0]
        pts = unique_points[idx]
        # If cluster size small, quick check
        ok = True
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                if haversine_km(pts[i, 0], pts[i, 1], pts[j, 0], pts[j, 1]) > max_diameter_km:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            continue
        # Split by greedy center contraction
        remaining = list(range(len(pts)))
        local_labels = [-1] * len(pts)
        local_next = 0
        while remaining:
            center = remaining[0]
            center_pt = pts[center]
            members = []
            rest = []
            for rid in remaining:
                d = haversine_km(center_pt[0], center_pt[1], pts[rid, 0], pts[rid, 1])
                if d <= max_diameter_km:
                    members.append(rid)
                else:
                    rest.append(rid)
            for m in members:
                local_labels[m] = local_next
            local_next += 1
            remaining = rest
        # map to global labels
        for local_id in range(local_next):
            mask = [k for k, v in enumerate(local_labels) if v == local_id]
            for m in mask:
                new_labels[idx[m]] = next_label
            next_label += 1
    return new_labels


def cluster_coordinates(df: pd.DataFrame,
                        lat_col: str,
                        lon_col: str,
                        method: Literal["hdbscan", "dbscan", "fixed_radius"] = "hdbscan",
                        radius_km: float = 75.0,
                        min_cluster_size: int = 5,
                        min_samples: Optional[int] = None,
                        diameter_guard_km: float = 75.0) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    # Prepare unique points
    points = df[[lat_col, lon_col]].dropna().values
    unique_points = np.unique(points, axis=0)
    if len(unique_points) == 0:
        return df.assign(cluster=-1), unique_points, np.array([])

    if method == "hdbscan":
        # Convert to radians for haversine metric if needed, but we will use precomputed distances via great-circle
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                    min_samples=min_samples if min_samples is not None else max(2, min_cluster_size // 2),
                                    cluster_selection_epsilon=0,
                                    metric='haversine')
        unique_rad = np.radians(unique_points)
        labels = clusterer.fit_predict(unique_rad)
    elif method == "dbscan":
        eps_rad = radius_km / 6371.0
        unique_rad = np.radians(unique_points)
        clusterer = DBSCAN(eps=eps_rad, min_samples=min_cluster_size, metric='haversine')
        labels = clusterer.fit_predict(unique_rad)
    else:
        # fixed_radius greedy clustering (center-based, centroid constraint)
        labels = np.full(len(unique_points), -1)
        assigned = np.zeros(len(unique_points), dtype=bool)
        next_label = 0
        for i in range(len(unique_points)):
            if assigned[i]:
                continue
            center = unique_points[i]
            members = []
            for j in range(len(unique_points)):
                if assigned[j]:
                    continue
                d = haversine_km(center[0], center[1], unique_points[j][0], unique_points[j][1])
                if d <= radius_km:
                    members.append(j)
            for m in members:
                labels[m] = next_label
                assigned[m] = True
            next_label += 1

    # Enforce diameter guard
    labels = ensure_diameter_guard(unique_points, labels, diameter_guard_km)

    # Map back to df
    index_map = {(round(lat, 6), round(lon, 6)): lab for (lat, lon), lab in zip(unique_points, labels)}
    df_out = df.copy()
    df_out["cluster"] = df_out.apply(lambda r: index_map.get((round(r[lat_col], 6), round(r[lon_col], 6)), -1), axis=1)
    return df_out, unique_points, labels


def assign_to_nearest_train_clusters(test_df: pd.DataFrame,
                                     train_unique_points: np.ndarray,
                                     train_labels: np.ndarray,
                                     lat_col: str,
                                     lon_col: str) -> pd.Series:
    """Assign each test point to the nearest train cluster centroid.

    Points farther than the max intra-cluster diameter among train clusters are assigned -1.
    """
    if len(train_unique_points) == 0 or len(np.unique(train_labels[train_labels >= 0])) == 0:
        return pd.Series([-1] * len(test_df), index=test_df.index)

    # compute centroids per label
    centroids = []
    labels_unique = []
    max_diameter_by_label: Dict[int, float] = {}
    for lab in np.unique(train_labels):
        if lab == -1:
            continue
        pts = train_unique_points[train_labels == lab]
        if len(pts) == 0:
            continue
        lat_c = pts[:, 0].mean()
        lon_c = pts[:, 1].mean()
        centroids.append((lat_c, lon_c))
        labels_unique.append(lab)
        # compute max pair distance within cluster for guard
        md = 0.0
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                md = max(md, haversine_km(pts[i, 0], pts[i, 1], pts[j, 0], pts[j, 1]))
        max_diameter_by_label[lab] = md

    if not centroids:
        return pd.Series([-1] * len(test_df), index=test_df.index)

    def pick_cluster(lat: float, lon: float) -> int:
        best_lab = -1
        best_dist = float('inf')
        for (clat, clon), lab in zip(centroids, labels_unique):
            d = haversine_km(lat, lon, clat, clon)
            if d < best_dist:
                best_dist = d
                best_lab = lab
        # reject if outside cluster diameter guard
        if best_lab == -1:
            return -1
        if best_dist > max_diameter_by_label.get(best_lab, float('inf')):
            return -1
        return int(best_lab)

    assigned = test_df.apply(lambda r: pick_cluster(r[lat_col], r[lon_col]), axis=1)
    return assigned



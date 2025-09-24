from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import pandas as pd
import numpy as np


@dataclass
class Columns:
    order_id: str = "order_number"
    distance_km: str = "distance_km"
    published_at: str = "published_at"
    customer_name: str = "customer_name"
    manager_name: str = "manager_name"
    start_dt: str = "first_point_date_time_start"
    start_city: str = "first_point_location_city"
    start_region: str = "first_point_location_region"
    end_dt: str = "last_point_date_time_until"
    end_city: str = "last_point_location_city"
    end_region: str = "last_point_location_region"
    price_carrier: str = "carrier_price_with_vat"
    price_customer: str = "customer_price_with_vat"
    price_auction: str = "auction_price"
    start_lat: str = "first_point_location_lat"
    start_lon: str = "first_point_location_lon"
    end_lat: str = "last_point_location_lat"
    end_lon: str = "last_point_location_lon"


def load_orders_csv(path: str, columns: Columns = Columns()) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Parse datetimes
    for col in [columns.published_at, columns.start_dt, columns.end_dt]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    # Basic cleaning
    df = df.dropna(subset=[columns.start_lat, columns.start_lon, columns.end_lat, columns.end_lon])
    df = df[(df[columns.distance_km] > 0)]
    if columns.price_auction in df.columns:
        df = df[df[columns.price_auction].fillna(0) > 0]
    return df


def add_temporal_fields(df: pd.DataFrame, columns: Columns = Columns()) -> pd.DataFrame:
    out = df.copy()
    out["weekday"] = out[columns.start_dt].dt.weekday
    out["hour"] = out[columns.start_dt].dt.hour
    out["date"] = out[columns.start_dt].dt.date
    return out


def clean_and_prepare(path: str) -> pd.DataFrame:
    df = load_orders_csv(path)
    df = add_temporal_fields(df)
    return df



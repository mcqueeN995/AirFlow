from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from typing import Dict, List, Tuple
from datetime import timedelta
import networkx as nx
import math


@dataclass
class CostModel:
    revenue_per_day_min: float = 27000
    revenue_per_km_min: float = 55
    min_hourly_prob: float = 0.02


class CircularRoute:
    def __init__(self, start_city: str, start_time, unloading_hours: int, stopover_hours: int, radius: float):
        self.start_city = start_city
        self.unloading_hours = unloading_hours
        self.stopover_hours = stopover_hours
        self.radius = radius
        self.nodes: Dict[str, Dict] = {}
        self.edges: List[Tuple[str, str, Dict]] = []
        # initialize start with load window
        self.nodes[start_city] = {
            'arrival_time': None,
            'unloading_end_time': None,
            'load_start_time': start_time,
            'load_end_time': start_time + timedelta(hours=unloading_hours),
        }

    def get_last_city(self) -> str:
        with_times = [(c, d) for c, d in self.nodes.items() if d.get('unloading_end_time') is not None]
        if not with_times:
            # return latest load_end_time node
            return max(self.nodes.items(), key=lambda x: x[1]['load_end_time'])[0]
        return max(with_times, key=lambda x: x[1]['unloading_end_time'])[0]

    def step_count(self) -> int:
        return len(self.edges)

    def add_step(self, g: nx.DiGraph, to_city: str, edge_data: Dict, travel_group: int):
        current = self.get_last_city()
        cur = self.nodes[current]
        # next load window in current
        cur['load_start_time'] = cur['unloading_end_time'] + timedelta(hours=self.stopover_hours) if cur['unloading_end_time'] else cur['load_start_time']
        cur['load_end_time'] = cur['load_start_time'] + timedelta(hours=self.unloading_hours)
        departure_time = cur['load_start_time']
        arrival = cur['load_end_time'] + timedelta(hours=edge_data['avg_time'][travel_group])
        if to_city not in self.nodes:
            self.nodes[to_city] = {
                'arrival_time': arrival,
                'unloading_end_time': arrival + timedelta(hours=self.unloading_hours),
                'load_start_time': None,
                'load_end_time': None,
            }
        else:
            self.nodes[to_city]['arrival_time'] = arrival
            self.nodes[to_city]['unloading_end_time'] = arrival + timedelta(hours=self.unloading_hours)
        self.edges.append((current, to_city, {
            'step_price': edge_data['avg_price'][travel_group],
            'step_distance': edge_data['median_distance'][travel_group],
            'step_frequency': edge_data['weekday_frequencies'][travel_group][cur['load_start_time'].weekday()],
            'step_top_customers': edge_data['top_customers'][travel_group],
            'departure_time': departure_time,
            'travel_group': travel_group
        }))

    def remove_last(self):
        if not self.edges:
            return
        frm, to, _ = self.edges.pop()
        # remove node if it has no incoming edges
        if to != self.start_city and not any(e[1] == to for e in self.edges):
            self.nodes.pop(to, None)

    def total_price(self) -> float:
        return sum(e[2]['step_price'] for e in self.edges)

    def total_distance(self) -> float:
        if not self.edges:
            return 0.0
        base = sum(e[2]['step_distance'] for e in self.edges)
        add = (self.radius / 2.0) * max(0, len(self.edges) - 1)
        return base + add

    def total_hours(self) -> float:
        if not self.edges:
            return 0.0
        start_load = self.nodes[self.start_city]['load_start_time']
        last = self.get_last_city()
        end_unload = self.nodes[last]['unloading_end_time']
        return (end_unload - start_load).total_seconds() / 3600.0 if start_load and end_unload else 0.0

    def revenue_per_day(self) -> float:
        h = self.total_hours()
        return (self.total_price() / h) * 24.0 if h > 0 else 0.0

    def revenue_per_km(self) -> float:
        d = self.total_distance()
        return (self.total_price() / d) if d > 0 else 0.0


def find_routes(g: nx.DiGraph,
                start_city: str,
                start_time,
                max_duration_days: int,
                max_routes: int,
                stopover_hours: int,
                unloading_hours: int,
                min_weekday_freq: float,
                radius: float,
                min_hourly_prob: float = 0.0,
                revenue_per_day_min: float = 0.0,
                revenue_per_km_min: float = 0.0) -> List[CircularRoute]:
    routes: List[CircularRoute] = []

    cr = CircularRoute(start_city, start_time, unloading_hours, stopover_hours, radius)

    def dfs(route: CircularRoute):
        current = route.get_last_city()
        # Ensure current node has a load window before expanding
        cur_node = route.nodes[current]
        if cur_node.get('load_start_time') is None and cur_node.get('unloading_end_time') is not None:
            cur_node['load_start_time'] = cur_node['unloading_end_time'] + timedelta(hours=stopover_hours)
            cur_node['load_end_time'] = cur_node['load_start_time'] + timedelta(hours=unloading_hours)

        # close cycle
        if current != start_city and g.has_edge(current, start_city):
            # can we return respecting min_weekday_freq?
            cur_node = route.nodes[current]
            candidate = g[current][start_city]
            for tg in candidate['weight'].keys():
                weekday_prob = candidate['weekday_frequencies'][tg][cur_node['load_start_time'].weekday()]
                hour_prob = candidate['hour_hist'][tg].get(cur_node['load_start_time'].hour, 0.0)
                if weekday_prob >= min_weekday_freq and hour_prob >= min_hourly_prob:
                    route.add_step(g, start_city, candidate, tg)
                    # check cost model thresholds
                    if (revenue_per_day_min <= 0 or route.revenue_per_day() >= revenue_per_day_min) and \
                       (revenue_per_km_min <= 0 or route.revenue_per_km() >= revenue_per_km_min):
                        routes.append(deepcopy(route))
                    route.remove_last()
                    break

        if route.step_count() >= max_routes:
            return

        # expand
        cur_node = route.nodes[current]
        for nbr in g.neighbors(current):
            if nbr in route.nodes and nbr != start_city:
                continue
            e = g[current][nbr]
            for tg in e['weight'].keys():
                weekday_prob = e['weekday_frequencies'][tg][cur_node['load_start_time'].weekday()]
                hour_prob = e['hour_hist'][tg].get(cur_node['load_start_time'].hour, 0.0)
                if weekday_prob < min_weekday_freq or hour_prob < min_hourly_prob:
                    continue
                # time feasibility
                hours_next = route.total_hours() + stopover_hours + e['avg_time'][tg] + unloading_hours
                if hours_next > max_duration_days * 24.0:
                    continue
                route.add_step(g, nbr, e, tg)
                # prune if cost model already below thresholds and cannot recover (simple check: after first step)
                if (route.step_count() == 0) or \
                   ((revenue_per_day_min <= 0 or route.revenue_per_day() >= revenue_per_day_min * 0.5) and \
                    (revenue_per_km_min <= 0 or route.revenue_per_km() >= revenue_per_km_min * 0.5)):
                    dfs(route)
                route.remove_last()

    dfs(cr)
    return routes


def find_routes_baseline(g: nx.DiGraph,
                         start_city: str,
                         start_time,
                         max_duration_days: int,
                         max_routes: int,
                         stopover_hours: int,
                         unloading_hours: int,
                         min_days_freq: float,
                         radius: float) -> List[CircularRoute]:
    """
    Baseline DFS that uses aggregated per-day frequency over the period (ignores weekday profile).
    """
    routes: List[CircularRoute] = []
    cr = CircularRoute(start_city, start_time, unloading_hours, stopover_hours, radius)

    def dfs(route: CircularRoute):
        current = route.get_last_city()
        # try close cycle
        if current != start_city and g.has_edge(current, start_city):
            e = g[current][start_city]
            feasible = any(freq >= min_days_freq for freq in e['days_frequency'].values())
            if feasible:
                # pick any travel group with sufficient freq
                for tg, freq in e['days_frequency'].items():
                    if freq >= min_days_freq:
                        route.add_step(g, start_city, e, tg)
                        routes.append(deepcopy(route))
                        route.remove_last()
                        break
        if route.step_count() >= max_routes:
            return
        for nbr in g.neighbors(current):
            if nbr in route.nodes and nbr != start_city:
                continue
            e = g[current][nbr]
            # any travel group with sufficient period frequency
            feasible_tg = [tg for tg, freq in e['days_frequency'].items() if freq >= min_days_freq]
            if not feasible_tg:
                continue
            for tg in feasible_tg:
                hours_next = route.total_hours() + stopover_hours + e['avg_time'][tg] + unloading_hours
                if hours_next > max_duration_days * 24.0:
                    continue
                route.add_step(g, nbr, e, tg)
                dfs(route)
                route.remove_last()

    dfs(cr)
    return routes



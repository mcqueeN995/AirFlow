import os
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Circular Routes Explorer", layout="wide")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output')
orders_path = os.path.join(OUTPUT_DIR, 'orders_with_clusters.csv')
edges_path = os.path.join(OUTPUT_DIR, 'graph_edges.csv')
routes_path = os.path.join(OUTPUT_DIR, 'circular_routes.csv')

@st.cache_data
def load_data():
    orders = pd.read_csv(orders_path) if os.path.exists(orders_path) else pd.DataFrame()
    edges = pd.read_csv(edges_path) if os.path.exists(edges_path) else pd.DataFrame()
    routes = pd.read_csv(routes_path) if os.path.exists(routes_path) else pd.DataFrame()
    return orders, edges, routes

orders, edges, routes = load_data()

st.title("FTL Circular Routes Explorer")

with st.sidebar:
    st.header("Filters")
    min_km = st.number_input("Min revenue per KM", value=0.0, step=1.0)
    min_day = st.number_input("Min revenue per Day", value=0.0, step=1000.0)
    max_steps = st.number_input("Max steps", value=10, step=1)

st.subheader("Graph edges summary")
if edges.empty:
    st.info("No edges data found. Run pipeline first.")
else:
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(edges, x='avg_price', nbins=30, title='Avg price per edge')
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.histogram(edges, x='avg_time_hours', nbins=30, title='Avg travel time(hours)')
        st.plotly_chart(fig, use_container_width=True)

st.subheader("Circular routes")
if routes.empty:
    st.info("No routes found.")
else:
    r = routes.copy()
    if 'revenue_per_km' in r.columns:
        r = r[r['revenue_per_km'] >= min_km]
    if 'revenue_per_day' in r.columns:
        r = r[r['revenue_per_day'] >= min_day]
    if 'steps' in r.columns:
        r = r[r['steps'] <= max_steps]
    st.write(f"Routes shown: {len(r)}")
    st.dataframe(r.head(1000))

st.caption("Data from AirFlow/output")

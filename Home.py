import streamlit as st
import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import zipfile
import io
import requests
from datetime import datetime

# =============================================================================
# Page Configuration & Global Settings
# =============================================================================
st.set_page_config(
    page_title="Dashboard Analisis Anggaran dan Realisasi Belanja Negara",
    page_icon=":analytics:",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.kemenkeu.go.id',
        'Report a bug': 'https://github.com/tubankum3/dashpmk/issues',
        'About': "Dashboard Anggaran Bidang PMK"
    }
)

# =============================================================================
# Material Design Styled CSS
# =============================================================================
st.markdown("""
<style>
:root {
    --primary: #1a73e8;
    --primary-dark: #0d47a1;
    --secondary: #34a853;
    --warning: #f9ab00;
    --error: #ea4335;
    --surface: #f6f6f6;
    --background: #f8f9fa;
    --on-surface: #202124;
    --on-primary: #ffffff;
    --shadow-1: 0 1px 2px rgba(60,64,67,0.3), 0 1px 3px 1px rgba(60,64,67,0.15);
    --shadow-2: 0 1px 2px rgba(60,64,67,0.3), 0 2px 6px 2px rgba(60,64,67,0.15);
    --shadow-3: 0 1px 3px rgba(60,64,67,0.3), 0 4px 8px 3px rgba(60,64,67,0.15);
    --border-radius: 8px;
    --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

/* Main App Styling */
.stApp {
    background-color: var(--background);
    font-family: 'Google Sans', 'Roboto', 'Inter', sans-serif;
}

/* Header with Breadcrumb Navigation */
.dashboard-header {
    background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
    padding: 2rem;
    border-radius: var(--border-radius);
    margin-bottom: 1.5rem;
    color: var(--on-primary);
    box-shadow: var(--shadow-2);
}

.breadcrumb {
    font-size: 0.875rem;
    opacity: 0.9;
    margin-bottom: 0.5rem;
}

/* Card System */
.material-card {
    background: var(--surface);
    border-radius: var(--border-radius);
    box-shadow: var(--shadow-1);
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    transition: var(--transition);
    border: 1px solid #e8eaed;
}

.material-card:hover {
    box-shadow: var(--shadow-2);
    transform: translateY(-2px);
}

.material-card-elevated {
    box-shadow: var(--shadow-3) !important;
}

/* Typography Scale */
.dashboard-title {
    font-family: 'Google Sans', sans-serif;
    font-weight: 700;
    font-size: 2.25rem;
    line-height: 1.2;
    margin: 0;
}

.dashboard-subtitle {
    font-family: 'Google Sans', sans-serif;
    font-weight: 400;
    font-size: 1.125rem;
    opacity: 0.9;
    margin: 0.5rem 0 0 0;
}

.section-title {
    font-family: 'Google Sans', sans-serif;
    font-weight: 600;
    font-size: 1.25rem;
    color: var(--on-surface);
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid var(--primary);
}

/* Metric Cards */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 1rem;
    margin: 1.5rem 0;
}

.metric-card {
    background: var(--surface);
    border-radius: var(--border-radius);
    padding: 1.5rem;
    box-shadow: var(--shadow-1);
    transition: var(--transition);
    border-left: 4px solid var(--primary);
    position: relative;
    overflow: hidden;
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--primary), var(--secondary));
}

.metric-card:hover {
    box-shadow: var(--shadow-2);
    transform: translateY(-2px);
}

.metric-value {
    font-size: 1.75rem;
    font-weight: 700;
    color: var(--on-surface);
    margin: 0.5rem 0;
    line-height: 1.2;
}

.metric-label {
    font-size: 0.875rem;
    color: #5f6368;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.metric-sublabel {
    font-size: 0.675rem;
    color: #5f6368;
    font-weight: 500;
    text-transform: none;
    letter-spacing: 0.5px;
}

.metric-trend {
    display: inline-flex;
    align-items: center;
    padding: 0.25rem 0.5rem;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-top: 0.5rem;
}

.trend-positive {
    background: #e6f4ea;
    color: var(--secondary);
}

.trend-negative {
    background: #fce8e6;
    color: var(--error);
}

/* Interactive Elements */
.stButton>button {
    background: var(--primary);
    color: var(--on-primary);
    border: none;
    border-radius: var(--border-radius);
    padding: 0.75rem 1.5rem;
    font-weight: 500;
    text-transform: none;
    transition: var(--transition);
    box-shadow: var(--shadow-1);
}

.stButton>button:hover {
    background: var(--primary-dark);
    box-shadow: var(--shadow-2);
    transform: translateY(-1px);
}

/* Sidebar */
.stSidebar {
    background: var(--surface);
    border-right: 1px solid #e8eaed;
}

.sidebar-section {
    background: var(--surface);
    border-radius: var(--border-radius);
    padding: 0.1rem;
    margin-bottom: 1rem;
    box-shadow: var(--shadow-1);
}

/* Tab */
.stTabs [data-baseweb="tab-list"] {
    flex-wrap: wrap !important;
    gap: 0.5rem;
}

.stTabs [data-baseweb="tab"] {
    background: var(--surface);
    border-radius: var(--border-radius) var(--border-radius) 0 0;
    padding: 0.75rem 1.5rem;
    border: 1px solid #e8eaed;
    transition: var(--transition);
    flex: initial !important;
    white-space: nowrap !important;
    margin-bottom: 6px;
}

.stTabs [aria-selected="true"] {
    background: var(--primary);
    color: var(--on-primary);
}

/* Focus indicators */
.stButton>button:focus, .stSelectbox:focus, .stSlider:focus {
    outline: 2px solid var(--primary);
    outline-offset: 2px;
}

/* Responsive design */
@media (max-width: 768px) {
    .metric-grid {
        grid-template-columns: 1fr;
    }
    
    .dashboard-title {
        font-size: 1.75rem;
    }
    
    .material-card {
        padding: 1rem;
        margin-bottom: 1rem;
    }
}

/* Style for columns containing charts */
[data-testid="column"] > div {
    background: white;
    border-radius: var(--border-radius);
    padding: 1.5rem;
    box-shadow: var(--shadow-1);
    border: 0px solid #e8eaed;
}

/* Chart container */
.chart-container {
    background: white;
    border-radius: var(--border-radius);
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    box-shadow: var(--shadow-1);
    border: 0px solid #e8eaed;
}

/* Data table */
.data-table {
    background: var(--surface);
    border-radius: var(--border-radius);
    overflow: hidden;
    box-shadow: var(--shadow-1);
}

</style>
""", unsafe_allow_html=True)

# =============================================================================
# Data Loading
# =============================================================================
@st.cache_data(show_spinner=True)
def load_data():
    """
    Load and preprocess budget data from GitHub with error handling and validation.
    
    Returns:
        pd.DataFrame: Preprocessed budget data
    """
    url = "https://raw.githubusercontent.com/tubankum3/dashpmk/main/df.csv.zip"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            with z.open("df.csv") as file:
                df = pd.read_csv(file, low_memory=False)
        
        if df.empty:
            st.error("Dataset kosong atau tidak valid")
            return pd.DataFrame()
        
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])
        
        if "Tahun" in df.columns:
            df["Tahun"] = pd.to_numeric(df["Tahun"], errors="coerce").astype("int64")
            
        required_columns = ["KEMENTERIAN/LEMBAGA", "Tahun"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.error(f"Kolom yang diperlukan tidak ditemukan: {missing_columns}")
            return pd.DataFrame()
            
        return df
        
    except FileNotFoundError:
        st.error("❌ File dataset tidak ditemukan. Pastikan 'df.csv' tersedia.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Gagal memuat data: {str(e)}")
        return pd.DataFrame()

# =============================================================================
# Utility Functions
# =============================================================================
def format_rupiah(value: float) -> str:
    """Format numeric value to Indonesian Rupiah currency format"""
    if pd.isna(value) or value == 0:
        return "Rp 0"
    
    abs_value = abs(value)
    
    if abs_value >= 1_000_000_000_000:
        return f"Rp {value/1_000_000_000_000:.2f} T"
    elif abs_value >= 1_000_000_000:
        return f"Rp {value/1_000_000_000:.2f} M"
    elif abs_value >= 1_000_000:
        return f"Rp {value/1_000_000:.2f} Jt"
    else:
        return f"Rp {value:,.0f}"

def rupiah_separator(x):
    try:
        x = float(x)
    except:
        return x
    return f"Rp {x:,.0f}".replace(",", ".")

# =============================================================================
# Chart Functions
# =============================================================================
def create_time_series_chart(df, selected_kls, selected_years, primary, secondary):
    """Create time series chart with bar and scatter"""
    # Filter data
    filtered_df = df.copy()
    filtered_df = filtered_df[
        (filtered_df["Tahun"] >= selected_years[0]) & 
        (filtered_df["Tahun"] <= selected_years[1])
    ]
    
    if selected_kls != "Semua":
        filtered_df = filtered_df[filtered_df["KEMENTERIAN/LEMBAGA"] == selected_kls]
    
    # Aggregate by year
    agg = (
        filtered_df.groupby("Tahun", as_index=False)[
            [secondary, primary]
        ].sum()
    )
    
    # Calculate realization percentage
    agg["Persentase Realisasi"] = (agg[secondary] / agg[primary] * 100)
    agg = agg.sort_values("Tahun")
    
    # Convert Tahun to string for x-axis
    agg["Tahun"] = agg["Tahun"].astype(str)
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add bar chart for primary metric
    fig.add_trace(
        go.Bar(
            x=agg["Tahun"],
            y=agg[primary],
            name=primary.replace("PAGU DIPA REVISI EFEKTIF", "Pagu DIPA Revisi (Efektif)"),
            marker=dict(color="#005FAC"),
            hovertemplate=f"{primary.replace('PAGU DIPA REVISI EFEKTIF', 'Pagu')}: %{{y:,.0f}}<extra></extra>"
        ),
        secondary_y=False
    )
    
    # Add scatter plot for secondary metric
    fig.add_trace(
        go.Scatter(
            x=agg["Tahun"],
            y=agg[secondary],
            mode="markers+lines",
            name=secondary.replace("REALISASI BELANJA KL (SAKTI)", "Realisasi Belanja"),
            marker=dict(color="#FAB715", size=12, line=dict(color="white", width=1.5)),
            line=dict(color="#FAB715", width=2),
            customdata=agg["Persentase Realisasi"],
            hovertemplate=f"{secondary.replace('REALISASI BELANJA KL (SAKTI)', 'Realisasi')}: %{{y:,.0f}} (%{{customdata:.2f}}%)<extra></extra>"
        ),
        secondary_y=True
    )
    
    # Add data labels for the bar (primary) trace
    fig.update_traces(
        selector=dict(type="bar"),
        text=agg[primary].apply(lambda v: format_rupiah(v)),
        textposition="outside",
        textfont=dict(size=11)
    )
    
    # Add data labels for the scatter (secondary) trace - percentage
    scatter_texts = [
        f"({pct:.1f}%)" 
        for val, pct in zip(agg[secondary], agg["Persentase Realisasi"])
    ]
    fig.update_traces(
        selector=dict(type="scatter"),
        text=scatter_texts,
        textposition="bottom center",
        textfont=dict(size=10, color="lightgrey"),
        mode="markers+lines+text"
    )
        
    # Calculate max value for synchronized y-axes
    max_value = max(agg[primary].max(), agg[secondary].max())
    
    # Update layout
    title_kl = f" {selected_kls}" if selected_kls != "Semua" else ""
    fig.update_layout(
        title=dict(
            text=f"PERBANDINGAN {primary} TERHADAP {secondary}<br>PERIODE {selected_years[0]} - {selected_years[1]}<br>{title_kl}",
            x=0.5,
            xanchor='center',
            font=dict(size=14)
        ),
        xaxis_title="Tahun",
        template="plotly_white",
        height=500,
        margin=dict(t=130, b=30, l=30, r=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified"
    )
    
    # Hide y-axes but keep synchronized ranges
    fig.update_yaxes(
        showticklabels=False, 
        title_text="", 
        secondary_y=False, 
        range=[0, max_value * 1.1]
    )
    fig.update_yaxes(
        showticklabels=False, 
        title_text="", 
        secondary_y=True, 
        range=[0, max_value * 1.1]
    )
    
    return fig

def create_sankey_chart(df, selected_kl, selected_year, metric, parent_col, child_col):
    """Create Sankey diagram for budget flow visualization"""
    
    # Filter by K/L first
    if selected_kl != "Semua":
        df = df[df["KEMENTERIAN/LEMBAGA"] == selected_kl].copy()
    
    # Filter by year
    df_year = df[df["Tahun"] == int(selected_year)].copy()
    
    if df_year.empty or metric not in df_year.columns:
        # Return empty figure if no data
        fig = go.Figure()
        fig.add_annotation(
            text="Tidak ada data untuk ditampilkan",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=9)
        )
        fig.update_layout(height=500)
        return fig
        
    # Aggregations
    agg_parent = df_year.groupby(parent_col, as_index=False)[metric].sum().query(f"`{metric}`>0")
    agg_parent_child = df_year.groupby([parent_col, child_col], as_index=False)[metric].sum().query(f"`{metric}`>0")
    agg_child = df_year.groupby(child_col, as_index=False)[metric].sum().query(f"`{metric}`>0")
    
    # Calculate total for percentages
    total_value = df_year[metric].sum()
    
    if total_value <= 0:
        fig = go.Figure()
        fig.add_annotation(
            text="Tidak ada data untuk ditampilkan",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=9)
        )
        fig.update_layout(height=500)
        return fig
    
    total_label = f"{metric}"
    parent_list = agg_parent[parent_col].astype(str).tolist()
    child_list = agg_child[child_col].astype(str).tolist()
    
    labels = [total_label] + [f"{p}" for p in parent_list] + [f"{c}" for c in child_list]
    index_map = {lab: i for i, lab in enumerate(labels)}
    
    sources, targets, values, link_labels = [], [], [], []
    
    # Total -> Parent
    for _, r in agg_parent.iterrows():
        p = str(r[parent_col])
        v = float(r[metric])
        if v <= 0: 
            continue
        sources.append(index_map[total_label])
        targets.append(index_map[f"{p}"])
        values.append(v)
        link_labels.append(f"{total_label} → {p}: {format_rupiah(v)}")
    
    # Parent -> Child
    for _, r in agg_parent_child.iterrows():
        p = str(r[parent_col])
        c = str(r[child_col])
        v = float(r[metric])
        if v <= 0:
            continue
        sources.append(index_map[f"{p}"])
        targets.append(index_map[f"{c}"])
        values.append(v)
        link_labels.append(f"{p} → {c}: {format_rupiah(v)}")
    
    # Color functions
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    def rgb_to_hex(rgb):
        return '#{:02x}{:02x}{:02x}'.format(*rgb)
    
    def lighten_color(hex_color, factor=0.3):
        rgb = hex_to_rgb(hex_color)
        light_rgb = tuple(min(255, int(c + (255 - c) * factor)) for c in rgb)
        return rgb_to_hex(light_rgb)
    
    # Base color and gradient definitions
    base_color = "#005FAC"
    parent_color = base_color
    child_color = lighten_color(base_color, 0.4)
    
    # Build node colors
    node_colors = ["#005FAC"]  # Total
    node_colors += [parent_color for p in parent_list]  # Parent nodes
    node_colors += [child_color for c in child_list]  # Child nodes
    
    # Calculate node values for hover information
    node_values = [total_value]  # Total node value
    node_values += [float(agg_parent[agg_parent[parent_col] == p][metric].iloc[0]) for p in parent_list]
    node_values += [float(agg_child[agg_child[child_col] == c][metric].iloc[0]) for c in child_list]
    
    # Create node hover texts with percentages
    node_hover_texts = []
    for i, (label, value) in enumerate(zip(labels, node_values)):
        percentage = (value / total_value) * 100
        if i == 0:
            node_hover_texts.append(f"<b>{label}</b><br>{format_rupiah(value)}<br>{percentage:.1f}% dari {metric}")
        else:
            node_hover_texts.append(f"<b>{label}</b><br>{format_rupiah(value)}<br>{percentage:.1f}% dari {metric}")
    
    # Helper function to distribute nodes vertically centered at 0.5
    def distribute_y(n):
        """Distribute n nodes symmetrically around center (0.5)"""
        if n == 0:
            return []
        if n == 1:
            return [0.5]  # Single node at center
        
        # Create gap around center for visual separation
        gap = 0.02
        
        if n % 2 == 0:  # Even number of nodes
            half = n // 2
            positions = []
            
            # Bottom half: from 0.1 to (0.5 - gap)
            bottom_range = (0.5 - gap) - 0.1
            if half > 1:
                positions += [0.1 + bottom_range * i / (half - 1) for i in range(half)]
            else:
                positions.append(0.5 - gap - bottom_range / 2)
            
            # Top half: from (0.5 + gap) to 0.9
            top_range = 0.9 - (0.5 + gap)
            if half > 1:
                positions += [0.5 + gap + top_range * i / (half - 1) for i in range(half)]
            else:
                positions.append(0.5 + gap + top_range / 2)
                
        else:  # Odd number of nodes
            half = n // 2
            positions = []
            
            # Bottom half
            if half > 0:
                bottom_range = (0.5 - gap) - 0.1
                if half > 1:
                    positions += [0.1 + bottom_range * i / (half - 1) for i in range(half)]
                else:
                    positions.append(0.3)
            
            # Center node at 0.5
            positions.append(0.5)
            
            # Top half
            if half > 0:
                top_range = 0.9 - (0.5 + gap)
                if half > 1:
                    positions += [0.5 + gap + top_range * i / (half - 1) for i in range(half)]
                else:
                    positions.append(0.7)
        
        return positions
    
    # Build node positions
    node_x = []
    node_y = []
    
    # Total node
    node_x.append(0.01)
    node_y.append(0.5)
    
    # Parent nodes
    parent_y_positions = distribute_y(len(parent_list))
    for y_pos in parent_y_positions:
        node_x.append(0.25)
        node_y.append(y_pos)
    
    # Child nodes - large x gap from parent
    child_y_positions = distribute_y(len(child_list))
    for y_pos in child_y_positions:
        node_x.append(0.99)
        node_y.append(y_pos)
    
    # Clamp values to avoid edge issues (as in the example)
    node_x = [0.001 if v == 0 else 0.999 if v == 1 else v for v in node_x]
    node_y = [0.001 if v == 0 else 0.999 if v == 1 else v for v in node_y]
    
    # Build sankey
    sankey = go.Sankey(
        arrangement="snap",  # Changed from "freeform"
        node=dict(
            label=labels,
            color=node_colors, 
            pad=15,  # Changed from 10
            thickness=20,  # Changed from 10
            line=dict(color="white", width=1),  # Added
            customdata=node_hover_texts,
            hovertemplate="%{customdata}<extra></extra>",
            x=node_x,
            y=node_y,
        ),
        link=dict(
            source=sources, 
            target=targets, 
            value=values, 
            customdata=link_labels, 
            hovertemplate="%{customdata}<extra></extra>",
            color="rgba(0, 95, 172, 0.2)",
            hovercolor="gold",
        )
    )
    fig = go.Figure(sankey)
    
    kl_text = f"<br>{selected_kl}" if selected_kl != "Semua" else ""

    total_nodes = 1 + len(parent_list) + len(child_list)
    chart_height = max(500, total_nodes * 35)
    
    fig.update_layout(
        title=dict(
            text=f"ALOKASI {metric}<br>BERDASARKAN {parent_col} & {child_col}<br>TAHUN {selected_year}{kl_text}",
            x=0.5,
            xanchor='center',
            font=dict(size=14)
        ),
        font=dict(size=9), 
        height=chart_height,
        margin=dict(l=20, r=20, t=130, b=20)
    )
    
    fig.update_traces(
        textfont_color="#005FAC",
        textfont_shadow=False,
        node_align="center"
    )
    return fig
    
def create_placeholder_chart(title, chart_type="bar"):
    """Create placeholder charts for the other 3 positions"""
    fig = go.Figure()
    
    if chart_type == "bar":
        fig.add_trace(go.Bar(
            x=["2023", "2024", "2025"],
            y=[100, 150, 120],
            marker=dict(color="#1a73e8")
        ))
    elif chart_type == "line":
        fig.add_trace(go.Scatter(
            x=["2023", "2024", "2025"],
            y=[100, 150, 120],
            mode="lines+markers",
            marker=dict(color="#34a853", size=10)
        ))
    elif chart_type == "pie":
        fig.add_trace(go.Pie(
            labels=["A", "B", "C"],
            values=[30, 40, 30],
            marker=dict(colors=["#1a73e8", "#34a853", "#f9ab00"])
        ))
    
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=500
    )
    
    return fig

# =============================================================================
# Component Architecture
# =============================================================================
def header(selected_kl: str | None = None):
    """Create comprehensive dashboard header"""
    kl_text = "Semua K/L" if selected_kl == "Semua" else (selected_kl)
    st.markdown(f"""
    <div class="dashboard-header">
        <div class="breadcrumb">DASHBOARD / ANALISIS / {kl_text}</div>
        <h1 class="dashboard-title">📊 Dashboard Analisis Anggaran & Realisasi Belanja Negara</h1>
    </div>
    """, unsafe_allow_html=True)

def sidebar(df):
    """Create sidebar with filters"""
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-section">
            <h3 style='margin: 0.1rem 0.1rem 0.1rem 0.1rem; color: var(--on-surface);'>Filter Data</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Select K/L
        kl_list = sorted(df["KEMENTERIAN/LEMBAGA"].dropna().unique())
        kl_list.insert(0, "Semua")  # add "Semua" at the beginning
        
        selected_kl = st.selectbox(
            "Pilih Kementerian/Lembaga",
            kl_list,
            key="ministry_select",
            help="Pilih kementerian/lembaga untuk melihat analisis anggaran"
        )
        
    return selected_kl

# =============================================================================
# Main Application
# =============================================================================
def main():
    # Load data
    df = load_data()
    
    if df.empty:
        st.warning("⚠️ Tidak ada data untuk ditampilkan")
        return
    
    # Sidebar
    selected_kl = sidebar(df)
    
    # Header
    header(selected_kl)
    
    # Filter data by KL
    if selected_kl == "Semua":
        df_filtered = df.copy()
    else:
        df_filtered = df[df["KEMENTERIAN/LEMBAGA"] == selected_kl]

    # Detect numeric columns
    numeric_cols = df_filtered.select_dtypes(include=["int64", "float64"]).columns.tolist()
    if "Tahun" in numeric_cols:
        numeric_cols.remove("Tahun")
    
    # Get categorical columns for parent/child selection
    categorical_cols = df_filtered.select_dtypes(include=['object']).columns.tolist()
    # Remove some columns that shouldn't be used
    exclude_cols = ['KEMENTERIAN/LEMBAGA', 'Tahun']
    categorical_cols = [col for col in categorical_cols if col not in exclude_cols]
    
    # === Dispay Charts ===
    col1, col2 = st.columns(2)
    
    # Column 1: Year slider, metric selectors, and chart
    # Use st.container
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        with st.container():     
            # Row 1: Year range slider (full width of column)
            year_options = sorted(df_filtered["Tahun"].dropna().unique())
            if len(year_options) >= 2:
                selected_years = st.slider(
                    "Rentang Tahun",
                    min_value=int(min(year_options)),
                    max_value=int(max(year_options)),
                    value=(int(min(year_options)), int(max(year_options))),
                    step=1,
                    key="year_range_main"
                )
            else:
                st.info("Hanya satu tahun tersedia dalam data")
                selected_years = (int(year_options[0]), int(year_options[0]))
            
            # Row 2: Two columns for metric selection
            colA, colB = st.columns(2)
            with colA:
                # Primary metric selector
                primary = st.selectbox(
                    "Pilih metrik pertama",
                    numeric_cols,
                    index=numeric_cols.index("PAGU DIPA REVISI EFEKTIF") if "PAGU DIPA REVISI EFEKTIF" in numeric_cols else 0,
                    key="primary_metric",
                    label_visibility="visible"
                )
            
            with colB:
                # Secondary metric selector
                secondary = st.selectbox(
                    "Pilih metrik kedua",
                    numeric_cols,
                    index=numeric_cols.index("REALISASI BELANJA KL (SAKTI)") if "REALISASI BELANJA KL (SAKTI)" in numeric_cols else 0,
                    key="secondary_metric",
                    label_visibility="visible"
                )
            
            # Row 3: Chart
            fig1 = create_time_series_chart(df, selected_kl, selected_years, primary, secondary)
            st.plotly_chart(fig1, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Column 2: Placeholder chart
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        fig2 = create_placeholder_chart("Grafik 2", "line")
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    col3, col4 = st.columns(2)
    
    # Column 3: Sankey Chart with selectors
    with col3:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        # Row 1: Year and Metric selectors
        colC, colD = st.columns(2)
        with colC:
            year_options_sankey = sorted(df_filtered["Tahun"].dropna().unique())
            if year_options_sankey:
                selected_year_sankey = st.selectbox(
                    "Tahun",
                    year_options_sankey,
                    key="year_sankey",
                    label_visibility="visible"
                )
            else:
                selected_year_sankey = 2025
        
        with colD:
            selected_metric_sankey = st.selectbox(
                "Metrik",
                numeric_cols,
                index=numeric_cols.index("REALISASI BELANJA KL (SAKTI)") if "REALISASI BELANJA KL (SAKTI)" in numeric_cols else 0,
                key="metric_sankey",
                label_visibility="visible"
            )
            
        # Row 2: Parent and Child selectors
        colE, colF = st.columns(2)
        with colE:
            parent_sankey = st.selectbox(
                "Parent",
                categorical_cols,
                index=categorical_cols.index("JENIS BELANJA") if "JENIS BELANJA" in categorical_cols else 0,
                key="parent_sankey",
                label_visibility="visible"
            )
        
        with colF:
            child_sankey = st.selectbox(
                "Child",
                categorical_cols,
                index=categorical_cols.index("FUNGSI") if "FUNGSI" in categorical_cols else 0,
                key="child_sankey",
                label_visibility="visible"
            )
            
        # NOW calculate dynamic container height from data
        df_temp = df[df["Tahun"] == int(selected_year_sankey)].copy()
        if selected_kl != "Semua":
            df_temp = df_temp[df_temp["KEMENTERIAN/LEMBAGA"] == selected_kl]
            
        num_parents = df_temp[parent_sankey].nunique() if parent_sankey in df_temp.columns else 0
        num_children = df_temp[child_sankey].nunique() if child_sankey in df_temp.columns else 0
        total_nodes = 1 + num_parents + num_children
            
        container_height = min(800, max(500, total_nodes * 25))  # Dynamic height, capped at 800px
            
        # Create and display Sankey chart
        fig3 = create_sankey_chart(df, selected_kl, selected_year_sankey, selected_metric_sankey, parent_sankey, child_sankey)
        
        with st.container(height=600, border=False):
            st.plotly_chart(fig3, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    # Column 4: Placeholder chart
    with col4:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        fig4 = create_placeholder_chart("Grafik 4", "bar")
        st.plotly_chart(fig4, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # --- Footer ---
    st.markdown("---")
    footer_col1, footer_col2 = st.columns([3, 1])
    with footer_col1:
        st.caption("📊 Sumber Data: bidja.kemenkeu.go.id")
    with footer_col2:
        st.caption(f"🕐 Diperbarui: {datetime.now().strftime('%d %B %Y %H:%M')}")

# =============================================================================
# Error Handling & Entry Point
# =============================================================================
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Terjadi kesalahan dalam aplikasi: {str(e)}")
        st.info("Silakan refresh halaman atau hubungi administrator.")

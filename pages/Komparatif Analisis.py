import streamlit as st
import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import zipfile
import io
import requests
from datetime import datetime

# =============================================================================
# Page Configuration
# =============================================================================
st.set_page_config(
    page_title="Dashboard Komparasi Realisasi vs Pagu DIPA",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://www.kemenkeu.go.id",
        "Report a bug": "https://github.com/tubankum3/dashpmk/issues",
        "About": "Dashboard Anggaran Bidang PMK"
    }
)

st.title("Dashboard Komparasi Realisasi vs Pagu DIPA")
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
    --surface: #ffffff;
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
    padding: 1.5rem;
    margin-bottom: 1rem;
    box-shadow: var(--shadow-1);
}

/* Tab */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
}

.stTabs [data-baseweb="tab"] {
    background: var(--surface);
    border-radius: var(--border-radius) var(--border-radius) 0 0;
    padding: 0.75rem 1.5rem;
    border: 1px solid #e8eaed;
    transition: var(--transition);
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

/* High contrast support */
@media (prefers-contrast: high) {
    .metric-card {
        border: 2px solid var(--on-surface);
    }
}

/* Reduced motion support */
@media (prefers-reduced-motion: reduce) {
    .material-card, .metric-card, .stButton>button {
        transition: none;
    }
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

/* Loading states */
.loading-skeleton {
    background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
    background-size: 200% 100%;
    animation: loading 1.5s infinite;
}

@keyframes loading {
    0% { background-position: 200% 0; }
    100% { background-position: -200% 0; }
}

/* Chart container */
.chart-container {
    background: var(--surface);
    border-radius: var(--border-radius);
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    box-shadow: var(--shadow-1);
    border: 1px solid #e8eaed;
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
@st.cache_data(show_spinner="Memuat dataset anggaran...")
def load_data():
    url = "https://raw.githubusercontent.com/tubankum3/dashpmk/main/df.csv.zip"
    try:
        response = requests.get(url)
        response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            with z.open("df.csv") as file:
                df = pd.read_csv(file, low_memory=False)

        if "Unnamed: 0" in df.columns:
            df.drop(columns=["Unnamed: 0"], inplace=True)
        if "Tahun" in df.columns:
            df["Tahun"] = df["Tahun"].astype(str)
        return df

    except Exception as e:
        st.error(f"Gagal memuat data: {e}")
        return pd.DataFrame()

# =============================================================================
# Utility
# =============================================================================
def format_rupiah(value):
    if pd.isna(value) or value == 0:
        return "Rp 0"
    abs_val = abs(value)
    if abs_val >= 1_000_000_000_000:
        return f"Rp {value/1_000_000_000_000:.2f} T"
    elif abs_val >= 1_000_000_000:
        return f"Rp {value/1_000_000_000:.2f} M"
    elif abs_val >= 1_000_000:
        return f"Rp {value/1_000_000:.2f} Jt"
    return f"Rp {value:,.0f}"

# =============================================================================
# Component Architecture
# =============================================================================
def header(selected_kl: str | None = None, selected_metric: str | None = None):
    """Create comprehensive dashboard header with breadcrumb and key info"""
    kl_text = selected_kl if selected_kl else "Overview"
    metric_text = f" {selected_metric}" if selected_metric else ""
    st.markdown(f"""
    <div class="dashboard-header">
        <div class="breadcrumb">DASHBOARD / KOMPARASI {metric_text} / {kl_text}</div>
        <h1 class="dashboard-title">📊 Dashboard Komparasi Realisasi vs Pagu DIPA</h1>
    </div>
    """, unsafe_allow_html=True)
    
def sidebar(df):
    with st.sidebar:
        st.title("Komparasi Anggaran & Realisasi Belanja Negara")
        st.markdown("---")

        kl_list = sorted(df["KEMENTERIAN/LEMBAGA"].dropna().unique())
        selected_kl = st.selectbox("Pilih Kementerian/Lembaga", kl_list)

        df_filtered = df[df["KEMENTERIAN/LEMBAGA"] == selected_kl]
        numeric_cols = [c for c in df_filtered.select_dtypes(include=["int64", "float64"]).columns if c != "Tahun"]
        selected_metric = st.selectbox("Metrik Anggaran", numeric_cols)

        years = sorted(df_filtered["Tahun"].astype(int).unique())
        selected_years = st.slider(
            "Rentang Tahun",
            min_value=int(min(years)),
            max_value=int(max(years)),
            value=(int(min(years)), int(max(years)))
        )
    return df_filtered, selected_kl, selected_metric, selected_years

def chart(df: pd.DataFrame, selected_metric: str, selected_years: tuple, top_n: int = 10):
    """
    Create an auto-scaled line chart highlighting top N ministries by budget.
    """
    # Ensure Tahun numeric and valid
    df["Tahun"] = pd.to_numeric(df["Tahun"], errors="coerce")
    df = df.dropna(subset=["Tahun", "KEMENTERIAN/LEMBAGA", "Nilai"])
    
    # Filter year range
    if selected_years and selected_years != (None, None):
        start_year, end_year = selected_years
        df = df[(df["Tahun"] >= start_year) & (df["Tahun"] <= end_year)]

    # Aggregate total per K/L per year
    df_grouped = (
        df.groupby(["KEMENTERIAN/LEMBAGA", "Tahun"], as_index=False)["Nilai"]
          .sum()
    )

    # Find top N ministries by the latest year's total
    latest_year = df_grouped["Tahun"].max()
    top_ministries = (
        df_grouped[df_grouped["Tahun"] == latest_year]
        .nlargest(top_n, "Nilai")["KEMENTERIAN/LEMBAGA"]
        .tolist()
    )

    # Assign colors — highlight top N, fade others
    df_grouped["Highlight"] = df_grouped["KEMENTERIAN/LEMBAGA"].apply(
        lambda x: "Top" if x in top_ministries else "Others"
    )

    # Calculate height dynamically
    n_kl = df_grouped["KEMENTERIAN/LEMBAGA"].nunique()
    height = 500 + (n_kl * 12 if n_kl > 15 else 0)

    # Plot
    fig = px.line(
        df_grouped,
        x="Tahun",
        y="Nilai",
        color="KEMENTERIAN/LEMBAGA",
        line_group="KEMENTERIAN/LEMBAGA",
        markers=True,
        title=f"📊 Tren {selected_metric} — Top {top_n} K/L berdasarkan Anggaran {latest_year}",
        labels={"Nilai": "Nilai (Rp)", "Tahun": "Tahun"},
        template="plotly_white",
        height=height,
    )

    # Adjust color opacity
    for trace in fig.data:
        if trace.name not in top_ministries:
            trace.line.color = "lightgray"
            trace.line.width = 1.5
            trace.opacity = 0.4
        else:
            trace.line.width = 3.5

    fig.update_layout(
        hovermode="x unified",
        legend_title_text="Kementerian/Lembaga",
        title_x=0,
        margin=dict(l=40, r=40, t=80, b=40),
        font=dict(family="Google Sans, Roboto, Arial", size=12),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    fig.update_traces(
        hovertemplate="<b>%{fullData.name}</b><br>Tahun: %{x}<br>Rp %{y:,.0f}<extra></extra>"
    )

    return fig

# =============================================================================
# Main
# =============================================================================
def main():
    df = load_data()
    if df.empty:
        st.error("Data gagal dimuat.")
        return

    df_filtered, selected_kl, selected_metric, selected_years = sidebar(df)
    st.markdown(f"### 📘 {selected_kl}")
    st.plotly_chart(chart(df_filtered, selected_metric, selected_years), use_container_width=True)
    
    # --- Footer ---
    st.markdown("---")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.caption("📊 Sumber Data: bidja.kemenkeu.go.id")
    with col2:
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

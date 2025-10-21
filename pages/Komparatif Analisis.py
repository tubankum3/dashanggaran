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
    page_icon=":analytics:",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://www.kemenkeu.go.id",
        "Report a bug": "https://github.com/tubankum3/dashpmk/issues",
        "About": "Dashboard Anggaran Bidang PMK"
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
    padding: 0.5rem;
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
def header(selected_year: str | None = None):
    """Create comprehensive dashboard header with breadcrumb and key info"""
    year_text = selected_year if selected_year else "Overview"
    st.markdown(f"""
    <div class="dashboard-header">
        <div class="breadcrumb">DASHBOARD / KOMPARASI / TAHUN {year_text}</div>
        <h1 class="dashboard-title">Dashboard Komparasi Realisasi vs Pagu DIPA</h1>
    </div>
    """, unsafe_allow_html=True)
    
def sidebar(df):
    with st.sidebar:
        kl_list = sorted(df["KEMENTERIAN/LEMBAGA"].dropna().unique())
        selected_kls = st.multiselect(
            "Pilih Kementerian/Lembaga (bisa lebih dari satu)",
            options=kl_list,
            default=[]
        )

        top_n = st.number_input(
            "Tampilkan Top K/L berdasarkan Pagu DIPA Revisi",
            min_value=1,
            max_value=50,
            value=10,
            step=1,
            help="Jumlah Kementerian/Lembaga yang ditampilkan pada grafik."
        )

    return selected_kls, top_n

def chart(df: pd.DataFrame, year: int, top_n: int = 10):
    df["Tahun"] = pd.to_numeric(df["Tahun"], errors="coerce")
    df_year = df[df["Tahun"] == year].copy()

    # Remove placeholder KL
    df_year = df_year[df_year["KEMENTERIAN/LEMBAGA"] != "999 BAGIAN ANGGARAN BENDAHARA UMUM NEGARA"]

    # Aggregate
    agg = (
        df_year.groupby("KEMENTERIAN/LEMBAGA", as_index=False)[
            ["REALISASI BELANJA KL (SAKTI)", "PAGU DIPA AWAL EFEKTIF", "PAGU DIPA REVISI EFEKTIF"]
        ].sum()
    )

    # Sort and select top_n
    agg = agg.sort_values("PAGU DIPA REVISI EFEKTIF", ascending=True).tail(top_n)

    # Range bar colors
    colors = [
        "#b2dfdb" if row["PAGU DIPA REVISI EFEKTIF"] >= row["PAGU DIPA AWAL EFEKTIF"] else "#d0d0d0"
        for _, row in agg.iterrows()
    ]

    # Marker colors
    marker_colors = [
        "#e53935" if row["REALISASI BELANJA KL (SAKTI)"] < min(row["PAGU DIPA AWAL EFEKTIF"], row["PAGU DIPA REVISI EFEKTIF"]) else "#00897b"
        for _, row in agg.iterrows()
    ]

    # --- Figure ---
    fig = go.Figure()

    # Range bars
    fig.add_trace(go.Bar(
        y=agg["KEMENTERIAN/LEMBAGA"],
        x=(agg["PAGU DIPA REVISI EFEKTIF"] - agg["PAGU DIPA AWAL EFEKTIF"]).abs(),
        base=agg[["PAGU DIPA REVISI EFEKTIF", "PAGU DIPA AWAL EFEKTIF"]].min(axis=1),
        orientation="h",
        marker=dict(color=colors),
        name="Rentang Pagu DIPA Efektif (Awal–Revisi)",
        hovertemplate=(
            "Pagu Awal: %{base:,.0f}<br>"
            "Pagu Revisi: %{customdata:,.0f}<extra></extra>"
        ),
        customdata=agg["PAGU DIPA REVISI EFEKTIF"]
    ))

    # Realisasi markers
    fig.add_trace(go.Scatter(
        y=agg["KEMENTERIAN/LEMBAGA"],
        x=agg["REALISASI BELANJA KL (SAKTI)"],
        mode="markers",
        marker=dict(color=marker_colors, size=12, line=dict(color="white", width=1.5)),
        name="Realisasi Belanja (SAKTI)",
        hovertemplate="Realisasi: %{x:,.0f}<extra></extra>"
    ))

    # Format x-axis with rupiah labels
    tickvals = np.linspace(0, agg["PAGU DIPA REVISI EFEKTIF"].max(), num=6)
    ticktext = [format_rupiah(val) for val in tickvals]

    # Layout
    fig.update_layout(
        title=f"Perbandingan Realisasi Belanja dengan Rentang Pagu DIPA Awal dan Revisi (Efektif)<br>Tahun {year}",
        xaxis_title="Jumlah (Rupiah)",
        yaxis_title="Kementerian / Lembaga",
        barmode="overlay",
        template="plotly_white",
        height=800,
        xaxis=dict(
            showgrid=True,
            zeroline=False,
            tickvals=tickvals,
            ticktext=ticktext
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=100, b=40)
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

    # Sidebar
    selected_kls, top_n = sidebar(df)

    # Show header at the top
    header(str(selected_year))

    # Filter K/L if selected
    if selected_kls:
        df = df[df["KEMENTERIAN/LEMBAGA"].isin(selected_kls)]
    
    # year selection to main
    years = sorted(df["Tahun"].astype(int).unique())
    selected_year = st.selectbox("Pilih Tahun", options=years, index=len(years)-1)
    
    # Chart
    st.plotly_chart(chart(df, selected_year, top_n), use_container_width=True)

    # Footer
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







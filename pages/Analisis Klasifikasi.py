import streamlit as st
import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events
import zipfile
import io
import requests
from datetime import datetime

# =============================================================================
# Page Configuration
# =============================================================================
st.set_page_config(
    page_title="Dashboard Klasifikasi Anggaran",
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
        <div class="breadcrumb">DASHBOARD / KLASIFIKASI / TAHUN {year_text}</div>
        <h1 class="dashboard-title">Dashboard Klasifikasi Anggaran</h1>
    </div>
    """, unsafe_allow_html=True)
    
def sidebar(df):
    with st.sidebar:
        st.markdown("### ⚙️ Pengaturan Dashboard")

        # --- Tahun ---
        years = sorted(df["Tahun"].astype(int).unique())
        default_year_index = years.index(2025) if 2025 in years else len(years) - 1
        selected_year = st.selectbox("Pilih Tahun", years, index=default_year_index)

        # --- Filter K/L ---
        kl_list = sorted(df["KEMENTERIAN/LEMBAGA"].dropna().unique())
        selected_kls = st.multiselect(
            "Pilih Kementerian/Lembaga (bisa lebih dari satu)",
            options=["Semua"] + kl_list,
            default=["Semua"]
        )

        # --- Jumlah Top K/L ---
        top_n = st.number_input(
            "Tampilkan Top K/L berdasarkan nilai metrik",
            min_value=1,
            max_value=50,
            value=10,
            step=1,
            help="Jumlah Kementerian/Lembaga yang ditampilkan pada grafik."
        )

        # --- Pilih Metrik ---
        df_filtered = df.copy()
        numeric_cols = df_filtered.select_dtypes(include=["int64", "float64"]).columns.tolist()
        if "Tahun" in numeric_cols:
            numeric_cols.remove("Tahun")

        metric_options = numeric_cols if numeric_cols else ["(Tidak ada kolom numerik)"]
        selected_metric = st.selectbox(
            "Metrik Anggaran",
            options=metric_options,
            index=metric_options.index("REALISASI BELANJA KL (SAKTI)") 
            if "REALISASI BELANJA KL (SAKTI)" in metric_options else 0,
            key="metric_select",
            help="Pilih jenis anggaran yang akan dianalisis."
        )

    # === Return clean values ===
    # If user picks “Semua”, treat as full dataset
    if "Semua" in selected_kls:
        selected_kls = []

    return selected_year, selected_kls, top_n, selected_metric

# =============================================================================
# Charts
# =============================================================================
def treemap_chart(df):
    agg = df.groupby(["FUNGSI", "SUB FUNGSI"], as_index=False)["REALISASI BELANJA KL (SAKTI)"].sum()
    total = agg["REALISASI BELANJA KL (SAKTI)"].sum()
    agg["Share (%)"] = 100 * agg["REALISASI BELANJA KL (SAKTI)"] / total

    fig = px.treemap(
        agg, path=["FUNGSI", "SUB FUNGSI"], values="REALISASI BELANJA KL (SAKTI)",
        color="REALISASI BELANJA KL (SAKTI)", color_continuous_scale="Blues"
    )
    fig.update_traces(
        hovertemplate="<b>%{label}</b><br>Realisasi: %{value:,.0f}<br>Share: %{percentParent:.2%}<extra></extra>",
        textinfo="label+value+percent parent"
    )
    fig.update_layout(title="Realisasi per Fungsi dan Sub Fungsi", height=500, margin=dict(t=40, l=0, r=0, b=0))

    st.subheader("🧭 Peta Fungsi dan Sub Fungsi (Treemap)")
    selected = plotly_events(fig, click_event=True, hover_event=False)
    if selected:
        clicked = selected[0]["label"]
        st.info(f"Menampilkan rincian untuk: **{clicked}**")
        return clicked
    return None

def bar_chart(df, group_col, title, color="#1a73e8"):
    agg = df.groupby(group_col, as_index=False)["REALISASI BELANJA KL (SAKTI)"].sum()
    agg = agg.sort_values("REALISASI BELANJA KL (SAKTI)", ascending=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=agg["REALISASI BELANJA KL (SAKTI)"],
        y=agg[group_col],
        orientation="h",
        text=agg["REALISASI BELANJA KL (SAKTI)"].apply(format_rupiah),
        textposition="outside",
        marker_color=color,
        hovertemplate="%{y}<br>Realisasi: %{x:,.0f}<extra></extra>"
    ))

    tickvals = np.linspace(0, agg["REALISASI BELANJA KL (SAKTI)"].max(), num=6)
    ticktext = [format_rupiah(val) for val in tickvals]
    fig.update_layout(
        title=title,
        xaxis=dict(tickvals=tickvals, ticktext=ticktext),
        yaxis=dict(title=None),
        template="plotly_white",
        height=400,
        margin=dict(l=0, r=0, t=40, b=40)
    )
    return fig

def show_child_charts(df, clicked_node):
    if clicked_node:
        df_filtered = df[
            (df["FUNGSI"] == clicked_node) | (df["SUB FUNGSI"] == clicked_node)
        ]
    else:
        df_filtered = df

    st.subheader("📈 Rincian Realisasi")
    st.plotly_chart(bar_chart(df_filtered, "PROGRAM", "Realisasi per Program", color="#1a73e8"), use_container_width=True)
    st.plotly_chart(bar_chart(df_filtered, "KEGIATAN", "Realisasi per Kegiatan", color="#34a853"), use_container_width=True)
    st.plotly_chart(bar_chart(df_filtered, "OUTPUT (KRO)", "Realisasi per Output (KRO)", color="#fbbc04"), use_container_width=True)
    st.plotly_chart(bar_chart(df_filtered, "SUB OUTPUT (RO)", "Realisasi per Sub Output (RO)", color="#ea4335"), use_container_width=True)

# =============================================================================
# Main
# =============================================================================
def main():
    df = load_data()
    if df.empty:
        st.error("Data gagal dimuat.")
        return

    selected_year, selected_kls, top_n, selected_metric = sidebar(df)

    header(selected_year)

    df_year = df[df["Tahun"] == str(selected_year)]
    if selected_kls:
        df_year = df_year[df_year["KEMENTERIAN/LEMBAGA"].isin(selected_kls)]

    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    clicked_node = treemap_chart(df_year)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    show_child_charts(df_year, clicked_node)
    st.markdown('</div>', unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.caption("📊 Sumber Data: bidja.kemenkeu.go.id")
    with col2:
        st.caption(f"🕐 Diperbarui: {datetime.now().strftime('%d %B %Y %H:%M')}")

# =============================================================================
# Entry Point
# =============================================================================
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Terjadi kesalahan dalam aplikasi: {str(e)}")
        st.info("Silakan refresh halaman atau hubungi administrator.")
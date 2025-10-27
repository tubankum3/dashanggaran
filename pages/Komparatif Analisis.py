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
def header(selected_year: str | None = None, selected_metric: str | None = None, selected_kls: list | None = None):
    """Create comprehensive dashboard header with breadcrumb and key info"""
    year_text = selected_year if selected_year else "Overview"
    metric_text = f" {selected_metric}" if selected_metric else ""
    kl_text = ", ".join(selected_kls) if selected_kls else "KEMENTERIAN/LEMBAGA"
    st.markdown(f"""
    <div class="dashboard-header">
        <div class="breadcrumb">DASHBOARD / KOMPARASI {metric_text} / {kl_text} / TAHUN {year_text}</div>
        <h1 class="dashboard-title">Dashboard Komparasi Realisasi vs Pagu DIPA</h1>
    </div>
    """, unsafe_allow_html=True)
    
def sidebar(df):
    with st.sidebar:
        years = sorted(df["Tahun"].astype(int).unique())
        selected_year = st.selectbox("Pilih Tahun", options=years, index=len(years)-1)

        category_cols = [col for col in df.select_dtypes(include="object").columns if col != "Tahun"]
        
        selected_metric = st.sidebar.selectbox(
            "Kategori / Dimensi Analisis",
            options=category_cols,
            index=category_cols.index("KEMENTERIAN/LEMBAGA") if "KEMENTERIAN/LEMBAGA" in category_cols else 0,
        )

        top_n = st.number_input(
            "Tampilkan Top K/L berdasarkan Pagu DIPA Revisi",
            min_value=1,
            max_value=50,
            value=10,
            step=1,
            help="Jumlah Kementerian/Lembaga yang ditampilkan pada grafik."
        )
        
        kl_list = sorted(df["KEMENTERIAN/LEMBAGA"].dropna().unique())
        selected_kls = st.multiselect(
            "Pilih Kementerian/Lembaga (bisa lebih dari satu)",
            options=kl_list,
            default=[]
        )

    return selected_year, selected_kls, top_n, selected_metric

# =============================================================================
# Chart
# =============================================================================
def comparison_chart(df, year, top_n, col_start, col_end, title_suffix,
                     color_range="#b2dfdb", color_marker="#1a73e8"):
    """Chart builder showing Pagu ranges, Realisasi markers, and Varian lines with end caps.
       Y axis is numeric positions so we can draw caps reliably; labels are set as ticktext.
    """
    df_year = df[df["Tahun"].astype(int) == year].copy()
    df_year = df_year[df_year["KEMENTERIAN/LEMBAGA"] != "999 BAGIAN ANGGARAN BENDAHARA UMUM NEGARA"]

    agg = (
        df_year.groupby("KEMENTERIAN/LEMBAGA", as_index=False)[
            ["REALISASI BELANJA KL (SAKTI)", col_start, col_end]
        ].sum()
    ).sort_values("REALISASI BELANJA KL (SAKTI)", ascending=False).head(top_n).reset_index(drop=True)

    # Calculate variance
    agg["VARIANS"] = agg[col_end] - agg["REALISASI BELANJA KL (SAKTI)"]
                         
    # numeric y positions (0..n-1)
    y_pos = np.arange(len(agg))

    fig = go.Figure()

    # Range Bar (Awal–Revisi)
    fig.add_trace(go.Bar(
        y=y_pos,
        x=(agg[col_end] - agg[col_start]),
        base=agg[col_start],
        orientation="h",
        width=0.6,
        marker=dict(color=color_range, cornerradius=15, line=dict(color=color_range, width=0.5)),
        name=f"Rentang {' '.join(col_start.split()[-3:])}–{' '.join(col_end.split()[-3:])}",
        hovertemplate=(f"{col_start}: %{{base:,.0f}}<br>"
                       f"{col_end}: %{{customdata:,.0f}}<extra></extra>"),
        customdata=agg[col_end]
    ))

    # Varians line + caps
    cap_size = 0.1  # adjust for visual; smaller if many rows
    for i, row in agg.iterrows():
        x_real = row["REALISASI BELANJA KL (SAKTI)"]
        x_pagu = row[col_end]
        y = y_pos[i]

        # choose color: black if Realisasi < Pagu (underspend), red if overspend
        var_color = "black" if x_real < x_pagu else "red"

        # main horizontal line
        fig.add_trace(go.Scatter(
            x=[x_real, x_pagu],
            y=[y, y],
            mode="lines",
            line=dict(color=var_color, width=1),
            showlegend=False,
            hoverinfo="skip"
        ))

        # left cap (at Realisasi)
        fig.add_trace(go.Scatter(
            x=[x_real, x_real],
            y=[y + cap_size, y - cap_size],
            mode="lines",
            line=dict(color=var_color, width=1),
            showlegend=False,
            hoverinfo="skip"
        ))

        # right cap (at Pagu)
        fig.add_trace(go.Scatter(
            x=[x_pagu, x_pagu],
            y=[y + cap_size, y - cap_size],
            mode="lines",
            line=dict(color=var_color, width=1),
            showlegend=False,
            hoverinfo="skip"
        ))

    # Realisasi Marker
    fig.add_trace(go.Scatter(
        y=y_pos,
        x=agg["REALISASI BELANJA KL (SAKTI)"],
        mode="markers",
        marker=dict(color=color_marker, size=12, line=dict(color="white", width=1)),
        name="Realisasi Belanja (SAKTI)",
        hovertemplate=(
            "Realisasi: %{x:,.0f}<br>"
            "Varian (Pagu Efektif-Realisasi): %{customdata:,.0f}<extra></extra>"
        ),
        customdata=agg["VARIANS"]
    ))

    # x ticks (formatted rupiah)
    tickvals = np.linspace(0, max(agg[col_end].max(), agg["REALISASI BELANJA KL (SAKTI)"].max()), num=6)
    ticktext = [format_rupiah(val) for val in tickvals]

    # set y axis tick labels to K/L names
    y_ticktext = agg["KEMENTERIAN/LEMBAGA"].tolist()

    fig.update_layout(
        title=f"Perbandingan Realisasi Belanja {title_suffix}<br>Tahun {year}",
        xaxis_title="Jumlah (Rupiah)",
        yaxis_title="Kementerian / Lembaga",
        template="plotly_white",
        height= max(500, 50 * len(agg)),  # adapt height a bit to rows
        xaxis=dict(showgrid=True, zeroline=False, tickvals=tickvals, ticktext=ticktext),
        yaxis=dict(tickmode="array", tickvals=list(y_pos), ticktext=y_ticktext, autorange="reversed"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=250, r=40, t=100, b=40)  # left margin bigger to fit long labels
    )

    return fig

def comparison_chart_by_category(df, year, selected_kls, selected_metric, top_n,
                                 col_start, col_end, title_suffix,
                                 color_range="#b2dfdb", color_marker="#1a73e8"):
    """Chart showing Pagu ranges, Realisasi markers, and Varian lines by selected category.
       Automatically aggregates all K/Ls if none selected. Supports top_n limiting.
    """
    df_year = df[df["Tahun"].astype(int) == year].copy()
    df_year = df_year[df_year["KEMENTERIAN/LEMBAGA"] != "999 BAGIAN ANGGARAN BENDAHARA UMUM NEGARA"]

    # ✅ If user selected K/Ls, filter by them. Otherwise use all data (aggregate nationally)
    if selected_kls:
        df_filtered = df_year[df_year["KEMENTERIAN/LEMBAGA"].isin(selected_kls)]
    else:
        df_filtered = df_year.copy()

    # Group by selected category
    agg = (
        df_filtered.groupby(selected_metric, as_index=False)[
            ["REALISASI BELANJA KL (SAKTI)", col_start, col_end]
        ].sum()
    ).sort_values("REALISASI BELANJA KL (SAKTI)", ascending=False).head(top_n).reset_index(drop=True)

    if agg.empty:
        st.warning(f"Tidak ada data untuk kategori '{selected_metric}' di tahun {year}.")
        return None

    # Calculate variance (Pagu - Realisasi)
    agg["VARIANS"] = agg[col_end] - agg["REALISASI BELANJA KL (SAKTI)"]

    # numeric y positions for consistent line plotting
    y_pos = np.arange(len(agg))

    fig = go.Figure()

    # Range Bar (Awal–Revisi)
    fig.add_trace(go.Bar(
        y=y_pos,
        x=(agg[col_end] - agg[col_start]),
        base=agg[col_start],
        orientation="h",
        width=0.6,
        marker=dict(color=color_range, cornerradius=15, line=dict(color=color_range, width=0.5)),
        name=f"Rentang {' '.join(col_start.split()[-3:])}–{' '.join(col_end.split()[-3:])}",
        hovertemplate=(f"{col_start}: %{{base:,.0f}}<br>"
                       f"{col_end}: %{{customdata:,.0f}}<extra></extra>"),
        customdata=agg[col_end]
    ))

    # Varians line + caps
    cap_size = max(0.05, 0.5 / len(agg))  # dynamically scale cap size based on number of rows
    for i, row in agg.iterrows():
        x_real = row["REALISASI BELANJA KL (SAKTI)"]
        x_pagu = row[col_end]
        y = y_pos[i]
        var_color = "black" if x_real < x_pagu else "red"

        # main variance line
        fig.add_trace(go.Scatter(
            x=[x_real, x_pagu],
            y=[y, y],
            mode="lines",
            line=dict(color=var_color, width=1),
            showlegend=False,
            hoverinfo="skip"
        ))

        # vertical caps
        for x_cap in [x_real, x_pagu]:
            fig.add_trace(go.Scatter(
                x=[x_cap, x_cap],
                y=[y + cap_size, y - cap_size],
                mode="lines",
                line=dict(color=var_color, width=1),
                showlegend=False,
                hoverinfo="skip"
            ))

    # Realisasi marker
    fig.add_trace(go.Scatter(
        y=y_pos,
        x=agg["REALISASI BELANJA KL (SAKTI)"],
        mode="markers",
        marker=dict(color=color_marker, size=12, line=dict(color="white", width=1)),
        name="Realisasi Belanja (SAKTI)",
        hovertemplate=(
            "Realisasi: %{x:,.0f}<br>"
            "Varian (Pagu Efektif-Realisasi): %{customdata:,.0f}<extra></extra>"
        ),
        customdata=agg["VARIANS"]
    ))

    tickvals = np.linspace(0, max(agg[col_end].max(), agg["REALISASI BELANJA KL (SAKTI)"].max()), num=6)
    ticktext = [format_rupiah(val) for val in tickvals]
    y_ticktext = agg[selected_metric].astype(str).tolist()

    fig.update_layout(
        title=(
            f"Perbandingan Realisasi Belanja per {selected_metric}<br>"
            f"Tahun {year}"
            + (f" — K/L Terpilih" if selected_kls else " — Seluruh K/L")
        ),
        xaxis_title="Jumlah (Rupiah)",
        yaxis_title=selected_metric,
        template="plotly_white",
        height=max(500, 50 * len(agg)),
        xaxis=dict(showgrid=True, zeroline=False, tickvals=tickvals, ticktext=ticktext),
        yaxis=dict(tickmode="array", tickvals=list(y_pos), ticktext=y_ticktext, autorange="reversed"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=250, r=40, t=100, b=40)
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

    # Sidebar selections
    selected_year, selected_kls, top_n, selected_metric = sidebar(df)
    
    # Header displayed at the top
    header(str(selected_year), selected_metric, selected_kls)

    # Filter by selected K/Ls (if any)
    if selected_kls:
        df = df[df["KEMENTERIAN/LEMBAGA"].isin(selected_kls)]

    # Tabs for charts
    tab1, tab2, tab3 = st.tabs([
        "1️⃣ Realisasi vs Pagu DIPA Awal dan Revisi (Efektif)",
        "2️⃣ Realisasi vs Pagu DIPA Awal Efektif",
        "3️⃣ Realisasi vs Pagu DIPA Revisi Efektif"
    ])

    with tab1:
        fig1 = comparison_chart(
            df, selected_year, top_n,
            "PAGU DIPA AWAL EFEKTIF", "PAGU DIPA REVISI EFEKTIF",
            "dengan Rentang Pagu DIPA Awal dan Revisi (Efektif)",
            color_range="#b2dfdb", color_marker="#00897b"
        )
        st.plotly_chart(fig1, use_container_width=True)
               
        if selected_metric != "KEMENTERIAN/LEMBAGA":
            fig11 = comparison_chart_by_category(
                df, selected_year, selected_kls, selected_metric, top_n,
                "PAGU DIPA AWAL EFEKTIF", "PAGU DIPA REVISI EFEKTIF",
                "dengan Rentang Pagu per Kategori",
                color_range="#aed581", color_marker="#33691e"
            )
            st.plotly_chart(fig11, use_container_width=True)
        st.caption("*Rentang merupakan _selisih_ antara Pagu Revisi Efektif dan Pagu Awal Efektif")

    with tab2:
        fig2 = comparison_chart(
            df, selected_year, top_n,
            "PAGU DIPA AWAL", "PAGU DIPA AWAL EFEKTIF",
            "dengan Rentang Pagu DIPA Awal dikurangi Blokir DIPA Awal",
            color_range="#c5cae9", color_marker="#1a73e8"
        )
        st.plotly_chart(fig2, use_container_width=True)

        if selected_metric != "KEMENTERIAN/LEMBAGA":
            fig22 = comparison_chart_by_category(
                df, selected_year, selected_kls, selected_metric, top_n,
                "PAGU DIPA AWAL", "PAGU DIPA AWAL EFEKTIF",
                "dengan Rentang Pagu DIPA Awal dikurangi Blokir DIPA Awal per Kategori",
                color_range="#aed581", color_marker="#33691e"
            )
            st.plotly_chart(fig22, use_container_width=True)
        st.caption("*Rentang merupakan besaran :red[Blokir] DIPA Awal")

    with tab3:
        fig3 = comparison_chart(
            df, selected_year, top_n,
            "PAGU DIPA REVISI", "PAGU DIPA REVISI EFEKTIF",
            "dengan Rentang Pagu DIPA Revisi dikurangi Blokir DIPA Revisi",
            color_range="#ffe082", color_marker="#e53935"
        )
        st.plotly_chart(fig3, use_container_width=True)

        if selected_metric != "KEMENTERIAN/LEMBAGA":
            fig33 = comparison_chart_by_category(
                df, selected_year, selected_kls, selected_metric, top_n,
                "PAGU DIPA REVISI", "PAGU DIPA REVISI EFEKTIF",
                "dengan Rentang Pagu DIPA Revisi dikurangi Blokir DIPA Revisi per Kategori",
                color_range="#aed581", color_marker="#33691e"
            )
            st.plotly_chart(fig33, use_container_width=True)
        st.caption("*Rentang merupakan besaran :red[Blokir] DIPA Revisi")

# =============================================================================
# Error Handling & Entry Point
# =============================================================================
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Terjadi kesalahan dalam aplikasi: {str(e)}")

        st.info("Silakan refresh halaman atau hubungi administrator.")

























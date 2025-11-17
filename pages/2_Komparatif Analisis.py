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
    page_title="Analisis Komparasi Realisasi vs Pagu DIPA",
    page_icon=":material/split_scene:",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://www.kemenkeu.go.id",
        "Report a bug": "https://github.com/tubankum3/dashpmk/issues",
        "About": "Dashboard Anggaran Bidang PMK"
    }
)

# =============================================================================
# Modern Dashboard Design CSS
# =============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    /* Modern Color Palette */
    --primary: #0066FF;
    --primary-dark: #0052CC;
    --primary-light: #4D94FF;
    --secondary: #10B981;
    --warning: #F59E0B;
    --error: #EF4444;
    --success: #10B981;
    
    /* Neutral Colors */
    --gray-50: #F9FAFB;
    --gray-100: #F3F4F6;
    --gray-200: #E5E7EB;
    --gray-300: #D1D5DB;
    --gray-400: #9CA3AF;
    --gray-500: #6B7280;
    --gray-600: #4B5563;
    --gray-700: #374151;
    --gray-800: #1F2937;
    --gray-900: #111827;
    
    /* Surface & Background */
    --surface: #FFFFFF;
    --background: #F9FAFB;
    --on-surface: #111827;
    --on-primary: #FFFFFF;
    
    /* Shadows - More subtle and modern */
    --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    --shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
    
    /* Border Radius */
    --radius-sm: 6px;
    --radius-md: 8px;
    --radius-lg: 12px;
    --radius-xl: 16px;
    --radius-full: 9999px;
    
    /* Spacing */
    --space-xs: 0.5rem;
    --space-sm: 0.75rem;
    --space-md: 1rem;
    --space-lg: 1.5rem;
    --space-xl: 2rem;
    --space-2xl: 3rem;
    
    /* Transitions */
    --transition-fast: 150ms cubic-bezier(0.4, 0, 0.2, 1);
    --transition-base: 200ms cubic-bezier(0.4, 0, 0.2, 1);
    --transition-slow: 300ms cubic-bezier(0.4, 0, 0.2, 1);
}

/* Global Styles */
* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

.stApp {
    background-color: var(--background);
}

/* Header - More modern gradient */
.dashboard-header {
    background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
    padding: var(--space-xl);
    border-radius: var(--radius-lg);
    margin-bottom: var(--space-xl);
    color: var(--on-primary);
    box-shadow: var(--shadow-lg);
}

.breadcrumb {
    font-size: 0.875rem;
    font-weight: 500;
    opacity: 0.9;
    margin-bottom: var(--space-sm);
    letter-spacing: 0.025em;
}

/* Card System - Cleaner, more minimal */
.material-card {
    background: var(--surface);
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow-sm);
    padding: var(--space-xl);
    margin-bottom: var(--space-lg);
    transition: all var(--transition-base);
    border: 1px solid var(--gray-200);
}

.material-card:hover {
    box-shadow: var(--shadow-md);
    transform: translateY(-2px);
    border-color: var(--gray-300);
}

/* Typography - Modern scale */
.dashboard-title {
    font-weight: 700;
    font-size: 2rem;
    line-height: 1.2;
    margin: 0;
    letter-spacing: -0.025em;
}

.dashboard-subtitle {
    font-weight: 400;
    font-size: 1rem;
    opacity: 0.9;
    margin: var(--space-sm) 0 0 0;
}

.section-title {
    font-weight: 600;
    font-size: 1.125rem;
    color: var(--gray-900);
    margin-bottom: var(--space-lg);
    padding-bottom: var(--space-md);
    border-bottom: 2px solid var(--gray-200);
}

/* Metric Cards - More modern design */
.metric-card {
    background: var(--surface);
    border-radius: var(--radius-lg);
    padding: var(--space-lg);
    box-shadow: var(--shadow-sm);
    transition: all var(--transition-base);
    border: 1px solid var(--gray-200);
    position: relative;
    overflow: hidden;
}

.metric-card:hover {
    box-shadow: var(--shadow-md);
    transform: translateY(-2px);
    border-color: var(--primary-light);
}

.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: var(--gray-900);
    margin: var(--space-sm) 0;
    line-height: 1;
}

.metric-label {
    font-size: 0.875rem;
    color: var(--gray-600);
    font-weight: 500;
    text-transform: none;
    letter-spacing: 0;
}

.metric-trend {
    display: inline-flex;
    align-items: center;
    padding: 0.25rem 0.75rem;
    border-radius: var(--radius-full);
    font-size: 0.75rem;
    font-weight: 600;
    margin-top: var(--space-sm);
}

.trend-positive {
    background: #ECFDF5;
    color: var(--success);
}

.trend-negative {
    background: #FEF2F2;
    color: var(--error);
}

/* Buttons - Modern, minimal */
.stButton>button {
    background: var(--primary);
    color: var(--on-primary);
    border: none;
    border-radius: var(--radius-md);
    padding: 0.625rem 1.25rem;
    font-weight: 500;
    font-size: 0.875rem;
    transition: all var(--transition-fast);
    box-shadow: var(--shadow-sm);
}

.stButton>button:hover {
    background: var(--primary-dark);
    box-shadow: var(--shadow-md);
    transform: translateY(-1px);
}

.stButton>button:active {
    transform: translateY(0);
}

/* Sidebar - Cleaner */
.stSidebar {
    background: var(--surface);
    border-right: 1px solid var(--gray-200);
}

.sidebar-section {
    background: var(--surface);
    border-radius: var(--radius-md);
    padding: var(--space-md);
    margin-bottom: var(--space-md);
    border: 1px solid var(--gray-200);
}

/* Tabs - Modern style */
.stTabs [data-baseweb="tab-list"] {
    gap: var(--space-sm);
    border-bottom: 1px solid var(--gray-200);
}

.stTabs [data-baseweb="tab"] {
    background: transparent;
    border: none;
    border-bottom: 2px solid transparent;
    padding: var(--space-md) var(--space-lg);
    color: var(--gray-600);
    font-weight: 500;
    transition: all var(--transition-fast);
}

.stTabs [data-baseweb="tab"]:hover {
    color: var(--gray-900);
    background: var(--gray-50);
    border-radius: var(--radius-md) var(--radius-md) 0 0;
}

.stTabs [aria-selected="true"] {
    background: transparent;
    color: var(--primary);
    border-bottom-color: var(--primary);
}

/* Input Fields */
.stSelectbox, .stTextInput, .stNumberInput {
    border-radius: var(--radius-md);
}

.stSelectbox > div > div,
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    border-radius: var(--radius-md) !important;
    border-color: var(--gray-300) !important;
    transition: all var(--transition-fast);
}

.stSelectbox > div > div:focus-within,
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px rgba(0, 102, 255, 0.1) !important;
}

/* Slider */
.stSlider > div > div > div {
    background: var(--primary) !important;
}

/* Chart Container - Minimal */
[data-testid="column"] > div {
    background: var(--surface);
    border-radius: var(--radius-lg);
    padding: var(--space-xl);
    box-shadow: var(--shadow-sm);
    border: 1px solid var(--gray-200);
    transition: all var(--transition-base);
}

[data-testid="column"] > div:hover {
    box-shadow: var(--shadow-md);
}

/* Responsive Design */
@media (max-width: 768px) {
    .dashboard-title {
        font-size: 1.5rem;
    }
    
    .material-card {
        padding: var(--space-md);
    }
    
    [data-testid="column"] > div {
        padding: var(--space-lg);
    }
}

/* Scrollbar Styling */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: var(--gray-100);
    border-radius: var(--radius-full);
}

::-webkit-scrollbar-thumb {
    background: var(--gray-400);
    border-radius: var(--radius-full);
}

::-webkit-scrollbar-thumb:hover {
    background: var(--gray-500);
}

/* Focus States - Accessibility */
*:focus-visible {
    outline: 2px solid var(--primary);
    outline-offset: 2px;
}

/* Loading States */
.loading-skeleton {
    background: linear-gradient(90deg, var(--gray-200) 25%, var(--gray-300) 50%, var(--gray-200) 75%);
    background-size: 200% 100%;
    animation: loading 1.5s ease-in-out infinite;
    border-radius: var(--radius-md);
}

@keyframes loading {
    0% { background-position: 200% 0; }
    100% { background-position: -200% 0; }
}

/* Utility Classes */
.text-primary { color: var(--primary); }
.text-secondary { color: var(--secondary); }
.text-muted { color: var(--gray-600); }
.bg-primary { background-color: var(--primary); }
.bg-surface { background-color: var(--surface); }

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

def rupiah_separator(x):
    try:
        x = float(x)
    except:
        return x
    return f"Rp {x:,.0f}".replace(",", ".")
    
def generate_table(df, year, selected_kls, selected_metric, col_start, col_end):
    df_year = df[df["Tahun"].astype(int) == year].copy()
    if selected_kls:
        df_year = df_year[df_year["KEMENTERIAN/LEMBAGA"].isin(selected_kls)]

    agg = (
        df_year.groupby(selected_metric, as_index=False)[
            ["REALISASI BELANJA KL (SAKTI)", col_start, col_end]
        ].sum()
    )

    # Calculate derived columns
    agg["VARIANS"] = agg[col_end] - agg["REALISASI BELANJA KL (SAKTI)"]

    # Avoid division by zero → result NaN
    agg["PERSEN_REALISASI"] = np.where(
        agg[col_end] == 0,
        np.nan,
        (agg["REALISASI BELANJA KL (SAKTI)"] / agg[col_end]) * 100
    )

    # Display version (friendly formatting)
    display_df = agg.copy()

    for c in ["REALISASI BELANJA KL (SAKTI)", col_start, col_end, "VARIANS"]:
        display_df[c] = display_df[c].apply(rupiah_separator)

    # Format percentage; replace NaN with "-"
    display_df["PERSEN_REALISASI"] = display_df["PERSEN_REALISASI"].apply(
        lambda x: "-" if pd.isna(x) else f"{x:.1f}%"
    )

    # Rename columns for final display
    display_df = display_df.rename(columns={
        "REALISASI BELANJA KL (SAKTI)": "Realisasi Belanja (SAKTI) [A]",
        col_start: f"{col_start} [B]",
        col_end: f"{col_end} [C]",
        "VARIANS": "Varians [C - A]",
        "PERSEN_REALISASI": "% Realisasi [A/C]"
    })

    return agg, display_df

def download_excel(df_raw, filename):
    import io
    from openpyxl import Workbook

    output = io.BytesIO()
    wb = Workbook()
    ws = wb.active
    ws.title = "Data"

    ws.append(list(df_raw.columns))
    for row in df_raw.values:
        ws.append(row.tolist())

    wb.save(output)
    output.seek(0)
    return output

# =============================================================================
# Component Architecture
# =============================================================================
def header(selected_year: str | None = None, selected_metric: str | None = None, selected_kls: list | None = None):
    """Create comprehensive dashboard header with breadcrumb and key info"""
    year_text = selected_year if selected_year else "Overview"
    metric_text = f" {selected_metric}" if selected_metric else ""
    kl_text = ", ".join(selected_kls) if selected_kls else "SELURUH K/L"
    st.markdown(f"""
    <div class="dashboard-header">
        <div class="breadcrumb">DASHBOARD / KOMPARASI {metric_text} / {kl_text} / TAHUN {year_text}</div>
        <h1 class="dashboard-title">Analisis Komparasi Realisasi vs Pagu DIPA</h1>
    </div>
    """, unsafe_allow_html=True)

def sidebar(df):
    with st.sidebar:
        years = sorted(df["Tahun"].astype(int).unique())
        default_year_index = years.index(2025) if 2025 in years else len(years) - 1
        selected_year = st.selectbox("Pilih Tahun", options=years, index=default_year_index)
        
        # Add Top/Bottom selector
        sort_order = st.radio(
            "Tampilkan Data",
            options=["Top", "Bottom"],
            index=0,
            horizontal=True,
            help="Top: Data tertinggi | Bottom: Data terendah"
        )
        
        top_n = st.number_input(
            f"Tampilkan {sort_order}-N Data",
            min_value=1,
            max_value=50,
            value=10,
            step=1,
            help=f"Jumlah Data {'tertinggi' if sort_order == 'Top' else 'terendah'} yang ditampilkan pada grafik berdasarkan Realisasi Belanja."
        )
        
        category_cols = [col for col in df.select_dtypes(include="object").columns if col != "Tahun"]
        
        selected_metric = st.sidebar.selectbox(
            "Kategori/Klasifikasi",
            options=category_cols,
            index=category_cols.index("KEMENTERIAN/LEMBAGA") if "KEMENTERIAN/LEMBAGA" in category_cols else 0,
        )

        kl_list = sorted(df["KEMENTERIAN/LEMBAGA"].dropna().unique())
        selected_kls = st.multiselect(
            "Pilih Kementerian/Lembaga (bisa lebih dari satu)",
            options=kl_list,
            default=[]
        )

    return selected_year, selected_kls, top_n, selected_metric, sort_order
    
# =============================================================================
# Chart
# =============================================================================
def comparison_chart(df, year, top_n, col_start, col_end, title_suffix,
                     color_range="#b2dfdb", color_marker="#1a73e8", sort_order="Top"):
    """Chart builder showing Pagu ranges, Realisasi markers, and Varian lines with end caps."""
    df_year = df[df["Tahun"].astype(int) == year].copy()
    df_year = df_year[df_year["KEMENTERIAN/LEMBAGA"] != "999 BAGIAN ANGGARAN BENDAHARA UMUM NEGARA"]

    # Determine ascending order based on sort_order
    ascending = (sort_order == "Bottom")
    
    agg = (
        df_year.groupby("KEMENTERIAN/LEMBAGA", as_index=False)[
            ["REALISASI BELANJA KL (SAKTI)", col_start, col_end]
        ].sum()
    ).sort_values("REALISASI BELANJA KL (SAKTI)", ascending=ascending).head(top_n).reset_index(drop=True)


    # Calculate variance and realization percentage
    agg["VARIANS"] = agg[col_end] - agg["REALISASI BELANJA KL (SAKTI)"]
    agg["PERSEN_REALISASI"] = (agg["REALISASI BELANJA KL (SAKTI)"] / agg[col_end]) * 100
                   
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
            "Realisasi: %{x:,.0f} "
            "(%{customdata[1]:.1f}%)<br>"
            "Varian (Pagu Efektif-Realisasi): %{customdata[0]:,.0f}<extra></extra>"
        ),
        customdata=np.stack((agg["VARIANS"], agg["PERSEN_REALISASI"]), axis=-1)
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
                                 color_range="#b2dfdb", color_marker="#1a73e8", sort_order="Top"):
    """Chart showing Pagu ranges, Realisasi markers, and Varian lines by selected category."""
    df_year = df[df["Tahun"].astype(int) == year].copy()
    df_year = df_year[df_year["KEMENTERIAN/LEMBAGA"] != "999 BAGIAN ANGGARAN BENDAHARA UMUM NEGARA"]

    # If user selected K/Ls, filter by them. Otherwise use all data
    if selected_kls:
        df_filtered = df_year[df_year["KEMENTERIAN/LEMBAGA"].isin(selected_kls)]
    else:
        df_filtered = df_year.copy()

    # Determine ascending order based on sort_order
    ascending = (sort_order == "Bottom")

    # Group by selected category
    agg = (
        df_filtered.groupby(selected_metric, as_index=False)[
            ["REALISASI BELANJA KL (SAKTI)", col_start, col_end]
        ].sum()
    ).sort_values("REALISASI BELANJA KL (SAKTI)", ascending=ascending).head(top_n).reset_index(drop=True)

    if agg.empty:
        st.warning(f"Tidak ada data untuk kategori '{selected_metric}' di tahun {year}.")
        return None

    # Calculate variance and realization percentage
    agg["VARIANS"] = agg[col_end] - agg["REALISASI BELANJA KL (SAKTI)"]
    agg["PERSEN_REALISASI"] = (agg["REALISASI BELANJA KL (SAKTI)"] / agg[col_end]) * 100

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
            "Realisasi: %{x:,.0f} "
            "(%{customdata[1]:.1f}%)<br>"
            "Varian: %{customdata[0]:,.0f}<extra></extra>"
        ),
        customdata=np.stack((agg["VARIANS"], agg["PERSEN_REALISASI"]), axis=-1)
    ))

    tickvals = np.linspace(0, max(agg[col_end].max(), agg["REALISASI BELANJA KL (SAKTI)"].max()), num=6)
    ticktext = [format_rupiah(val) for val in tickvals]
    y_ticktext = agg[selected_metric].astype(str).tolist()

    fig.update_layout(
        title=(
            f"Perbandingan Realisasi Belanja {title_suffix} berdasarkan {selected_metric}<br>"
            f"Tahun {year}"
            + (f" untuk K/L Terpilih" if selected_kls else " untuk Seluruh K/L")
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

    # Sidebar selections - now includes sort_order
    selected_year, selected_kls, top_n, selected_metric, sort_order = sidebar(df)
    
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
            color_range="#b2dfdb", color_marker="#00897b",
            sort_order=sort_order 
        )
        st.plotly_chart(fig1, use_container_width=True)
               
        if selected_metric != "KEMENTERIAN/LEMBAGA":
            fig11 = comparison_chart_by_category(
                df, selected_year, selected_kls, selected_metric, top_n,
                "PAGU DIPA AWAL EFEKTIF", "PAGU DIPA REVISI EFEKTIF",
                "dengan Rentang Pagu DIPA Efektif",
                color_range="#aed581", color_marker="#33691e",
                sort_order=sort_order
                
        st.caption("*Rentang merupakan _selisih_ antara Pagu Revisi Efektif dan Pagu Awal Efektif")
        st.caption("**Persentase Realisasi Belanja *terhadap* Pagu DIPA Revisi Efektif")
        st.caption("***Varian adalah Pagu Efektif *dikurangi* Realisasi Belanja")

        with st.expander("Tabel Rincian Data"):
            raw_table, display_table = generate_table(
                df, selected_year, selected_kls, selected_metric,
                "PAGU DIPA AWAL EFEKTIF", "PAGU DIPA REVISI EFEKTIF"
            )
    
            st.dataframe(display_table, use_container_width=True, hide_index=True)
    
            excel_data = download_excel(raw_table, "tabel_realisasi_vs_pagu_awal_revisi.xlsx")
            st.download_button(
                label="Download Excel",
                data=excel_data,
                file_name=f"Tabel_Realisasi_vs_Pagu_{selected_year}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"download_tab1_{selected_year}_{selected_metric}"
            )
           
    with tab2:
        fig2 = comparison_chart(
            df, selected_year, top_n,
            "PAGU DIPA AWAL", "PAGU DIPA AWAL EFEKTIF",
            "dengan Rentang Pagu DIPA Awal dikurangi Blokir DIPA Awal",
            color_range="#c5cae9", color_marker="#1a73e8",
            sort_order=sort_order
        )
        st.plotly_chart(fig2, use_container_width=True)

        if selected_metric != "KEMENTERIAN/LEMBAGA":
            fig22 = comparison_chart_by_category(
                df, selected_year, selected_kls, selected_metric, top_n,
                "PAGU DIPA AWAL", "PAGU DIPA AWAL EFEKTIF",
                "dengan Rentang Pagu DIPA Awal Efektif",
                color_range="#aed581", color_marker="#33691e",
            sort_order=sort_order
            )
            st.plotly_chart(fig22, use_container_width=True)
        st.caption("*Rentang merupakan besaran :red[Blokir] DIPA Awal")
        st.caption("**Persentase Realisasi Belanja *terhadap* Pagu DIPA Awal Efektif")
        st.caption("***Varian adalah Pagu Efektif *dikurangi* Realisasi Belanja")
        
        with st.expander("Tabel Rincian Data"):
            raw_table, display_table = generate_table(
                df, selected_year, selected_kls, selected_metric,
                "PAGU DIPA AWAL", "PAGU DIPA AWAL EFEKTIF"
            )
            
            st.dataframe(display_table, use_container_width=True, hide_index=True)
            
            excel_data = download_excel(raw_table, "tabel_realisasi_vs_pagu_awal_efektif.xlsx")
            st.download_button(
                label="Download Excel",
                data=excel_data,
                file_name=f"Tabel_Realisasi_vs_Pagu_{selected_year}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"download_tab2_{selected_year}_{selected_metric}"
            )

    with tab3:
        fig3 = comparison_chart(
            df, selected_year, top_n,
            "PAGU DIPA REVISI", "PAGU DIPA REVISI EFEKTIF",
            "dengan Rentang Pagu DIPA Revisi dikurangi Blokir DIPA Revisi",
            color_range="#ffe082", color_marker="#e53935",
            sort_order=sort_order
        )
        st.plotly_chart(fig3, use_container_width=True)

        if selected_metric != "KEMENTERIAN/LEMBAGA":
            fig33 = comparison_chart_by_category(
                df, selected_year, selected_kls, selected_metric, top_n,
                "PAGU DIPA REVISI", "PAGU DIPA REVISI EFEKTIF",
                "dengan Rentang Pagu DIPA Revisi Efektif",
                color_range="#aed581", color_marker="#33691e",
            sort_order=sort_order
            )
            st.plotly_chart(fig33, use_container_width=True)
        st.caption("*Rentang merupakan besaran :red[Blokir] DIPA Revisi")
        st.caption("**Persentase Realisasi Belanja *terhadap* Pagu DIPA Revisi Efektif")
        st.caption("***Varian adalah Pagu Efektif *dikurangi* Realisasi Belanja")
        
        with st.expander("Tabel Rincian Data"):
            raw_table, display_table = generate_table(
                df, selected_year, selected_kls, selected_metric,
                "PAGU DIPA REVISI", "PAGU DIPA REVISI EFEKTIF"
            )
            
            st.dataframe(display_table, use_container_width=True, hide_index=True)
            
            excel_data = download_excel(raw_table, "tabel_realisasi_vs_pagu_revisi_efektif.xlsx")
            st.download_button(
                label="Download Excel",
                data=excel_data,
                file_name=f"Tabel_Realisasi_vs_Pagu_{selected_year}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"download_tab3_{selected_year}_{selected_metric}"
            )
        
# =============================================================================
# Error Handling & Entry Point
# =============================================================================
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Terjadi kesalahan dalam aplikasi: {str(e)}")

        st.info("Silakan refresh halaman atau hubungi administrator.")















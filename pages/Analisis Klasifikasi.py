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
# Utilities
# =============================================================================
def format_rupiah(value: float) -> str:
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

def aggregate_level(df, group_cols, metric, top_n=None):
    agg = df.groupby(group_cols, as_index=False)[metric].sum()
    # drop rows where grouping columns are missing
    agg = agg.dropna(subset=[c for c in group_cols if c])
    if top_n:
        top = agg.nlargest(top_n, metric)
        agg = agg[agg[group_cols[-1]].isin(top[group_cols[-1]])]
    return agg

def create_bar_chart(df, metric, y_col, color_col=None, title="", stacked=False, height_override=None, colors="Tealgrn"):
    df_plot = df.copy()
    df_plot["__formatted"] = df_plot[metric].apply(format_rupiah)
    fig = px.bar(
        df_plot.sort_values(metric, ascending=True),
        x=metric, y=y_col, color=color_col,
        orientation="h", text="__formatted", custom_data=[y_col, metric],
        title=title, labels={y_col: y_col.title(), metric: "Jumlah"},
        color_continuous_scale=colors if color_col else None
    )
    fig.update_traces(
        hovertemplate=f"%{{y}}<br>Jumlah: %{{customdata[1]:,.0f}}<br>",
        textposition="auto",
    )
    fig.update_layout(
        showlegend=bool(color_col),
        barmode="stack" if stacked else "relative",
        yaxis={"categoryorder": "total ascending"},
        margin=dict(t=60, l=240, r=25, b=25),
        height=height_override if height_override else 500 + max(0, (len(df_plot) - 10) * 15),
    )
    fig.update_xaxes(title_text="Jumlah")
    return fig

def create_treemap(df, metric, title, path):
    df_plot = df.copy()
    if metric in df_plot.columns:
        df_plot["__formatted"] = df_plot[metric].apply(format_rupiah)
    fig = px.treemap(
        df_plot,
        path=path,
        values=metric,
        color=metric,
        color_continuous_scale="Tealgrn",
        title=title
    )
    fig.update_traces(
        hovertemplate="%{label}<br>Jumlah: %{customdata[0]}<br>Persentase dari Induk: %{percentParent:.2%}<extra></extra>",
        custom_data=[df_plot["__formatted"]],
        textinfo="label+percent parent",
        textfont_size=12
    )
    fig.update_layout(margin=dict(t=70, l=25, r=25, b=25))
    return fig

def display_breadcrumbs():
    path = [f"{col}: {st.session_state.drill[col]}" for _, col in HIERARCHY if st.session_state.drill.get(col)]
    if path:
        st.markdown(f"**Navigasi:** {' > '.join(path)}", unsafe_allow_html=True)
    else:
        st.markdown("**Navigasi:** Top Level", unsafe_allow_html=True)
        
# =============================================================================
# Header
# =============================================================================
def header(selected_year: str | None = None):
    year_text = selected_year if selected_year else "Overview"
    st.markdown(f"""
    <div class="dashboard-header" role="banner" aria-label="Header Dashboard Klasifikasi Anggaran">
        <div class="breadcrumb">DASHBOARD / KLASIFIKASI / TAHUN {year_text}</div>
        <h1 class="dashboard-title">Dashboard Klasifikasi Anggaran</h1>
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# Sidebar
# =============================================================================
def sidebar(df):
    with st.sidebar:
        st.markdown("### ⚙️ Filter Data")

        # Pastikan kolom Tahun ada
        if "Tahun" not in df.columns:
            st.error("Kolom 'Tahun' tidak ditemukan di dataset.")
            st.stop()

        # Bersihkan nilai kosong dan ekstrak 4 digit tahun
        df = df[df["Tahun"].notna()]
        df["Tahun"] = df["Tahun"].astype(str).str.extract(r"(\d{4})")[0]
        df = df[df["Tahun"].notna()]

        years = sorted(df["Tahun"].astype(int).unique().tolist())
        if len(years) == 0:
            st.error("Tidak ada data tahun yang valid di dataset.")
            st.stop()

        default_year_index = years.index(2025) if 2025 in years else len(years) - 1
        selected_year = st.selectbox("Pilih Tahun", years, index=default_year_index)

        # Jumlah Top
        top_n = st.number_input(
            "Tampilkan Top K/L berdasarkan nilai metrik",
            min_value=1,
            max_value=50,
            value=10,
            step=1,
        )

        # Pilih metrik
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        if not numeric_cols:
            st.error("Tidak ada kolom numerik yang dapat dipilih sebagai metrik.")
            st.stop()

        selected_metric = st.selectbox(
            "Metrik Anggaran",
            options=numeric_cols,
            index=numeric_cols.index("REALISASI BELANJA KL (SAKTI)")
            if "REALISASI BELANJA KL (SAKTI)" in numeric_cols else 0,
        )
        
        # Filter K/L
        if "KEMENTERIAN/LEMBAGA" not in df.columns:
            st.error("Kolom 'KEMENTERIAN/LEMBAGA' tidak ditemukan di dataset.")
            st.stop()
        kl_list = sorted(df["KEMENTERIAN/LEMBAGA"].dropna().unique().tolist())
        selected_kls = st.multiselect(
            "Pilih Kementerian/Lembaga (bisa lebih dari satu)",
            options=["Semua"] + kl_list,
            default=["Semua"]
        )

    if "Semua" in selected_kls:
        selected_kls = []
    return selected_year, selected_kls, top_n, selected_metric

# =============================================================================
# Drill state helpers
# =============================================================================
HIERARCHY = [
    ("FUNGSI", "FUNGSI"),
    ("SUB FUNGSI", "SUB FUNGSI"),
    ("PROGRAM", "PROGRAM"),
    ("KEGIATAN", "KEGIATAN"),
    ("OUTPUT (KRO)", "OUTPUT (KRO)"),
    ("SUB OUTPUT (RO)", "SUB OUTPUT (RO)"),
]

def init_session_state():
    defaults = {
        "drill": {lvl: None for _, lvl in HIERARCHY},
        "level_index": 0,
        "click_key": 0
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def reset_drill():
    for k in st.session_state.drill.keys():
        st.session_state.drill[k] = None
    st.session_state.level_index = 0
    st.session_state.click_key += 1

def go_back():
    if st.session_state.level_index > 0:
        # clear current level selection
        current_level = HIERARCHY[st.session_state.level_index][1]
        st.session_state.drill[current_level] = None
        st.session_state.level_index -= 1
        st.session_state.click_key += 1

# =============================================================================
# Main
# =============================================================================
def validate_columns(df):
    required_cols = ["Tahun", "KEMENTERIAN/LEMBAGA"] + [col for _, col in HIERARCHY]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        st.error(f"Kolom berikut hilang dari dataset: {', '.join(missing)}")
        return False
    return True
    
def main():
    df = load_data()
    if df.empty or not validate_columns(df):
        st.warning("Data tidak valid atau tidak tersedia.")
        return

    init_session_state()
    selected_year, selected_kls, top_n, selected_metric = sidebar(df)
    header(selected_year)

    df_filtered = df[df["Tahun"] == str(selected_year)].copy()
    if selected_kls:
        df_filtered = df_filtered[df_filtered["KEMENTERIAN/LEMBAGA"].isin(selected_kls)]

    df_active = df_filtered.copy()
    for _, col in HIERARCHY:
        if sel := st.session_state.drill.get(col):
            df_active = df_active[df_active[col] == sel]

    available_levels = [col for _, col in HIERARCHY if col in df_filtered.columns]
    if not available_levels:
        st.error("Kolom hierarki tidak lengkap di dataset.")
        return

    agg_for_treemap = aggregate_level(df_filtered, available_levels, selected_metric)
    if agg_for_treemap.empty:
        st.info("Tidak ada data untuk treemap.")
        return

    display_breadcrumbs()
    st.markdown("## Treemap — klik node untuk drill-down <span title='Klik node untuk melihat detail level berikutnya'>ℹ️</span>", unsafe_allow_html=True)
    treemap_path = [px.Constant("All")] + available_levels
    fig_treemap = create_treemap(agg_for_treemap, selected_metric, f"DISTRIBUSI {selected_metric} — {selected_year}", treemap_path)
    treemap_height = 500 + len(agg_for_treemap) * 5  # Adjust multiplier as needed
    fig_treemap.update_layout(margin=dict(t=70, l=25, r=25, b=25), height=treemap_height,
        title=dict(text=f"DISTRIBUSI {selected_metric} — {selected_year}", x=0.5, xanchor="center"),
        accessibility=dict(description="Treemap distribusi anggaran berdasarkan hierarki")
    )
    events = plotly_events(fig_treemap, click_event=True, key=f"treemap-{st.session_state.click_key}")

    if events:
        ev = events[0]
        clicked_path = ev.get("currentPath", "").split("/") if ev.get("currentPath") else []
        clicked_path = [p for p in clicked_path if p]  # Remove empty
        if clicked_path and clicked_path[0] == "All":
            clicked_path = clicked_path[1:]  # Skip root
        for i, (level_name, level_col) in enumerate(HIERARCHY):
            if i < len(clicked_path) and clicked_path[i]:
                st.session_state.drill[level_col] = clicked_path[i]
            else:
                st.session_state.drill[level_col] = None
        st.session_state.level_index = min(len(clicked_path), len(HIERARCHY) - 1)
        st.session_state.click_key += 1

    deepest_selected_idx = max((-1, *(i for i, (_, col) in enumerate(HIERARCHY) if st.session_state.drill.get(col))))
    bar_level = HIERARCHY[deepest_selected_idx + 1][1] if deepest_selected_idx + 1 < len(HIERARCHY) and HIERARCHY[deepest_selected_idx + 1][1] in df_filtered.columns else HIERARCHY[deepest_selected_idx][1] if deepest_selected_idx >= 0 else available_levels[0]

    st.markdown("## Detail — bar chart untuk level saat ini")
    if bar_level in df_active.columns:
        agg_bar = aggregate_level(df_active, [bar_level], selected_metric, top_n)
        if not agg_bar.empty:
            fig_bar = create_bar_chart(agg_bar, selected_metric, bar_level, title=f"{bar_level} — (Top {top_n}) — berdasarkan pilihan saat ini", colors="Tealgrn")
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("Tidak ada data untuk level yang dipilih.")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Current filters")
    st.sidebar.write(f"**Tahun:** {selected_year}")
    st.sidebar.write(f"**Metrik:** {selected_metric}")
    if selected_kls:
        st.sidebar.write("**K/L:**")
        for k in selected_kls:
            st.sidebar.write(f"- {k}")

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








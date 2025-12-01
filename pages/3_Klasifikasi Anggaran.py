import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events
import zipfile
import io
import requests
from datetime import datetime
from io import BytesIO

# =============================================================================
# Page Configuration
# =============================================================================
st.set_page_config(
    page_title="Analisis Klasifikasi Anggaran",
    page_icon=":material/category:",
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
        response = requests.get(url, timeout=30)
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
    """Format numeric value to Rupiah string with units (T/M/Jt)"""
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
    
def aggregate_level(df, group_cols, metric, top_n=None, sort_order="Top"):
    group_cols = [c for c in group_cols if c in df.columns]
    if not group_cols:
        return pd.DataFrame()
    agg = df.groupby(group_cols, as_index=False)[metric].sum()
    agg = agg.dropna(subset=[group_cols[-1]])
    if top_n:
        if sort_order == "Top":
            sample = agg.nlargest(top_n, metric)
        else:
            sample = agg.nsmallest(top_n, metric)
        agg = agg[agg[group_cols[-1]].isin(sample[group_cols[-1]])]
    return agg

def create_bar_chart(df, metric, y_col, color_col=None, title="", stacked=False, max_height=None, sort_order="Top"):

    """
    Create horizontal bar chart with:
    - X-axis: numeric/continuous float metric values (Rupiah)
    - Bar labels: percentage of total
    - Hover: both percentage and Rupiah value
    - Dynamic height scaling based on number of bars
    """
    df_plot = df.copy()
    
    # Validate columns exist
    if metric not in df_plot.columns or y_col not in df_plot.columns:
        st.error(f"Column '{metric}' or '{y_col}' not found in data")
        return go.Figure()
    
    # Ensure metric is numeric
    df_plot[metric] = pd.to_numeric(df_plot[metric], errors="coerce").fillna(0.0).astype(float)
    
    # Compute total and percentages
    total = float(df_plot[metric].sum())
    if total > 0:
        df_plot["__percentage"] = (df_plot[metric] / total * 100).round(1)
    else:
        df_plot["__percentage"] = 0.0
    
    # Format display strings
    df_plot["__pct_label"] = df_plot["__percentage"].apply(lambda x: f"{x:.2f}%")
    df_plot["__rupiah_formatted"] = df_plot[metric].apply(format_rupiah)
    
    # Sort ascending by metric
    ascending = True if sort_order == "Bottom" else False
    df_plot = df_plot.sort_values(metric, ascending=ascending).reset_index(drop=True)
    
    # Wrap long y-axis labels (wrap at spaces)
    import textwrap
    max_chars = 32
    df_plot["__wrapped_label"] = df_plot[y_col].astype(str).apply(
        lambda lbl: "<br>".join(textwrap.wrap(lbl, width=max_chars, break_long_words=False))
    )

    
    # Get X-axis limits
    x_min = 0.0
    x_max = float(df_plot[metric].max()) if len(df_plot) > 0 and df_plot[metric].max() > 0 else 100.0
    
    # Determine display scale & unit
    if x_max >= 1e12:
        scale, unit = 1e12, "T"   # Triliun
    elif x_max >= 1e9:
        scale, unit = 1e9, "M"   # Miliar
    elif x_max >= 1e6:
        scale, unit = 1e6, "Jt"    # Juta
    else:
        scale, unit = 1, ""
    
    # Compute tick values (nice intervals)
    if x_max > 0:
        target_ticks = 6
        raw_interval = x_max / target_ticks
        magnitude = 10 ** int(np.floor(np.log10(raw_interval)))
        nice_interval = np.ceil(raw_interval / magnitude) * magnitude
        last_tick = np.ceil(x_max / nice_interval) * nice_interval
        tick_vals = list(np.arange(0, last_tick + nice_interval, nice_interval))
    else:
        tick_vals = [0, 50, 100]
        last_tick = 100
    
    # Format tick labels
    if unit:
        tick_texts = [f"Rp {v/scale:.0f} {unit}" for v in tick_vals]
    else:
        tick_texts = [f"Rp {v:,.0f}" for v in tick_vals]
    
    # Create figure
    fig = go.Figure()
    
    for _, row in df_plot.iterrows():
        fig.add_trace(go.Bar(
            x=[row[metric]],
            y=[row[y_col]],
            orientation='h',
            text=row["__pct_label"],
            textposition="outside",
            textfont=dict(size=11, color="#333"),
            marker=dict(color="#1a73e8"),
            hovertemplate=(
                f"{row[y_col]}<br>"
                f"Jumlah: {row['__rupiah_formatted']}<br>"
                f"Persentase: {row['__pct_label']}<extra></extra>"
            ),
            showlegend=False,
        ))
    
    # ✅ Dynamic height adjustment
    n_groups = df_plot[y_col].nunique()
    base_height = 400
    extra_height_per_line = 5
    minus_height_per_line = 20
    
    if n_groups > 10:
        height = base_height + (n_groups - 10) * extra_height_per_line
    elif n_groups < 10:
        height = base_height - (10 - n_groups) * minus_height_per_line
    else:
        height = base_height
    
    # Bound height
    height = max(300, min(height, 1200))
    
    # Allow manual override
    final_height = int(max_height) if max_height is not None else height
    
    # Layout
    fig.update_layout(
        title=title,
        showlegend=False,
        barmode="relative",
        margin=dict(t=70, l=250, r=80, b=50),
        height=final_height,
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis_title="",
        yaxis_title="",
        hovermode="closest",
    )
    fig.update_traces(
        hoverlabel=dict(align="left", bgcolor="white", font_size=10, font_color="#333"),
        hoverinfo="text",
    )
    fig.update_layout(
        hovermode="closest",  # or "y unified" if you prefer static hover line
    )

    # Axis styling
    fig.update_xaxes(
        type="linear",
        tickmode="array",
        tickvals=tick_vals,
        ticktext=tick_texts,
        range=[0, last_tick * 1.1],
        showgrid=True,
        gridcolor="rgba(200,200,200,0.3)",
        zeroline=True,
        zerolinecolor="rgba(150,150,150,0.5)",
    )
    fig.update_yaxes(
        tickvals=df_plot[y_col],
        ticktext=df_plot["__wrapped_label"],
        categoryorder="trace",
        automargin=True,
    )
    
    return fig

# =============================================================================
# Hierarchy and session helpers
# =============================================================================
HIERARCHY = [
    ("FUNGSI", "FUNGSI"),
    ("SUB FUNGSI", "SUB FUNGSI"),
    ("PROGRAM", "PROGRAM"),
    ("KEGIATAN", "KEGIATAN"),
    ("OUTPUT (KRO)", "OUTPUT (KRO)"),
    ("SUB OUTPUT (RO)", "SUB OUTPUT (RO)"),
    ("KOMPONEN", "KOMPONEN")
]

def init_session_state():
    if "drill" not in st.session_state:
        st.session_state.drill = {lvl: None for _, lvl in HIERARCHY}
    if "level_index" not in st.session_state:
        st.session_state.level_index = 0
    if "click_key" not in st.session_state:
        st.session_state.click_key = 0

def reset_drill():
    for k in st.session_state.drill.keys():
        st.session_state.drill[k] = None
    st.session_state.level_index = 0
    st.session_state.click_key += 1

# =============================================================================
# Header / Sidebar
# =============================================================================
def header(selected_year: str | None = None, selected_metric: str | None = None, selected_kls: list | None = None):
    year_text = selected_year if selected_year else "OVERVIEW"
    metric_text = f" {selected_metric}" if selected_metric else "KLASIFIKASI"
    kl_text = ", ".join(selected_kls) if selected_kls else "KEMENTERIAN/LEMBAGA"
    st.markdown(f"""
    <div class="dashboard-header" role="banner" aria-label="Header Dashboard Klasifikasi Anggaran">
        <div class="breadcrumb">DASHBOARD / KLASIFIKASI {metric_text} / {kl_text} / TAHUN {year_text}</div>
        <h1 class="dashboard-title">Analisis Klasifikasi Anggaran</h1>
    </div>
    """, unsafe_allow_html=True)

def sidebar(df):
    with st.sidebar:
        st.markdown("### ⚙️ Filter Data")
        if "Tahun" not in df.columns:
            st.error("Kolom 'Tahun' tidak ditemukan di dataset.")
            st.stop()
        df_year = df[df["Tahun"].notna()].copy()
        df_year["Tahun"] = df_year["Tahun"].astype(str).str.extract(r"(\d{4})")[0]
        years = sorted(df_year["Tahun"].dropna().astype(int).unique().tolist())
        if len(years) == 0:
            st.error("Tidak ada data tahun yang valid di dataset.")
            st.stop()
        default_year_index = years.index(2025) if 2025 in years else len(years) - 1
        selected_year = st.selectbox("Pilih Tahun", years, index=default_year_index)

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
            max_value=500,
            value=11,
            step=1,
            help=f"Jumlah Data {'tertinggi' if sort_order == 'Top' else 'terendah'} yang ditampilkan pada grafik berdasarkan Metrik yang dipilih."
        )

        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        if not numeric_cols:
            st.error("Tidak ada kolom numerik yang dapat dipilih sebagai metrik.")
            st.stop()
        selected_metric = st.selectbox(
            "Metrik Anggaran",
            options=numeric_cols,
            index=numeric_cols.index("REALISASI BELANJA KL (SAKTI)") if "REALISASI BELANJA KL (SAKTI)" in numeric_cols else 0,
        )

        if "KEMENTERIAN/LEMBAGA" not in df.columns:
            st.error("Kolom 'KEMENTERIAN/LEMBAGA' tidak ditemukan di dataset.")
            st.stop()
        kl_list = sorted(df["KEMENTERIAN/LEMBAGA"].dropna().unique().tolist())
        selected_kls = st.multiselect("Pilih Kementerian/Lembaga (opsional)", options=["Semua"] + kl_list, default=["Semua"])
        if "Semua" in selected_kls:
            selected_kls = []

    return selected_year, selected_kls, top_n, selected_metric, sort_order

# =============================================================================
# Drill-down UI
# =============================================================================
def general_drill_down(df_filtered, available_levels, selected_metric, selected_year, top_n, sort_order="Top"):

    """
    Main drill-down interface with breadcrumb navigation and interactive chart
    
    Args:
        df_filtered: Pre-filtered dataframe by year and K/L
        available_levels: List of hierarchy column names available in data
        selected_metric: The numeric metric column to aggregate and display
        selected_year: Selected year for display
        top_n: Number of top items to show
    """
    placeholder = st.empty()
    with placeholder.container():
        # === Breadcrumb navigation ===
        active_drills = [
            (i, lvl, st.session_state.drill.get(lvl))
            for i, lvl in enumerate(available_levels)
            if st.session_state.drill.get(lvl)
        ]
        
        st.markdown(f"##### KLASIFIKASI {selected_metric} TAHUN {selected_year}")
        if active_drills:
            st.markdown("BERDASARKAN:")
            for idx, (i, lvl, val) in enumerate(active_drills):
                row = st.columns([1, 5])
                with row[0]:
                    st.markdown(f"<div class='drill-label'>{lvl}</div>", unsafe_allow_html=True)
                with row[1]:
                    if st.button(f"{val}", key=f"crumb-{lvl}-{val}-{st.session_state.click_key}", use_container_width=True):
                        for j in range(i + 1, len(available_levels)):
                            st.session_state.drill[available_levels[j]] = None
                        st.session_state.level_index = i + 1 if i + 1 < len(available_levels) else i
                        st.session_state.click_key += 1
                        st.rerun()
        
        # === Back / Reset row ===
        left_col, mid_col, right_col = st.columns([1, 10, 1])
        
        # Back button
        with left_col:
            if st.button(":arrow_backward:", help="Kembali satu tingkat"):
                if st.session_state.level_index > 0:
                    prev_idx = max(0, st.session_state.level_index - 1)
                    prev_col = HIERARCHY[prev_idx][1]
                    st.session_state.drill[prev_col] = None
                    st.session_state.level_index = prev_idx
                    st.session_state.click_key += 1
                    st.rerun()
        
        # Reset button
        with right_col:
            if st.button(":arrows_counterclockwise:", help="Kembali ke tampilan awal"):
                reset_drill()
                st.rerun()

        # === Determine current view level ===
        view_idx = min(st.session_state.level_index, len(available_levels) - 1)
        view_row = available_levels[view_idx]

        # === Filter data by ancestor selections ===
        df_view = df_filtered.copy()
        
        # === Ensure selected_metric is numeric in the filtered dataframe ===
        if selected_metric in df_view.columns:
            df_view[selected_metric] = pd.to_numeric(df_view[selected_metric], errors="coerce").fillna(0.0)
        else:
            st.error(f"Metric column '{selected_metric}' not found in data")
            return
        
        for j in range(view_idx):
            anc_row = available_levels[j]
            anc_val = st.session_state.drill.get(anc_row)
            if anc_val is not None:
                df_view = df_view[df_view[anc_row] == anc_val]

        # === Aggregate data for current level ===
        aagg = aggregate_level(df_view, [view_row], selected_metric, top_n, sort_order=sort_order)
        
        if agg.empty:
            st.info("Tidak ada data untuk level ini.")
            return

        # === Create and display chart ===
        title = f"TOP {top_n} {view_row} (Level {view_idx + 1} dari {len(available_levels)})"
        agg_df = aggregate_level(df, ["KEMENTERIAN/LEMBAGA"], selected_metric, top_n=top_n, sort_order=sort_order)
        fig = create_bar_chart(agg_df, selected_metric, view_row, title=title, max_height=600, sort_order=sort_order)

        # ✅ Show chart and capture click events
        events = plotly_events(fig, click_event=True, key=f"drill-{st.session_state.click_key}", override_height=600)

        # === Display Detailed Table with Grand Total & Rupiah Formatting ===
        with st.expander(f"Tabel Rincian Data {view_row}"):
            display_cols = ["KEMENTERIAN/LEMBAGA", "Tahun"] + available_levels + [selected_metric]
            display_cols = [c for c in display_cols if c in df_view.columns]
        
            df_table = df_view[display_cols].copy()
        
            # Ensure numeric
            df_table[selected_metric] = pd.to_numeric(df_table[selected_metric], errors="coerce").fillna(0)
        
            # Sort largest first
            df_table = df_table.sort_values(by=selected_metric, ascending=False).reset_index(drop=True)
        
            # Hidden numeric column for export
            hidden_numeric_col = f"_numeric_{selected_metric}"
            df_table[hidden_numeric_col] = df_table[selected_metric]
        
            # Convert visible metric to Rupiah display strings
            df_table[selected_metric] = df_table[selected_metric].apply(rupiah_separator)
        
            # —— ADD GRAND TOTAL ROW ——
            grand_total = df_table[hidden_numeric_col].sum()
        
            total_row = {col: "" for col in df_table.columns}
            label_col = next((col for col in available_levels if col in df_table.columns), "KEMENTERIAN/LEMBAGA")
            total_row[label_col] = "TOTAL"
            total_row[hidden_numeric_col] = grand_total
            total_row[selected_metric] = rupiah_separator(grand_total)
        
            df_table = pd.concat([df_table, pd.DataFrame([total_row])], ignore_index=True)
        
            # Display table (drop hidden numeric column)
            df_display = df_table.drop(columns=[hidden_numeric_col])
            st.dataframe(df_display, use_container_width=True, hide_index=True)
        
            # —— EXCEL DOWNLOAD ——
            buffer = BytesIO()
            df_table.rename(columns={hidden_numeric_col: f"{selected_metric} (numeric)"}).to_excel(
                buffer,
                index=False,
                sheet_name="Data"
            )
            buffer.seek(0)
        
            st.download_button(
                label="Download Excel",
                data=buffer,
                file_name=f"drill_view_{selected_metric}_{selected_year}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        # === Handle click events for drill-down ===
        if events:
            ev = events[0]
            # Try to get clicked value from y-axis (category name)
            clicked = ev.get("y") or ev.get("label")
            
            # Fallback: try customdata
            if not clicked and ev.get("customdata"):
                cd = ev.get("customdata")
                if isinstance(cd, list) and len(cd) > 0:
                    clicked = cd[0][0] if isinstance(cd[0], (list, tuple)) else cd[0]
            
            if clicked:
                # Store the clicked value in session state
                st.session_state.drill[view_row] = clicked
                
                # Move to next level if available
                if view_idx + 1 < len(available_levels):
                    st.session_state.level_index = view_idx + 1
                
                st.session_state.click_key += 1
                st.rerun()

# =============================================================================
# Main
# =============================================================================
def main():
    init_session_state()
    df = load_data()
    if df.empty:
        st.warning("Data tidak tersedia.")
        return

    if "Tahun" not in df.columns or "KEMENTERIAN/LEMBAGA" not in df.columns:
        st.error("Kolom 'Tahun' atau 'KEMENTERIAN/LEMBAGA' tidak ditemukan.")
        return

    selected_year, selected_kls, top_n, selected_metric, sort_order = sidebar(df)
    header(selected_year, selected_metric, selected_kls)

    # Filter base data by year + K/L
    df_filtered = df[df["Tahun"] == str(selected_year)].copy()
    if selected_kls:
        df_filtered = df_filtered[df_filtered["KEMENTERIAN/LEMBAGA"].isin(selected_kls)]

    # Determine available hierarchy columns in order
    available_levels = [col for _, col in HIERARCHY if col in df_filtered.columns]
    if not available_levels:
        st.error("Kolom hierarki tidak ditemukan di dataset.")
        return

    # Run the drill-down interface
    general_drill_down(df_filtered, available_levels, selected_metric, selected_year, top_n, sort_order=sort_order)

    # Sidebar: current filters and drill state
    st.sidebar.markdown("---")
    if selected_kls:
        st.sidebar.write("**K/L:**")
        for k in selected_kls:
            st.sidebar.write(f"- {k}")

    # Footer
    st.markdown("---")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.caption("📊 Sumber Data: bidja.kemenkeu.go.id")
    with col2:
        st.caption(f"🕐 Diperbarui: {datetime.now().strftime('%d %B %Y %H:%M')}")

# =============================================================================
# Entry
# =============================================================================
if __name__ == "__main__":
    try:
        init_session_state()
        main()
    except Exception as e:
        st.error(f"Terjadi kesalahan dalam aplikasi: {str(e)}")
        st.info("Silakan refresh halaman atau hubungi administrator.")



































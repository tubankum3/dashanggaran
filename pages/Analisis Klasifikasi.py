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
    # Only keep group columns that exist in df to avoid KeyError
    group_cols = [c for c in group_cols if c in df.columns]
    if not group_cols:
        return pd.DataFrame()
    agg = df.groupby(group_cols, as_index=False)[metric].sum()
    # drop rows where grouping columns are missing
    agg = agg.dropna(subset=[group_cols[-1]])
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
    """
    Expects df to include a column '__custom' which is a list of path values for each row
    and the columns referenced in 'path' (path should include px.Constant root then hierarchy cols).
    """
    df_plot = df.copy()
    if metric in df_plot.columns:
        df_plot["__formatted"] = df_plot[metric].apply(format_rupiah)
    # Ensure customdata is present as a column (list per row)
    if "__custom" not in df_plot.columns:
        df_plot["__custom"] = df_plot.apply(lambda r: [r[c] if c in df_plot.columns else None for c in path[1:]], axis=1)

    fig = px.treemap(
        df_plot,
        path=path,
        values=metric,
        color=metric,
        custom_data=["__custom"],
        color_continuous_scale="Tealgrn",
        title=title
    )
    fig.update_traces(
        hovertemplate="%{label}<br>Jumlah: Rp%{value:,.0f}<br>Persentase dari Induk: %{percentParent:.2%}<extra></extra>",
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
            max_value=200,
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
# Main & Validation
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
    if df.empty:
        st.warning("Data tidak tersedia.")
        return

    # Validate presence of core columns, but proceed if some hierarchy levels are missing (we adapt)
    # Only raise if 'Tahun' or 'KEMENTERIAN/LEMBAGA' missing
    if "Tahun" not in df.columns or "KEMENTERIAN/LEMBAGA" not in df.columns:
        st.error("Kolom 'Tahun' atau 'KEMENTERIAN/LEMBAGA' tidak ditemukan. Aplikasi butuh keduanya.")
        return

    init_session_state()
    selected_year, selected_kls, top_n, selected_metric = sidebar(df)
    header(selected_year)

    # filter by year and K/L
    df_filtered = df[df["Tahun"] == str(selected_year)].copy()
    if selected_kls:
        df_filtered = df_filtered[df_filtered["KEMENTERIAN/LEMBAGA"].isin(selected_kls)]

    # apply active drill selections to get df_active (used for bar aggregation)
    df_active = df_filtered.copy()
    for _, col in HIERARCHY:
        if (sel := st.session_state.drill.get(col)):
            df_active = df_active[df_active[col] == sel]

    # determine which hierarchy columns exist in dataset (in order)
    available_levels = [col for _, col in HIERARCHY if col in df_filtered.columns]
    if not available_levels:
        st.error("Kolom hierarki tidak ditemukan di dataset.")
        return

    # aggregate for treemap at the deepest available level for full hierarchy view
    agg_for_treemap = aggregate_level(df_filtered, available_levels, selected_metric)
    if agg_for_treemap.empty:
        st.info("Tidak ada data untuk treemap.")
        return

    # create a customdata list per row containing path values (in available_levels order)
    def _row_custom(r):
        return [r[col] if col in r.index else None for col in available_levels]
    agg_for_treemap["__custom"] = agg_for_treemap.apply(_row_custom, axis=1)

    # breadcrumbs & info
    display_breadcrumbs()
    st.markdown("## Treemap — klik node untuk drill-down")
    treemap_path = [px.Constant("All")] + available_levels
    fig_treemap = create_treemap(agg_for_treemap, selected_metric, f"DISTRIBUSI {selected_metric} — {selected_year}", treemap_path)

    # make treemap height reasonable relative to number of leaves (cap it)
    treemap_height = min(1000, 400 + len(agg_for_treemap) * 3)
    fig_treemap.update_layout(height=treemap_height)

    events = plotly_events(fig_treemap, click_event=True, key=f"treemap-{st.session_state.click_key}")

    # parse click robustly via customdata (preferred) or currentPath fallback
    if events:
        ev = events[0]
        clicked_list = None

        # Try to extract customdata (we packed a list per row under "__custom")
        # plotly_events sometimes returns customdata nested inside lists
        cd = ev.get("customdata")
        if cd:
            # cd may be like [[a,b,c]] or [a,b,c] depending on point
            if isinstance(cd, list) and len(cd) > 0 and isinstance(cd[0], list):
                clicked_list = cd[0]
            else:
                clicked_list = cd
        else:
            # fallback to 'points' structure
            pts = ev.get("points")
            if pts and isinstance(pts, list) and pts[0].get("customdata"):
                cd2 = pts[0]["customdata"]
                if isinstance(cd2, list) and len(cd2) > 0 and isinstance(cd2[0], list):
                    clicked_list = cd2[0]
                else:
                    clicked_list = cd2

        # final fallback: use currentPath string (less robust if labels contain '/')
        if not clicked_list:
            cp = ev.get("currentPath") or ev.get("pointPath") or None
            if cp and isinstance(cp, str):
                parts = [p for p in cp.split("/") if p and p != "All"]
                if parts:
                    # map parts into available_levels order (prefix)
                    clicked_list = [parts[i] if i < len(parts) else None for i in range(len(available_levels))]

        if clicked_list:
            # normalize nan -> None
            clicked_normal = [None if (isinstance(x, float) and np.isnan(x)) else x for x in clicked_list]
            # find deepest index with a non-null value
            deepest_idx = None
            for i in reversed(range(len(clicked_normal))):
                if clicked_normal[i] is not None:
                    deepest_idx = i
                    break
            if deepest_idx is not None:
                # set session_state.drill for each level up to deepest_idx
                for j in range(deepest_idx + 1):
                    lvl = available_levels[j]
                    st.session_state.drill[lvl] = clicked_normal[j]
                # clear deeper levels beyond deepest_idx
                for k in range(deepest_idx + 1, len(available_levels)):
                    st.session_state.drill[available_levels[k]] = None
                # update numerical level_index relative to HIERARCHY full list
                mapped_idx = None
                for idx_full, (_, col_full) in enumerate(HIERARCHY):
                    if col_full == available_levels[deepest_idx]:
                        mapped_idx = idx_full
                        break
                if mapped_idx is not None:
                    st.session_state.level_index = mapped_idx
                st.session_state.click_key += 1

    # After processing clicks, recompute df_active (apply current drill)
    df_active = df_filtered.copy()
    for _, col in HIERARCHY:
        if (sel := st.session_state.drill.get(col)):
            df_active = df_active[df_active[col] == sel]

    # Determine which single bar level to show:
    # - If nothing selected, show top-level summary (FUNGSI if present, else first available)
    # - If selection exists at depth d, show next deeper level (d+1) if available; otherwise show current level d
    deepest_selected_idx = -1
    for idx_full, (_, col_full) in enumerate(HIERARCHY):
        if col_full in available_levels and st.session_state.drill.get(col_full):
            deepest_selected_idx = idx_full

    # compute bar_level using available_levels order
    if deepest_selected_idx == -1:
        # no selection yet -> show first available top-level (prefer FUNGSI)
        bar_level = "FUNGSI" if "FUNGSI" in available_levels else available_levels[0]
    else:
        # find the index of the deepest selected level inside available_levels
        selected_col = HIERARCHY[deepest_selected_idx][1]
        try:
            idx_in_avail = available_levels.index(selected_col)
        except ValueError:
            idx_in_avail = None
        bar_level = None
        if idx_in_avail is not None:
            if idx_in_avail + 1 < len(available_levels):
                bar_level = available_levels[idx_in_avail + 1]  # show next deeper level
            else:
                bar_level = available_levels[idx_in_avail]  # no deeper level -> show current

    # Render single bar chart for the computed bar_level (non-clickable)
    st.markdown("## Detail — bar chart untuk level saat ini")
    if bar_level and bar_level in df_active.columns:
        agg_bar = aggregate_level(df_active, [bar_level], selected_metric, top_n)
        if agg_bar.empty:
            st.info("Tidak ada data untuk level yang dipilih.")
        else:
            fig_bar = create_bar_chart(
                agg_bar,
                selected_metric,
                bar_level,
                title=f"{bar_level} — (Top {top_n}) — berdasarkan pilihan saat ini",
                colors="Tealgrn"
            )
            st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("Level bar chart tidak tersedia untuk data saat ini.")

    # Sidebar: current filters and drill state
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Current filters")
    st.sidebar.write(f"**Tahun:** {selected_year}")
    st.sidebar.write(f"**Metrik:** {selected_metric}")
    if selected_kls:
        st.sidebar.write("**K/L:**")
        for k in selected_kls:
            st.sidebar.write(f"- {k}")
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Drill state")
    for _, col in HIERARCHY:
        st.sidebar.write(f"- {col}: {st.session_state.drill.get(col) if st.session_state.drill.get(col) else '-'}")

    # footer
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

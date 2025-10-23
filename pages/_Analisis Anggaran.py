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
# Material Design Styled CSS (kept from your file)
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

/* Breadcrumb row */
.breadcrumb-row {
    display: flex;
    gap: 8px;
    align-items: center;
    margin-bottom: 8px;
}

/* Simple breadcrumb button style */
.breadcrumb-btn {
    background: transparent;
    border: 1px solid rgba(0,0,0,0.08);
    padding: 6px 10px;
    border-radius: 6px;
    cursor: pointer;
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
    # Only keep group columns that exist in df to avoid KeyError and to speed up
    group_cols = [c for c in group_cols if c in df.columns]
    if not group_cols:
        return pd.DataFrame()
    agg = df.groupby(group_cols, as_index=False)[metric].sum()
    agg = agg.dropna(subset=[group_cols[-1]])
    if top_n:
        top = agg.nlargest(top_n, metric)
        agg = agg[agg[group_cols[-1]].isin(top[group_cols[-1]])]
    return agg

def create_bar_chart(df, metric, y_col, title="", max_height=None):
    df_plot = df.copy()
    df_plot["__fmt"] = df_plot[metric].apply(format_rupiah)
    fig = px.bar(
        df_plot.sort_values(metric, ascending=True),
        x=metric, y=y_col, orientation="h",
        text="__fmt",
        custom_data=[y_col, metric],
        title=title,
        labels={y_col: y_col, metric: "Jumlah"}
    )
    fig.update_traces(textposition="auto", hovertemplate="%{y}<br>Jumlah: %{customdata[1]:,.0f}<extra></extra>")
    fig.update_layout(margin=dict(t=60, l=240, r=25, b=25), height=max_height or (500 + max(0, (len(df_plot) - 10) * 15)))
    fig.update_xaxes(title_text="Jumlah (Rp)")
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
    ("SUB OUTPUT (RO)", "SUB OUTPUT (RO)")
]

def init_session_state():
    if "drill" not in st.session_state:
        st.session_state.drill = {lvl: None for _, lvl in HIERARCHY}
    if "level_index" not in st.session_state:
        st.session_state.level_index = -1  # -1 => no selection yet, showing top level (FUNGSI)
    if "click_key" not in st.session_state:
        st.session_state.click_key = 0

def reset_drill():
    for k in st.session_state.drill.keys():
        st.session_state.drill[k] = None
    st.session_state.level_index = -1
    st.session_state.click_key += 1

def jump_to_level(target_col):
    """
    Keep selections up to target_col, clear deeper ones, and set level_index accordingly.
    If target_col is None (Home), reset everything.
    """
    if target_col is None:
        reset_drill()
        return
    # find index in HIERARCHY
    idx = None
    for i, (_, col) in enumerate(HIERARCHY):
        if col == target_col:
            idx = i
            break
    if idx is None:
        return
    # keep existing parents, clear deeper
    for j, (_, col) in enumerate(HIERARCHY):
        if j > idx:
            st.session_state.drill[col] = None
    st.session_state.level_index = idx
    st.session_state.click_key += 1

# Breadcrumb renderer (simple text + buttons)
def render_breadcrumb_buttons(available_levels):
    # build items as (col, value) for selected ancestors in order
    items = [(col, st.session_state.drill[col]) for col in available_levels if st.session_state.drill.get(col)]
    # render Home + ancestor buttons
    cols = st.columns(max(1, len(items) + 1))
    if cols[0].button("Home"):
        reset_drill()
        st.experimental_rerun()
    for i, (col, val) in enumerate(items, start=1):
        if cols[i].button(f"{col}: {val}", key=f"crumb-{col}-{val}-{st.session_state.click_key}"):
            # jump to this ancestor
            jump_to_level(col)
            st.experimental_rerun()

# =============================================================================
# Header / Sidebar (kept your implementations w/ minor adjustments)
# =============================================================================
def header(selected_year: str | None = None):
    year_text = selected_year if selected_year else "Overview"
    st.markdown(f"""
    <div class="dashboard-header" role="banner" aria-label="Header Dashboard Klasifikasi Anggaran">
        <div class="breadcrumb">DASHBOARD / KLASIFIKASI / TAHUN {year_text}</div>
        <h1 class="dashboard-title">Dashboard Klasifikasi Anggaran</h1>
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

        top_n = st.number_input("Tampilkan Top (N)", min_value=1, max_value=500, value=20, step=1)

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

        # Back / Reset quick controls
        st.markdown("---")
        if st.button("Kembali (Back)"):
            # step back one level
            if st.session_state.level_index >= 0:
                # clear current level selection
                col = HIERARCHY[st.session_state.level_index][1]
                st.session_state.drill[col] = None
                st.session_state.level_index = max(-1, st.session_state.level_index - 1)
                st.session_state.click_key += 1
                st.experimental_rerun()
        if st.button("Reset"):
            reset_drill()
            st.experimental_rerun()

    return selected_year, selected_kls, top_n, selected_metric

# =============================================================================
# Main
# =============================================================================
def main():
    init_session_state()
    df = load_data()
    if df.empty:
        st.warning("Data tidak tersedia.")
        return

    # basic validation columns
    if "Tahun" not in df.columns or "KEMENTERIAN/LEMBAGA" not in df.columns:
        st.error("Kolom 'Tahun' atau 'KEMENTERIAN/LEMBAGA' tidak ditemukan.")
        return

    selected_year, selected_kls, top_n, selected_metric = sidebar(df)
    header(selected_year)

    # filter by year and KL
    df_filtered = df[df["Tahun"] == str(selected_year)].copy()
    if selected_kls:
        df_filtered = df_filtered[df_filtered["KEMENTERIAN/LEMBAGA"].isin(selected_kls)]

    # determine which hierarchy columns are available (preserve order)
    available_levels = [col for _, col in HIERARCHY if col in df_filtered.columns]
    if not available_levels:
        st.error("Kolom hierarki tidak ditemukan di dataset.")
        return

    # Render breadcrumbs above chart
    st.markdown("### Navigasi")
    render_breadcrumb_buttons(available_levels)
    # show a simple path text
    path_text = " > ".join([f"{col}: {st.session_state.drill[col]}" for col in available_levels if st.session_state.drill.get(col)])
    if path_text:
        st.markdown(f"**Path:** {path_text}")
    else:
        st.markdown("**Path:** Top level (FUNGSI)")

    st.markdown("---")

    # Decide which level to display as bars:
    # If no selection yet (level_index == -1), show top (FUNGSI).
    # If selection at depth d, show next level (d+1) if exists, else show current.
    deepest_selected_idx = -1
    for idx, (_, col) in enumerate(HIERARCHY):
        if col in available_levels and st.session_state.drill.get(col):
            deepest_selected_idx = idx

    if deepest_selected_idx == -1:
        # show FUNGSI summary
        bar_level = "FUNGSI" if "FUNGSI" in available_levels else available_levels[0]
    else:
        # try to show next deeper level
        selected_col = HIERARCHY[deepest_selected_idx][1]
        try:
            idx_in_avail = available_levels.index(selected_col)
        except ValueError:
            idx_in_avail = None
        if idx_in_avail is not None and idx_in_avail + 1 < len(available_levels):
            bar_level = available_levels[idx_in_avail + 1]
        else:
            bar_level = selected_col

    # Filter df_active according to current drill path (so bars show only children of current selection)
    df_active = df_filtered.copy()
    for _, col in HIERARCHY:
        if st.session_state.drill.get(col):
            df_active = df_active[df_active[col] == st.session_state.drill[col]]

    # Aggregate for the bar chart (group by bar_level)
    agg_bar = aggregate_level(df_active, [bar_level], selected_metric, top_n)
    if agg_bar.empty:
        st.info("Tidak ada data untuk level yang dipilih.")
        return

    # Create bar chart and render it
    title = f"{bar_level} — (Top {top_n}) — Berdasarkan pilihan saat ini"
    fig = create_bar_chart(agg_bar, selected_metric, bar_level, title=title, max_height=600)
    st.plotly_chart(fig, use_container_width=True)

    # Capture clicks on the bar chart to drill deeper
    events = plotly_events(fig, click_event=True, key=f"bar-{st.session_state.click_key}")
    if events:
        ev = events[0]
        # Try multiple places to get the clicked label
        clicked = None
        if ev.get("y"):
            clicked = ev.get("y")
        elif ev.get("label"):
            clicked = ev.get("label")
        else:
            # customdata first element is the label
            cd = ev.get("customdata")
            if cd and isinstance(cd, list):
                if isinstance(cd[0], (list, tuple)) and len(cd[0]) > 0:
                    clicked = cd[0][0]
                else:
                    clicked = cd[0]
        if clicked:
            # Find which hierarchy column corresponds to the clicked label
            # We already aggregated by bar_level, so set drill[bar_level] = clicked
            st.session_state.drill[bar_level] = clicked
            # update level_index to the index of bar_level in HIERARCHY
            for idx_full, (_, col_full) in enumerate(HIERARCHY):
                if col_full == bar_level:
                    st.session_state.level_index = idx_full
                    break
            st.session_state.click_key += 1
            # Rerun so the chart updates to the next level
            st.experimental_rerun()

    # Sidebar: show current filters and drill state
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
        init_session_state()
        main()
    except Exception as e:
        st.error(f"Terjadi kesalahan dalam aplikasi: {str(e)}")
        st.info("Silakan refresh halaman atau hubungi administrator.")

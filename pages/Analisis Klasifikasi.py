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
# =============================================================================
# Header
# =============================================================================
def header(selected_year: str | None = None):
    year_text = selected_year if selected_year else "Overview"
    st.markdown(f"""
    <div class="dashboard-header">
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
# Hierarchical Drilldown Treemap
# =============================================================================
# ======================================================
# Hierarchical Drilldown Treemap (dengan ekstraksi klik yang tahan banting)
# ======================================================
def treemap_chart(df, selected_year, selected_kls, top_n, selected_metric):
    """
    Menampilkan treemap pada level saat ini berdasarkan st.session_state.level & path.
    Mengembalikan tuple (fig, clicked_label_or_None).
    """
    hierarchy = ["FUNGSI", "SUB FUNGSI", "PROGRAM", "KEGIATAN", "OUTPUT (KRO)", "SUB OUTPUT (RO)"]

    # Initialize session state for hierarchy navigation
    if "level" not in st.session_state:
        st.session_state.level = 0
        st.session_state.path = []

    level = st.session_state.level
    path = st.session_state.path

    df_filtered = df[df["Tahun"] == str(selected_year)].copy()
    if selected_kls:
        df_filtered = df_filtered[df_filtered["KEMENTERIAN/LEMBAGA"].isin(selected_kls)]

    # Apply path filters (keadaan saat ini)
    for i, node in enumerate(path):
        if hierarchy[i] in df_filtered.columns:
            df_filtered = df_filtered[df_filtered[hierarchy[i]] == node]

    # Tentukan kolom saat ini dan kolom berikutnya (untuk path treemap)
    current_level = hierarchy[level]
    next_level = hierarchy[level + 1] if level + 1 < len(hierarchy) else None

    # Grup sesuai level yang tersedia di data
    group_cols = [current_level] if not next_level else [current_level, next_level]
    group_cols = [c for c in group_cols if c in df_filtered.columns]

    if len(group_cols) == 0:
        st.warning("Tidak ada kolom hirarki yang tersedia pada data untuk ditampilkan.")
        return None, None

    agg = (
        df_filtered.groupby(group_cols, as_index=False)[selected_metric]
        .sum()
        .sort_values(selected_metric, ascending=False)
    )

    # Batasi top_n jika ada (berlaku pada jumlah baris parent-child teratas)
    agg = agg.head(top_n)

    # Siapkan judul yang informatif
    title_path = " / ".join(path) if path else "FUNGSI"
    title = f"{title_path} — {selected_metric} — Tahun {selected_year}"

    # Buat treemap
    fig = px.treemap(
        agg,
        path=group_cols,
        values=selected_metric,
        color=selected_metric,
        color_continuous_scale="Blues",
        title=title,
    )

    fig.update_traces(
        texttemplate="%{label}<br>%{value:,.0f}",
        hovertemplate="%{currentPath}<br>Nilai: %{value:,.0f}<br>Share terhadap induk: %{percentParent:.2%}<extra></extra>",
    )
    fig.update_layout(margin=dict(t=50, l=25, r=25, b=25), height=520)

    # Render chart (tapi jangan langsung rerun di sini — kembalikan fig + label)
    return fig, agg

# ======================================================
# Main 
# ======================================================
def main():
    df = load_data()
    if df.empty:
        st.error("Data gagal dimuat.")
        return

    selected_year, selected_kls, top_n, selected_metric = sidebar(df)

    header(selected_year)
    st.markdown(f"Menampilkan: **{selected_metric}** — Tahun **{selected_year}**")
    st.markdown("---")

    # Ambil figure treemap
    fig, agg = treemap_chart(df, selected_year, selected_kls, top_n, selected_metric)
    if fig is None:
        return

    # Tampilkan dan tangkap event klik (plotly_events kadang mengembalikan struktur yang berbeda)
    st.plotly_chart(fig, use_container_width=True)
    clicked = plotly_events(fig, click_event=True, hover_event=False, select_event=False)

    # Robust extraction of clicked label
    clicked_label = None
    if clicked and isinstance(clicked, list) and len(clicked) > 0:
        ev = clicked[0]
        # possible keys: 'label', 'id', or nested 'points'
        if isinstance(ev, dict):
            if "label" in ev and ev["label"]:
                clicked_label = ev["label"]
            elif "id" in ev and ev["id"]:
                clicked_label = ev["id"]
            elif "points" in ev and isinstance(ev["points"], list) and len(ev["points"]) > 0:
                pt = ev["points"][0]
                # check several possible keys inside a point
                for k in ("label", "text", "hovertext", "customdata", "id"):
                    if k in pt and pt[k]:
                        # if customdata is an array, pick first element if stringable
                        val = pt[k]
                        if isinstance(val, (list, tuple)) and len(val) > 0:
                            val = val[0]
                        clicked_label = str(val)
                        break

    # If we found a clicked label, advance level/path appropriately
    if clicked_label:
        # Only advance when there is a next level available
        hierarchy = ["FUNGSI", "SUB FUNGSI", "PROGRAM", "KEGIATAN", "OUTPUT (KRO)", "SUB OUTPUT (RO)"]
        level = st.session_state.get("level", 0)
        if level + 1 < len(hierarchy):
            # append clicked label into path and increment level
            st.session_state.path = st.session_state.get("path", []) + [clicked_label]
            st.session_state.level = level + 1
            st.experimental_rerun()
    else:
        # show breadcrumb + navigation controls (Back button)
        path = st.session_state.get("path", [])
        level = st.session_state.get("level", 0)
        breadcrumb = " / ".join(path) if path else "FUNGSI"
        st.markdown(f"**📍 Posisi Saat Ini:** {breadcrumb}")

        # Back button
        col1, col2 = st.columns([1, 3])
        with col1:
            if level > 0 and st.button("⬅️ Kembali"):
                st.session_state.level = max(0, level - 1)
                # pop last path element safely
                if st.session_state.get("path"):
                    st.session_state.path = st.session_state.path[:-1]
                st.experimental_rerun()

    # After navigation, show a small note on how to proceed
    st.caption("Klik kotak pada treemap untuk masuk ke level berikutnya. Gunakan 'Kembali' untuk naik satu level.")

    # Optionally: show child-level aggregated tables or charts here (if you want)
    # ... (Anda bisa memanggil fungsi show_child_charts yang sudah disesuaikan untuk level saat ini)

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












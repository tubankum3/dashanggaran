import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
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
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://www.kemenkeu.go.id",
        "Report a bug": "https://github.com/tubankum3/dashpmk/issues",
        "About": "Dashboard Anggaran Bidang PMK"
    }
)

# =============================================================================
# Material Design Styled CSS (Kept as is)
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
.stApp { background-color: var(--background); font-family: 'Google Sans', 'Roboto', 'Inter', sans-serif; }
.dashboard-header { background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%); padding: 2rem; border-radius: var(--border-radius); margin-bottom: 1.5rem; color: var(--on-primary); box-shadow: var(--shadow-2); }
.breadcrumb { font-size: 0.875rem; opacity: 0.9; margin-bottom: 0.5rem; }
.dashboard-title { font-family: 'Google Sans', sans-serif; font-weight: 700; font-size: 2.25rem; line-height: 1.2; margin: 0; }
.stButton>button { background: var(--primary); color: var(--on-primary); border: none; border-radius: var(--border-radius); padding: 0.5rem 1rem; font-weight: 500; transition: var(--transition); box-shadow: var(--shadow-1); }
.stButton>button:hover { background: var(--primary-dark); box-shadow: var(--shadow-2); transform: translateY(-1px); }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Data Loading (Kept as is)
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
# Utilities (Kept as is)
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
    group_cols = [c for c in group_cols if c in df.columns]
    if not group_cols:
        return pd.DataFrame()
    agg = df.groupby(group_cols, as_index=False)[metric].sum()
    agg = agg.dropna(subset=[group_cols[-1]])
    if top_n:
        agg = agg.nlargest(top_n, metric)
    return agg

def create_bar_chart(df, metric, y_col, title=""):
    df_plot = df.copy()
    df_plot["__formatted"] = df_plot[metric].apply(format_rupiah)
    fig = px.bar(
        df_plot.sort_values(metric, ascending=True),
        x=metric, y=y_col,
        orientation="h", text="__formatted",
        title=title, labels={y_col: "", metric: "Jumlah"},
    )
    fig.update_traces(
        hovertemplate=f"%{{y}}<br><b>Jumlah:</b> Rp%{{x:,.0f}}<extra></extra>",
        textposition="auto",
        marker_color='#1a73e8'
    )
    fig.update_layout(
        yaxis={"categoryorder": "total ascending"},
        margin=dict(t=60, l=250, r=25, b=25),
        height=400 + max(0, (len(df_plot) - 5) * 20),
    )
    fig.update_xaxes(title_text="Jumlah Anggaran")
    return fig

def create_treemap(df, metric, title, path):
    df_plot = df.copy()
    df_plot["__formatted"] = df_plot[metric].apply(format_rupiah)
    fig = px.treemap(
        df_plot,
        path=path,
        values=metric,
        color=metric,
        custom_data=["__formatted"],
        color_continuous_scale="Tealgrn",
        title=title
    )
    fig.update_traces(
        hovertemplate="%{label}<br><b>Jumlah:</b> %{customdata[0]}<br><b>Persentase:</b> %{percentParent:.2%}<extra></extra>",
        textinfo="label+percent parent",
        textfont_size=12
    )
    fig.update_layout(margin=dict(t=70, l=25, r=25, b=25), height=600)
    return fig

def display_breadcrumbs():
    path_parts = []
    # Create breadcrumbs in the correct hierarchy order
    for _, col in HIERARCHY:
        if value := st.session_state.drill.get(col):
            path_parts.append(f"{value}")
        else:
            break # Stop when a level is not selected
    
    if path_parts:
        st.markdown(f"**Navigasi:** {' > '.join(path_parts)}")
    else:
        st.markdown("**Navigasi:** Level Teratas")

# =============================================================================
# Header (Kept as is)
# =============================================================================
def header(selected_year: str | None = None):
    year_text = selected_year if selected_year else "Overview"
    st.markdown(f"""
    <div class="dashboard-header" role="banner">
        <div class="breadcrumb">DASHBOARD / KLASIFIKASI / TAHUN {year_text}</div>
        <h1 class="dashboard-title">Dashboard Klasifikasi Anggaran</h1>
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# Sidebar (Kept as is)
# =============================================================================
def sidebar(df):
    with st.sidebar:
        st.markdown("### ⚙️ Filter Data")
        if "Tahun" not in df.columns:
            st.error("Kolom 'Tahun' tidak ditemukan di dataset.")
            st.stop()

        df = df[df["Tahun"].notna()]
        df["Tahun"] = df["Tahun"].astype(str).str.extract(r"(\d{4})")[0]
        df = df[df["Tahun"].notna()]

        years = sorted(df["Tahun"].astype(int).unique().tolist())
        if not years:
            st.error("Tidak ada data tahun yang valid di dataset.")
            st.stop()

        default_year_index = years.index(2025) if 2025 in years else len(years) - 1
        selected_year = st.selectbox("Pilih Tahun", years, index=default_year_index)

        top_n = st.number_input(
            "Tampilkan Top N Anggaran", min_value=1, max_value=50, value=10, step=1
        )

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if not numeric_cols:
            st.error("Tidak ada kolom numerik yang dapat dipilih sebagai metrik.")
            st.stop()

        default_metric = "REALISASI BELANJA KL (SAKTI)"
        selected_metric = st.selectbox(
            "Metrik Anggaran", options=numeric_cols,
            index=numeric_cols.index(default_metric) if default_metric in numeric_cols else 0
        )
        
        if "KEMENTERIAN/LEMBAGA" not in df.columns:
            st.error("Kolom 'KEMENTERIAN/LEMBAGA' tidak ditemukan.")
            st.stop()
        kl_list = sorted(df["KEMENTERIAN/LEMBAGA"].dropna().unique().tolist())
        selected_kls = st.multiselect(
            "Pilih Kementerian/Lembaga", options=["Semua"] + kl_list, default=["Semua"]
        )

    if "Semua" in selected_kls or not selected_kls:
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
    if "drill" not in st.session_state:
        st.session_state.drill = {lvl: None for _, lvl in HIERARCHY}
    if "click_key" not in st.session_state:
        st.session_state.click_key = 0

def reset_drill():
    st.session_state.drill = {lvl: None for _, lvl in HIERARCHY}
    st.session_state.click_key += 1
    st.rerun()

# =============================================================================
# Main Application
# =============================================================================
def main():
    df = load_data()
    if df.empty:
        st.warning("Data tidak tersedia.")
        return

    init_session_state()
    selected_year, selected_kls, top_n, selected_metric = sidebar(df)
    header(selected_year)

    # Filter by sidebar selections
    df_filtered = df[df["Tahun"] == str(selected_year)].copy()
    if selected_kls:
        df_filtered = df_filtered[df_filtered["KEMENTERIAN/LEMBAGA"].isin(selected_kls)]

    # Determine which hierarchy columns are actually in the dataset
    available_levels = [col for _, col in HIERARCHY if col in df_filtered.columns]
    if not available_levels:
        st.error("Tidak ada kolom hierarki (FUNGSI, PROGRAM, dll.) yang ditemukan di dataset.")
        return

    # Aggregate data for the complete treemap view
    agg_for_treemap = aggregate_level(df_filtered, available_levels, selected_metric)
    if agg_for_treemap.empty:
        st.info("Tidak ada data untuk filter yang dipilih.")
        return
    
    # --- Display Treemap and Handle Clicks ---
    col1, col2 = st.columns([10, 1])
    with col1:
        display_breadcrumbs()
    with col2:
        if any(st.session_state.drill.values()):
            st.button("Reset View", on_click=reset_drill, use_container_width=True)

    st.markdown("#### Treemap — Klik pada kotak untuk menelusuri lebih dalam")
    treemap_path = [px.Constant("Semua Anggaran")] + available_levels
    fig_treemap = create_treemap(agg_for_treemap, selected_metric, f"Distribusi {selected_metric} — {selected_year}", treemap_path)
    events = plotly_events(fig_treemap, click_event=True, key=f"treemap-{st.session_state.click_key}")

    # --- Process Click Event to Update Drill State ---
    if events:
        # Use the most reliable path info from the event
        point = events[0]
        if 'points' in point and point['points']:
            entry = point['points'][0].get('entry')
            if entry and 'labels' in entry:
                clicked_path = entry['labels']
                
                # Update drill state based on the clicked path
                for i, level_col in enumerate(available_levels):
                    if i < len(clicked_path):
                        st.session_state.drill[level_col] = clicked_path[i]
                    else:
                        st.session_state.drill[level_col] = None # Clear deeper levels
                
                st.session_state.click_key += 1
                st.rerun()

    # --- Determine Bar Chart Content Based on Drill State ---
    df_active = df_filtered.copy()
    deepest_selected_idx = -1
    for i, col in enumerate(available_levels):
        if value := st.session_state.drill.get(col):
            df_active = df_active[df_active[col] == value]
            deepest_selected_idx = i
        else:
            break
            
    if deepest_selected_idx == -1:
        # Nothing selected, show the top level
        bar_level = available_levels[0]
    elif deepest_selected_idx + 1 < len(available_levels):
        # A level is selected, show the next level down
        bar_level = available_levels[deepest_selected_idx + 1]
    else:
        # At the deepest level, show the current level's breakdown
        bar_level = available_levels[deepest_selected_idx]

    # --- Display the Single Bar Chart ---
    st.markdown(f"#### Detail Anggaran: {bar_level}")
    if bar_level and bar_level in df_active.columns:
        agg_bar = aggregate_level(df_active, [bar_level], selected_metric, top_n)
        if not agg_bar.empty:
            fig_bar = create_bar_chart(agg_bar, selected_metric, bar_level, title=f"Top {top_n} Anggaran untuk {bar_level}")
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info(f"Tidak ada data detail untuk level '{bar_level}' pada pilihan ini.")
    
    # --- Footer ---
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
        st.exception(e) # Also log the full traceback for debugging
        st.info("Silakan segarkan (refresh) halaman atau hubungi administrator.")
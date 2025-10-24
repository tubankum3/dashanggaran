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

.drill-btn {
    background-color: #f0f2f6;
    border: none;
    color: #333;
    padding: 6px 12px;
    border-radius: 6px;
    cursor: pointer;
    text-align: left;
    font-size: 0.9rem;
    width: 100%;
    transition: background-color 0.2s;
}

.drill-btn:hover {
    background-color: #e2e6eb;
}

.drill-controls {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
}

.drill-label {
    font-weight: 600;
    color: #444;
    padding-top: 4px;
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

/* Breadcrumb buttons row */
.breadcrumb-row {
    display: flex;
    gap: 8px;
    align-items: center;
    margin-bottom: 8px;
}

/* Small breadcrumb button style */
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

def aggregate_level(df, group_cols, metric, top_n=None):
    """Aggregate data by grouping columns and return top N"""
    group_cols = [c for c in group_cols if c in df.columns]
    if not group_cols:
        return pd.DataFrame()
    agg = df.groupby(group_cols, as_index=False)[metric].sum()
    agg = agg.dropna(subset=[group_cols[-1]])
    if top_n:
        top = agg.nlargest(top_n, metric)
        agg = agg[agg[group_cols[-1]].isin(top[group_cols[-1]])]
    return agg

def create_bar_chart(df, metric, y_col, color_col=None, title="", stacked=False, max_height=None):
    """
    Create horizontal bar chart with:
    - X-axis: numeric/continuous float metric values (Rupiah) - NO AUTOSCALING
    - Bar labels: percentage of total
    - Hover: both percentage and Rupiah value
    """
    df_plot = df.copy()
    
    # Validate columns exist
    if metric not in df_plot.columns or y_col not in df_plot.columns:
        st.error(f"Column '{metric}' or '{y_col}' not found in data")
        return go.Figure()
    
    # ✅ CRITICAL: Ensure metric is numeric float
    df_plot[metric] = pd.to_numeric(df_plot[metric], errors="coerce").fillna(0.0).astype(float)
    
    # Calculate total and percentage
    total = float(df_plot[metric].sum())
    if total > 0:
        df_plot["__percentage"] = (df_plot[metric] / total * 100).round(1)
    else:
        df_plot["__percentage"] = 0.0
    
    # Create formatted strings for display
    df_plot["__pct_label"] = df_plot["__percentage"].apply(lambda x: f"{x:.1f}%")
    df_plot["__rupiah_formatted"] = df_plot[metric].apply(format_rupiah)
    
    # Sort by metric value ascending for better visualization
    df_plot = df_plot.sort_values(metric, ascending=True).reset_index(drop=True)
    
    # ✅ Wrap long y-axis labels
    cat_labels = df_plot[y_col].astype(str).tolist()
    max_chars = 30
    wrapped = []
    for lbl in cat_labels:
        if len(lbl) <= max_chars:
            wrapped.append(lbl)
        else:
            parts = [lbl[i:i + max_chars] for i in range(0, len(lbl), max_chars)]
            wrapped.append("<br>".join(parts))
    
    df_plot["__wrapped_label"] = wrapped
    
    # ✅ Get x-axis range from actual metric values
    x_min = 0.0
    x_max = float(df_plot[metric].max()) if len(df_plot) > 0 and df_plot[metric].max() > 0 else 100.0
    
    # Determine scale and unit for x-axis labels (for display only, NOT for data)
    if x_max >= 1e12:
        scale, unit = 1e12, "T"
    elif x_max >= 1e9:
        scale, unit = 1e9, "M"
    elif x_max >= 1e6:
        scale, unit = 1e6, "Jt"
    else:
        scale, unit = 1, ""
    
    # Calculate nice tick intervals based on ACTUAL metric values
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
    
    # Format tick labels with units
    if unit:
        tick_texts = [f"{v/scale:.0f} {unit}" for v in tick_vals]
    else:
        tick_texts = [f"Rp {v:,.0f}" for v in tick_vals]
    
    # ✅ Create bar chart using RAW metric values (no scaling)
    fig = go.Figure()
    
    for idx, row in df_plot.iterrows():
        fig.add_trace(go.Bar(
            x=[row[metric]],  # ✅ Use actual raw metric value
            y=[row["__wrapped_label"]],  # ✅ Use wrapped labels
            orientation='h',
            text=row["__pct_label"],
            textposition="auto",
            textfont=dict(size=11, color="#333"),
            marker=dict(color='#1a73e8'),
            hovertemplate=(
                f"<b>{row[y_col]}</b><br>"
                f"Jumlah: {row['__rupiah_formatted']}<br>"
                f"Persentase: {row['__pct_label']}<extra></extra>"
            ),
            showlegend=False,
        ))
    
    # Calculate dynamic height
    base_height = 600 + max(0, (len(df_plot) - 10) * 15)
    final_height = int(max_height) if max_height is not None else base_height
    
    # Update layout
    fig.update_layout(
        title=title,
        showlegend=False,
        barmode="relative",
        margin=dict(t=70, l=250, r=80, b=50),
        height=final_height,
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis_title="Jumlah (Rp)",
        yaxis_title="",
    )
    
    # ✅ Update x-axis: use ACTUAL metric values, format labels only
    fig.update_xaxes(
        type="linear",
        tickmode="array",
        tickvals=tick_vals,
        ticktext=tick_texts,
        range=[0, last_tick * 1.1],  # 10% padding for outside labels
        showgrid=True,
        gridcolor="rgba(200,200,200,0.3)",
        zeroline=True,
        zerolinecolor="rgba(150,150,150,0.5)",
    )
    
    # Update y-axis - keep category order
    fig.update_yaxes(
        categoryorder="trace",  # Maintain the order from df_plot
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
    ("SUB OUTPUT (RO)", "SUB OUTPUT (RO)")
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

def jump_to_ancestor(idx):
    """Jump to ancestor level and clear deeper selections"""
    for j in range(idx + 1, len(HIERARCHY)):
        st.session_state.drill[HIERARCHY[j][1]] = None
    st.session_state.level_index = idx + 1 if idx + 1 < len(HIERARCHY) else idx
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

        top_n = st.number_input("Tampilkan Top (N)", min_value=1, max_value=500, value=10, step=1)

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

    return selected_year, selected_kls, top_n, selected_metric

# =============================================================================
# Drill-down UI
# =============================================================================
def general_drill_down(df_filtered, available_levels, selected_metric, selected_year, top_n):
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

        # === Determine current view level ===
        view_idx = min(st.session_state.level_index, len(available_levels) - 1)
        view_row = available_levels[view_idx]

        # === Filter data by ancestor selections ===
        df_view = df_filtered.copy()
        
        # ✅ Ensure selected_metric is numeric in the filtered dataframe
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
        agg = aggregate_level(df_view, [view_row], selected_metric, top_n)
        
        if agg.empty:
            st.info("Tidak ada data untuk level ini.")
            return

        # === Create and display chart ===
        title = f"TOP {top_n} {view_row} (Level {view_idx + 1} dari {len(available_levels)})"
        fig = create_bar_chart(agg, selected_metric, view_row, title=title, max_height=600)

        # ✅ Show chart and capture click events
        events = plotly_events(fig, click_event=True, key=f"drill-{st.session_state.click_key}", override_height=600)

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

    selected_year, selected_kls, top_n, selected_metric = sidebar(df)
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
    general_drill_down(df_filtered, available_levels, selected_metric, selected_year, top_n)

    # Sidebar: current filters and drill state
    st.sidebar.markdown("---")
    if selected_kls:
        st.sidebar.write("**K/L:**")
        for k in selected_kls:
            st.sidebar.write(f"- {k}")
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Drill state")
    for _, col in HIERARCHY:
        st.sidebar.write(f"- {col}: {st.session_state.drill.get(col) if st.session_state.drill.get(col) else '-'}")

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



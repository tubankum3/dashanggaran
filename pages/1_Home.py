import streamlit as st
import os
import pandas as pd
import numpy as np
import plotly.express as px
import zipfile
import io
import requests
from datetime import datetime

# =============================================================================
# Page Configuration & Global Settings
# =============================================================================
st.set_page_config(
    page_title="Dashboard Analisis Anggaran dan Realisasi Belanja Negara",
    page_icon=":analytics:",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.kemenkeu.go.id',
        'Report a bug': 'https://github.com/tubankum3/dashpmk/issues',
        'About': "Dashboard Anggaran Bidang PMK"
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
    padding: 1.5rem;
    margin-bottom: 1rem;
    box-shadow: var(--shadow-1);
}

/* Tab */
/* Tab container: allow wrapping */
.stTabs [data-baseweb="tab-list"] {
    flex-wrap: wrap !important;
    gap: 0.5rem;
}

/* Each tab */
.stTabs [data-baseweb="tab"] {
    background: var(--surface);
    border-radius: var(--border-radius) var(--border-radius) 0 0;
    padding: 0.75rem 1.5rem;
    border: 1px solid #e8eaed;
    transition: var(--transition);
    flex: initial !important; /* Prevent tabs from shrinking */
    white-space: nowrap !important; /* Keep text on one line */
    margin-bottom: 6px; /* Add vertical spacing between rows */
}

/* Active (selected) tab */
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
    """
    Load and preprocess budget data from GitHub with error handling and validation.
    
    Returns:
        pd.DataFrame: Preprocessed budget data
    """
    url = "https://raw.githubusercontent.com/tubankum3/dashpmk/main/df.csv.zip"
    
    try:
        response = requests.get(url)
        response.raise_for_status()  # raise error if download failed
        
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            # adjust if filename inside zip is different
            with z.open("df.csv") as file:
                df = pd.read_csv(file, low_memory=False)
        
        # Data validation and cleaning
        if df.empty:
            st.error("Dataset kosong atau tidak valid")
            return pd.DataFrame()
        
        # Remove index column if exists
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])
        
        # Data type validation
        if "Tahun" in df.columns:
            df["Tahun"] = df["Tahun"].astype(str)
            
        # Data quality checks
        required_columns = ["KEMENTERIAN/LEMBAGA", "Tahun"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.error(f"Kolom yang diperlukan tidak ditemukan: {missing_columns}")
            return pd.DataFrame()
            
        # st.success("✅ Data berhasil dimuat dan divalidasi")
        return df
        
    except FileNotFoundError:
        st.error("❌ File dataset tidak ditemukan. Pastikan 'df.csv' tersedia.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Gagal memuat data: {str(e)}")
        return pd.DataFrame()

# =============================================================================
# Utility Functions
# =============================================================================
def format_rupiah(value: float) -> str:
    """
    Format numeric value to Indonesian Rupiah currency format
    with appropriate scaling (Thousand, Million, Billion, Trillion).
    
    Args:
        value (float): Numeric value to format
        
    Returns:
        str: Formatted currency string
    """
    if pd.isna(value) or value == 0:
        return "Rp 0"
    
    abs_value = abs(value)
    
    if abs_value >= 1_000_000_000_000:
        return f"Rp {value/1_000_000_000_000:.2f} T"
    elif abs_value >= 1_000_000_000:
        return f"Rp {value/1_000_000_000:.2f} M"
    elif abs_value >= 1_000_000:
        return f"Rp {value/1_000_000:.2f} Jt"
    else:
        return f"Rp {value:,.0f}"

def calculate_financial_metrics(df: pd.DataFrame) -> dict:
    """
    Calculate comprehensive financial metrics including totals, 
    growth rates, and performance indicators.
    
    Args:
        df (pd.DataFrame): Filtered budget data
        
    Returns:
        dict: Dictionary containing all calculated metrics
    """
    metrics = {}
    
    if df.empty:
        return metrics
        
    df = df.sort_values("Tahun")
    yearly_sums = df.groupby("Tahun", as_index=False)["Nilai"].sum()
    
    metrics['yearly_totals'] = yearly_sums
    
    if len(yearly_sums) > 1:
        first_value = yearly_sums["Nilai"].iloc[0]
        last_value = yearly_sums["Nilai"].iloc[-1]
        n_years = len(yearly_sums) - 1

        # Calculate growth metrics
        yearly_sums["YoY_Growth"] = yearly_sums["Nilai"].pct_change() * 100
        metrics['aagr'] = yearly_sums["YoY_Growth"].mean(skipna=True)
        metrics['cagr'] = ((last_value / first_value) ** (1 / n_years) - 1) * 100
        metrics['latest_growth'] = yearly_sums["YoY_Growth"].iloc[-1]
        metrics['last_tahun'] = yearly_sums['Tahun'].max()
    else:
        metrics.update({
            'aagr': 0, 'cagr': 0, 'latest_growth': 0, 'last_tahun': 0
        })
    
    return metrics

# =============================================================================
# Component Architecture
# =============================================================================
def header(selected_kl: str | None = None, selected_metric: str | None = None):
    """Create comprehensive dashboard header with breadcrumb and key info"""
    kl_text = selected_kl if selected_kl else "Overview"
    metric_text = f" {selected_metric}" if selected_metric else ""
    st.markdown(f"""
    <div class="dashboard-header">
        <div class="breadcrumb">DASHBOARD / ANALISIS {metric_text} / {kl_text}</div>
        <h1 class="dashboard-title">📊 Dashboard Analisis Anggaran & Realisasi Belanja Negara</h1>
    </div>
    """, unsafe_allow_html=True)
    
def cards(metrics: dict, selected_kl=None, selected_metric=None):
    """
    Create metric cards with visual hierarchy and interactive elements.
    """
    kl_text = selected_kl or "—"
    metric_text = selected_metric or "—"

    if not metrics:
        return
    
    col1, col2, col3 = st.columns([1,2,2])
    
    with col1:
        # Total Metric Card
        latest_total = metrics['yearly_totals']["Nilai"].iloc[-1]
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Tahun {metrics['last_tahun']}</div>
            <div class="metric-value">{format_rupiah(latest_total)}</div>
            <div class="metric-trend {'trend-positive' if metrics['latest_growth'] >= 0 else 'trend-negative'}">
                {'↗' if metrics['latest_growth'] >= 0 else '↘'} {metrics['latest_growth']:+.1f}% YoY
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # CAGR Card
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Tingkat Pertumbuhan Tahunan Majemuk (CAGR)</div>
            <div class="metric-value">{metrics['cagr']:+.1f}%</div>
            <div class="metric-label">Pertumbuhan tahunan rata-rata selama rentang periode waktu tertentu</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # AAGR Card
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Tingkat Pertumbuhan Tahunan Rata-rata (AAGR)</div>
            <div class="metric-value">{metrics['aagr']:+.1f}%</div>
            <div class="metric-label">Rata-rata tingkat pertumbuhan tahunan</div>
        </div>
        """, unsafe_allow_html=True)

def sidebar(df):
    with st.sidebar:
        st.title("Analisis Anggaran & Realisasi Belanja Negara")
        st.markdown("---")
        st.markdown("""
        <div class="sidebar-section">
            <h3 style='margin: 0 0 1rem 0; color: var(--on-surface);'>🔍 Filter Data</h3>
        """, unsafe_allow_html=True)

        # === Ensure Tahun is numeric ===
        df["Tahun"] = pd.to_numeric(df["Tahun"], errors="coerce").astype("int64")

        # === Select K/L ===
        kl_list = sorted(df["KEMENTERIAN/LEMBAGA"].dropna().unique())
        selected_kl = st.selectbox(
            "Pilih Kementerian/Lembaga",
            kl_list,
            key="ministry_select",
            help="Pilih kementerian/lembaga untuk melihat analisis anggaran"
        )

        df_filtered = df[df["KEMENTERIAN/LEMBAGA"] == selected_kl]

        # === Detect numeric columns for metric choices ===
        numeric_cols = df_filtered.select_dtypes(include=["int64", "float64"]).columns.tolist()
        numeric_cols.remove('Tahun')
        metric_options = numeric_cols if numeric_cols else ["(Tidak ada kolom numerik)"]
        selected_metric = st.selectbox(
            "Metrik Anggaran",
            metric_options,
            key="metric_select",
            help="Pilih jenis anggaran yang akan dianalisis"
        )

        # === Advanced filters ===
        with st.expander("⚙️ Filter Lanjutan"):
            # --- Year range ---
            year_options = sorted(df_filtered["Tahun"].dropna().unique())
            if len(year_options) == 0:
                selected_years = (None, None)
            elif len(year_options) == 1:
                # only one year available — show it and set selected_years to that single year
                single_year = int(year_options[0])
                st.markdown(f"**Tahun tersedia:** {single_year}")
                # store in session_state to keep key consistent if needed
                st.session_state["filter__year_range"] = (single_year, single_year)
                selected_years = (single_year, single_year)
            else:
                # normal slider for multiple years — use a unique key per K/L if you prefer
                min_year, max_year = int(min(year_options)), int(max(year_options))
                # prefill from session_state if exists
                default_range = st.session_state.get("filter__year_range", (min_year, max_year))
                # ensure default_range is within min/max bounds
                default_range = (
                    max(min_year, default_range[0]),
                    min(max_year, default_range[1])
                )
                selected_years = st.slider(
                    "Rentang Tahun",
                    min_value=min_year,
                    max_value=max_year,
                    value=default_range,
                    step=1,
                    key="filter__year_range"
                )


            # --- Categorical filters ---
            st.markdown("### Filter Berdasarkan Nilai Kategorikal")
            cat_cols = [
                col for col in df_filtered.select_dtypes(include=["object"]).columns
                if col not in ["KEMENTERIAN/LEMBAGA", "Tahun"]
            ]

            active_filters = {}
            for cat_col in cat_cols:
                options = sorted(df_filtered[cat_col].dropna().unique())
                selected_values = st.multiselect(
                    f"Pilih {cat_col.replace('_', ' ').title()}",
                    options=options,
                    default=options,
                    key=f"filter__{cat_col}"
                )
                active_filters[cat_col] = selected_values

        st.markdown("</div>", unsafe_allow_html=True)

    # === Apply filters to dataframe ===
    df_filtered = df_filtered[
        (df_filtered["Tahun"] >= selected_years[0]) &
        (df_filtered["Tahun"] <= selected_years[1])
    ]

    for cat_col, values in active_filters.items():
        if values:
            df_filtered = df_filtered[df_filtered[cat_col].isin(values)]

    return df_filtered, selected_kl, selected_metric, selected_years

def chart(df: pd.DataFrame, category_col: str, selected_metric: str, selected_kl: str, base_height=600, extra_height_per_line=10):
    df_grouped = (
        df.groupby(["KEMENTERIAN/LEMBAGA", "Tahun", category_col], as_index=False)["Nilai"]
          .sum()
    )

    # Ensure Tahun is sorted and string
    df_grouped["Tahun"] = df_grouped["Tahun"].astype(str)
    df_grouped = df_grouped.sort_values("Tahun")

    # Adjust height dynamically
    n_groups = df_grouped[category_col].nunique()
    height = base_height + (n_groups * extra_height_per_line if n_groups > 10 else 0)

    # Create the line chart
    fig = px.line(
        df_grouped,
        x="Tahun",
        y="Nilai",
        color=category_col,
        markers=True,
        title=f"📈 {selected_metric} BERDASARKAN {category_col} — {selected_kl}",
        labels={
            "Tahun": "Tahun",
            "Nilai": "Jumlah (Rp)",
            category_col: category_col.replace("_", " ").title(),
        },
        template="plotly_white",
        height=height,
    )

    years = sorted(df_grouped["Tahun"].unique())
    min_year, max_year = years[0], years[-1]

    fig.update_layout(
        hovermode="closest",
        title_x=0,
        legend_title_text=category_col.replace("_", " ").title(),
        margin=dict(l=40, r=40, t=80, b=40),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Google Sans, Roboto, Arial"),
    )

    fig.update_traces(
        hovertemplate="<b>%{fullData.name}</b><br>Tahun: %{x}<br>Rp %{y:,.0f}<extra></extra>",
        line=dict(width=2.5),
        marker=dict(size=7)
    )

    return fig, df_grouped

def apply_advanced_filters(df_filtered):
    """
    Apply multiselect filters created in the sidebar (stored in st.session_state).
    Returns the filtered dataframe.
    """
    # Apply categorical filters that were created with keys 'filter__<colname>'
    # Find relevant keys:
    filter_keys = [k for k in st.session_state.keys() if k.startswith("filter__") and k != "filter__year_range"]
    for key in filter_keys:
        col_name = key.replace("filter__", "")
        selected_vals = st.session_state.get(key)
        if selected_vals:  # if user selected some values (or default exists)
            # only apply column filter if column exists in df_filtered (safety)
            if col_name in df_filtered.columns:
                df_filtered = df_filtered[df_filtered[col_name].isin(selected_vals)]

    # Apply year range filter
    year_range = st.session_state.get("filter__year_range")
    if year_range and year_range[0] is not None:
        # ensure Tahun is int-compatible
        try:
            df_filtered = df_filtered[df_filtered["Tahun"].astype(int).between(year_range[0], year_range[1])]
        except Exception:
            # fallback: if Tahun already string but numerically sortable
            df_filtered = df_filtered[df_filtered["Tahun"].astype(str).between(str(year_range[0]), str(year_range[1]))]

    return df_filtered

# =============================================================================
# Main
# =============================================================================
def main():  
    # Load data with loading state
    with st.spinner("Memuat data anggaran..."):
        df = load_data()
    
    if df.empty:
        st.error("Tidak dapat memuat data. Silakan periksa file dataset.")
        return
    
    # --- Sidebar ---
    df_filtered, selected_kl, selected_metric, selected_years = sidebar(df)
    
    # --- Header ---
    header(selected_kl=selected_kl, selected_metric=selected_metric)

    # --- Ensure Tahun numeric (convert once) ---
    df["Tahun"] = pd.to_numeric(df["Tahun"], errors="coerce")
    df_filtered["Tahun"] = pd.to_numeric(df_filtered["Tahun"], errors="coerce")

    # --- Apply year range filter ---
    if selected_years and selected_years != (None, None):
        start_year, end_year = selected_years
        df_filtered = df_filtered[
            (df_filtered["Tahun"] >= start_year) &
            (df_filtered["Tahun"] <= end_year)
        ]

    # --- Apply categorical filters (from session_state) ---
    filters = {}
    for key, value in st.session_state.items():
        if key.startswith("filter__") and key != "filter__year_range":
            col_name = key.replace("filter__", "")
            filters[col_name] = value

    for col, allowed_vals in filters.items():
        if allowed_vals:
            df_filtered = df_filtered[df_filtered[col].isin(allowed_vals)]

    # --- Apply advanced filters (Filter Lanjutan) ---
    df_filtered = apply_advanced_filters(df_filtered)

    # --- Validate selected metric ---
    if selected_metric not in df_filtered.columns:
        st.warning("Kolom metrik tidak ditemukan di dataset untuk K/L ini.")
        return
    df_filtered = df_filtered.rename(columns={selected_metric: "Nilai"})

    # --- Calculate summary metrics ---
    metrics = calculate_financial_metrics(df_filtered)
    
    # --- Display summary cards ---
    st.markdown(f"<div class='section-title'>RINGKASAN KINERJA {selected_metric}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='material-card'>{selected_kl}</div>", unsafe_allow_html=True)
    cards(metrics, selected_kl=selected_kl, selected_metric=selected_metric)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # --- Visualization Section ---
    st.markdown("<div class='section-title'>📊 Visualisasi Data</div>", unsafe_allow_html=True)
    
    # Categorical columns for visualization
    cat_cols = [
        col for col in df_filtered.select_dtypes(include=["object"]).columns
        if col not in ["KEMENTERIAN/LEMBAGA", "Tahun"]
    ]
    
    if cat_cols:
        tabs = st.tabs([f"📈 {col.replace('_', ' ').title()}" for col in cat_cols])
        
        for tab, col in zip(tabs, cat_cols):
            with tab:
                fig, grouped_df = chart(df_filtered, col, selected_metric, selected_kl)
                st.plotly_chart(fig, use_container_width=True)
                
                # --- Data table (wide format: metric x year) ---
                with st.expander("📋 Data Tabel", expanded=True):
                    display_col = col
                    df_display = grouped_df[["Tahun", display_col, "Nilai"]].copy()

                    # Pivot to wide format: rows=metric, cols=years
                    df_pivot = (
                        df_display
                        .pivot_table(
                            index=display_col,
                            columns="Tahun",
                            values="Nilai",
                            aggfunc="sum"
                        )
                        .fillna(0)
                        .reset_index()
                    )

                    # Sort columns: Tahun ascending
                    tahun_cols = sorted([c for c in df_pivot.columns if c != display_col])
                    df_pivot = df_pivot[[display_col] + tahun_cols]

                    # Format numeric values as currency
                    for c in tahun_cols:
                        df_pivot[c] = df_pivot[c].apply(lambda x: f"Rp {x:,.0f}")

                    # Rename first column to show selected metric
                    df_pivot = df_pivot.rename(columns={display_col: selected_metric})

                    st.dataframe(
                        df_pivot,
                        use_container_width=True,
                        hide_index=True
                    )
    else:
        st.info("ℹ️ Tidak ada kolom kategorikal yang tersedia untuk visualisasi.")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # --- Footer ---
    st.markdown("---")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.caption("📊 Sumber Data: bidja.kemenkeu.go.id")
    with col2:
        st.caption(f"🕐 Diperbarui: {datetime.now().strftime('%d %B %Y %H:%M')}")

# =============================================================================
# Error Handling & Entry Point
# =============================================================================
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Terjadi kesalahan dalam aplikasi: {str(e)}")
        st.info("Silakan refresh halaman atau hubungi administrator.")



















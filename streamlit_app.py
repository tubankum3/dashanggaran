import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import numpy as np

# =============================================================================
# Page Configuration & Global Settings
# =============================================================================
st.set_page_config(
    page_title="Dashboard Analisis Anggaran dan Belanja Negara",
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
    """
    Load and preprocess budget data with error handling and data validation.
    
    Returns:
        pd.DataFrame: Preprocessed budget data
    """
    try:
        df = pd.read_csv("df23-25.csv")
        
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
        st.error("❌ File dataset tidak ditemukan. Pastikan 'df23-25.csv' tersedia.")
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
def header():
    """Create comprehensive dashboard header with breadcrumb and key info"""
    st.markdown(f"""
    <div class="dashboard-header">
        <div class="breadcrumb">Dashboard / Analisis Anggaran / {selected_kl if 'selected_kl' in locals() else 'Overview'}</div>
        <h1 class="dashboard-title">📊 Dashboard Analisis Anggaran & Realisasi Belanja Negara</h1>
        <p class="dashboard-subtitle">Visualisasi dan analisis anggaran Kementerian/Lembaga</p>
    </div>
    """, unsafe_allow_html=True)

def cards(metrics: dict):
    """
    Create enhanced metric cards with improved visual hierarchy
    and interactive elements.
    """
    if not metrics:
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Total Budget Card
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
        st.markdown("""
        <div class="sidebar-section">
            <h3 style='margin: 0 0 1rem 0; color: var(--on-surface);'>🔍 Filter Data</h3>
        """, unsafe_allow_html=True)

        # --- Select K/L ---
        kl_list = sorted(df["KEMENTERIAN/LEMBAGA"].dropna().unique())
        selected_kl = st.selectbox(
            "Pilih Kementerian/Lembaga",
            kl_list,
            key="ministry_select",
            help="Pilih kementerian/lembaga untuk melihat analisis anggaran"
        )

        # Filter dataframe by selected K/L
        df_filtered = df[df["KEMENTERIAN/LEMBAGA"] == selected_kl]

        # --- Detect numeric columns dynamically ---
        numeric_cols = df_filtered.select_dtypes(include=["int64", "float64"]).columns.tolist()
        metric_options = numeric_cols if numeric_cols else ["(Tidak ada kolom numerik)"]
        selected_metric = st.selectbox(
            "Metrik Anggaran",
            metric_options,
            key="metric_select",
            help="Pilih jenis anggaran yang akan dianalisis"
        )

        # Filter again based on selected metric (if needed)
        df_filtered_metric = df_filtered[["KEMENTERIAN/LEMBAGA", "Tahun", selected_metric] + 
                                         [col for col in df_filtered.columns if df_filtered[col].dtype == "object"]]

        # --- Filter tambahan ---
        with st.expander("⚙️ Filter Lanjutan"):
            st.markdown("### Filter Berdasarkan Nilai Kategorikal")

            # Dynamically detect categorical columns (after selecting K/L and metric)
            cat_cols = [
                col for col in df_filtered_metric.select_dtypes(include=["object"]).columns
                if col not in ["KEMENTERIAN/LEMBAGA", "Tahun"]
            ]

            filters = {}
            for cat_col in cat_cols:
                unique_vals = sorted(df_filtered_metric[cat_col].dropna().unique())
                filters[cat_col] = st.multiselect(
                    f"Pilih {cat_col.replace('_', ' ').title()}",
                    options=unique_vals,
                    default=unique_vals
                )

            # --- Year filter ---
            year_options = sorted(df_filtered_metric["Tahun"].dropna().unique())
            if len(year_options) > 1:
                selected_years = st.slider(
                    "Rentang Tahun",
                    min_value=int(min(year_options)),
                    max_value=int(max(year_options)),
                    value=(int(min(year_options)), int(max(year_options))),
                    step=1
                )
            else:
                selected_years = (int(year_options[0]), int(year_options[0]))

        st.markdown("</div>", unsafe_allow_html=True)

    return selected_kl, selected_metric, selected_years

def chart(df: pd.DataFrame, category_col: str, base_height=600, extra_height_per_line=10):
    df_grouped = (
        df.groupby(["KEMENTERIAN/LEMBAGA", "Tahun", category_col], as_index=False)["Nilai"]
          .sum()
    )

    # Sort and ensure Tahun is string
    df_grouped["Tahun"] = df_grouped["Tahun"].astype(str)
    df_grouped = df_grouped.sort_values("Tahun")

    # Adjust chart height for large categories
    n_groups = df_grouped[category_col].nunique()
    height = base_height + (n_groups * extra_height_per_line if n_groups > 10 else 0)

    # Create interactive line chart with year slider
    fig = px.line(
        df_grouped,
        x="Tahun",
        y="Nilai",
        color=category_col,
        markers=True,
        title=f"📈 {selected_metric} per {category_col} — {selected_kl}",
        labels={
            "Tahun": "Tahun",
            "Nilai": "Jumlah (Rp)",
            category_col: category_col.replace("_", " ").title(),
        },
        template="plotly_white",
        height=height,
        animation_frame="Tahun",  # 🎯 adds the year slider
    )

    fig.update_layout(
        hovermode="closest",
        title_x=0,
        legend_title_text=category_col.replace("_", " ").title(),
        margin=dict(l=40, r=40, t=80, b=40),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Google Sans, Roboto, Arial"),
        updatemenus=[{
            "buttons": [
                {"args": [None, {"frame": {"duration": 1000, "redraw": True},
                                 "fromcurrent": True, "mode": "immediate"}],
                 "label": "▶️ Play",
                 "method": "animate"},
                {"args": [[None], {"frame": {"duration": 0, "redraw": True},
                                   "mode": "immediate"}],
                 "label": "⏸️ Pause",
                 "method": "animate"}
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 70},
            "showactive": True,
            "type": "buttons",
            "x": 0.05,
            "xanchor": "right",
            "y": -0.05,
            "yanchor": "top"
        }]
    )

    fig.update_traces(
        hovertemplate="<b>%{fullData.name}</b><br>Tahun: %{x}<br>Rp %{y:,.0f}<extra></extra>",
        line=dict(width=2.5),
        marker=dict(size=7)
    )

    return fig, df_grouped

# =============================================================================
# Main Application
# =============================================================================
def main():  
    # Load data with loading state
    with st.spinner("Memuat data anggaran..."):
        global df
        df = load_data()
    
    if df.empty:
        st.error("Tidak dapat memuat data. Silakan periksa file dataset.")
        return
    
    # Create enhanced UI components
    header()
    
    # Sidebar with filters
    global selected_kl, selected_metric, selected_years
    selected_kl, selected_metric, selected_years = sidebar()
    
    # Filter data based on selections
    df_filtered = df[df["KEMENTERIAN/LEMBAGA"] == selected_kl].copy()
    
    # Apply advanced filters
    for col, selected_values in filters.items():
        if selected_values:
            df_filtered = df_filtered[df_filtered[col].isin(selected_values)]
    
    # Filter by selected year range
    df_filtered = df_filtered[
        df_filtered["Tahun"].astype(int).between(selected_years[0], selected_years[1])
    ]

    df_filtered = df_filtered.rename(columns={selected_metric: "Nilai"})
    
    # Calculate metrics
    metrics = calculate_financial_metrics(df_filtered)
    
    # Display metrics in cards
    # st.markdown("<div class='material-card'>", unsafe_allow_html=True)
    st.markdown(f"<div class='section-title'>📈 {selected_kl}: Ringkasan Kinerja {selected_metric}</div>", unsafe_allow_html=True)
    cards(metrics)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Visualization section
    # st.markdown("<div class='material-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>📊 Visualisasi Data</div>", unsafe_allow_html=True)
    
    # Get categorical columns for visualization
    cat_cols = [
        col for col in df_filtered.select_dtypes(include=["object"]).columns
        if col not in ["KEMENTERIAN/LEMBAGA", "Tahun"]
    ]
    
    if cat_cols:
        # Create tabs
        tabs = st.tabs([f"📈 {col.replace('_', ' ').title()}" for col in cat_cols])
        
        for tab, col in zip(tabs, cat_cols):
            with tab:
                # Chart container
                st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
                fig, grouped_df = chart(df_filtered, col)
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Data table with enhanced presentation
                with st.expander("📋 Data Tabel", expanded=True):
                    display_col = col
                    df_display = grouped_df[["Tahun", display_col, "Nilai"]].copy()
                    
                    # Year-wise tables
                    years_sorted = sorted(df_display["Tahun"].unique(), reverse=True)
                    for year in years_sorted:
                        with st.container():
                            st.markdown(f"**Tahun {year}**")
                            year_df = df_display[df_display["Tahun"] == year][[display_col, "Nilai"]]
                            year_df = year_df.sort_values("Nilai", ascending=False)
                            
                            # Format and display
                            display_df = year_df.copy()
                            display_df["Nilai"] = display_df["Nilai"].apply(
                                lambda x: f"Rp {x:,.0f}"
                            )
                            
                            st.dataframe(
                                display_df,
                                use_container_width=True,
                                hide_index=True
                            )
                            st.markdown("---")
    else:
        st.info("ℹ️ Tidak ada kolom kategorikal yang tersedia untuk visualisasi.")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Footer with enhanced information
    st.markdown("---")
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.caption("📊 Sumber Data: bidja.kemenkeu.go.id")
    with col2:
        st.caption(f"🕐 Diperbarui: {datetime.now().strftime('%d %B %Y %H:%M')}")
    with col3:
        st.caption("👨‍💻 Built with Streamlit")

# =============================================================================
# Error Handling & Entry Point
# =============================================================================
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Terjadi kesalahan dalam aplikasi: {str(e)}")
        st.info("Silakan refresh halaman atau hubungi administrator.")

















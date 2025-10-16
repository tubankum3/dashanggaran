import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import numpy as np

# =============================================================================
# IMPROVEMENT 1: Enhanced Page Configuration & Global Settings
# =============================================================================
st.set_page_config(
    page_title="National Budget Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.kemenkeu.go.id',
        'Report a bug': 'https://github.com/your-repo/issues',
        'About': "Official National Budget Analysis Dashboard"
    }
)

# =============================================================================
# IMPROVEMENT 2: Advanced Material Design Styled CSS
# =============================================================================
st.markdown("""
<style>
/* === GOOGLE MATERIAL DESIGN THEME === */
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

/* === IMPROVEMENT 3: Enhanced Information Architecture === */
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

/* Enhanced Card System */
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

/* === IMPROVEMENT 4: Advanced Visual Hierarchy === */
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

/* Metric Cards with Enhanced Visual Design */
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

/* === IMPROVEMENT 5: Enhanced Interactivity === */
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

/* Enhanced Sidebar */
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

/* Tab Enhancements */
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

/* === IMPROVEMENT 6: Advanced Accessibility === */
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

/* === IMPROVEMENT 7: Performance Optimized Layout === */
/* Responsive design improvements */
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

/* Chart container enhancements */
.chart-container {
    background: var(--surface);
    border-radius: var(--border-radius);
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    box-shadow: var(--shadow-1);
    border: 1px solid #e8eaed;
}

/* Data table enhancements */
.data-table {
    background: var(--surface);
    border-radius: var(--border-radius);
    overflow: hidden;
    box-shadow: var(--shadow-1);
}

</style>
""", unsafe_allow_html=True)

# =============================================================================
# IMPROVEMENT 8: Enhanced Data Loading with Error Handling & Performance
# =============================================================================
@st.cache_data(show_spinner="Memuat dataset anggaran...")
def load_data():
    """
    Load and preprocess budget data with comprehensive error handling
    and data validation.
    
    Returns:
        pd.DataFrame: Preprocessed budget data
    """
    try:
        # Simulate data loading - replace with actual CSV path
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
            
        st.success("✅ Data berhasil dimuat dan divalidasi")
        return df
        
    except FileNotFoundError:
        st.error("❌ File dataset tidak ditemukan. Pastikan 'df23-25.csv' tersedia.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Gagal memuat data: {str(e)}")
        return pd.DataFrame()

# =============================================================================
# IMPROVEMENT 9: Enhanced Utility Functions with Documentation
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
        metrics['total_growth'] = ((last_value - first_value) / first_value) * 100
    else:
        metrics.update({
            'aagr': 0, 'cagr': 0, 'latest_growth': 0, 'total_growth': 0
        })
    
    return metrics

# =============================================================================
# IMPROVEMENT 10: Enhanced Component Architecture
# =============================================================================
def create_enhanced_header():
    """Create comprehensive dashboard header with breadcrumb and key info"""
    st.markdown(f"""
    <div class="dashboard-header">
        <div class="breadcrumb">Dashboard / Analisis Anggaran / {selected_kl if 'selected_kl' in locals() else 'Overview'}</div>
        <h1 class="dashboard-title">📊 Dashboard Analisis Anggaran Belanja Negara</h1>
        <p class="dashboard-subtitle">Visualisasi dan analisis komprehensif anggaran kementerian/lembaga</p>
    </div>
    """, unsafe_allow_html=True)

def create_metric_cards(metrics: dict):
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
            <div class="metric-label">Total Anggaran Terkini</div>
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
            <div class="metric-label">Pertumbuhan Tahunan (CAGR)</div>
            <div class="metric-value">{metrics['cagr']:+.1f}%</div>
            <div class="metric-label">Rata-rata pertumbuhan tahunan</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # AAGR Card
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Pertumbuhan Rata-rata (AAGR)</div>
            <div class="metric-value">{metrics['aagr']:+.1f}%</div>
            <div class="metric-label">Pertumbuhan tahunan rata-rata</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        # Total Growth Card
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Pertumbuhan Periode</div>
            <div class="metric-value">{metrics['total_growth']:+.1f}%</div>
            <div class="metric-label">Pertumbuhan sejak tahun awal</div>
        </div>
        """, unsafe_allow_html=True)

def create_enhanced_sidebar():
    """Create organized sidebar with clear information architecture"""
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-section">
            <h3 style='margin: 0 0 1rem 0; color: var(--on-surface);'>🔍 Filter Data</h3>
        """, unsafe_allow_html=True)
        
        # Ministry selection with search
        kl_list = sorted(df["KEMENTERIAN/LEMBAGA"].dropna().unique())
        selected_kl = st.selectbox(
            "Pilih Kementerian/Lembaga",
            kl_list,
            key="ministry_select",
            help="Pilih kementerian/lembaga untuk melihat analisis anggaran"
        )
        
        # Metric selection with descriptions
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        metric_options = numeric_cols if numeric_cols else ["(Tidak ada kolom numerik)"]
        selected_metric = st.selectbox(
            "Metrik Anggaran",
            metric_options,
            key="metric_select",
            help="Pilih jenis anggaran yang akan dianalisis"
        )
        
        # Additional filters
        with st.expander("⚙️ Filter Lanjutan"):
            year_options = sorted(df["Tahun"].unique())
            selected_years = st.multiselect(
                "Filter Tahun",
                options=year_options,
                default=year_options,
                help="Pilih tahun yang akan ditampilkan"
            )
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Information panel
        st.markdown("""
        <div class="sidebar-section">
            <h4 style='margin: 0 0 0.5rem 0;'>ℹ️ Informasi</h4>
            <p style='font-size: 0.8rem; margin: 0; color: #5f6368;'>
                Data sumber: bidja.kemenkeu.go.id<br>
                Terakhir diperbarui: {update_date}
            </p>
        </div>
        """.format(update_date=datetime.now().strftime("%d %b %Y")), unsafe_allow_html=True)
    
    return selected_kl, selected_metric

def create_interactive_chart(df: pd.DataFrame, category_col: str):
    """
    Create enhanced interactive chart with improved visual design
    and interactivity features.
    """
    df_grouped = (
        df.groupby(["KEMENTERIAN/LEMBAGA", "Tahun", category_col], as_index=False)["Nilai"]
          .sum()
    )
    
    # Create interactive figure
    fig = px.line(
        df_grouped,
        x="Tahun",
        y="Nilai",
        color=category_col,
        markers=True,
        title=f"📈 {selected_metric} per {category_col} — {selected_kl}",
        labels={
            "Tahun": "Tahun",
            "Nilai": "Jumlah Anggaran",
            category_col: category_col.replace("_", " ").title(),
        },
        template="plotly_white",
        height=500
    )
    
    # Enhanced styling
    fig.update_layout(
        hovermode="x unified",
        title_x=0,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=40, r=40, t=80, b=40),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Google Sans, Roboto, Arial")
    )
    
    # Enhanced hover information
    fig.update_traces(
        hovertemplate="<b>%{fullData.name}</b><br>" +
                     "Tahun: %{x}<br>" +
                     "Anggaran: %{y:,.0f} Rupiah<extra></extra>",
        line=dict(width=3),
        marker=dict(size=8)
    )
    
    return fig, df_grouped

# =============================================================================
# IMPROVEMENT 11: Main Application with Enhanced Architecture
# =============================================================================
def main():
    """Main application function with improved error handling and user flow"""
    
    # Load data with loading state
    with st.spinner("Memuat data anggaran..."):
        global df
        df = load_data()
    
    if df.empty:
        st.error("Tidak dapat memuat data. Silakan periksa file dataset.")
        return
    
    # Create enhanced UI components
    create_enhanced_header()
    
    # Sidebar with filters
    global selected_kl, selected_metric
    selected_kl, selected_metric = create_enhanced_sidebar()
    
    # Filter data based on selections
    df_filtered = df[df["KEMENTERIAN/LEMBAGA"] == selected_kl].copy()
    df_filtered = df_filtered.rename(columns={selected_metric: "Nilai"})
    
    # Calculate metrics
    metrics = calculate_financial_metrics(df_filtered)
    
    # Display metrics in enhanced cards
    st.markdown("<div class='material-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>📈 Ringkasan Kinerja Anggaran</div>", unsafe_allow_html=True)
    create_metric_cards(metrics)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Visualization section
    st.markdown("<div class='material-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>📊 Visualisasi Data</div>", unsafe_allow_html=True)
    
    # Get categorical columns for visualization
    cat_cols = [
        col for col in df_filtered.select_dtypes(include=["object"]).columns
        if col not in ["KEMENTERIAN/LEMBAGA", "Tahun"] and df_filtered[col].nunique() <= 20
    ]
    
    if cat_cols:
        # Create interactive tabs
        tabs = st.tabs([f"📈 {col.replace('_', ' ').title()}" for col in cat_cols])
        
        for tab, col in zip(tabs, cat_cols):
            with tab:
                # Chart container
                st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
                fig, grouped_df = create_interactive_chart(df_filtered, col)
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Data table with enhanced presentation
                with st.expander("📋 Lihat Data Tabel", expanded=False):
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
        st.caption("📊 Dashboard Analisis Anggaran Belanja Negara - Sumber Data: bidja.kemenkeu.go.id")
    with col2:
        st.caption(f"🕐 Diperbarui: {datetime.now().strftime('%d %B %Y %H:%M')}")
    with col3:
        st.caption("👨‍💻 Built with Streamlit")

# =============================================================================
# IMPROVEMENT 12: Enhanced Error Handling & Entry Point
# =============================================================================
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Terjadi kesalahan dalam aplikasi: {str(e)}")
        st.info("Silakan refresh halaman atau hubungi administrator.")

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
    page_title="Analisis Tren Anggaran dan Realisasi Belanja Negara",
    page_icon=":material/line_axis:",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.kemenkeu.go.id',
        'Report a bug': 'https://github.com/tubankum3/dashpmk/issues',
        'About': "Dashboard Anggaran Bidang PMK"
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
@st.cache_data(show_spinner=True)
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

def rupiah_separator(x):
    try:
        x = float(x)
    except:
        return x
    return f"Rp {x:,.0f}".replace(",", ".")

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

# Header =============================================================================
def header(selected_kl: str | None = None, selected_metric: str | None = None):
    """Create comprehensive dashboard header with breadcrumb and key info"""
    kl_text = "Semua K/L" if selected_kl == "Semua" else (selected_kl)
    metric_text = f" {selected_metric}" if selected_metric else ""
    st.markdown(f"""
    <div class="dashboard-header">
        <div class="breadcrumb">DASHBOARD / ANALISIS {metric_text} / {kl_text}</div>
        <h1 class="dashboard-title">Analisis Tren Anggaran & Realisasi Belanja Negara</h1>
    </div>
    """, unsafe_allow_html=True)

# Cards =============================================================================
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
            <div class="metric-sublabel">Pertumbuhan tahunan rata-rata selama rentang periode waktu tertentu</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # AAGR Card
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Tingkat Pertumbuhan Tahunan Rata-rata (AAGR)</div>
            <div class="metric-value">{metrics['aagr']:+.1f}%</div>
            <div class="metric-sublabel">Rata-rata tingkat pertumbuhan tahunan</div>
        </div>
        """, unsafe_allow_html=True)
        
# Sidebar =============================================================================
def sidebar(df):
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-section">
            <h3 style='margin: 0.1rem 0.1rem 0.1rem 0.1rem; color: var(--on-surface);'>Filter Data</h3>
        """, unsafe_allow_html=True)

        # === Ensure Tahun is numeric ===
        df["Tahun"] = pd.to_numeric(df["Tahun"], errors="coerce").astype("int64")

        # === Select K/L ===
        kl_list = sorted(df["KEMENTERIAN/LEMBAGA"].dropna().unique())
        kl_list.append("Semua")  # add "Semua" at the end
        
        selected_kl = st.selectbox(
            "Pilih Kementerian/Lembaga",
            kl_list,
            key="ministry_select",
            help="Pilih kementerian/lembaga untuk melihat analisis anggaran"
        )
        
        # === Filter data ===
        if selected_kl == "Semua":
            df_filtered = df.copy()
        else:
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
            current_year = int(datetime.now().strftime('%Y'))
            min_year = int(min(year_options))
            max_year_data = int(max(year_options))
        
            # default end-year → current year but capped by data max
            default_end = min(current_year, max_year_data)
        
            # default start-year → read from session or fall back to min_year
            default_start = st.session_state.get("filter__year_range", (min_year, default_end))[0]
            default_start = max(min_year, default_start)  # clamp to range
        
            selected_years = st.slider(
                "Rentang Tahun",
                min_value=min_year,
                max_value=max_year_data,  # slider max equals dataset max year
                value=(default_start, default_end),
                step=1,
                key="filter__year_range"
            )
                
        # === Advanced filters ===
        with st.expander("⚙️ Filter Lanjutan"):
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

def short_label_from_code(value, col_type):
    value = str(value)
    if col_type in ["SUMBER DANA", "FUNGSI", "JENIS BELANJA"]:
        return value[:2]  # first 2 digits
    elif col_type == "SUB FUNGSI":
        return value[:2] + " " + value[3:5]  # 2 digit + space + 2 digit
    elif col_type == "PROGRAM":
        return value[:2] + " " + value[3:5] # 2 digit 2 letters
    elif col_type == "KEGIATAN":
        return value[:4]  # first 4 digits
    elif col_type == "OUTPUT (KRO)":
        return value[:4] + " " + value[5:8]  # 4 digits + 3 letters
    elif col_type == "SUB OUTPUT (RO)":
        return value[:4] + " " + value[5:8] + " " + value[9:12]  # 4 digits + 3 letters + 3 digits
    elif col_type == "KOMPONEN":
        return value[:3] + " " + value[4:7] + " " + value[8:11]  # 3 letters + 3 digits + 3 digits
    elif col_type == "AKUN 4 DIGIT":
        return value[:4]
    else:
        return value.split(" ")[0]

def chart(df: pd.DataFrame, category_col: str, selected_metric: str, selected_kl: str, base_height=400, extra_height_per_line=3):
    """Create line chart with shortened labels for legend only"""
    
    # Validate inputs
    if df.empty:
        st.warning("Dataframe kosong, tidak dapat membuat chart")
        return None, None
        
    if category_col not in df.columns:
        st.error(f"Kolom '{category_col}' tidak ditemukan dalam dataframe")
        return None, None
    
    # Check if there's data to display
    if df[category_col].isna().all():
        st.warning(f"Kolom '{category_col}' tidak memiliki data yang valid")
        return None, None
    
    try:
        # Create short_label column for legend only
        df = df.copy()
        df["short_label"] = df[category_col].apply(lambda x: short_label_from_code(x, category_col))
        
        # Group by Tahun, category_col, AND short_label to preserve original values
        df_grouped = (
            df.groupby(["Tahun", category_col, "short_label"], as_index=False)["Nilai"]
              .sum()
        )

        # Check if we have data after grouping
        if df_grouped.empty:
            st.warning("Tidak ada data setelah pengelompokan")
            return None, None

        # Ensure Tahun is sorted and string
        df_grouped["Tahun"] = df_grouped["Tahun"].astype(str)
        df_grouped = df_grouped.sort_values("Tahun")

        # Adjust height dynamically
        n_groups = df_grouped["short_label"].nunique()
        height = base_height + (n_groups * extra_height_per_line if n_groups > 10 else 0)

        # Create the line chart
        fig = px.line(
            df_grouped,
            x="Tahun",
            y="Nilai",
            color="short_label",  # Use shortened label for color/legend only
            markers=True,
            title=f"📈 {selected_metric} BERDASARKAN {category_col} — {selected_kl}",
            labels={
                "Tahun": "Tahun",
                "Nilai": "Jumlah (Rp)",
                "short_label": category_col.replace("_", " ").title(),  # legend name
            },
            template="plotly_white",
            height=height,
        )

        # Update layout
        fig.update_layout(
            hovermode="closest",
            title_x=0,
            legend_title_text=category_col.replace("_", " ").title(),
            margin=dict(l=40, r=40, t=80, b=40),
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(family="Google Sans, Roboto, Arial"),
        )

        # Use original category_col values for hover, but show short_label in legend
        fig.update_traces(
            hovertemplate=(
                f"<b>%{{customdata}}</b><br>"  # Original category_col value
                "Tahun: %{x}<br>"
                "Rp %{y:,.0f}<extra></extra>"
            ),
            customdata=df_grouped[category_col],  # Pass original values for hover
            line=dict(width=2.5),
            marker=dict(size=7)
        )

        return fig, df_grouped
        
    except Exception as e:
        st.error(f"Error dalam membuat chart: {str(e)}")
        return None, None
    
# advanced filter =============================================================================
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
    st.markdown(f"<div class='material-card'>{'Semua Kementerian/Lembaga' if selected_kl == 'Semua' else selected_kl}</div>", unsafe_allow_html=True)
    cards(metrics, selected_kl=selected_kl, selected_metric=selected_metric)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # --- Visualization Section ---
    st.markdown("<div class='section-title'>TREN ANALISIS</div>", unsafe_allow_html=True)
    
    # Categorical columns for visualization - with proper validation
    cat_cols = [
        col for col in df_filtered.select_dtypes(include=["object"]).columns
        if col not in ["KEMENTERIAN/LEMBAGA", "Tahun"] and col in df_filtered.columns
    ]
        
    if cat_cols:
        # Show ALL categorical columns, not just first 5
        tab_labels = [f"📈 {col.replace('_', ' ').title()}" for col in cat_cols]
        
        # Create tabs for all categorical columns
        tabs = st.tabs(tab_labels)
        
        for tab, col in zip(tabs, cat_cols):
            with tab:
                if col in df_filtered.columns:
                    try:
                        # Check if this column has any non-null data
                        if df_filtered[col].notna().sum() == 0:
                            st.warning(f"Kolom '{col}' tidak memiliki data yang valid")
                            continue
                            
                        fig, grouped_df = chart(df_filtered, col, selected_metric, selected_kl)
                        if fig is not None:
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # --- Data table (wide format: metric x year) ---
                            with st.expander("📋 Data Tabel", expanded=True):
                                if grouped_df is not None and not grouped_df.empty:
                                    display_col = col
                                    df_display = grouped_df[["Tahun", display_col, "Nilai"]].copy()
                            
                                    # Pivot data
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
                            
                                    # Sort year columns
                                    tahun_cols = sorted([c for c in df_pivot.columns if c != display_col])
                                    df_pivot = df_pivot[[display_col] + tahun_cols]
                            
                                    # Keep numeric copy for Excel export
                                    df_excel = df_pivot.copy()
                            
                                    # Apply Rupiah formatting for display
                                    for c in tahun_cols:
                                        df_pivot[c] = df_pivot[c].apply(rupiah_separator)
                            
                                    df_pivot = df_pivot.rename(columns={display_col: selected_metric})
                            
                                    st.dataframe(df_pivot, use_container_width=True, hide_index=True)
                            
                                    # Download button (numeric values, not formatted strings)
                                    excel_buffer = io.BytesIO()
                                    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                                        df_excel.to_excel(writer, sheet_name="Data", index=False)
                                    st.download_button(
                                        label="Download Excel",
                                        data=excel_buffer.getvalue(),
                                        file_name=f"{selected_metric}_{col}_{selected_kl}.xlsx",
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                    )
                                else:
                                    st.info("Tidak ada data untuk ditampilkan dalam tabel")
                        else:
                            st.warning(f"Tidak dapat membuat chart untuk {col}")
                    except Exception as e:
                        st.error(f"Error creating chart for {col}: {str(e)}")
                        # Show raw data for debugging
                        with st.expander("🔧 Debug Data"):
                            st.write(f"Sample values in {col}:", df_filtered[col].head(10).tolist())
                else:
                    st.warning(f"Kolom {col} tidak ditemukan dalam data yang difilter")
    else:
        st.info("ℹ️ Tidak ada kolom kategorikal yang tersedia untuk visualisasi.")
    
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




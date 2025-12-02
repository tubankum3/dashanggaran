# =============================================================================
# Monitoring Anggaran - Revised Version v2
# Changes:
# 1. Ringkasan Perbandingan moved to main area (replaces metrics row)
# 2. Removed tab1 (Agregasi Data) - only Perbandingan tab remains
# 3. When no date comparison: show numeric columns comparison if > 1 selected
# =============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import io
import requests
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional, Tuple

# =============================================================================
# Page Configuration
# =============================================================================
st.set_page_config(
    page_title="Dashboard Analisis Anggaran dan Realisasi Belanja Negara",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.kemenkeu.go.id',
        'Report a bug': 'https://github.com/tubankum3/dashanggaran/issues',
        'About': "Monitoring Anggaran Belanja - Perbandingan Data Berdasarkan Tanggal Update"
    }
)

# =============================================================================
# Modern Dashboard Design CSS
# =============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --primary: #0066FF;
    --primary-dark: #0052CC;
    --primary-light: #4D94FF;
    --secondary: #10B981;
    --warning: #F59E0B;
    --error: #EF4444;
    --success: #10B981;
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
    --surface: #FFFFFF;
    --background: #F9FAFB;
    --on-surface: #111827;
    --on-primary: #FFFFFF;
    --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    --radius-sm: 6px;
    --radius-md: 8px;
    --radius-lg: 12px;
    --radius-xl: 16px;
    --radius-full: 9999px;
    --space-xs: 0.5rem;
    --space-sm: 0.75rem;
    --space-md: 1rem;
    --space-lg: 1.5rem;
    --space-xl: 2rem;
    --transition-fast: 150ms cubic-bezier(0.4, 0, 0.2, 1);
    --transition-base: 200ms cubic-bezier(0.4, 0, 0.2, 1);
}

* { font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
.stApp { background-color: var(--background); }

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

.material-card {
    background: var(--surface);
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow-sm);
    padding: var(--space-xl);
    margin-bottom: var(--space-lg);
    border: 1px solid var(--gray-200);
}

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

.stSidebar { background: var(--surface); border-right: 1px solid var(--gray-200); }

.stTabs [data-baseweb="tab-list"] { gap: var(--space-sm); border-bottom: 1px solid var(--gray-200); }
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border: none;
    border-bottom: 2px solid transparent;
    padding: var(--space-md) var(--space-lg);
    color: var(--gray-600);
    font-weight: 500;
    transition: all var(--transition-fast);
}
.stTabs [data-baseweb="tab"]:hover { color: var(--gray-900); background: var(--gray-50); }
.stTabs [aria-selected="true"] { color: var(--primary); border-bottom-color: var(--primary); }

.availability-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 14px;
    border-radius: var(--radius-full);
    font-size: 13px;
    font-weight: 500;
    margin: 8px 0;
}
.badge-available { background-color: #ECFDF5; color: #059669; border: 1px solid #A7F3D0; }
.badge-unavailable { background-color: #FEF2F2; color: #DC2626; border: 1px solid #FECACA; }

.comparison-header {
    background: var(--gray-100);
    padding: var(--space-md);
    border-radius: var(--radius-md);
    margin-bottom: var(--space-md);
}

.filter-container {
    background: var(--gray-50);
    padding: var(--space-md);
    border-radius: var(--radius-md);
    margin-bottom: var(--space-lg);
    border: 1px solid var(--gray-200);
}

::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: var(--gray-100); border-radius: var(--radius-full); }
::-webkit-scrollbar-thumb { background: var(--gray-400); border-radius: var(--radius-full); }
::-webkit-scrollbar-thumb:hover { background: var(--gray-500); }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Configuration
# =============================================================================
BASE_URL = "https://raw.githubusercontent.com/tubankum3/dashanggaran/main/"
FILENAME_PATTERN = "df_{YYYYMMDD}.csv.zip"
CSV_FILENAME_IN_ZIP = "df.csv"

DATE_START = date(2024, 1, 1)
DATE_END = date.today()

CONNECT_TIMEOUT = 10
READ_TIMEOUT = 300

# =============================================================================
# URL Builder Functions
# =============================================================================
def build_url(data_date: date) -> str:
    """Build the full URL for a given date."""
    filename = FILENAME_PATTERN.replace("{YYYYMMDD}", data_date.strftime('%Y%m%d'))
    return BASE_URL + filename


def parse_date_string(date_str: str) -> date:
    """Parse date string to date object."""
    return datetime.strptime(date_str, "%Y-%m-%d").date()


def format_date_label(data_date: date) -> str:
    """Format date as Indonesian label."""
    months_id = {
        1: 'Januari', 2: 'Februari', 3: 'Maret', 4: 'April',
        5: 'Mei', 6: 'Juni', 7: 'Juli', 8: 'Agustus',
        9: 'September', 10: 'Oktober', 11: 'November', 12: 'Desember'
    }
    return f"{data_date.day} {months_id[data_date.month]} {data_date.year}"


# =============================================================================
# Availability Check
# =============================================================================
@st.cache_data(ttl=300, show_spinner=False)
def check_data_availability(data_date: str) -> Tuple[bool, Optional[int]]:
    """Check if data file exists and get file size."""
    dt = parse_date_string(data_date)
    url = build_url(dt)
    
    try:
        response = requests.head(url, timeout=10, allow_redirects=True)
        if response.status_code == 200:
            content_length = response.headers.get('Content-Length')
            size = int(content_length) if content_length else None
            return True, size
        return False, None
    except:
        return False, None


def format_file_size(size_bytes: Optional[int]) -> str:
    """Format file size in human readable format."""
    if size_bytes is None:
        return "Unknown"
    if size_bytes >= 1_000_000_000:
        return f"{size_bytes / 1_000_000_000:.2f} GB"
    elif size_bytes >= 1_000_000:
        return f"{size_bytes / 1_000_000:.2f} MB"
    elif size_bytes >= 1_000:
        return f"{size_bytes / 1_000:.2f} KB"
    return f"{size_bytes} bytes"


def render_availability_badge(is_available: bool, file_size: Optional[int] = None) -> str:
    """Return HTML badge for availability status."""
    if is_available:
        size_text = f" • {format_file_size(file_size)}" if file_size else ""
        return f"""
        <div class="availability-badge badge-available">
            <span>✓</span>
            <span>Data Tersedia{size_text}</span>
        </div>
        """
    return """
    <div class="availability-badge badge-unavailable">
        <span>✗</span>
        <span>Data Tidak Tersedia</span>
    </div>
    """


# =============================================================================
# Streaming Download Functions
# =============================================================================
def download_with_progress(url: str, progress_placeholder) -> Optional[bytes]:
    """Download file with progress bar for large files."""
    try:
        response = requests.get(
            url, stream=True, 
            timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
            allow_redirects=True
        )
        response.raise_for_status()
        
        total_size = response.headers.get('Content-Length')
        total_size = int(total_size) if total_size else None
        
        chunks = []
        downloaded = 0
        chunk_size = 1024 * 1024
        
        progress_bar = progress_placeholder.progress(0, text="📥 Downloading...")
        
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                chunks.append(chunk)
                downloaded += len(chunk)
                
                if total_size:
                    progress = min(downloaded / total_size, 1.0)
                    progress_bar.progress(progress, text=f"📥 Downloading... {format_file_size(downloaded)} / {format_file_size(total_size)}")
                else:
                    progress_bar.progress(0.5, text=f"📥 Downloading... {format_file_size(downloaded)}")
        
        progress_bar.progress(1.0, text="✅ Download complete!")
        return b''.join(chunks)
        
    except requests.exceptions.Timeout:
        progress_placeholder.error("⏱️ Download timeout")
        return None
    except Exception as e:
        progress_placeholder.error(f"❌ Error: {e}")
        return None


# =============================================================================
# Data Loading Functions
# =============================================================================
def load_data_by_date_with_progress(data_date: str, progress_placeholder) -> pd.DataFrame:
    """Load budget data with progress indicator."""
    dt = parse_date_string(data_date)
    url = build_url(dt)
    
    content = download_with_progress(url, progress_placeholder)
    
    if content is None:
        return pd.DataFrame()
    
    if len(content) < 4 or content[:2] != b'PK':
        st.error("❌ File bukan ZIP yang valid")
        with st.expander("🔍 Debug Info"):
            st.write(f"File size: {format_file_size(len(content))}")
            st.code(content[:200].decode('utf-8', errors='replace'))
        return pd.DataFrame()
    
    try:
        progress_placeholder.info("📦 Extracting ZIP...")
        
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            csv_files = [f for f in z.namelist() if f.endswith('.csv')]
            
            if not csv_files:
                st.error(f"❌ No CSV files in ZIP")
                return pd.DataFrame()
            
            target_file = CSV_FILENAME_IN_ZIP if CSV_FILENAME_IN_ZIP in csv_files else csv_files[0]
            progress_placeholder.info(f"📄 Reading {target_file}...")
            
            with z.open(target_file) as file:
                df = pd.read_csv(file, low_memory=False)
        
        progress_placeholder.empty()
        df = preprocess_dataframe(df)
        df["_DATA_DATE"] = data_date
        return df
        
    except Exception as e:
        st.error(f"❌ Error: {e}")
        return pd.DataFrame()


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the loaded dataframe."""
    if df.empty:
        return df
    
    unnamed_cols = [c for c in df.columns if 'Unnamed' in str(c)]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)
    
    if "Tahun" in df.columns:
        df["Tahun"] = pd.to_numeric(df["Tahun"], errors="coerce").astype("Int64")
    
    return df


# =============================================================================
# Column Definitions
# =============================================================================
STRING_COLUMNS = [
    'KEMENTERIAN/LEMBAGA', 'SUMBER DANA', 'FUNGSI', 'SUB FUNGSI',
    'PROGRAM', 'KEGIATAN', 'OUTPUT (KRO)', 'SUB OUTPUT (RO)',
    'KOMPONEN', 'JENIS BELANJA', 'AKUN 4 DIGIT', 'Tahun'
]

NUMERIC_COLUMNS = [
    'REALISASI BELANJA KL (SAKTI)', 'PAGU DIPA REVISI', 'BLOKIR DIPA REVISI',
    'PAGU DIPA AWAL', 'BLOKIR DIPA AWAL', 'PAGU DIPA AWAL EFEKTIF',
    'PAGU DIPA REVISI EFEKTIF'
]


# =============================================================================
# Aggregation & Comparison Functions  
# =============================================================================
def aggregate_data(
    df: pd.DataFrame,
    group_cols: List[str],
    numeric_cols: List[str],
    agg_func: str = 'sum'
) -> pd.DataFrame:
    """Aggregate numeric columns by selected group columns."""
    available_group = [c for c in group_cols if c in df.columns]
    available_numeric = [c for c in numeric_cols if c in df.columns]
    
    if not available_group or not available_numeric:
        return pd.DataFrame()
    
    agg_dict = {col: agg_func for col in available_numeric}
    return df.groupby(available_group, as_index=False).agg(agg_dict)


def compare_datasets(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    group_cols: List[str],
    numeric_cols: List[str],
    date1: str,
    date2: str
) -> pd.DataFrame:
    """
    Compare two datasets and calculate differences.
    """
    if date1 == date2:
        st.warning("⚠️ Tanggal utama dan pembanding sama. Pilih tanggal yang berbeda untuk perbandingan.")
        return pd.DataFrame()
    
    available_group = [c for c in group_cols if c in df1.columns and c in df2.columns]
    available_numeric = [c for c in numeric_cols if c in df1.columns and c in df2.columns]
    
    if not available_group or not available_numeric:
        st.warning("⚠️ Tidak ada kolom yang cocok untuk perbandingan")
        return pd.DataFrame()
    
    agg1 = aggregate_data(df1, available_group, available_numeric)
    agg2 = aggregate_data(df2, available_group, available_numeric)
    
    if agg1.empty or agg2.empty:
        st.warning("⚠️ Data agregasi kosong")
        return pd.DataFrame()
    
    rename1 = {c: f"{c}_{date1}" for c in available_numeric}
    rename2 = {c: f"{c}_{date2}" for c in available_numeric}
    
    agg1_renamed = agg1.rename(columns=rename1)
    agg2_renamed = agg2.rename(columns=rename2)
    
    comparison = pd.merge(agg1_renamed, agg2_renamed, on=available_group, how='outer')
    
    for col in available_numeric:
        col1 = f"{col}_{date1}"
        col2 = f"{col}_{date2}"
        
        if col1 not in comparison.columns or col2 not in comparison.columns:
            continue
        
        comparison[f"SELISIH_{col}"] = comparison[col2].fillna(0) - comparison[col1].fillna(0)
        comparison[f"PCT_{col}"] = np.where(
            comparison[col1].fillna(0) != 0,
            (comparison[f"SELISIH_{col}"] / comparison[col1].fillna(0).abs()) * 100,
            np.nan
        )
    
    return comparison


# =============================================================================
# Utility Functions
# =============================================================================
def format_rupiah(value: float) -> str:
    """Format numeric value to Indonesian Rupiah currency format (short)."""
    if pd.isna(value) or value == 0:
        return "Rp 0"
    
    abs_value = abs(value)
    sign = "-" if value < 0 else ""
    
    if abs_value >= 1_000_000_000_000:
        return f"{sign}Rp {abs_value/1_000_000_000_000:.2f} T"
    elif abs_value >= 1_000_000_000:
        return f"{sign}Rp {abs_value/1_000_000_000:.2f} M"
    elif abs_value >= 1_000_000:
        return f"{sign}Rp {abs_value/1_000_000:.2f} Jt"
    return f"{sign}Rp {abs_value:,.0f}"


def rupiah_separator(x):
    """Format number with Indonesian thousand separator (full number)."""
    try:
        x = float(x)
    except:
        return x
    if pd.isna(x):
        return "-"
    return f"Rp {x:,.0f}".replace(",", ".")


def format_percentage(x):
    """Format percentage value."""
    try:
        x = float(x)
    except:
        return x
    if pd.isna(x):
        return "-"
    return f"{x:,.2f}%"


def format_dataframe_for_display(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    """
    Format dataframe with rupiah separator for numeric columns.
    Returns a copy with formatted values for display.
    """
    if df.empty:
        return df
    
    df_display = df.copy()
    
    for col in df_display.columns:
        # Check if column is a numeric column or derived from numeric columns
        is_numeric_col = any(nc in col for nc in numeric_cols)
        is_selisih_col = col.startswith('SELISIH_')
        is_pct_col = col.startswith('PCT_')
        
        if is_pct_col:
            # Format as percentage
            df_display[col] = df_display[col].apply(format_percentage)
        elif is_numeric_col or is_selisih_col:
            # Format as rupiah
            df_display[col] = df_display[col].apply(rupiah_separator)
    
    return df_display


def filter_dataframe(df: pd.DataFrame, filters: Dict[str, List]) -> pd.DataFrame:
    """
    Filter dataframe based on selected filter values.
    """
    if df.empty:
        return df
    
    filtered_df = df.copy()
    
    for col, values in filters.items():
        if col in filtered_df.columns and values:
            filtered_df = filtered_df[filtered_df[col].astype(str).isin([str(v) for v in values])]
    
    return filtered_df


# =============================================================================
# Sidebar Components
# =============================================================================
def render_date_selector_with_availability(key_prefix: str = "") -> Tuple[Optional[str], Optional[str], bool, bool, bool, Optional[int]]:
    """Render date selection with availability indicator."""
    st.sidebar.markdown("### 📅 Pilih Tanggal Data")
    
    primary_dt = st.sidebar.date_input(
        "Tanggal Data Utama",
        value=date(2025, 10, 27),
        min_value=DATE_START,
        max_value=DATE_END,
        key=f"{key_prefix}primary_date",
        help="Pilih tanggal update data"
    )
    primary_date = primary_dt.strftime("%Y-%m-%d")
    
    with st.sidebar:
        with st.spinner("Memeriksa..."):
            primary_available, primary_size = check_data_availability(primary_date)
        
        st.markdown(render_availability_badge(primary_available, primary_size), unsafe_allow_html=True)
        
    st.sidebar.markdown("---")
    
    enable_comparison = st.sidebar.checkbox(
        "🔄 Bandingkan dengan tanggal lain",
        value=False,
        key=f"{key_prefix}enable_comparison"
    )
    
    comparison_date = None
    comparison_available = False
    
    if enable_comparison:
        default_comparison = date(2025, 11, 11) if primary_dt != date(2025, 11, 11) else date(2025, 10, 27)
        
        comparison_dt = st.sidebar.date_input(
            "Tanggal Pembanding",
            value=default_comparison,
            min_value=DATE_START,
            max_value=DATE_END,
            key=f"{key_prefix}comparison_date"
        )
        comparison_date = comparison_dt.strftime("%Y-%m-%d")
        
        if comparison_date == primary_date:
            st.sidebar.error("⚠️ Pilih tanggal yang berbeda!")
        
        with st.sidebar:
            with st.spinner("Memeriksa..."):
                comparison_available, comp_size = check_data_availability(comparison_date)
            
            st.markdown(render_availability_badge(comparison_available, comp_size), unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**📋 Tanggal Tersedia:**")
    st.sidebar.markdown("- 27 Oktober 2025\n- 11 November 2025")
    
    return primary_date, comparison_date, enable_comparison, primary_available, comparison_available, primary_size


def render_aggregation_options(key_prefix: str = "") -> Tuple[List[str], List[str], str]:
    """Render aggregation options sidebar component with minimum 1 selection."""
    st.sidebar.markdown("### 📊 Opsi Agregasi")
    
    group_cols = st.sidebar.multiselect(
        "Group By (Kolom String)",
        options=STRING_COLUMNS,
        default=['Tahun', 'KEMENTERIAN/LEMBAGA'],
        help="Pilih minimal 1 kolom untuk pengelompokan",
        key=f"{key_prefix}group_cols"
    )
    
    if len(group_cols) < 1:
        st.sidebar.error("⚠️ Pilih minimal 1 kolom string")
    
    numeric_cols = st.sidebar.multiselect(
        "Agregasi (Kolom Numerik)",
        options=NUMERIC_COLUMNS,
        default=['REALISASI BELANJA KL (SAKTI)', 'PAGU DIPA REVISI'],
        help="Pilih minimal 1 kolom untuk agregasi",
        key=f"{key_prefix}numeric_cols"
    )
    
    if len(numeric_cols) < 1:
        st.sidebar.error("⚠️ Pilih minimal 1 kolom numerik")
    
    agg_func = st.sidebar.selectbox(
        "Fungsi Agregasi",
        options=['sum', 'mean', 'count', 'min', 'max'],
        format_func=lambda x: {
            'sum': 'Jumlah (Sum)',
            'mean': 'Rata-rata (Mean)', 
            'count': 'Hitung (Count)',
            'min': 'Minimum',
            'max': 'Maksimum'
        }.get(x, x),
        key=f"{key_prefix}agg_func"
    )
    
    return group_cols, numeric_cols, agg_func


def render_data_info(df: pd.DataFrame, label: str, key_prefix: str = ""):
    """Render data info expander in sidebar."""
    with st.sidebar.expander(f"📋 Info Data - {label}", expanded=False):
        col1, col2, col3 = st.columns(3)
        col1.metric("📊 Baris", f"{len(df):,}")
        col2.metric("📋 Kolom", len(df.columns))
        if "Tahun" in df.columns:
            tahun_list = df["Tahun"].dropna().unique()
            col3.metric("📆 Tahun", ", ".join(map(str, sorted(tahun_list))))


def render_comparison_filters(df: pd.DataFrame, group_cols: List[str], key_prefix: str = "") -> Dict[str, List]:
    """
    Render filter selectors for comparison table.
    """
    filters = {}
    
    st.markdown("""
    <div class="filter-container">
        <strong>🔍 Filter Data</strong>
    </div>
    """, unsafe_allow_html=True)
    
    available_cols = [col for col in group_cols if col in df.columns]
    
    if not available_cols:
        return filters
    
    n_filters = len(available_cols)
    cols_per_row = min(n_filters, 4)
    
    filter_cols = st.columns(cols_per_row)
    
    for idx, col in enumerate(available_cols):
        col_idx = idx % cols_per_row
        
        with filter_cols[col_idx]:
            unique_values = df[col].dropna().unique().tolist()
            unique_values = sorted([str(v) for v in unique_values])
            
            selected = st.multiselect(
                f"Filter: {col}",
                options=unique_values,
                default=[],
                key=f"{key_prefix}filter_{col}",
                help=f"Pilih nilai untuk filter {col}. Kosongkan untuk tampilkan semua."
            )
            
            if selected:
                filters[col] = selected
    
    if filters:
        active_filters = sum(1 for v in filters.values() if v)
        st.caption(f"🔹 {active_filters} filter aktif")
    
    return filters


# =============================================================================
# Main Application
# =============================================================================
def main():
    """Main application."""
    
    # Header
    st.markdown("""
    <div class="dashboard-header">
        <p class="breadcrumb">KEMENTERIAN KEUANGAN RI • BIDJA</p>
        <h1 class="dashboard-title">Monitoring Anggaran</h1>
        <p class="dashboard-subtitle">Perbandingan Data Anggaran Berdasarkan Tanggal Update Database</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Pengaturan")
        result = render_date_selector_with_availability()
        primary_date, comparison_date, enable_comparison, primary_available, comparison_available, primary_size = result
        st.divider()
        group_cols, numeric_cols, agg_func = render_aggregation_options()
    
    # Main content
    if not primary_available:
        st.markdown("""
        <div class="material-card">
            <h3>⚠️ Data Tidak Tersedia</h3>
            <p>Data tidak ditemukan untuk tanggal yang dipilih.</p>
            <p><strong>Tanggal yang tersedia:</strong></p>
            <ul>
                <li>27 Oktober 2025</li>
                <li>11 November 2025</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        return
    
    if primary_size and primary_size > 50_000_000:
        st.info(f"📦 File besar ({format_file_size(primary_size)}). Download mungkin memakan waktu...")
    
    progress_placeholder = st.empty()
    df_primary = load_data_by_date_with_progress(primary_date, progress_placeholder)
    
    if df_primary.empty:
        return
    
    st.success(f"✅ Data dimuat: **{len(df_primary):,}** baris | {format_date_label(parse_date_string(primary_date))}")
    
    # Render Info Data in sidebar for primary data
    render_data_info(df_primary, format_date_label(parse_date_string(primary_date)), "primary_")
    
    # Validate minimum selections
    if len(group_cols) < 1 or len(numeric_cols) < 1:
        st.info("👆 Pilih minimal 1 kolom string dan 1 kolom numerik di sidebar")
        with st.expander("📋 Preview Data Mentah"):
            st.dataframe(df_primary.head(100), width="stretch")
        return
    
    agg_primary = aggregate_data(df_primary, group_cols, numeric_cols, agg_func)
    
    # Format for display with rupiah separator
    agg_primary_display = format_dataframe_for_display(agg_primary, numeric_cols)
    
    # =========================================================================
    # Comparison mode - comparing between dates
    # =========================================================================
    if enable_comparison and comparison_date and comparison_available:
        if comparison_date == primary_date:
            st.error("⚠️ **Tanggal sama!** Pilih tanggal pembanding yang berbeda dari tanggal utama.")
            st.dataframe(agg_primary_display, width="stretch", hide_index=True)
            return
        
        progress_placeholder2 = st.empty()
        df_comparison = load_data_by_date_with_progress(comparison_date, progress_placeholder2)
        
        if not df_comparison.empty:
            st.success(f"✅ Data pembanding dimuat: **{len(df_comparison):,}** baris")
            
            # Render Info Data in sidebar for comparison data
            render_data_info(df_comparison, format_date_label(parse_date_string(comparison_date)), "comparison_")
            
            comparison_df = compare_datasets(
                df_primary, df_comparison,
                group_cols, numeric_cols,
                primary_date, comparison_date
            )
            
            if comparison_df.empty:
                st.dataframe(agg_primary_display, width="stretch", hide_index=True)
                return
            
            # Ringkasan Perbandingan - Show at top
            st.markdown("### 📈 Ringkasan Perbandingan Antar Tanggal")
            
            for col in numeric_cols:
                if col in df_primary.columns and col in df_comparison.columns:
                    st.markdown(f"**{col}**")
                    
                    c1, c2, c3 = st.columns(3)
                    
                    val1 = df_primary[col].sum()
                    val2 = df_comparison[col].sum()
                    diff = val2 - val1
                    pct = (diff / abs(val1) * 100) if val1 != 0 else 0
                    
                    c1.metric(format_date_label(parse_date_string(primary_date)), format_rupiah(val1))
                    c2.metric(format_date_label(parse_date_string(comparison_date)), format_rupiah(val2))
                    c3.metric("Selisih", format_rupiah(diff), f"{pct:+.2f}%")
            
            st.divider()
            
            # Perbandingan Detail (No tabs - direct display)
            st.markdown("### 🔄 Perbandingan Detail")
            st.markdown(f"""
            <div class="comparison-header">
                <strong>{format_date_label(parse_date_string(primary_date))}</strong> vs 
                <strong>{format_date_label(parse_date_string(comparison_date))}</strong>
            </div>
            """, unsafe_allow_html=True)
            
            # Filter section
            filters = render_comparison_filters(comparison_df, group_cols, key_prefix="comp_")
            
            # Apply filters if any
            filtered_comparison_df = filter_dataframe(comparison_df, filters)
            
            # Show filter result info
            if filters:
                st.info(f"Menampilkan **{len(filtered_comparison_df):,}** dari **{len(comparison_df):,}** baris")
            
            with st.expander("ℹ️ Keterangan Kolom", expanded=False):
                st.markdown(f"""
                - `[kolom]_{primary_date}` = Nilai tanggal utama
                - `[kolom]_{comparison_date}` = Nilai tanggal pembanding
                - `SELISIH_[kolom]` = Selisih (pembanding - utama)
                - `PCT_[kolom]` = Persentase perubahan
                """)
            
            # Format comparison dataframe for display
            comparison_display = format_dataframe_for_display(filtered_comparison_df, numeric_cols)
            st.dataframe(comparison_display, width="stretch", hide_index=True)
            
            # Download button
            csv2 = filtered_comparison_df.to_csv(index=False)
            st.download_button(
                "📥 Download Perbandingan", 
                csv2, 
                f"perbandingan_{primary_date}_vs_{comparison_date}.csv", 
                "text/csv", 
                key="dl2"
            )
        else:
            st.dataframe(agg_primary_display, width="stretch", hide_index=True)
    
    # =========================================================================
    # No date comparison mode
    # =========================================================================
    else:
        # If more than 1 numeric column selected, show comparison between columns
        if len(numeric_cols) > 1:
            st.markdown("### 📈 Ringkasan Kolom Numerik")
            
            # Create metrics for each numeric column (display in rows of 3)
            cols_per_row = 3
            for i in range(0, len(numeric_cols), cols_per_row):
                chunk = numeric_cols[i:i + cols_per_row]
                cols = st.columns(len(chunk))
                
                for idx, col in enumerate(chunk):
                    if col in df_primary.columns:
                        val = df_primary[col].sum()
                        cols[idx].metric(col, format_rupiah(val))
            
            # Show comparison table between numeric columns
            st.divider()
            st.markdown("### 📊 Perbandingan Antar Kolom Numerik")
            
            # Create summary dataframe for numeric columns comparison
            summary_data = []
            for col in numeric_cols:
                if col in df_primary.columns:
                    summary_data.append({
                        'Kolom': col,
                        'Total': df_primary[col].sum(),
                        'Rata-rata': df_primary[col].mean(),
                        'Min': df_primary[col].min(),
                        'Max': df_primary[col].max(),
                        'Count': df_primary[col].count()
                    })
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                
                # Format for display
                summary_display = summary_df.copy()
                for col in ['Total', 'Rata-rata', 'Min', 'Max']:
                    summary_display[col] = summary_display[col].apply(rupiah_separator)
                summary_display['Count'] = summary_display['Count'].apply(lambda x: f"{x:,}")
                
                st.dataframe(summary_display, width="stretch", hide_index=True)
            
            st.divider()
        
        # Always show the aggregated data table
        st.markdown(f"### 📋 Data Agregasi - {format_date_label(parse_date_string(primary_date))}")
        st.dataframe(agg_primary_display, width="stretch", hide_index=True)
        csv = agg_primary.to_csv(index=False)
        st.download_button("📥 Download CSV", csv, f"agregasi_{primary_date}.csv", "text/csv")


if __name__ == "__main__":
    main()

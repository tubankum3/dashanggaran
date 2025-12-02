# =============================================================================
# Data Loading with Streaming Download + Modern CSS Styling
# Fixed: Same date comparison error + Streamlit deprecation warnings
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
        'About': "Dashboard Anggaran - Perbandingan Data Berdasarkan Tanggal Update"
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
    
    FIXED: Now handles same date comparison and missing columns gracefully.
    """
    # Validate dates are different
    if date1 == date2:
        st.warning("⚠️ Tanggal utama dan pembanding sama. Pilih tanggal yang berbeda untuk perbandingan.")
        return pd.DataFrame()
    
    available_group = [c for c in group_cols if c in df1.columns and c in df2.columns]
    available_numeric = [c for c in numeric_cols if c in df1.columns and c in df2.columns]
    
    if not available_group or not available_numeric:
        st.warning("⚠️ Tidak ada kolom yang cocok untuk perbandingan")
        return pd.DataFrame()
    
    # Aggregate both datasets
    agg1 = aggregate_data(df1, available_group, available_numeric)
    agg2 = aggregate_data(df2, available_group, available_numeric)
    
    if agg1.empty or agg2.empty:
        st.warning("⚠️ Data agregasi kosong")
        return pd.DataFrame()
    
    # Rename columns before merge (to avoid suffix conflicts)
    rename1 = {c: f"{c}_{date1}" for c in available_numeric}
    rename2 = {c: f"{c}_{date2}" for c in available_numeric}
    
    agg1_renamed = agg1.rename(columns=rename1)
    agg2_renamed = agg2.rename(columns=rename2)
    
    # Merge datasets
    comparison = pd.merge(agg1_renamed, agg2_renamed, on=available_group, how='outer')
    
    # Calculate differences - with column existence check
    for col in available_numeric:
        col1 = f"{col}_{date1}"
        col2 = f"{col}_{date2}"
        
        # Verify columns exist after merge
        if col1 not in comparison.columns:
            st.warning(f"⚠️ Kolom {col1} tidak ditemukan setelah merge")
            continue
        if col2 not in comparison.columns:
            st.warning(f"⚠️ Kolom {col2} tidak ditemukan setelah merge")
            continue
        
        # Calculate difference
        comparison[f"SELISIH_{col}"] = comparison[col2].fillna(0) - comparison[col1].fillna(0)
        
        # Calculate percentage change
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
    """Format numeric value to Indonesian Rupiah currency format."""
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
    """Format number with Indonesian thousand separator."""
    try:
        x = float(x)
    except:
        return x
    return f"Rp {x:,.0f}".replace(",", ".")


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
        
        with st.expander("🔗 Info URL", expanded=False):
            url = build_url(primary_dt)
            st.code(url, language=None)
            if primary_available:
                st.success(f"✓ File size: {format_file_size(primary_size)}")
    
    st.sidebar.markdown("---")
    
    enable_comparison = st.sidebar.checkbox(
        "🔄 Bandingkan dengan tanggal lain",
        value=False,
        key=f"{key_prefix}enable_comparison"
    )
    
    comparison_date = None
    comparison_available = False
    
    if enable_comparison:
        # Default to a different date
        default_comparison = date(2025, 11, 11) if primary_dt != date(2025, 11, 11) else date(2025, 10, 27)
        
        comparison_dt = st.sidebar.date_input(
            "Tanggal Pembanding",
            value=default_comparison,
            min_value=DATE_START,
            max_value=DATE_END,
            key=f"{key_prefix}comparison_date"
        )
        comparison_date = comparison_dt.strftime("%Y-%m-%d")
        
        # Check if same date selected
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
    """Render aggregation options sidebar component."""
    st.sidebar.markdown("### 📊 Opsi Agregasi")
    
    group_cols = st.sidebar.multiselect(
        "Group By (Kolom String)",
        options=STRING_COLUMNS,
        default=['KEMENTERIAN/LEMBAGA', 'PROGRAM'],
        help="Pilih minimal 2 kolom untuk pengelompokan",
        key=f"{key_prefix}group_cols"
    )
    
    if len(group_cols) < 2:
        st.sidebar.warning("⚠️ Pilih minimal 2 kolom string")
    
    numeric_cols = st.sidebar.multiselect(
        "Agregasi (Kolom Numerik)",
        options=NUMERIC_COLUMNS,
        default=['REALISASI BELANJA KL (SAKTI)', 'PAGU DIPA REVISI'],
        help="Pilih minimal 2 kolom untuk agregasi",
        key=f"{key_prefix}numeric_cols"
    )
    
    if len(numeric_cols) < 2:
        st.sidebar.warning("⚠️ Pilih minimal 2 kolom numerik")
    
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


# =============================================================================
# Main Application
# =============================================================================
def main():
    """Main application."""
    
    # Header
    st.markdown("""
    <div class="dashboard-header">
        <p class="breadcrumb">KEMENTERIAN KEUANGAN RI • BIDJA</p>
        <h1 class="dashboard-title">Dashboard Analisis Anggaran</h1>
        <p class="dashboard-subtitle">Perbandingan Data Berdasarkan Tanggal Update</p>
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
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📊 Total Baris", f"{len(df_primary):,}")
    col2.metric("📋 Total Kolom", len(df_primary.columns))
    col3.metric("📅 Tanggal Data", primary_date)
    if "Tahun" in df_primary.columns:
        tahun_list = df_primary["Tahun"].dropna().unique()
        col4.metric("📆 Tahun Anggaran", ", ".join(map(str, sorted(tahun_list))))
    
    st.divider()
    
    if len(group_cols) < 2 or len(numeric_cols) < 2:
        st.info("👆 Pilih minimal 2 kolom string dan 2 kolom numerik di sidebar")
        with st.expander("📋 Preview Data Mentah"):
            st.dataframe(df_primary.head(100), width="stretch")
        return
    
    agg_primary = aggregate_data(df_primary, group_cols, numeric_cols, agg_func)
    
    # Comparison mode - with same date check
    if enable_comparison and comparison_date and comparison_available:
        # Check for same date
        if comparison_date == primary_date:
            st.error("⚠️ **Tanggal sama!** Pilih tanggal pembanding yang berbeda dari tanggal utama.")
            st.markdown(f"### Data Agregasi - {format_date_label(parse_date_string(primary_date))}")
            st.dataframe(agg_primary, width="stretch", hide_index=True)
            return
        
        progress_placeholder2 = st.empty()
        df_comparison = load_data_by_date_with_progress(comparison_date, progress_placeholder2)
        
        if not df_comparison.empty:
            st.success(f"✅ Data pembanding dimuat: **{len(df_comparison):,}** baris")
            
            comparison_df = compare_datasets(
                df_primary, df_comparison,
                group_cols, numeric_cols,
                primary_date, comparison_date
            )
            
            if comparison_df.empty:
                st.dataframe(agg_primary, width="stretch", hide_index=True)
                return
            
            # Tabs
            tab1, tab2, tab3 = st.tabs([
                f"📊 Data {format_date_label(parse_date_string(primary_date))}",
                "🔄 Perbandingan",
                "📈 Ringkasan"
            ])
            
            with tab1:
                st.markdown(f"### Agregasi Data - {format_date_label(parse_date_string(primary_date))}")
                st.dataframe(agg_primary, width="stretch", hide_index=True)
                csv1 = agg_primary.to_csv(index=False)
                st.download_button("📥 Download CSV", csv1, f"agregasi_{primary_date}.csv", "text/csv", key="dl1")
            
            with tab2:
                st.markdown("### Perbandingan Data")
                st.markdown(f"""
                <div class="comparison-header">
                    <strong>{format_date_label(parse_date_string(primary_date))}</strong> vs 
                    <strong>{format_date_label(parse_date_string(comparison_date))}</strong>
                </div>
                """, unsafe_allow_html=True)
                
                st.info(f"""
                **Keterangan Kolom:**
                - `[kolom]_{primary_date}` = Nilai tanggal utama
                - `[kolom]_{comparison_date}` = Nilai tanggal pembanding
                - `SELISIH_[kolom]` = Selisih (pembanding - utama)
                - `PCT_[kolom]` = Persentase perubahan
                """)
                
                st.dataframe(comparison_df, width="stretch", hide_index=True)
                csv2 = comparison_df.to_csv(index=False)
                st.download_button("📥 Download Perbandingan", csv2, f"perbandingan_{primary_date}_vs_{comparison_date}.csv", "text/csv", key="dl2")
            
            with tab3:
                st.markdown("### Ringkasan Perbandingan")
                
                for col in numeric_cols:
                    if col in df_primary.columns and col in df_comparison.columns:
                        st.markdown(f"#### {col}")
                        
                        c1, c2, c3 = st.columns(3)
                        
                        val1 = df_primary[col].sum()
                        val2 = df_comparison[col].sum()
                        diff = val2 - val1
                        pct = (diff / abs(val1) * 100) if val1 != 0 else 0
                        
                        c1.metric(format_date_label(parse_date_string(primary_date)), format_rupiah(val1))
                        c2.metric(format_date_label(parse_date_string(comparison_date)), format_rupiah(val2))
                        c3.metric("Selisih", format_rupiah(diff), f"{pct:+.2f}%")
                        st.divider()
        else:
            st.dataframe(agg_primary, width="stretch", hide_index=True)
    else:
        st.markdown(f"### Data Agregasi - {format_date_label(parse_date_string(primary_date))}")
        st.dataframe(agg_primary, width="stretch", hide_index=True)
        csv = agg_primary.to_csv(index=False)
        st.download_button("📥 Download CSV", csv, f"agregasi_{primary_date}.csv", "text/csv")


if __name__ == "__main__":
    main()

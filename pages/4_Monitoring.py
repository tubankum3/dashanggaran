# =============================================================================
# Data Loading with Date Picker + Enhanced Debugging
# =============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import gzip
import io
import requests
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional, Tuple

# =============================================================================
# Configuration
# =============================================================================
BASE_URL = "https://raw.githubusercontent.com/tubankum3/dashpmk/main/"

# File patterns to try (in order)
FILENAME_PATTERNS = [
    "df_{YYYYMMDD}.csv.zip",
    "df_{YYYYMMDD}.csv.gz",
    "df_{YYYYMMDD}.csv",
]

CSV_FILENAME_IN_ZIP = "df.csv"

DATE_START = date(2024, 1, 1)
DATE_END = date.today()

# =============================================================================
# URL Builder Functions
# =============================================================================
def build_url(data_date: date, pattern: str) -> str:
    """Build the full URL for a given date and pattern."""
    filename = pattern.replace("{YYYYMMDD}", data_date.strftime('%Y%m%d'))
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
# File Type Detection
# =============================================================================
def detect_file_type(content: bytes) -> str:
    """
    Detect file type from content bytes.
    
    Returns: 'zip', 'gzip', 'csv', 'html', or 'unknown'
    """
    if len(content) < 4:
        return 'unknown'
    
    # Check magic bytes
    if content[:2] == b'PK':
        return 'zip'
    
    if content[:2] == b'\x1f\x8b':
        return 'gzip'
    
    # Check if HTML (error page)
    try:
        text_start = content[:500].decode('utf-8', errors='ignore').lower()
        if '<html' in text_start or '<!doctype' in text_start or '404' in text_start:
            return 'html'
    except:
        pass
    
    # Check if CSV-like
    try:
        text = content[:2000].decode('utf-8')
        lines = text.split('\n')
        if len(lines) > 1:
            # Check if first line looks like headers (contains letters and commas/semicolons)
            first_line = lines[0]
            if (',' in first_line or ';' in first_line) and any(c.isalpha() for c in first_line):
                return 'csv'
    except:
        pass
    
    return 'unknown'


def get_content_preview(content: bytes, max_chars: int = 200) -> str:
    """Get a readable preview of content."""
    try:
        text = content[:max_chars].decode('utf-8', errors='replace')
        # Clean up for display
        text = text.replace('\n', '\\n').replace('\r', '\\r')
        return text
    except:
        return content[:max_chars].hex()


# =============================================================================
# Availability Check Functions
# =============================================================================
@st.cache_data(ttl=300, show_spinner=False)
def check_data_availability(data_date: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Check if data file exists for a given date.
    
    Returns:
        Tuple of (is_available, working_url, file_type)
    """
    dt = parse_date_string(data_date)
    
    for pattern in FILENAME_PATTERNS:
        url = build_url(dt, pattern)
        try:
            # Use GET with stream to check content type
            response = requests.get(url, timeout=10, stream=True)
            if response.status_code == 200:
                # Read first chunk to detect type
                first_chunk = next(response.iter_content(chunk_size=1024), b'')
                file_type = detect_file_type(first_chunk)
                
                if file_type in ['zip', 'gzip', 'csv']:
                    return True, url, file_type
                    
        except Exception as e:
            continue
    
    return False, None, None


def render_availability_badge(is_available: bool, file_type: str = None) -> str:
    """Return HTML badge for availability status."""
    if is_available:
        type_text = f" ({file_type.upper()})" if file_type else ""
        return f"""
        <span style="
            background-color: #10B981;
            color: white;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: 500;
        ">✓ Data Tersedia{type_text}</span>
        """
    else:
        return """
        <span style="
            background-color: #EF4444;
            color: white;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: 500;
        ">✗ Data Tidak Tersedia</span>
        """


# =============================================================================
# Data Loading Functions
# =============================================================================
@st.cache_data(show_spinner=True, ttl=3600)
def load_data_by_date(data_date: str, debug: bool = False) -> pd.DataFrame:
    """
    Load budget data for a specific date.
    Automatically detects file format.
    """
    dt = parse_date_string(data_date)
    
    errors = []
    
    for pattern in FILENAME_PATTERNS:
        url = build_url(dt, pattern)
        
        try:
            response = requests.get(url, timeout=60)
            
            if response.status_code == 404:
                errors.append(f"{pattern}: 404 Not Found")
                continue
            
            response.raise_for_status()
            content = response.content
            
            if len(content) == 0:
                errors.append(f"{pattern}: Empty response")
                continue
            
            # Detect type
            file_type = detect_file_type(content)
            
            if debug:
                st.write(f"**URL:** `{url}`")
                st.write(f"**Size:** {len(content):,} bytes")
                st.write(f"**Detected type:** {file_type}")
                st.write(f"**Preview:** `{get_content_preview(content, 100)}`")
            
            # Parse based on type
            if file_type == 'zip':
                df = load_from_zip(content, data_date)
                if not df.empty:
                    return df
                errors.append(f"{pattern}: ZIP parsing failed")
                    
            elif file_type == 'gzip':
                df = load_from_gzip(content, data_date)
                if not df.empty:
                    return df
                errors.append(f"{pattern}: GZIP parsing failed")
                    
            elif file_type == 'csv':
                df = load_from_csv(content, data_date)
                if not df.empty:
                    return df
                errors.append(f"{pattern}: CSV parsing failed")
                
            elif file_type == 'html':
                preview = get_content_preview(content, 100)
                errors.append(f"{pattern}: Received HTML (error page?): {preview}")
                
            else:
                preview = get_content_preview(content, 100)
                errors.append(f"{pattern}: Unknown format. Preview: {preview}")
                
        except requests.exceptions.Timeout:
            errors.append(f"{pattern}: Timeout")
        except requests.exceptions.HTTPError as e:
            errors.append(f"{pattern}: HTTP {e.response.status_code}")
        except Exception as e:
            errors.append(f"{pattern}: {str(e)}")
    
    # All patterns failed
    st.error("❌ Gagal memuat data")
    with st.expander("🔍 Detail Error", expanded=True):
        for err in errors:
            st.write(f"- {err}")
    
    return pd.DataFrame()


def load_from_zip(content: bytes, data_date: str) -> pd.DataFrame:
    """Load DataFrame from ZIP content."""
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            file_list = z.namelist()
            csv_files = [f for f in file_list if f.endswith('.csv')]
            
            if not csv_files:
                st.warning(f"ZIP contains: {file_list}")
                return pd.DataFrame()
            
            target_file = CSV_FILENAME_IN_ZIP if CSV_FILENAME_IN_ZIP in csv_files else csv_files[0]
            
            with z.open(target_file) as file:
                df = pd.read_csv(file, low_memory=False)
        
        df = preprocess_dataframe(df)
        df["_DATA_DATE"] = data_date
        return df
        
    except Exception as e:
        st.warning(f"ZIP error: {e}")
        return pd.DataFrame()


def load_from_gzip(content: bytes, data_date: str) -> pd.DataFrame:
    """Load DataFrame from GZIP content."""
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(content)) as gz:
            df = pd.read_csv(gz, low_memory=False)
        
        df = preprocess_dataframe(df)
        df["_DATA_DATE"] = data_date
        return df
        
    except Exception as e:
        st.warning(f"GZIP error: {e}")
        return pd.DataFrame()


def load_from_csv(content: bytes, data_date: str) -> pd.DataFrame:
    """Load DataFrame from CSV content."""
    # Try different encodings and separators
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    separators = [',', ';', '\t']
    
    for encoding in encodings:
        for sep in separators:
            try:
                df = pd.read_csv(
                    io.BytesIO(content), 
                    low_memory=False, 
                    encoding=encoding,
                    sep=sep
                )
                
                # Check if parsing was successful (more than 1 column)
                if len(df.columns) > 1:
                    df = preprocess_dataframe(df)
                    df["_DATA_DATE"] = data_date
                    return df
                    
            except Exception:
                continue
    
    st.warning("CSV parsing failed with all encoding/separator combinations")
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
    """Compare two datasets and calculate differences."""
    available_group = [c for c in group_cols if c in df1.columns and c in df2.columns]
    available_numeric = [c for c in numeric_cols if c in df1.columns and c in df2.columns]
    
    if not available_group or not available_numeric:
        return pd.DataFrame()
    
    agg1 = aggregate_data(df1, available_group, available_numeric)
    agg2 = aggregate_data(df2, available_group, available_numeric)
    
    agg1 = agg1.rename(columns={c: f"{c}_{date1}" for c in available_numeric})
    agg2 = agg2.rename(columns={c: f"{c}_{date2}" for c in available_numeric})
    
    comparison = pd.merge(agg1, agg2, on=available_group, how='outer')
    
    for col in available_numeric:
        col1, col2 = f"{col}_{date1}", f"{col}_{date2}"
        comparison[f"SELISIH_{col}"] = comparison[col2].fillna(0) - comparison[col1].fillna(0)
        comparison[f"PCT_{col}"] = np.where(
            comparison[col1] != 0,
            (comparison[f"SELISIH_{col}"] / comparison[col1].abs()) * 100,
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
    else:
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
def render_date_selector_with_availability(key_prefix: str = "") -> Tuple[Optional[str], Optional[str], bool, bool, bool]:
    """
    Render date selection with availability indicator.
    """
    st.sidebar.markdown("### 📅 Pilih Tanggal Data")
    
    # Primary date picker
    primary_dt = st.sidebar.date_input(
        "Tanggal Data Utama",
        value=date(2025, 10, 27),
        min_value=DATE_START,
        max_value=DATE_END,
        key=f"{key_prefix}primary_date",
        help="Pilih tanggal update data"
    )
    primary_date = primary_dt.strftime("%Y-%m-%d")
    
    # Check availability
    with st.sidebar:
        with st.spinner("Memeriksa..."):
            primary_available, primary_url, primary_type = check_data_availability(primary_date)
        
        st.markdown(render_availability_badge(primary_available, primary_type), unsafe_allow_html=True)
        
        with st.expander("🔗 Debug Info", expanded=False):
            st.markdown("**URLs yang dicoba:**")
            for pattern in FILENAME_PATTERNS:
                url = build_url(primary_dt, pattern)
                st.code(url, language=None)
            if primary_available:
                st.success(f"✓ Ditemukan: {primary_type.upper()}")
                st.code(primary_url)
    
    st.sidebar.markdown("---")
    
    # Comparison toggle
    enable_comparison = st.sidebar.checkbox(
        "🔄 Bandingkan dengan tanggal lain",
        value=False,
        key=f"{key_prefix}enable_comparison"
    )
    
    comparison_date = None
    comparison_available = False
    
    if enable_comparison:
        comparison_dt = st.sidebar.date_input(
            "Tanggal Pembanding",
            value=date(2025, 11, 11),
            min_value=DATE_START,
            max_value=DATE_END,
            key=f"{key_prefix}comparison_date"
        )
        comparison_date = comparison_dt.strftime("%Y-%m-%d")
        
        with st.sidebar:
            with st.spinner("Memeriksa..."):
                comparison_available, comp_url, comp_type = check_data_availability(comparison_date)
            
            st.markdown(render_availability_badge(comparison_available, comp_type), unsafe_allow_html=True)
    
    # Quick reference
    st.sidebar.markdown("---")
    st.sidebar.markdown("**📋 Tanggal Tersedia:**")
    st.sidebar.markdown("- 27 Oktober 2025\n- 11 November 2025")
    
    return primary_date, comparison_date, enable_comparison, primary_available, comparison_available


def render_aggregation_options(key_prefix: str = "") -> Tuple[List[str], List[str], str]:
    """Render aggregation options sidebar component."""
    st.sidebar.markdown("### 📊 Opsi Agregasi")
    
    group_cols = st.sidebar.multiselect(
        "Group By (Kolom String)",
        options=STRING_COLUMNS,
        default=['KEMENTERIAN/LEMBAGA', 'PROGRAM'],
        help="Pilih minimal 2 kolom",
        key=f"{key_prefix}group_cols"
    )
    
    if len(group_cols) < 2:
        st.sidebar.warning("⚠️ Pilih minimal 2 kolom")
    
    numeric_cols = st.sidebar.multiselect(
        "Agregasi (Kolom Numerik)",
        options=NUMERIC_COLUMNS,
        default=['REALISASI BELANJA KL (SAKTI)', 'PAGU DIPA REVISI'],
        help="Pilih minimal 2 kolom",
        key=f"{key_prefix}numeric_cols"
    )
    
    if len(numeric_cols) < 2:
        st.sidebar.warning("⚠️ Pilih minimal 2 kolom")
    
    agg_func = st.sidebar.selectbox(
        "Fungsi Agregasi",
        options=['sum', 'mean', 'count', 'min', 'max'],
        format_func=lambda x: {'sum': 'Jumlah', 'mean': 'Rata-rata', 'count': 'Hitung', 'min': 'Min', 'max': 'Max'}.get(x, x),
        key=f"{key_prefix}agg_func"
    )
    
    return group_cols, numeric_cols, agg_func


# =============================================================================
# Debug Tool
# =============================================================================
def render_debug_tool():
    """Render a debug tool to test URLs directly."""
    st.markdown("---")
    st.subheader("🔧 Debug Tool")
    
    test_url = st.text_input(
        "Test URL langsung:",
        value="https://raw.githubusercontent.com/tubankum3/dashpmk/main/df_20251027.csv.zip",
        help="Masukkan URL file untuk di-test"
    )
    
    if st.button("🧪 Test URL"):
        with st.spinner("Testing..."):
            try:
                response = requests.get(test_url, timeout=30)
                
                st.write(f"**Status Code:** {response.status_code}")
                st.write(f"**Content-Type:** {response.headers.get('Content-Type', 'N/A')}")
                st.write(f"**Content-Length:** {len(response.content):,} bytes")
                
                content = response.content
                file_type = detect_file_type(content)
                st.write(f"**Detected Type:** {file_type}")
                
                st.write("**First 500 bytes (hex):**")
                st.code(content[:500].hex())
                
                st.write("**First 500 bytes (text):**")
                st.code(get_content_preview(content, 500))
                
                # Try to load
                if file_type == 'csv':
                    df = load_from_csv(content, "test")
                    if not df.empty:
                        st.success(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
                        st.write("**Columns:**", list(df.columns))
                        st.dataframe(df.head())
                        
                elif file_type == 'zip':
                    with zipfile.ZipFile(io.BytesIO(content)) as z:
                        st.write("**ZIP contents:**", z.namelist())
                    df = load_from_zip(content, "test")
                    if not df.empty:
                        st.success(f"✅ Loaded {len(df)} rows")
                        st.dataframe(df.head())
                        
                elif file_type == 'gzip':
                    df = load_from_gzip(content, "test")
                    if not df.empty:
                        st.success(f"✅ Loaded {len(df)} rows")
                        st.dataframe(df.head())
                
            except Exception as e:
                st.error(f"Error: {e}")


# =============================================================================
# Main Application
# =============================================================================
def main():
    """Main application."""
    
    st.set_page_config(
        page_title="Dashboard Anggaran - Debug",
        page_icon="🔧",
        layout="wide"
    )
    
    # Header
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #0066FF 0%, #0052CC 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    ">
        <p style="font-size: 0.875rem; opacity: 0.9; margin-bottom: 0.5rem;">DEBUG MODE</p>
        <h1 style="margin: 0; font-size: 1.75rem;">Dashboard Analisis Anggaran</h1>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Perbandingan Data dengan Enhanced Debugging</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Debug mode toggle
    debug_mode = st.checkbox("🔧 Enable Debug Mode", value=True)
    
    if debug_mode:
        render_debug_tool()
        st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Pengaturan")
        
        result = render_date_selector_with_availability()
        primary_date, comparison_date, enable_comparison, primary_available, comparison_available = result
        
        st.divider()
        group_cols, numeric_cols, agg_func = render_aggregation_options()
    
    # Main content
    if not primary_available:
        st.warning(f"⚠️ Data tidak tersedia untuk {format_date_label(parse_date_string(primary_date))}")
        st.info("Gunakan Debug Tool di atas untuk test URL secara langsung")
        return
    
    # Load primary data
    with st.spinner(f"Memuat data..."):
        df_primary = load_data_by_date(primary_date, debug=debug_mode)
    
    if df_primary.empty:
        return
    
    st.success(f"✅ Data dimuat: **{len(df_primary):,}** baris | {format_date_label(parse_date_string(primary_date))}")
    
    # Show columns for debugging
    if debug_mode:
        with st.expander("📋 Kolom tersedia di data"):
            st.write(list(df_primary.columns))
    
    # Rest of the app...
    if len(group_cols) < 2 or len(numeric_cols) < 2:
        st.info("👆 Pilih minimal 2 kolom string dan 2 kolom numerik")
        with st.expander("📋 Preview Data"):
            st.dataframe(df_primary.head(100), use_container_width=True)
        return
    
    agg_primary = aggregate_data(df_primary, group_cols, numeric_cols, agg_func)
    st.dataframe(agg_primary, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()

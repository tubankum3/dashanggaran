# =============================================================================
# Data Loading with Streaming Download for Large Files
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
# Configuration
# =============================================================================
BASE_URL = "https://raw.githubusercontent.com/tubankum3/dashanggaran/main/"
FILENAME_PATTERN = "df_{YYYYMMDD}.csv.zip"
CSV_FILENAME_IN_ZIP = "df.csv"

DATE_START = date(2024, 1, 1)
DATE_END = date.today()

# Timeout settings for large files
CONNECT_TIMEOUT = 10  # seconds to establish connection
READ_TIMEOUT = 300    # seconds to read data (5 minutes for large files)

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
    """
    Check if data file exists and get file size.
    
    Returns:
        Tuple of (is_available, file_size_bytes)
    """
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
    else:
        return f"{size_bytes} bytes"


def render_availability_badge(is_available: bool, file_size: Optional[int] = None) -> str:
    """Return HTML badge for availability status."""
    if is_available:
        size_text = f" • {format_file_size(file_size)}" if file_size else ""
        return f"""
        <span style="
            background-color: #10B981;
            color: white;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: 500;
        ">✓ Data Tersedia{size_text}</span>
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
# Streaming Download Functions
# =============================================================================
def download_with_progress(url: str, progress_placeholder) -> Optional[bytes]:
    """
    Download file with progress bar for large files.
    
    Args:
        url: URL to download
        progress_placeholder: Streamlit placeholder for progress bar
        
    Returns:
        File content as bytes, or None if failed
    """
    try:
        # Start streaming download
        response = requests.get(
            url, 
            stream=True, 
            timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
            allow_redirects=True
        )
        response.raise_for_status()
        
        # Get total size
        total_size = response.headers.get('Content-Length')
        total_size = int(total_size) if total_size else None
        
        # Download in chunks
        chunks = []
        downloaded = 0
        chunk_size = 1024 * 1024  # 1MB chunks
        
        progress_bar = progress_placeholder.progress(0, text="Downloading...")
        
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                chunks.append(chunk)
                downloaded += len(chunk)
                
                if total_size:
                    progress = min(downloaded / total_size, 1.0)
                    progress_bar.progress(
                        progress, 
                        text=f"Downloading... {format_file_size(downloaded)} / {format_file_size(total_size)}"
                    )
                else:
                    progress_bar.progress(0.5, text=f"Downloading... {format_file_size(downloaded)}")
        
        progress_bar.progress(1.0, text="Download complete!")
        
        content = b''.join(chunks)
        return content
        
    except requests.exceptions.Timeout:
        progress_placeholder.error("⏱️ Download timeout - file terlalu besar atau koneksi lambat")
        return None
    except requests.exceptions.RequestException as e:
        progress_placeholder.error(f"❌ Download error: {e}")
        return None
    except Exception as e:
        progress_placeholder.error(f"❌ Error: {e}")
        return None


# =============================================================================
# Data Loading Functions
# =============================================================================
def load_data_by_date_with_progress(data_date: str, progress_placeholder) -> pd.DataFrame:
    """
    Load budget data with progress indicator.
    
    Args:
        data_date: Date string in 'YYYY-MM-DD' format
        progress_placeholder: Streamlit placeholder for progress
        
    Returns:
        pd.DataFrame: Preprocessed budget data
    """
    dt = parse_date_string(data_date)
    url = build_url(dt)
    
    # Download with progress
    content = download_with_progress(url, progress_placeholder)
    
    if content is None:
        return pd.DataFrame()
    
    # Check if it's a valid ZIP
    if len(content) < 4:
        st.error("❌ Downloaded file is too small")
        return pd.DataFrame()
    
    # Debug: show first bytes
    first_bytes = content[:20]
    is_zip = content[:2] == b'PK'
    
    if not is_zip:
        st.error(f"❌ File bukan ZIP yang valid")
        with st.expander("🔍 Debug Info"):
            st.write(f"File size: {format_file_size(len(content))}")
            st.write(f"First 20 bytes (hex): {first_bytes.hex()}")
            st.write(f"First 20 bytes (text): {first_bytes.decode('utf-8', errors='replace')}")
            
            # Check if it's HTML error
            try:
                text_preview = content[:500].decode('utf-8', errors='ignore')
                if '<html' in text_preview.lower() or '404' in text_preview:
                    st.write("⚠️ Received HTML error page instead of file")
                st.code(text_preview)
            except:
                pass
        return pd.DataFrame()
    
    # Parse ZIP
    try:
        progress_placeholder.info("📦 Extracting ZIP...")
        
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            file_list = z.namelist()
            csv_files = [f for f in file_list if f.endswith('.csv')]
            
            if not csv_files:
                st.error(f"❌ No CSV files in ZIP. Contents: {file_list}")
                return pd.DataFrame()
            
            target_file = CSV_FILENAME_IN_ZIP if CSV_FILENAME_IN_ZIP in csv_files else csv_files[0]
            
            progress_placeholder.info(f"📄 Reading {target_file}...")
            
            with z.open(target_file) as file:
                df = pd.read_csv(file, low_memory=False)
        
        progress_placeholder.empty()
        
        df = preprocess_dataframe(df)
        df["_DATA_DATE"] = data_date
        
        return df
        
    except zipfile.BadZipFile as e:
        st.error(f"❌ Invalid ZIP file: {e}")
        with st.expander("🔍 Debug Info"):
            st.write(f"File size: {format_file_size(len(content))}")
            st.write(f"First bytes: {content[:50].hex()}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Error reading ZIP: {e}")
        return pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=3600)
def load_data_by_date_cached(data_date: str) -> pd.DataFrame:
    """
    Cached version - downloads without progress (for subsequent loads).
    """
    dt = parse_date_string(data_date)
    url = build_url(dt)
    
    try:
        response = requests.get(
            url, 
            timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
            allow_redirects=True
        )
        response.raise_for_status()
        content = response.content
        
        if content[:2] != b'PK':
            return pd.DataFrame()
        
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            csv_files = [f for f in z.namelist() if f.endswith('.csv')]
            if not csv_files:
                return pd.DataFrame()
            
            target_file = CSV_FILENAME_IN_ZIP if CSV_FILENAME_IN_ZIP in csv_files else csv_files[0]
            
            with z.open(target_file) as file:
                df = pd.read_csv(file, low_memory=False)
        
        df = preprocess_dataframe(df)
        df["_DATA_DATE"] = data_date
        return df
        
    except:
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
def render_date_selector_with_availability(key_prefix: str = "") -> Tuple[Optional[str], Optional[str], bool, bool, bool, Optional[int]]:
    """
    Render date selection with availability indicator.
    
    Returns:
        Tuple of (primary_date, comparison_date, enable_comparison, primary_available, comparison_available, primary_size)
    """
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
# Main Application
# =============================================================================
def main():
    """Main application."""
    
    st.set_page_config(
        page_title="Dashboard Anggaran",
        page_icon="📊",
        layout="wide"
    )
    
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #0066FF 0%, #0052CC 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    ">
        <p style="font-size: 0.875rem; opacity: 0.9; margin-bottom: 0.5rem;">PROTOTIPE</p>
        <h1 style="margin: 0; font-size: 1.75rem;">Dashboard Analisis Anggaran</h1>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Perbandingan Data Berdasarkan Tanggal Update</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("## ⚙️ Pengaturan")
        result = render_date_selector_with_availability()
        primary_date, comparison_date, enable_comparison, primary_available, comparison_available, primary_size = result
        st.divider()
        group_cols, numeric_cols, agg_func = render_aggregation_options()
    
    if not primary_available:
        st.warning(f"⚠️ Data tidak tersedia untuk {format_date_label(parse_date_string(primary_date))}")
        return
    
    # Show file size warning for large files
    if primary_size and primary_size > 50_000_000:  # > 50MB
        st.info(f"📦 File besar ({format_file_size(primary_size)}). Download mungkin memakan waktu...")
    
    # Progress placeholder
    progress_placeholder = st.empty()
    
    # Load data with progress
    df_primary = load_data_by_date_with_progress(primary_date, progress_placeholder)
    
    if df_primary.empty:
        return
    
    st.success(f"✅ Data dimuat: **{len(df_primary):,}** baris | {format_date_label(parse_date_string(primary_date))}")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📊 Total Baris", f"{len(df_primary):,}")
    col2.metric("📋 Total Kolom", len(df_primary.columns))
    col3.metric("📅 Tanggal Data", primary_date)
    if "Tahun" in df_primary.columns:
        tahun_list = df_primary["Tahun"].dropna().unique()
        col4.metric("📆 Tahun Anggaran", ", ".join(map(str, sorted(tahun_list))))
    
    st.divider()
    
    if len(group_cols) < 2 or len(numeric_cols) < 2:
        st.info("👆 Pilih minimal 2 kolom string dan 2 kolom numerik")
        with st.expander("📋 Preview Data"):
            st.dataframe(df_primary.head(100), use_container_width=True)
        return
    
    agg_primary = aggregate_data(df_primary, group_cols, numeric_cols, agg_func)
    
    if enable_comparison and comparison_date and comparison_available:
        progress_placeholder2 = st.empty()
        df_comparison = load_data_by_date_with_progress(comparison_date, progress_placeholder2)
        
        if not df_comparison.empty:
            st.success(f"✅ Data pembanding dimuat: **{len(df_comparison):,}** baris")
            
            comparison_df = compare_datasets(
                df_primary, df_comparison,
                group_cols, numeric_cols,
                primary_date, comparison_date
            )
            
            tab1, tab2, tab3 = st.tabs([
                f"📊 Data {format_date_label(parse_date_string(primary_date))}",
                "🔄 Perbandingan",
                "📈 Ringkasan"
            ])
            
            with tab1:
                st.dataframe(agg_primary, use_container_width=True, hide_index=True)
                csv1 = agg_primary.to_csv(index=False)
                st.download_button("📥 Download CSV", csv1, f"agregasi_{primary_date}.csv", "text/csv", key="dl1")
            
            with tab2:
                st.info(f"""
                **Kolom:** `[kolom]_{primary_date}` = Utama | `[kolom]_{comparison_date}` = Pembanding | `SELISIH_` = Selisih | `PCT_` = %
                """)
                st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                csv2 = comparison_df.to_csv(index=False)
                st.download_button("📥 Download CSV", csv2, f"perbandingan_{primary_date}_vs_{comparison_date}.csv", "text/csv", key="dl2")
            
            with tab3:
                for col in numeric_cols:
                    if col in df_primary.columns and col in df_comparison.columns:
                        st.markdown(f"#### {col}")
                        c1, c2, c3 = st.columns(3)
                        val1, val2 = df_primary[col].sum(), df_comparison[col].sum()
                        diff = val2 - val1
                        pct = (diff / abs(val1) * 100) if val1 != 0 else 0
                        c1.metric(primary_date, format_rupiah(val1))
                        c2.metric(comparison_date, format_rupiah(val2))
                        c3.metric("Selisih", format_rupiah(diff), f"{pct:+.2f}%")
                        st.divider()
        else:
            st.dataframe(agg_primary, use_container_width=True, hide_index=True)
    else:
        st.dataframe(agg_primary, use_container_width=True, hide_index=True)
        csv = agg_primary.to_csv(index=False)
        st.download_button("📥 Download CSV", csv, f"agregasi_{primary_date}.csv", "text/csv")


if __name__ == "__main__":
    main()

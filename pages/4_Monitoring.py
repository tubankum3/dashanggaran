# =============================================================================
# Data Loading with BASE_URL_PATTERN Approach
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
# Configuration - BASE URL PATTERN
# =============================================================================

BASE_URL = "https://raw.githubusercontent.com/tubankum3/dashpmk/main/"
FILENAME_PATTERN = "df_{YYYYMMDD}.csv.zip"

# CSV filename
CSV_FILENAME_IN_ZIP = "df.csv"  # or None to auto-detect

# Date range configuration
DATE_START = date(2024, 1, 1)    # Earliest available date
DATE_END = date.today()          # Latest available date

# Date selection mode: 'picker', 'list', or 'both'
DATE_SELECTION_MODE = 'both'

# If using 'list' mode, define available dates here
AVAILABLE_DATES = [
    "2024-12-01",
    "2024-11-01", 
    "2024-10-01",
    "2024-09-01",
    # Add more dates as needed
]

# =============================================================================
# URL Builder Functions
# =============================================================================
def build_url(data_date: date) -> str:
    """
    Build the full URL for a given date using the pattern.
    
    Args:
        data_date: Date object
        
    Returns:
        Full URL string
    """
    # Create all possible placeholders
    placeholders = {
        '{YYYY}': data_date.strftime('%Y'),
        '{MM}': data_date.strftime('%m'),
        '{DD}': data_date.strftime('%d'),
        '{YYYYMMDD}': data_date.strftime('%Y%m%d'),
        '{YYYY-MM-DD}': data_date.strftime('%Y-%m-%d'),
        '{YY}': data_date.strftime('%y'),
        '{M}': str(data_date.month),  # Without leading zero
        '{D}': str(data_date.day),    # Without leading zero
    }
    
    filename = FILENAME_PATTERN
    for placeholder, value in placeholders.items():
        filename = filename.replace(placeholder, value)
    
    return BASE_URL + filename


def parse_date_string(date_str: str) -> date:
    """Parse date string to date object."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        try:
            return datetime.strptime(date_str, "%Y%m%d").date()
        except ValueError:
            raise ValueError(f"Cannot parse date: {date_str}")


def format_date_label(data_date: date) -> str:
    """Format date as Indonesian label."""
    months_id = {
        1: 'Januari', 2: 'Februari', 3: 'Maret', 4: 'April',
        5: 'Mei', 6: 'Juni', 7: 'Juli', 8: 'Agustus',
        9: 'September', 10: 'Oktober', 11: 'November', 12: 'Desember'
    }
    return f"{data_date.day} {months_id[data_date.month]} {data_date.year}"


# =============================================================================
# Data Loading Functions
# =============================================================================
@st.cache_data(show_spinner=True, ttl=3600)
def load_data_by_date(data_date: str) -> pd.DataFrame:
    """
    Load budget data for a specific date.
    
    Args:
        data_date: Date string in 'YYYY-MM-DD' format
        
    Returns:
        pd.DataFrame: Preprocessed budget data with _DATA_DATE column
    """
    # Parse date
    if isinstance(data_date, str):
        dt = parse_date_string(data_date)
    else:
        dt = data_date
    
    # Build URL
    url = build_url(dt)
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            # Find CSV file in zip
            csv_files = [f for f in z.namelist() if f.endswith('.csv')]
            
            if not csv_files:
                st.error(f"❌ Tidak ada file CSV dalam zip untuk tanggal {data_date}")
                return pd.DataFrame()
            
            # Use configured filename or first CSV found
            if CSV_FILENAME_IN_ZIP and CSV_FILENAME_IN_ZIP in csv_files:
                target_file = CSV_FILENAME_IN_ZIP
            else:
                target_file = csv_files[0]
            
            with z.open(target_file) as file:
                df = pd.read_csv(file, low_memory=False)
        
        # Preprocessing
        df = preprocess_dataframe(df)
        
        # Add data date column
        df["_DATA_DATE"] = data_date if isinstance(data_date, str) else data_date.strftime("%Y-%m-%d")
        
        return df
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            st.error(f"❌ Data tidak tersedia untuk tanggal {format_date_label(dt)}")
            st.info(f"URL: {url}")
        else:
            st.error(f"❌ HTTP Error: {e}")
        return pd.DataFrame()
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Gagal mengunduh data: {str(e)}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Gagal memuat data: {str(e)}")
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def check_url_exists(url: str) -> bool:
    """Check if a URL exists (returns 200)."""
    try:
        response = requests.head(url, timeout=5)
        return response.status_code == 200
    except:
        return False


def get_available_dates_from_range(
    start_date: date = DATE_START,
    end_date: date = DATE_END,
    check_existence: bool = False
) -> List[str]:
    """
    Generate list of dates in range, optionally checking if files exist.
    
    Args:
        start_date: Start of range
        end_date: End of range
        check_existence: If True, verify each URL exists (slower)
        
    Returns:
        List of date strings (YYYY-MM-DD format)
    """
    dates = []
    current = end_date
    
    while current >= start_date:
        date_str = current.strftime("%Y-%m-%d")
        
        if check_existence:
            url = build_url(current)
            if check_url_exists(url):
                dates.append(date_str)
        else:
            dates.append(date_str)
        
        # Move to previous month (first day)
        current = current.replace(day=1) - timedelta(days=1)
        current = current.replace(day=1)
    
    return dates


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the loaded dataframe."""
    if df.empty:
        return df
    
    # Remove unnamed columns
    unnamed_cols = [c for c in df.columns if 'Unnamed' in str(c)]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)
    
    # Convert Tahun to integer
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
    
    # Aggregate
    agg1 = aggregate_data(df1, available_group, available_numeric)
    agg2 = aggregate_data(df2, available_group, available_numeric)
    
    # Rename for merge
    agg1 = agg1.rename(columns={c: f"{c}_{date1}" for c in available_numeric})
    agg2 = agg2.rename(columns={c: f"{c}_{date2}" for c in available_numeric})
    
    # Merge
    comparison = pd.merge(agg1, agg2, on=available_group, how='outer')
    
    # Calculate differences
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
def render_date_selector(key_prefix: str = "") -> Tuple[Optional[str], Optional[str], bool]:
    """
    Render date selection sidebar component.
    
    Returns:
        Tuple of (primary_date, comparison_date, enable_comparison)
    """
    st.sidebar.markdown("### 📅 Pilih Tanggal Data")
    
    if DATE_SELECTION_MODE == 'picker':
        # Date picker mode
        primary_dt = st.sidebar.date_input(
            "Tanggal Data Utama",
            value=DATE_END,
            min_value=DATE_START,
            max_value=DATE_END,
            key=f"{key_prefix}primary_date"
        )
        primary_date = primary_dt.strftime("%Y-%m-%d")
        
    elif DATE_SELECTION_MODE == 'list':
        # List selection mode
        primary_date = st.sidebar.selectbox(
            "Tanggal Data Utama",
            options=AVAILABLE_DATES,
            format_func=lambda x: format_date_label(parse_date_string(x)),
            key=f"{key_prefix}primary_date"
        )
        
    else:  # 'both'
        # Combined mode
        use_picker = st.sidebar.checkbox("Gunakan date picker", value=True, key=f"{key_prefix}use_picker")
        
        if use_picker:
            primary_dt = st.sidebar.date_input(
                "Tanggal Data Utama",
                value=DATE_END,
                min_value=DATE_START,
                max_value=DATE_END,
                key=f"{key_prefix}primary_date"
            )
            primary_date = primary_dt.strftime("%Y-%m-%d")
        else:
            primary_date = st.sidebar.selectbox(
                "Tanggal Data Utama",
                options=AVAILABLE_DATES,
                format_func=lambda x: format_date_label(parse_date_string(x)),
                key=f"{key_prefix}primary_date_list"
            )
    
    # Show generated URL for debugging
    with st.sidebar.expander("🔗 URL Info", expanded=False):
        url = build_url(parse_date_string(primary_date))
        st.code(url, language=None)
    
    # Comparison toggle
    st.sidebar.markdown("---")
    enable_comparison = st.sidebar.checkbox(
        "🔄 Bandingkan dengan tanggal lain",
        value=False,
        key=f"{key_prefix}enable_comparison"
    )
    
    comparison_date = None
    if enable_comparison:
        if DATE_SELECTION_MODE == 'picker':
            comparison_dt = st.sidebar.date_input(
                "Tanggal Pembanding",
                value=DATE_END - timedelta(days=30),
                min_value=DATE_START,
                max_value=DATE_END,
                key=f"{key_prefix}comparison_date"
            )
            comparison_date = comparison_dt.strftime("%Y-%m-%d")
        else:
            available_for_compare = [d for d in AVAILABLE_DATES if d != primary_date]
            if available_for_compare:
                comparison_date = st.sidebar.selectbox(
                    "Tanggal Pembanding",
                    options=available_for_compare,
                    format_func=lambda x: format_date_label(parse_date_string(x)),
                    key=f"{key_prefix}comparison_date"
                )
        
        if comparison_date:
            with st.sidebar.expander("🔗 URL Pembanding", expanded=False):
                url2 = build_url(parse_date_string(comparison_date))
                st.code(url2, language=None)
    
    return primary_date, comparison_date, enable_comparison


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
        <p class="breadcrumb">KEMENTERIAN KEUANGAN RI</p>
        <h1 class="dashboard-title">Dashboard Analisis Anggaran</h1>
        <p class="dashboard-subtitle">Perbandingan Data Berdasarkan Tanggal Update</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Pengaturan")
        primary_date, comparison_date, enable_comparison = render_date_selector()
        st.divider()
        group_cols, numeric_cols, agg_func = render_aggregation_options()
    
    # Main content
    if primary_date:
        # Load primary data
        with st.spinner(f"Memuat data {format_date_label(parse_date_string(primary_date))}..."):
            df_primary = load_data_by_date(primary_date)
        
        if df_primary.empty:
            st.warning("⚠️ Data tidak tersedia. Silakan pilih tanggal lain.")
            return
        
        # Success message
        st.success(f"✅ Data dimuat: **{len(df_primary):,}** baris | {format_date_label(parse_date_string(primary_date))}")
        
        # Info metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📊 Total Baris", f"{len(df_primary):,}")
        col2.metric("📋 Total Kolom", len(df_primary.columns))
        col3.metric("📅 Tanggal Data", primary_date)
        if "Tahun" in df_primary.columns:
            tahun_list = df_primary["Tahun"].dropna().unique()
            col4.metric("📆 Tahun Anggaran", ", ".join(map(str, sorted(tahun_list))))
        
        st.divider()
        
        # Check column selection
        if len(group_cols) < 2 or len(numeric_cols) < 2:
            st.info("👆 Pilih minimal 2 kolom string dan 2 kolom numerik di sidebar untuk melihat agregasi")
            return
        
        # Aggregate primary data
        agg_primary = aggregate_data(df_primary, group_cols, numeric_cols, agg_func)
        
        if enable_comparison and comparison_date:
            # Load comparison data
            with st.spinner(f"Memuat data pembanding {format_date_label(parse_date_string(comparison_date))}..."):
                df_comparison = load_data_by_date(comparison_date)
            
            if df_comparison.empty:
                st.warning("⚠️ Data pembanding tidak tersedia")
                st.dataframe(agg_primary, use_container_width=True, hide_index=True)
                return
            
            st.success(f"✅ Data pembanding dimuat: **{len(df_comparison):,}** baris")
            
            # Create comparison
            comparison_df = compare_datasets(
                df_primary, df_comparison,
                group_cols, numeric_cols,
                primary_date, comparison_date
            )
            
            # Tabs for different views
            tab1, tab2, tab3 = st.tabs([
                f"📊 Data {primary_date}",
                "🔄 Perbandingan",
                "📈 Ringkasan"
            ])
            
            with tab1:
                st.subheader(f"Agregasi - {format_date_label(parse_date_string(primary_date))}")
                st.dataframe(agg_primary, use_container_width=True, hide_index=True)
            
            with tab2:
                st.subheader("Perbandingan Data")
                st.caption(f"{format_date_label(parse_date_string(primary_date))} vs {format_date_label(parse_date_string(comparison_date))}")
                st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                
                csv = comparison_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Perbandingan (CSV)",
                    data=csv,
                    file_name=f"perbandingan_{primary_date}_vs_{comparison_date}.csv",
                    mime="text/csv"
                )
            
            with tab3:
                st.subheader("Ringkasan Perbandingan")
                
                for col in numeric_cols:
                    if col in df_primary.columns and col in df_comparison.columns:
                        st.markdown(f"**{col}**")
                        
                        c1, c2, c3 = st.columns(3)
                        
                        val1 = df_primary[col].sum()
                        val2 = df_comparison[col].sum()
                        diff = val2 - val1
                        pct = (diff / abs(val1) * 100) if val1 != 0 else 0
                        
                        c1.metric(
                            format_date_label(parse_date_string(primary_date)),
                            format_rupiah(val1)
                        )
                        c2.metric(
                            format_date_label(parse_date_string(comparison_date)),
                            format_rupiah(val2)
                        )
                        c3.metric(
                            "Selisih",
                            format_rupiah(diff),
                            f"{pct:+.2f}%"
                        )
                        st.divider()
        
        else:
            # Single date view
            st.subheader(f"Data Agregasi - {format_date_label(parse_date_string(primary_date))}")
            st.dataframe(agg_primary, use_container_width=True, hide_index=True)
            
            csv = agg_primary.to_csv(index=False)
            st.download_button(
                "📥 Download Data (CSV)",
                data=csv,
                file_name=f"agregasi_{primary_date}.csv",
                mime="text/csv"
            )


if __name__ == "__main__":
    main()

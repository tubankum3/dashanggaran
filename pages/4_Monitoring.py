# =============================================================================
# Data Loading with Date Picker + Availability Indicator
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

# Date range for picker
DATE_START = date(2024, 1, 1)
DATE_END = date.today()

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
# Availability Check Functions
# =============================================================================
@st.cache_data(ttl=300, show_spinner=False)  # Cache for 5 minutes
def check_data_availability(data_date: str) -> bool:
    """
    Check if data file exists for a given date.
    
    Args:
        data_date: Date string in 'YYYY-MM-DD' format
        
    Returns:
        bool: True if file exists, False otherwise
    """
    dt = parse_date_string(data_date)
    url = build_url(dt)
    
    try:
        response = requests.head(url, timeout=5)
        return response.status_code == 200
    except:
        return False


def render_availability_badge(is_available: bool) -> str:
    """Return HTML badge for availability status."""
    if is_available:
        return """
        <span style="
            background-color: #10B981;
            color: white;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: 500;
        ">✓ Data Tersedia</span>
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
def load_data_by_date(data_date: str) -> pd.DataFrame:
    """
    Load budget data for a specific date.
    
    Args:
        data_date: Date string in 'YYYY-MM-DD' format
        
    Returns:
        pd.DataFrame: Preprocessed budget data with _DATA_DATE column
    """
    dt = parse_date_string(data_date)
    url = build_url(dt)
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            csv_files = [f for f in z.namelist() if f.endswith('.csv')]
            
            if not csv_files:
                st.error(f"❌ Tidak ada file CSV dalam zip untuk tanggal {data_date}")
                return pd.DataFrame()
            
            target_file = CSV_FILENAME_IN_ZIP if CSV_FILENAME_IN_ZIP in csv_files else csv_files[0]
            
            with z.open(target_file) as file:
                df = pd.read_csv(file, low_memory=False)
        
        df = preprocess_dataframe(df)
        df["_DATA_DATE"] = data_date
        
        return df
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            st.error(f"❌ Data tidak tersedia untuk tanggal {format_date_label(dt)}")
        else:
            st.error(f"❌ HTTP Error: {e}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Gagal memuat data: {str(e)}")
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
    
    Returns:
        Tuple of (primary_date, comparison_date, enable_comparison, primary_available, comparison_available)
    """
    st.sidebar.markdown("### 📅 Pilih Tanggal Data")
    
    # Primary date picker
    primary_dt = st.sidebar.date_input(
        "Tanggal Data Utama",
        value=date(2025, 10, 27),  # Default to known available date
        min_value=DATE_START,
        max_value=DATE_END,
        key=f"{key_prefix}primary_date",
        help="Pilih tanggal update data"
    )
    primary_date = primary_dt.strftime("%Y-%m-%d")
    
    # Check availability
    with st.sidebar:
        with st.spinner("Memeriksa ketersediaan..."):
            primary_available = check_data_availability(primary_date)
        
        st.markdown(render_availability_badge(primary_available), unsafe_allow_html=True)
        
        # Show URL info
        with st.expander("🔗 Info URL", expanded=False):
            url = build_url(primary_dt)
            st.code(url, language=None)
            if primary_available:
                st.success("File ditemukan")
            else:
                st.error("File tidak ditemukan")
    
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
            value=date(2025, 11, 11),  # Default to other known available date
            min_value=DATE_START,
            max_value=DATE_END,
            key=f"{key_prefix}comparison_date",
            help="Pilih tanggal untuk dibandingkan"
        )
        comparison_date = comparison_dt.strftime("%Y-%m-%d")
        
        # Check availability
        with st.sidebar:
            with st.spinner("Memeriksa ketersediaan..."):
                comparison_available = check_data_availability(comparison_date)
            
            st.markdown(render_availability_badge(comparison_available), unsafe_allow_html=True)
            
            with st.expander("🔗 Info URL Pembanding", expanded=False):
                url2 = build_url(comparison_dt)
                st.code(url2, language=None)
    
    # Quick select known dates
    st.sidebar.markdown("---")
    st.sidebar.markdown("**📋 Tanggal Tersedia (Prototipe):**")
    st.sidebar.markdown("""
    - 27 Oktober 2025
    - 11 November 2025
    """)
    
    return primary_date, comparison_date, enable_comparison, primary_available, comparison_available


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
    
    st.set_page_config(
        page_title="Dashboard Anggaran - Prototipe",
        page_icon="📊",
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
        <p style="font-size: 0.875rem; opacity: 0.9; margin-bottom: 0.5rem;">PROTOTIPE</p>
        <h1 style="margin: 0; font-size: 1.75rem;">Dashboard Analisis Anggaran</h1>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Perbandingan Data Berdasarkan Tanggal Update</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Pengaturan")
        
        result = render_date_selector_with_availability()
        primary_date, comparison_date, enable_comparison, primary_available, comparison_available = result
        
        st.divider()
        group_cols, numeric_cols, agg_func = render_aggregation_options()
    
    # Main content
    if not primary_available:
        st.warning(f"""
        ⚠️ **Data tidak tersedia untuk tanggal {format_date_label(parse_date_string(primary_date))}**
        
        Silakan pilih tanggal lain. Data yang tersedia:
        - **27 Oktober 2025** (`df_20251027.csv.zip`)
        - **11 November 2025** (`df_20251111.csv.zip`)
        """)
        return
    
    # Load primary data
    with st.spinner(f"Memuat data {format_date_label(parse_date_string(primary_date))}..."):
        df_primary = load_data_by_date(primary_date)
    
    if df_primary.empty:
        st.error("Gagal memuat data")
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
        
        # Show raw data preview
        with st.expander("📋 Preview Data Mentah"):
            st.dataframe(df_primary.head(100), use_container_width=True)
        return
    
    # Aggregate primary data
    agg_primary = aggregate_data(df_primary, group_cols, numeric_cols, agg_func)
    
    # Comparison mode
    if enable_comparison and comparison_date:
        if not comparison_available:
            st.warning(f"⚠️ Data pembanding tidak tersedia untuk tanggal {format_date_label(parse_date_string(comparison_date))}")
            st.dataframe(agg_primary, use_container_width=True, hide_index=True)
            return
        
        # Load comparison data
        with st.spinner(f"Memuat data pembanding {format_date_label(parse_date_string(comparison_date))}..."):
            df_comparison = load_data_by_date(comparison_date)
        
        if df_comparison.empty:
            st.warning("⚠️ Gagal memuat data pembanding")
            st.dataframe(agg_primary, use_container_width=True, hide_index=True)
            return
        
        st.success(f"✅ Data pembanding dimuat: **{len(df_comparison):,}** baris")
        
        # Create comparison
        comparison_df = compare_datasets(
            df_primary, df_comparison,
            group_cols, numeric_cols,
            primary_date, comparison_date
        )
        
        # Tabs
        tab1, tab2, tab3 = st.tabs([
            f"📊 Data {format_date_label(parse_date_string(primary_date))}",
            "🔄 Perbandingan",
            "📈 Ringkasan"
        ])
        
        with tab1:
            st.subheader(f"Agregasi - {format_date_label(parse_date_string(primary_date))}")
            st.dataframe(agg_primary, use_container_width=True, hide_index=True)
            
            csv1 = agg_primary.to_csv(index=False)
            st.download_button(
                "📥 Download CSV",
                data=csv1,
                file_name=f"agregasi_{primary_date}.csv",
                mime="text/csv",
                key="download_primary"
            )
        
        with tab2:
            st.subheader("Perbandingan Data")
            st.caption(f"**{format_date_label(parse_date_string(primary_date))}** vs **{format_date_label(parse_date_string(comparison_date))}**")
            
            # Highlight columns info
            st.info(f"""
            **Kolom hasil perbandingan:**
            - `[kolom]_{primary_date}` = Nilai dari tanggal utama
            - `[kolom]_{comparison_date}` = Nilai dari tanggal pembanding  
            - `SELISIH_[kolom]` = Selisih (pembanding - utama)
            - `PCT_[kolom]` = Persentase perubahan
            """)
            
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
            
            csv2 = comparison_df.to_csv(index=False)
            st.download_button(
                "📥 Download Perbandingan (CSV)",
                data=csv2,
                file_name=f"perbandingan_{primary_date}_vs_{comparison_date}.csv",
                mime="text/csv",
                key="download_comparison"
            )
        
        with tab3:
            st.subheader("Ringkasan Perbandingan")
            
            for col in numeric_cols:
                if col in df_primary.columns and col in df_comparison.columns:
                    st.markdown(f"#### {col}")
                    
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

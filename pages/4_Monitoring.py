"""
Monitoring Anggaran
=================================================
Dashboard untuk mengkomparasi data anggaran pemerintah antar tanggal update database yang berbeda.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st

# =============================================================================
# CONSTANTS & CONFIGURATION
# =============================================================================

class AggregationFunction(Enum):
    """Supported aggregation functions for data analysis."""
    SUM = "sum"
    MEAN = "mean"
    COUNT = "count"
    MIN = "min"
    MAX = "max"
    
    @property
    def display_name(self) -> str:
        names = {
            "sum": "Jumlah (Sum)",
            "mean": "Rata-rata (Mean)",
            "count": "Hitung (Count)",
            "min": "Minimum",
            "max": "Maksimum"
        }
        return names.get(self.value, self.value)


@dataclass(frozen=True)
class AppConfig:
    """Application configuration settings."""
    base_url: str = "https://raw.githubusercontent.com/tubankum3/dashanggaran/main/"
    filename_pattern: str = "df_{date}.csv.zip"
    csv_filename_in_zip: str = "df.csv"
    date_start: date = field(default_factory=lambda: date(2024, 1, 1))
    date_end: date = field(default_factory=date.today)
    connect_timeout: int = 10
    read_timeout: int = 300
    chunk_size: int = 1024 * 1024
    cache_ttl: int = 300
    large_file_threshold: int = 50_000_000
    max_filters_per_row: int = 4
    metrics_per_row: int = 3


@dataclass
class ColumnConfig:
    """Column definitions for the budget dataset."""
    
    # Potential columns - actual columns determined from data
    POTENTIAL_STRING_COLUMNS: Tuple[str, ...] = (
        'KEMENTERIAN/LEMBAGA', 'SUMBER DANA', 'FUNGSI', 'SUB FUNGSI',
        'PROGRAM', 'KEGIATAN', 'OUTPUT (KRO)', 'SUB OUTPUT (RO)',
        'KOMPONEN', 'JENIS BELANJA', 'AKUN 4 DIGIT', 'Tahun'
    )
    POTENTIAL_NUMERIC_COLUMNS: Tuple[str, ...] = (
        'REALISASI BELANJA KL (SAKTI)', 'PAGU DIPA REVISI', 'BLOKIR DIPA REVISI',
        'PAGU DIPA AWAL', 'BLOKIR DIPA AWAL', 'PAGU DIPA AWAL EFEKTIF',
        'PAGU DIPA REVISI EFEKTIF'
    )
    
    # Defaults
    DEFAULT_GROUP_COLS: Tuple[str, ...] = ('KEMENTERIAN/LEMBAGA','FUNGSI','PROGRAM')
    DEFAULT_NUMERIC_COLS: Tuple[str, ...] = ('PAGU DIPA REVISI',)
    
    # Actual available columns (set after loading data)
    string_columns: List[str] = field(default_factory=list)
    numeric_columns: List[str] = field(default_factory=list)
    available_years: List[int] = field(default_factory=list)
    
    def update_from_dataframe(self, df: pd.DataFrame) -> None:
        """Update available columns based on actual DataFrame columns."""
        if df.empty:
            return
        
        # Find string columns that exist in the DataFrame
        self.string_columns = [
            col for col in self.POTENTIAL_STRING_COLUMNS 
            if col in df.columns
        ]
        
        # Find numeric columns that exist in the DataFrame
        self.numeric_columns = [
            col for col in self.POTENTIAL_NUMERIC_COLUMNS 
            if col in df.columns
        ]
        
        # Extract available years
        if "Tahun" in df.columns:
            self.available_years = sorted(df["Tahun"].dropna().unique().tolist())
    
    def get_available_group_defaults(self) -> List[str]:
        """Get default group columns that are available in data."""
        return [col for col in self.DEFAULT_GROUP_COLS if col in self.string_columns]
    
    def get_available_numeric_defaults(self) -> List[str]:
        """Get default numeric columns that are available in data."""
        return [col for col in self.DEFAULT_NUMERIC_COLS if col in self.numeric_columns]


@dataclass
class DataAvailability:
    """Represents the availability status of a data file."""
    is_available: bool
    file_size: Optional[int] = None
    error_message: Optional[str] = None


# Indonesian month names
INDONESIAN_MONTHS: Dict[int, str] = {
    1: 'Januari', 2: 'Februari', 3: 'Maret', 4: 'April',
    5: 'Mei', 6: 'Juni', 7: 'Juli', 8: 'Agustus',
    9: 'September', 10: 'Oktober', 11: 'November', 12: 'Desember'
}

AVAILABLE_DATES: List[str] = ["27 Oktober 2025", "11 November 2025"]

# =============================================================================
# STYLING
# =============================================================================

CSS_STYLES = """
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
    --on-primary: #FFFFFF;
    --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    --radius-md: 8px;
    --radius-lg: 12px;
    --radius-full: 9999px;
    --space-md: 1rem;
    --space-lg: 1.5rem;
    --space-xl: 2rem;
    --transition-fast: 150ms cubic-bezier(0.4, 0, 0.2, 1);
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
    margin-bottom: 0.75rem;
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
    margin: 0.75rem 0 0 0;
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

.stTabs [data-baseweb="tab-list"] { gap: 0.75rem; border-bottom: 1px solid var(--gray-200); }
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
"""


# =============================================================================
# UTILITY CLASSES
# =============================================================================

class Formatter:
    """Utility class for formatting values."""
    
    @staticmethod
    def to_rupiah_short(value: float) -> str:
        """Format value as Indonesian Rupiah with abbreviated units."""
        if pd.isna(value) or value == 0:
            return "Rp 0"
        
        abs_value = abs(value)
        sign = "-" if value < 0 else ""
        
        if abs_value >= 1_000_000_000_000:
            return f"{sign}Rp {abs_value / 1_000_000_000_000:.2f} T"
        elif abs_value >= 1_000_000_000:
            return f"{sign}Rp {abs_value / 1_000_000_000:.2f} M"
        elif abs_value >= 1_000_000:
            return f"{sign}Rp {abs_value / 1_000_000:.2f} Jt"
        return f"{sign}Rp {abs_value:,.0f}"
    
    @staticmethod
    def to_rupiah_full(value: Any) -> str:
        """Format value as Indonesian Rupiah with full number."""
        try:
            value = float(value)
        except (ValueError, TypeError):
            return str(value)
        
        if pd.isna(value):
            return "-"
        
        return f"Rp {value:,.0f}".replace(",", ".")
    
    @staticmethod
    def to_percentage(value: Any) -> str:
        """Format value as percentage."""
        try:
            value = float(value)
        except (ValueError, TypeError):
            return str(value)
        
        if pd.isna(value):
            return "-"
        
        return f"{value:,.2f}%"
    
    @staticmethod
    def to_file_size(size_bytes: Optional[int]) -> str:
        """Format byte count as human-readable file size."""
        if size_bytes is None:
            return "Unknown"
        
        units = [(1_000_000_000, "GB"), (1_000_000, "MB"), (1_000, "KB")]
        
        for threshold, unit in units:
            if size_bytes >= threshold:
                return f"{size_bytes / threshold:.2f} {unit}"
        
        return f"{size_bytes} bytes"
    
    @staticmethod
    def to_indonesian_date(dt: date) -> str:
        """Format date using Indonesian month names."""
        month_name = INDONESIAN_MONTHS.get(dt.month, str(dt.month))
        return f"{dt.day} {month_name} {dt.year}"
    
    @staticmethod
    def to_number_with_separator(value: Any) -> str:
        """Format number with thousand separator."""
        try:
            return f"{int(value):,}"
        except (ValueError, TypeError):
            return str(value)


class DataFrameFormatter:
    """Formats DataFrames for display."""
    
    def __init__(self, numeric_columns: List[str]):
        self.numeric_columns = numeric_columns
    
    def format_for_display(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply formatting to DataFrame for UI display."""
        if df.empty:
            return df
        
        df_display = df.copy()
        
        for col in df_display.columns:
            if self._is_percentage_column(col):
                df_display[col] = df_display[col].apply(Formatter.to_percentage)
            elif self._is_numeric_column(col):
                df_display[col] = df_display[col].apply(Formatter.to_rupiah_full)
        
        return df_display
    
    def _is_numeric_column(self, column_name: str) -> bool:
        return (
            any(nc in column_name for nc in self.numeric_columns) or
            column_name.startswith('SELISIH_')
        )
    
    def _is_percentage_column(self, column_name: str) -> bool:
        return column_name.startswith('%CHG_')


class ExcelExporter:
    """Handles Excel file export."""
    
    @staticmethod
    def to_excel_bytes(df: pd.DataFrame, sheet_name: str = "Data") -> bytes:
        """Export DataFrame to Excel bytes."""
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
        excel_buffer.seek(0)
        return excel_buffer.getvalue()


# =============================================================================
# CACHED FUNCTIONS
# =============================================================================

@st.cache_data(ttl=300, show_spinner=False)
def check_availability_cached(
    base_url: str,
    filename_pattern: str,
    timeout: int,
    date_str: str
) -> Tuple[bool, Optional[int], Optional[str]]:
    """
    Cached availability check.
    
    Returns tuple instead of dataclass for pickle compatibility.
    
    Returns:
        Tuple of (is_available, file_size, error_message)
    """
    dt = datetime.strptime(date_str, "%Y-%m-%d").date()
    date_formatted = dt.strftime('%Y%m%d')
    filename = filename_pattern.replace("{date}", date_formatted)
    url = f"{base_url}{filename}"
    
    try:
        response = requests.head(url, timeout=timeout, allow_redirects=True)
        
        if response.status_code == 200:
            content_length = response.headers.get('Content-Length')
            size = int(content_length) if content_length else None
            return (True, size, None)
        
        return (False, None, f"HTTP {response.status_code}")
        
    except requests.exceptions.Timeout:
        return (False, None, "Connection timeout")
    except requests.exceptions.RequestException as e:
        return (False, None, str(e))


# =============================================================================
# DATA LAYER
# =============================================================================

class URLBuilder:
    """Constructs URLs for data files based on date patterns."""
    
    def __init__(self, config: AppConfig):
        self.config = config
    
    def build(self, data_date: date) -> str:
        """Build full URL for a given date."""
        date_str = data_date.strftime('%Y%m%d')
        filename = self.config.filename_pattern.replace("{date}", date_str)
        return f"{self.config.base_url}{filename}"


class DataLoader:
    """Handles data loading from remote sources with progress tracking."""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.url_builder = URLBuilder(config)
    
    def check_availability(self, date_str: str) -> DataAvailability:
        """Check if data file exists for the given date."""
        result = check_availability_cached(
            self.config.base_url,
            self.config.filename_pattern,
            self.config.connect_timeout,
            date_str
        )
        return DataAvailability(
            is_available=result[0],
            file_size=result[1],
            error_message=result[2]
        )
    
    def load_with_progress(self, data_date: date, progress_placeholder) -> pd.DataFrame:
        """Load data with progress indicator."""
        url = self.url_builder.build(data_date)
        
        content = self._download_with_progress(url, progress_placeholder)
        if content is None:
            return pd.DataFrame()
        
        if not self._is_valid_zip(content):
            self._show_invalid_file_error(content)
            return pd.DataFrame()
        
        return self._extract_and_read(content, data_date, progress_placeholder)
    
    def _download_with_progress(self, url: str, progress_placeholder) -> Optional[bytes]:
        """Download file with streaming and progress bar."""
        try:
            response = requests.get(
                url,
                stream=True,
                timeout=(self.config.connect_timeout, self.config.read_timeout),
                allow_redirects=True
            )
            response.raise_for_status()
            
            total_size = response.headers.get('Content-Length')
            total_size = int(total_size) if total_size else None
            
            chunks: List[bytes] = []
            downloaded = 0
            
            progress_bar = progress_placeholder.progress(0, text="📥 Downloading...")
            
            for chunk in response.iter_content(chunk_size=self.config.chunk_size):
                if chunk:
                    chunks.append(chunk)
                    downloaded += len(chunk)
                    
                    if total_size:
                        progress = min(downloaded / total_size, 1.0)
                        text = f"📥 Downloading... {Formatter.to_file_size(downloaded)} / {Formatter.to_file_size(total_size)}"
                    else:
                        progress = 0.5
                        text = f"📥 Downloading... {Formatter.to_file_size(downloaded)}"
                    
                    progress_bar.progress(progress, text=text)
            
            progress_bar.progress(1.0, text="✅ Download complete!")
            return b''.join(chunks)
            
        except requests.exceptions.Timeout:
            progress_placeholder.error("⏱️ Download timeout - connection too slow")
            return None
        except requests.exceptions.RequestException as e:
            progress_placeholder.error(f"❌ Download error: {e}")
            return None
    
    def _is_valid_zip(self, content: bytes) -> bool:
        """Check if content starts with ZIP magic bytes."""
        return len(content) >= 4 and content[:2] == b'PK'
    
    def _show_invalid_file_error(self, content: bytes) -> None:
        """Display error message for invalid file content."""
        st.error("❌ File bukan ZIP yang valid")
        with st.expander("🔍 Debug Info"):
            st.write(f"File size: {Formatter.to_file_size(len(content))}")
            st.code(content[:200].decode('utf-8', errors='replace'))
    
    def _extract_and_read(self, content: bytes, data_date: date, progress_placeholder) -> pd.DataFrame:
        """Extract ZIP and read CSV content."""
        try:
            progress_placeholder.info("📦 Extracting ZIP...")
            
            with zipfile.ZipFile(io.BytesIO(content)) as z:
                csv_files = [f for f in z.namelist() if f.endswith('.csv')]
                
                if not csv_files:
                    st.error("❌ No CSV files found in ZIP archive")
                    return pd.DataFrame()
                
                target = (
                    self.config.csv_filename_in_zip 
                    if self.config.csv_filename_in_zip in csv_files 
                    else csv_files[0]
                )
                
                progress_placeholder.info(f"📄 Reading {target}...")
                
                with z.open(target) as file:
                    df = pd.read_csv(file, low_memory=False)
            
            progress_placeholder.empty()
            
            df = self._preprocess(df)
            df["_DATA_DATE"] = data_date.strftime("%Y-%m-%d")
            
            return df
            
        except zipfile.BadZipFile as e:
            st.error(f"❌ Invalid ZIP file: {e}")
            return pd.DataFrame()
        except pd.errors.ParserError as e:
            st.error(f"❌ CSV parsing error: {e}")
            return pd.DataFrame()
        except Exception as e:
            st.error(f"❌ Unexpected error: {e}")
            return pd.DataFrame()
    
    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply preprocessing transformations to DataFrame."""
        if df.empty:
            return df
        
        unnamed_cols = [c for c in df.columns if 'Unnamed' in str(c)]
        if unnamed_cols:
            df = df.drop(columns=unnamed_cols)
        
        if "Tahun" in df.columns:
            df["Tahun"] = pd.to_numeric(df["Tahun"], errors="coerce").astype("Int64")
        
        return df


# =============================================================================
# ANALYSIS LAYER
# =============================================================================

class DataAggregator:
    """Handles data aggregation operations."""
    
    @staticmethod
    def aggregate(
        df: pd.DataFrame,
        group_columns: List[str],
        numeric_columns: List[str],
        agg_function: str = 'sum'
    ) -> pd.DataFrame:
        """Aggregate numeric columns by group columns."""
        available_groups = [c for c in group_columns if c in df.columns]
        available_numerics = [c for c in numeric_columns if c in df.columns]
        
        if not available_groups or not available_numerics:
            return pd.DataFrame()
        
        agg_dict = {col: agg_function for col in available_numerics}
        return df.groupby(available_groups, as_index=False).agg(agg_dict)


class DatasetComparator:
    """Compares two datasets and calculates differences."""
    
    def compare(
        self,
        df_primary: pd.DataFrame,
        df_comparison: pd.DataFrame,
        group_columns: List[str],
        numeric_columns: List[str],
        date_primary: str,
        date_comparison: str
    ) -> pd.DataFrame:
        """Compare two datasets and calculate differences."""
        if date_primary == date_comparison:
            st.warning("⚠️ Tanggal utama dan pembanding sama. Pilih tanggal berbeda.")
            return pd.DataFrame()
        
        available_groups = [
            c for c in group_columns 
            if c in df_primary.columns and c in df_comparison.columns
        ]
        available_numerics = [
            c for c in numeric_columns 
            if c in df_primary.columns and c in df_comparison.columns
        ]
        
        if not available_groups or not available_numerics:
            st.warning("⚠️ Tidak ada kolom yang cocok untuk perbandingan")
            return pd.DataFrame()
        
        agg_primary = DataAggregator.aggregate(df_primary, available_groups, available_numerics)
        agg_comparison = DataAggregator.aggregate(df_comparison, available_groups, available_numerics)
        
        if agg_primary.empty or agg_comparison.empty:
            st.warning("⚠️ Data agregasi kosong")
            return pd.DataFrame()
        
        rename_primary = {c: f"{c}_{date_primary}" for c in available_numerics}
        rename_comparison = {c: f"{c}_{date_comparison}" for c in available_numerics}
        
        agg_primary = agg_primary.rename(columns=rename_primary)
        agg_comparison = agg_comparison.rename(columns=rename_comparison)
        
        comparison = pd.merge(agg_primary, agg_comparison, on=available_groups, how='outer')
        
        for col in available_numerics:
            col_primary = f"{col}_{date_primary}"
            col_comparison = f"{col}_{date_comparison}"
            
            if col_primary not in comparison.columns or col_comparison not in comparison.columns:
                continue
            
            comparison[f"SELISIH_{col}"] = (
                comparison[col_primary].fillna(0) - 
                comparison[col_comparison].fillna(0)
            )
            
            comparison[f"%CHG_{col}"] = np.where(
                comparison[col_comparison].fillna(0) != 0,
                (comparison[f"SELISIH_{col}"] / comparison[col_comparison].fillna(0).abs()) * 100,
                np.nan
            )
        
        return comparison


# =============================================================================
# UI COMPONENTS
# =============================================================================

class UIComponents:
    """Reusable UI component generators."""
    
    @staticmethod
    def render_header() -> None:
        """Render the main dashboard header."""
        st.markdown("""
        <div class="dashboard-header">
            <p class="breadcrumb">DASHBOARD / MONITORING</p>
            <h1 class="dashboard-title">Monitoring Anggaran</h1>
            <p class="dashboard-subtitle">Perbandingan Data Anggaran Berdasarkan Tanggal Update Database</p>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_availability_badge(availability: DataAvailability) -> str:
        """Generate HTML for availability badge."""
        if availability.is_available:
            size_text = f" • {Formatter.to_file_size(availability.file_size)}" if availability.file_size else ""
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
    
    @staticmethod
    def render_data_unavailable_message() -> None:
        """Render message when data is not available."""
        dates_list = "\n".join(f"<li>{d}</li>" for d in AVAILABLE_DATES)
        st.markdown(f"""
        <div class="material-card">
            <h3>⚠️ Data Tidak Tersedia</h3>
            <p>Data tidak ditemukan untuk tanggal yang dipilih.</p>
            <p><strong>Tanggal yang tersedia:</strong></p>
            <ul>{dates_list}</ul>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_comparison_header(date1: str, date2: str) -> None:
        """Render comparison section header."""
        st.markdown(f"""
        <div class="comparison-header">
            <strong>{date1}</strong> vs <strong>{date2}</strong>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_filter_header() -> None:
        """Render filter section header."""
        st.markdown("""
        <div class="filter-container">
            <strong>🔍 Filter Data</strong>
        </div>
        """, unsafe_allow_html=True)


class SidebarController:
    """Controls sidebar UI elements and state."""
    
    def __init__(self, config: AppConfig, column_config: ColumnConfig):
        self.config = config
        self.column_config = column_config
        self.data_loader = DataLoader(config)
    
    def render_date_selector(
        self, 
        key_prefix: str = ""
    ) -> Tuple[str, Optional[str], bool, DataAvailability, Optional[DataAvailability]]:
        """Render date selection controls in sidebar."""
        st.sidebar.markdown("### 📅 Pengaturan Tanggal Data Update")
        help_text = "Tanggal tersedia:\n" + "\n".join(f"- {d}" for d in AVAILABLE_DATES)
        
        primary_dt = st.sidebar.date_input(
            "Pilih Tanggal Data Terkini",
            value=date(2025, 11, 11),
            min_value=self.config.date_start,
            max_value=self.config.date_end,
            key=f"{key_prefix}primary_date",
            help=help_text
        )
        primary_date = primary_dt.strftime("%Y-%m-%d")
        
        primary_availability = self.data_loader.check_availability(primary_date)
        st.sidebar.markdown(
            UIComponents.render_availability_badge(primary_availability),
            unsafe_allow_html=True
        )
        
        st.sidebar.markdown("---")
        
        enable_comparison = st.sidebar.checkbox(
            "Bandingkan dengan tanggal update data sebelumnya",
            value=False,
            key=f"{key_prefix}enable_comparison"
        )
        
        comparison_date = None
        comparison_availability = None
        
        if enable_comparison:
            default_comparison = (
                date(2025, 10, 27) 
                if primary_dt != date(2025, 10, 27) 
                else date(2025, 11, 11)
            )
            
            comparison_dt = st.sidebar.date_input(
                "Pilih Tanggal Pembanding",
                value=default_comparison,
                min_value=self.config.date_start,
                max_value=self.config.date_end,
                key=f"{key_prefix}comparison_date",
                help=help_text
            )
            comparison_date = comparison_dt.strftime("%Y-%m-%d")
            
            if comparison_date == primary_date:
                st.sidebar.error("⚠️ Pilih tanggal yang berbeda!")
            
            comparison_availability = self.data_loader.check_availability(comparison_date)
            st.sidebar.markdown(
                UIComponents.render_availability_badge(comparison_availability),
                unsafe_allow_html=True
            )
        
        return (
            primary_date, 
            comparison_date, 
            enable_comparison,
            primary_availability,
            comparison_availability
        )
    
    def render_aggregation_options(
        self, 
        df: pd.DataFrame,
        key_prefix: str = ""
    ) -> Tuple[List[int], List[str], List[str], str]:
        """Render aggregation options in sidebar including Tahun Anggaran selector."""
        st.sidebar.markdown("### Pilih Kolom Data")
        
        # Use actual available columns (or fall back to potential if not set)
        string_options = (
            self.column_config.string_columns 
            if self.column_config.string_columns 
            else list(self.column_config.POTENTIAL_STRING_COLUMNS)
        )
        numeric_options = (
            self.column_config.numeric_columns 
            if self.column_config.numeric_columns 
            else list(self.column_config.POTENTIAL_NUMERIC_COLUMNS)
        )
        
        # Get available defaults
        default_groups = self.column_config.get_available_group_defaults() or string_options[:2]
        default_numerics = self.column_config.get_available_numeric_defaults() or numeric_options[:1]
        
        # Tahun Anggaran selector (before group_cols)
        available_years = self.column_config.available_years
        if not available_years and "Tahun" in df.columns:
            available_years = sorted(df["Tahun"].dropna().unique().tolist())
        
        selected_years: List[int] = []
        if available_years:
            # Get current year as default if available, otherwise use all years
            current_year = datetime.now().year
            year_options = [int(y) for y in available_years]
            default_years = [current_year] if current_year in year_options else year_options
            
            selected_years = st.sidebar.multiselect(
                "Tahun Anggaran",
                options=year_options,
                default=default_years,
                help="Pilih tahun anggaran yang ingin ditampilkan",
                key=f"{key_prefix}tahun_anggaran"
            )
            
            if not selected_years:
                st.sidebar.warning("⚠️ Pilih minimal 1 tahun anggaran")
        
        # Group columns
        group_cols = st.sidebar.multiselect(
            "Kolom Kategori",
            options=string_options,
            default=default_groups,
            help="Pilih minimal 1 kolom untuk pengelompokan",
            key=f"{key_prefix}group_cols"
        )
        
        if len(group_cols) < 1:
            st.sidebar.error("⚠️ Pilih minimal 1 kolom kategori")
        
        # Numeric columns
        numeric_cols = st.sidebar.multiselect(
            "Kolom Numerik",
            options=numeric_options,
            default=default_numerics,
            help="Pilih minimal 1 kolom untuk agregasi",
            key=f"{key_prefix}numeric_cols"
        )
        
        if len(numeric_cols) < 1:
            st.sidebar.error("⚠️ Pilih minimal 1 kolom numerik")
        
        # Aggregation function
        agg_func = st.sidebar.selectbox(
            "Fungsi Agregasi",
            options=[f.value for f in AggregationFunction],
            format_func=lambda x: AggregationFunction(x).display_name,
            key=f"{key_prefix}agg_func"
        )
        
        return selected_years, group_cols, numeric_cols, agg_func


class FilterController:
    """Handles data filtering UI and logic."""
    
    def __init__(self, config: AppConfig):
        self.config = config
    
    def render_filters(
        self, 
        df: pd.DataFrame, 
        group_columns: List[str],
        key_prefix: str = ""
    ) -> Dict[str, List[str]]:
        """Render filter controls for the given columns."""
        UIComponents.render_filter_header()
        
        available_cols = [col for col in group_columns if col in df.columns]
        
        if not available_cols:
            return {}
        
        filters: Dict[str, List[str]] = {}
        cols_per_row = min(len(available_cols), self.config.max_filters_per_row)
        filter_columns = st.columns(cols_per_row)
        
        for idx, col in enumerate(available_cols):
            col_idx = idx % cols_per_row
            
            with filter_columns[col_idx]:
                unique_values = sorted(str(v) for v in df[col].dropna().unique())
                
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
            active_count = sum(1 for v in filters.values() if v)
            st.caption(f"🔹 {active_count} filter aktif")
        
        return filters
    
    @staticmethod
    def apply_filters(df: pd.DataFrame, filters: Dict[str, List[str]]) -> pd.DataFrame:
        """Apply filters to DataFrame."""
        if df.empty or not filters:
            return df
        
        filtered_df = df.copy()
        
        for col, values in filters.items():
            if col in filtered_df.columns and values:
                filtered_df = filtered_df[filtered_df[col].astype(str).isin(values)]
        
        return filtered_df
    
    @staticmethod
    def filter_by_years(df: pd.DataFrame, years: List[int]) -> pd.DataFrame:
        """Filter DataFrame by selected years."""
        if df.empty or not years or "Tahun" not in df.columns:
            return df
        
        return df[df["Tahun"].isin(years)]


# =============================================================================
# MAIN APPLICATION
# =============================================================================

class MonitoringDashboard:
    """Main application controller for the Budget Monitoring Dashboard."""
    
    def __init__(self):
        self.config = AppConfig()
        self.column_config = ColumnConfig()
        self.data_loader = DataLoader(self.config)
        self.sidebar = SidebarController(self.config, self.column_config)
        self.filter_controller = FilterController(self.config)
        self.comparator = DatasetComparator()
    
    def run(self) -> None:
        """Main application entry point."""
        self._configure_page()
        self._apply_styles()
        
        UIComponents.render_header()
        
        (
            primary_date, comparison_date, enable_comparison,
            primary_availability, comparison_availability
        ) = self.sidebar.render_date_selector()
        
        if not primary_availability.is_available:
            UIComponents.render_data_unavailable_message()
            return
        
        if (primary_availability.file_size and 
            primary_availability.file_size > self.config.large_file_threshold):
            size_str = Formatter.to_file_size(primary_availability.file_size)
            st.info(f"📦 File besar ({size_str}). Download mungkin memakan waktu...")
        
        progress_placeholder = st.empty()
        primary_dt = datetime.strptime(primary_date, "%Y-%m-%d").date()
        df_primary = self.data_loader.load_with_progress(primary_dt, progress_placeholder)
        
        if df_primary.empty:
            return
        
        # Update column config based on actual data
        self.column_config.update_from_dataframe(df_primary)
        
        date_label = Formatter.to_indonesian_date(primary_dt)
        st.success(f"✅ Data dimuat: **{len(df_primary):,}** baris | {date_label}")
        
        st.sidebar.divider()
        selected_years, group_cols, numeric_cols, agg_func = self.sidebar.render_aggregation_options(
            df_primary
        )
        
        # Filter by selected years
        if selected_years:
            df_primary = FilterController.filter_by_years(df_primary, selected_years)
            if df_primary.empty:
                st.warning("⚠️ Tidak ada data untuk tahun yang dipilih")
                return
        
        if len(group_cols) < 1 or len(numeric_cols) < 1:
            st.info("👆 Pilih minimal 1 kolom kategori dan 1 kolom numerik di sidebar")
            with st.expander("📋 Preview Data Mentah"):
                st.dataframe(df_primary.head(100), width="stretch")
            return
        
        agg_primary = DataAggregator.aggregate(df_primary, group_cols, numeric_cols, agg_func)
        formatter = DataFrameFormatter(numeric_cols)
        agg_primary_display = formatter.format_for_display(agg_primary)
        
        if enable_comparison and comparison_date and comparison_availability and comparison_availability.is_available:
            self._render_comparison_view(
                df_primary, primary_date,
                comparison_date, comparison_availability,
                group_cols, numeric_cols, agg_func,
                agg_primary_display, formatter,
                selected_years
            )
        else:
            self._render_single_date_view(
                df_primary, primary_date,
                group_cols, numeric_cols,
                agg_primary_display, formatter
            )
    
    def _configure_page(self) -> None:
        """Configure Streamlit page settings."""
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
    
    def _apply_styles(self) -> None:
        """Apply CSS styles to the page."""
        st.markdown(CSS_STYLES, unsafe_allow_html=True)
    
    def _render_comparison_view(
        self,
        df_primary: pd.DataFrame,
        primary_date: str,
        comparison_date: str,
        comparison_availability: DataAvailability,
        group_cols: List[str],
        numeric_cols: List[str],
        agg_func: str,
        agg_primary_display: pd.DataFrame,
        formatter: DataFrameFormatter,
        selected_years: List[int]
    ) -> None:
        """Render the date comparison view."""
        if comparison_date == primary_date:
            st.error("⚠️ **Tanggal sama!** Pilih tanggal pembanding yang berbeda dari tanggal utama.")
            st.dataframe(agg_primary_display, width="stretch", hide_index=True)
            return
        
        progress_placeholder = st.empty()
        comparison_dt = datetime.strptime(comparison_date, "%Y-%m-%d").date()
        df_comparison = self.data_loader.load_with_progress(comparison_dt, progress_placeholder)
        
        if df_comparison.empty:
            st.dataframe(agg_primary_display, width="stretch", hide_index=True)
            return
        
        # Filter comparison data by selected years
        if selected_years:
            df_comparison = FilterController.filter_by_years(df_comparison, selected_years)
        
        comparison_label = Formatter.to_indonesian_date(comparison_dt)
        st.success(f"✅ Data pembanding dimuat: **{len(df_comparison):,}** baris")
        
        comparison_df = self.comparator.compare(
            df_primary, df_comparison,
            group_cols, numeric_cols,
            primary_date, comparison_date
        )
        
        if comparison_df.empty:
            st.dataframe(agg_primary_display, width="stretch", hide_index=True)
            return
        
        primary_dt = datetime.strptime(primary_date, "%Y-%m-%d").date()
        primary_label = Formatter.to_indonesian_date(primary_dt)
        
        UIComponents.render_comparison_header(primary_label, comparison_label)
        
        # Render filters - affects both summary and detail
        filters = self.filter_controller.render_filters(
            comparison_df, group_cols, key_prefix="comp_"
        )
        
        # Apply filters to all data
        df_primary_filtered = self.filter_controller.apply_filters(df_primary, filters)
        df_comparison_filtered = self.filter_controller.apply_filters(df_comparison, filters)
        filtered_comparison_df = self.filter_controller.apply_filters(comparison_df, filters)
        
        if filters:
            st.info(f"Filter diterapkan: **{len(filtered_comparison_df):,}** dari **{len(comparison_df):,}** baris")
        
        st.divider()
        
        # Render comparison summary (uses filtered data)
        self._render_comparison_summary(
            df_primary_filtered, df_comparison_filtered,
            numeric_cols,
            primary_label, comparison_label
        )
        
        st.divider()
        
        # Render comparison detail table (uses filtered data)
        self._render_comparison_detail(
            filtered_comparison_df, group_cols, numeric_cols,
            primary_date, comparison_date,
            primary_label, comparison_label,
            formatter
        )
    
    def _render_comparison_summary(
        self,
        df_primary_filtered: pd.DataFrame,
        df_comparison_filtered: pd.DataFrame,
        numeric_cols: List[str],
        primary_label: str,
        comparison_label: str
    ) -> None:
        """Render summary metrics for comparison (uses pre-filtered data)."""
        st.markdown("### 📈 Ringkasan Perbandingan Antar Data")
        
        for col in numeric_cols:
            if col not in df_primary_filtered.columns or col not in df_comparison_filtered.columns:
                continue
            
            st.markdown(f"**{col}**")
            c1, c2, c3 = st.columns(3)
            
            val_primary = df_primary_filtered[col].sum()
            val_comparison = df_comparison_filtered[col].sum()
            diff = val_primary - val_comparison
            pct = (diff / abs(val_comparison) * 100) if val_comparison != 0 else 0
            
            c1.metric(primary_label, Formatter.to_rupiah_short(val_primary))
            c2.metric(comparison_label, Formatter.to_rupiah_short(val_comparison))
            c3.metric("Selisih", Formatter.to_rupiah_short(diff), f"{pct:+.2f}%")
    
    def _render_comparison_detail(
        self,
        filtered_df: pd.DataFrame,
        group_cols: List[str],
        numeric_cols: List[str],
        primary_date: str,
        comparison_date: str,
        primary_label: str,
        comparison_label: str,
        formatter: DataFrameFormatter
    ) -> None:
        """Render detailed comparison table (uses pre-filtered data)."""
        st.markdown("### 📋 Detail Perbandingan Data")
        
        with st.expander("ℹ️ Keterangan Kolom", expanded=False):
            st.markdown(f"""
            - `[kolom]_{primary_date}` = Nilai tanggal utama
            - `[kolom]_{comparison_date}` = Nilai tanggal pembanding
            - `SELISIH_[kolom]` = Selisih (utama - pembanding)
            - `%CHG_[kolom]` = Persentase perubahan
            """)
        
        display_df = formatter.format_for_display(filtered_df)
        st.dataframe(display_df, width="stretch", hide_index=True)
        
        # Excel download
        excel_data = ExcelExporter.to_excel_bytes(
            filtered_df, 
            sheet_name="Perbandingan"
        )
        st.download_button(
            "📥 Download Perbandingan (Excel)",
            excel_data,
            f"perbandingan_{primary_date}_vs_{comparison_date}.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="dl_comparison"
        )
        
    def _render_single_date_view(
        self,
        df_primary: pd.DataFrame,
        primary_date: str,
        group_cols: List[str],
        numeric_cols: List[str],
        agg_primary: pd.DataFrame,
        agg_primary_display: pd.DataFrame,
        formatter: DataFrameFormatter,
        selected_years: List[int]
    ) -> None:
        """Render single date view with filters affecting both summary and detail."""
        primary_dt = datetime.strptime(primary_date, "%Y-%m-%d").date()
        date_label = Formatter.to_indonesian_date(primary_dt)
        
        # Render header and filters first
        st.markdown(f"### 📊 Analisis Data - {date_label}")
        
        # Render filters - affects both summary and detail
        filters = self.filter_controller.render_filters(
            agg_primary, group_cols, key_prefix="single_"
        )
        
        # Apply filters to aggregated data
        filtered_agg = self.filter_controller.apply_filters(agg_primary, filters)
        
        # Also filter raw data for summary calculations
        df_filtered = self.filter_controller.apply_filters(df_primary, filters)
        
        if filters:
            st.info(f"Filter diterapkan: **{len(filtered_agg):,}** dari **{len(agg_primary):,}** baris")
        
        st.divider()
        
        # Render column summary with comparison (if more than 1 numeric column)
        if len(numeric_cols) > 1:
            self._render_column_summary(df_filtered, numeric_cols)
            st.divider()
            self._render_column_comparison_table(df_filtered, numeric_cols)
            st.divider()
        
        # Render data table
        st.markdown(f"### 📋 Data Agregasi")
        filtered_display = formatter.format_for_display(filtered_agg)
        st.dataframe(filtered_display, width="stretch", hide_index=True)
        
        # Excel download
        excel_data = ExcelExporter.to_excel_bytes(
            filtered_agg,
            sheet_name="Agregasi"
        )
        st.download_button(
            "📥 Download Excel",
            excel_data,
            f"agregasi_{primary_date}.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="dl_single"
        )
    
    def _render_column_summary(self, df: pd.DataFrame, numeric_cols: List[str]) -> None:
        """
        Render summary metrics for numeric columns.
        First column is primary (baseline), others show diff and %chg from primary.
        """
        st.markdown("### 📈 Ringkasan Perbandingan Kolom Numerik")
        
        if len(numeric_cols) < 1:
            return
        
        # First column is the primary (baseline)
        primary_col = numeric_cols[0]
        if primary_col not in df.columns:
            return
        
        primary_value = df[primary_col].sum()
        
        # Show primary column
        st.markdown(f"**Kolom Utama (Baseline): {primary_col}**")
        st.metric(primary_col, Formatter.to_rupiah_short(primary_value))
        
        # Show comparison columns (2nd and onwards)
        if len(numeric_cols) > 1:
            st.markdown("**Perbandingan dengan Kolom Lain:**")
            
            comparison_cols = numeric_cols[1:]
            cols_per_row = min(len(comparison_cols), self.config.metrics_per_row)
            
            for i in range(0, len(comparison_cols), cols_per_row):
                chunk = comparison_cols[i:i + cols_per_row]
                cols = st.columns(len(chunk))
                
                for idx, col in enumerate(chunk):
                    if col in df.columns:
                        value = df[col].sum()
                        diff = value - primary_value
                        pct = (diff / abs(primary_value) * 100) if primary_value != 0 else 0
                        
                        cols[idx].metric(
                            col, 
                            Formatter.to_rupiah_short(value),
                            f"{pct:+.2f}% vs {primary_col[:20]}..."
                        )
    
    def _render_column_comparison_table(self, df: pd.DataFrame, numeric_cols: List[str]) -> None:
        """
        Render comparison table for numeric columns.
        Shows each column's stats with diff and %chg from primary (1st column).
        """
        st.markdown("### 📊 Tabel Perbandingan Antar Kolom Numerik")
        
        if len(numeric_cols) < 1:
            return
        
        # First column is the primary (baseline)
        primary_col = numeric_cols[0]
        if primary_col not in df.columns:
            return
        
        primary_total = df[primary_col].sum()
        
        summary_data = []
        for col in numeric_cols:
            if col not in df.columns:
                continue
            
            total = df[col].sum()
            diff = total - primary_total
            pct = (diff / abs(primary_total) * 100) if primary_total != 0 else 0
            
            row_data = {
                'Kolom': col,
                'Total': total,
                'Rata-rata': df[col].mean(),
                'Min': df[col].min(),
                'Max': df[col].max(),
                'Count': df[col].count(),
            }
            
            # Add diff and %chg columns (except for primary column itself)
            if col == primary_col:
                row_data[f'Selisih vs {primary_col[:15]}...'] = '-'
                row_data['%Chg'] = '-'
            else:
                row_data[f'Selisih vs {primary_col[:15]}...'] = diff
                row_data['%Chg'] = pct
            
            summary_data.append(row_data)
        
        if not summary_data:
            return
        
        summary_df = pd.DataFrame(summary_data)
        
        # Format display
        display_df = summary_df.copy()
        for col in ['Total', 'Rata-rata', 'Min', 'Max']:
            display_df[col] = display_df[col].apply(Formatter.to_rupiah_full)
        display_df['Count'] = display_df['Count'].apply(Formatter.to_number_with_separator)
        
        # Format diff column (handle '-' for primary row)
        diff_col = f'Selisih vs {primary_col[:15]}...'
        display_df[diff_col] = display_df[diff_col].apply(
            lambda x: Formatter.to_rupiah_full(x) if x != '-' else '-'
        )
        display_df['%Chg'] = display_df['%Chg'].apply(
            lambda x: Formatter.to_percentage(x) if x != '-' else '-'
        )
        
        st.dataframe(display_df, width="stretch", hide_index=True)
        
        with st.expander("ℹ️ Keterangan", expanded=False):
            st.markdown(f"""
            - **Kolom Utama (Baseline)**: `{primary_col}`
            - **Selisih**: Nilai kolom - Nilai {primary_col}
            - **%Chg**: Persentase perubahan terhadap {primary_col}
            """)


# =============================================================================
# ENTRY POINT
# =============================================================================

def main() -> None:
    """Application entry point."""
    dashboard = MonitoringDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()





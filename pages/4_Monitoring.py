"""
Monitoring Anggaran - Budget Monitoring Dashboard
=================================================

A Streamlit dashboard for comparing Indonesian government budget data
across different update dates. Supports data aggregation, filtering,
and comparison analysis.

Author: Budget Analysis Team
Version: 3.0.0
"""

from __future__ import annotations

import io
import zipfile
from abc import ABC, abstractmethod
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
        """Return human-readable name for UI display."""
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
    """
    Application configuration settings.
    
    Frozen dataclass ensures configuration immutability after initialization.
    """
    # Data source settings
    base_url: str = "https://raw.githubusercontent.com/tubankum3/dashanggaran/main/"
    filename_pattern: str = "df_{date}.csv.zip"
    csv_filename_in_zip: str = "df.csv"
    
    # Date range settings
    date_start: date = field(default_factory=lambda: date(2024, 1, 1))
    date_end: date = field(default_factory=date.today)
    
    # Network settings
    connect_timeout: int = 10
    read_timeout: int = 300
    chunk_size: int = 1024 * 1024  # 1MB chunks for streaming
    
    # Cache settings
    cache_ttl: int = 300  # 5 minutes
    
    # UI settings
    large_file_threshold: int = 50_000_000  # 50MB
    max_filters_per_row: int = 4
    metrics_per_row: int = 3


@dataclass(frozen=True)
class ColumnConfig:
    """Column definitions for the budget dataset."""
    
    STRING_COLUMNS: Tuple[str, ...] = (
        'KEMENTERIAN/LEMBAGA', 'SUMBER DANA', 'FUNGSI', 'SUB FUNGSI',
        'PROGRAM', 'KEGIATAN', 'OUTPUT (KRO)', 'SUB OUTPUT (RO)',
        'KOMPONEN', 'JENIS BELANJA', 'AKUN 4 DIGIT', 'Tahun'
    )
    
    NUMERIC_COLUMNS: Tuple[str, ...] = (
        'REALISASI BELANJA KL (SAKTI)', 'PAGU DIPA REVISI', 'BLOKIR DIPA REVISI',
        'PAGU DIPA AWAL', 'BLOKIR DIPA AWAL', 'PAGU DIPA AWAL EFEKTIF',
        'PAGU DIPA REVISI EFEKTIF'
    )
    
    # Default selections
    DEFAULT_GROUP_COLS: Tuple[str, ...] = ('Tahun', 'KEMENTERIAN/LEMBAGA')
    DEFAULT_NUMERIC_COLS: Tuple[str, ...] = ('PAGU DIPA REVISI')


# Indonesian month names for date formatting
INDONESIAN_MONTHS: Dict[int, str] = {
    1: 'Januari', 2: 'Februari', 3: 'Maret', 4: 'April',
    5: 'Mei', 6: 'Juni', 7: 'Juli', 8: 'Agustus',
    9: 'September', 10: 'Oktober', 11: 'November', 12: 'Desember'
}

# Known available dates (for user reference)
AVAILABLE_DATES: List[str] = ["27 Oktober 2025", "11 November 2025"]


# =============================================================================
# STYLING
# =============================================================================

CSS_STYLES = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    /* Color Palette */
    --primary: #0066FF;
    --primary-dark: #0052CC;
    --primary-light: #4D94FF;
    --secondary: #10B981;
    --warning: #F59E0B;
    --error: #EF4444;
    --success: #10B981;
    
    /* Grayscale */
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
    
    /* Surfaces */
    --surface: #FFFFFF;
    --background: #F9FAFB;
    --on-primary: #FFFFFF;
    
    /* Effects */
    --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    
    /* Spacing & Radius */
    --radius-md: 8px;
    --radius-lg: 12px;
    --radius-full: 9999px;
    --space-md: 1rem;
    --space-lg: 1.5rem;
    --space-xl: 2rem;
    
    /* Transitions */
    --transition-fast: 150ms cubic-bezier(0.4, 0, 0.2, 1);
}

* { font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
.stApp { background-color: var(--background); }

/* Header */
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

/* Cards */
.material-card {
    background: var(--surface);
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow-sm);
    padding: var(--space-xl);
    margin-bottom: var(--space-lg);
    border: 1px solid var(--gray-200);
}

/* Buttons */
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

/* Sidebar */
.stSidebar { background: var(--surface); border-right: 1px solid var(--gray-200); }

/* Tabs */
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

/* Badges */
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

/* Comparison Header */
.comparison-header {
    background: var(--gray-100);
    padding: var(--space-md);
    border-radius: var(--radius-md);
    margin-bottom: var(--space-md);
}

/* Filter Container */
.filter-container {
    background: var(--gray-50);
    padding: var(--space-md);
    border-radius: var(--radius-md);
    margin-bottom: var(--space-lg);
    border: 1px solid var(--gray-200);
}

/* Scrollbar */
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
    """
    Utility class for formatting values in various formats.
    
    Provides static methods for currency, percentage, file size,
    and date formatting with Indonesian locale support.
    """
    
    @staticmethod
    def to_rupiah_short(value: float) -> str:
        """
        Format value as Indonesian Rupiah with abbreviated units.
        
        Args:
            value: Numeric value to format
            
        Returns:
            Formatted string (e.g., "Rp 1.5 T" for trillion)
        """
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
        """
        Format value as Indonesian Rupiah with full number and dot separator.
        
        Args:
            value: Numeric value to format
            
        Returns:
            Formatted string (e.g., "Rp 1.500.000.000")
        """
        try:
            value = float(value)
        except (ValueError, TypeError):
            return str(value)
        
        if pd.isna(value):
            return "-"
        
        return f"Rp {value:,.0f}".replace(",", ".")
    
    @staticmethod
    def to_percentage(value: Any) -> str:
        """
        Format value as percentage.
        
        Args:
            value: Numeric value to format
            
        Returns:
            Formatted percentage string (e.g., "12.34%")
        """
        try:
            value = float(value)
        except (ValueError, TypeError):
            return str(value)
        
        if pd.isna(value):
            return "-"
        
        return f"{value:,.2f}%"
    
    @staticmethod
    def to_file_size(size_bytes: Optional[int]) -> str:
        """
        Format byte count as human-readable file size.
        
        Args:
            size_bytes: File size in bytes
            
        Returns:
            Formatted size string (e.g., "150.23 MB")
        """
        if size_bytes is None:
            return "Unknown"
        
        units = [
            (1_000_000_000, "GB"),
            (1_000_000, "MB"),
            (1_000, "KB"),
        ]
        
        for threshold, unit in units:
            if size_bytes >= threshold:
                return f"{size_bytes / threshold:.2f} {unit}"
        
        return f"{size_bytes} bytes"
    
    @staticmethod
    def to_indonesian_date(dt: date) -> str:
        """
        Format date using Indonesian month names.
        
        Args:
            dt: Date object to format
            
        Returns:
            Formatted date string (e.g., "27 Oktober 2025")
        """
        month_name = INDONESIAN_MONTHS.get(dt.month, str(dt.month))
        return f"{dt.day} {month_name} {dt.year}"
    
    @staticmethod
    def to_number_with_separator(value: Any) -> str:
        """
        Format number with thousand separator.
        
        Args:
            value: Numeric value to format
            
        Returns:
            Formatted number string (e.g., "1,500,000")
        """
        try:
            return f"{int(value):,}"
        except (ValueError, TypeError):
            return str(value)


class DataFrameFormatter:
    """Formats DataFrames for display with appropriate value formatting."""
    
    def __init__(self, numeric_columns: List[str]):
        """
        Initialize formatter with column configuration.
        
        Args:
            numeric_columns: List of column names that contain numeric values
        """
        self.numeric_columns = numeric_columns
    
    def format_for_display(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply formatting to DataFrame for UI display.
        
        Formats numeric columns as Rupiah and percentage columns appropriately.
        
        Args:
            df: DataFrame to format
            
        Returns:
            Formatted DataFrame copy (original unchanged)
        """
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
        """Check if column should be formatted as currency."""
        return (
            any(nc in column_name for nc in self.numeric_columns) or
            column_name.startswith('SELISIH_')
        )
    
    def _is_percentage_column(self, column_name: str) -> bool:
        """Check if column should be formatted as percentage."""
        return column_name.startswith('PCT_')


# =============================================================================
# DATA LAYER
# =============================================================================

@dataclass
class DataAvailability:
    """Represents the availability status of a data file."""
    is_available: bool
    file_size: Optional[int] = None
    error_message: Optional[str] = None


class URLBuilder:
    """Constructs URLs for data files based on date patterns."""
    
    def __init__(self, config: AppConfig):
        self.config = config
    
    def build(self, data_date: date) -> str:
        """
        Build full URL for a given date.
        
        Args:
            data_date: Date for which to build URL
            
        Returns:
            Complete URL string
        """
        date_str = data_date.strftime('%Y%m%d')
        filename = self.config.filename_pattern.replace("{date}", date_str)
        return f"{self.config.base_url}{filename}"


class DataLoader:
    """
    Handles data loading from remote sources with progress tracking.
    
    Supports streaming downloads for large files and ZIP extraction.
    """
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.url_builder = URLBuilder(config)
    
    def check_availability(self, data_date: date) -> DataAvailability:
        """
        Check if data file exists for the given date.
        
        Args:
            data_date: Date to check
            
        Returns:
            DataAvailability object with status and file size
        """
        url = self.url_builder.build(data_date)
        
        try:
            response = requests.head(
                url, 
                timeout=self.config.connect_timeout,
                allow_redirects=True
            )
            
            if response.status_code == 200:
                content_length = response.headers.get('Content-Length')
                size = int(content_length) if content_length else None
                return DataAvailability(is_available=True, file_size=size)
            
            return DataAvailability(
                is_available=False,
                error_message=f"HTTP {response.status_code}"
            )
            
        except requests.exceptions.Timeout:
            return DataAvailability(
                is_available=False,
                error_message="Connection timeout"
            )
        except requests.exceptions.RequestException as e:
            return DataAvailability(
                is_available=False,
                error_message=str(e)
            )
    
    def load_with_progress(
        self, 
        data_date: date, 
        progress_placeholder
    ) -> pd.DataFrame:
        """
        Load data with progress indicator.
        
        Args:
            data_date: Date of data to load
            progress_placeholder: Streamlit placeholder for progress display
            
        Returns:
            Loaded and preprocessed DataFrame, or empty DataFrame on error
        """
        url = self.url_builder.build(data_date)
        
        # Download with progress
        content = self._download_with_progress(url, progress_placeholder)
        if content is None:
            return pd.DataFrame()
        
        # Validate ZIP file
        if not self._is_valid_zip(content):
            self._show_invalid_file_error(content)
            return pd.DataFrame()
        
        # Extract and read CSV
        return self._extract_and_read(content, data_date, progress_placeholder)
    
    def _download_with_progress(
        self, 
        url: str, 
        progress_placeholder
    ) -> Optional[bytes]:
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
    
    def _extract_and_read(
        self, 
        content: bytes, 
        data_date: date,
        progress_placeholder
    ) -> pd.DataFrame:
        """Extract ZIP and read CSV content."""
        try:
            progress_placeholder.info("📦 Extracting ZIP...")
            
            with zipfile.ZipFile(io.BytesIO(content)) as z:
                csv_files = [f for f in z.namelist() if f.endswith('.csv')]
                
                if not csv_files:
                    st.error("❌ No CSV files found in ZIP archive")
                    return pd.DataFrame()
                
                # Use configured filename or first CSV
                target = (
                    self.config.csv_filename_in_zip 
                    if self.config.csv_filename_in_zip in csv_files 
                    else csv_files[0]
                )
                
                progress_placeholder.info(f"📄 Reading {target}...")
                
                with z.open(target) as file:
                    df = pd.read_csv(file, low_memory=False)
            
            progress_placeholder.empty()
            
            # Preprocess and add metadata
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
        
        # Remove unnamed columns (often index columns from CSV)
        unnamed_cols = [c for c in df.columns if 'Unnamed' in str(c)]
        if unnamed_cols:
            df = df.drop(columns=unnamed_cols)
        
        # Convert Tahun to nullable integer
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
        """
        Aggregate numeric columns by group columns.
        
        Args:
            df: Source DataFrame
            group_columns: Columns to group by
            numeric_columns: Columns to aggregate
            agg_function: Aggregation function name
            
        Returns:
            Aggregated DataFrame
        """
        # Filter to available columns
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
        """
        Compare two datasets and calculate differences.
        
        Args:
            df_primary: Primary/base dataset
            df_comparison: Comparison dataset
            group_columns: Columns to group and merge by
            numeric_columns: Columns to compare
            date_primary: Label for primary data
            date_comparison: Label for comparison data
            
        Returns:
            Comparison DataFrame with difference and percentage columns
        """
        # Validate different dates
        if date_primary == date_comparison:
            st.warning("⚠️ Tanggal utama dan pembanding sama. Pilih tanggal berbeda.")
            return pd.DataFrame()
        
        # Find common columns
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
        
        # Aggregate both datasets
        agg_primary = DataAggregator.aggregate(
            df_primary, available_groups, available_numerics
        )
        agg_comparison = DataAggregator.aggregate(
            df_comparison, available_groups, available_numerics
        )
        
        if agg_primary.empty or agg_comparison.empty:
            st.warning("⚠️ Data agregasi kosong")
            return pd.DataFrame()
        
        # Rename columns with date suffix before merge
        rename_primary = {c: f"{c}_{date_primary}" for c in available_numerics}
        rename_comparison = {c: f"{c}_{date_comparison}" for c in available_numerics}
        
        agg_primary = agg_primary.rename(columns=rename_primary)
        agg_comparison = agg_comparison.rename(columns=rename_comparison)
        
        # Merge datasets
        comparison = pd.merge(
            agg_primary, agg_comparison,
            on=available_groups,
            how='outer'
        )
        
        # Calculate differences and percentages
        for col in available_numerics:
            col_primary = f"{col}_{date_primary}"
            col_comparison = f"{col}_{date_comparison}"
            
            if col_primary not in comparison.columns or col_comparison not in comparison.columns:
                continue
            
            # Difference: comparison - primary
            comparison[f"SELISIH_{col}"] = (
                comparison[col_comparison].fillna(0) - 
                comparison[col_primary].fillna(0)
            )
            
            # Percentage change
            comparison[f"PCT_{col}"] = np.where(
                comparison[col_primary].fillna(0) != 0,
                (comparison[f"SELISIH_{col}"] / comparison[col_primary].fillna(0).abs()) * 100,
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
            <p class="breadcrumb">KEMENTERIAN KEUANGAN RI • BIDJA</p>
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

    @st.cache_data(ttl=300, show_spinner=False)
    def _check_availability_cached_impl(
        base_url: str,
        filename_pattern: str,
        timeout: int,
        date_str: str
    ) -> Tuple[bool, Optional[int], Optional[str]]:
        """
        Cached availability check implementation.
        
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
        """
        Render date selection controls in sidebar.
        
        Returns:
            Tuple of (primary_date, comparison_date, enable_comparison,
                     primary_availability, comparison_availability)
        """
        st.sidebar.markdown("### 📅 Pilih Tanggal Data")
        
        # Primary date selection
        primary_dt = st.sidebar.date_input(
            "Tanggal Data Utama",
            value=date(2025, 10, 27),
            min_value=self.config.date_start,
            max_value=self.config.date_end,
            key=f"{key_prefix}primary_date",
            help="Pilih tanggal update data"
        )
        primary_date = primary_dt.strftime("%Y-%m-%d")
        
        # Check availability with caching
        primary_availability = self._check_availability_cached(primary_date)
        st.sidebar.markdown(
            UIComponents.render_availability_badge(primary_availability),
            unsafe_allow_html=True
        )
        
        st.sidebar.markdown("---")
        
        # Comparison toggle
        enable_comparison = st.sidebar.checkbox(
            "🔄 Bandingkan dengan tanggal lain",
            value=False,
            key=f"{key_prefix}enable_comparison"
        )
        
        comparison_date = None
        comparison_availability = None
        
        if enable_comparison:
            # Default to different date
            default_comparison = (
                date(2025, 11, 11) 
                if primary_dt != date(2025, 11, 11) 
                else date(2025, 10, 27)
            )
            
            comparison_dt = st.sidebar.date_input(
                "Tanggal Pembanding",
                value=default_comparison,
                min_value=self.config.date_start,
                max_value=self.config.date_end,
                key=f"{key_prefix}comparison_date"
            )
            comparison_date = comparison_dt.strftime("%Y-%m-%d")
            
            # Warn if same date
            if comparison_date == primary_date:
                st.sidebar.error("⚠️ Pilih tanggal yang berbeda!")
            
            comparison_availability = self._check_availability_cached(comparison_date)
            st.sidebar.markdown(
                UIComponents.render_availability_badge(comparison_availability),
                unsafe_allow_html=True
            )
        
        # Available dates reference
        st.sidebar.markdown("---")
        st.sidebar.markdown("**📋 Tanggal Tersedia:**")
        st.sidebar.markdown("\n".join(f"- {d}" for d in AVAILABLE_DATES))
        
        return (
            primary_date, 
            comparison_date, 
            enable_comparison,
            primary_availability,
            comparison_availability
        )
    
   def _check_availability_cached(self, date_str: str) -> DataAvailability:
        """Check availability with caching (returns DataAvailability)."""
        result = _check_availability_cached_impl(
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
    
    def render_aggregation_options(
        self, 
        key_prefix: str = ""
    ) -> Tuple[List[str], List[str], str]:
        """
        Render aggregation options in sidebar.
        
        Returns:
            Tuple of (group_columns, numeric_columns, aggregation_function)
        """
        st.sidebar.markdown("### 📊 Opsi Agregasi")
        
        # Group columns selection
        group_cols = st.sidebar.multiselect(
            "Group By (Kolom String)",
            options=list(self.column_config.STRING_COLUMNS),
            default=list(self.column_config.DEFAULT_GROUP_COLS),
            help="Pilih minimal 1 kolom untuk pengelompokan",
            key=f"{key_prefix}group_cols"
        )
        
        if len(group_cols) < 1:
            st.sidebar.error("⚠️ Pilih minimal 1 kolom string")
        
        # Numeric columns selection
        numeric_cols = st.sidebar.multiselect(
            "Agregasi (Kolom Numerik)",
            options=list(self.column_config.NUMERIC_COLUMNS),
            default=list(self.column_config.DEFAULT_NUMERIC_COLS),
            help="Pilih minimal 1 kolom untuk agregasi",
            key=f"{key_prefix}numeric_cols"
        )
        
        if len(numeric_cols) < 1:
            st.sidebar.error("⚠️ Pilih minimal 1 kolom numerik")
        
        # Aggregation function selection
        agg_func = st.sidebar.selectbox(
            "Fungsi Agregasi",
            options=[f.value for f in AggregationFunction],
            format_func=lambda x: AggregationFunction(x).display_name,
            key=f"{key_prefix}agg_func"
        )
        
        return group_cols, numeric_cols, agg_func
    
    def render_data_info(
        self, 
        df: pd.DataFrame, 
        label: str, 
        key_prefix: str = ""
    ) -> None:
        """Render data info expander in sidebar."""
        with st.sidebar.expander(f"📋 Info Data - {label}", expanded=False):
            col1, col2, col3 = st.columns(3)
            col1.metric("📊 Baris", Formatter.to_number_with_separator(len(df)))
            col2.metric("📋 Kolom", len(df.columns))
            
            if "Tahun" in df.columns:
                years = sorted(df["Tahun"].dropna().unique())
                col3.metric("📆 Tahun", ", ".join(map(str, years)))


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
        """
        Render filter controls for the given columns.
        
        Args:
            df: DataFrame to filter
            group_columns: Columns to create filters for
            key_prefix: Prefix for widget keys
            
        Returns:
            Dictionary mapping column names to selected filter values
        """
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
    def apply_filters(
        df: pd.DataFrame, 
        filters: Dict[str, List[str]]
    ) -> pd.DataFrame:
        """
        Apply filters to DataFrame.
        
        Args:
            df: DataFrame to filter
            filters: Dictionary of column -> selected values
            
        Returns:
            Filtered DataFrame
        """
        if df.empty or not filters:
            return df
        
        filtered_df = df.copy()
        
        for col, values in filters.items():
            if col in filtered_df.columns and values:
                filtered_df = filtered_df[
                    filtered_df[col].astype(str).isin(values)
                ]
        
        return filtered_df


# =============================================================================
# MAIN APPLICATION
# =============================================================================

class MonitoringDashboard:
    """
    Main application controller for the Budget Monitoring Dashboard.
    
    Orchestrates data loading, processing, and UI rendering.
    """
    
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
        
        # Render sidebar and get selections
        (
            primary_date, comparison_date, enable_comparison,
            primary_availability, comparison_availability
        ) = self.sidebar.render_date_selector()
        
        st.sidebar.divider()
        group_cols, numeric_cols, agg_func = self.sidebar.render_aggregation_options()
        
        # Check data availability
        if not primary_availability.is_available:
            UIComponents.render_data_unavailable_message()
            return
        
        # Show large file warning
        if (primary_availability.file_size and 
            primary_availability.file_size > self.config.large_file_threshold):
            size_str = Formatter.to_file_size(primary_availability.file_size)
            st.info(f"📦 File besar ({size_str}). Download mungkin memakan waktu...")
        
        # Load primary data
        progress_placeholder = st.empty()
        primary_dt = datetime.strptime(primary_date, "%Y-%m-%d").date()
        df_primary = self.data_loader.load_with_progress(primary_dt, progress_placeholder)
        
        if df_primary.empty:
            return
        
        # Show success message
        date_label = Formatter.to_indonesian_date(primary_dt)
        st.success(f"✅ Data dimuat: **{len(df_primary):,}** baris | {date_label}")
        
        # Render data info in sidebar
        self.sidebar.render_data_info(df_primary, date_label, "primary_")
        
        # Validate column selections
        if len(group_cols) < 1 or len(numeric_cols) < 1:
            st.info("👆 Pilih minimal 1 kolom string dan 1 kolom numerik di sidebar")
            with st.expander("📋 Preview Data Mentah"):
                st.dataframe(df_primary.head(100), width="stretch")
            return
        
        # Aggregate primary data
        agg_primary = DataAggregator.aggregate(df_primary, group_cols, numeric_cols, agg_func)
        formatter = DataFrameFormatter(numeric_cols)
        agg_primary_display = formatter.format_for_display(agg_primary)
        
        # Route to appropriate view
        if enable_comparison and comparison_date and comparison_availability and comparison_availability.is_available:
            self._render_comparison_view(
                df_primary, primary_date,
                comparison_date, comparison_availability,
                group_cols, numeric_cols, agg_func,
                agg_primary_display, formatter
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
        formatter: DataFrameFormatter
    ) -> None:
        """Render the date comparison view."""
        # Validate different dates
        if comparison_date == primary_date:
            st.error("⚠️ **Tanggal sama!** Pilih tanggal pembanding yang berbeda dari tanggal utama.")
            st.dataframe(agg_primary_display, width="stretch", hide_index=True)
            return
        
        # Load comparison data
        progress_placeholder = st.empty()
        comparison_dt = datetime.strptime(comparison_date, "%Y-%m-%d").date()
        df_comparison = self.data_loader.load_with_progress(comparison_dt, progress_placeholder)
        
        if df_comparison.empty:
            st.dataframe(agg_primary_display, width="stretch", hide_index=True)
            return
        
        # Show success and sidebar info
        comparison_label = Formatter.to_indonesian_date(comparison_dt)
        st.success(f"✅ Data pembanding dimuat: **{len(df_comparison):,}** baris")
        self.sidebar.render_data_info(df_comparison, comparison_label, "comparison_")
        
        # Compare datasets
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
        
        # Render summary metrics
        self._render_comparison_summary(
            df_primary, df_comparison,
            numeric_cols,
            primary_label, comparison_label
        )
        
        st.divider()
        
        # Render detailed comparison table
        self._render_comparison_detail(
            comparison_df, group_cols, numeric_cols,
            primary_date, comparison_date,
            primary_label, comparison_label,
            formatter
        )
    
    def _render_comparison_summary(
        self,
        df_primary: pd.DataFrame,
        df_comparison: pd.DataFrame,
        numeric_cols: List[str],
        primary_label: str,
        comparison_label: str
    ) -> None:
        """Render summary metrics for comparison."""
        st.markdown("### 📈 Ringkasan Perbandingan Antar Tanggal")
        
        for col in numeric_cols:
            if col not in df_primary.columns or col not in df_comparison.columns:
                continue
            
            st.markdown(f"**{col}**")
            c1, c2, c3 = st.columns(3)
            
            val_primary = df_primary[col].sum()
            val_comparison = df_comparison[col].sum()
            diff = val_comparison - val_primary
            pct = (diff / abs(val_primary) * 100) if val_primary != 0 else 0
            
            c1.metric(primary_label, Formatter.to_rupiah_short(val_primary))
            c2.metric(comparison_label, Formatter.to_rupiah_short(val_comparison))
            c3.metric("Selisih", Formatter.to_rupiah_short(diff), f"{pct:+.2f}%")
    
    def _render_comparison_detail(
        self,
        comparison_df: pd.DataFrame,
        group_cols: List[str],
        numeric_cols: List[str],
        primary_date: str,
        comparison_date: str,
        primary_label: str,
        comparison_label: str,
        formatter: DataFrameFormatter
    ) -> None:
        """Render detailed comparison table with filters."""
        st.markdown("### 🔄 Perbandingan Detail")
        UIComponents.render_comparison_header(primary_label, comparison_label)
        
        # Filters
        filters = self.filter_controller.render_filters(
            comparison_df, group_cols, key_prefix="comp_"
        )
        filtered_df = self.filter_controller.apply_filters(comparison_df, filters)
        
        if filters:
            st.info(f"Menampilkan **{len(filtered_df):,}** dari **{len(comparison_df):,}** baris")
        
        # Column legend
        with st.expander("ℹ️ Keterangan Kolom", expanded=False):
            st.markdown(f"""
            - `[kolom]_{primary_date}` = Nilai tanggal utama
            - `[kolom]_{comparison_date}` = Nilai tanggal pembanding
            - `SELISIH_[kolom]` = Selisih (pembanding - utama)
            - `PCT_[kolom]` = Persentase perubahan
            """)
        
        # Display table
        display_df = formatter.format_for_display(filtered_df)
        st.dataframe(display_df, width="stretch", hide_index=True)
        
        # Download button
        csv_data = filtered_df.to_csv(index=False)
        st.download_button(
            "📥 Download Perbandingan",
            csv_data,
            f"perbandingan_{primary_date}_vs_{comparison_date}.csv",
            "text/csv",
            key="dl_comparison"
        )
    
    def _render_single_date_view(
        self,
        df_primary: pd.DataFrame,
        primary_date: str,
        group_cols: List[str],
        numeric_cols: List[str],
        agg_primary_display: pd.DataFrame,
        formatter: DataFrameFormatter
    ) -> None:
        """Render single date view with optional column comparison."""
        primary_dt = datetime.strptime(primary_date, "%Y-%m-%d").date()
        date_label = Formatter.to_indonesian_date(primary_dt)
        
        # Show column comparison if multiple numeric columns selected
        if len(numeric_cols) > 1:
            self._render_column_summary(df_primary, numeric_cols)
            st.divider()
            self._render_column_comparison_table(df_primary, numeric_cols)
            st.divider()
        
        # Always show aggregated data table
        st.markdown(f"### 📋 Data Agregasi - {date_label}")
        st.dataframe(agg_primary_display, width="stretch", hide_index=True)
        
        # Download button
        # Need to get original (unformatted) aggregated data for download
        agg_primary = DataAggregator.aggregate(
            df_primary, group_cols, numeric_cols
        )
        csv_data = agg_primary.to_csv(index=False)
        st.download_button(
            "📥 Download CSV",
            csv_data,
            f"agregasi_{primary_date}.csv",
            "text/csv",
            key="dl_single"
        )
    
    def _render_column_summary(
        self, 
        df: pd.DataFrame, 
        numeric_cols: List[str]
    ) -> None:
        """Render summary metrics for numeric columns."""
        st.markdown("### 📈 Ringkasan Kolom Numerik")
        
        for i in range(0, len(numeric_cols), self.config.metrics_per_row):
            chunk = numeric_cols[i:i + self.config.metrics_per_row]
            cols = st.columns(len(chunk))
            
            for idx, col in enumerate(chunk):
                if col in df.columns:
                    value = df[col].sum()
                    cols[idx].metric(col, Formatter.to_rupiah_short(value))
    
    def _render_column_comparison_table(
        self, 
        df: pd.DataFrame, 
        numeric_cols: List[str]
    ) -> None:
        """Render comparison table for numeric columns."""
        st.markdown("### 📊 Perbandingan Antar Kolom Numerik")
        
        summary_data = []
        for col in numeric_cols:
            if col not in df.columns:
                continue
            
            summary_data.append({
                'Kolom': col,
                'Total': df[col].sum(),
                'Rata-rata': df[col].mean(),
                'Min': df[col].min(),
                'Max': df[col].max(),
                'Count': df[col].count()
            })
        
        if not summary_data:
            return
        
        summary_df = pd.DataFrame(summary_data)
        
        # Format for display
        display_df = summary_df.copy()
        for col in ['Total', 'Rata-rata', 'Min', 'Max']:
            display_df[col] = display_df[col].apply(Formatter.to_rupiah_full)
        display_df['Count'] = display_df['Count'].apply(Formatter.to_number_with_separator)
        
        st.dataframe(display_df, width="stretch", hide_index=True)


# =============================================================================
# ENTRY POINT
# =============================================================================

def main() -> None:
    """Application entry point."""
    dashboard = MonitoringDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()



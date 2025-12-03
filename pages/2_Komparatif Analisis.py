"""
Komparasi Realisasi vs Pagu DIPA Dashboard
==========================================
Dashboard interaktif untuk membandingkan realisasi belanja terhadap alokasi DIPA.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from openpyxl import Workbook


# =============================================================================
# ENUMS & CONSTANTS
# =============================================================================

class SortOrder(Enum):
    """Sort order for data display."""
    TOP = "Top"
    BOTTOM = "Bottom"
    
    @property
    def is_ascending(self) -> bool:
        """Returns True if sorting should be ascending (smallest first)."""
        return self == SortOrder.BOTTOM
    
    @property
    def display_label(self) -> str:
        """Human-readable label for UI."""
        return "tertinggi" if self == SortOrder.TOP else "terendah"


class BudgetColumn(Enum):
    """Standard budget column names in the dataset."""
    REALISASI = "REALISASI BELANJA KL (SAKTI)"
    PAGU_AWAL = "PAGU DIPA AWAL"
    PAGU_AWAL_EFEKTIF = "PAGU DIPA AWAL EFEKTIF"
    PAGU_REVISI = "PAGU DIPA REVISI"
    PAGU_REVISI_EFEKTIF = "PAGU DIPA REVISI EFEKTIF"
    BLOKIR_AWAL = "BLOKIR DIPA AWAL"
    BLOKIR_REVISI = "BLOKIR DIPA REVISI"
    KL = "KEMENTERIAN/LEMBAGA"
    TAHUN = "Tahun"


@dataclass
class ComparisonConfig:
    """Configuration for a budget comparison view."""
    col_start: str
    col_end: str
    title_suffix: str
    color_range: str
    color_marker: str
    caption_range: str
    caption_persen: str
    caption_varian: str


# Pre-defined comparison configurations for each tab
COMPARISON_CONFIGS = {
    "awal_revisi_efektif": ComparisonConfig(
        col_start=BudgetColumn.PAGU_AWAL_EFEKTIF.value,
        col_end=BudgetColumn.PAGU_REVISI_EFEKTIF.value,
        title_suffix="dengan Rentang Pagu DIPA Awal dan Revisi (Efektif)",
        color_range="#b2dfdb",
        color_marker="#00897b",
        caption_range="*Rentang merupakan _selisih_ antara Pagu Revisi Efektif dan Pagu Awal Efektif",
        caption_persen="**Persentase Realisasi Belanja *terhadap* Pagu DIPA Revisi Efektif",
        caption_varian="***Varian adalah Pagu Efektif *dikurangi* Realisasi Belanja"
    ),
    "awal_efektif": ComparisonConfig(
        col_start=BudgetColumn.PAGU_AWAL.value,
        col_end=BudgetColumn.PAGU_AWAL_EFEKTIF.value,
        title_suffix="dengan Rentang Pagu DIPA Awal dikurangi Blokir DIPA Awal",
        color_range="#c5cae9",
        color_marker="#1a73e8",
        caption_range="*Rentang merupakan besaran :red[Blokir] DIPA Awal",
        caption_persen="**Persentase Realisasi Belanja *terhadap* Pagu DIPA Awal Efektif",
        caption_varian="***Varian adalah Pagu Efektif *dikurangi* Realisasi Belanja"
    ),
    "revisi_efektif": ComparisonConfig(
        col_start=BudgetColumn.PAGU_REVISI.value,
        col_end=BudgetColumn.PAGU_REVISI_EFEKTIF.value,
        title_suffix="dengan Rentang Pagu DIPA Revisi dikurangi Blokir DIPA Revisi",
        color_range="#ffe082",
        color_marker="#e53935",
        caption_range="*Rentang merupakan besaran :red[Blokir] DIPA Revisi",
        caption_persen="**Persentase Realisasi Belanja *terhadap* Pagu DIPA Revisi Efektif",
        caption_varian="***Varian adalah Pagu Efektif *dikurangi* Realisasi Belanja"
    )
}

# Secondary chart colors for category breakdown
CATEGORY_CHART_COLORS = {
    "color_range": "#aed581",
    "color_marker": "#33691e"
}


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class AppConfig:
    """Application configuration constants."""
    
    # Data source
    DATA_URL: str = "https://raw.githubusercontent.com/tubankum3/dashanggaran/main/df.csv.zip"
    CSV_FILENAME: str = "df.csv"
    REQUEST_TIMEOUT: int = 30
    
    # UI defaults
    DEFAULT_YEAR: int = 2025
    DEFAULT_TOP_N: int = 10
    MIN_TOP_N: int = 1
    MAX_TOP_N: int = 50
    
    # Chart settings
    CHART_MIN_HEIGHT: int = 500
    CHART_HEIGHT_PER_ROW: int = 50
    CHART_TICK_COUNT: int = 6
    VARIANCE_CAP_BASE_SIZE: float = 0.1
    
    # Excluded entities
    EXCLUDED_KL: str = "999 BAGIAN ANGGARAN BENDAHARA UMUM NEGARA"
    
    # Page configuration
    PAGE_TITLE: str = "Analisis Komparasi Realisasi vs Pagu DIPA"
    PAGE_ICON: str = ":material/split_scene:"


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
    --radius-sm: 6px;
    --radius-md: 8px;
    --radius-lg: 12px;
    --radius-full: 9999px;
    --space-sm: 0.75rem;
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

.material-card {
    background: var(--surface);
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow-sm);
    padding: var(--space-xl);
    margin-bottom: var(--space-lg);
    border: 1px solid var(--gray-200);
    transition: all var(--transition-fast);
}

.material-card:hover {
    box-shadow: var(--shadow-md);
    transform: translateY(-2px);
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

::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: var(--gray-100); border-radius: var(--radius-full); }
::-webkit-scrollbar-thumb { background: var(--gray-400); border-radius: var(--radius-full); }
::-webkit-scrollbar-thumb:hover { background: var(--gray-500); }
</style>
"""


# =============================================================================
# FORMATTING UTILITIES
# =============================================================================

class Formatter:
    """Utility class for formatting values for display."""
    
    TRILLION = 1_000_000_000_000
    BILLION = 1_000_000_000
    MILLION = 1_000_000
    
    @classmethod
    def to_rupiah_short(cls, value: float) -> str:
        """
        Format numeric value to abbreviated Rupiah string.
        
        Args:
            value: Numeric value to format
            
        Returns:
            Formatted string like "Rp 1.23 T" or "Rp 500 M"
        """
        if pd.isna(value) or value == 0:
            return "Rp 0"
        
        abs_val = abs(value)
        sign = "-" if value < 0 else ""
        
        if abs_val >= cls.TRILLION:
            return f"{sign}Rp {abs_val / cls.TRILLION:.2f} T"
        elif abs_val >= cls.BILLION:
            return f"{sign}Rp {abs_val / cls.BILLION:.2f} M"
        elif abs_val >= cls.MILLION:
            return f"{sign}Rp {abs_val / cls.MILLION:.2f} Jt"
        return f"{sign}Rp {abs_val:,.0f}"
    
    @classmethod
    def to_rupiah_full(cls, value: Any) -> str:
        """
        Format numeric value to full Rupiah string with dot separators.
        
        Args:
            value: Value to format
            
        Returns:
            Formatted string like "Rp 1.234.567.890"
        """
        try:
            value = float(value)
        except (ValueError, TypeError):
            return str(value)
        
        if pd.isna(value):
            return "-"
        
        return f"Rp {value:,.0f}".replace(",", ".")
    
    @staticmethod
    def to_percentage(value: float, decimals: int = 1) -> str:
        """Format value as percentage string, handling NaN."""
        if pd.isna(value):
            return "-"
        return f"{value:.{decimals}f}%"


# =============================================================================
# DATA LAYER
# =============================================================================

class DataLoadError(Exception):
    """Custom exception for data loading failures."""
    pass


class DataLoader:
    """Handles loading and caching of budget data."""
    
    def __init__(self, config: AppConfig = AppConfig()):
        self.config = config
    
    @staticmethod
    @st.cache_data(show_spinner="Memuat dataset anggaran...")
    def load_data(url: str, csv_filename: str, timeout: int) -> pd.DataFrame:
        """
        Load and cache budget data from remote ZIP file.
        
        Args:
            url: URL to the ZIP file containing CSV data
            csv_filename: Name of the CSV file inside the ZIP
            timeout: Request timeout in seconds
            
        Returns:
            DataFrame with budget data
            
        Raises:
            DataLoadError: If data cannot be loaded
        """
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                with z.open(csv_filename) as file:
                    df = pd.read_csv(file, low_memory=False)
            
            # Clean up unnamed columns
            unnamed_cols = [c for c in df.columns if "Unnamed" in str(c)]
            if unnamed_cols:
                df = df.drop(columns=unnamed_cols)
            
            # Ensure Tahun is string for consistent filtering
            if BudgetColumn.TAHUN.value in df.columns:
                df[BudgetColumn.TAHUN.value] = df[BudgetColumn.TAHUN.value].astype(str)
            
            return df
            
        except requests.exceptions.RequestException as e:
            raise DataLoadError(f"Network error: {e}")
        except zipfile.BadZipFile as e:
            raise DataLoadError(f"Invalid ZIP file: {e}")
        except Exception as e:
            raise DataLoadError(f"Failed to load data: {e}")
    
    def get_data(self) -> pd.DataFrame:
        """Load data using configured settings."""
        return self.load_data(
            self.config.DATA_URL,
            self.config.CSV_FILENAME,
            self.config.REQUEST_TIMEOUT
        )


# =============================================================================
# DATA AGGREGATION & TABLE GENERATION
# =============================================================================

@dataclass
class AggregatedData:
    """Container for raw and display-formatted aggregated data."""
    raw: pd.DataFrame
    display: pd.DataFrame


class DataAggregator:
    """Handles data aggregation and table generation."""
    
    def __init__(self, config: AppConfig = AppConfig()):
        self.config = config
    
    def aggregate_for_chart(
        self,
        df: pd.DataFrame,
        year: int,
        group_col: str,
        col_start: str,
        col_end: str,
        top_n: int,
        sort_order: SortOrder,
        selected_kls: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Aggregate data for chart display.
        
        Args:
            df: Source DataFrame
            year: Year to filter
            group_col: Column to group by
            col_start: Start range column (e.g., Pagu Awal)
            col_end: End range column (e.g., Pagu Revisi)
            top_n: Number of rows to return
            sort_order: Top or Bottom N
            selected_kls: Optional K/L filter
            
        Returns:
            Aggregated DataFrame with variance calculations
        """
        # Filter by year
        df_filtered = df[df[BudgetColumn.TAHUN.value].astype(int) == year].copy()
        
        # Exclude special K/L
        df_filtered = df_filtered[
            df_filtered[BudgetColumn.KL.value] != self.config.EXCLUDED_KL
        ]
        
        # Apply K/L filter if specified
        if selected_kls:
            df_filtered = df_filtered[
                df_filtered[BudgetColumn.KL.value].isin(selected_kls)
            ]
        
        if df_filtered.empty:
            return pd.DataFrame()
        
        # Aggregate
        realisasi_col = BudgetColumn.REALISASI.value
        agg_cols = [realisasi_col, col_start, col_end]
        available_cols = [c for c in agg_cols if c in df_filtered.columns]
        
        agg = df_filtered.groupby(group_col, as_index=False)[available_cols].sum()
        
        # Sort and limit
        agg = agg.sort_values(
            realisasi_col, 
            ascending=sort_order.is_ascending
        ).head(top_n).reset_index(drop=True)
        
        # Calculate derived columns
        agg["VARIANS"] = agg[col_end] - agg[realisasi_col]
        agg["PERSEN_REALISASI"] = np.where(
            agg[col_end] == 0,
            np.nan,
            (agg[realisasi_col] / agg[col_end]) * 100
        )
        
        return agg
    
    def generate_table(
        self,
        df: pd.DataFrame,
        year: int,
        group_col: str,
        col_start: str,
        col_end: str,
        selected_kls: Optional[List[str]] = None
    ) -> AggregatedData:
        """
        Generate display and raw tables for data export.
        
        Args:
            df: Source DataFrame
            year: Year to filter
            group_col: Column to group by
            col_start: Start column
            col_end: End column
            selected_kls: Optional K/L filter
            
        Returns:
            AggregatedData with raw and display DataFrames
        """
        # Filter by year
        df_year = df[df[BudgetColumn.TAHUN.value].astype(int) == year].copy()
        
        # Apply K/L filter
        if selected_kls:
            df_year = df_year[df_year[BudgetColumn.KL.value].isin(selected_kls)]
        
        realisasi_col = BudgetColumn.REALISASI.value
        
        # Aggregate
        agg = df_year.groupby(group_col, as_index=False)[
            [realisasi_col, col_start, col_end]
        ].sum()
        
        # Calculate derived columns
        agg["VARIANS"] = agg[col_end] - agg[realisasi_col]
        agg["PERSEN_REALISASI"] = np.where(
            agg[col_end] == 0,
            np.nan,
            (agg[realisasi_col] / agg[col_end]) * 100
        )
        
        # Create display version
        display_df = agg.copy()
        
        # Format currency columns
        for col in [realisasi_col, col_start, col_end, "VARIANS"]:
            display_df[col] = display_df[col].apply(Formatter.to_rupiah_full)
        
        # Format percentage
        display_df["PERSEN_REALISASI"] = display_df["PERSEN_REALISASI"].apply(
            Formatter.to_percentage
        )
        
        # Rename columns for display
        display_df = display_df.rename(columns={
            realisasi_col: "Realisasi Belanja (SAKTI) [A]",
            col_start: f"{col_start} [B]",
            col_end: f"{col_end} [C]",
            "VARIANS": "Varians [C - A]",
            "PERSEN_REALISASI": "% Realisasi [A/C]"
        })
        
        return AggregatedData(raw=agg, display=display_df)


# =============================================================================
# EXCEL EXPORT
# =============================================================================

class ExcelExporter:
    """Handles Excel file export functionality."""
    
    @staticmethod
    def export_dataframe(df: pd.DataFrame, sheet_name: str = "Data") -> bytes:
        """
        Export DataFrame to Excel bytes.
        
        Args:
            df: DataFrame to export
            sheet_name: Name for the Excel sheet
            
        Returns:
            Excel file as bytes
        """
        output = io.BytesIO()
        wb = Workbook()
        ws = wb.active
        ws.title = sheet_name
        
        # Write headers
        ws.append(list(df.columns))
        
        # Write data rows
        for row in df.values:
            ws.append(row.tolist())
        
        wb.save(output)
        output.seek(0)
        return output.getvalue()


# =============================================================================
# CHART BUILDER
# =============================================================================

class ComparisonChartBuilder:
    """Builds comparison charts showing Pagu ranges and Realisasi markers."""
    
    def __init__(self, config: AppConfig = AppConfig()):
        self.config = config
    
    def create_comparison_chart(
        self,
        agg: pd.DataFrame,
        group_col: str,
        col_start: str,
        col_end: str,
        title: str,
        color_range: str,
        color_marker: str
    ) -> go.Figure:
        """
        Create a comparison chart with range bars, variance lines, and markers.
        
        Args:
            agg: Aggregated DataFrame with VARIANS and PERSEN_REALISASI columns
            group_col: Column for y-axis labels
            col_start: Start range column
            col_end: End range column
            title: Chart title
            color_range: Color for range bars
            color_marker: Color for realisasi markers
            
        Returns:
            Plotly Figure object
        """
        if agg.empty:
            return self._create_empty_chart("Tidak ada data untuk ditampilkan")
        
        realisasi_col = BudgetColumn.REALISASI.value
        
        # Numeric y positions for consistent line plotting
        y_pos = np.arange(len(agg))
        
        fig = go.Figure()
        
        # Add range bars
        self._add_range_bars(fig, agg, y_pos, col_start, col_end, color_range)
        
        # Add variance lines with caps
        self._add_variance_lines(fig, agg, y_pos, realisasi_col, col_end)
        
        # Add realisasi markers
        self._add_realisasi_markers(fig, agg, y_pos, realisasi_col, color_marker)
        
        # Configure layout
        self._apply_chart_layout(fig, agg, y_pos, group_col, col_end, realisasi_col, title)
        
        return fig
    
    def _add_range_bars(
        self,
        fig: go.Figure,
        agg: pd.DataFrame,
        y_pos: np.ndarray,
        col_start: str,
        col_end: str,
        color: str
    ) -> None:
        """Add horizontal range bars showing Pagu range."""
        # Extract short names for legend
        start_short = " ".join(col_start.split()[-3:])
        end_short = " ".join(col_end.split()[-3:])
        
        fig.add_trace(go.Bar(
            y=y_pos,
            x=(agg[col_end] - agg[col_start]),
            base=agg[col_start],
            orientation="h",
            width=0.6,
            marker=dict(
                color=color,
                cornerradius=15,
                line=dict(color=color, width=0.5)
            ),
            name=f"Rentang {start_short}–{end_short}",
            hovertemplate=(
                f"{col_start}: %{{base:,.0f}}<br>"
                f"{col_end}: %{{customdata:,.0f}}<extra></extra>"
            ),
            customdata=agg[col_end]
        ))
    
    def _add_variance_lines(
        self,
        fig: go.Figure,
        agg: pd.DataFrame,
        y_pos: np.ndarray,
        realisasi_col: str,
        col_end: str
    ) -> None:
        """Add variance lines with end caps between Realisasi and Pagu."""
        # Dynamic cap size based on number of rows
        cap_size = max(0.05, self.config.VARIANCE_CAP_BASE_SIZE / max(1, len(agg) / 10))
        
        for i, row in agg.iterrows():
            x_real = row[realisasi_col]
            x_pagu = row[col_end]
            y = y_pos[i]
            
            # Color: black if underspend, red if overspend
            var_color = "black" if x_real < x_pagu else "red"
            
            # Main horizontal variance line
            fig.add_trace(go.Scatter(
                x=[x_real, x_pagu],
                y=[y, y],
                mode="lines",
                line=dict(color=var_color, width=1),
                showlegend=False,
                hoverinfo="skip"
            ))
            
            # Vertical end caps
            for x_cap in [x_real, x_pagu]:
                fig.add_trace(go.Scatter(
                    x=[x_cap, x_cap],
                    y=[y + cap_size, y - cap_size],
                    mode="lines",
                    line=dict(color=var_color, width=1),
                    showlegend=False,
                    hoverinfo="skip"
                ))
    
    def _add_realisasi_markers(
        self,
        fig: go.Figure,
        agg: pd.DataFrame,
        y_pos: np.ndarray,
        realisasi_col: str,
        color: str
    ) -> None:
        """Add markers showing Realisasi values."""
        fig.add_trace(go.Scatter(
            y=y_pos,
            x=agg[realisasi_col],
            mode="markers",
            marker=dict(
                color=color,
                size=12,
                line=dict(color="white", width=1)
            ),
            name="Realisasi Belanja (SAKTI)",
            hovertemplate=(
                "Realisasi: %{x:,.0f} "
                "(%{customdata[1]:.1f}%)<br>"
                "Varian (Pagu Efektif-Realisasi): %{customdata[0]:,.0f}<extra></extra>"
            ),
            customdata=np.stack((agg["VARIANS"], agg["PERSEN_REALISASI"]), axis=-1)
        ))
    
    def _apply_chart_layout(
        self,
        fig: go.Figure,
        agg: pd.DataFrame,
        y_pos: np.ndarray,
        group_col: str,
        col_end: str,
        realisasi_col: str,
        title: str
    ) -> None:
        """Apply layout configuration to the chart."""
        # Calculate x-axis ticks
        x_max = max(agg[col_end].max(), agg[realisasi_col].max())
        tickvals = np.linspace(0, x_max, num=self.config.CHART_TICK_COUNT)
        ticktext = [Formatter.to_rupiah_short(val) for val in tickvals]
        
        # Y-axis labels
        y_ticktext = agg[group_col].astype(str).tolist()
        
        # Dynamic height
        height = max(
            self.config.CHART_MIN_HEIGHT,
            self.config.CHART_HEIGHT_PER_ROW * len(agg)
        )
        
        fig.update_layout(
            title=title,
            xaxis_title="Jumlah (Rupiah)",
            yaxis_title=group_col,
            template="plotly_white",
            height=height,
            xaxis=dict(
                showgrid=True,
                zeroline=False,
                tickvals=tickvals,
                ticktext=ticktext
            ),
            yaxis=dict(
                tickmode="array",
                tickvals=list(y_pos),
                ticktext=y_ticktext,
                autorange="reversed"
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=250, r=40, t=100, b=40)
        )
    
    def _create_empty_chart(self, message: str) -> go.Figure:
        """Create an empty chart with a message."""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color="#666")
        )
        fig.update_layout(
            height=300,
            plot_bgcolor="white",
            paper_bgcolor="white"
        )
        return fig


# =============================================================================
# UI COMPONENTS
# =============================================================================

class UIComponents:
    """Reusable UI component generators."""
    
    @staticmethod
    def render_header(
        year: Optional[str] = None,
        metric: Optional[str] = None,
        kl_list: Optional[List[str]] = None
    ) -> None:
        """Render the main dashboard header."""
        year_text = year or "Overview"
        metric_text = f" {metric}" if metric else ""
        kl_text = ", ".join(kl_list) if kl_list else "SELURUH K/L"
        
        st.markdown(f"""
        <div class="dashboard-header" role="banner">
            <div class="breadcrumb">DASHBOARD / KOMPARASI{metric_text} / {kl_text} / TAHUN {year_text}</div>
            <h1 class="dashboard-title">Analisis Komparasi Realisasi vs Pagu DIPA</h1>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_captions(config: ComparisonConfig) -> None:
        """Render explanation captions for a comparison view."""
        st.caption(config.caption_range)
        st.caption(config.caption_persen)
        st.caption(config.caption_varian)
    
    @staticmethod
    def apply_styles() -> None:
        """Apply CSS styles to the page."""
        st.markdown(CSS_STYLES, unsafe_allow_html=True)


class SidebarController:
    """Manages sidebar filter controls."""
    
    def __init__(self, df: pd.DataFrame, config: AppConfig = AppConfig()):
        self.df = df
        self.config = config
    
    def render(self) -> Tuple[int, List[str], int, str, SortOrder]:
        """
        Render sidebar controls and return selected values.
        
        Returns:
            Tuple of (year, kl_list, top_n, category_metric, sort_order)
        """
        with st.sidebar:
            year = self._render_year_selector()
            sort_order = self._render_sort_order_selector()
            top_n = self._render_top_n_input(sort_order)
            category = self._render_category_selector()
            kl_list = self._render_kl_selector()
            
            return year, kl_list, top_n, category, sort_order
    
    def _render_year_selector(self) -> int:
        """Render year selection dropdown."""
        years = sorted(
            self.df[BudgetColumn.TAHUN.value].astype(int).unique()
        )
        
        default_idx = (
            years.index(self.config.DEFAULT_YEAR)
            if self.config.DEFAULT_YEAR in years
            else len(years) - 1
        )
        
        return st.selectbox("Pilih Tahun", options=years, index=default_idx)
    
    def _render_sort_order_selector(self) -> SortOrder:
        """Render Top/Bottom radio selector."""
        selection = st.radio(
            "Tampilkan Data",
            options=[SortOrder.TOP.value, SortOrder.BOTTOM.value],
            index=0,
            horizontal=True,
            help="Top: Data tertinggi | Bottom: Data terendah"
        )
        return SortOrder(selection)
    
    def _render_top_n_input(self, sort_order: SortOrder) -> int:
        """Render top N input field."""
        return st.number_input(
            f"Tampilkan {sort_order.value}-N Data",
            min_value=self.config.MIN_TOP_N,
            max_value=self.config.MAX_TOP_N,
            value=self.config.DEFAULT_TOP_N,
            step=1,
            help=f"Jumlah data {sort_order.display_label} berdasarkan Realisasi Belanja"
        )
    
    def _render_category_selector(self) -> str:
        """Render category/classification selector."""
        # Get object columns excluding Tahun
        category_cols = [
            col for col in self.df.select_dtypes(include="object").columns
            if col != BudgetColumn.TAHUN.value
        ]
        
        default_idx = (
            category_cols.index(BudgetColumn.KL.value)
            if BudgetColumn.KL.value in category_cols
            else 0
        )
        
        return st.selectbox(
            "Kategori/Klasifikasi",
            options=category_cols,
            index=default_idx
        )
    
    def _render_kl_selector(self) -> List[str]:
        """Render K/L multiselect."""
        kl_list = sorted(
            self.df[BudgetColumn.KL.value].dropna().unique()
        )
        
        return st.multiselect(
            "Pilih Kementerian/Lembaga (bisa lebih dari satu)",
            options=kl_list,
            default=[]
        )


# =============================================================================
# TAB VIEW RENDERER
# =============================================================================

class ComparisonTabView:
    """Renders a single comparison tab with charts and table."""
    
    def __init__(
        self,
        df: pd.DataFrame,
        year: int,
        top_n: int,
        sort_order: SortOrder,
        selected_kls: List[str],
        selected_metric: str,
        comparison_config: ComparisonConfig,
        config: AppConfig = AppConfig()
    ):
        self.df = df
        self.year = year
        self.top_n = top_n
        self.sort_order = sort_order
        self.selected_kls = selected_kls
        self.selected_metric = selected_metric
        self.comparison_config = comparison_config
        self.config = config
        
        self.aggregator = DataAggregator(config)
        self.chart_builder = ComparisonChartBuilder(config)
    
    def render(self, tab_key: str) -> None:
        """
        Render the complete tab view.
        
        Args:
            tab_key: Unique key for download button
        """
        cfg = self.comparison_config
        
        # Main K/L comparison chart
        self._render_kl_chart(cfg)
        
        # Category breakdown chart (if not K/L)
        if self.selected_metric != BudgetColumn.KL.value:
            self._render_category_chart(cfg)
        
        # Captions
        UIComponents.render_captions(cfg)
        
        # Detail table
        self._render_detail_table(cfg, tab_key)
    
    def _render_kl_chart(self, cfg: ComparisonConfig) -> None:
        """Render the main K/L comparison chart."""
        # Aggregate data
        agg = self.aggregator.aggregate_for_chart(
            df=self.df,
            year=self.year,
            group_col=BudgetColumn.KL.value,
            col_start=cfg.col_start,
            col_end=cfg.col_end,
            top_n=self.top_n,
            sort_order=self.sort_order,
            selected_kls=self.selected_kls if self.selected_kls else None
        )
        
        if agg.empty:
            st.warning(f"Tidak ada data untuk tahun {self.year}")
            return
        
        # Build title
        title = (
            f"Perbandingan Realisasi Belanja {cfg.title_suffix}<br>"
            f"Tahun {self.year}"
        )
        
        fig = self.chart_builder.create_comparison_chart(
            agg=agg,
            group_col=BudgetColumn.KL.value,
            col_start=cfg.col_start,
            col_end=cfg.col_end,
            title=title,
            color_range=cfg.color_range,
            color_marker=cfg.color_marker
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_category_chart(self, cfg: ComparisonConfig) -> None:
        """Render the category breakdown chart."""
        agg = self.aggregator.aggregate_for_chart(
            df=self.df,
            year=self.year,
            group_col=self.selected_metric,
            col_start=cfg.col_start,
            col_end=cfg.col_end,
            top_n=self.top_n,
            sort_order=self.sort_order,
            selected_kls=self.selected_kls if self.selected_kls else None
        )
        
        if agg.empty:
            st.warning(f"Tidak ada data untuk kategori '{self.selected_metric}'")
            return
        
        # Build title
        kl_text = "K/L Terpilih" if self.selected_kls else "Seluruh K/L"
        title = (
            f"Perbandingan Realisasi Belanja berdasarkan {self.selected_metric}<br>"
            f"Tahun {self.year} untuk {kl_text}"
        )
        
        fig = self.chart_builder.create_comparison_chart(
            agg=agg,
            group_col=self.selected_metric,
            col_start=cfg.col_start,
            col_end=cfg.col_end,
            title=title,
            color_range=CATEGORY_CHART_COLORS["color_range"],
            color_marker=CATEGORY_CHART_COLORS["color_marker"]
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_detail_table(self, cfg: ComparisonConfig, tab_key: str) -> None:
        """Render the expandable detail table with Excel export."""
        with st.expander("Tabel Rincian Data"):
            table_data = self.aggregator.generate_table(
                df=self.df,
                year=self.year,
                group_col=self.selected_metric,
                col_start=cfg.col_start,
                col_end=cfg.col_end,
                selected_kls=self.selected_kls if self.selected_kls else None
            )
            
            st.dataframe(
                table_data.display,
                use_container_width=True,
                hide_index=True
            )
            
            # Excel download
            excel_data = ExcelExporter.export_dataframe(table_data.raw)
            st.download_button(
                label="📥 Download Excel",
                data=excel_data,
                file_name=f"Tabel_Realisasi_vs_Pagu_{self.year}_{tab_key}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"download_{tab_key}_{self.year}_{self.selected_metric}"
            )


# =============================================================================
# MAIN APPLICATION
# =============================================================================

class BudgetComparisonApp:
    """Main application controller."""
    
    def __init__(self):
        self.config = AppConfig()
        self.data_loader = DataLoader(self.config)
    
    def run(self) -> None:
        """Run the application."""
        self._configure_page()
        UIComponents.apply_styles()
        
        # Load data
        try:
            df = self.data_loader.get_data()
        except DataLoadError as e:
            st.error(f"Gagal memuat data: {e}")
            return
        
        if df.empty:
            st.error("Data gagal dimuat.")
            return
        
        # Render sidebar and get selections
        sidebar = SidebarController(df, self.config)
        year, kl_list, top_n, category, sort_order = sidebar.render()
        
        # Render header
        UIComponents.render_header(str(year), category, kl_list)
        
        # Filter data by K/L if selected
        df_filtered = df.copy()
        if kl_list:
            df_filtered = df_filtered[
                df_filtered[BudgetColumn.KL.value].isin(kl_list)
            ]
        
        # Render tabs
        self._render_tabs(df_filtered, year, top_n, sort_order, kl_list, category)
    
    def _configure_page(self) -> None:
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title=self.config.PAGE_TITLE,
            page_icon=self.config.PAGE_ICON,
            layout="wide",
            initial_sidebar_state="expanded",
            menu_items={
                "Get Help": "https://www.kemenkeu.go.id",
                "Report a bug": "https://github.com/tubankum3/dashanggaran/issues",
                "About": "Dashboard Anggaran Bidang PMK"
            }
        )
    
    def _render_tabs(
        self,
        df: pd.DataFrame,
        year: int,
        top_n: int,
        sort_order: SortOrder,
        kl_list: List[str],
        category: str
    ) -> None:
        """Render the comparison tabs."""
        tab_labels = [
            "1️⃣ Realisasi vs Pagu DIPA Awal dan Revisi (Efektif)",
            "2️⃣ Realisasi vs Pagu DIPA Awal Efektif",
            "3️⃣ Realisasi vs Pagu DIPA Revisi Efektif"
        ]
        
        tab_configs = [
            ("awal_revisi_efektif", "tab1"),
            ("awal_efektif", "tab2"),
            ("revisi_efektif", "tab3")
        ]
        
        tabs = st.tabs(tab_labels)
        
        for tab, (config_key, tab_key) in zip(tabs, tab_configs):
            with tab:
                view = ComparisonTabView(
                    df=df,
                    year=year,
                    top_n=top_n,
                    sort_order=sort_order,
                    selected_kls=kl_list,
                    selected_metric=category,
                    comparison_config=COMPARISON_CONFIGS[config_key],
                    config=self.config
                )
                view.render(tab_key)


# =============================================================================
# ENTRY POINT
# =============================================================================

def main() -> None:
    """Application entry point."""
    try:
        app = BudgetComparisonApp()
        app.run()
    except Exception as e:
        st.error(f"Terjadi kesalahan dalam aplikasi: {str(e)}")
        st.info("Silakan refresh halaman atau hubungi administrator.")


if __name__ == "__main__":
    main()

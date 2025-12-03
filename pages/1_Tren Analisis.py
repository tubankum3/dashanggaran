"""
Tren Anggaran Dashboard
=======================
Dashboard interaktif untuk menganalisis tren anggaran Pemerintah.
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
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st


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
    
    # Chart settings
    CHART_BASE_HEIGHT: int = 400
    CHART_HEIGHT_PER_EXTRA_LINE: int = 3
    CHART_GROUP_THRESHOLD: int = 10
    
    # Required columns
    YEAR_COLUMN: str = "Tahun"
    KL_COLUMN: str = "KEMENTERIAN/LEMBAGA"
    
    # Page configuration
    PAGE_TITLE: str = "Analisis Tren Anggaran dan Realisasi Belanja Negara"
    PAGE_ICON: str = ":material/line_axis:"


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
    --on-surface: #111827;
    --on-primary: #FFFFFF;
    --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    --radius-sm: 6px;
    --radius-md: 8px;
    --radius-lg: 12px;
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

.breadcrumb { font-size: 0.875rem; font-weight: 500; opacity: 0.9; margin-bottom: var(--space-sm); letter-spacing: 0.025em; }
.dashboard-title { font-weight: 700; font-size: 2rem; line-height: 1.2; margin: 0; letter-spacing: -0.025em; }
.dashboard-subtitle { font-weight: 400; font-size: 1rem; opacity: 0.9; margin: 0.75rem 0 0 0; }

.section-title { font-weight: 600; font-size: 1.125rem; color: var(--gray-900); margin-bottom: var(--space-lg); padding-bottom: var(--space-md); border-bottom: 2px solid var(--gray-200); }

.material-card { background: var(--surface); border-radius: var(--radius-lg); box-shadow: var(--shadow-sm); padding: var(--space-xl); margin-bottom: var(--space-lg); border: 1px solid var(--gray-200); transition: all var(--transition-base); }
.material-card:hover { box-shadow: var(--shadow-md); transform: translateY(-2px); }

.chart-container { background: var(--surface); border-radius: var(--radius-lg); padding: var(--space-lg); box-shadow: var(--shadow-sm); border: 1px solid var(--gray-200); }

.metric-card { background: var(--surface); border-radius: var(--radius-lg); padding: var(--space-lg); box-shadow: var(--shadow-sm); transition: all var(--transition-base); border: 1px solid var(--gray-200); position: relative; overflow: hidden; }
.metric-card:hover { box-shadow: var(--shadow-md); transform: translateY(-2px); border-color: var(--primary-light); }
.metric-value { font-size: 2rem; font-weight: 700; color: var(--gray-900); margin: var(--space-sm) 0; line-height: 1; }
.metric-label { font-size: 0.875rem; color: var(--gray-600); font-weight: 500; }
.metric-sublabel { font-size: 0.75rem; color: var(--gray-500); margin-top: var(--space-xs); }
.metric-trend { display: inline-flex; align-items: center; padding: 0.25rem 0.75rem; border-radius: var(--radius-full); font-size: 0.75rem; font-weight: 600; margin-top: var(--space-sm); }
.trend-positive { background: #ECFDF5; color: var(--success); }
.trend-negative { background: #FEF2F2; color: var(--error); }

.availability-badge { display: inline-flex; align-items: center; gap: 6px; padding: 6px 14px; border-radius: var(--radius-full); font-size: 13px; font-weight: 500; margin: 8px 0; }
.badge-available { background-color: #ECFDF5; color: #059669; border: 1px solid #A7F3D0; }
.badge-unavailable { background-color: #FEF2F2; color: #DC2626; border: 1px solid #FECACA; }

.comparison-header { background: var(--gray-100); padding: var(--space-md); border-radius: var(--radius-md); margin-bottom: var(--space-md); }
.filter-container { background: var(--gray-50); padding: var(--space-md); border-radius: var(--radius-md); margin-bottom: var(--space-lg); border: 1px solid var(--gray-200); }

.stButton > button { background: var(--primary); color: var(--on-primary); border: none; border-radius: var(--radius-md); padding: 0.625rem 1.25rem; font-weight: 500; font-size: 0.875rem; transition: all var(--transition-fast); box-shadow: var(--shadow-sm); }
.stButton > button:hover { background: var(--primary-dark); box-shadow: var(--shadow-md); transform: translateY(-1px); }

.stSidebar { background: var(--surface); border-right: 1px solid var(--gray-200); }
.sidebar-section { background: var(--surface); border-radius: var(--radius-md); padding: var(--space-md); margin-bottom: var(--space-md); border: 1px solid var(--gray-200); }

.stTabs [data-baseweb="tab-list"] { display: flex; flex-wrap: wrap; gap: var(--space-xs); border-bottom: 1px solid var(--gray-200); padding-bottom: var(--space-sm); row-gap: var(--space-sm); }
.stTabs [data-baseweb="tab"] { background: var(--gray-50); border: 1px solid var(--gray-200); border-bottom: 2px solid transparent; border-radius: var(--radius-md); padding: var(--space-sm) var(--space-md); color: var(--gray-600); font-weight: 500; font-size: 0.8rem; transition: all var(--transition-fast); white-space: nowrap; flex-shrink: 0; }
.stTabs [data-baseweb="tab"]:hover { color: var(--gray-900); background: var(--gray-100); border-color: var(--gray-300); }
.stTabs [aria-selected="true"] { background: var(--primary-light); color: white; border-color: var(--primary); border-bottom-color: var(--primary); }
.stTabs [data-baseweb="tab-panel"] { padding-top: var(--space-lg); }

::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: var(--gray-100); border-radius: var(--radius-full); }
::-webkit-scrollbar-thumb { background: var(--gray-400); border-radius: var(--radius-full); }
::-webkit-scrollbar-thumb:hover { background: var(--gray-500); }
</style>
"""

CSS_TABS_UNDERLINE = """
<style>
.stTabs [data-baseweb="tab-list"] { display: flex; flex-wrap: nowrap; gap: var(--space-sm); border-bottom: 1px solid var(--gray-200); padding-bottom: 0; }
.stTabs [data-baseweb="tab"] { background: transparent; border: none; border-bottom: 2px solid transparent; border-radius: 0; padding: var(--space-md) var(--space-lg); color: var(--gray-600); font-weight: 500; font-size: 0.875rem; }
.stTabs [data-baseweb="tab"]:hover { color: var(--gray-900); background: var(--gray-50); }
.stTabs [aria-selected="true"] { background: transparent; color: var(--primary); border-bottom-color: var(--primary); }
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
        """Format numeric value to abbreviated Rupiah string."""
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
        """Format numeric value to full Rupiah string with dot separators."""
        try:
            value = float(value)
        except (ValueError, TypeError):
            return str(value)
        
        if pd.isna(value):
            return "-"
        
        return f"Rp {value:,.0f}".replace(",", ".")


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
        """Load and cache budget data from remote ZIP file."""
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
            
            # Ensure Tahun is string for consistent handling
            if "Tahun" in df.columns:
                df["Tahun"] = df["Tahun"].astype(str)
            
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
# METRICS CALCULATOR
# =============================================================================

@dataclass
class FinancialMetrics:
    """Container for calculated financial metrics."""
    yearly_totals: pd.DataFrame
    aagr: float = 0.0
    cagr: float = 0.0
    latest_growth: float = 0.0
    last_tahun: str = ""
    
    @property
    def has_data(self) -> bool:
        """Check if metrics contain valid data."""
        return not self.yearly_totals.empty


class MetricsCalculator:
    """Calculates financial metrics from budget data."""
    
    @staticmethod
    def calculate(df: pd.DataFrame, value_col: str = "Nilai") -> FinancialMetrics:
        """
        Calculate comprehensive financial metrics.
        
        Args:
            df: DataFrame with Tahun and value columns
            value_col: Name of the value column
            
        Returns:
            FinancialMetrics object with calculated values
        """
        if df.empty or value_col not in df.columns:
            return FinancialMetrics(yearly_totals=pd.DataFrame())
        
        # Sort and aggregate by year
        df_sorted = df.sort_values("Tahun")
        yearly_sums = df_sorted.groupby("Tahun", as_index=False)[value_col].sum()
        
        metrics = FinancialMetrics(yearly_totals=yearly_sums)
        
        if len(yearly_sums) > 1:
            first_value = yearly_sums[value_col].iloc[0]
            last_value = yearly_sums[value_col].iloc[-1]
            n_years = len(yearly_sums) - 1
            
            # Calculate YoY growth
            yearly_sums["YoY_Growth"] = yearly_sums[value_col].pct_change() * 100
            
            # AAGR: Average Annual Growth Rate
            metrics.aagr = yearly_sums["YoY_Growth"].mean(skipna=True)
            
            # CAGR: Compound Annual Growth Rate
            if first_value > 0:
                metrics.cagr = ((last_value / first_value) ** (1 / n_years) - 1) * 100
            
            metrics.latest_growth = yearly_sums["YoY_Growth"].iloc[-1]
            metrics.last_tahun = str(yearly_sums["Tahun"].iloc[-1])
        elif len(yearly_sums) == 1:
            metrics.last_tahun = str(yearly_sums["Tahun"].iloc[0])
        
        return metrics


# =============================================================================
# LABEL SHORTENER
# =============================================================================

class LabelShortener:
    """Creates shortened labels for chart legends based on column type."""
    
    # Mapping of column types to their shortening rules
    SHORTENING_RULES = {
        "SUMBER DANA": lambda v: v[:2],
        "FUNGSI": lambda v: v[:2],
        "JENIS BELANJA": lambda v: v[:2],
        "SUB FUNGSI": lambda v: f"{v[:2]} {v[3:5]}" if len(v) >= 5 else v,
        "PROGRAM": lambda v: f"{v[:2]} {v[3:5]}" if len(v) >= 5 else v,
        "KEGIATAN": lambda v: v[:4],
        "OUTPUT (KRO)": lambda v: f"{v[:4]} {v[5:8]}" if len(v) >= 8 else v,
        "SUB OUTPUT (RO)": lambda v: f"{v[:4]} {v[5:8]} {v[9:12]}" if len(v) >= 12 else v,
        "KOMPONEN": lambda v: f"{v[:4]} {v[5:8]} {v[9:12] {v[13:15]}" if len(v) >= 11 else v,
        "AKUN 4 DIGIT": lambda v: v[:4],
    }
    
    @classmethod
    def shorten(cls, value: Any, col_type: str) -> str:
        """
        Create shortened label for legend display.
        
        Args:
            value: Original value
            col_type: Column type to determine shortening rule
            
        Returns:
            Shortened string label
        """
        value_str = str(value)
        
        rule = cls.SHORTENING_RULES.get(col_type)
        if rule:
            try:
                return rule(value_str)
            except (IndexError, TypeError):
                pass
        
        # Default: return first word
        return value_str.split(" ")[0]


# =============================================================================
# CHART BUILDER
# =============================================================================

class TrendChartBuilder:
    """Builds trend line charts for budget analysis."""
    
    def __init__(self, config: AppConfig = AppConfig()):
        self.config = config
    
    def create_line_chart(
        self,
        df: pd.DataFrame,
        category_col: str,
        selected_metric: str,
        selected_kl: str,
        value_col: str = "Nilai"
    ) -> Tuple[Optional[go.Figure], Optional[pd.DataFrame]]:
        """
        Create a line chart showing trends by category.
        
        Args:
            df: Source DataFrame
            category_col: Column to use for line grouping
            selected_metric: Name of the metric for title
            selected_kl: Selected K/L for title
            value_col: Column containing values
            
        Returns:
            Tuple of (Figure, grouped DataFrame) or (None, None) on error
        """
        # Validate inputs
        if df.empty:
            return None, None
        
        if category_col not in df.columns:
            return None, None
        
        if df[category_col].isna().all():
            return None, None
        
        try:
            # Prepare data
            df_work = df.copy()
            df_work["short_label"] = df_work[category_col].apply(
                lambda x: LabelShortener.shorten(x, category_col)
            )
            
            # Group data
            df_grouped = (
                df_work
                .groupby(["Tahun", category_col, "short_label"], as_index=False)[value_col]
                .sum()
            )
            
            if df_grouped.empty:
                return None, None
            
            # Ensure proper sorting
            df_grouped["Tahun"] = df_grouped["Tahun"].astype(str)
            df_grouped = df_grouped.sort_values("Tahun")
            
            # Calculate dynamic height
            n_groups = df_grouped["short_label"].nunique()
            height = self.config.CHART_BASE_HEIGHT
            if n_groups > self.config.CHART_GROUP_THRESHOLD:
                height += n_groups * self.config.CHART_HEIGHT_PER_EXTRA_LINE
            
            # Create chart
            fig = px.line(
                df_grouped,
                x="Tahun",
                y=value_col,
                color="short_label",
                markers=True,
                title=f"📈 {selected_metric} BERDASARKAN {category_col} — {selected_kl}",
                labels={
                    "Tahun": "Tahun",
                    value_col: "Jumlah (Rp)",
                    "short_label": category_col.replace("_", " ").title(),
                },
                template="plotly_white",
                height=height,
            )
            
            # Apply layout
            self._apply_layout(fig, df_grouped, category_col)
            
            return fig, df_grouped
            
        except Exception as e:
            st.error(f"Error creating chart: {str(e)}")
            return None, None
    
    def _apply_layout(
        self,
        fig: go.Figure,
        df_grouped: pd.DataFrame,
        category_col: str
    ) -> None:
        """Apply layout settings to figure."""
        fig.update_layout(
            hovermode="closest",
            title_x=0,
            legend_title_text=category_col.replace("_", " ").title(),
            margin=dict(l=40, r=40, t=80, b=40),
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(family="Inter, Google Sans, Roboto, Arial"),
        )
        
        # Custom hover template with original category values
        fig.update_traces(
            hovertemplate=(
                f"<b>%{{customdata}}</b><br>"
                "Tahun: %{x}<br>"
                "Rp %{y:,.0f}<extra></extra>"
            ),
            customdata=df_grouped[category_col],
            line=dict(width=2.5),
            marker=dict(size=7)
        )


# =============================================================================
# UI COMPONENTS
# =============================================================================

class UIComponents:
    """Reusable UI component generators."""
    
    @staticmethod
    def render_header(selected_kl: Optional[str], selected_metric: Optional[str]) -> None:
        """Render the main dashboard header."""
        kl_text = "Semua K/L" if selected_kl == "Semua" else (selected_kl or "—")
        metric_text = f" {selected_metric}" if selected_metric else ""
        
        st.markdown(f"""
        <div class="dashboard-header" role="banner">
            <div class="breadcrumb">DASHBOARD / ANALISIS{metric_text} / {kl_text}</div>
            <h1 class="dashboard-title">Analisis Tren Anggaran & Realisasi Belanja Negara</h1>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_metric_cards(
        metrics: FinancialMetrics,
        selected_kl: Optional[str],
        selected_metric: Optional[str]
    ) -> None:
        """Render the summary metric cards."""
        if not metrics.has_data:
            return
        
        col1, col2, col3 = st.columns([1, 2, 2])
        
        with col1:
            latest_total = metrics.yearly_totals["Nilai"].iloc[-1]
            trend_class = "trend-positive" if metrics.latest_growth >= 0 else "trend-negative"
            trend_arrow = "↗" if metrics.latest_growth >= 0 else "↘"
            
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Tahun {metrics.last_tahun}</div>
                <div class="metric-value">{Formatter.to_rupiah_short(latest_total)}</div>
                <div class="metric-trend {trend_class}">
                    {trend_arrow} {metrics.latest_growth:+.1f}% YoY
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Tingkat Pertumbuhan Tahunan Majemuk (CAGR)</div>
                <div class="metric-value">{metrics.cagr:+.1f}%</div>
                <div class="metric-sublabel">Pertumbuhan tahunan rata-rata selama rentang periode waktu tertentu</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Tingkat Pertumbuhan Tahunan Rata-rata (AAGR)</div>
                <div class="metric-value">{metrics.aagr:+.1f}%</div>
                <div class="metric-sublabel">Rata-rata tingkat pertumbuhan tahunan</div>
            </div>
            """, unsafe_allow_html=True)
    
    @staticmethod
    def render_footer() -> None:
        """Render page footer."""
        st.markdown("---")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.caption("📊 Sumber Data: bidja.kemenkeu.go.id")
        with col2:
            st.caption(f"🕐 Diperbarui: {datetime.now().strftime('%d %B %Y %H:%M')}")
    
    @staticmethod
    def apply_styles() -> None:
        """Apply CSS styles to the page."""
        st.markdown(CSS_STYLES, unsafe_allow_html=True)


# =============================================================================
# SIDEBAR CONTROLLER
# =============================================================================

@dataclass
class SidebarSelections:
    """Container for sidebar filter selections."""
    df_filtered: pd.DataFrame
    selected_kl: str
    selected_metric: str
    selected_years: Tuple[int, int]
    active_filters: Dict[str, List[str]]


class SidebarController:
    """Manages sidebar filter controls."""
    
    def __init__(self, df: pd.DataFrame, config: AppConfig = AppConfig()):
        self.df = df
        self.config = config
    
    def render(self) -> SidebarSelections:
        """Render sidebar controls and return selections."""
        with st.sidebar:
            st.markdown("""
            <div class="sidebar-section">
                <h3 style='margin: 0.1rem; color: var(--on-surface);'>Filter Data</h3>
            """, unsafe_allow_html=True)
            
            # Ensure Tahun is numeric
            df_work = self.df.copy()
            df_work["Tahun"] = pd.to_numeric(df_work["Tahun"], errors="coerce").astype("Int64")
            
            # K/L selector
            selected_kl = self._render_kl_selector(df_work)
            
            # Filter by K/L
            if selected_kl == "Semua":
                df_filtered = df_work.copy()
            else:
                df_filtered = df_work[df_work[self.config.KL_COLUMN] == selected_kl]
            
            # Metric selector
            selected_metric = self._render_metric_selector(df_filtered)
            
            # Year range selector
            selected_years = self._render_year_selector(df_filtered)
            
            # Advanced filters
            active_filters = self._render_advanced_filters(df_filtered)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Apply filters
        df_filtered = self._apply_filters(df_filtered, selected_years, active_filters)
        
        return SidebarSelections(
            df_filtered=df_filtered,
            selected_kl=selected_kl,
            selected_metric=selected_metric,
            selected_years=selected_years,
            active_filters=active_filters
        )
    
    def _render_kl_selector(self, df: pd.DataFrame) -> str:
        """Render K/L selection dropdown."""
        kl_list = sorted(df[self.config.KL_COLUMN].dropna().unique())
        kl_list.append("Semua")
        
        return st.selectbox(
            "Pilih Kementerian/Lembaga",
            kl_list,
            key="ministry_select",
            help="Pilih kementerian/lembaga untuk melihat analisis anggaran"
        )
    
    def _render_metric_selector(self, df: pd.DataFrame) -> str:
        """Render metric selection dropdown."""
        numeric_cols = df.select_dtypes(include=["int64", "float64", "Int64"]).columns.tolist()
        if "Tahun" in numeric_cols:
            numeric_cols.remove("Tahun")
        
        if not numeric_cols:
            return "(Tidak ada kolom numerik)"
        
        return st.selectbox(
            "Metrik Anggaran",
            numeric_cols,
            key="metric_select",
            help="Pilih jenis anggaran yang akan dianalisis"
        )
    
    def _render_year_selector(self, df: pd.DataFrame) -> Tuple[int, int]:
        """Render year range selector."""
        year_options = sorted(df["Tahun"].dropna().unique())
        
        if len(year_options) == 0:
            return (0, 0)
        
        if len(year_options) == 1:
            single_year = int(year_options[0])
            st.markdown(f"**Tahun tersedia:** {single_year}")
            return (single_year, single_year)
        
        current_year = datetime.now().year
        min_year = int(min(year_options))
        max_year = int(max(year_options))
        
        default_end = min(current_year, max_year)
        default_start = st.session_state.get("filter__year_range", (min_year, default_end))[0]
        default_start = max(min_year, default_start)
        
        return st.slider(
            "Rentang Tahun",
            min_value=min_year,
            max_value=max_year,
            value=(default_start, default_end),
            step=1,
            key="filter__year_range"
        )
    
    def _render_advanced_filters(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Render advanced categorical filters."""
        active_filters = {}
        
        with st.expander("⚙️ Filter Lanjutan"):
            st.markdown("### Filter Berdasarkan Nilai Kategorikal")
            
            cat_cols = [
                col for col in df.select_dtypes(include=["object"]).columns
                if col not in [self.config.KL_COLUMN, self.config.YEAR_COLUMN]
            ]
            
            for cat_col in cat_cols:
                options = sorted(df[cat_col].dropna().unique())
                selected_values = st.multiselect(
                    f"Pilih {cat_col.replace('_', ' ').title()}",
                    options=options,
                    default=options,
                    key=f"filter__{cat_col}"
                )
                active_filters[cat_col] = selected_values
        
        return active_filters
    
    def _apply_filters(
        self,
        df: pd.DataFrame,
        years: Tuple[int, int],
        filters: Dict[str, List[str]]
    ) -> pd.DataFrame:
        """Apply year and categorical filters."""
        # Year filter
        if years[0] and years[1]:
            df = df[
                (df["Tahun"] >= years[0]) &
                (df["Tahun"] <= years[1])
            ]
        
        # Categorical filters
        for col, values in filters.items():
            if values and col in df.columns:
                df = df[df[col].isin(values)]
        
        return df


# =============================================================================
# CATEGORY TAB VIEW
# =============================================================================

class CategoryTabView:
    """Renders a category analysis tab with chart and data table."""
    
    def __init__(
        self,
        df: pd.DataFrame,
        category_col: str,
        selected_metric: str,
        selected_kl: str,
        config: AppConfig = AppConfig()
    ):
        self.df = df
        self.category_col = category_col
        self.selected_metric = selected_metric
        self.selected_kl = selected_kl
        self.config = config
        self.chart_builder = TrendChartBuilder(config)
    
    def render(self) -> None:
        """Render the complete tab view."""
        # Validate column exists
        if self.category_col not in self.df.columns:
            st.warning(f"Kolom {self.category_col} tidak ditemukan")
            return
        
        # Check for valid data
        if self.df[self.category_col].notna().sum() == 0:
            st.warning(f"Kolom '{self.category_col}' tidak memiliki data yang valid")
            return
        
        try:
            # Create and display chart
            fig, grouped_df = self.chart_builder.create_line_chart(
                self.df,
                self.category_col,
                self.selected_metric,
                self.selected_kl
            )
            
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
                self._render_data_table(grouped_df)
            else:
                st.warning(f"Tidak dapat membuat chart untuk {self.category_col}")
                
        except Exception as e:
            st.error(f"Error creating chart for {self.category_col}: {str(e)}")
            with st.expander("🔧 Debug Data"):
                st.write(f"Sample values:", self.df[self.category_col].head(10).tolist())
    
    def _render_data_table(self, grouped_df: Optional[pd.DataFrame]) -> None:
        """Render expandable data table with Excel export."""
        with st.expander("📋 Data Tabel", expanded=True):
            if grouped_df is None or grouped_df.empty:
                st.info("Tidak ada data untuk ditampilkan dalam tabel")
                return
            
            # Prepare display dataframe
            df_display = grouped_df[["Tahun", self.category_col, "Nilai"]].copy()
            
            # Pivot for wide format
            df_pivot = (
                df_display
                .pivot_table(
                    index=self.category_col,
                    columns="Tahun",
                    values="Nilai",
                    aggfunc="sum"
                )
                .fillna(0)
                .reset_index()
            )
            
            # Sort year columns
            tahun_cols = sorted([c for c in df_pivot.columns if c != self.category_col])
            df_pivot = df_pivot[[self.category_col] + tahun_cols]
            
            # Keep numeric copy for Excel
            df_excel = df_pivot.copy()
            
            # Format for display
            for col in tahun_cols:
                df_pivot[col] = df_pivot[col].apply(Formatter.to_rupiah_full)
            
            df_pivot = df_pivot.rename(columns={self.category_col: self.selected_metric})
            
            st.dataframe(df_pivot, use_container_width=True, hide_index=True)
            
            # Excel download
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                df_excel.to_excel(writer, sheet_name="Data", index=False)
            
            st.download_button(
                label="📥 Download Excel",
                data=excel_buffer.getvalue(),
                file_name=f"{self.selected_metric}_{self.category_col}_{self.selected_kl}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )


# =============================================================================
# MAIN APPLICATION
# =============================================================================

class TrendAnalysisApp:
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
            with st.spinner("Memuat data anggaran..."):
                df = self.data_loader.get_data()
        except DataLoadError as e:
            st.error(f"Gagal memuat data: {e}")
            return
        
        if df.empty:
            st.error("Tidak dapat memuat data. Silakan periksa file dataset.")
            return
        
        # Render sidebar and get selections
        sidebar = SidebarController(df, self.config)
        selections = sidebar.render()
        
        # Render header
        UIComponents.render_header(selections.selected_kl, selections.selected_metric)
        
        # Validate metric
        if selections.selected_metric not in selections.df_filtered.columns:
            st.warning("Kolom metrik tidak ditemukan di dataset untuk K/L ini.")
            return
        
        # Rename metric column for processing
        df_work = selections.df_filtered.rename(columns={selections.selected_metric: "Nilai"})
        
        # Calculate and display metrics
        metrics = MetricsCalculator.calculate(df_work)
        
        kl_display = (
            "Semua Kementerian/Lembaga"
            if selections.selected_kl == "Semua"
            else selections.selected_kl
        )
        
        st.markdown(
            f"<div class='section-title'>RINGKASAN KINERJA {selections.selected_metric}</div>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<div class='material-card'>{kl_display}</div>",
            unsafe_allow_html=True
        )
        
        UIComponents.render_metric_cards(metrics, selections.selected_kl, selections.selected_metric)
        
        # Visualization section
        st.markdown("<div class='section-title'>TREN ANALISIS</div>", unsafe_allow_html=True)
        
        # Get categorical columns for tabs
        cat_cols = [
            col for col in df_work.select_dtypes(include=["object"]).columns
            if col not in [self.config.KL_COLUMN, self.config.YEAR_COLUMN]
        ]
        
        if cat_cols:
            self._render_category_tabs(df_work, cat_cols, selections)
        else:
            st.info("ℹ️ Tidak ada kolom kategorikal yang tersedia untuk visualisasi.")
        
        # Footer
        UIComponents.render_footer()
    
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
    
    def _render_category_tabs(
        self,
        df: pd.DataFrame,
        cat_cols: List[str],
        selections: SidebarSelections
    ) -> None:
        """Render tabs for each categorical column."""
        # Create tab labels
        tab_labels = [f"📈 {col.replace('_', ' ').title()}" for col in cat_cols]
        
        # Create tabs (CSS will handle multi-line wrapping)
        tabs = st.tabs(tab_labels)
        
        for tab, col in zip(tabs, cat_cols):
            with tab:
                view = CategoryTabView(
                    df=df,
                    category_col=col,
                    selected_metric=selections.selected_metric,
                    selected_kl=selections.selected_kl,
                    config=self.config
                )
                view.render()


# =============================================================================
# ENTRY POINT
# =============================================================================

def main() -> None:
    """Application entry point."""
    try:
        app = TrendAnalysisApp()
        app.run()
    except Exception as e:
        st.error(f"Terjadi kesalahan dalam aplikasi: {str(e)}")
        st.info("Silakan refresh halaman atau hubungi administrator.")


if __name__ == "__main__":
    main()




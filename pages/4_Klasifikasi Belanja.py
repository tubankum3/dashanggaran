"""
Dashboard Klasifikasi Belanja 
==========================
Dashboard drill-down interaktif untuk menganalisis klasifikasi akun Jenis Belanja Pemerintah.
"""

from __future__ import annotations

import io
import textwrap
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from streamlit_plotly_events import plotly_events


# =============================================================================
# ENUMS & CONSTANTS
# =============================================================================

class SortOrder(Enum):
    """Sort order for data display."""
    TOP = "Top"
    BOTTOM = "Bottom"
    
    @property
    def is_descending(self) -> bool:
        """Returns True if sorting should be descending (largest first)."""
        return self == SortOrder.TOP
    
    @property
    def display_label(self) -> str:
        """Human-readable label for UI."""
        return "tertinggi" if self == SortOrder.TOP else "terendah"
    
    @property
    def chart_color(self) -> str:
        """Bar chart color based on sort order."""
        return "#1a73e8" if self == SortOrder.TOP else "#dc3545"


class HierarchyLevel(Enum):
    """AKUN classification hierarchy levels in order (6 levels)."""
    AKUN_1_DIGIT = "AKUN 1 DIGIT"
    AKUN_2_DIGIT = "AKUN 2 DIGIT"
    AKUN_3_DIGIT = "AKUN 3 DIGIT"
    AKUN_4_DIGIT = "AKUN 4 DIGIT"
    AKUN_5_DIGIT = "AKUN 5 DIGIT"
    AKUN_6_DIGIT = "AKUN 6 DIGIT"
    
    @classmethod
    def ordered_columns(cls) -> List[str]:
        """Return hierarchy columns in drill-down order."""
        return [level.value for level in cls]


# Available metrics for this dashboard
AVAILABLE_METRICS: List[str] = [
    'REALISASI BELANJA KL (SAKTI)',
    'BLOKIR DIPA REVISI',
    'PAGU DIPA REVISI',
    'BLOKIR DIPA AWAL',
    'PAGU DIPA AWAL',
    'BLOKIR PERPRES',
    'PAGU PERPRES',
    'BLOKIR HIMPUNAN',
    'PAGU HIMPUNAN',
    'PAGU DIPA AWAL EFEKTIF',
    'PAGU DIPA REVISI EFEKTIF',
    'PAGU HIMPUNAN EFEKTIF',
    'PAGU PERPRES EFEKTIF'
]


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class AppConfig:
    """Application configuration constants."""
    
    # Data source - using jb.csv
    DATA_URL: str = "https://raw.githubusercontent.com/tubankum3/dashanggaran/main/jb.csv"
    REQUEST_TIMEOUT: int = 30
    
    # UI defaults
    DEFAULT_YEAR: int = 2025
    DEFAULT_TOP_N: int = 10
    MIN_TOP_N: int = 1
    MAX_TOP_N: int = 500
    DEFAULT_METRIC: str = "REALISASI BELANJA KL (SAKTI)"
    
    # Chart settings
    CHART_BASE_HEIGHT: int = 400
    CHART_MIN_HEIGHT: int = 300
    CHART_MAX_HEIGHT: int = 1200
    CHART_HEIGHT_PER_EXTRA_ITEM: int = 5
    CHART_HEIGHT_REDUCTION_PER_MISSING: int = 20
    CHART_ITEMS_BASELINE: int = 10
    LABEL_WRAP_WIDTH: int = 32
    TARGET_TICK_COUNT: int = 6
    
    # Required columns
    YEAR_COLUMN: str = "Tahun"
    KL_COLUMN: str = "KEMENTERIAN/LEMBAGA"
    
    # Page configuration
    PAGE_TITLE: str = "Analisis Klasifikasi Akun Anggaran"
    PAGE_ICON: str = "💰"


@dataclass
class SessionKeys:
    """Session state key names to avoid magic strings."""
    DRILL_STATE: str = "akun_drill"
    LEVEL_INDEX: str = "akun_level_index"
    CLICK_KEY: str = "akun_click_key"


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

.drill-label {
    font-size: 0.75rem;
    color: var(--gray-500);
    font-weight: 500;
    padding: 0.25rem 0;
}
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
    
    # Value thresholds for unit conversion
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
    
    @staticmethod
    def to_percentage(value: float, decimals: int = 2) -> str:
        """Format value as percentage string."""
        return f"{value:.{decimals}f}%"
    
    @staticmethod
    def wrap_label(label: str, max_width: int = 32) -> str:
        """Wrap long labels for chart display using HTML line breaks."""
        return "<br>".join(
            textwrap.wrap(str(label), width=max_width, break_long_words=False)
        )


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
    @st.cache_data(show_spinner="Memuat dataset akun anggaran...")
    def load_data(url: str, timeout: int) -> pd.DataFrame:
        """Load and cache budget data from remote CSV file."""
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            
            # Read CSV directly from response content
            df = pd.read_csv(io.StringIO(response.text), low_memory=False)
            
            # Clean up unnamed columns
            unnamed_cols = [c for c in df.columns if "Unnamed" in str(c)]
            if unnamed_cols:
                df = df.drop(columns=unnamed_cols)
            
            # Ensure Tahun is string for consistent filtering
            if "Tahun" in df.columns:
                df["Tahun"] = df["Tahun"].astype(str)
            
            return df
            
        except requests.exceptions.RequestException as e:
            raise DataLoadError(f"Network error: {e}")
        except Exception as e:
            raise DataLoadError(f"Failed to load data: {e}")
    
    def get_data(self) -> pd.DataFrame:
        """Load data using configured settings."""
        return self.load_data(
            self.config.DATA_URL,
            self.config.REQUEST_TIMEOUT
        )


class DataAggregator:
    """Handles data aggregation for drill-down views."""
    
    @staticmethod
    def aggregate_by_level(
        df: pd.DataFrame,
        group_cols: List[str],
        metric: str,
        top_n: Optional[int] = None,
        sort_order: SortOrder = SortOrder.TOP
    ) -> pd.DataFrame:
        """Aggregate data by grouping columns and optionally limit to top/bottom N."""
        # Filter to valid columns
        valid_cols = [c for c in group_cols if c in df.columns]
        if not valid_cols or metric not in df.columns:
            return pd.DataFrame()
        
        # Perform aggregation
        agg = df.groupby(valid_cols, as_index=False)[metric].sum()
        agg = agg.dropna(subset=[valid_cols[-1]])
        
        # Apply top/bottom N filter
        if top_n and len(agg) > 0:
            if sort_order == SortOrder.TOP:
                selected = agg.nlargest(top_n, metric)
            else:
                selected = agg.nsmallest(top_n, metric)
            agg = agg[agg[valid_cols[-1]].isin(selected[valid_cols[-1]])]
        
        return agg
    
    @staticmethod
    def filter_by_ancestors(
        df: pd.DataFrame,
        hierarchy: List[str],
        current_level: int,
        drill_state: Dict[str, Optional[str]]
    ) -> pd.DataFrame:
        """Filter DataFrame by ancestor selections in drill-down hierarchy."""
        df_filtered = df.copy()
        
        for i in range(current_level):
            col = hierarchy[i]
            selected_value = drill_state.get(col)
            if selected_value is not None and col in df_filtered.columns:
                df_filtered = df_filtered[df_filtered[col] == selected_value]
        
        return df_filtered


# =============================================================================
# CHART BUILDER
# =============================================================================

class ChartBuilder:
    """Builds Plotly charts for budget visualization."""
    
    def __init__(self, config: AppConfig = AppConfig()):
        self.config = config
    
    def create_horizontal_bar_chart(
        self,
        df: pd.DataFrame,
        metric: str,
        category_col: str,
        title: str = "",
        sort_order: SortOrder = SortOrder.TOP,
        max_height: Optional[int] = None
    ) -> go.Figure:
        """Create a horizontal bar chart with percentage labels and Rupiah formatting."""
        if df.empty or metric not in df.columns or category_col not in df.columns:
            return self._create_empty_chart("No data available")
        
        # Prepare data
        plot_df = self._prepare_plot_data(df, metric, category_col, sort_order)
        
        # Calculate axis configuration
        axis_config = self._calculate_axis_config(plot_df[metric])
        
        # Build figure
        fig = self._build_figure(plot_df, metric, category_col, sort_order)
        
        # Apply layout
        chart_height = max_height or self._calculate_dynamic_height(plot_df[category_col].nunique())
        self._apply_layout(fig, title, chart_height, axis_config)
        
        return fig
    
    def _prepare_plot_data(
        self,
        df: pd.DataFrame,
        metric: str,
        category_col: str,
        sort_order: SortOrder
    ) -> pd.DataFrame:
        """Prepare DataFrame for plotting with calculated fields."""
        plot_df = df.copy()
        
        # Ensure numeric
        plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce").fillna(0.0)
        
        # Calculate percentages
        total = plot_df[metric].sum()
        plot_df["_pct"] = (plot_df[metric] / total * 100).round(2) if total > 0 else 0.0
        plot_df["_pct_label"] = plot_df["_pct"].apply(lambda x: f"{x:.2f}%")
        plot_df["_rupiah_fmt"] = plot_df[metric].apply(Formatter.to_rupiah_short)
        plot_df["_wrapped_label"] = plot_df[category_col].apply(
            lambda x: Formatter.wrap_label(x, self.config.LABEL_WRAP_WIDTH)
        )
        
        # Sort for display (ascending puts largest at top in horizontal bar)
        ascending = sort_order == SortOrder.TOP
        plot_df = plot_df.sort_values(metric, ascending=ascending).reset_index(drop=True)
        
        return plot_df
    
    def _calculate_axis_config(self, values: pd.Series) -> Dict[str, Any]:
        """Calculate x-axis tick values and labels."""
        x_max = float(values.max()) if len(values) > 0 and values.max() > 0 else 100.0
        
        # Determine scale and unit
        if x_max >= Formatter.TRILLION:
            scale, unit = Formatter.TRILLION, "T"
        elif x_max >= Formatter.BILLION:
            scale, unit = Formatter.BILLION, "M"
        elif x_max >= Formatter.MILLION:
            scale, unit = Formatter.MILLION, "Jt"
        else:
            scale, unit = 1, ""
        
        # Calculate nice tick intervals
        if x_max > 0:
            raw_interval = x_max / self.config.TARGET_TICK_COUNT
            magnitude = 10 ** int(np.floor(np.log10(raw_interval)))
            nice_interval = np.ceil(raw_interval / magnitude) * magnitude
            last_tick = np.ceil(x_max / nice_interval) * nice_interval
            tick_vals = list(np.arange(0, last_tick + nice_interval, nice_interval))
        else:
            tick_vals = [0, 50, 100]
            last_tick = 100
        
        # Format tick labels
        if unit:
            tick_texts = [f"Rp {v/scale:.0f} {unit}" for v in tick_vals]
        else:
            tick_texts = [f"Rp {v:,.0f}" for v in tick_vals]
        
        return {
            "tick_vals": tick_vals,
            "tick_texts": tick_texts,
            "last_tick": last_tick
        }
    
    def _build_figure(
        self,
        plot_df: pd.DataFrame,
        metric: str,
        category_col: str,
        sort_order: SortOrder
    ) -> go.Figure:
        """Build the Plotly figure with traces."""
        fig = go.Figure()
        
        for _, row in plot_df.iterrows():
            fig.add_trace(go.Bar(
                x=[row[metric]],
                y=[row[category_col]],
                orientation='h',
                text=row["_pct_label"],
                textposition="outside",
                textfont=dict(size=11, color="#333"),
                marker=dict(color=sort_order.chart_color),
                hovertemplate=(
                    f"{row[category_col]}<br>"
                    f"Jumlah: {row['_rupiah_fmt']}<br>"
                    f"Persentase: {row['_pct_label']}<extra></extra>"
                ),
                showlegend=False,
            ))
        
        return fig
    
    def _calculate_dynamic_height(self, n_items: int) -> int:
        """Calculate chart height based on number of items."""
        config = self.config
        
        if n_items > config.CHART_ITEMS_BASELINE:
            height = config.CHART_BASE_HEIGHT + (n_items - config.CHART_ITEMS_BASELINE) * config.CHART_HEIGHT_PER_EXTRA_ITEM
        elif n_items < config.CHART_ITEMS_BASELINE:
            height = config.CHART_BASE_HEIGHT - (config.CHART_ITEMS_BASELINE - n_items) * config.CHART_HEIGHT_REDUCTION_PER_MISSING
        else:
            height = config.CHART_BASE_HEIGHT
        
        return max(config.CHART_MIN_HEIGHT, min(height, config.CHART_MAX_HEIGHT))
    
    def _apply_layout(
        self,
        fig: go.Figure,
        title: str,
        height: int,
        axis_config: Dict[str, Any]
    ) -> None:
        """Apply layout settings to figure."""
        fig.update_layout(
            title=title,
            showlegend=False,
            barmode="relative",
            margin=dict(t=70, l=250, r=80, b=50),
            height=height,
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis_title="",
            yaxis_title="",
            hovermode="closest",
        )
        
        fig.update_traces(
            hoverlabel=dict(align="left", bgcolor="white", font_size=10, font_color="#333"),
        )
        
        fig.update_xaxes(
            type="linear",
            tickmode="array",
            tickvals=axis_config["tick_vals"],
            ticktext=axis_config["tick_texts"],
            range=[0, axis_config["last_tick"] * 1.1],
            showgrid=True,
            gridcolor="rgba(200,200,200,0.3)",
            zeroline=True,
            zerolinecolor="rgba(150,150,150,0.5)",
        )
        
        fig.update_yaxes(
            categoryorder="trace",
            automargin=True,
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
# SESSION STATE MANAGEMENT
# =============================================================================

class DrillDownState:
    """Manages drill-down navigation state in Streamlit session."""
    
    def __init__(self, hierarchy: List[str]):
        self.hierarchy = hierarchy
        self._keys = SessionKeys()
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize session state if not exists."""
        if self._keys.DRILL_STATE not in st.session_state:
            st.session_state[self._keys.DRILL_STATE] = {col: None for col in self.hierarchy}
        if self._keys.LEVEL_INDEX not in st.session_state:
            st.session_state[self._keys.LEVEL_INDEX] = 0
        if self._keys.CLICK_KEY not in st.session_state:
            st.session_state[self._keys.CLICK_KEY] = 0
    
    @property
    def drill_selections(self) -> Dict[str, Optional[str]]:
        """Get current drill-down selections."""
        return st.session_state[self._keys.DRILL_STATE]
    
    @property
    def current_level(self) -> int:
        """Get current drill-down level index."""
        return st.session_state[self._keys.LEVEL_INDEX]
    
    @current_level.setter
    def current_level(self, value: int) -> None:
        """Set current drill-down level index."""
        st.session_state[self._keys.LEVEL_INDEX] = min(value, len(self.hierarchy) - 1)
    
    @property
    def click_key(self) -> int:
        """Get unique key for chart interactions."""
        return st.session_state[self._keys.CLICK_KEY]
    
    def select_value(self, level: str, value: str) -> None:
        """Record a selection at the given hierarchy level."""
        st.session_state[self._keys.DRILL_STATE][level] = value
    
    def clear_from_level(self, level_index: int) -> None:
        """Clear selections from the given level onwards."""
        for i in range(level_index, len(self.hierarchy)):
            col = self.hierarchy[i]
            st.session_state[self._keys.DRILL_STATE][col] = None
    
    def go_back(self) -> bool:
        """Go back one level. Returns True if successful."""
        if self.current_level > 0:
            new_level = self.current_level - 1
            self.clear_from_level(new_level)
            self.current_level = new_level
            self._increment_click_key()
            return True
        return False
    
    def reset(self) -> None:
        """Reset to initial state."""
        for col in self.hierarchy:
            st.session_state[self._keys.DRILL_STATE][col] = None
        st.session_state[self._keys.LEVEL_INDEX] = 0
        self._increment_click_key()
    
    def advance_level(self) -> None:
        """Move to next drill-down level if available."""
        if self.current_level + 1 < len(self.hierarchy):
            self.current_level = self.current_level + 1
        self._increment_click_key()
    
    def _increment_click_key(self) -> None:
        """Increment click key to force chart re-render."""
        st.session_state[self._keys.CLICK_KEY] += 1
    
    def get_active_breadcrumbs(self) -> List[Tuple[int, str, str]]:
        """Get list of (level_index, column_name, selected_value) for active selections."""
        return [
            (i, col, self.drill_selections[col])
            for i, col in enumerate(self.hierarchy)
            if self.drill_selections.get(col) is not None
        ]


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
        year_text = year or "OVERVIEW"
        metric_text = f" {metric}" if metric else "KLASIFIKASI AKUN"
        kl_text = ", ".join(kl_list) if kl_list else "SEMUA K/L"
        
        st.markdown(f"""
        <div class="dashboard-header" role="banner">
            <div class="breadcrumb">DASHBOARD / KLASIFIKASI BELANJA {metric_text} / {kl_text} / TAHUN {year_text}</div>
            <h1 class="dashboard-title">Analisis Klasifikasi Jenis Belanja</h1>
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
        st.markdown(CSS_TABS_UNDERLINE, unsafe_allow_html=True)


class SidebarController:
    """Manages sidebar filter controls."""
    
    def __init__(self, df: pd.DataFrame, config: AppConfig = AppConfig()):
        self.df = df
        self.config = config
    
    def render(self) -> Tuple[int, List[str], int, str, SortOrder]:
        """
        Render sidebar controls and return selected values.
        
        Returns:
            Tuple of (year, kl_list, top_n, metric, sort_order)
        """
        with st.sidebar:
            st.markdown("### ⚙️ Filter Data")
            
            year = self._render_year_selector()
            sort_order = self._render_sort_order_selector()
            top_n = self._render_top_n_input(sort_order)
            metric = self._render_metric_selector()
            kl_list = self._render_kl_selector()
            
            return year, kl_list, top_n, metric, sort_order
    
    def _render_year_selector(self) -> int:
        """Render year selection dropdown."""
        self._validate_column_exists(self.config.YEAR_COLUMN)
        
        # Extract valid years
        df_year = self.df[self.df[self.config.YEAR_COLUMN].notna()].copy()
        df_year[self.config.YEAR_COLUMN] = (
            df_year[self.config.YEAR_COLUMN]
            .astype(str)
            .str.extract(r"(\d{4})")[0]
        )
        years = sorted(df_year[self.config.YEAR_COLUMN].dropna().astype(int).unique().tolist())
        
        if not years:
            st.error("Tidak ada data tahun yang valid di dataset.")
            st.stop()
        
        default_idx = years.index(self.config.DEFAULT_YEAR) if self.config.DEFAULT_YEAR in years else len(years) - 1
        
        return st.selectbox("Pilih Tahun", years, index=default_idx)
    
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
            help=f"Jumlah data {sort_order.display_label} yang ditampilkan"
        )
    
    def _render_metric_selector(self) -> str:
        """Render metric selection dropdown."""
        # Use predefined metrics list, filter to those available in data
        available_metrics = [m for m in AVAILABLE_METRICS if m in self.df.columns]
        
        if not available_metrics:
            # Fallback to all numeric columns if no predefined metrics found
            available_metrics = self.df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        
        if not available_metrics:
            st.error("Tidak ada kolom numerik yang dapat dipilih sebagai metrik.")
            st.stop()
        
        default_idx = (
            available_metrics.index(self.config.DEFAULT_METRIC)
            if self.config.DEFAULT_METRIC in available_metrics
            else 0
        )
        
        return st.selectbox("Metrik Anggaran", options=available_metrics, index=default_idx)
    
    def _render_kl_selector(self) -> List[str]:
        """Render K/L multiselect."""
        self._validate_column_exists(self.config.KL_COLUMN)
        
        kl_list = sorted(self.df[self.config.KL_COLUMN].dropna().unique().tolist())
        selected = st.multiselect(
            "Pilih Kementerian/Lembaga (opsional)",
            options=["Semua"] + kl_list,
            default=["Semua"]
        )
        
        return [] if "Semua" in selected else selected
    
    def _validate_column_exists(self, column: str) -> None:
        """Validate that a required column exists in the DataFrame."""
        if column not in self.df.columns:
            st.error(f"Kolom '{column}' tidak ditemukan di dataset.")
            st.stop()


class ExcelExporter:
    """Handles Excel file export functionality."""
    
    @staticmethod
    def export_to_excel(
        df: pd.DataFrame,
        metric: str,
        formatter_func: callable = Formatter.to_rupiah_full
    ) -> bytes:
        """Export DataFrame to Excel bytes with formatted and numeric columns."""
        export_df = df.copy()
        
        # Add hidden numeric column for calculations
        numeric_col = f"{metric} (numeric)"
        export_df[numeric_col] = pd.to_numeric(export_df[metric], errors="coerce").fillna(0)
        
        # Format display column
        export_df[metric] = export_df[numeric_col].apply(formatter_func)
        
        # Write to buffer
        buffer = BytesIO()
        export_df.to_excel(buffer, index=False, sheet_name="Data")
        buffer.seek(0)
        
        return buffer.getvalue()


# =============================================================================
# DRILL-DOWN INTERFACE
# =============================================================================

class DrillDownView:
    """Main drill-down visualization interface."""
    
    def __init__(
        self,
        df: pd.DataFrame,
        hierarchy: List[str],
        metric: str,
        year: int,
        top_n: int,
        sort_order: SortOrder,
        config: AppConfig = AppConfig()
    ):
        self.df = df
        self.hierarchy = hierarchy
        self.metric = metric
        self.year = year
        self.top_n = top_n
        self.sort_order = sort_order
        self.config = config
        
        self.state = DrillDownState(hierarchy)
        self.chart_builder = ChartBuilder(config)
        self.aggregator = DataAggregator()
    
    def render(self) -> None:
        """Render the complete drill-down interface."""
        # Ensure metric is numeric
        self.df[self.metric] = pd.to_numeric(self.df[self.metric], errors="coerce").fillna(0.0)
        
        # Render components
        self._render_title()
        self._render_breadcrumbs()
        self._render_navigation_buttons()
        self._render_chart_and_handle_clicks()
        self._render_detail_table()
    
    def _render_title(self) -> None:
        """Render section title."""
        st.markdown(f"##### KLASIFIKASI BELANJA {self.metric} TAHUN {self.year}")
    
    def _render_breadcrumbs(self) -> None:
        """Render breadcrumb navigation for active drill-down selections."""
        breadcrumbs = self.state.get_active_breadcrumbs()
        
        if not breadcrumbs:
            return
        
        st.markdown("BERDASARKAN:")
        
        for level_idx, col, value in breadcrumbs:
            row = st.columns([1, 5])
            with row[0]:
                st.markdown(f"<div class='drill-label'>{col}</div>", unsafe_allow_html=True)
            with row[1]:
                if st.button(
                    f"{value}",
                    key=f"crumb-{col}-{value}-{self.state.click_key}",
                    use_container_width=True
                ):
                    # Clear selections after this level and go to next
                    self.state.clear_from_level(level_idx + 1)
                    self.state.current_level = min(level_idx + 1, len(self.hierarchy) - 1)
                    self.state._increment_click_key()
                    st.rerun()
    
    def _render_navigation_buttons(self) -> None:
        """Render back and reset navigation buttons."""
        left_col, _, right_col = st.columns([1, 10, 1])
        
        with left_col:
            if st.button(":arrow_backward:", help="Kembali satu tingkat"):
                if self.state.go_back():
                    st.rerun()
        
        with right_col:
            if st.button(":arrows_counterclockwise:", help="Kembali ke tampilan awal"):
                self.state.reset()
                st.rerun()
    
    def _render_chart_and_handle_clicks(self) -> None:
        """Render the chart and handle click events for drill-down."""
        # Get current view data
        current_col = self.hierarchy[self.state.current_level]
        filtered_df = self.aggregator.filter_by_ancestors(
            self.df, self.hierarchy, self.state.current_level, self.state.drill_selections
        )
        
        # Aggregate for current level
        agg_df = self.aggregator.aggregate_by_level(
            filtered_df, [current_col], self.metric, self.top_n, self.sort_order
        )
        
        if agg_df.empty:
            st.info("Tidak ada data untuk level ini.")
            return
        
        # Build and display chart
        title = (
            f"{self.sort_order.value.upper()} {self.top_n} {current_col} "
            f"(Level {self.state.current_level + 1} dari {len(self.hierarchy)})"
        )
        fig = self.chart_builder.create_horizontal_bar_chart(
            agg_df, self.metric, current_col,
            title=title, sort_order=self.sort_order, max_height=600
        )
        
        # Check if we can drill down further
        can_drill = self.state.current_level + 1 < len(self.hierarchy)
        
        if can_drill:
            # Display with click events
            events = plotly_events(
                fig, click_event=True,
                key=f"drill-{self.state.click_key}",
                override_height=600
            )
            
            # Handle clicks
            if events:
                clicked_value = self._extract_clicked_value(events[0])
                if clicked_value:
                    self.state.select_value(current_col, clicked_value)
                    self.state.advance_level()
                    st.rerun()
        else:
            # Last level - just display chart without click events
            st.plotly_chart(fig, use_container_width=True)
    
    def _extract_clicked_value(self, event: Dict[str, Any]) -> Optional[str]:
        """Extract the clicked category value from a Plotly event."""
        # Try y-axis value first
        clicked = event.get("y") or event.get("label")
        
        # Fallback to customdata
        if not clicked and event.get("customdata"):
            cd = event.get("customdata")
            if isinstance(cd, list) and len(cd) > 0:
                clicked = cd[0][0] if isinstance(cd[0], (list, tuple)) else cd[0]
        
        return clicked
    
    def _render_detail_table(self) -> None:
        """Render the expandable detail table with export."""
        current_col = self.hierarchy[self.state.current_level]
        
        with st.expander(f"📋 Tabel Rincian Data {current_col}"):
            # Get filtered data
            filtered_df = self.aggregator.filter_by_ancestors(
                self.df, self.hierarchy, self.state.current_level, self.state.drill_selections
            )
            
            # Prepare display columns
            display_cols = [self.config.KL_COLUMN, self.config.YEAR_COLUMN] + self.hierarchy + [self.metric]
            display_cols = [c for c in display_cols if c in filtered_df.columns]
            
            table_df = filtered_df[display_cols].copy()
            table_df[self.metric] = pd.to_numeric(table_df[self.metric], errors="coerce").fillna(0)
            
            # Sort based on sort order
            table_df = table_df.sort_values(
                by=self.metric,
                ascending=(self.sort_order == SortOrder.BOTTOM)
            ).reset_index(drop=True)
            
            # Add grand total row
            grand_total = table_df[self.metric].sum()
            total_row = {col: "" for col in table_df.columns}
            label_col = next((c for c in self.hierarchy if c in table_df.columns), self.config.KL_COLUMN)
            total_row[label_col] = "TOTAL"
            total_row[self.metric] = grand_total
            
            # Format for display
            display_df = table_df.copy()
            display_df[self.metric] = display_df[self.metric].apply(Formatter.to_rupiah_full)
            total_row_formatted = total_row.copy()
            total_row_formatted[self.metric] = Formatter.to_rupiah_full(grand_total)
            
            display_df = pd.concat([display_df, pd.DataFrame([total_row_formatted])], ignore_index=True)
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            # Export button
            export_df = pd.concat([table_df, pd.DataFrame([total_row])], ignore_index=True)
            excel_data = ExcelExporter.export_to_excel(export_df, self.metric)
            
            st.download_button(
                label="📥 Download Excel",
                data=excel_data,
                file_name=f"akun_drill_{self.sort_order.value}_{self.top_n}_{self.metric}_{self.year}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )


# =============================================================================
# MAIN APPLICATION
# =============================================================================

class AkunClassificationApp:
    """Main application controller for AKUN classification drill-down."""
    
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
            st.warning("Data tidak tersedia.")
            return
        
        # Validate required columns
        if not self._validate_required_columns(df):
            return
        
        # Render sidebar and get selections
        sidebar = SidebarController(df, self.config)
        year, kl_list, top_n, metric, sort_order = sidebar.render()
        
        # Render header
        UIComponents.render_header(str(year), metric, kl_list)
        
        # Filter data
        df_filtered = self._apply_base_filters(df, year, kl_list)
        
        # Determine available hierarchy levels (AKUN 1-6 DIGIT)
        available_hierarchy = [
            col for col in HierarchyLevel.ordered_columns()
            if col in df_filtered.columns
        ]
        
        if not available_hierarchy:
            st.error("Kolom hierarki Jenis Belanja tidak ditemukan di dataset.")
            st.info("Kolom yang diharapkan: " + ", ".join(HierarchyLevel.ordered_columns()))
            with st.expander("🔍 Kolom yang tersedia di dataset"):
                st.write(df_filtered.columns.tolist())
            return
        
        # Show hierarchy info
        st.markdown(f"**Hierarki tersedia:** {' → '.join(available_hierarchy)}")
        
        # Render drill-down view
        drill_view = DrillDownView(
            df_filtered, available_hierarchy, metric, year, top_n, sort_order, self.config
        )
        drill_view.render()
        
        # Render sidebar info and footer
        self._render_sidebar_info(sort_order, top_n, kl_list)
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
                "About": "Dashboard Klasifikasi Belanja"
            }
        )
    
    def _validate_required_columns(self, df: pd.DataFrame) -> bool:
        """Validate that required columns exist in the DataFrame."""
        required = [self.config.YEAR_COLUMN, self.config.KL_COLUMN]
        missing = [col for col in required if col not in df.columns]
        
        if missing:
            st.error(f"Kolom berikut tidak ditemukan: {', '.join(missing)}")
            return False
        return True
    
    def _apply_base_filters(
        self,
        df: pd.DataFrame,
        year: int,
        kl_list: List[str]
    ) -> pd.DataFrame:
        """Apply year and K/L filters to the DataFrame."""
        filtered = df[df[self.config.YEAR_COLUMN] == str(year)].copy()
        
        if kl_list:
            filtered = filtered[filtered[self.config.KL_COLUMN].isin(kl_list)]
        
        return filtered
    
    def _render_sidebar_info(
        self,
        sort_order: SortOrder,
        top_n: int,
        kl_list: List[str]
    ) -> None:
        """Render additional sidebar information."""
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"**Mode:** {sort_order.value} {top_n}")
        st.sidebar.markdown(f"**Hierarki:** 6 Level AKUN")
        
        if kl_list:
            st.sidebar.write("**K/L:**")
            for kl in kl_list:
                st.sidebar.write(f"- {kl}")


# =============================================================================
# ENTRY POINT
# =============================================================================

def main() -> None:
    """Application entry point."""
    try:
        app = AkunClassificationApp()
        app.run()
    except Exception as e:
        st.error(f"Terjadi kesalahan dalam aplikasi: {str(e)}")
        st.info("Silakan refresh halaman atau hubungi administrator.")


if __name__ == "__main__":
    main()

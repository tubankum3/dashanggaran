"""
Dashboard Analisis Anggaran dan Realisasi Belanja Negara
========================================================
Dashboard interaktif untuk menganalisis alokasi anggaran dan realisasi belanja Pemerintah.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from datetime import date
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
    
    # Column names
    YEAR_COLUMN: str = "Tahun"
    KL_COLUMN: str = "KEMENTERIAN/LEMBAGA"
    
    # Default metrics
    DEFAULT_PRIMARY_METRIC: str = "PAGU DIPA REVISI EFEKTIF"
    DEFAULT_SECONDARY_METRIC: str = "REALISASI BELANJA KL (SAKTI)"
    DEFAULT_SANKEY_PARENT: str = "JENIS BELANJA"
    DEFAULT_SANKEY_CHILD: str = "FUNGSI"
    
    # Chart settings
    TIME_SERIES_HEIGHT: int = 500
    SANKEY_MIN_HEIGHT: int = 500
    SANKEY_MAX_HEIGHT: int = 800
    SANKEY_HEIGHT_PER_NODE: int = 25
    SANKEY_CONTAINER_HEIGHT: int = 600
    
    # Colors
    PRIMARY_BAR_COLOR: str = "#005FAC"
    SECONDARY_MARKER_COLOR: str = "#FAB715"
    SANKEY_LINK_COLOR: str = "rgba(0, 95, 172, 0.2)"
    SANKEY_LINK_HOVER_COLOR: str = "gold"
    
    # Page configuration
    PAGE_TITLE: str = "Dashboard Analisis Anggaran dan Realisasi Belanja Negara"
    PAGE_ICON: str = ":analytics:"


@dataclass
class TimeSeriesConfig:
    """Configuration for time series chart."""
    primary_metric: str
    secondary_metric: str
    year_range: Tuple[int, int]
    selected_kl: str


@dataclass
class SankeyConfig:
    """Configuration for Sankey diagram."""
    metric: str
    year: int
    parent_col: str
    child_col: str
    selected_kl: str


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
# COLOR UTILITIES
# =============================================================================

class ColorUtils:
    """Utility class for color manipulation."""
    
    @staticmethod
    def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
        """Convert hex color to RGB tuple."""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    @staticmethod
    def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
        """Convert RGB tuple to hex color."""
        return '#{:02x}{:02x}{:02x}'.format(*rgb)
    
    @classmethod
    def lighten(cls, hex_color: str, factor: float = 0.3) -> str:
        """
        Lighten a hex color by the given factor.
        
        Args:
            hex_color: Hex color string (e.g., "#005FAC")
            factor: Lightening factor (0.0 to 1.0)
            
        Returns:
            Lightened hex color string
        """
        rgb = cls.hex_to_rgb(hex_color)
        light_rgb = tuple(min(255, int(c + (255 - c) * factor)) for c in rgb)
        return cls.rgb_to_hex(light_rgb)


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
            
            # Ensure Tahun is numeric
            if "Tahun" in df.columns:
                df["Tahun"] = pd.to_numeric(df["Tahun"], errors="coerce").astype("Int64")
            
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
# TIME SERIES CHART BUILDER
# =============================================================================

class TimeSeriesChartBuilder:
    """Builds time series comparison charts with bar and scatter traces."""
    
    def __init__(self, config: AppConfig = AppConfig()):
        self.config = config
    
    def create_chart(
        self,
        df: pd.DataFrame,
        ts_config: TimeSeriesConfig
    ) -> go.Figure:
        """
        Create time series chart comparing two metrics.
        
        Args:
            df: Source DataFrame
            ts_config: Chart configuration
            
        Returns:
            Plotly Figure with bar and scatter traces
        """
        # Filter and aggregate data
        agg = self._prepare_data(df, ts_config)
        
        if agg.empty:
            return self._create_empty_chart("Tidak ada data untuk ditampilkan")
        
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add traces
        self._add_bar_trace(fig, agg, ts_config.primary_metric)
        self._add_scatter_trace(fig, agg, ts_config.secondary_metric)
        
        # Apply layout
        self._apply_layout(fig, agg, ts_config)
        
        return fig
    
    def _prepare_data(
        self,
        df: pd.DataFrame,
        ts_config: TimeSeriesConfig
    ) -> pd.DataFrame:
        """Filter and aggregate data for the chart."""
        filtered_df = df.copy()
        
        # Year filter
        filtered_df = filtered_df[
            (filtered_df[self.config.YEAR_COLUMN] >= ts_config.year_range[0]) &
            (filtered_df[self.config.YEAR_COLUMN] <= ts_config.year_range[1])
        ]
        
        # K/L filter
        if ts_config.selected_kl != "Semua":
            filtered_df = filtered_df[
                filtered_df[self.config.KL_COLUMN] == ts_config.selected_kl
            ]
        
        if filtered_df.empty:
            return pd.DataFrame()
        
        # Aggregate by year
        agg = filtered_df.groupby(
            self.config.YEAR_COLUMN, as_index=False
        )[[ts_config.secondary_metric, ts_config.primary_metric]].sum()
        
        # Calculate realization percentage
        agg["Persentase Realisasi"] = np.where(
            agg[ts_config.primary_metric] > 0,
            (agg[ts_config.secondary_metric] / agg[ts_config.primary_metric] * 100),
            0
        )
        
        agg = agg.sort_values(self.config.YEAR_COLUMN)
        agg[self.config.YEAR_COLUMN] = agg[self.config.YEAR_COLUMN].astype(str)
        
        return agg
    
    def _add_bar_trace(
        self,
        fig: go.Figure,
        agg: pd.DataFrame,
        metric: str
    ) -> None:
        """Add bar chart trace for primary metric."""
        # Create display name
        display_name = metric.replace("PAGU DIPA REVISI EFEKTIF", "Pagu DIPA Revisi (Efektif)")
        
        fig.add_trace(
            go.Bar(
                x=agg[self.config.YEAR_COLUMN],
                y=agg[metric],
                name=display_name,
                marker=dict(color=self.config.PRIMARY_BAR_COLOR),
                text=agg[metric].apply(Formatter.to_rupiah_short),
                textposition="outside",
                textfont=dict(size=11),
                hovertemplate=f"Pagu: %{{y:,.0f}}<extra></extra>"
            ),
            secondary_y=False
        )
    
    def _add_scatter_trace(
        self,
        fig: go.Figure,
        agg: pd.DataFrame,
        metric: str
    ) -> None:
        """Add scatter plot trace for secondary metric."""
        # Create display name
        display_name = metric.replace("REALISASI BELANJA KL (SAKTI)", "Realisasi Belanja")
        
        # Create percentage labels
        scatter_texts = [f"({pct:.1f}%)" for pct in agg["Persentase Realisasi"]]
        
        fig.add_trace(
            go.Scatter(
                x=agg[self.config.YEAR_COLUMN],
                y=agg[metric],
                mode="markers+lines+text",
                name=display_name,
                marker=dict(
                    color=self.config.SECONDARY_MARKER_COLOR,
                    size=12,
                    line=dict(color="white", width=1.5)
                ),
                line=dict(color=self.config.SECONDARY_MARKER_COLOR, width=2),
                text=scatter_texts,
                textposition="bottom center",
                textfont=dict(size=10, color="lightgrey"),
                customdata=agg["Persentase Realisasi"],
                hovertemplate=f"Realisasi: %{{y:,.0f}} (%{{customdata:.2f}}%)<extra></extra>"
            ),
            secondary_y=True
        )
    
    def _apply_layout(
        self,
        fig: go.Figure,
        agg: pd.DataFrame,
        ts_config: TimeSeriesConfig
    ) -> None:
        """Apply layout settings to the figure."""
        # Calculate max value for synchronized y-axes
        max_value = max(
            agg[ts_config.primary_metric].max(),
            agg[ts_config.secondary_metric].max()
        )
        
        # Build title
        title_kl = f" {ts_config.selected_kl}" if ts_config.selected_kl != "Semua" else ""
        title_text = (
            f"PERBANDINGAN {ts_config.primary_metric} TERHADAP {ts_config.secondary_metric}<br>"
            f"PERIODE {ts_config.year_range[0]} - {ts_config.year_range[1]}<br>{title_kl}"
        )
        
        fig.update_layout(
            title=dict(
                text=title_text,
                x=0.5,
                xanchor='center',
                font=dict(size=14)
            ),
            xaxis_title="Tahun",
            template="plotly_white",
            height=self.config.TIME_SERIES_HEIGHT,
            margin=dict(t=130, b=30, l=30, r=30),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            hovermode="x unified"
        )
        
        # Synchronized y-axes (hidden labels)
        y_range = [0, max_value * 1.1]
        fig.update_yaxes(
            showticklabels=False,
            title_text="",
            secondary_y=False,
            range=y_range
        )
        fig.update_yaxes(
            showticklabels=False,
            title_text="",
            secondary_y=True,
            range=y_range
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
            height=self.config.TIME_SERIES_HEIGHT,
            plot_bgcolor="white",
            paper_bgcolor="white"
        )
        return fig


# =============================================================================
# SANKEY CHART BUILDER
# =============================================================================

class SankeyChartBuilder:
    """Builds Sankey diagrams for budget flow visualization."""
    
    def __init__(self, config: AppConfig = AppConfig()):
        self.config = config
    
    def create_chart(
        self,
        df: pd.DataFrame,
        sankey_config: SankeyConfig
    ) -> go.Figure:
        """
        Create Sankey diagram showing budget flow.
        
        Args:
            df: Source DataFrame
            sankey_config: Chart configuration
            
        Returns:
            Plotly Figure with Sankey diagram
        """
        # Filter data
        df_filtered = self._filter_data(df, sankey_config)
        
        if df_filtered.empty or sankey_config.metric not in df_filtered.columns:
            return self._create_empty_chart("Tidak ada data untuk ditampilkan")
        
        # Calculate aggregations
        aggregations = self._calculate_aggregations(df_filtered, sankey_config)
        
        if aggregations['total'] <= 0:
            return self._create_empty_chart("Tidak ada data untuk ditampilkan")
        
        # Build Sankey components
        labels, node_colors, node_positions = self._build_nodes(aggregations, sankey_config)
        sources, targets, values, link_labels = self._build_links(aggregations, labels, sankey_config)
        node_hover_texts = self._build_node_hover_texts(labels, aggregations, sankey_config)
        
        # Create figure
        fig = self._create_sankey_figure(
            labels, node_colors, node_positions,
            sources, targets, values, link_labels,
            node_hover_texts, sankey_config, aggregations
        )
        
        return fig
    
    def _filter_data(
        self,
        df: pd.DataFrame,
        sankey_config: SankeyConfig
    ) -> pd.DataFrame:
        """Filter data by K/L and year."""
        df_filtered = df.copy()
        
        # K/L filter
        if sankey_config.selected_kl != "Semua":
            df_filtered = df_filtered[
                df_filtered[self.config.KL_COLUMN] == sankey_config.selected_kl
            ]
        
        # Year filter
        df_filtered = df_filtered[
            df_filtered[self.config.YEAR_COLUMN] == sankey_config.year
        ]
        
        return df_filtered
    
    def _calculate_aggregations(
        self,
        df: pd.DataFrame,
        sankey_config: SankeyConfig
    ) -> Dict[str, Any]:
        """Calculate aggregations for Sankey diagram."""
        metric = sankey_config.metric
        parent_col = sankey_config.parent_col
        child_col = sankey_config.child_col
        
        # Parent aggregation
        agg_parent = (
            df.groupby(parent_col, as_index=False)[metric]
            .sum()
            .query(f"`{metric}` > 0")
        )
        
        # Parent-Child aggregation
        agg_parent_child = (
            df.groupby([parent_col, child_col], as_index=False)[metric]
            .sum()
            .query(f"`{metric}` > 0")
        )
        
        # Child aggregation
        agg_child = (
            df.groupby(child_col, as_index=False)[metric]
            .sum()
            .query(f"`{metric}` > 0")
        )
        
        # Total
        total = df[metric].sum()
        
        return {
            'parent': agg_parent,
            'parent_child': agg_parent_child,
            'child': agg_child,
            'total': total,
            'parent_list': agg_parent[parent_col].astype(str).tolist(),
            'child_list': agg_child[child_col].astype(str).tolist()
        }
    
    def _build_nodes(
        self,
        aggregations: Dict[str, Any],
        sankey_config: SankeyConfig
    ) -> Tuple[List[str], List[str], Tuple[List[float], List[float]]]:
        """Build node labels, colors, and positions."""
        parent_list = aggregations['parent_list']
        child_list = aggregations['child_list']
        
        # Build labels
        total_label = sankey_config.metric
        labels = [total_label] + parent_list + child_list
        
        # Build colors
        base_color = self.config.PRIMARY_BAR_COLOR
        child_color = ColorUtils.lighten(base_color, 0.4)
        
        node_colors = [base_color]  # Total
        node_colors += [base_color] * len(parent_list)  # Parents
        node_colors += [child_color] * len(child_list)  # Children
        
        # Build positions
        node_x, node_y = self._calculate_node_positions(
            len(parent_list), len(child_list)
        )
        
        return labels, node_colors, (node_x, node_y)
    
    def _calculate_node_positions(
        self,
        num_parents: int,
        num_children: int
    ) -> Tuple[List[float], List[float]]:
        """Calculate x, y positions for Sankey nodes."""
        node_x = []
        node_y = []
        
        # Total node (leftmost, center)
        node_x.append(0.01)
        node_y.append(0.5)
        
        # Parent nodes (middle column)
        parent_y = self._distribute_y_positions(num_parents)
        for y_pos in parent_y:
            node_x.append(0.25)
            node_y.append(y_pos)
        
        # Child nodes (rightmost column)
        child_y = self._distribute_y_positions(num_children)
        for y_pos in child_y:
            node_x.append(0.99)
            node_y.append(y_pos)
        
        # Clamp values to avoid edge issues
        node_x = [0.001 if v <= 0 else 0.999 if v >= 1 else v for v in node_x]
        node_y = [0.001 if v <= 0 else 0.999 if v >= 1 else v for v in node_y]
        
        return node_x, node_y
    
    def _distribute_y_positions(self, n: int) -> List[float]:
        """
        Distribute n nodes vertically, centered around 0.5.
        
        Creates a visually balanced distribution with a gap around the center.
        """
        if n == 0:
            return []
        if n == 1:
            return [0.5]
        
        gap = 0.02  # Gap around center for visual separation
        
        if n % 2 == 0:  # Even number
            half = n // 2
            positions = []
            
            # Bottom half: 0.1 to (0.5 - gap)
            bottom_range = (0.5 - gap) - 0.1
            if half > 1:
                positions += [0.1 + bottom_range * i / (half - 1) for i in range(half)]
            else:
                positions.append(0.5 - gap - bottom_range / 2)
            
            # Top half: (0.5 + gap) to 0.9
            top_range = 0.9 - (0.5 + gap)
            if half > 1:
                positions += [0.5 + gap + top_range * i / (half - 1) for i in range(half)]
            else:
                positions.append(0.5 + gap + top_range / 2)
        else:  # Odd number
            half = n // 2
            positions = []
            
            # Bottom half
            if half > 0:
                bottom_range = (0.5 - gap) - 0.1
                if half > 1:
                    positions += [0.1 + bottom_range * i / (half - 1) for i in range(half)]
                else:
                    positions.append(0.3)
            
            # Center node
            positions.append(0.5)
            
            # Top half
            if half > 0:
                top_range = 0.9 - (0.5 + gap)
                if half > 1:
                    positions += [0.5 + gap + top_range * i / (half - 1) for i in range(half)]
                else:
                    positions.append(0.7)
        
        return positions
    
    def _build_links(
        self,
        aggregations: Dict[str, Any],
        labels: List[str],
        sankey_config: SankeyConfig
    ) -> Tuple[List[int], List[int], List[float], List[str]]:
        """Build link sources, targets, values, and labels."""
        metric = sankey_config.metric
        parent_col = sankey_config.parent_col
        child_col = sankey_config.child_col
        total_label = metric
        
        index_map = {lab: i for i, lab in enumerate(labels)}
        
        sources, targets, values, link_labels = [], [], [], []
        
        # Total -> Parent links
        for _, row in aggregations['parent'].iterrows():
            parent = str(row[parent_col])
            value = float(row[metric])
            if value <= 0:
                continue
            
            sources.append(index_map[total_label])
            targets.append(index_map[parent])
            values.append(value)
            link_labels.append(f"{total_label} → {parent}: {Formatter.to_rupiah_short(value)}")
        
        # Parent -> Child links
        for _, row in aggregations['parent_child'].iterrows():
            parent = str(row[parent_col])
            child = str(row[child_col])
            value = float(row[metric])
            if value <= 0:
                continue
            
            sources.append(index_map[parent])
            targets.append(index_map[child])
            values.append(value)
            link_labels.append(f"{parent} → {child}: {Formatter.to_rupiah_short(value)}")
        
        return sources, targets, values, link_labels
    
    def _build_node_hover_texts(
        self,
        labels: List[str],
        aggregations: Dict[str, Any],
        sankey_config: SankeyConfig
    ) -> List[str]:
        """Build hover text for each node."""
        metric = sankey_config.metric
        parent_col = sankey_config.parent_col
        child_col = sankey_config.child_col
        total = aggregations['total']
        
        hover_texts = []
        
        # Calculate node values
        node_values = [total]  # Total
        
        for parent in aggregations['parent_list']:
            val = float(
                aggregations['parent'][
                    aggregations['parent'][parent_col] == parent
                ][metric].iloc[0]
            )
            node_values.append(val)
        
        for child in aggregations['child_list']:
            val = float(
                aggregations['child'][
                    aggregations['child'][child_col] == child
                ][metric].iloc[0]
            )
            node_values.append(val)
        
        # Build hover texts
        for label, value in zip(labels, node_values):
            percentage = (value / total) * 100 if total > 0 else 0
            hover_texts.append(
                f"<b>{label}</b><br>"
                f"{Formatter.to_rupiah_short(value)}<br>"
                f"{percentage:.1f}% dari {metric}"
            )
        
        return hover_texts
    
    def _create_sankey_figure(
        self,
        labels: List[str],
        node_colors: List[str],
        node_positions: Tuple[List[float], List[float]],
        sources: List[int],
        targets: List[int],
        values: List[float],
        link_labels: List[str],
        node_hover_texts: List[str],
        sankey_config: SankeyConfig,
        aggregations: Dict[str, Any]
    ) -> go.Figure:
        """Create the Sankey figure with all components."""
        node_x, node_y = node_positions
        
        sankey = go.Sankey(
            arrangement="snap",
            node=dict(
                label=labels,
                color=node_colors,
                pad=15,
                thickness=20,
                line=dict(color="white", width=1),
                customdata=node_hover_texts,
                hovertemplate="%{customdata}<extra></extra>",
                x=node_x,
                y=node_y,
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                customdata=link_labels,
                hovertemplate="%{customdata}<extra></extra>",
                color=self.config.SANKEY_LINK_COLOR,
                hovercolor=self.config.SANKEY_LINK_HOVER_COLOR,
            )
        )
        
        fig = go.Figure(sankey)
        
        # Build title
        kl_text = (
            f"<br>{sankey_config.selected_kl}"
            if sankey_config.selected_kl != "Semua"
            else ""
        )
        
        title_text = (
            f"ALOKASI {sankey_config.metric}<br>"
            f"BERDASARKAN {sankey_config.parent_col} & {sankey_config.child_col}<br>"
            f"TAHUN {sankey_config.year}{kl_text}"
        )
        
        # Calculate dynamic height
        total_nodes = 1 + len(aggregations['parent_list']) + len(aggregations['child_list'])
        chart_height = max(
            self.config.SANKEY_MIN_HEIGHT,
            min(self.config.SANKEY_MAX_HEIGHT, total_nodes * 35)
        )
        
        fig.update_layout(
            title=dict(
                text=title_text,
                x=0.5,
                xanchor='center',
                font=dict(size=14)
            ),
            font=dict(size=9),
            height=chart_height,
            margin=dict(l=20, r=20, t=130, b=20)
        )
        
        fig.update_traces(
            textfont_color=self.config.PRIMARY_BAR_COLOR,
            node_align="center"
        )
        
        return fig
    
    def _create_empty_chart(self, message: str) -> go.Figure:
        """Create an empty chart with a message."""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=9)
        )
        fig.update_layout(height=self.config.SANKEY_MIN_HEIGHT)
        return fig


# =============================================================================
# UI COMPONENTS
# =============================================================================

class UIComponents:
    """Reusable UI component generators."""
    
    @staticmethod
    def render_header(selected_kl: Optional[str]) -> None:
        """Render the main dashboard header."""
        kl_text = "Semua K/L" if selected_kl == "Semua" else (selected_kl or "—")
        
        st.markdown(f"""
        <div class="dashboard-header" role="banner">
            <div class="breadcrumb">DASHBOARD / ANALISIS / {kl_text}</div>
            <h1 class="dashboard-title">📊 Dashboard Analisis Anggaran & Realisasi Belanja Negara</h1>
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


class SidebarController:
    """Manages sidebar filter controls."""
    
    def __init__(self, df: pd.DataFrame, config: AppConfig = AppConfig()):
        self.df = df
        self.config = config
    
    def render(self) -> str:
        """Render sidebar and return selected K/L."""
        with st.sidebar:
            st.markdown("""
            <div class="sidebar-section">
                <h3 style='margin: 0.1rem; color: var(--on-surface);'>Filter Data</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # K/L selector
            kl_list = sorted(self.df[self.config.KL_COLUMN].dropna().unique())
            kl_list.insert(0, "Semua")
            
            selected_kl = st.selectbox(
                "Pilih Kementerian/Lembaga",
                kl_list,
                key="ministry_select",
                help="Pilih kementerian/lembaga untuk melihat analisis anggaran"
            )
        
        return selected_kl


# =============================================================================
# CHART CONTROL PANELS
# =============================================================================

class TimeSeriesControlPanel:
    """Control panel for time series chart configuration."""
    
    def __init__(self, df: pd.DataFrame, config: AppConfig = AppConfig()):
        self.df = df
        self.config = config
    
    def render(self, selected_kl: str) -> TimeSeriesConfig:
        """Render controls and return configuration."""
        # Filter by K/L for year options
        df_filtered = self.df.copy()
        if selected_kl != "Semua":
            df_filtered = df_filtered[
                df_filtered[self.config.KL_COLUMN] == selected_kl
            ]
        
        # Get numeric columns
        numeric_cols = df_filtered.select_dtypes(include=["int64", "float64", "Int64"]).columns.tolist()
        if self.config.YEAR_COLUMN in numeric_cols:
            numeric_cols.remove(self.config.YEAR_COLUMN)
        
        # Year range slider
        year_options = sorted(df_filtered[self.config.YEAR_COLUMN].dropna().unique())
        
        if len(year_options) >= 2:
            year_range = st.slider(
                "Rentang Tahun",
                min_value=int(min(year_options)),
                max_value=int(max(year_options)),
                value=(int(min(year_options)), int(max(year_options))),
                step=1,
                key="year_range_main"
            )
        else:
            single_year = int(year_options[0]) if year_options else 2025
            st.info("Hanya satu tahun tersedia dalam data")
            year_range = (single_year, single_year)
        
        # Metric selectors
        colA, colB = st.columns(2)
        
        with colA:
            primary_idx = (
                numeric_cols.index(self.config.DEFAULT_PRIMARY_METRIC)
                if self.config.DEFAULT_PRIMARY_METRIC in numeric_cols
                else 0
            )
            primary_metric = st.selectbox(
                "Pilih metrik pertama",
                numeric_cols,
                index=primary_idx,
                key="primary_metric"
            )
        
        with colB:
            secondary_idx = (
                numeric_cols.index(self.config.DEFAULT_SECONDARY_METRIC)
                if self.config.DEFAULT_SECONDARY_METRIC in numeric_cols
                else 0
            )
            secondary_metric = st.selectbox(
                "Pilih metrik kedua",
                numeric_cols,
                index=secondary_idx,
                key="secondary_metric"
            )
        
        return TimeSeriesConfig(
            primary_metric=primary_metric,
            secondary_metric=secondary_metric,
            year_range=year_range,
            selected_kl=selected_kl
        )


class SankeyControlPanel:
    """Control panel for Sankey chart configuration."""
    
    def __init__(self, df: pd.DataFrame, config: AppConfig = AppConfig()):
        self.df = df
        self.config = config
    
    def render(self, selected_kl: str) -> SankeyConfig:
        """Render controls and return configuration."""
        # Filter by K/L
        df_filtered = self.df.copy()
        if selected_kl != "Semua":
            df_filtered = df_filtered[
                df_filtered[self.config.KL_COLUMN] == selected_kl
            ]
        
        # Get column options
        numeric_cols = df_filtered.select_dtypes(include=["int64", "float64", "Int64"]).columns.tolist()
        if self.config.YEAR_COLUMN in numeric_cols:
            numeric_cols.remove(self.config.YEAR_COLUMN)
        
        categorical_cols = df_filtered.select_dtypes(include=['object']).columns.tolist()
        exclude_cols = [self.config.KL_COLUMN, self.config.YEAR_COLUMN]
        categorical_cols = [col for col in categorical_cols if col not in exclude_cols]
        
        # Row 1: Year and Metric
        colC, colD = st.columns(2)
        
        with colC:
            year_options = sorted(int(y) for y in df_filtered[self.config.YEAR_COLUMN].dropna().unique())
            if not year_options:
                year_options = [date.today().year]
                
            current_year = date.today().year
            default_index = year_options.index(current_year) if current_year in year_options else -1
            
            selected_year = st.selectbox(
                "Tahun",
                year_options,
                index=default_index,
                key="year_sankey"
            )
            
        with colD:
            metric_idx = (
                numeric_cols.index(self.config.DEFAULT_SECONDARY_METRIC)
                if self.config.DEFAULT_SECONDARY_METRIC in numeric_cols
                else 0
            )
            selected_metric = st.selectbox(
                "Metrik",
                numeric_cols,
                index=metric_idx,
                key="metric_sankey"
            )
        
        # Row 2: Parent and Child
        colE, colF = st.columns(2)
        
        with colE:
            parent_idx = (
                categorical_cols.index(self.config.DEFAULT_SANKEY_PARENT)
                if self.config.DEFAULT_SANKEY_PARENT in categorical_cols
                else 0
            )
            parent_col = st.selectbox(
                "Parent",
                categorical_cols,
                index=parent_idx,
                key="parent_sankey"
            )
        
        with colF:
            child_idx = (
                categorical_cols.index(self.config.DEFAULT_SANKEY_CHILD)
                if self.config.DEFAULT_SANKEY_CHILD in categorical_cols
                else 0
            )
            child_col = st.selectbox(
                "Child",
                categorical_cols,
                index=child_idx,
                key="child_sankey"
            )
        
        return SankeyConfig(
            metric=selected_metric,
            year=int(selected_year),
            parent_col=parent_col,
            child_col=child_col,
            selected_kl=selected_kl
        )


# =============================================================================
# MAIN APPLICATION
# =============================================================================

class BudgetAnalysisDashboard:
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
            st.warning("⚠️ Tidak ada data untuk ditampilkan")
            return
        
        # Sidebar
        sidebar = SidebarController(df, self.config)
        selected_kl = sidebar.render()
        
        # Header
        UIComponents.render_header(selected_kl)
        
        # Main content - two columns
        col1, col2 = st.columns(2)
        
        # Column 1: Time Series Chart
        with col1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            
            ts_panel = TimeSeriesControlPanel(df, self.config)
            ts_config = ts_panel.render(selected_kl)
            
            ts_builder = TimeSeriesChartBuilder(self.config)
            fig_ts = ts_builder.create_chart(df, ts_config)
            st.plotly_chart(fig_ts, use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Column 2: Sankey Chart
        with col2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            
            sankey_panel = SankeyControlPanel(df, self.config)
            sankey_config = sankey_panel.render(selected_kl)
            
            sankey_builder = SankeyChartBuilder(self.config)
            fig_sankey = sankey_builder.create_chart(df, sankey_config)
            
            with st.container(height=self.config.SANKEY_CONTAINER_HEIGHT, border=False):
                st.plotly_chart(fig_sankey, use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
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

# =============================================================================
# ENTRY POINT
# =============================================================================

def main() -> None:
    """Application entry point."""
    try:
        app = BudgetAnalysisDashboard()
        app.run()
    except Exception as e:
        st.error(f"Terjadi kesalahan dalam aplikasi: {str(e)}")
        st.info("Silakan refresh halaman atau hubungi administrator.")


if __name__ == "__main__":
    main()








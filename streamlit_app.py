import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

# Set up page configuration
st.set_page_config(
    page_title="Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
/* Main container styling */
.stApp {
    background-color: #f8fafc;
    font-family: 'Inter', sans-serif;
}

/* Card styling */
.card {
    background-color: white;
    border-radius: 12px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
    padding: 24px;
    margin-bottom: 24px;
    transition: all 0.3s ease;
}

.card:hover {
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.12);
}

/* Metric card styling */
.metric-card {
    background-color: white;
    border-radius: 12px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
    padding: 24px;
    text-align: center;
    transition: transform 0.2s;
    height: 100%;
}

.metric-card:hover {
    transform: translateY(-3px);
}

/* Chart container styling */
.chart-container {
    background-color: white;
    border-radius: 12px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
    padding: 24px;
    margin-bottom: 24px;
}

/* Sidebar styling */
.stSidebar {
    background-color: #ffffff;
    border-right: 1px solid #e2e8f0;
}

/* Button styling */
.stButton>button {
    background-color: #3b82f6;
    color: white;
    border-radius: 8px;
    padding: 10px 20px;
    font-weight: 500;
    transition: background-color 0.3s ease;
}

.stButton>button:hover {
    background-color: #2563eb;
}

/* Tab styling */
.tab-content {
    background-color: white;
    border-radius: 12px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
    padding: 24px;
    margin-top: 10px;
}

/* Typography */
h1 {
    color: #1e293b;
    font-weight: 700;
    margin-bottom: 24px;
}

h2 {
    color: #334155;
    font-weight: 600;
    margin-bottom: 20px;
}

h3 {
    color: #475569;
    font-weight: 600;
    margin-bottom: 12px;
}

p {
    color: #64748b;
    font-size: 14px;
    margin: 0;
}

/* Progress bar styling */
.progress-bar {
    height: 8px;
    border-radius: 4px;
    background-color: #e2e8f0;
}

.progress-fill {
    height: 100%;
    border-radius: 4px;
    background-color: #3b82f6;
}

/* Responsive adjustments */
@media (max-width: 768px) {
    .card {
        padding: 16px;
    }
    
    .metric-card {
        padding: 16px;
    }
    
    .chart-container {
        padding: 16px;
    }
}
</style>
""", unsafe_allow_html=True)

# Title with icon
st.markdown("<h1 style='color: #1e293b; font-weight: 700;'>📊 Dashboard Analisis Keuangan</h1>", unsafe_allow_html=True)

# Load data function
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("df23-25.csv")
        
        # Remove index column if exists
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])
        
        # Ensure Tahun is string
        if "Tahun" in df.columns:
            df["Tahun"] = df["Tahun"].astype(str)
            
        return df
    except Exception as e:
        st.error(f"Gagal memuat data: {str(e)}")
        return pd.DataFrame()

df = load_data()

# Check if data loaded successfully
if df.empty:
    st.stop()

# Sidebar filters with better organization
with st.sidebar:
    st.markdown("<h2 style='font-size: 18px; margin-bottom: 24px; color: #334155;'>🔍 Filter Data</h2>", unsafe_allow_html=True)
    
    # Ministry selection
    kl_list = sorted(df["KEMENTERIAN/LEMBAGA"].dropna().unique())
    selected_kl = st.selectbox("Pilih Kementerian/Lembaga", kl_list, key="ministry_select")
    
    # Metric selection
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    metric_options = numeric_cols if numeric_cols else ["(Tidak ada kolom numerik ditemukan)"]
    selected_metric = st.selectbox("Pilih Jenis Nilai", metric_options, key="metric_select")
    
    # Add some space
    st.markdown("<div style='margin-top: 40px;'></div>", unsafe_allow_html=True)
    
    # Additional filter options could be added here
    
    # Footer in sidebar
    st.markdown("---")
    st.caption("Data: bidja.kemenkeu.go.id | Dibuat dengan Streamlit")

# Filter and prepare data
df_filtered = df[df["KEMENTERIAN/LEMBAGA"] == selected_kl].copy()
df_filtered = df_filtered.rename(columns={selected_metric: "Nilai"})

# Helper functions
def format_rupiah(value):
    """Format currency values"""
    if value >= 1_000_000_000_000:
        return f"{value/1_000_000_000_000:.2f} T"
    elif value >= 1_000_000_000:
        return f"{value/1_000_000_000:.2f} M"
    elif value >= 1_000_000:
        return f"{value/1_000_000:.2f} Jt"
    else:
        return f"{value:,.0f}"

def create_summary_cards(df):
    """Create summary cards for key metrics"""
    total_value = df["Nilai"].sum()
    avg_value = df["Nilai"].mean()
    max_value = df["Nilai"].max()
    
    # Calculate year-over-year growth
    yearly_sums = df.groupby("Tahun")["Nilai"].sum()
    if len(yearly_sums) > 1:
        latest_year = yearly_sums.index[-1]
        previous_year = yearly_sums.index[-2]
        growth_rate = ((yearly_sums[latest_year] - yearly_sums[previous_year]) / yearly_sums[previous_year]) * 100
    else:
        growth_rate = 0
    
    # Create cards
    cols = st.columns(4)
    
    with cols[0]:
        st.markdown(f"""
        <div class='metric-card'>
            <h3>Total Nilai</h3>
            <p style='font-size: 24px; font-weight: 700; color: #1e293b;'>Rp {format_rupiah(total_value)}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[1]:
        st.markdown(f"""
        <div class='metric-card'>
            <h3>Rata-rata Nilai</h3>
            <p style='font-size: 24px; font-weight: 700; color: #1e293b;'>Rp {format_rupiah(avg_value)}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[2]:
        st.markdown(f"""
        <div class='metric-card'>
            <h3>Maksimum Nilai</h3>
            <p style='font-size: 24px; font-weight: 700; color: #1e293b;'>Rp {format_rupiah(max_value)}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[3]:
        st.markdown(f"""
        <div class='metric-card'>
            <h3>Pertumbuhan YoY</h3>
            <p style='font-size: 24px; font-weight: 700; color: {'#22c55e' if growth_rate >= 0 else '#ef4444'};'>{growth_rate:+.1f}%</p>
        </div>
        """, unsafe_allow_html=True)

def make_line_chart(df, group_col, base_height=600, extra_height_per_line=10):
    """Create line chart with improved formatting"""
    df_grouped = (
        df.groupby(["KEMENTERIAN/LEMBAGA", "Tahun", group_col], as_index=False)["Nilai"]
          .sum()
    )
    
    # Ensure Tahun is string
    df_grouped["Tahun"] = df_grouped["Tahun"].astype(str)
    
    # Adjust height based on number of groups
    n_groups = df_grouped[group_col].nunique()
    height = base_height + (n_groups * extra_height_per_line if n_groups > 10 else 0)
    
    fig = px.line(
        df_grouped,
        x="Tahun",
        y="Nilai",
        color=group_col,
        markers=True,
        title=f"{selected_metric} per {group_col} — {selected_kl}",
        labels={
            "Tahun": "Tahun",
            "Nilai": "Jumlah (Rp)",
            group_col: group_col.replace("_", " ").title(),
        },
        template="plotly_white"
    )
    
    # Update y-axis with Indonesian currency format
    fig.update_yaxes(
        tickprefix="Rp ",
        tickformat=",",
        tickvals=None,
        ticktext=None,
    )
    
    # Update hover template
    fig.update_traces(
        hovertemplate="<b>Tahun: %{x}</b><br>" +
                     "%{fullData.name}: Rp %{y:,.0f}<extra></extra>"
    )
    
    # Update layout
    fig.update_layout(
        height=height,
        hovermode="closest",
        title_x=0,
        legend_title_text=group_col.replace("_", " ").title(),
        margin=dict(l=40, r=40, t=80, b=40),
        paper_bgcolor="white",
        plot_bgcolor="white"
    )
    
    # Format y-axis labels
    y_ticks = fig.data[0].y
    fig.update_yaxes(
        ticktext=[format_rupiah(v) for v in sorted(set(y_ticks))],
        tickvals=sorted(set(y_ticks))
    )
    
    return fig

# Main content area
st.markdown("<div class='card'>", unsafe_allow_html=True)
create_summary_cards(df_filtered)
st.markdown("</div>", unsafe_allow_html=True)

# Display charts
st.markdown("<h2 style='color: #334155; font-weight: 600; margin-top: 32px;'>Visualisasi Data</h2>", unsafe_allow_html=True)

# Get categorical columns
cat_cols = [
    col for col in df_filtered.select_dtypes(include=["object"]).columns
    if col not in ["KEMENTERIAN/LEMBAGA", "Tahun"]
]

if cat_cols:
    tabs = st.tabs([f"📊 {col.replace('_', ' ').title()}" for col in cat_cols])
    
    for tab, col in zip(tabs, cat_cols):
        with tab:
            st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
            st.plotly_chart(make_line_chart(df_filtered, col), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
else:
    st.warning("Tidak ada kolom kategorikal yang tersedia untuk divisualisasikan.")

# Footer
st.markdown("---")
st.caption("Data: bidja.kemenkeu.go.id")

import streamlit as st
import pandas as pd
import plotly.express as px

# === Page config ===
st.set_page_config(
    page_title="Dashboard Realisasi Belanja KL",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Dashboard Realisasi Belanja KL 2023–2025")

# === Load data ===
@st.cache_data
def load_data():
    df = pd.read_csv("df23-25.csv")

    # 1️⃣ Remove index column if exists
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # 2️⃣ Ensure Tahun is string (important for plotting)
    if "Tahun" in df.columns:
        df["Tahun"] = df["Tahun"].astype(str)

    return df

df = load_data()

# === Sidebar filters ===
st.sidebar.header("🔍 Filter")

kl_list = sorted(df["KEMENTERIAN/LEMBAGA"].dropna().unique())
selected_kl = st.sidebar.selectbox("Pilih Kementerian/Lembaga", kl_list)

# Automatically detect numeric columns for metrics
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
metric_options = numeric_cols if numeric_cols else ["(Tidak ada kolom numerik ditemukan)"]

selected_metric = st.sidebar.selectbox("Pilih Jenis Nilai (Metric)", metric_options)

# === Filter and prepare data ===
df_filtered = df[df["KEMENTERIAN/LEMBAGA"] == selected_kl].copy()
df_filtered = df_filtered.rename(columns={selected_metric: "Nilai"})

# === Helper: format Rupiah values into Jt, M, T ===
def format_rupiah(value):
    if value >= 1_000_000_000_000:
        return f"{value/1_000_000_000_000:.2f} T"
    elif value >= 1_000_000_000:
        return f"{value/1_000_000_000:.2f} M"
    elif value >= 1_000_000:
        return f"{value/1_000_000:.2f} Jt"
    else:
        return f"{value:,.0f}"

# === Function to make line charts ===
def make_line_chart(df, group_col, base_height=600, extra_height_per_line=10):
    df_grouped = (
        df.groupby(["KEMENTERIAN/LEMBAGA", "Tahun", group_col], as_index=False)["Nilai"]
          .sum()
    )

    # 3️⃣ Convert Tahun to string again (ensure categorical x-axis)
    df_grouped["Tahun"] = df_grouped["Tahun"].astype(str)

    # Adjust height if too many groups
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
    )

    # 4️⃣ Format y-axis with Indonesian short scale
    fig.update_yaxes(
        tickprefix="Rp ",
        tickformat=",",
        tickvals=None,
        ticktext=None,
    )
    fig.update_traces(
        hovertemplate="<b>%{x}</b><br>%{fullData.name}: Rp %{y:,.0f}"
    )

    # Custom y-axis tick labels in formatted form
    fig.update_layout(
        height=height,
        hovermode="x unified",
        title_x=0.5,
        legend_title_text=group_col.replace("_", " ").title(),
        margin=dict(l=40, r=40, t=80, b=40)
    )

    # Format y-axis labels using the custom Rupiah formatter
    y_ticks = fig.data[0].y
    fig.update_yaxes(
        ticktext=[format_rupiah(v) for v in sorted(set(y_ticks))],
        tickvals=sorted(set(y_ticks))
    )

    return fig

# === Display charts ===
# Automatically detect all categorical columns (object/string)
cat_cols = [
    col for col in df_filtered.select_dtypes(include=["object"]).columns
    if col not in ["KEMENTERIAN/LEMBAGA", "Tahun"]
]

# Create a tab for each categorical column
tabs = st.tabs([f"📊 {col.replace('_', ' ').title()}" for col in cat_cols])

# Loop through each tab and display the corresponding chart
for tab, col in zip(tabs, cat_cols):
    with tab:
        st.plotly_chart(make_line_chart(df_filtered, col), use_container_width=True)

# === Footer ===
st.markdown("---")
st.caption("Data: bidja.kemenkeu.go.id | Dashboard dibuat dengan Streamlit dan Plotly")

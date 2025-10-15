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
    return pd.read_csv("df23-25.csv")

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
df_filtered["Tahun"] = df_filtered["Tahun"].astype(str)

# === Function to make line charts ===
def make_line_chart(df, group_col, base_height=600, extra_height_per_line=10):
    df_grouped = (
        df.groupby(["KEMENTERIAN/LEMBAGA", "Tahun", group_col], as_index=False)["Nilai"]
          .sum()
    )

    n_groups = df_grouped[group_col].nunique()
    height = base_height + (n_groups * extra_height_per_line if n_groups > 10 else 0)

    fig = px.line(
        df_grouped,
        x="Tahun",
        y="Nilai",
        color=group_col,
        markers=True,
        title=f"{selected_metric} per {group_col} — {selected_kl}",
        labels={"Nilai": "Jumlah (Rp)", group_col: group_col.replace("_", " ").title()},
    )

    fig.update_layout(
        height=height,
        hovermode="x",
        title_x=0.5,
        legend_title_text=group_col.replace("_", " ").title(),
        margin=dict(l=40, r=40, t=80, b=40)
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
st.caption("Data: Kementerian/Lembaga 2023–2025 | Dashboard dibuat dengan Streamlit dan Plotly")

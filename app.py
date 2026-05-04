import streamlit as st
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.express as px
from pathlib import Path
from PIL import Image
import re

# -----------------------------
# PAGE CONFIG (HCSS STYLE)
# -----------------------------
st.set_page_config(page_title="Political Compass Explorer", layout="wide")

st.title("Political Positions of Speakers in Climate Policy")
st.markdown("Structured linguistic analysis of political positons of speakers in climate policy discussion across dimensions of social reform, climate stance, and political agency.")
logo = Image.open(Path("..")/"HCSS_Exam_2026"/"HCSS-New-Logo-Set-2021"/"HCSS_Beeldmerk_Blauw_RGB1200 ppi.png")

st.sidebar.image(logo, use_container_width=True)
st.markdown("---")

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_excel(
        Path("..") / "HCSS_Exam_2026" / "Intermediary_files" / "ParlaMint_PS2.xlsx"
    )

df = load_data().copy()

# -----------------------------
# DIMENSIONS
# -----------------------------
dimensions = {
    "Policy Aggressiveness vs Agency of Speaker": ("A_score", "D_score"),
    "Policy Aggressiveness vs Climate Stance": ("A_score", "stance_score"),
    "Agency of Speaker vs Climate Stance": ("D_score", "stance_score")
}

# -----------------------------
# CLEAN ACTOR LABELS (NO "Party:" PREFIX)
# -----------------------------
def clean_actor_name(x):
    if pd.isna(x):
        return x

    x = str(x).strip()

    # remove dataset artifacts like "party.VVD"
    if x.lower().startswith("party."):
        x = x.split(".", 1)[1]

    # special case: TK
    if x == "TK":
        return "TK (Tweede Kamer)"

    return x

# -----------------------------
# SIDEBAR CONTROLS
# -----------------------------
st.sidebar.header("Controls")

party_col = st.sidebar.selectbox(
    "Actor column",
    [c for c in df.columns if "party" in c.lower() or "Party" in c]
)

selected_dim = st.sidebar.selectbox(
    "Analysis dimension",
    list(dimensions.keys())
)

# create display column
df["actor_display"] = df[party_col].apply(clean_actor_name)

selected_actor_display = st.sidebar.selectbox(
    "Focus actor",
    sorted(df["actor_display"].dropna().unique().tolist())
)

show_background = st.sidebar.checkbox("Show other actors (faint)", value=False)
show_clusters = st.sidebar.checkbox("Show clusters", value=False)
normalize = st.sidebar.checkbox("Comparative scale to each other (Normalization)", value=True)
show_all = st.sidebar.checkbox("Show all actors", value=False)

# map display → raw
selected_actor_raw = df.loc[
    df["actor_display"] == selected_actor_display, party_col
].iloc[0]


#st.markdown("## View configuration")

x_col, y_col = dimensions[selected_dim]
df_plot = df.copy()

# -----------------------------
# MODE: RAW vs NORMALIZED
# -----------------------------
if normalize:

    for col in [x_col, y_col]:
        low = df_plot[col].quantile(0.05)
        high = df_plot[col].quantile(0.95)

        if high - low != 0:
            df_plot[col] = (df_plot[col] - low) / (high - low)
            df_plot[col] = df_plot[col].clip(0, 1)
            df_plot[col] = df_plot[col] * 2 - 1
        else:
            df_plot[col] = 0

    x_range = [-1, 1]
    y_range = [-1, 1]

else:
    x_range = [df_plot[x_col].min(), df_plot[x_col].max()]
    y_range = [df_plot[y_col].min(), df_plot[y_col].max()]

# -----------------------------
# CLUSTERING (OPTIONAL)
# -----------------------------
if show_clusters:
    X = df_plot[[x_col, y_col]].values
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df_plot["cluster"] = kmeans.fit_predict(X)
else:
    df_plot["cluster"] = None

# -----------------------------
# DATA SELECTION LOGIC
# -----------------------------
if show_all:
    plot_df = df_plot
else:
    plot_df = df_plot[df_plot[party_col] == selected_actor_raw]

background = (
    df_plot[df_plot[party_col] != selected_actor_raw]
    if show_background and not show_all else None
)

# -----------------------------
# VISUALIZATION
# -----------------------------
st.markdown("## Visualization")

fig = px.scatter(
    plot_df,
    x=x_col,
    y=y_col,
    color="cluster" if show_clusters else None,
    custom_data=["actor_display"],
    opacity=0.9
)

fig.update_traces(
    hovertemplate=(
        "Actor: %{customdata[0]}<br>"
        f"{x_col}: %{{x}}<br>"
        f"{y_col}: %{{y}}<extra></extra>"
    )
)

# background layer
if background is not None:
    fig.add_scatter(
        x=background[x_col],
        y=background[y_col],
        mode="markers",
        marker=dict(size=5, opacity=0.12),
        hoverinfo="skip",
        showlegend=False
    )

# -----------------------------
# AXES STYLING
# -----------------------------
fig.update_layout(
    template="simple_white",
    height=650,
    showlegend=False,
    paper_bgcolor="white",
    plot_bgcolor="white",
    xaxis_title=x_col,
    yaxis_title=y_col
)

fig.update_xaxes(range=x_range, zeroline=True, zerolinecolor="gray")
fig.update_yaxes(range=y_range, zeroline=True, zerolinecolor="gray")

fig.update_traces(marker=dict(size=9))

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# INTERPRETATION
# -----------------------------
st.markdown("## Discussion")

mode = "Normalized compass view" if normalize else "Raw data view"
scope = "All actors" if show_all else f"Focused on {selected_actor_display}"

st.write(
    f"""
- **Mode:** {mode}  
- **Scope:** {scope}  
- **Dimension:** {selected_dim}  

The examinee was interested 
This visualization shows political positioning across selected ideological axes. There are three main dimensions of note, those described as "Policy Aggressiveness", "Climate Stance", 
and "Agency of Speaker".
\n
**Policy Aggressiveness**: A measure of how strongly a speaker frames political positions in conservative vs liberal terms. 
Higher values indicate more assertive, conservative-leaning or status-quo reinforcing language, while lower values reflect more liberal, reform-oriented or progressive framing. This was calculated using "Arousal" from an examination of English lemmas
 from Warriner, A.B., Kuperman, V., & Brysbaert, M. (2013); or as shown here the A_score. It indicates the intensity of the emotion provoked by a verb. \n
\n
**Climate Stance**: A measure of how a speaker positions themselves on climate-related issues, ranging from skeptical or low-priority framing to strong pro-climate action orientation.
Higher values indicate stronger stance on climate action and environmental urgency, the direction dependent on the whether it is positive or negative.\n
\n
**Agency of Speaker**: A measure of how independently and concretely a speaker formulates policy or action proposals.
Higher values indicate statements where the policy or action stands on its own as clear, executable intent, rather than being vague, dependent, or rhetorically diffuse. This was calculated using "Dominance" from an examination of English lemmas
 from Warriner, A.B., Kuperman, V., & Brysbaert, M. (2013); or as shown here the A_score. It indicates the degree of control exerted or perceived to be exerted by a verb.

"""
)

st.markdown("---")
st.caption("The Hague Centre for Strategic Studies")
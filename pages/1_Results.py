import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import os

st.set_page_config(
    page_title="A/B Test Results — Sunscreen",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        section[data-testid="stSidebar"] { background: #0a2540; }
        section[data-testid="stSidebar"] * { color: #e8edf3 !important; }
        section[data-testid="stSidebar"] hr { border-color: #1e3a5f; }
        .header-title {
            color: #0D6E8A;
            font-size: 2.4rem;
            font-weight: 800;
            margin-bottom: 0.3rem;
        }
        .header-sub {
            color: #E87830;
            font-size: 1.1rem;
            font-weight: 500;
            margin-bottom: 1.5rem;
        }
        .metric-card {
            background: #fff;
            border-radius: 12px;
            padding: 1.2rem 1.4rem;
            box-shadow: 0 2px 12px rgba(0,0,0,0.07);
            border-left: 5px solid #0D6E8A;
        }
        .metric-label { font-size: 0.8rem; color: #888; font-weight: 600; text-transform: uppercase; }
        .metric-value { font-size: 1.9rem; font-weight: 800; color: #0D6E8A; }
        .metric-delta { font-size: 0.85rem; color: #27AE60; font-weight: 600; }
    </style>
""", unsafe_allow_html=True)

# ── Data ─────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("BAMA580A+A_B+Test+-+Sunscreen_labels.csv", skiprows=[1, 2])
    if "Finished" in df.columns:
        df = df[df["Finished"] == True].copy()
    df = df[["likelihood", "gender", "age", "Stimuli_DO"]].dropna()
    likert_map = {
        "Extremely unlikely": 1, "Somewhat unlikely": 2,
        "Neither likely nor unlikely": 3, "Somewhat likely": 4, "Extremely likely": 5,
    }
    df["likelihood_num"] = df["likelihood"].map(likert_map)
    df = df.dropna(subset=["likelihood_num"])
    df["ironcoast"] = df["Stimuli_DO"].str.lower().str.contains("ironcoast").astype(int)
    df["brand"] = df["ironcoast"].map({1: "Ironcoast", 0: "Banana Boat"})
    df["male"] = (df["gender"] == "Male").astype(int)
    age_order = ["21-30", "31-40", "41-50", "50+"]
    df["age"] = pd.Categorical(df["age"], categories=age_order, ordered=True)
    df["age_21_30"] = (df["age"] == "21-30").astype(int)
    return df

df = load_data()

# ── Palette ───────────────────────────────────────────────────────────────────
C_IRON  = "#0D6E8A"
C_BB    = "#E87830"
C_MALE  = "#2980B9"
C_FEM   = "#C0392B"
C_SIG   = "#27AE60"
C_NSIG  = "#E74C3C"

color_brand  = {"Ironcoast": C_IRON, "Banana Boat": C_BB}
color_gender = {"Male": C_MALE, "Female": C_FEM}
sub_colors = {
    "Male - Ironcoast":    C_IRON,
    "Male - Banana Boat":  C_BB,
    "Female - Ironcoast":  "#5dade2",
    "Female - Banana Boat": "#f5b041",
}

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="header-title">📊 A/B Test Results</div>', unsafe_allow_html=True)
st.markdown('<div class="header-sub">Ironcoast vs. Banana Boat — Purchase Likelihood Analysis</div>', unsafe_allow_html=True)

# ── Stimuli (inline, compact) ─────────────────────────────────────────────────
with st.expander("🖼️ View Experimental Stimuli", expanded=False):
    c_bb, c_gap, c_ic = st.columns([1, 0.06, 1])
    with c_bb:
        st.markdown("""
        <div style="background:#FFF7F2;border:2px solid #E87830;border-radius:12px;
                    padding:12px 16px 8px;text-align:center;margin-bottom:8px;">
            <span style="font-size:1rem;font-weight:700;color:#E87830;">
                🏖️ Condition B &nbsp;—&nbsp; Banana Boat</span>
        </div>""", unsafe_allow_html=True)
        st.image(Image.open("bananaboat.PNG"), use_container_width=True)
    with c_ic:
        st.markdown("""
        <div style="background:#F0F8FB;border:2px solid #0D6E8A;border-radius:12px;
                    padding:12px 16px 8px;text-align:center;margin-bottom:8px;">
            <span style="font-size:1rem;font-weight:700;color:#0D6E8A;">
                🌊 Condition A &nbsp;—&nbsp; Ironcoast</span>
        </div>""", unsafe_allow_html=True)
        st.image(Image.open("ironcoast.PNG"), use_container_width=True)

# ── KPI Row ───────────────────────────────────────────────────────────────────
N = len(df)
n_iron = df["ironcoast"].sum()
n_bb   = N - n_iron
n_male = df["male"].sum()
ir_mean = df[df["ironcoast"]==1]["likelihood_num"].mean()
bb_mean = df[df["ironcoast"]==0]["likelihood_num"].mean()

ir_m21 = df[(df["ironcoast"]==1)&(df["male"]==1)&(df["age"]=="21-30")]["likelihood_num"]
bb_m21 = df[(df["ironcoast"]==0)&(df["male"]==1)&(df["age"]=="21-30")]["likelihood_num"]
focal_delta = ir_m21.mean() - bb_m21.mean()

k1, k2, k3, k4 = st.columns(4)
def kpi(col, label, value, delta=None):
    with col:
        delta_html = f'<div class="metric-delta">{delta}</div>' if delta else ""
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            {delta_html}
        </div>""", unsafe_allow_html=True)

kpi(k1, "Total Respondents", N)
kpi(k2, "Ironcoast / Banana Boat", f"{n_iron} / {n_bb}")
kpi(k3, "Overall Δ Likelihood", f"{ir_mean-bb_mean:+.2f}",
    f"Ironcoast {ir_mean:.2f}  |  Banana Boat {bb_mean:.2f}")
kpi(k4, "Focal: Males 21-30 Δ", f"{focal_delta:+.2f}",
    f"IR {ir_m21.mean():.2f}  |  BB {bb_m21.mean():.2f}  (n={len(ir_m21)},{len(bb_m21)})")

st.divider()

# ── Section A: Main Effects ───────────────────────────────────────────────────
st.markdown("### A · Mean Purchase Likelihood by Gender × Brand")
colA, colB = st.columns(2)

with colA:
    marg = (df.groupby(["gender", "brand"])["likelihood_num"]
              .agg(mean="mean", sem="sem", count="count")
              .reset_index())
    marg["ci"] = marg["sem"] * 1.96
    fig_a = px.bar(marg, x="gender", y="mean", color="brand", barmode="group",
                   error_y="ci", text=marg["mean"].round(2),
                   color_discrete_map=color_brand,
                   labels={"mean": "Mean Likelihood (1–5)", "gender": "Gender", "brand": "Brand"})
    fig_a.update_traces(textposition="outside", textfont_size=12)
    fig_a.update_layout(yaxis_range=[1, 5.8], height=400,
                        margin=dict(t=20, b=20, l=20, r=20),
                        legend_title_text="Brand")
    st.plotly_chart(fig_a, use_container_width=True)

with colB:
    int_data = (df.groupby(["brand", "gender"])["likelihood_num"]
                  .mean().reset_index(name="mean"))
    fig_b = px.line(int_data, x="brand", y="mean", color="gender", markers=True,
                    color_discrete_map=color_gender,
                    labels={"mean": "Mean Likelihood (1–5)", "brand": "Brand", "gender": "Gender"})
    fig_b.update_traces(marker=dict(size=13, line=dict(width=2, color="white")), line=dict(width=3))
    for row in int_data.itertuples():
        fig_b.add_annotation(x=row.brand, y=row.mean + 0.12,
                              text=f"{row.mean:.2f}", showarrow=False,
                              font=dict(size=12, color=color_gender[row.gender]))
    fig_b.update_layout(yaxis_range=[1, 5.2], height=400,
                        margin=dict(t=20, b=20, l=20, r=20),
                        legend_title_text="Gender",
                        annotations=[a for a in fig_b.layout.annotations])
    st.plotly_chart(fig_b, use_container_width=True)

# ── Section B: Age × Gender × Brand ──────────────────────────────────────────
st.markdown("### B · Purchase Likelihood by Brand × Gender × Age Group")
age_data = (df.groupby(["age", "gender", "brand"])["likelihood_num"]
              .mean().reset_index(name="mean"))
age_data["Subgroup"] = age_data["gender"] + " – " + age_data["brand"]

fig_c = px.bar(age_data, x="age", y="mean", color="Subgroup", barmode="group",
               text=age_data["mean"].round(2),
               color_discrete_map={k.replace(" - ", " – "): v for k, v in sub_colors.items()},
               labels={"mean": "Mean Likelihood (1–5)", "age": "Age Group", "Subgroup": "Subgroup"})
fig_c.update_traces(textposition="outside", textfont_size=11)

# Annotate the focal bar
focal_val = df[(df["male"]==1)&(df["ironcoast"]==1)&(df["age"]=="21-30")]["likelihood_num"].mean()
fig_c.add_annotation(
    x="21-30", y=focal_val + 0.55, text="⭐ Focal Hypothesis",
    showarrow=True, arrowhead=2, arrowcolor=C_SIG, font=dict(color=C_SIG, size=12))

fig_c.update_layout(yaxis_range=[1, 6.5], height=450,
                    margin=dict(t=20, b=20, l=20, r=20),
                    legend_title_text="Subgroup")
st.plotly_chart(fig_c, use_container_width=True)

st.divider()

# ── Section C: Effect Sizes ───────────────────────────────────────────────────
st.markdown("### C · Cohen's d Effect Sizes by Subgroup")
se_path = "output/simple_effects.csv"
if os.path.exists(se_path):
    effects = pd.read_csv(se_path).sort_values("d", ascending=True)
    effects["Significance"] = np.where(effects["p"] < 0.05, "p < .05", "p ≥ .05")
    sig_colors = {"p < .05": C_SIG, "p ≥ .05": C_NSIG}

    fig_d = go.Figure()
    for _, row in effects.iterrows():
        col = sig_colors[row["Significance"]]
        fig_d.add_shape(type="line", x0=0, x1=row["d"], y0=row["label"], y1=row["label"],
                        line=dict(color=col, width=3))
        fig_d.add_trace(go.Scatter(
            x=[row["d"]], y=[row["label"]], mode="markers+text",
            marker=dict(color=col, size=13, line=dict(width=2, color="white")),
            text=[f"d={row['d']:+.2f}, p={row['p']:.3f}"],
            textposition="middle right" if row["d"] >= 0 else "middle left",
            textfont=dict(size=10),
            name=row["Significance"],
            showlegend=False,
            hovertemplate=f"<b>{row['label']}</b><br>d={row['d']:.3f}<br>p={row['p']:.4f}<extra></extra>"
        ))

    # Reference lines
    for thresh, lbl in [(0.2, "small"), (0.5, "medium"), (0.8, "large")]:
        fig_d.add_vline(x=thresh, line_dash="dot", line_color="gray", opacity=0.5)
        fig_d.add_vline(x=-thresh, line_dash="dot", line_color="gray", opacity=0.5)

    # Legend patches
    for sig_label, col in sig_colors.items():
        fig_d.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(color=col, size=12), name=sig_label))

    fig_d.add_vline(x=0, line_color="black", line_width=1.5)
    fig_d.update_layout(height=500, margin=dict(t=20, b=20, l=220, r=20),
                        xaxis_title="Cohen's d (Ironcoast − Banana Boat)",
                        yaxis_title="",
                        legend_title_text="Significance")
    st.plotly_chart(fig_d, use_container_width=True)
else:
    st.info("Run `analysis.py` to generate `output/simple_effects.csv`.")

# ── Section D: Likert Distribution ───────────────────────────────────────────
st.markdown("### D · Response Distribution — Focal Subgroup vs. Others")
df["Focal"] = np.where((df["male"]==1) & (df["age"]=="21-30"), "Male 21–30", "All Others")
dist = (df.groupby(["Focal", "brand", "likelihood_num"])
          .size().reset_index(name="count"))
dist["total"] = dist.groupby(["Focal", "brand"])["count"].transform("sum")
dist["pct"] = dist["count"] / dist["total"] * 100
dist["Group"] = dist["Focal"] + " · " + dist["brand"]

likert_names = {1: "Extremely Unlikely", 2: "Somewhat Unlikely", 3: "Neutral",
                4: "Somewhat Likely", 5: "Extremely Likely"}
likert_colors = {1: "#C0392B", 2: "#E67E22", 3: "#95A5A6", 4: "#27AE60", 5: "#1A5276"}

fig_e = go.Figure()
for val in [1, 2, 3, 4, 5]:
    subset = dist[dist["likelihood_num"] == val]
    fig_e.add_trace(go.Bar(
        y=subset["Group"], x=subset["pct"],
        name=likert_names[val],
        orientation="h",
        marker_color=likert_colors[val],
        text=[f"{v:.0f}%" if v > 7 else "" for v in subset["pct"]],
        textposition="inside",
        insidetextanchor="middle",
        hovertemplate="%{y}<br>" + likert_names[val] + ": %{x:.1f}%<extra></extra>",
    ))

fig_e.update_layout(barmode="stack", height=380,
                    xaxis_title="% of Respondents", yaxis_title="",
                    margin=dict(t=20, b=20, l=20, r=20),
                    legend_title_text="Response")
st.plotly_chart(fig_e, use_container_width=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("☀️ Sunscreen A/B Test")
st.sidebar.markdown("---")
st.sidebar.info(f"""
**N = {N}** respondents  
**{n_iron}** Ironcoast · **{n_bb}** Banana Boat  
**{n_male}** Male · **{N-n_male}** Female
""")
st.sidebar.markdown("---")
st.sidebar.markdown("Navigate to **Survey** in the sidebar to view & take the survey.")

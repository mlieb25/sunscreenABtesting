import streamlit as st

st.set_page_config(
    page_title="Sunscreen A/B Test",
    page_icon="☀️",
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
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="header-title">📋 Sunscreen Experience Survey</div>', unsafe_allow_html=True)
st.markdown('<div class="header-sub">Participate in the A/B test by completing the survey below</div>', unsafe_allow_html=True)
st.divider()

st.components.v1.iframe(
    "https://ubc.ca1.qualtrics.com/jfe/form/SV_ebMsvKs6HVWOEOG",
    height=870,
    scrolling=True,
)

st.sidebar.title("☀️ Sunscreen A/B Test")
st.sidebar.markdown("---")
st.sidebar.info("""
**Study:** Ironcoast vs. Banana Boat  
**Measure:** Purchase likelihood (1–5 Likert)  
**Hypothesis:** Ironcoast ↑ purchase likelihood in males aged 21–30

Navigate to **Results** in the sidebar to explore the findings.
""")

import streamlit as st

st.set_page_config(
    page_title="Sunscreen A/B Test",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

survey  = st.Page("pages/0_Survey.py",  title="Survey",  icon="📋", default=True)
results = st.Page("pages/1_Results.py", title="Results", icon="📊")

pg = st.navigation([survey, results])
pg.run()

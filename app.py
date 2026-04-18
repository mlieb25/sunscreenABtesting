import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

# Set page configuration with a modern, clean look
st.set_page_config(
    page_title="Sunscreen A/B Test Dashboard",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for enhanced aesthetics
st.markdown("""
    <style>
        .main {
            background-color: #F8F9FA;
            font-family: 'Inter', sans-serif;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: #fff;
            border-radius: 8px 8px 0 0;
            padding: 10px 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            color: #333;
            font-weight: 600;
        }
        .stTabs [aria-selected="true"] {
            background-color: #0D6E8A !important;
            color: #FFF !important;
        }
        .header-container {
            padding: 2rem 0 1rem 0;
            text-align: center;
        }
        .header-title {
            color: #0D6E8A;
            font-size: 2.5rem;
            font-weight: 800;
            margin-bottom: 0.5rem;
        }
        .header-subtitle {
            color: #E87830;
            font-size: 1.2rem;
            font-weight: 500;
        }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv("BAMA580A+A_B+Test+-+Sunscreen_labels.csv", skiprows=[1, 2])
    if 'Finished' in df.columns:
        df = df[df['Finished'] == True].copy()
    
    df = df[['likelihood', 'gender', 'age', 'Stimuli_DO']].copy()
    df = df.dropna()
    
    likert_map = {
        'Extremely unlikely': 1,
        'Somewhat unlikely': 2,
        'Neither likely nor unlikely': 3,
        'Somewhat likely': 4,
        'Extremely likely': 5,
    }
    df['likelihood_num'] = df['likelihood'].map(likert_map)
    df = df.dropna(subset=['likelihood_num'])
    
    df['ironcoast'] = df['Stimuli_DO'].str.lower().str.contains('ironcoast').astype(int)
    df['brand'] = df['ironcoast'].map({1: 'Ironcoast', 0: 'Banana Boat'})
    df['male'] = (df['gender'] == 'Male').astype(int)
    
    age_order = ['21-30', '31-40', '41-50', '50+']
    df['age'] = pd.Categorical(df['age'], categories=age_order, ordered=True)
    df['age_21_30'] = (df['age'] == '21-30').astype(int)
    return df

try:
    df = load_data()
    data_loaded = True
except Exception as e:
    st.error(f"Failed to load data: {e}")
    data_loaded = False

st.markdown("""
<div class="header-container">
    <div class="header-title">☀️ Sunscreen A/B Test Dashboard</div>
    <div class="header-subtitle">Interactive Results & Survey View</div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📊 Interactive Results", "📋 Survey View", "📈 Effect Sizes & Distributions"])

color_brand = {'Ironcoast': '#0D6E8A', 'Banana Boat': '#E87830'}
color_gender = {'Male': '#2980B9', 'Female': '#C0392B'}

if data_loaded:
    with tab1:
        st.markdown("### Experiment Findings: Main Effects & Interaction")
        colA, colB = st.columns(2)
        
        with colA:
            st.markdown("**Mean Purchase Likelihood by Gender × Brand**")
            marg = df.groupby(['gender', 'brand'])['likelihood_num'].agg(['mean', 'sem', 'count']).reset_index()
            marg['ci'] = marg['sem'] * 1.96
            fig_a = px.bar(marg, x='gender', y='mean', color='brand', barmode='group',
                           error_y='ci', text=marg['mean'].round(2),
                           color_discrete_map=color_brand)
            fig_a.update_layout(yaxis_title="Mean Purchase Likelihood (1-5)", xaxis_title="Gender", 
                                yaxis_range=[1, 5.5], margin=dict(t=20, b=20, l=20, r=20), height=450)
            fig_a.update_traces(textposition='outside')
            st.plotly_chart(fig_a, use_container_width=True)
            
        with colB:
            st.markdown("**Interaction Plot: Brand × Gender**")
            int_data = df.groupby(['brand', 'gender'])['likelihood_num'].mean().reset_index()
            fig_b = px.line(int_data, x='brand', y='likelihood_num', color='gender', markers=True,
                            color_discrete_map=color_gender)
            fig_b.update_traces(marker=dict(size=12, line=dict(width=2, color='white')))
            fig_b.update_layout(yaxis_title="Mean Purchase Likelihood (1-5)", xaxis_title="Brand",
                                yaxis_range=[1, 5.2], margin=dict(t=20, b=20, l=20, r=20), height=450)
            st.plotly_chart(fig_b, use_container_width=True)

        st.markdown("---")
        st.markdown("**Purchase Likelihood by Brand × Gender × Age Group**")
        
        age_data = df.groupby(['age', 'gender', 'brand'])['likelihood_num'].mean().reset_index()
        age_data['Subgroup'] = age_data['gender'] + " - " + age_data['brand']
        
        sub_colors = {
            'Male - Ironcoast': '#0D6E8A',
            'Male - Banana Boat': '#E87830',
            'Female - Ironcoast': '#5dade2',
            'Female - Banana Boat': '#f5b041'
        }
        
        fig_c = px.bar(age_data, x='age', y='likelihood_num', color='Subgroup', barmode='group',
                       text=age_data['likelihood_num'].round(2),
                       color_discrete_map=sub_colors)
        fig_c.update_layout(yaxis_title="Mean Purchase Likelihood (1-5)", xaxis_title="Age Group",
                            yaxis_range=[1, 6], margin=dict(t=20, b=20, l=20, r=20), height=450)
        fig_c.update_traces(textposition='outside')
        st.plotly_chart(fig_c, use_container_width=True)

    with tab2:
        st.markdown("### Sunscreen Experience Survey")
        st.components.v1.iframe("https://ubc.ca1.qualtrics.com/jfe/form/SV_ebMsvKs6HVWOEOG", height=850, scrolling=True)

    with tab3:
        st.markdown("### Detailed Effect Sizes & Response Distributions")
        colC, colD = st.columns([1.2, 1])
        
        with colC:
            st.markdown("**Effect Sizes (Cohen's d) by Subgroup**")
            se_path = "output/simple_effects.csv"
            if os.path.exists(se_path):
                effects = pd.read_csv(se_path)
                effects = effects.sort_values('d', ascending=True)
                effects['Significance'] = np.where(effects['p'] < 0.05, 'p < .05 (Significant)', 'p ≥ .05 (Not Significant)')
                sig_colors = {'p < .05 (Significant)': '#27AE60', 'p ≥ .05 (Not Significant)': '#E74C3C'}
                
                fig_d = px.scatter(effects, x='d', y='label', color='Significance', 
                                   color_discrete_map=sig_colors, hover_data=['p', 't'])
                fig_d.update_traces(marker=dict(size=14))
                for i, row in effects.iterrows():
                    fig_d.add_shape(type="line", x0=0, y0=row['label'], x1=row['d'], y1=row['label'], 
                                    line=dict(color=sig_colors[row['Significance']], width=3), layer="below")
                fig_d.add_vline(x=0, line_dash="dash", line_color="black")
                fig_d.update_layout(xaxis_title="Cohen's d (Ironcoast - Banana Boat)", yaxis_title="",
                                    margin=dict(t=20, b=20, l=20, r=20), height=550)
                st.plotly_chart(fig_d, use_container_width=True)
            else:
                st.info("Run analysis.py to generate simple_effects.csv for this chart.")
                
        with colD:
            st.markdown("**Response Distribution (Focal vs Others)**")
            df['Focal Group'] = np.where((df['male']==1) & (df['age']=='21-30'), 'Male 21-30', 'Others')
            dist_data = df.groupby(['Focal Group', 'brand', 'likelihood_num']).size().reset_index(name='count')
            dist_data['total'] = dist_data.groupby(['Focal Group', 'brand'])['count'].transform('sum')
            dist_data['pct'] = (dist_data['count'] / dist_data['total']) * 100
            
            dist_data['Group'] = dist_data['Focal Group'] + " - " + dist_data['brand']
            
            likert_colors = {
                1: '#C0392B', 2: '#E67E22', 3: '#95A5A6', 4: '#27AE60', 5: '#1A5276'
            }
            
            fig_e = px.bar(dist_data, y='Group', x='pct', color='likelihood_num', orientation='h',
                           title="",
                           labels={'pct': '% of respondents', 'Group': 'Subgroup'})
            
            # Update specific colors dynamically mapping the literal values
            fig_e.for_each_trace(lambda t: t.update(
                marker_color=likert_colors.get(int(t.name), '#000'),
                name={
                    '1': 'Extr. Unlikely', '2': 'Smwt. Unlikely', '3': 'Neutral', 
                    '4': 'Smwt. Likely', '5': 'Extr. Likely'
                }.get(t.name, t.name)
            ))
            fig_e.update_layout(xaxis_title="% of Respondents", yaxis_title="", legend_title="Likelihood (1-5)",
                                barmode='stack', margin=dict(t=20, b=20, l=20, r=20), height=550)
            st.plotly_chart(fig_e, use_container_width=True)

st.sidebar.title("About the Experiment")
st.sidebar.info('''
**Objective:**
Analyze the effectiveness of Ironcoast sunscreen brand compared to Banana Boat, particularly on male demographics aged 21-30.

**Hypothesis:**
Ironcoast increases purchase likelihood in males, especially young males aged 21-30.

**Data Source:**
Analysis derived from an A/B test with an embedded Qualtrics survey measuring purchase likelihood on a 1-5 Likert scale.
''')
st.sidebar.markdown("---")
st.sidebar.caption("Data is processed interactively on-the-fly.")

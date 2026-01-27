"""
DSA 9 MVP - Streamlit UI
User-friendly interface for LLM-based hiring system
"""

import streamlit as st
import requests
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import tab modules
from tabs import job_description_tab, resumes_tab, matching_tab, verification_tab, pipeline_tab

# API Configuration
API_BASE_URL = "http://localhost:8000"

# Page config
st.set_page_config(
    page_title="DSA 9 MVP - LLM Hiring System",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'jd_data' not in st.session_state:
    st.session_state.jd_data = None
if 'resumes_data' not in st.session_state:
    st.session_state.resumes_data = []
if 'pipeline_results' not in st.session_state:
    st.session_state.pipeline_results = None

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/200x100.png?text=DSA+9+MVP", use_container_width=True)
    st.markdown("### 💼 LLM Hiring System")
    st.markdown("**Features:**")
    st.markdown("- 📄 Job Description Analysis")
    st.markdown("- 📋 Resume Parsing (BERT NER)")
    st.markdown("- ✅ Hallucination Detection")
    st.markdown("- 🎯 Job-Resume Matching")
    st.markdown("- ⚖️ Fairness-Aware Ranking")
    
    st.markdown("---")
    st.markdown("**API Status:**")
    
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        if response.status_code == 200:
            st.success("✅ API Connected")
        else:
            st.error("❌ API Error")
    except:
        st.error("❌ API Offline")
        st.info("Start server: `python backend/app.py`")

# Main content
st.markdown('<div class="main-header">💼 DSA 9 MVP - LLM Hiring System</div>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📄 Job Description", 
    "📋 Resumes", 
    "🎯 Matching & Ranking",
    "✅ Verification",
    "🚀 Complete Pipeline"
])

with tab1:
    job_description_tab.render(API_BASE_URL)

with tab2:
    resumes_tab.render(API_BASE_URL)

with tab3:
    matching_tab.render(API_BASE_URL)

with tab4:
    verification_tab.render(API_BASE_URL)

with tab5:
    pipeline_tab.render(API_BASE_URL)

# Footer
st.markdown("---")
st.markdown("**DSA 9 MVP** | LLM-based Hiring System with Bias Mitigation & Hallucination Detection")
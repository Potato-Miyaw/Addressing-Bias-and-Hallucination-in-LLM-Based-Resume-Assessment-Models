"""
HR Portal - Full Resume Screening & Management System
"""

import streamlit as st
import requests
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Import tab modules
from tabs import job_description_tab, resumes_tab, matching_tab, verification_tab, pipeline_tab, feedback_tab, bias_tab



# API Configuration
API_BASE_URL = "http://localhost:8000"

# Page config
st.set_page_config(
    page_title="HR Portal - DSA 9 MVP",
    page_icon="👨‍💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #f5576c;
        text-align: center;
        margin-bottom: 2rem;
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
if 'experiment_status' not in st.session_state:
    st.session_state.experiment_status = {}
if 'bias_results' not in st.session_state:
    st.session_state.bias_results = None

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
    st.markdown("- 🎓 Feedback & Pattern Learning")
    st.markdown("- 🔬 Multi-Model Bias Detection")

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
        st.info("Start server: `uvicorn backend.app:app --reload`")

# Main content
st.markdown('<div class="main-header">👨‍💼 HR Portal - Resume Screening System</div>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📄 Job Description", 
    "📋 Resumes", 
    "🎯 Matching & Ranking",
    "✅ Verification",
    "🚀 Complete Pipeline",
    "🎓 HR Feedback",
    "🔬 Multi-Model Bias Detection" #feature 6
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

with tab6:
    feedback_tab.render(API_BASE_URL)

with tab7:
    bias_tab.render(API_BASE_URL)  # Feature 6
    


# Footer
st.markdown("---")
st.markdown("**HR Portal** | LLM-based Hiring System with Bias Mitigation & Hallucination Detection")
st.markdown("🔒 Secure • ⚖️ Fair • 🤖 AI-Powered")

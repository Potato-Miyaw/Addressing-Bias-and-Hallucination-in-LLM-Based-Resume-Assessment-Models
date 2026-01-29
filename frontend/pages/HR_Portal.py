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
from tabs import job_description_tab, resumes_tab, matching_tab, verification_tab, pipeline_tab, feedback_tab, bias_tab, multilang_tab, questionnaire_tab



# API Configuration
API_BASE_URL = "http://localhost:8000"

# Page config
st.set_page_config(
    page_title="HR Portal - DSA 9 MVP",
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
if 'experiment_status' not in st.session_state:
    st.session_state.experiment_status = {}
if 'bias_results' not in st.session_state:
    st.session_state.bias_results = None

# Sidebar
with st.sidebar:
    st.markdown("### Connection Status")

    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        if response.status_code == 200:
            st.success("API Connected")
        else:
            st.error("API Error")
    except:
        st.error("API Offline")
        st.info("Please contact the administrator")

# Main content
st.markdown('<div class="main-header">HR Portal - Resume Screening System</div>', unsafe_allow_html=True)

# Tabs
tabs = st.tabs([
    "Job Description", 
    "Resumes", 
    "Multi-Language", # Moved here
    "Matching & Ranking",
    "Questionnaire",  # NEW
    "Verification",
    "HR Feedback",
    "Multi-Model Bias Detection"
])

(
    job_description_tab_ui, 
    resumes_tab_ui, 
    multilang_tab_ui, 
    matching_tab_ui,
    questionnaire_tab_ui,  # NEW
    verification_tab_ui, 
    feedback_tab_ui, 
    bias_tab_ui
) = tabs

with job_description_tab_ui:
    job_description_tab.render(API_BASE_URL)

with resumes_tab_ui:
    resumes_tab.render(API_BASE_URL, portal_type="hr_upload")

with multilang_tab_ui:
    multilang_tab.render(API_BASE_URL)

with matching_tab_ui:
    matching_tab.render(API_BASE_URL, portal_type="hr_portal")

with questionnaire_tab_ui:
    questionnaire_tab.render(API_BASE_URL)

with verification_tab_ui:
    verification_tab.render(API_BASE_URL)

with feedback_tab_ui:
    feedback_tab.render(API_BASE_URL)

with bias_tab_ui:
    bias_tab.render(API_BASE_URL)
    

# Footer
st.markdown("---")
st.markdown("**HR Portal** | Powered by AI-based Resume Screening System")
st.markdown("Secure • Fair • AI-Powered")

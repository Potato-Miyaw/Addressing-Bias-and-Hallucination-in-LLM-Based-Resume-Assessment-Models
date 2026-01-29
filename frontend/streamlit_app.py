"""
DSA 9 MVP - Landing Page
Choose your portal: Candidate or HR
"""
import os
import sys
import streamlit as st
# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

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
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        text-align: center;
        color: #666;
        margin-bottom: 3rem;
    }
    .portal-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        color: white;
        margin: 1rem 0;
    }
    .portal-card-hr {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        color: white;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<div class="main-header">💼 LLM Hiring System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Bias-Aware Resume Screening with Hallucination Detection</div>', unsafe_allow_html=True)

st.markdown("---")

# Portal selection
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="portal-card">
        <h2>👔 Candidate Portal</h2>
        <p style="font-size: 1.1rem; margin: 1.5rem 0;">
            Upload your resume and see how you match with job openings
        </p>
        <ul style="text-align: left; list-style: none; padding: 0;">
            <li>📄 Upload Resume</li>
            <li>🎯 View Match Score</li>
            <li>📊 See Ranking</li>
            <li>✅ Verify Information</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🚀 Go to Candidate Portal", type="primary", width='stretch'):
        st.switch_page("pages/Candidate_Portal.py")

with col2:
    st.markdown("""
    <div class="portal-card-hr">
        <h2>👨‍💼 HR Portal</h2>
        <p style="font-size: 1.1rem; margin: 1.5rem 0;">
            Manage job descriptions, screen resumes, and ensure fair hiring
        </p>
        <ul style="text-align: left; list-style: none; padding: 0;">
            <li>📋 Manage Job Descriptions</li>
            <li>🔍 Screen Resumes with AI</li>
            <li>✅ Detect Hallucinations</li>
            <li>⚖️ Bias Auditing</li>
            <li>🎓 Provide Feedback</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🚀 Go to HR Portal", type="primary", width='stretch'):
        st.switch_page("pages/HR_Portal.py")

st.markdown("---")

# Features
st.subheader("🌟 System Features")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **🤖 AI-Powered Screening**
    - BERT-based NER extraction
    - Hybrid model combining Generic + Resume-specific NER
    - Automatic skills matching
    """)

with col2:
    st.markdown("""
    **✅ Hallucination Detection**
    - Token overlap + BERTScore verification
    - Ground truth extraction from resumes
    - Confidence scoring for each claim
    """)

with col3:
    st.markdown("""
    **⚖️ Fairness & Bias Mitigation**
    - Bias auditing in rankings
    - Fair scoring algorithms
    - Pattern learning from HR feedback
    """)

st.markdown("---")
st.markdown("**DSA 9 MVP** | LLM-based Hiring System with Bias Mitigation & Hallucination Detection")
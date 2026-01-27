"""
DSA 9 MVP - Streamlit UI
User-friendly interface for LLM-based hiring system
"""

import streamlit as st
import requests
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, List
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

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

# ============================================
# TAB 1: JOB DESCRIPTION
# ============================================
with tab1:
    st.header("📄 Job Description Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Enter Job Description")
        
        job_title = st.text_input("Job Title", placeholder="e.g., Senior Python Developer")
        
        jd_text = st.text_area(
            "Job Description",
            height=300,
            placeholder="""Enter the full job description here...

Example:
We are seeking a Senior Python Developer with 5+ years of experience.

Requirements:
- Strong knowledge of Python, Django, FastAPI
- Experience with AWS, Docker, Kubernetes
- Master's degree in Computer Science preferred
"""
        )
        
        if st.button("🔍 Analyze Job Description", type="primary"):
            if jd_text:
                with st.spinner("Analyzing job description..."):
                    try:
                        response = requests.post(
                            f"{API_BASE_URL}/api/jd/extract",
                            json={"jd_text": jd_text, "job_title": job_title}
                        )
                        
                        if response.status_code == 200:
                            st.session_state.jd_data = response.json()
                            st.success("✅ Job description analyzed successfully!")
                            st.rerun()
                        else:
                            st.error(f"Error: {response.text}")
                    except Exception as e:
                        st.error(f"API Error: {str(e)}")
            else:
                st.warning("Please enter a job description")
    
    with col2:
        if st.session_state.jd_data:
            st.subheader("📊 Extracted Requirements")
            
            data = st.session_state.jd_data
            
            st.markdown(f"**Job ID:** `{data['job_id']}`")
            st.markdown(f"**Status:** {data['status']}")
            
            if data.get('job_title'):
                st.markdown(f"**Title:** {data['job_title']}")
            
            st.markdown("---")
            
            st.metric("Required Experience", f"{data['required_experience']} years")
            st.metric("Education Level", data['required_education'])
            
            st.markdown("**Required Skills:**")
            if data['required_skills']:
                for skill in data['required_skills']:
                    st.markdown(f"- {skill}")
            else:
                st.info("No skills extracted")
            
            st.markdown("**Certifications:**")
            if data['certifications']:
                for cert in data['certifications']:
                    st.markdown(f"- {cert}")
            else:
                st.info("No certifications required")

# ============================================
# TAB 2: RESUMES
# ============================================
with tab2:
    st.header("📋 Resume Parsing")
    
    st.markdown("Upload resumes (PDF, DOCX, DOC, TXT) to extract structured information")
    
    uploaded_files = st.file_uploader(
        "Upload Resume Files",
        type=['txt', 'pdf', 'docx', 'doc'],
        accept_multiple_files=True,
        help="Upload one or more resume files in TXT, PDF, DOCX, or DOC format"
    )
    
    if uploaded_files:
        if st.button("📤 Parse Resumes", type="primary"):
            st.session_state.resumes_data = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Parsing {uploaded_file.name}...")
                
                try:
                    # Reset file pointer to beginning
                    uploaded_file.seek(0)
                    
                    # Send file to upload endpoint (supports PDF, DOCX, DOC, TXT)
                    files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                    response = requests.post(
                        f"{API_BASE_URL}/api/resume/upload",
                        files=files,
                        params={"save_to_db": True}
                    )
                    
                    if response.status_code == 200:
                        resume_data = response.json()
                        st.session_state.resumes_data.append({
                            "filename": uploaded_file.name,
                            "data": resume_data
                        })
                    else:
                        st.error(f"API Error: {response.json().get('detail', 'Unknown error')}")
                    
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                    
                except Exception as e:
                    st.error(f"Error parsing {uploaded_file.name}: {str(e)}")
            
            status_text.text("✅ All resumes parsed!")
            st.success(f"Successfully parsed {len(st.session_state.resumes_data)} resumes")
            st.rerun()
    
    # Display parsed resumes
    if st.session_state.resumes_data:
        st.markdown("---")
        st.subheader("📊 Stored Resumes")
        st.info("ℹ️ Basic info extracted. Full NER extraction happens during matching for better performance.")
        
        for idx, resume in enumerate(st.session_state.resumes_data):
            with st.expander(f"📄 {resume['filename']}", expanded=(idx == 0)):
                data = resume['data']
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Resume ID", data.get('resume_id', 'N/A')[:8])
                    st.metric("Name", data.get('name', 'Unknown'))
                
                with col2:
                    st.metric("Email", data.get('email', 'N/A')[:20])
                    st.metric("Phone", data.get('phone', 'N/A'))
                
                with col3:
                    st.metric("Status", data.get('status', 'UNKNOWN'))
                
                st.markdown("**Storage Info:**")
                st.write(data.get('message', 'Resume stored successfully'))
                
                # Extract and display skills
                st.markdown("---")
                st.markdown("**🛠️ Extracted Skills:**")
                
                # Get resume text from data
                resume_text = data.get('text', '')
                
                if resume_text:
                    # Make API call to extract skills
                    try:
                        skill_response = requests.post(
                            f"{API_BASE_URL}/api/resume/extract-skills",
                            json={"resume_text": resume_text},
                            timeout=10
                        )
                        
                        if skill_response.status_code == 200:
                            skills_data = skill_response.json()
                            extracted_skills = skills_data.get('skills', [])
                            
                            if extracted_skills:
                                # Display skills as colored badges
                                st.write("")
                                cols = st.columns(len(extracted_skills) if len(extracted_skills) <= 6 else 6)
                                
                                for i, skill in enumerate(extracted_skills[:24]):  # Show max 24 skills
                                    col = cols[i % 6]
                                    with col:
                                        st.markdown(f"<span style='background-color: #1f77b4; color: white; padding: 5px 10px; border-radius: 15px; display: inline-block; font-size: 12px; margin: 3px;'>🔧 {skill}</span>", unsafe_allow_html=True)
                                
                                st.caption(f"📊 Total skills found: {len(extracted_skills)}")
                                
                                if len(extracted_skills) > 24:
                                    with st.expander(f"View all {len(extracted_skills)} skills"):
                                        st.write(", ".join(extracted_skills))
                            else:
                                st.info("No skills detected in this resume")
                        else:
                            st.warning("Could not extract skills - API error")
                    except Exception as e:
                        st.warning(f"Skills extraction failed: {str(e)}")
                else:
                    st.info("Resume text not available for skill extraction")

# ============================================
# TAB 3: MATCHING & RANKING
# ============================================
with tab3:
    st.header("🎯 Job-Resume Matching & Ranking")
    
    if not st.session_state.jd_data:
        st.warning("⚠️ Please analyze a job description first (Tab 1)")
    elif not st.session_state.resumes_data:
        st.warning("⚠️ Please upload and parse resumes first (Tab 2)")
    else:
        st.success("✅ Ready to match and rank candidates!")
        
        use_fairness = st.checkbox("Use Fairness-Aware Ranking", value=False,
                                   help="Apply fairness constraints using Fairlearn")
        
        if st.button("🚀 Match & Rank Candidates", type="primary"):
            with st.spinner("Processing candidates..."):
                try:
                    # Prepare candidates data
                    candidates = []
                    
                    for resume in st.session_state.resumes_data:
                        resume_data = resume['data']
                        
                        # Match to job (NER extraction happens inside matching router)
                        match_response = requests.post(
                            f"{API_BASE_URL}/api/match/",
                            json={
                                "resume_id": resume_data.get('resume_id', 'unknown'),
                                "job_id": st.session_state.jd_data['job_id'],
                                "resume_data": resume_data,  # Pass full data (includes 'text' field)
                                "jd_data": {
                                    "required_skills": st.session_state.jd_data['required_skills'],
                                    "required_experience": st.session_state.jd_data['required_experience'],
                                    "required_education": st.session_state.jd_data['required_education']
                                }
                            }
                        )
                        
                        if match_response.status_code == 200:
                            match_result = match_response.json()['match_result']
                            
                            candidates.append({
                                "resume_id": resume_data['resume_id'],
                                "resume_data": resume_data,
                                "match_data": match_result,
                                "demographics": {"gender": 1, "race_gender": "Unknown"}
                            })
                    
                    # Rank candidates
                    rank_response = requests.post(
                        f"{API_BASE_URL}/api/rank/",
                        json={
                            "job_id": st.session_state.jd_data['job_id'],
                            "candidates": candidates,
                            "jd_data": st.session_state.jd_data,
                            "use_fairness": use_fairness
                        }
                    )
                    
                    if rank_response.status_code == 200:
                        ranked_results = rank_response.json()
                        st.session_state.ranked_candidates = ranked_results['ranked_candidates']
                        st.success("✅ Candidates ranked successfully!")
                        st.rerun()
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        # Display results
        if hasattr(st.session_state, 'ranked_candidates') and st.session_state.ranked_candidates:
            st.markdown("---")
            st.subheader("📊 Ranking Results")
            
            # Create ranking table
            ranking_data = []
            for candidate in st.session_state.ranked_candidates:
                ranking_data.append({
                    "Name": candidate['resume_data'].get('name', 'Unknown'),
                    "Rank": candidate['rank'],
                    "Candidate": candidate.get('candidate_name', candidate['resume_id'][:8]),
                    "Match Score": f"{candidate['match_data']['match_score']:.1f}%",
                    "Skill Match": f"{candidate['match_data']['skill_match']:.1f}%",
                    "Experience Match": f"{candidate['match_data']['experience_match']:.1f}%",
                    "Ranking Score": f"{candidate['ranking_score']:.2f}"
                })
            
            df = pd.DataFrame(ranking_data)
            st.dataframe(df, width='stretch', hide_index=True)
            
            # Visualization
            st.subheader("📈 Candidate Comparison")
            
            fig = go.Figure()
            
            candidates_list = [c.get('candidate_name', c['resume_id'][:8]) 
                             for c in st.session_state.ranked_candidates]
            
            fig.add_trace(go.Bar(
                x=candidates_list,
                y=[c['match_data']['match_score'] for c in st.session_state.ranked_candidates],
                name='Overall Match',
                marker_color='lightblue'
            ))
            
            fig.add_trace(go.Bar(
                x=candidates_list,
                y=[c['match_data']['skill_match'] for c in st.session_state.ranked_candidates],
                name='Skill Match',
                marker_color='lightgreen'
            ))
            
            fig.update_layout(
                title="Candidate Match Scores",
                xaxis_title="Candidate",
                yaxis_title="Match Score (%)",
                barmode='group',
                height=400
            )
            
            st.plotly_chart(fig, width='stretch')
            
            # Detailed view
            st.subheader("📋 Detailed Candidate Reports")
            
            for candidate in st.session_state.ranked_candidates:
                with st.expander(f"Rank #{candidate['rank']}: {candidate['resume_data'].get('name', candidate.get('candidate_name', candidate['resume_id'][:8]))}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Match Details:**")
                        st.write(f"Overall: {candidate['match_data']['match_score']:.1f}%")
                        st.write(f"Skills: {candidate['match_data']['skill_match']:.1f}%")
                        st.write(f"Experience: {candidate['match_data']['experience_match']:.1f}%")
                        st.write(f"Education: {candidate['match_data']['education_match']:.1f}%")
                        
                        st.markdown("**Skill Gaps:**")
                        if candidate['match_data']['skill_gaps']:
                            for gap in candidate['match_data']['skill_gaps']:
                                st.write(f"❌ {gap}")
                        else:
                            st.write("✅ No skill gaps")
                    
                    with col2:
                        st.markdown("**Matched Skills:**")
                        if candidate['match_data']['matched_skills']:
                            for skill in candidate['match_data']['matched_skills']:
                                st.write(f"✅ {skill}")
                        else:
                            st.write("No matched skills")

# ============================================
# TAB 4: VERIFICATION
# ============================================
with tab4:
    st.header("✅ Hallucination Detection")
    
    if not st.session_state.resumes_data:
        st.warning("⚠️ Please upload and parse resumes first (Tab 2)")
    else:
        st.markdown("Verify resume claims for hallucinations using Token Overlap + BERTScore")
        
        selected_resume = st.selectbox(
            "Select Resume to Verify",
            options=range(len(st.session_state.resumes_data)),
            format_func=lambda x: st.session_state.resumes_data[x]['filename']
        )
        
        if st.button("🔍 Verify Resume", type="primary"):
            with st.spinner("Verifying claims..."):
                try:
                    resume_data = st.session_state.resumes_data[selected_resume]['data']
                    
                    response = requests.post(
                        f"{API_BASE_URL}/api/verify/resume",
                        json={
                            "resume_id": resume_data['resume_id'],
                            "resume_extractions": resume_data
                        }
                    )
                    
                    if response.status_code == 200:
                        verification = response.json()['verification_report']
                        st.session_state.verification_report = verification
                        st.rerun()
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        # Display verification results
        if hasattr(st.session_state, 'verification_report'):
            report = st.session_state.verification_report
            
            st.markdown("---")
            st.subheader("📊 Verification Report")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Overall Confidence", f"{report['overall_confidence']:.1%}")
            with col2:
                st.metric("Hallucination Rate", f"{report['hallucination_rate']:.1%}")
            with col3:
                st.metric("Verified Claims", report['verified_claims'])
            with col4:
                st.metric("Flagged Claims", report['flagged_claims'])
            
            # Verdict
            if report['verdict'] == "VERIFIED":
                st.success("✅ All claims verified!")
            elif report['verdict'] == "NO_CLAIMS":
                st.info("ℹ️ No ground truth provided for verification")
            else:
                st.warning(f"⚠️ {report['verdict']}")

# ============================================
# TAB 5: COMPLETE PIPELINE
# ============================================
with tab5:
    st.header("🚀 Complete Pipeline")
    
    st.markdown("""
    Run the complete end-to-end pipeline:
    1. Extract JD requirements
    2. Parse all resumes
    3. Verify claims for hallucinations
    4. Match resumes to job
    5. Rank candidates with fairness
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        pipeline_jd = st.text_area(
            "Job Description",
            height=200,
            placeholder="Enter job description..."
        )
        
        pipeline_resumes = st.text_area(
            "Resumes (one per line, or paste multiple)",
            height=200,
            placeholder="""Resume 1: Jane Smith. 5 years Python experience...

Resume 2: Bob Jones. 3 years Python experience..."""
        )
    
    with col2:
        use_pipeline_fairness = st.checkbox("Enable Fairness", value=False)
        
        if st.button("🚀 Run Complete Pipeline", type="primary"):
            if pipeline_jd and pipeline_resumes:
                with st.spinner("Running pipeline..."):
                    try:
                        # Split resumes
                        resume_texts = [r.strip() for r in pipeline_resumes.split('\n\n') if r.strip()]
                        
                        response = requests.post(
                            f"{API_BASE_URL}/api/pipeline/complete",
                            json={
                                "jd_text": pipeline_jd,
                                "resume_texts": resume_texts,
                                "use_fairness": use_pipeline_fairness
                            }
                        )
                        
                        if response.status_code == 200:
                            st.session_state.pipeline_results = response.json()
                            st.success("✅ Pipeline completed!")
                            st.rerun()
                        else:
                            st.error(f"Error: {response.text}")
                    
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            else:
                st.warning("Please provide both JD and resumes")
    
    # Display pipeline results
    if st.session_state.pipeline_results:
        results = st.session_state.pipeline_results
        
        st.markdown("---")
        st.subheader("📊 Pipeline Results")
        
        st.metric("Total Candidates Processed", results['total_candidates'])
        
        # Show top 3 candidates
        st.subheader("🏆 Top Candidates")
        
        for candidate in results['ranked_candidates'][:3]:
            with st.container():
                col1, col2, col3 = st.columns([1, 2, 1])
                
                with col1:
                    st.markdown(f"### Rank #{candidate['rank']}")
                
                with col2:
                    st.markdown(f"**{candidate['candidate_name']}**")
                    st.progress(candidate['match_data']['match_score'] / 100)
                
                with col3:
                    st.metric("Match", f"{candidate['match_data']['match_score']:.1f}%")
                
                st.markdown("---")

# Footer
st.markdown("---")
st.markdown("**DSA 9 MVP** | LLM-based Hiring System with Bias Mitigation & Hallucination Detection")
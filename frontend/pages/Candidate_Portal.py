"""
Candidate Portal - Upload Resume & View Match Results
"""

import streamlit as st
import requests
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# API Configuration
API_BASE_URL = "http://localhost:8000"

# Page config
st.set_page_config(
    page_title="Candidate Portal - DSA 9 MVP",
    page_icon="👔",
    layout="wide"
)

# Initialize session state
if 'candidate_resume' not in st.session_state:
    st.session_state.candidate_resume = None
if 'candidate_results' not in st.session_state:
    st.session_state.candidate_results = None

# Header
st.title("👔 Candidate Portal")
st.markdown("Upload your resume and see how you match with job openings")

# API Status check
with st.sidebar:
    st.markdown("### 🔌 Connection Status")
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        if response.status_code == 200:
            st.success("✅ API Connected")
        else:
            st.error("❌ API Error")
    except:
        st.error("❌ API Offline")
        st.info("Please contact the administrator")

st.markdown("---")

# Step 1: Upload Resume
st.header("📄 Step 1: Upload Your Resume")
st.markdown("We support PDF, DOCX, and TXT formats")

uploaded_file = st.file_uploader(
    "Choose your resume file",
    type=['pdf', 'docx', 'txt'],
    help="Upload your resume in PDF, DOCX, or TXT format"
)

if uploaded_file:
    st.success(f"✅ File uploaded: {uploaded_file.name}")
    
    if st.button("🚀 Parse Resume", type="primary"):
        with st.spinner("Analyzing your resume..."):
            try:
                # Send file to backend
                files = {'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                response = requests.post(
                    f"{API_BASE_URL}/api/resume/parse-hybrid",
                    files=files
                )
                
                if response.status_code == 200:
                    resume_data = response.json()
                    st.session_state.candidate_resume = {
                        'filename': uploaded_file.name,
                        'data': resume_data
                    }
                    st.success("✅ Resume parsed successfully!")
                    st.rerun()
                else:
                    st.error(f"❌ Failed to parse resume: {response.text}")
            
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# Step 2: View Extracted Information
if st.session_state.candidate_resume:
    st.markdown("---")
    st.header("📊 Step 2: Your Extracted Information")
    
    resume_data = st.session_state.candidate_resume['data']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👤 Personal Information")
        st.markdown(f"**Name:** {resume_data.get('name', 'Unknown')}")
        st.markdown(f"**Email:** {resume_data.get('email', 'Not found')}")
        st.markdown(f"**Phone:** {resume_data.get('phone', 'Not found')}")
    
    with col2:
        st.subheader("📈 Resume Statistics")
        st.metric("Resume ID", resume_data.get('resume_id', 'N/A'))
        st.metric("File Type", resume_data.get('file_type', 'N/A'))
        st.metric("Text Length", f"{resume_data.get('text_length', 0)} chars")
    
    # Skills
    if resume_data.get('skills'):
        st.subheader("💡 Extracted Skills")
        
        skills = resume_data.get('skills', [])
        if len(skills) > 0:
            # Display skills in two columns
            col1, col2 = st.columns(2)
            mid = len(skills) // 2
            
            with col1:
                for skill in skills[:mid]:
                    st.markdown(f"- {skill}")
            
            with col2:
                for skill in skills[mid:]:
                    st.markdown(f"- {skill}")
    
    # Verify Information
    st.markdown("---")
    st.header("✅ Step 3: Verify Your Information")
    st.markdown("We use AI to extract information. Please verify if everything is correct.")
    
    # Initialize verification state
    if 'verification_done' not in st.session_state:
        st.session_state.verification_done = False
    if 'verification_result' not in st.session_state:
        st.session_state.verification_result = None
    
    if st.button("🔍 Verify Accuracy", type="primary"):
        with st.spinner("Verifying information accuracy..."):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/api/verify/resume",
                    json={
                        "resume_id": resume_data['resume_id'],
                        "resume_extractions": resume_data,
                        "auto_extract_ground_truth": True
                    }
                )
                
                if response.status_code == 200:
                    st.session_state.verification_result = response.json()['verification_report']
                    st.session_state.verification_done = True
                else:
                    st.error("Failed to verify information")
            
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # Show verification results and editable fields
    if st.session_state.verification_done and st.session_state.verification_result:
        verification = st.session_state.verification_result
        
        # Internal fields to exclude from candidate view
        INTERNAL_FIELDS = {'success', 'model_type', 'entities', 'resume_id', 'filename', 
                          'file_type', 'text', 'text_length', 'saved_to_db', 'status', 'message'}
        
        st.subheader("📊 Verification Results")
        
        col1, col2, col3 = st.columns(3)
        
        # Recalculate metrics excluding internal fields
        verified_count = len([item for item in verification.get('verified', []) 
                             if item['field'] not in INTERNAL_FIELDS])
        flagged_count = len([item for item in verification.get('flagged', []) 
                            if item['field'] not in INTERNAL_FIELDS])
        total_fields = verified_count + flagged_count
        
        with col1:
            confidence = (verified_count / total_fields * 100) if total_fields > 0 else 0
            st.metric("Overall Confidence", f"{confidence:.1f}%")
        with col2:
            st.metric("Verified Fields", verified_count)
        with col3:
            st.metric("Flagged Fields", flagged_count)
        
        if flagged_count == 0:
            st.success("✅ All information verified successfully!")
        else:
            st.warning("⚠️ Some information may need your attention")
        
        st.markdown("---")
        st.subheader("📝 Review and Correct Your Information")
        st.markdown("Update any incorrect information below:")
        
        # Show editable fields for flagged items
        flagged_fields = [item for item in verification.get('flagged', []) 
                         if item['field'] not in INTERNAL_FIELDS]
        
        if flagged_fields:
            st.markdown("#### ⚠️ Fields Needing Attention:")
            for item in flagged_fields:
                field_name = item['field']
                current_value = item.get('extraction', '')
                
                with st.expander(f"❌ {field_name.replace('_', ' ').title()}", expanded=True):
                    st.caption(f"Confidence: {item.get('confidence', 0):.1%} | Status: {item['verdict']}")
                    
                    # Handle different field types
                    if field_name == 'skills':
                        current_skills = current_value if isinstance(current_value, list) else []
                        corrected = st.text_area(
                            "Enter skills (one per line):",
                            value="\n".join(current_skills),
                            key=f"correct_{field_name}",
                            help="List your skills, one per line"
                        )
                    elif field_name == 'experience':
                        if isinstance(current_value, dict):
                            col_a, col_b = st.columns(2)
                            with col_a:
                                years = st.number_input("Years", value=current_value.get('years', 0), 
                                                       min_value=0, key=f"correct_{field_name}_years")
                            with col_b:
                                months = st.number_input("Months", value=current_value.get('months', 0), 
                                                        min_value=0, max_value=11, key=f"correct_{field_name}_months")
                    elif field_name == 'education':
                        current_edu = current_value if isinstance(current_value, list) else []
                        corrected = st.text_area(
                            "Enter education (one per line):",
                            value="\n".join(current_edu),
                            key=f"correct_{field_name}",
                            help="List your degrees and institutions"
                        )
                    else:
                        # Simple text field
                        display_value = str(current_value) if current_value else ""
                        if isinstance(current_value, list) and len(current_value) > 0:
                            display_value = current_value[0] if len(current_value) == 1 else ", ".join(map(str, current_value))
                        
                        corrected = st.text_input(
                            "Current Value:",
                            value=display_value,
                            key=f"correct_{field_name}",
                            help=f"Update your {field_name.replace('_', ' ')}"
                        )
        
        # Show verified fields (collapsed by default)
        verified_fields = [item for item in verification.get('verified', []) 
                          if item['field'] not in INTERNAL_FIELDS]
        
        if verified_fields:
            with st.expander(f"✅ Verified Fields ({len(verified_fields)})", expanded=False):
                for item in verified_fields:
                    field_name = item['field']
                    value = item.get('extraction', '')
                    confidence = item.get('confidence', 0)
                    
                    # Format value for display
                    if isinstance(value, list):
                        display_value = ", ".join(map(str, value[:3]))
                        if len(value) > 3:
                            display_value += f" ... (+{len(value)-3} more)"
                    elif isinstance(value, dict):
                        display_value = str(value)
                    else:
                        display_value = str(value)
                    
                    st.success(f"**{field_name.replace('_', ' ').title()}**: {display_value} ({confidence:.1%})")
        
        if flagged_count > 0:
            st.markdown("---")
            if st.button("💾 Save Corrections", type="primary", key="save_corrections"):
                st.success("✅ Your corrections have been noted!")
                st.info("💡 In a full system, these would update your profile.")
    
    # Match with Jobs
    st.markdown("---")
    st.header("🎯 Step 4: Match with Job Openings")
    st.markdown("See how well your profile matches with available positions")
    
    st.info("💡 This feature requires a job description to be uploaded by HR. Please check back later or contact the recruiter.")
    
    # If you want to enable matching, you'd need a way for candidates to select from available JDs
    # For now, just show this is available

# Call to action
if not st.session_state.candidate_resume:
    st.markdown("---")
    st.info("👆 Upload your resume above to get started!")

# Footer
st.markdown("---")
st.markdown("**Candidate Portal** | Powered by AI-based Resume Screening System")
st.markdown("🔒 Your data is secure and used only for matching purposes")

"""Resume Parsing Tab"""
import streamlit as st
import requests


def render(api_base_url: str):
    """Render the Resume Parsing tab"""
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
                        f"{api_base_url}/api/resume/upload",
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
                    resume_id = data.get('resume_id', 'N/A')
                    st.metric("Resume ID", resume_id[:8] if resume_id else 'N/A')
                    st.metric("Name", data.get('name', 'Unknown'))
                
                with col2:
                    email = data.get('email', 'N/A') or 'N/A'
                    st.metric("Email", email[:20] if email else 'N/A')
                    st.metric("Phone", data.get('phone', 'N/A') or 'N/A')
                
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
                            f"{api_base_url}/api/resume/extract-skills",
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
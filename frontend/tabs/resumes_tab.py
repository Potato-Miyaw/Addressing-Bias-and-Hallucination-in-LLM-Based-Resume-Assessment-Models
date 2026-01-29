"""Resume Parsing Tab"""
import streamlit as st
import requests


def render(api_base_url: str, portal_type: str = "hr_upload"):
    """
    Render the Resume Parsing tab
    
    Args:
        api_base_url: Base URL for API calls
        portal_type: "hr_upload" for HR Portal or "candidate_self" for Candidate Portal
    """
    st.header("Resume Management")
    
    # Tabs for Upload vs View
    upload_tab, view_tab = st.tabs(["Upload New Resumes", "View Database Resumes"])
    
    with upload_tab:
        st.markdown("Upload resumes (PDF, DOCX, DOC, TXT) to extract structured information and save to database")
        
        # NER Model selection
        ner_model = st.selectbox(
            "Select NER Model:",
            ["Hybrid NER (Recommended)", "Generic BERT NER", "Resume-Specific NER"],
            help="Hybrid combines both models for best results. Generic BERT is good at names. Resume-Specific excels at skills."
        )
        
        uploaded_files = st.file_uploader(
            "Upload Resume Files",
            type=['txt', 'pdf', 'docx', 'doc'],
            accept_multiple_files=True,
            help="Upload one or more resume files in TXT, PDF, DOCX, or DOC format"
        )
    
    if uploaded_files:
        if st.button("Upload Resumes", type="primary"):
            st.session_state.resumes_data = []
            
            # Map NER model selection to endpoint
            if "Hybrid" in ner_model:
                endpoint = "/api/resume/parse-hybrid"
                model_name = "Hybrid NER"
            elif "Generic" in ner_model:
                endpoint = "/api/resume/parse-generic-file"
                model_name = "Generic BERT NER"
            else:
                endpoint = "/api/resume/parse-v2-file"
                model_name = "Resume-Specific NER"
            
            st.info(f"Uploading {len(uploaded_files)} file(s) using {model_name}...")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Parsing {uploaded_file.name}...")
                
                try:
                    # Reset file pointer to beginning
                    uploaded_file.seek(0)
                    
                    # Send file to selected endpoint
                    files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                    response = requests.post(
                        f"{api_base_url}{endpoint}",
                        files=files,
                        data={
                            "upload_source": portal_type,
                            "uploaded_by": "hr_portal" if portal_type == "hr_upload" else "candidate"
                        }
                    )
                    
                    if response.status_code == 200:
                        resume_data = response.json()
                        st.session_state.resumes_data.append({
                            "filename": uploaded_file.name,
                            "data": resume_data
                        })
                    else:
                        st.error(f"Failed to parse {uploaded_file.name}: {response.status_code}")
                
                except Exception as e:
                    st.error(f"Error parsing {uploaded_file.name}: {str(e)}")
                
                progress_bar.progress((idx + 1) / len(uploaded_files))
            
            status_text.text("All files processed!")
            progress_bar.empty()
            st.success(f"Successfully processed {len(st.session_state.resumes_data)} resumes")
            st.rerun()
    
    # Display parsed resumes
    if st.session_state.resumes_data:
        st.markdown("---")
        st.subheader("Recently Uploaded Resumes")
        st.info("These resumes have been saved to the database")
        
        for idx, resume in enumerate(st.session_state.resumes_data):
            with st.expander(f"{resume['filename']}", expanded=(idx == 0)):
                data = resume['data']
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    resume_id = data.get('resume_id', 'N/A')
                    st.metric("Resume ID", resume_id[:12] if resume_id else 'N/A')
                    st.metric("Name", data.get('name', 'Unknown'))
                
                with col2:
                    email = data.get('email', 'N/A') or 'N/A'
                    st.metric("Email", email if email else 'N/A')
                    st.metric("Phone", data.get('phone', 'N/A') or 'N/A')
                
                with col3:
                    st.metric("Status", data.get('status', 'UNKNOWN'))
                    saved = data.get('saved_to_db', False)
                    st.metric("Saved to DB", "Yes" if saved else "No")
                
                # Show skills if available
                skills = data.get('skills', [])
                if skills:
                    st.markdown("**Extracted Skills:**")
                    st.write(", ".join(skills[:10]))
                    if len(skills) > 10:
                        st.caption(f"... and {len(skills)-10} more")
    
    with view_tab:
        st.markdown("### Resumes in Database")
        st.markdown("View all resumes that have been uploaded by candidates and HR")
        
        # Fetch resumes from database
        try:
            response = requests.get(f"{api_base_url}/api/data/resumes?limit=100")
            
            if response.status_code == 200:
                resumes_data = response.json()
                resumes = resumes_data.get('resumes', [])
                total = resumes_data.get('total', 0)
                
                if not resumes:
                    st.info("No resumes in database yet. Upload some resumes to get started!")
                else:
                    st.success(f"Found {total} resume(s) in database")
                    
                    # Filters
                    col_filter1, col_filter2 = st.columns(2)
                    
                    with col_filter1:
                        source_filter = st.selectbox(
                            "Filter by source:",
                            ["All", "Candidate Self-Upload", "HR Upload"]
                        )
                    
                    with col_filter2:
                        file_type_filter = st.selectbox(
                            "Filter by file type:",
                            ["All", "PDF", "DOCX", "TXT"]
                        )
                    
                    # Apply filters
                    filtered_resumes = resumes
                    
                    if source_filter != "All":
                        source_val = "candidate_self" if "Candidate" in source_filter else "hr_upload"
                        filtered_resumes = [r for r in filtered_resumes if r.get('upload_source') == source_val]
                    
                    if file_type_filter != "All":
                        filtered_resumes = [r for r in filtered_resumes if r.get('file_type') == file_type_filter]
                    
                    st.markdown(f"**Showing {len(filtered_resumes)} resume(s)**")
                    st.markdown("---")
                    
                    # Display resumes
                    for idx, resume in enumerate(filtered_resumes):
                        with st.expander(
                            f"{resume.get('filename', 'Unknown')} - {resume.get('candidate_name', 'No name')}", 
                            expanded=(idx == 0)
                        ):
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Resume ID", resume.get('resume_id', 'N/A')[:12])
                            with col2:
                                st.metric("Candidate", resume.get('candidate_name', 'Unknown'))
                            with col3:
                                st.metric("Email", resume.get('candidate_email', 'Not extracted'))
                            with col4:
                                source = resume.get('upload_source', 'unknown')
                                source_display = "Candidate" if source == "candidate_self" else "HR"
                                st.metric("Uploaded by", source_display)
                            
                            st.markdown("**File Info:**")
                            st.write(f"- File type: {resume.get('file_type', 'N/A')}")
                            st.write(f"- Text length: {resume.get('text_length', 0)} characters")
                            st.write(f"- Uploaded: {resume.get('uploaded_at', 'N/A')[:19] if resume.get('uploaded_at') else 'N/A'}")
                            
                            # Show raw text preview
                            if st.button(f"View Raw Text", key=f"view_{resume.get('resume_id')}"):
                                raw_text = resume.get('raw_text', 'No text available')
                                st.text_area("Raw Resume Text:", raw_text[:2000], height=200)
                                if len(raw_text) > 2000:
                                    st.caption(f"Showing first 2000 of {len(raw_text)} characters")
            else:
                st.error("Could not fetch resumes from database")
        
        except Exception as e:
            st.error(f"Error loading resumes: {str(e)}")
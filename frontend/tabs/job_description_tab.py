"""Job Description Analysis Tab"""
import streamlit as st
import requests


def render(api_base_url: str):
    """Render the Job Description Analysis tab"""
    st.header("📄 Job Description Analysis")
    
    # Add option to load existing job or create new
    st.subheader("Load Existing Job from Database")
    
    try:
        jobs_response = requests.get(f"{api_base_url}/api/data/jobs?limit=50")
        
        if jobs_response.status_code == 200:
            jobs_data = jobs_response.json()
            jobs_list = jobs_data.get('jobs', [])
            
            if jobs_list:
                job_options = {
                    "-- Select a job --": None,
                    **{
                        f"{job['job_title']} ({job['job_id'][:12]}...) - {len(job.get('required_skills', []))} skills": job
                        for job in jobs_list
                    }
                }
                
                selected_job_key = st.selectbox(
                    "Load Previously Saved Job:",
                    options=list(job_options.keys()),
                    key="load_job_selector"
                )
                
                if selected_job_key != "-- Select a job --":
                    selected_job = job_options[selected_job_key]
                    
                    col_load, col_details = st.columns([1, 3])
                    
                    with col_load:
                        if st.button("📥 Load This Job", type="primary"):
                            st.session_state.jd_data = selected_job
                            st.success(f"✅ Loaded: {selected_job['job_title']}")
                            st.rerun()
                    
                    with col_details:
                        st.caption(f"Created: {selected_job.get('created_at', 'N/A')[:10]}")
                        st.caption(f"Experience: {selected_job.get('required_experience', 0)} years | Education: {selected_job.get('required_education', 'N/A')}")
            else:
                st.info("No saved jobs found. Create a new one below.")
    except Exception as e:
        st.warning(f"Could not load jobs from database: {str(e)}")
    
    st.markdown("---")
    st.subheader(" Create New Job Description")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write("Enter Job Description")
        
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
        
        save_to_db = st.checkbox("💾 Save to Database", value=True, help="Save this job description for future use")
        
        if st.button("🔍 Analyze Job Description", type="primary"):
            if jd_text:
                with st.spinner("Analyzing job description..."):
                    try:
                        response = requests.post(
                            f"{api_base_url}/api/jd/extract",
                            json={
                                "jd_text": jd_text,
                                "job_title": job_title,
                                "save_to_db": save_to_db
                            }
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            st.session_state.jd_data = result
                            
                            if result.get('saved_to_db'):
                                st.success(f"✅ Job analyzed and saved to database! ID: `{result['job_id'][:16]}...`")
                            elif result.get('duplicate_warning'):
                                st.warning(f"⚠️ {result['duplicate_warning']}")
                                st.info("Job analyzed but not saved (duplicate content detected)")
                            else:
                                st.success("✅ Job analyzed successfully!")
                            
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
            
            # Handle both 'status' (from API) and 'extraction_status' (from DB)
            status = data.get('status') or data.get('extraction_status', 'N/A')
            st.markdown(f"**Status:** {status}")
            
            if data.get('job_title'):
                st.markdown(f"**Title:** {data['job_title']}")
            
            st.markdown("---")
            
            st.metric("Required Experience", f"{data.get('required_experience', 0)} years")
            st.metric("Education Level", data.get('required_education', 'Not specified'))
            
            st.markdown("**Required Skills:**")
            if data.get('required_skills'):
                for skill in data['required_skills']:
                    st.markdown(f"- {skill}")
            else:
                st.info("No skills extracted")
            
            st.markdown("**Certifications:**")
            if data.get('certifications'):
                for cert in data['certifications']:
                    st.markdown(f"- {cert}")
            else:
                st.info("No certifications required")
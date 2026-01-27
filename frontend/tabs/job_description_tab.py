"""Job Description Analysis Tab"""
import streamlit as st
import requests


def render(api_base_url: str):
    """Render the Job Description Analysis tab"""
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
                            f"{api_base_url}/api/jd/extract",
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
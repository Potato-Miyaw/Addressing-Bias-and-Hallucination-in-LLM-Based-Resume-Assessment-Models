"""Complete Pipeline Tab"""
import streamlit as st
import requests


def render(api_base_url: str):
    """Render the Complete Pipeline tab"""
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
                            f"{api_base_url}/api/pipeline/complete",
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
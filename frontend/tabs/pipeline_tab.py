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
        fairness_method = None
        if use_pipeline_fairness:
            fairness_method = st.selectbox("Fairness Method", ["expgrad", "reweighing", "threshold"], index=0, key="pipeline_fairness_method")

        
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
                                "use_fairness": use_pipeline_fairness,
                                "fairness_method": fairness_method
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

        fairness_metrics = results.get('fairness_metrics')
        if isinstance(fairness_metrics, dict):
            st.subheader("Fairness Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric("Impact Ratio", fairness_metrics.get("impact_ratio"))
            col2.metric("DP Diff", fairness_metrics.get("demographic_parity"))
            col3.metric("EO Diff", fairness_metrics.get("equal_opportunity"))

            st.subheader("Exports")
            export_payload = {
                "job_id": results.get('job_id', 'JOB'),
                "ranked_candidates": results.get('ranked_candidates', []),
                "fairness_metrics": fairness_metrics or {},
                "fairness_enabled": results.get('fairness_enabled', False),
                "hire_threshold": 0.5,
            }
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Prepare Pipeline Shortlist CSV"):
                    try:
                        csv_resp = requests.post(f"{api_base_url}/api/rank/export/shortlist", json=export_payload)
                        if csv_resp.status_code == 200:
                            st.session_state.pipeline_shortlist_csv = csv_resp.content
                            st.session_state.pipeline_shortlist_name = f"shortlist_{export_payload['job_id']}.csv"
                        else:
                            st.error(f"CSV export failed: {csv_resp.text}")
                    except Exception as e:
                        st.error(f"CSV export failed: {str(e)}")
                if hasattr(st.session_state, 'pipeline_shortlist_csv'):
                    st.download_button(
                        label="Download Pipeline Shortlist CSV",
                        data=st.session_state.pipeline_shortlist_csv,
                        file_name=getattr(st.session_state, 'pipeline_shortlist_name', 'shortlist.csv'),
                        mime="text/csv",
                    )

            with col_b:
                if st.button("Prepare Pipeline Audit PDF"):
                    try:
                        pdf_resp = requests.post(f"{api_base_url}/api/rank/export/audit", json=export_payload)
                        if pdf_resp.status_code == 200:
                            st.session_state.pipeline_audit_pdf = pdf_resp.content
                            st.session_state.pipeline_audit_name = f"audit_{export_payload['job_id']}.pdf"
                        else:
                            st.error(f"Audit export failed: {pdf_resp.text}")
                    except Exception as e:
                        st.error(f"Audit export failed: {str(e)}")
                if hasattr(st.session_state, 'pipeline_audit_pdf'):
                    st.download_button(
                        label="Download Pipeline Audit PDF",
                        data=st.session_state.pipeline_audit_pdf,
                        file_name=getattr(st.session_state, 'pipeline_audit_name', 'audit.pdf'),
                        mime="application/pdf",
                    )
        
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
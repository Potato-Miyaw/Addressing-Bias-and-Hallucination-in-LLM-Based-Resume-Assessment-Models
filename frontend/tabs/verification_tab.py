"""Verification Tab"""
import streamlit as st
import requests


def render(api_base_url: str):
    """Render the Verification tab"""
    st.header("Hallucination Detection")
    
    if not st.session_state.resumes_data:
        st.warning("Please upload and parse resumes first (Tab 2)")
    else:
        st.markdown("Verify resume claims for hallucinations using Token Overlap + BERTScore")
        
        selected_resume = st.selectbox(
            "Select Resume to Verify",
            options=range(len(st.session_state.resumes_data)),
            format_func=lambda x: st.session_state.resumes_data[x]['filename']
        )
        
        if st.button("Verify Resume", type="primary"):
            with st.spinner("Verifying claims..."):
                try:
                    resume_data = st.session_state.resumes_data[selected_resume]['data']
                    
                    # Make sure we have the full extracted data
                    if 'text' not in resume_data:
                        st.error("Resume text not available. Please re-upload the resume.")
                    else:
                        response = requests.post(
                            f"{api_base_url}/api/verify/resume",
                            json={
                                "resume_id": resume_data['resume_id'],
                                "resume_extractions": resume_data,
                                "auto_extract_ground_truth": True  # Enable auto-extraction
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
            st.subheader("Verification Report")
            
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
                st.success("All claims verified!")
            elif report['verdict'] == "NO_CLAIMS":
                st.info("Ground truth auto-extracted from resume text")
            else:
                st.warning(f"{report['verdict']}")
            
            # Show details if available
            if report.get('details'):
                with st.expander("Detailed Verification Results"):
                    for claim in report['details']:
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            if claim['is_hallucination']:
                                st.error(f"**{claim['field']}**: {claim['verdict']} (confidence: {claim['confidence']:.2f})")
                            else:
                                st.success(f"**{claim['field']}**: {claim['verdict']} (confidence: {claim['confidence']:.2f})")
                            
                            # Show extraction vs ground truth
                            st.markdown(f"**Extracted:** `{claim['extraction']}`")
                            st.markdown(f"**Ground Truth:** `{claim['ground_truth']}`")
                            
                            # Show evidence snippet if available
                            if claim.get('evidence_snippet'):
                                with st.expander("Evidence from Resume"):
                                    st.code(claim['evidence_snippet'], language="text")
                        
                        with col2:
                            # Show metrics
                            st.metric("Token Overlap", f"{claim.get('token_overlap', 0):.2f}")
                            st.metric("BERTScore", f"{claim.get('bertscore_f1', 0):.2f}")
                        
                        st.markdown("---")
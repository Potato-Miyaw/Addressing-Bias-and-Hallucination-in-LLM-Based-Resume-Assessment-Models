"""Matching & Ranking Tab"""
import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go


def render(api_base_url: str):
    """Render the Matching & Ranking tab"""
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
                            f"{api_base_url}/api/match/",
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
                        f"{api_base_url}/api/rank/",
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
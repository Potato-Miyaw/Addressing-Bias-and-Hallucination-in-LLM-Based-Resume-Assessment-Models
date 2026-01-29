"""Matching & Ranking Tab"""
import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go


def render(api_base_url: str, portal_type: str = "hr_portal"):
    """Render the Matching & Ranking tab
    
    Args:
        api_base_url: Base URL for API calls
        portal_type: Type of portal (hr_portal or candidate_portal)
    """
    st.header("🎯 Job-Resume Matching & Ranking")
    
    # Job Selection Section
    st.subheader("1️⃣ Select Job Description")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Option to use existing job or current session job
        job_source = st.radio(
            "Job Description Source:",
            ["Use Current Session Job", "Select from Database"],
            horizontal=True
        )
    
    selected_job = None
    
    if job_source == "Use Current Session Job":
        if not st.session_state.jd_data:
            st.warning("⚠️ Please analyze a job description first (Tab 1)")
            return
        else:
            selected_job = st.session_state.jd_data
            st.info(f"📋 Using: **{selected_job.get('job_title', 'Current Job')}** (ID: `{selected_job['job_id'][:16]}...`)")
    
    else:  # Select from Database
        try:
            # Fetch jobs from database
            jobs_response = requests.get(f"{api_base_url}/api/data/jobs?limit=100")
            
            if jobs_response.status_code == 200:
                jobs_data = jobs_response.json()
                jobs_list = jobs_data.get('jobs', [])
                
                if not jobs_list:
                    st.warning("⚠️ No jobs found in database. Please add a job description first (Tab 1)")
                    return
                
                # Create job options for selectbox
                job_options = {
                    f"{job['job_title']} - {job['job_id'][:16]}... ({len(job.get('required_skills', []))} skills)": job
                    for job in jobs_list
                }
                
                selected_job_key = st.selectbox(
                    "Select Job Description:",
                    options=list(job_options.keys()),
                    key="job_selector"
                )
                
                selected_job = job_options[selected_job_key]
                
                # Show job details in expander
                with st.expander("📄 View Job Requirements"):
                    st.write(f"**Job ID:** `{selected_job['job_id']}`")
                    st.write(f"**Title:** {selected_job.get('job_title', 'N/A')}")
                    st.write(f"**Experience Required:** {selected_job.get('required_experience', 0)} years")
                    st.write(f"**Education:** {selected_job.get('required_education', 'N/A')}")
                    
                    st.write("**Required Skills:**")
                    for skill in selected_job.get('required_skills', []):
                        st.write(f"  • {skill}")
                    
                    if selected_job.get('certifications'):
                        st.write("**Certifications:**")
                        for cert in selected_job.get('certifications', []):
                            st.write(f"  • {cert}")
            else:
                st.error("Failed to fetch jobs from database")
                return
        except Exception as e:
            st.error(f"Error fetching jobs: {str(e)}")
            return
    
    st.markdown("---")
    st.subheader("2️⃣ Select Resumes and Match")
    
    # Resume source selection
    resume_source = st.radio(
        "Resume Source:",
        ["Use Current Session Resumes", "Select from Database"],
        horizontal=True,
        key="resume_source_selector"
    )
    
    selected_resumes = []
    
    if resume_source == "Use Current Session Resumes":
        if not st.session_state.resumes_data:
            st.warning("⚠️ Please upload and parse resumes first (Tab 2)")
            return
        else:
            selected_resumes = st.session_state.resumes_data
            st.info(f"📄 Using {len(selected_resumes)} resume(s) from current session")
    
    else:  # Select from Database
        try:
            # Fetch resumes from database (max 100 per API limit)
            resumes_response = requests.get(f"{api_base_url}/api/data/resumes?limit=100")
            
            if resumes_response.status_code == 200:
                resumes_data = resumes_response.json()
                db_resumes = resumes_data.get('resumes', [])
                
                if not db_resumes:
                    st.warning("⚠️ No resumes found in database. Please upload resumes first (Tab 2)")
                    return
                
                st.success(f"✅ Found {len(db_resumes)} resume(s) in database")
                
                # Filter options
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    upload_source_filter = st.selectbox(
                        "Filter by Source:",
                        ["All", "HR Uploaded", "Candidate Self-Upload"],
                        key="source_filter"
                    )
                
                with col2:
                    file_type_filter = st.selectbox(
                        "Filter by Type:",
                        ["All", "PDF", "DOCX", "TXT"],
                        key="type_filter"
                    )
                
                # Apply filters
                filtered_resumes = db_resumes
                if upload_source_filter == "HR Uploaded":
                    filtered_resumes = [r for r in filtered_resumes if r.get('upload_source') == 'hr_upload']
                elif upload_source_filter == "Candidate Self-Upload":
                    filtered_resumes = [r for r in filtered_resumes if r.get('upload_source') == 'candidate_self']
                
                if file_type_filter != "All":
                    filtered_resumes = [r for r in filtered_resumes if r.get('file_type', '').upper() == file_type_filter]
                
                if not filtered_resumes:
                    st.warning("⚠️ No resumes match the selected filters")
                    return
                
                st.info(f"📊 {len(filtered_resumes)} resume(s) after filtering")
                
                # Multi-select for resumes
                st.markdown("**Select Resumes to Match:**")
                
                # Create resume options with details
                resume_options = {}
                for resume in filtered_resumes:
                    name = resume.get('candidate_name', 'Unknown')
                    email = resume.get('candidate_email', 'No email')
                    resume_id = resume.get('resume_id', 'unknown')[:12]
                    source = "🏢 HR" if resume.get('upload_source') == 'hr_upload' else "👤 Self"
                    
                    label = f"{name} ({email}) - {resume_id}... {source}"
                    resume_options[label] = resume
                
                # Select all checkbox
                select_all = st.checkbox("Select All Resumes", key="select_all_resumes")
                
                if select_all:
                    selected_resume_keys = list(resume_options.keys())
                else:
                    selected_resume_keys = st.multiselect(
                        "Choose resumes:",
                        options=list(resume_options.keys()),
                        key="resume_multiselect",
                        help="Select one or more resumes to match against the job"
                    )
                
                # Convert to resume data format
                for key in selected_resume_keys:
                    resume = resume_options[key]
                    
                    # Ensure the resume has the 'text' field for parsing
                    # Ground truth has 'raw_text', but matcher expects 'text'
                    if 'raw_text' in resume and 'text' not in resume:
                        resume['text'] = resume['raw_text']
                    
                    # Format to match session state structure
                    selected_resumes.append({
                        'data': resume
                    })
                
                if selected_resumes:
                    st.success(f"✅ Selected {len(selected_resumes)} resume(s) for matching")
                else:
                    st.info("👆 Please select at least one resume")
                    return
                    
            else:
                st.error("Failed to fetch resumes from database")
                return
        except Exception as e:
            st.error(f"Error fetching resumes: {str(e)}")
            return
    
    if selected_resumes and selected_job:
        st.success("✅ Ready to match and rank candidates!")
        
        use_fairness = st.checkbox("Use Fairness-Aware Ranking", value=False,
                                   help="Apply fairness constraints using Fairlearn")
        fairness_method = None
        if use_fairness:
            fairness_method = st.selectbox("Fairness Method", ["expgrad", "reweighing", "threshold"], index=0)

        
        if st.button("🚀 Match & Rank Candidates", type="primary"):
            with st.spinner("Processing candidates..."):
                try:
                    # First, check for existing matches to avoid duplicates
                    st.info("🔍 Checking for existing matches...")
                    existing_matches_response = requests.get(
                        f"{api_base_url}/api/matches/job/{selected_job['job_id']}"
                    )
                    
                    existing_match_map = {}
                    if existing_matches_response.status_code == 200:
                        existing_matches = existing_matches_response.json()
                        # Map resume_id -> match for quick lookup
                        for match in existing_matches:
                            existing_match_map[match['resume_id']] = match
                        st.info(f"📊 Found {len(existing_match_map)} existing match(es) for this job")
                    
                    # Prepare candidates data
                    candidates = []
                    newly_matched = 0
                    reused_matches = 0
                    
                    for resume in selected_resumes:
                        resume_data = resume['data']
                        resume_id = resume_data.get('resume_id', 'unknown')
                        
                        # Check if match already exists
                        if resume_id in existing_match_map:
                            # Reuse existing match
                            existing_match = existing_match_map[resume_id]
                            reused_matches += 1
                            
                            # Convert existing match to expected format
                            match_result = {
                                "match_score": existing_match['overall_match_score'] * 100,
                                "skill_match": existing_match['skill_match_score'] * 100,
                                "experience_match": existing_match['experience_match_score'] * 100,
                                "education_match": existing_match['education_match_score'] * 100,
                                "matched_skills": existing_match['matched_skills'],
                                "skill_gaps": existing_match['missing_skills'],
                                "overall_score": existing_match['overall_match_score']
                            }
                            
                            candidates.append({
                                "resume_id": resume_id,
                                "resume_data": resume_data,
                                "match_data": match_result,
                                "demographics": {"gender": 1, "race_gender": "Unknown"}
                            })
                        else:
                            # Create new match
                            newly_matched += 1
                            
                            # Determine match_source based on portal type
                            match_source = "hr_initiated" if portal_type == "hr_portal" else "candidate_initiated"
                            
                            # Match to job using the new endpoint with job_id
                            match_response = requests.post(
                                f"{api_base_url}/api/match/with-job-id",
                                json={
                                    "resume_id": resume_data.get('resume_id', 'unknown'),
                                    "job_id": selected_job['job_id'],
                                    "resume_data": resume_data,  # Pass full data (includes 'text' field)
                                    "match_source": match_source
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
                    
                    # Show matching summary
                    if newly_matched > 0 or reused_matches > 0:
                        st.success(f"✅ Matching complete! {newly_matched} new match(es), {reused_matches} reused from database")
                    
                    # Rank candidates
                    rank_response = requests.post(
                        f"{api_base_url}/api/rank/",
                        json={
                            "job_id": selected_job['job_id'],
                            "candidates": candidates,
                            "jd_data": {
                                "required_skills": selected_job.get('required_skills', []),
                                "required_experience": selected_job.get('required_experience', 0),
                                "required_education": selected_job.get('required_education', '')
                            },
                            "use_fairness": use_fairness,
                            "fairness_method": fairness_method
                        }
                    )
                    
                    if rank_response.status_code == 200:
                        ranked_results = rank_response.json()
                        st.session_state.ranked_candidates = ranked_results['ranked_candidates']
                        st.session_state.rank_fairness_metrics = ranked_results.get('fairness_metrics')
                        st.session_state.rank_job_id = ranked_results.get('job_id', selected_job['job_id'])
                        st.session_state.rank_fairness_enabled = ranked_results.get('fairness_enabled', use_fairness)
                        st.session_state.selected_job_for_ranking = selected_job  # Store for display
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
            st.dataframe(df, hide_index=True)

            # Fairness metrics summary
            fairness_metrics = getattr(st.session_state, 'rank_fairness_metrics', None)
            if isinstance(fairness_metrics, dict):
                st.subheader("Fairness Metrics")
                col1, col2, col3 = st.columns(3)
                col1.metric("Impact Ratio", fairness_metrics.get("impact_ratio"))
                col2.metric("DP Diff", fairness_metrics.get("demographic_parity"))
                col3.metric("EO Diff", fairness_metrics.get("equal_opportunity"))

            # Exports
            st.subheader("Exports")
            export_payload = {
                "job_id": getattr(st.session_state, 'rank_job_id', 'JOB'),
                "ranked_candidates": st.session_state.ranked_candidates,
                "fairness_metrics": fairness_metrics or {},
                "fairness_enabled": getattr(st.session_state, 'rank_fairness_enabled', False),
                "hire_threshold": 0.5,
            }

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Prepare Shortlist CSV"):
                    try:
                        csv_resp = requests.post(f"{api_base_url}/api/rank/export/shortlist", json=export_payload)
                        if csv_resp.status_code == 200:
                            st.session_state.shortlist_csv = csv_resp.content
                            st.session_state.shortlist_csv_name = f"shortlist_{export_payload['job_id']}.csv"
                        else:
                            st.error(f"CSV export failed: {csv_resp.text}")
                    except Exception as e:
                        st.error(f"CSV export failed: {str(e)}")
                if hasattr(st.session_state, 'shortlist_csv'):
                    st.download_button(
                        label="Download Shortlist CSV",
                        data=st.session_state.shortlist_csv,
                        file_name=getattr(st.session_state, 'shortlist_csv_name', 'shortlist.csv'),
                        mime="text/csv",
                    )

            with col2:
                if st.button("Prepare Audit PDF"):
                    try:
                        pdf_resp = requests.post(f"{api_base_url}/api/rank/export/audit", json=export_payload)
                        if pdf_resp.status_code == 200:
                            st.session_state.audit_pdf = pdf_resp.content
                            st.session_state.audit_pdf_name = f"audit_{export_payload['job_id']}.pdf"
                        else:
                            st.error(f"Audit export failed: {pdf_resp.text}")
                    except Exception as e:
                        st.error(f"Audit export failed: {str(e)}")
                if hasattr(st.session_state, 'audit_pdf'):
                    st.download_button(
                        label="Download Audit PDF",
                        data=st.session_state.audit_pdf,
                        file_name=getattr(st.session_state, 'audit_pdf_name', 'audit.pdf'),
                        mime="application/pdf",
                    )
            # Send summary to Teams
            st.subheader("Send Summary to Teams")
            if st.button("Send to Teams"):
                try:
                    notify_payload = {
                        "job_id": getattr(st.session_state, 'rank_job_id', 'JOB'),
                        "ranked_candidates": st.session_state.ranked_candidates,
                        "fairness_metrics": fairness_metrics or {},
                    }
                    notify_resp = requests.post(f"{api_base_url}/api/notify/power-automate", json=notify_payload)
                    if notify_resp.status_code == 200:
                        st.success("Teams notification sent")
                    else:
                        st.error(f"Teams notification failed: {notify_resp.text}")
                except Exception as e:
                    st.error(f"Teams notification failed: {str(e)}")

            # Detailed view
            st.subheader("Detailed Candidate Reports")
            
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

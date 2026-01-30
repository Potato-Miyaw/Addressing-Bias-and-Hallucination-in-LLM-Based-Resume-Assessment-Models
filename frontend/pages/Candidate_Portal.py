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

from dotenv import load_dotenv as load
load()

# API Configuration
API_BASE_URL = os.getenv("API_BASE_URL")

# Page config
st.set_page_config(
    page_title="Candidate Portal - DSA 9 MVP",
    layout="wide"
)

# Initialize session state
if 'candidate_resume' not in st.session_state:
    st.session_state.candidate_resume = None
if 'candidate_results' not in st.session_state:
    st.session_state.candidate_results = None

# Header
st.title("Candidate Portal")
st.markdown("Upload your resume and see how you match with job openings")

# API Status check
with st.sidebar:
    st.markdown("### Connection Status")
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        if response.status_code == 200:
            st.success("API Connected")
        else:
            st.error("API Error")
    except:
        st.error("API Offline")
        st.info("Please contact the administrator")

st.markdown("---")

# Step 1: Upload Resume
st.header("Upload Your Resume")
st.markdown("PDF, DOCX, and TXT formats")

# NER Model Selection
st.subheader("Choose Extraction Model")
ner_model = st.radio(
    "Select the AI model for resume analysis:",
    [
        "Hybrid (Recommended) - Best of both worlds",
        "Generic BERT - Good for standard resumes",
        "Resume-Specific - Optimized for technical resumes"
    ],
    index=0,
    help="Hybrid combines multiple models for best accuracy"
)

# Map selection to endpoint
if "Hybrid" in ner_model:
    endpoint = "/api/resume/parse-hybrid"
    model_name = "Hybrid NER"
elif "Generic" in ner_model:
    endpoint = "/api/resume/parse-generic-file"
    model_name = "Generic BERT NER"
else:
    endpoint = "/api/resume/parse-v2-file"
    model_name = "Resume-Specific NER"

uploaded_file = st.file_uploader(
    "Choose your resume file",
    type=['pdf', 'docx', 'txt'],
    help="Upload your resume in PDF, DOCX, or TXT format"
)

if uploaded_file:
    st.success(f"File uploaded: {uploaded_file.name}")
    st.info(f"Selected model: **{model_name}**")
    
    if st.button("Parse Resume", type="primary"):
        with st.spinner(f"Analyzing your resume with {model_name}..."):
            try:
                # Send file to backend
                files = {'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                response = requests.post(
                    f"{API_BASE_URL}{endpoint}",
                    files=files
                )
                
                if response.status_code == 200:
                    resume_data = response.json()
                    st.session_state.candidate_resume = {
                        'filename': uploaded_file.name,
                        'data': resume_data
                    }
                    
                    # Show success message
                    if resume_data.get('saved_to_db'):
                        st.success("Resume parsed and saved to database!")
                        if resume_data.get('is_duplicate'):
                            st.warning(f"{resume_data.get('duplicate_warning', 'Duplicate detected')}")
                    else:
                        st.success("Resume parsed successfully!")
                    
                    st.rerun()
                else:
                    st.error(f"Failed to parse resume: {response.text}")
            
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Step 2: View Extracted Information
if st.session_state.candidate_resume:
    st.markdown("---")
    st.header("Your Extracted Information")
    
    resume_data = st.session_state.candidate_resume['data']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Personal Information")
        st.markdown(f"**Name:** {resume_data.get('name', 'Unknown')}")
        st.markdown(f"**Email:** {resume_data.get('email', 'Not found')}")
        st.markdown(f"**Phone:** {resume_data.get('phone', 'Not found')}")
        st.markdown(f"**Location:** {resume_data.get('location', 'Not specified')}")
    
    with col2:
        st.subheader("Resume Statistics")
        st.metric("Resume ID", resume_data.get('resume_id', 'N/A'))
        st.metric("File Type", resume_data.get('file_type', 'N/A'))
        st.metric("Text Length", f"{resume_data.get('text_length', 0)} chars")
    
    # Skills
    if resume_data.get('skills'):
        st.subheader("Extracted Skills")
        
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
    
    # Job History & Certifications
    exp_data = resume_data.get('experience', {})
    if exp_data.get('job_history') or resume_data.get('certifications'):
        st.markdown("---")
        col_job, col_cert = st.columns(2)
        
        with col_job:
            job_history = exp_data.get('job_history', [])
            if job_history:
                st.subheader("Work History")
                for i, job in enumerate(job_history[:5]):
                    if isinstance(job, dict):
                        st.markdown(f"{i+1}. **{job.get('title', 'Position')}** at {job.get('company', 'Company')}")
                    else:
                        st.markdown(f"{i+1}. {job}")
                if len(job_history) > 5:
                    st.caption(f"... and {len(job_history)-5} more")
            elif exp_data.get('companies'):
                st.subheader("Companies")
                for company in exp_data.get('companies', [])[:5]:
                    st.markdown(f"- {company}")
        
        with col_cert:
            certifications = resume_data.get('certifications', [])
            if certifications:
                st.subheader("Certifications")
                for cert in certifications[:5]:
                    st.markdown(f"- {cert}")
                if len(certifications) > 5:
                    st.caption(f"... and {len(certifications)-5} more")
    
    # Verify Information
    st.markdown("---")
    st.header("Verify & Update Your Information")
    st.markdown("We use AI to extract information. Please verify and correct any inaccuracies.")
    
    # Initialize verification state
    if 'verification_done' not in st.session_state:
        st.session_state.verification_done = False
    if 'verification_result' not in st.session_state:
        st.session_state.verification_result = None
    if 'corrections_made' not in st.session_state:
        st.session_state.corrections_made = {}
    
    # Show current extracted data for review
    with st.expander("Review Extracted Information", expanded=True):
        col_rev1, col_rev2 = st.columns(2)
        
        with col_rev1:
            st.markdown("**Contact Information:**")
            st.write(f"Name: {resume_data.get('name', 'Not extracted')}")
            st.write(f"Email: {resume_data.get('email', 'Not extracted')}")
            st.write(f"Phone: {resume_data.get('phone', 'Not extracted')}")
            st.write(f"Location: {resume_data.get('location', 'Not extracted')}")
        
        with col_rev2:
            st.markdown("**Professional Details:**")
            exp_data = resume_data.get('experience', {})
            if isinstance(exp_data, dict):
                st.write(f"Experience: {exp_data.get('years', 0)} years")
            else:
                st.write("Experience: Not extracted")
            
            skills = resume_data.get('skills', [])
            st.write(f"Skills: {len(skills)} extracted")
            
            education = resume_data.get('education', [])
            st.write(f"Education: {len(education)} entry/entries")
    
    col_verify, col_edit = st.columns(2)
    
    with col_verify:
        if st.button("Run AI Verification", type="primary"):
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
                        st.rerun()
                    else:
                        st.error("Failed to verify information")
                
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    with col_edit:
        if st.button("Edit My Information Manually", type="secondary"):
            st.session_state.show_edit_form = True
            st.rerun()
    
    # Show verification results
    if st.session_state.verification_done and st.session_state.verification_result:
        verification = st.session_state.verification_result
        
        # Internal fields to exclude from candidate view
        INTERNAL_FIELDS = {'success', 'model_type', 'entities', 'resume_id', 'filename', 
                          'file_type', 'text', 'text_length', 'saved_to_db', 'status', 'message'}
        
        st.markdown("---")
        st.subheader("Verification Results")
        
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
            st.success("All information verified successfully!")
        else:
            st.warning(f"{flagged_count} field(s) may need your attention")
        
        # Show flagged fields with edit capability
        flagged_fields = [item for item in verification.get('flagged', []) 
                         if item['field'] not in INTERNAL_FIELDS]
        
        if flagged_fields:
            st.markdown("---")
            st.markdown("### Fields Needing Attention - Please Review and Correct")
            
            for item in flagged_fields:
                field_name = item['field']
                st.info(f"**{field_name.replace('_', ' ').title()}** needs your review")
    
    # Manual edit form (shown when "Edit My Information" is clicked or when flagged fields exist)
    if st.session_state.get('show_edit_form', False) or (st.session_state.verification_done and 
                                                           st.session_state.verification_result and 
                                                           len([item for item in st.session_state.verification_result.get('flagged', []) 
                                                                if item['field'] not in {'success', 'model_type', 'entities', 'resume_id', 
                                                                                        'filename', 'file_type', 'text', 'text_length', 
                                                                                        'saved_to_db', 'status', 'message'}]) > 0):
        st.markdown("---")
        st.subheader("Update Your Information")
        st.markdown("Modify any incorrect information below. Changes will be used for job matching.")
        
        with st.form("correction_form"):
            st.markdown("#### Personal Information")
            
            col1, col2 = st.columns(2)
            
            with col1:
                corrected_name = st.text_input(
                    "Full Name",
                    value=resume_data.get('name', ''),
                    help="Your full name as you want it to appear"
                )
            
            with col2:
                corrected_email = st.text_input(
                    "Email Address",
                    value=resume_data.get('email', ''),
                    help="Your contact email"
                )
            
            col_phone, col_location = st.columns(2)
            
            with col_phone:
                corrected_phone = st.text_input(
                    "Phone Number",
                    value=resume_data.get('phone', ''),
                    help="Your contact phone number"
                )
            
            with col_location:
                corrected_location = st.text_input(
                    "Current Location",
                    value=resume_data.get('location', ''),
                    help="Your current city/location"
                )
            
            st.markdown("#### Professional Experience")
            
            col3, col4 = st.columns(2)
            
            exp_data = resume_data.get('experience', {})
            current_years = exp_data.get('years', 0) if isinstance(exp_data, dict) else 0
            current_months = exp_data.get('months', 0) if isinstance(exp_data, dict) else 0
            # Cap months at 11 (valid range is 0-11)
            current_months = min(int(current_months), 11)
            
            with col3:
                corrected_exp_years = st.number_input(
                    "Years of Experience",
                    min_value=0,
                    max_value=50,
                    value=int(current_years),
                    help="Total years of professional experience"
                )
            
            with col4:
                corrected_exp_months = st.number_input(
                    "Additional Months",
                    min_value=0,
                    max_value=11,
                    value=current_months,
                    help="Additional months (0-11)"
                )
            
            st.markdown("#### Skills")
            
            current_skills = resume_data.get('skills', [])
            corrected_skills = st.text_area(
                "Your Skills (one per line)",
                value="\n".join(current_skills) if current_skills else "",
                height=150,
                help="List your technical and professional skills, one per line"
            )
            
            st.markdown("#### Education")
            
            current_education = resume_data.get('education', [])
            if isinstance(current_education, list):
                # Convert education list to string format
                edu_text = ""
                for edu in current_education:
                    if isinstance(edu, dict):
                        degree = edu.get('degree', '')
                        institution = edu.get('institution', '')
                        edu_text += f"{degree} - {institution}\n"
                    elif isinstance(edu, str):
                        edu_text += f"{edu}\n"
            else:
                edu_text = str(current_education) if current_education else ""
            
            corrected_education = st.text_area(
                "Your Education (one per line)",
                value=edu_text,
                height=100,
                help="List your degrees and institutions, e.g., 'Bachelor of Science - MIT'"
            )
            
            st.markdown("#### Certifications")
            
            current_certs = resume_data.get('certifications', [])
            corrected_certs = st.text_area(
                "Your Certifications (one per line)",
                value="\n".join(current_certs) if current_certs else "",
                height=80,
                help="List your professional certifications"
            )
            
            # Submit button
            submitted = st.form_submit_button("Save All Changes", type="primary")
            
            if submitted:
                # Parse the corrected data
                skills_list = [s.strip() for s in corrected_skills.split('\n') if s.strip()]
                education_list = [e.strip() for e in corrected_education.split('\n') if e.strip()]
                certs_list = [c.strip() for c in corrected_certs.split('\n') if c.strip()]
                
                # Update the resume data in session state
                updated_data = resume_data.copy()
                updated_data['name'] = corrected_name
                updated_data['email'] = corrected_email
                updated_data['phone'] = corrected_phone
                updated_data['location'] = corrected_location
                updated_data['experience'] = {
                    'years': corrected_exp_years,
                    'months': corrected_exp_months
                }
                updated_data['total_experience_(years)'] = corrected_exp_years
                updated_data['total_experience_(months)'] = corrected_exp_years * 12 + corrected_exp_months
                updated_data['skills'] = skills_list
                updated_data['education'] = education_list
                updated_data['certifications'] = certs_list
                
                # Save corrections
                st.session_state.candidate_resume['data'] = updated_data
                st.session_state.corrections_made = {
                    'name': corrected_name != resume_data.get('name', ''),
                    'email': corrected_email != resume_data.get('email', ''),
                    'phone': corrected_phone != resume_data.get('phone', ''),
                    'location': corrected_location != resume_data.get('location', ''),
                    'experience': corrected_exp_years != current_years or corrected_exp_months != current_months,
                    'skills': skills_list != current_skills,
                    'education': education_list != current_education,
                    'certifications': certs_list != current_certs
                }
                
                st.success("Your information has been updated successfully!")
                st.info("Your updated information will now be used for job matching.")
                
                # Clear the job matches so they can be recalculated with new data
                if 'candidate_job_matches' in st.session_state:
                    st.session_state.candidate_job_matches = None
                
                st.balloons()
                st.rerun()
        
        # Show summary of corrections made
        if any(st.session_state.corrections_made.values()):
            st.markdown("---")
            st.success("### Recent Changes")
            st.markdown("You have updated the following information:")
            
            for field, changed in st.session_state.corrections_made.items():
                if changed:
                    st.write(f"✓ {field.replace('_', ' ').title()}")
    
    # Show verified fields summary
    if st.session_state.verification_done and st.session_state.verification_result:
        verified_fields = [item for item in st.session_state.verification_result.get('verified', []) 
                          if item['field'] not in {'success', 'model_type', 'entities', 'resume_id', 
                                                   'filename', 'file_type', 'text', 'text_length', 
                                                   'saved_to_db', 'status', 'message'}]
        
        if verified_fields:
            with st.expander(f"Verified Fields ({len(verified_fields)})", expanded=False):
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
    
    # Match with Jobs
    st.markdown("---")
    st.header("Match with Job Openings")
    st.markdown("See how well your profile matches with available positions")
    
    # Initialize match results in session state
    if 'candidate_job_matches' not in st.session_state:
        st.session_state.candidate_job_matches = None
    
    # Fetch available jobs from database
    try:
        jobs_response = requests.get(f"{API_BASE_URL}/api/data/jobs?limit=100")
        
        if jobs_response.status_code == 200:
            jobs_data = jobs_response.json()
            available_jobs = jobs_data.get('jobs', [])
            
            if not available_jobs:
                st.info("No job openings available at the moment. Please check back later!")
            else:
                st.success(f"Found {len(available_jobs)} job opening(s)")
                
                # Option to match against all jobs or select specific ones
                match_option = st.radio(
                    "Matching Options:",
                    ["Match Against All Jobs", "Select Specific Jobs"],
                    horizontal=True
                )
                
                selected_jobs = []
                
                if match_option == "Match Against All Jobs":
                    selected_jobs = available_jobs
                    st.info(f"Will match your resume against all {len(available_jobs)} job(s)")
                else:
                    # Let candidate select which jobs to match against
                    st.markdown("#### Select Jobs to Match:")
                    
                    for job in available_jobs:
                        job_title = job.get('job_title', 'Untitled Position')
                        job_id_short = job['job_id'][:16]
                        num_skills = len(job.get('required_skills', []))
                        exp_years = job.get('required_experience', 0)
                        
                        col_check, col_details = st.columns([1, 4])
                        
                        with col_check:
                            if st.checkbox(
                                f"{job_title}",
                                key=f"job_select_{job['job_id']}"
                            ):
                                selected_jobs.append(job)
                        
                        with col_details:
                            st.caption(f"ID: {job_id_short}... | {num_skills} skills | {exp_years} years exp")
                
                # Match button
                if selected_jobs:
                    st.markdown("---")
                    if st.button("Find My Best Matches", type="primary", key="match_jobs"):
                        with st.spinner(f"Analyzing matches against {len(selected_jobs)} job(s)..."):
                            try:
                                match_results = []
                                
                                for job in selected_jobs:
                                    # Match resume against this job
                                    match_response = requests.post(
                                        f"{API_BASE_URL}/api/match/with-job-id",
                                        json={
                                            "resume_id": resume_data['resume_id'],
                                            "job_id": job['job_id'],
                                            "resume_data": resume_data,
                                            "match_source": "candidate_initiated"
                                        }
                                    )
                                    
                                    if match_response.status_code == 200:
                                        match_result = match_response.json()
                                        match_results.append({
                                            'job': job,
                                            'match': match_result['match_result'],
                                            'job_title': job.get('job_title', 'Untitled')
                                        })
                                
                                # Sort by match score (highest first)
                                match_results.sort(key=lambda x: x['match']['match_score'], reverse=True)
                                
                                st.session_state.candidate_job_matches = match_results
                                st.success(f"Matched against {len(match_results)} job(s)!")
                                st.rerun()
                            
                            except Exception as e:
                                st.error(f"Error matching jobs: {str(e)}")
                
                # Display match results
                if st.session_state.candidate_job_matches:
                    st.markdown("---")
                    st.header("Your Job Match Results")
                    
                    results = st.session_state.candidate_job_matches
                    
                    # Summary metrics
                    st.subheader("Overview")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Jobs Analyzed", len(results))
                    with col2:
                        strong_matches = len([r for r in results if r['match']['match_score'] >= 70])
                        st.metric("Strong Matches (≥70%)", strong_matches)
                    with col3:
                        avg_score = sum(r['match']['match_score'] for r in results) / len(results)
                        st.metric("Average Match", f"{avg_score:.1f}%")
                    with col4:
                        best_score = results[0]['match']['match_score'] if results else 0
                        st.metric("Best Match", f"{best_score:.1f}%")
                    
                    st.markdown("---")
                    
                    # Categorize jobs
                    excellent_matches = [r for r in results if r['match']['match_score'] >= 80]
                    good_matches = [r for r in results if 60 <= r['match']['match_score'] < 80]
                    fair_matches = [r for r in results if 40 <= r['match']['match_score'] < 60]
                    low_matches = [r for r in results if r['match']['match_score'] < 40]
                    
                    # Display categorized results
                    if excellent_matches:
                        st.success(f"### Excellent Matches ({len(excellent_matches)})")
                        st.markdown("You're a great fit for these positions!")
                        
                        for result in excellent_matches:
                            job = result['job']
                            match = result['match']
                            
                            with st.expander(f"{result['job_title']} - {match['match_score']:.1f}% Match", expanded=True):
                                col_a, col_b = st.columns([2, 1])
                                
                                with col_a:
                                    st.markdown(f"**Position:** {result['job_title']}")
                                    st.markdown(f"**Experience Required:** {job.get('required_experience', 0)} years")
                                    st.markdown(f"**Education:** {job.get('required_education', 'Not specified')}")
                                    
                                    st.markdown("**Why you're a great match:**")
                                    st.write(f"Skills Match: {match['skill_match']:.1f}%")
                                    st.write(f"Experience Match: {match['experience_match']:.1f}%")
                                    st.write(f"Education Match: {match['education_match']:.1f}%")
                                
                                with col_b:
                                    st.metric("Overall Match", f"{match['match_score']:.1f}%")
                                    
                                    if match.get('matched_skills'):
                                        st.markdown(f"**Matching Skills ({len(match['matched_skills'])}):**")
                                        for skill in match['matched_skills'][:5]:
                                            st.write(f"{skill}")
                                        if len(match['matched_skills']) > 5:
                                            st.caption(f"... and {len(match['matched_skills'])-5} more")
                                
                                if match.get('skill_gaps'):
                                    st.warning("**Skills to develop:**")
                                    for gap in match['skill_gaps']:
                                        st.write(f"{gap}")
                                else:
                                    st.success("You have all the required skills!")
                    
                    if good_matches:
                        st.info(f"### Good Matches ({len(good_matches)})")
                        st.markdown("You're qualified for these positions with some skill development.")
                        
                        for result in good_matches:
                            job = result['job']
                            match = result['match']
                            
                            with st.expander(f"{result['job_title']} - {match['match_score']:.1f}% Match"):
                                col_a, col_b = st.columns(2)
                                
                                with col_a:
                                    st.markdown(f"**Position:** {result['job_title']}")
                                    st.write(f"Skills Match: {match['skill_match']:.1f}%")
                                    st.write(f"Experience Match: {match['experience_match']:.1f}%")
                                    st.write(f"Education Match: {match['education_match']:.1f}%")
                                
                                with col_b:
                                    if match.get('skill_gaps'):
                                        st.markdown("**Skills to develop:**")
                                        for gap in match['skill_gaps'][:3]:
                                            st.write(f"{gap}")
                                        if len(match['skill_gaps']) > 3:
                                            st.caption(f"... and {len(match['skill_gaps'])-3} more")
                    
                    if fair_matches:
                        with st.expander(f"Fair Matches ({len(fair_matches)}) - Click to expand"):
                            st.markdown("These positions might be suitable with additional training.")
                            for result in fair_matches:
                                match = result['match']
                                st.write(f"• {result['job_title']} - {match['match_score']:.1f}% match")
                    
                    if low_matches:
                        with st.expander(f"Other Positions ({len(low_matches)}) - Click to expand"):
                            st.markdown("These positions may require significant upskilling.")
                            for result in low_matches:
                                match = result['match']
                                st.write(f"• {result['job_title']} - {match['match_score']:.1f}% match")
                    
                    # Recommendations
                    st.markdown("---")
                    st.subheader("Personalized Recommendations")
                    
                    if excellent_matches:
                        st.success("Top Recommendation: Apply to your excellent matches immediately!")
                        top_job = excellent_matches[0]
                        st.write(f"Start with **{top_job['job_title']}** - you're a {top_job['match']['match_score']:.1f}% match!")
                    elif good_matches:
                        st.info("Recommendation: Focus on developing a few key skills to improve your match.")
                        top_job = good_matches[0]
                        if top_job['match'].get('skill_gaps'):
                            st.write(f"For **{top_job['job_title']}**, consider learning: {', '.join(top_job['match']['skill_gaps'][:3])}")
                    else:
                        st.warning("Recommendation: Consider upskilling in high-demand areas to improve your matches.")
                        
                        # Aggregate most common missing skills
                        all_gaps = []
                        for result in results:
                            all_gaps.extend(result['match'].get('skill_gaps', []))
                        
                        if all_gaps:
                            from collections import Counter
                            most_common = Counter(all_gaps).most_common(5)
                            st.write("**Most in-demand skills you're missing:**")
                            for skill, count in most_common:
                                st.write(f"• {skill} (required in {count} job(s))")
        else:
            st.error("Could not fetch jobs from database. Please try again later.")
    
    except Exception as e:
        st.warning(f"Unable to load job openings: {str(e)}")
        st.info("Please contact the administrator or try again later.")
# Call to action
if not st.session_state.candidate_resume:
    st.markdown("---")
    st.info("Upload your resume above to get started!")

# Footer
st.markdown("---")
st.markdown("**Candidate Portal** | Powered by AI-based Resume Screening System")
st.markdown("Your data is secure and used only for matching purposes")
st.markdown("Secure • Fair • AI-Powered")
